"""Parlor Vision — on-device, real-time multimodal AI (voice + vision) — Windows + Ollama Edition."""

import asyncio
import base64
import io
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import tts

# Configuración de Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. "
    "The user is talking to you through a microphone and showing you their camera. You can SEE the camera feed. "
    "If asked about what you see, describe the image provided. "
    "IMPORTANT: Always respond in the SAME LANGUAGE as the user. "
    "If the user speaks Spanish, YOU MUST RESPOND ONLY IN SPANISH. Do not use English words. "
    "Keep responses natural and conversational, 1-4 short sentences."
)

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

tts_backend = None
whisper_model = None
last_detected_lang = "es"

HISTORY_FILE = Path(__file__).parent / "history.json"

def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error cargando historial: {e}")
            return []
    return []

def save_history(history):
    try:
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Error guardando historial: {e}")


def check_ollama_connection():
    """Verify Ollama is running and model is available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned {response.status_code}")
        
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        
        if OLLAMA_MODEL not in model_names:
            print(f"\n⚠️  Model '{OLLAMA_MODEL}' not found!")
            raise RuntimeError(f"Model {OLLAMA_MODEL} not found")
        
        print(f"✅ Ollama connected with model: {OLLAMA_MODEL}")
        return True
    except Exception as e:
        raise RuntimeError(f"Ollama connection failed: {e}")


def load_models():
    """Initialize Whisper, TTS and verify Ollama."""
    global tts_backend, whisper_model
    
    print("\n" + "="*60)
    print("  Parlor Vision - Windows + Ollama Edition")
    print("="*60 + "\n")
    
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
        download_root="./models/whisper"
    )
    print("✅ Whisper loaded.\n")
    
    check_ollama_connection()
    
    print("\nLoading TTS backend...")
    tts_backend = tts.load()
    print("TTS loaded.\n")


@asynccontextmanager
async def lifespan(app):
    """Load models on startup."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


def transcribe_audio(audio_base64: str) -> tuple[str, str]:
    """Transcribe audio with language detection and filtering."""
    global last_detected_lang
    try:
        wav_bytes = base64.b64decode(audio_base64)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            temp_path = f.name
        
        # Forzar el idioma a español desde la raíz para evitar alucinaciones
        segments, info = whisper_model.transcribe(
            temp_path,
            beam_size=5,
            language="es",
            initial_prompt="Hola, ¿cómo estás? Una conversación normal.",
            vad_filter=True,
            condition_on_previous_text=False
        )
        
        transcript = " ".join([segment.text for segment in segments]).strip()
        lang_code = "es" # Siempre forzamos a español
        last_detected_lang = lang_code
        os.unlink(temp_path)
        return transcript, lang_code
    except Exception as e:
        print(f"⚠️ Transcription error: {e}")
        return "", last_detected_lang


async def ollama_chat(transcript: str, lang: str = "es", images: list[str] = None, user_text: str = None, history: list = None) -> str:
    """Send chat request to Ollama with language context and no-CoT config."""
    
    # Instrucción de idioma reforzada para evitar que gemma responda en inglés
    lang_inst = "EL USUARIO TE ESTÁ HABLANDO EN ESPAÑOL. DEBES RESPONDER **EXCLUSIVA Y ÚNICAMENTE** EN ESPAÑOL. CUALQUIER OTRA LENGUA ESTÁ ESTRICTAMENTE PROHIBIDA." if lang == "es" else "USER SPEAKS ENGLISH. ALWAYS RESPOND IN ENGLISH."
    
    # Prevención de descripción compulsiva al inicio
    if not transcript and not user_text and images:
        return "¡Hola! Veo que has activado la cámara. ¿En qué puedo ayudarte?" if lang == "es" else "Hello! I see your camera is on. How can I help you?"

    full_system_prompt = f"{SYSTEM_PROMPT}\n\n{lang_inst}"
    
    messages = [{"role": "system", "content": full_system_prompt}]
    if history:
        messages.extend(history)
    
    user_content = f"User said: \"{transcript}\"" if transcript else (user_text or "Hello!")
    if images and transcript:
        user_content += "\n(Note: User is showing you their camera. Reference it only if relevant.)"
    
    messages.append({"role": "user", "content": user_content})
    if images:
        clean_images = [img.split(",")[1] if "data:image" in img else img for img in images]
        messages[-1]["images"] = clean_images
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "think": False,  # Desactivar CoT para Gemma 4
                "options": {
                    "temperature": 0.5,
                    "num_predict": 150,
                    "stop": ["<think>", "</think>"]
                }
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error: {response.status_code}")
        
        res_json = response.json()
        msg_data = res_json.get("message", {})
        content = msg_data.get("content", "") or msg_data.get("thinking", "")
        
        # Limpieza final de etiquetas de pensamiento si aparecieran
        if "</think>" in content:
            content = content.split("</think>")[-1]
            
        return content.strip()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    
    conversation_history = load_history()
    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None: break

            interrupted.clear()
            
            # 1. Transcripción
            transcript = ""
            lang = last_detected_lang
            if msg.get("audio"):
                transcript, lang = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: transcribe_audio(msg["audio"])
                )
                if transcript:
                    print(f"🎤 Transcribed [{lang}]: {transcript[:60]}...")
            
            # 2. LLM
            t0 = time.time()
            try:
                text_response = await ollama_chat(
                    transcript=transcript,
                    lang=lang,
                    images=[msg["image"]] if msg.get("image") else None,
                    user_text=msg.get("text"),
                    history=conversation_history
                )
                llm_time = time.time() - t0
                print(f"✅ LLM ({llm_time:.2f}s): {text_response[:100]}...")
                
                # Actualizar historial
                conversation_history.append({"role": "user", "content": transcript or "User showed camera"})
                conversation_history.append({"role": "assistant", "content": text_response})
                if len(conversation_history) > 20:
                    conversation_history = conversation_history[-20:]
                
                # Autoguardado silencioso de la memoria en disco
                save_history(conversation_history)
                    
            except Exception as e:
                print(f"❌ LLM error: {e}")
                text_response = "Error de procesamiento." if lang == "es" else "Processing error."
                llm_time = 0

            if interrupted.is_set(): continue

            # ENVIAR TEXTO
            await ws.send_text(json.dumps({
                "type": "text", 
                "text": text_response, 
                "llm_time": round(llm_time, 2),
                "transcription": transcript,
                "language": lang
            }))

            # 3. TTS
            sentences = split_sentences(text_response)
            if not sentences: continue

            await ws.send_text(json.dumps({
                "type": "audio_start",
                "sample_rate": tts_backend.sample_rate,
                "sentence_count": len(sentences),
            }))

            for i, sentence in enumerate(sentences):
                if interrupted.is_set(): break
                try:
                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sentence: tts_backend.generate(s, lang=lang)
                    )
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_text(json.dumps({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                        "index": i,
                    }))
                except Exception as e:
                    print(f"TTS error: {e}")
            
            if not interrupted.is_set():
                await ws.send_text(json.dumps({"type": "audio_end"}))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print("\n" + "="*60)
    print("  Parlor Vision - Starting Server")
    print("="*60)
    print(f"  Ollama URL: {OLLAMA_BASE_URL}")
    print(f"  Model: {OLLAMA_MODEL}")
    print(f"  Port: {port}")
    print("="*60)
    print(f"\n  🌐 Open in browser: http://localhost:{port}")
    print("\n" + "="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
