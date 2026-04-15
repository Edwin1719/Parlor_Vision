"""TTS multiplataforma: Kokoro-ONNX (primario), edge-tts (fallback Windows)."""

import os
import sys
import numpy as np


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(self, text: str, voice: str = None, speed: float = 1.1, lang: str = "en") -> np.ndarray:
        raise NotImplementedError


class KokoroONNXBackend(TTSBackend):
    """kokoro-onnx backend."""

    def __init__(self):
        import kokoro_onnx
        from huggingface_hub import hf_hub_download

        print("Cargando Kokoro-ONNX...")
        model_path = hf_hub_download("fastrtc/kokoro-onnx", "kokoro-v1.0.onnx")
        voices_path = hf_hub_download("fastrtc/kokoro-onnx", "voices-v1.0.bin")

        self._model = kokoro_onnx.Kokoro(model_path, voices_path)
        self.sample_rate = 24000
        
        self.voice_map = {
            "en": "af_heart", # American Female
            "es": "ef_dora"   # Spanish Female (Nativa)
        }

    def generate(self, text: str, voice: str = None, speed: float = 1.1, lang: str = "en") -> np.ndarray:
        selected_voice = voice or self.voice_map.get(lang, "af_heart")
        print(f"🔊 Generando audio ({lang}) con voz: {selected_voice}")
        
        # Le pasamos explícitamente el idioma 'lang' a create() para que la 
        # fonetización (G2P) se haga con reglas de español y no de inglés.
        pcm, _sr = self._model.create(text, voice=selected_voice, speed=speed, lang=lang)
        return pcm


class EdgeTTSBackend(TTSBackend):
    """edge-tts backend (Microsoft) - CALIDAD PROFESIONAL PARA ESPAÑOL."""
    
    def __init__(self):
        self.sample_rate = 24000
        self.voice_map = {
            "en": "en-US-EmmaNeural",
            "es": "es-MX-DaliaNeural" # Voz mexicana muy natural
        }
        
    def generate(self, text: str, voice: str = None, speed: float = 1.1, lang: str = "en") -> np.ndarray:
        import asyncio
        import io
        import edge_tts
        import soundfile as sf
        
        selected_voice = voice or self.voice_map.get(lang, "en-US-EmmaNeural")
        print(f"🔊 Generando audio Edge-TTS ({lang}) con voz: {selected_voice}")
        
        rate = f"{int((speed - 1) * 100):+d}%"
            
        async def _generate():
            communicate = edge_tts.Communicate(text, selected_voice, rate=rate)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        
        loop = asyncio.get_event_loop()
        audio_data = loop.run_until_complete(_generate())
        audio, sr = sf.read(io.BytesIO(audio_data))
        
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        return audio


def load() -> TTSBackend:
    """Load the best available TTS backend para Windows."""
    # Intentar kokoro-onnx primero (recomendado, offline)
    try:
        backend = KokoroONNXBackend()
        print(f"TTS: kokoro-onnx (sample_rate={backend.sample_rate})")
        return backend
    except Exception as e:
        print(f"TTS: kokoro-onnx falló ({e}), intentando edge-tts...")
        
    # Fallback a edge-tts (requiere internet)
    try:
        backend = EdgeTTSBackend()
        print(f"TTS: edge-tts (sample_rate={backend.sample_rate})")
        return backend
    except Exception as e:
        print(f"TTS: edge-tts falló ({e})")
        
    raise RuntimeError(
        "No se pudo cargar ningún backend de TTS. "
        "Instala kokoro-onnx o edge-tts: pip install kokoro-onnx o pip install edge-tts"
    )
