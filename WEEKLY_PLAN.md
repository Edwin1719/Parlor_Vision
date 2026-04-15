# 📋 Plan de Mejoras — Semana Próxima

Objetivo: Transformar Parlor de un prototipo funcional a una aplicación conversacional robusta, rápida y verdaderamente multi-idioma.

---

## 🎯 Objetivos de la Semana

| # | Mejora | Impacto | Esfuerzo | Prioridad |
|---|--------|---------|----------|-----------|
| 1 | Reducir latencia primera respuesta | 🔴 Alto | 🟡 Medio | **P0** |
| 2 | Consistencia 100% multi-idioma | 🔴 Alto | 🟢 Bajo | **P0** |
| 3 | Evaluación de modelos alternativos | 🔴 Alto | 🟡 Medio | **P1** |
| 4 | Selección de voces TTS | 🟡 Medio | 🟢 Bajo | **P1** |
| 5 | Whisper con GPU (CUDA) | 🟡 Medio | 🔴 Alto | **P2** |
| 6 | Indicador de idioma en UI | 🟡 Medio | 🟢 Bajo | **P2** |
| 7 | Precarga de modelo al iniciar | 🟡 Medio | 🟢 Bajo | **P2** |
| 8 | Ajustes fin de VAD | 🟢 Bajo | 🟢 Bajo | **P3** |

---

## 🔥 P0 — Críticos

### 1. Reducir Latencia de Primera Respuesta

**Estado actual:** 15-20s primera vez, 2-4s después

**Causa raíz:**
- Ollama carga el modelo en GPU en la primera petición
- Warm-up de kernels CUDA
- Primer prompt requiere inicialización completa

**Soluciones a implementar:**

#### A. Precarga del Modelo al Iniciar

```python
# server.py — En load_models()
def load_models():
    # ... carga Whisper, TTS
    
    # Precarga Ollama con petición dummy
    print("🔥 Pre-loading Ollama model...")
    requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": "ready",
            "stream": False
        },
        timeout=60
    )
    print("✅ Model pre-loaded. First response will be fast.")
```

**Beneficio:** Primera respuesta ~3s en vez de ~20s
**Costo:** +30s al iniciar servidor (una sola vez)

#### B. Mantener Modelo Activo

Ollama descarga modelos de VRAM tras 5 minutos de inactividad. Podemos mantenerlo activo:

```python
# Background task para mantener modelo vivo
async def keep_model_alive():
    while True:
        await asyncio.sleep(240)  # Cada 4 minutos
        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": "ping", "stream": False},
            timeout=30
        )

# Iniciar en lifespan
```

---

### 2. Consistencia 100% Multi-idioma

**Problema:** A veces responde en inglés aunque detectamos español.

**Análisis de logs:**
```
Input (es): "hola como estás"
Output: "Hola, estoy bien..." ✅

Input (es): "que está viendo la imagen"
Output: "You are showing me..." ❌ (inglés)

Input (es): "responde en español"
Output: "Veo a un hombre..." ✅
```

**Hipótesis:** El modelo responde en inglés cuando no detecta suficientes "señales" de español en el prompt.

**Solución en 3 capas:**

#### Capa 1: Detección Mejorada

```python
def detect_language(text: str) -> str:
    """Mejor detección con scoring."""
    text_lower = text.lower()
    
    # Spanish score
    es_words = ['hola', 'qué', 'cómo', 'cuál', 'dónde', 'cuándo', 'quién',
                'estoy', 'estás', 'está', 'somos', 'son', 'tengo', 'tienes',
                'puedo', 'puedes', 'necesito', 'quiero', 'ayuda', 'por favor',
                'gracias', 'bien', 'mal', 'mejor', 'peor', 'muy', 'más',
                'menos', 'siempre', 'nunca', 'también', 'solo', 'con', 'sin',
                'para', 'sobre', 'entre', 'hasta', 'desde', 'aquí', 'ahí',
                'allí', 'este', 'ese', 'aquel', 'el', 'la', 'los', 'las',
                'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'porque',
                'que', 'si', 'como', 'cuando', 'donde', 'mientras']
    
    # English score
    en_words = ['hello', 'hi', 'hey', 'how', 'what', 'where', 'when', 'who',
                'am', 'is', 'are', 'have', 'has', 'can', 'could', 'would',
                'should', 'will', 'do', 'does', 'did', 'the', 'a', 'an',
                'and', 'or', 'but', 'because', 'if', 'as', 'when', 'where']
    
    es_score = sum(1 for w in es_words if w in text_lower)
    en_score = sum(1 for w in en_words if w in text_lower)
    
    if es_score > en_score:
        return "es"
    elif en_score > es_score:
        return "en"
    else:
        # Tie-breaker: check for Spanish-specific chars
        if any(c in text for c in ['á', 'é', 'í', 'ó', 'ú', 'ñ', '¿', '¡']):
            return "es"
        return "en"
```

#### Capa 2: System Prompt Reforzado

```python
SYSTEM_PROMPT_ES = """Eres un asistente conversacional amigable. El usuario te habla por micrófono y te muestra su cámara.

REGLA FUNDAMENTAL: Este usuario habla ESPAÑOL. DEBES responder EXCLUSIVAMENTE en español.
- NUNCA uses inglés en tu respuesta
- Si ves texto en inglés, tradúcelo y responde en español
- Describe lo que ves en la cámara en español
- Mantén respuestas cortas: 1-4 oraciones
- Sé natural y conversacional"""

SYSTEM_PROMPT_EN = """You are a friendly conversational AI assistant. The user talks to you through a microphone and shows you their camera.

Keep responses natural and conversational, 1-4 short sentences.
Describe what you see if the user shows you their camera."""
```

#### Capa 3: Verificación Post-Respuesta

```python
def ensure_language(response: str, target_lang: str) -> str:
    """Verificar que la respuesta está en el idioma correcto."""
    if target_lang == "es":
        # Detectar si la respuesta es mayormente en inglés
        detected = detect_language(response)
        if detected == "en":
            print("⚠️ Response was in English, requesting Spanish...")
            # Re-prompt con instrucción explícita
            retry = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "Traduce la siguiente respuesta al español, manteniendo el significado exacto:"},
                        {"role": "user", "content": response}
                    ],
                    "stream": False
                },
                timeout=30
            )
            return retry.json()["message"]["content"]
    return response
```

---

## 🎯 P1 — Importantes

### 3. Evaluación de Modelos Alternativos

**Objetivo:** Encontrar el mejor balance velocidad/capacidad/multi-idioma.

**Modelos a evaluar:**

#### A. `gemma3:4b` — Más rápido que gemma4

```bash
# Instalar
ollama pull gemma3:4b

# Probar
ollama run gemma3:4b "Hola, ¿cómo estás?"

# Medir
time ollama run gemma3:4b "Describe esta imagen" < imagen.jpg
```

**Criterios:**
- Velocidad de respuesta (<5s ideal)
- Calidad de descripción de imágenes
- Capacidad multi-idioma (español consistente)
- Uso de VRAM (<6GB para RTX 4050)

#### B. `llama3.2-vision` — Especializado en visión

```bash
ollama pull llama3.2-vision
```

**Ventajas:**
- Diseñado específicamente para visión + texto
- 3B parámetros (ligero y rápido)
- Buen soporte multi-idioma

#### C. `qwen2.5:7b` — Mejor español

```bash
ollama pull qwen2.5:7b
```

**Ventajas:**
- Excelente soporte para español (modelo chino, muy bueno en idiomas)
- 7B parámetros (más capaz)
- Rápido con GPU

**Desventajas:**
- Sin visión (necesitaríamos modelo separado)

#### Comparativa a documentar:

| Modelo | Primera Respuesta | Subsiguientes | Descripción Imagen | Español Consistente | VRAM |
|--------|-------------------|---------------|--------------------|---------------------|-------|
| `gemma4:e2b` | 20s | 3s | ✅ Buena | ⚠️ 70% | 6 GB |
| `gemma3:4b` | ? | ? | ? | ? | 4 GB |
| `llama3.2-vision` | ? | ? | ? | ? | 4 GB |
| `qwen2.5:7b` | ? | ? | ❌ Sin visión | ✅ 95% | 6 GB |

**Implementar script de benchmark:**

```python
# benchmarks/benchmark_models.py
import time
import requests

MODELS = ["gemma4:e2b", "gemma3:4b", "llama3.2-vision", "qwen2.5:7b"]
TEST_PROMPTS = [
    ("Hola, ¿cómo estás?", None),
    ("Describe lo que ves", "imagen_test.jpg"),
    ("¿Qué modelo eres?", None),
]

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"Testing: {model}")
    print(f"{'='*60}")
    
    for prompt, image in TEST_PROMPTS:
        t0 = time.time()
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )
        elapsed = time.time() - t0
        print(f"  Prompt: {prompt[:30]}... → {elapsed:.2f}s")
        print(f"  Response: {response.json()['message']['content'][:80]}...")
```

---

### 4. Selección de Voces TTS

**Problema:** Solo voz femenina `af_heart`.

**Kokoro tiene 10+ voces disponibles:**

| Voz | Género | Acento | Calidad |
|-----|--------|--------|---------|
| `af_heart` | Femenino | US | ⭐⭐⭐⭐ |
| `af_bella` | Femenino | US | ⭐⭐⭐⭐ |
| `am_adam` | Masculino | US | ⭐⭐⭐ |
| `am_michael` | Masculino | US | ⭐⭐⭐ |
| `bf_emma` | Femenino | UK | ⭐⭐⭐⭐ |
| `bf_isabella` | Femenino | UK | ⭐⭐⭐⭐ |
| `bm_george` | Masculino | UK | ⭐⭐⭐ |

**Implementación:**

#### A. Variable de entorno

```python
# .env
TTS_VOICE=am_adam  # Voz masculina estadounidense
```

#### B. Selector en UI (HTML)

```html
<div class="voice-selector">
  <label>Voz:</label>
  <select id="voiceSelect">
    <option value="af_heart">♀️ Bella (US)</option>
    <option value="am_adam">♂️ Adam (US)</option>
    <option value="bf_emma">♀️ Emma (UK)</option>
    <option value="bm_george">♂️ George (UK)</option>
  </select>
</div>
```

#### C. Pasar voz al servidor

```javascript
// index.html — WebSocket
ws.send(JSON.stringify({
  audio: wavBase64,
  image: imageBase64,
  voice: document.getElementById('voiceSelect').value
}));
```

#### D. Server.py usa la voz

```python
# En websocket_endpoint
voice = msg.get("voice", "af_heart")

# En generación TTS
pcm = tts_backend.generate(sentence, voice=voice)
```

---

## 📊 P2 — Secundarios

### 5. Whisper con GPU (CUDA)

**Prerrequisitos:**
1. Instalar CUDA Toolkit 12.x
2. Instalar cuDNN
3. Reinstalar `faster-whisper` con soporte GPU

**Pasos:**

```bash
# 1. Verificar GPU CUDA
nvidia-smi

# 2. Instalar CUDA Toolkit
# Descargar de: https://developer.nvidia.com/cuda-downloads
# Windows → x86_64 → 10/11 → exe (local)

# 3. Verificar instalación
nvcc --version

# 4. Reinstalar faster-whisper
pip uninstall -y faster-whisper ctranslate2
pip install faster-whisper

# 5. Probar con GPU
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny', device='cuda', compute_type='float16')
print('GPU Whisper funcionando!')
"
```

**Beneficio esperado:**
- CPU: ~1.0s → GPU: ~0.3s
- 3x más rápido

---

### 6. Indicador de Idioma en UI

**Frontend (index.html):**

```html
<div class="language-indicator" id="langIndicator">
  🇬🇧 English
</div>
```

```javascript
// Detectar idioma en respuesta del servidor
if (msg.type === 'text') {
  const lang = detectLanguage(msg.text);
  updateLanguageIndicator(lang);
}

function updateLanguageIndicator(lang) {
  const indicator = document.getElementById('langIndicator');
  const flags = {
    'es': '🇪🇸 Español',
    'en': '🇬🇧 English',
    'it': '🇮🇹 Italiano',
    'fr': '🇫🇷 Français'
  };
  indicator.textContent = flags[lang] || '🌐 Unknown';
  indicator.className = `language-indicator lang-${lang}`;
}
```

**CSS:**

```css
.language-indicator {
  font-size: 11px;
  padding: 4px 10px;
  border-radius: 100px;
  background: var(--c-surface);
  transition: all 0.3s ease;
}

.language-indicator.lang-es {
  background: rgba(255, 200, 0, 0.15);
  color: #ffd700;
}

.language-indicator.lang-en {
  background: rgba(100, 150, 255, 0.15);
  color: #6496ff;
}
```

---

### 7. Ajustes Fin de VAD

**Problema:** Corta audio demasiado pronto, frases incompletas.

**Frontend (index.html):**

```javascript
myvad = await vad.MicVAD.new({
  getStream: async () => new MediaStream(mediaStream.getAudioTracks()),
  positiveSpeechThreshold: 0.5,
  negativeSpeechThreshold: 0.25,
  redemptionMs: 800,       // Antes: 600 → Más tiempo para detectar pausa
  minSpeechMs: 400,        // Antes: 300 → Más estabilidad
  preSpeechPadMs: 500,     // Antes: 300 → Más contexto previo
  onSpeechStart: handleSpeechStart,
  onSpeechEnd: handleSpeechEnd,
  // ...
});
```

---

## 💡 Mejoras Adicionales (Bonus)

### 8. Modo "Chat de Texto" (Sin Audio)

**Escenario:** Usuario quiere conversar sin micrófono.

**Implementación:**

```html
<!-- Input de texto -->
<div class="text-input" id="textInputContainer" style="display: none;">
  <input type="text" id="textInput" placeholder="Escribe tu mensaje...">
  <button id="textSend">Enviar</button>
</div>

<!-- Toggle -->
<button id="modeToggle">🎤 Voz | ⌨️ Texto</button>
```

```javascript
// Toggle entre modo voz y texto
document.getElementById('modeToggle').addEventListener('click', () => {
  const isTextMode = textInputContainer.style.display !== 'none';
  textInputContainer.style.display = isTextMode ? 'none' : 'block';
  
  if (!isTextMode) {
    myvad.pause();  // Desactivar VAD
  } else {
    myvad.start();  // Reactivar VAD
  }
});
```

---

### 9. Persistencia de Conversaciones

**Guardar historial entre sesiones:**

```python
import json
from datetime import datetime

def save_conversation(transcript, response, llm_time, tts_time):
    """Guardar conversación a archivo JSON."""
    history_file = Path(__file__).parent / "conversation_history.json"
    
    if history_file.exists():
        history = json.loads(history_file.read_text())
    else:
        history = []
    
    history.append({
        "timestamp": datetime.now().isoformat(),
        "user": transcript,
        "assistant": response,
        "llm_time": llm_time,
        "tts_time": tts_time
    })
    
    # Mantener últimos 100 intercambios
    history = history[-100:]
    
    history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False))
```

---

### 10. Estadísticas de Uso

**Dashboard simple en UI:**

```html
<div class="stats-panel">
  <h3>Estadísticas</h3>
  <div>💬 Intercambios: <span id="exchangeCount">0</span></div>
  <div> ⏱️ Latencia promedio: <span id="avgLatency">0s</span></div>
  <div>🌐 Idiomas detectados: <span id="langStats">EN: 3, ES: 2</span></div>
</div>
```

---

### 11. Wake Word (Palabra de Activación)

**Activar solo cuando escucha "Hey Parlor" o similar:**

```python
# Usar modelo wake word ligero
# Opciones:
# - Porcupine (Picovoice)
# - OpenWakeWord (open source)
# - Simple keyword detection con Whisper tiny

# Implementación simple:
WAKE_WORDS = ["hey parlor", "ok parlor", "hey asistente"]

def check_wake_word(transcript: str) -> bool:
    return any(w in transcript.lower() for w in WAKE_WORDS)
```

---

## 📅 Planificación Semanal

### Lunes

- [ ] Implementar **detección de idioma mejorada** (scoring)
- [ ] Crear **system prompts separados** (ES/EN)
- [ ] Probar consistencia multi-idioma

### Martes

- [ ] **Precarga de modelo** al iniciar servidor
- [ ] Benchmark de **modelos alternativos** (gemma3, llama3.2-vision, qwen2.5)
- [ ] Documentar resultados

### Miércoles

- [ ] **Selección de voces TTS** en UI
- [ ] **Indicador de idioma** en interfaz
- [ ] Ajustes de **VAD** para frases más completas

### Jueves

- [ ] Intentar **Whisper con GPU** (instalar CUDA)
- [ ] **Persistencia de conversaciones**
- [ ] Estadísticas de uso

### Viernes

- [ ] Pruebas integrales
- [ ] Documentar resultados en README
- [ ] Crear script de benchmark automático

### Sábado/Domingo

- [ ] Mejoras bonus (modo texto, wake word)
- [ ] Optimización final
- [ ] Preparar demo

---

## 🎯 Criterios de Éxito

Al final de la semana, deberíamos tener:

| Métrica | Actual | Objetivo |
|---------|--------|----------|
| Primera respuesta | 20s | **<5s** |
| Respuestas siguientes | 7s | **<4s** |
| Consistencia español | ~70% | **>95%** |
| Voces disponibles | 1 | **4+** |
| Idiomas en UI | 0 | **Sí** |
| Whisper GPU | No | **Sí (opcional)** |
| Historial persistente | No | **Sí** |

---

## 📊 Script de Benchmark Automatizado

```python
# benchmarks/weekly_improvements.py
"""Benchmark semanal de mejoras."""

import time
import requests
import statistics

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:e2b"

def benchmark_first_response():
    """Medir latencia de primera respuesta tras reinicio."""
    # Reiniciar Ollama (opcional)
    # subprocess.run(["ollama", "stop", MODEL])
    
    t0 = time.time()
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hola"}],
            "stream": False
        },
        timeout=60
    )
    elapsed = time.time() - t0
    
    print(f"Primera respuesta: {elapsed:.2f}s")
    return elapsed

def benchmark_language_consistency():
    """Probar consistencia de español."""
    test_inputs_es = [
        "hola cómo estás",
        "qué ves en la imagen",
        "cuéntame un chiste",
        "qué tiempo hace hoy",
        "dime tu nombre",
    ]
    
    success_count = 0
    for text in test_inputs_es:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Responde siempre en español."},
                    {"role": "user", "content": text}
                ],
                "stream": False
            }
        )
        result = response.json()["message"]["content"]
        
        # Verificar que contiene palabras en español
        es_detected = any(w in result.lower() for w in ['estoy', 'veo', 'soy', 'puedo', 'hola'])
        
        if es_detected:
            success_count += 1
            print(f"  ✅ '{text}' → Español")
        else:
            print(f"  ❌ '{text}' → {result[:50]}")
    
    consistency = success_count / len(test_inputs_es) * 100
    print(f"\nConsistencia español: {consistency:.0f}%")
    return consistency

if __name__ == "__main__":
    print("="*60)
    print("  Benchmark Semanal de Mejoras")
    print("="*60)
    
    print("\n1. Latencia primera respuesta")
    first_latency = benchmark_first_response()
    
    print("\n2. Consistencia de idioma español")
    consistency = benchmark_language_consistency()
    
    print("\n" + "="*60)
    print("  Resultados")
    print("="*60)
    print(f"  Primera respuesta: {first_latency:.2f}s (objetivo: <5s)")
    print(f"  Consistencia ES: {consistency:.0f}% (objetivo: >95%)")
```

---

## 🔗 Recursos Útiles

- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **Ollama Models:** https://ollama.com/library
- **Kokoro Voices:** https://huggingface.co/hexgrad/Kokoro-82M
- **Faster-Whisper:** https://github.com/SYSTRAN/faster-whisper
- **Silero VAD:** https://github.com/snakers4/silero-vad

---

**Última actualización:** Sábado 11 de abril, 2026  
**Responsable:** Equipo Parlor
