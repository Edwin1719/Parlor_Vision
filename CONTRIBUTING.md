# Contribuir a Parlor Vision

Gracias por tu interés en contribuir a **Parlor Vision — Windows + Ollama Edition**. Este documento te guía para participar de forma efectiva.

---

## 🚀 Inicio Rápido para Contribuidores

### 1. Fork y clona

```bash
git clone https://github.com/TU_USUARIO/parlor.git
cd parlor
```

### 2. Crea tu entorno

```bash
conda create -n parlor-dev python=3.12 -y
conda activate parlor-dev
pip install -e .
```

### 3. Crea una rama

```bash
git checkout -b feature/tu-mejora
# o
git checkout -b fix/tu-correccion
```

---

## 📋 Cómo Contribuir

### 🐛 Reportar Bugs

- Usa **GitHub Issues** con el template `Bug Report`
- Incluye: versión de Python, Ollama, SO, y logs de error
- Adjunta `pyproject.toml` si modificaste dependencias

### ✨ Sugerir Funcionalidades

- Abre un **Feature Request** antes de implementar
- Describe el caso de uso y por qué es valioso
- Si ya lo implementaste, abre un PR directamente

### 🔧 Pull Requests

1. **Revisa** que no haya un PR similar abierto
2. **Sigue** el estilo de código existente
3. **Actualiza** el README si tu cambio afecta la documentación
4. **Prueba** localmente antes de enviar

---

## 📐 Estándares de Código

### Python

- Sigue **PEP 8** con línea máxima de 120 caracteres
- Usa **type hints** en funciones nuevas
- Docstrings en funciones públicas
- Nombres descriptivos en inglés (variables, funciones)

### Frontend (HTML/JS)

- Mantén consistencia con CSS variables existentes
- No agregues dependencias externas sin discutir primero
- Prueba en Chrome/Edge (principales targets)

### Commits

Usa mensajes claros y concisos:

```
fix: reduce TTS latency by caching voice models
feat: add screen sharing for vision analysis
docs: update README with industrial use cases
```

---

## 🧪 Testing

### Benchmarks

Antes y después de tu cambio:

```bash
python benchmarks/bench.py
python benchmarks/benchmark_tts.py
```

### Manual

- [ ] Server inicia sin errores
- [ ] Conversación por voz funciona
- [ ] Visión (cámara/pantalla) funciona
- [ ] TTS genera audio correctamente
- [ ] Barge-in interrumpe playback

---

## 🎯 Áreas de Mejora Prioritarias

Si no sabes por dónde empezar, estas mejoras son las más valiosas:

| Área | Descripción |
|------|-------------|
| **Latencia** | Reducir cold-start de primera respuesta |
| **Multi-idioma** | Consistencia 100% en español |
| **GPU Whisper** | Soporte CUDA para transcripción más rápida |
| **Monitor de Visión** | Modo continuo con detección de cambios |
| **Voces TTS** | Selector de voces masculinas/femeninas |

---

## 📞 Contacto

- **Issues:** Para bugs y feature requests
- **Discussions:** Para preguntas generales

¡Todas las contribuciones son bienvenidas! 🙌
