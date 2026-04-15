# Política de Seguridad

## Versiones Soportadas

| Versión | Soportada |
|---------|-----------|
| 0.3.x   | ✅ Sí |
| < 0.3.0 | ❌ No |

---

## Reportar una Vulnerabilidad

Dado que Parlor Vision es un proyecto de IA **100% local**, las preocupaciones de seguridad principales son:

### ⚠️ Áreas de Preocupación

1. **Inyección de prompts** — Un usuario malicioso podría intentar manipular el system prompt via WebSocket
2. **Ejecución de comandos** — Si se agrega funcionalidad de herramientas locales, garantizar sandboxing
3. **Datos sensibles** — Asegurar que audio/imágenes nunca se envían a servidores externos sin consentimiento
4. **Dependencias** — Vulnerabilidades en paquetes de terceros (kokoro-onnx, faster-whisper, etc.)

### 📩 Cómo Reportar

1. Abre un **Issue** con el label `security`
2. O contacta directamente al mantenedor via GitHub
3. Incluye:
   - Descripción de la vulnerabilidad
   - Pasos para reproducir
   - Impacto potencial
   - Sugerencia de fix (si aplica)

### ⏱️ Timeline Esperado

- **Reconocimiento:** 48 horas
- **Evaluación:** 1 semana
- **Fix:** 2-4 semanas según severidad

---

## Buenas Prácticas Actuales

- ✅ Todo el procesamiento es local (Ollama, Whisper, TTS)
- ✅ No se envían datos a APIs externas
- ✅ Historial de conversación guardado localmente
- ✅ Sin telemetría ni tracking
- ✅ WebSocket solo escucha en localhost
