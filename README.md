# Recomendador semántico de Pueblos Mágicos (Rest-Mex 2025)

> README en construcción: se completa al cerrar las etapas de evaluación y web.

## Configuración del LLM (HyDE / Anti-HyDE y etiquetas de arquetipos)

El LLM se usa en dos puntos: generar las reseñas hipotéticas (HyDE y Anti-HyDE) de cada
consulta y poner una etiqueta corta a cada arquetipo. Los **embeddings no cambian**: siempre
se calculan con `nomic-embed-text-v2-moe` en Ollama.

Cadena de respaldo (`src/restmex/llm.py`):

```
Gemini (gemini_model → gemini_fallback_models) → Ollama (ollama_model) → plantilla de antónimos
```

1. Copia `.env.example` como `.env` y pon tu clave de [Google AI Studio](https://aistudio.google.com/apikey):
   ```
   GEMINI_API=tu_clave
   ```
   `.env` está en `.gitignore`; la clave se carga con `python-dotenv` y nunca se registra ni se imprime.
2. Ajusta `config.yaml`:

   | Clave | Valor inicial | Qué hace |
   |---|---|---|
   | `llm.provider` | `gemini` | `gemini` usa la cadena completa; `ollama` usa solo el modelo local |
   | `llm.gemini_model` | `gemini-3.8-flash` | Modelo principal (el Flash más reciente disponible con la clave) |
   | `llm.gemini_fallback_models` | `[gemini-3.5-flash]` | Otros modelos de Gemini que se prueban si el principal falla |
   | `llm.gemini_thinking_level` | `low` | Razonamiento mínimo para bajar la latencia |
   | `llm.gemini_timeout` | `25` | Segundos antes de pasar al siguiente eslabón |
   | `llm.ollama_model` | `qwen2.5:3b` | Respaldo local (`ollama pull qwen2.5:3b`) |
   | `llm.retries` | `1` | Reintentos por proveedor ante un JSON mal formado |
   | `llm.cooldown` | `300` | Segundos que se omite un proveedor después de fallar |

Gemini se llama con el SDK oficial `google-genai` y salida JSON estructurada
(`response_mime_type="application/json"`). Si no hay clave o Gemini no responde, el sistema
sigue funcionando con el modelo local y, en último caso, con la plantilla; la respuesta de la
API (`llm_provider`, `llm_model`, `degraded`, `fallback`) y la interfaz web indican quién respondió.
