# ArkaAI Snapdragon Runtime

ArkaAI is **Snapdragon-first**. On Windows ARM64 Snapdragon systems, the main assistant uses Qualcomm GenieX through its local OpenAI-compatible server instead of using Ollama as the primary chat runtime. GenieX can run local GGUF models or Qualcomm AI Hub precompiled bundles on Qualcomm NPU/GPU/CPU paths.

Default tiers:
- Quality: `ai-hub-models/Qwen3-8B` when the detected memory budget allows it.
- Fast: `ai-hub-models/Qwen3-4B`.
- Fallback: `ai-hub-models/Qwen3-4B-Instruct-2507`, then `ai-hub-models/Qwen3-4B`.

The models can be changed without touching `app.py`:
- `ARKAAI_GENIEX_MODEL`
- `ARKAAI_GENIEX_FAST_MODEL`

Semantic memory uses a small local ONNX embedding engine (`all-MiniLM-L6-v2`), so semantic retrieval does not depend on the chat runtime.

On non-Snapdragon PCs, ArkaAI retains an Ollama compatibility fallback.
