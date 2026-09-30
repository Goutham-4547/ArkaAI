# Snapdragon optimization notes

ArkaAI is designed around a portable inference interface rather than a GUI hard-wired to one model.

## Hardware-aware routing

`model_router.py` detects:
- physical RAM
- GPU memory when available
- CPU architecture
- Snapdragon/Qualcomm CPU hints
- ARM64 status
- ONNX Runtime QNN provider availability

It selects a practical local LLM tier and a lower-latency companion. Setup runs a small local load/response benchmark and stores the verified runtime plan.

## Qualcomm path

`snapdragon_backend.py` detects `QNNExecutionProvider` and creates ONNX Runtime sessions with QNN/HTP first and CPU fallback. This is used by the local emotion classifier when the provider supports the model.

The app therefore has a real Qualcomm-specific execution path without making the whole product unusable on the developer's non-Snapdragon PC.

## Why the main LLM remains pluggable

Qualcomm AI Hub has multiple model/target/runtime combinations, and some current AI Hub model cards include application restrictions. The project does not silently choose a packaged AI Hub model that has restrictions incompatible with an educational/emotional-companion use case. Instead, the LLM adapter remains replaceable, and compatible QNN/ONNX model assets can be inserted later without changing semantic memory, learning memory, UI or study tools.

## Performance evidence

After installation, `.arkaai_model_plan.json` stores the host hardware profile and the measured local model probe. Use this file when preparing a presentation slide showing adaptive deployment.
