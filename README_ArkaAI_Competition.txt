ARKAAI — SNAPDRAGON-FIRST OFFLINE AI STUDY COMPANION
====================================================

This competition build is designed around a replaceable local-AI backend.
On Snapdragon Windows ARM64 PCs it prefers Qualcomm GenieX. On other PCs it
keeps an Ollama compatibility fallback.

CORE CAPABILITIES
- Natural chat with short/long-term conversational context
- Hardware-aware pluggable AI runtime
- Semantic memory using a local ONNX embedding engine
- Semantic document search over PDFs/PPTX/notes
- Persistent student learning profile
- Strength / mistake memory and quiz progress
- Emotion-aware response adaptation with safe local fallback
- PDF, PowerPoint and image ingestion
- Study Studio: summary, study guide, mind map, report, tables, flashcards,
  quiz, compare sources, slides, audio overview and video overview
- One-click Windows setup

SNAPDRAGON RUNTIME
- Qualcomm GenieX is the preferred chat runtime on Snapdragon Windows ARM64.
- Default quality model: ai-hub-models/Qwen3-8B when the detected memory budget allows it.
- Default fast model: ai-hub-models/Qwen3-4B.
- Fallback quality model: ai-hub-models/Qwen3-4B.
- The model can be replaced without editing app.py using:
    ARKAAI_GENIEX_MODEL
    ARKAAI_GENIEX_FAST_MODEL

WHY GENIEX?
GenieX provides a local OpenAI-compatible server and supports Qualcomm NPU,
GPU and CPU paths on Snapdragon Windows ARM64. This makes the runtime a better
fit for the Snapdragon competition target than making Ollama the primary path.

SEMANTIC MEMORY
Semantic retrieval is independent from the LLM runtime. ArkaAI uses a compact
quantized ONNX version of sentence-transformers/all-MiniLM-L6-v2 locally, so
memory and document retrieval continue to work even when the main chat runtime
is swapped.

FIRST RUN
1. Extract the ZIP.
2. Double-click INSTALL_AND_RUN.bat.
3. Allow the setup to download Python packages, GenieX and model/embedding assets.
4. ArkaAI launches automatically.

INTERNET REQUIREMENT
The first setup needs internet to download runtime/model assets. After setup,
chat, semantic memory, the student profile, quiz memory and the emotion fallback
can operate locally. Voice transcription and video helpers may use online tools
when enabled for better quality.

FILES
- app.py — GUI, study workflow, memory, source handling and UI.
- model_router.py — runtime abstraction and adaptive model selection.
- semantic_embeddings.py — local ONNX semantic embedding engine.
- setup_models.py — hardware detection, runtime/model preparation and checks.
- snapdragon_backend.py — Snapdragon/QNN capability bridge.
- INSTALL_AND_RUN.bat — one-click installer and launcher.
- run_ArkaAI.bat — launch after first setup.
- THIRD_PARTY_NOTICES.md — third-party software/model notices.
