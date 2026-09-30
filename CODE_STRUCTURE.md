ArkaAI code structure
=====================

app.py
  Main desktop application, GUI, source ingestion, memory, learning profile,
  quiz, notes and Study Studio.

model_router.py
  Hardware profiler, adaptive runtime selector, GenieX transport on Snapdragon,
  Ollama compatibility fallback, and stable chat/stream/embed interface.

semantic_embeddings.py
  Local ONNX embedding engine used for semantic memory/document search.

setup_models.py
  One-time setup: detects hardware, prepares GenieX or fallback runtime, selects
  models, downloads local embedding assets and performs smoke tests.

snapdragon_backend.py
  Snapdragon/QNN capability bridge used by the UI and documentation.

INSTALL_AND_RUN.bat
  One-click Windows setup and launch.
