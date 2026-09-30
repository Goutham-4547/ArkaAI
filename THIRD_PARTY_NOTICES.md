# Third-party notices

ArkaAI is an application assembled from open-source libraries, local model runtimes and model weights. Each dependency and model is subject to its own license/terms.

Important components include:

- Ollama — local model runtime/API.
- Qwen3.5 open model family — local model weights through Ollama; check the specific model license/terms before redistribution.
- Qwen3-Embedding — local semantic retrieval model; check the specific model license/terms.
- GoEmotions ONNX model from Sam Lowe — used as an emotion-response signal.
- ONNX Runtime / ONNX Runtime QNN — local ONNX inference; QNN provider is used when available on supported Snapdragon systems.
- PyPDF2, python-pptx, PyMuPDF, Pillow, pytesseract, SpeechRecognition, sounddevice, pyttsx3 and imageio-ffmpeg.

Model weights are not redistributed inside this ZIP. The installer fetches them from their respective sources during setup.

## Qualcomm GenieX
The Snapdragon runtime can use Qualcomm GenieX. See https://github.com/qualcomm/GenieX for its license and notices. The installer downloads the runtime from the official release rather than bundling a modified copy.

## all-MiniLM-L6-v2
Semantic embeddings use the Apache-2.0 licensed sentence-transformers/all-MiniLM-L6-v2 model exported to ONNX.
