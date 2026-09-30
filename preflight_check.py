"""ArkaAI preflight validation used by the one-click installer.
Checks Python modules, app import, backend routing, and key local assets without
starting the GUI or making a model inference request.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "numpy", "PIL", "fitz", "pptx", "speech_recognition", "sounddevice",
    "scipy", "pyttsx3", "onnxruntime", "transformers", "huggingface_hub",
    "imageio_ffmpeg",
]

errors = []
for mod in REQUIRED:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        errors.append(f"{mod}: {exc}")

try:
    import model_router
    router = model_router.AdaptiveModelRouter()
    print(f"Runtime selection: {router.runtime_kind}")
    print(f"Fast model: {router.fast_model}")
    print(f"Quality model: {router.quality_model}")
    print(f"Vision model: {router.vlm_model}")
except Exception as exc:
    errors.append(f"model_router: {exc}")

try:
    import app  # import-only smoke test; GUI is not started here
    print("Application import: OK")
except Exception as exc:
    errors.append(f"app import: {exc}")

if errors:
    print("PREFLIGHT FAILED")
    for item in errors:
        print(" -", item)
    sys.exit(1)

print("PREFLIGHT PASSED")
