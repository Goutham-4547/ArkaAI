"""ArkaAI model/runtime bootstrap.

Snapdragon Windows ARM64:
  - installs Qualcomm GenieX from PyPI (which auto-provisions the native SDK)
  - starts GenieX's local OpenAI-compatible server
  - pulls Qualcomm AI Hub LLM/VLM bundles selected by the adaptive router
  - installs the local ONNX semantic embedding model

Other PCs:
  - retains Ollama as a compatibility fallback for development/testing.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from model_router import (
    GENIEX_FAST_LADDER,
    GENIEX_QUALITY_LADDER,
    GENIEX_VLM_LADDER,
    GENIEX_MODEL_OVERRIDE,
    GENIEX_FAST_MODEL_OVERRIDE,
    GENIEX_VLM_MODEL_OVERRIDE,
    GENIEX_URL,
    PLAN_FILE,
    detect_hardware,
    geniex_models,
    ollama_models,
    probe_runtime,
    recommend,
)

BASE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BASE_DIR / ".runtime"
EMBED_DIR = RUNTIME_DIR / "embeddings"
GENIEX_PY_VERSION = "0.7.0"

EMBED_REPO = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/"
EMBED_TOKENIZER_ROOT = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/"


def run(cmd, timeout=3600, env=None, capture=False):
    print(">", " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, check=False, timeout=timeout, env=env, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


def install_geniex_python():
    print("Installing Qualcomm GenieX Python distribution...")
    # Qualcomm's package auto-fetches the matching Windows ARM64 SDK at install time.
    r = run([sys.executable, "-m", "pip", "install", "--upgrade", f"geniex=={GENIEX_PY_VERSION}"], timeout=1800)
    if r.returncode != 0:
        raise RuntimeError("GenieX Python package installation failed.")


def find_geniex():
    exe = shutil.which("geniex") or shutil.which("geniex.exe")
    if exe:
        return exe
    candidates = [
        Path(sys.executable).parent / "geniex.exe",
        Path(sys.executable).parent / "Scripts" / "geniex.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def start_geniex(exe):
    for _ in range(15):
        try:
            with urllib.request.urlopen(f"{GENIEX_URL}/models", timeout=2):
                return
        except Exception:
            time.sleep(0.3)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen([exe, "serve", "--host", "127.0.0.1"], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, creationflags=creationflags)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{GENIEX_URL}/models", timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("GenieX server did not become ready on http://127.0.0.1:18181.")


def geniex_pull(exe, model):
    names = geniex_models()
    if any(model == n or model.split("@", 1)[0] == n.split("@", 1)[0] for n in names):
        print(f"Already cached: {model}")
        return True
    r = run([exe, "pull", model], timeout=7200)
    return r.returncode == 0


def setup_embedding():
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    arch = platform.machine().upper()
    model_name = "model_qint8_arm64.onnx" if "ARM64" in arch or "AARCH64" in arch else "model_quint8_avx2.onnx"
    dest = EMBED_DIR / "model.onnx"
    if not dest.exists():
        print("Downloading local semantic embedding model...")
        urllib.request.urlretrieve(EMBED_REPO + model_name, dest)
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "config.json"]:
        target = EMBED_DIR / name
        if not target.exists():
            urllib.request.urlretrieve(EMBED_TOKENIZER_ROOT + name, target)
    test = run([sys.executable, "-c",
                "from semantic_embeddings import LocalSemanticEmbedder; print(len(LocalSemanticEmbedder.shared().encode(['ArkaAI semantic memory test'])[0]))"],
               timeout=300, capture=True)
    if test.returncode != 0:
        print(test.stdout or "")
        raise RuntimeError("Semantic embedding engine failed verification.")


def choose_candidate(ladder, profile):
    # Keep selection deterministic. Actual runtime availability is verified after pull.
    cap = max(0.0, profile.ram_gb * 0.72 - 3.0)
    for item in ladder:
        if cap >= item["size_gb"] * 1.18:
            return item["model"]
    return ladder[-1]["model"]


def setup_snapdragon():
    profile = detect_hardware()
    if not profile.is_arm64 or not profile.is_snapdragon:
        raise RuntimeError("This Snapdragon competition deployment requires Windows ARM64 Snapdragon hardware.")

    install_geniex_python()
    exe = find_geniex()
    if not exe:
        raise RuntimeError("GenieX CLI was not found after installation.")
    start_geniex(exe)

    quality = GENIEX_MODEL_OVERRIDE or choose_candidate(GENIEX_QUALITY_LADDER, profile)
    fast = GENIEX_FAST_MODEL_OVERRIDE or choose_candidate(GENIEX_FAST_LADDER, profile)
    vlm = GENIEX_VLM_MODEL_OVERRIDE or choose_candidate(GENIEX_VLM_LADDER, profile)

    quality_candidates = [quality] + [x["model"] for x in GENIEX_QUALITY_LADDER if x["model"] != quality]
    fast_candidates = [fast] + [x["model"] for x in GENIEX_FAST_LADDER if x["model"] != fast]
    vlm_candidates = [vlm] + [x["model"] for x in GENIEX_VLM_LADDER if x["model"] != vlm]

    def pull_and_probe(candidates, label):
        for model in candidates:
            print(f"Preparing {label}: {model}")
            if geniex_pull(exe, model):
                result = probe_runtime(model, runtime_kind="geniex", timeout=1200)
                print(f"{label} probe: {result}")
                if result.get("ok"):
                    return model, result
        raise RuntimeError(f"No working GenieX {label} model was available.")

    quality_ok, quality_bench = pull_and_probe(quality_candidates, "quality")
    fast_ok, fast_bench = pull_and_probe(fast_candidates, "fast")
    vlm_ok, vlm_bench = pull_and_probe(vlm_candidates, "vision")
    setup_embedding()

    plan = {
        "runtime": {"kind": "geniex", "fast": fast_ok, "quality": quality_ok, "vlm": vlm_ok},
        "benchmarks": {"fast": fast_bench, "quality": quality_bench, "vision": vlm_bench},
        "hardware": profile.as_dict(),
        "note": "Selected after hardware check and local inference probes.",
    }
    Path(PLAN_FILE).write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


def setup_generic():
    print("Generic-PC development mode: Ollama remains the compatibility path.")
    names = ollama_models()
    print("Installed Ollama models:", names)


def main():
    profile = detect_hardware()
    if profile.is_snapdragon and profile.is_arm64:
        setup_snapdragon()
    else:
        setup_generic()


if __name__ == "__main__":
    main()
