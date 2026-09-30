"""Small fully-local ONNX semantic embedding engine.

Uses sentence-transformers/all-MiniLM-L6-v2 exported ONNX weights. The setup
script downloads the architecture-appropriate quantized ONNX file and tokenizer.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ASSET_DIR = BASE_DIR / ".runtime" / "embeddings"
MODEL_PATH = ASSET_DIR / "model.onnx"
TOKENIZER_DIR = ASSET_DIR

_LOCK = threading.Lock()
_SHARED = None


class LocalSemanticEmbedder:
    _shared = None

    @classmethod
    def shared(cls):
        global _SHARED
        with _LOCK:
            if _SHARED is None:
                _SHARED = cls()
            return _SHARED

    def __init__(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        if not MODEL_PATH.exists() or not (TOKENIZER_DIR / "tokenizer.json").exists():
            raise RuntimeError("Semantic embedding assets are not installed yet.")
        self.tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), local_files_only=True, use_fast=True)
        providers = ort.get_available_providers()
        # QNN may not support this BERT graph reliably on every device, so CPU
        # is the stable default. The quantized ARM64 model keeps it lightweight.
        provider = "CPUExecutionProvider"
        self.session = ort.InferenceSession(str(MODEL_PATH), providers=[provider])

    def encode(self, texts: List[str]):
        clean = [str(t).strip() for t in texts if str(t).strip()]
        if not clean:
            return []
        enc = self.tokenizer(clean, padding=True, truncation=True, max_length=256, return_tensors="np")
        input_names = {x.name for x in self.session.get_inputs()}
        feed = {}
        for name in input_names:
            if name in enc:
                feed[name] = enc[name]
            elif name == "token_type_ids" and "input_ids" in enc:
                feed[name] = np.zeros_like(enc["input_ids"])
        outputs = self.session.run(None, feed)
        hidden = None
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 3:
                hidden = arr
                break
        if hidden is None:
            raise RuntimeError("Semantic embedding model returned an unexpected output shape.")
        mask = np.asarray(enc["attention_mask"], dtype=np.float32)[..., None]
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norm, 1e-9, None)
        return pooled.astype(np.float32).tolist()
