"""Optional Snapdragon acceleration services for ArkaAI.

This module never prevents the desktop app from running on a non-Snapdragon PC.
On Windows ARM64 Snapdragon systems it prefers ONNX Runtime's QNN execution
provider for the local emotion classifier when the provider and compatible model
are available. The same capability can be extended to additional QNN model
assets without changing the GUI/backend contract.
"""
from __future__ import annotations

import os
import platform
from typing import Any, Dict, Optional, Sequence


def platform_info() -> Dict[str, Any]:
    machine = (platform.machine() or "").upper()
    cpu = platform.processor() or ""
    is_arm64 = "ARM64" in machine or "AARCH64" in machine
    text = f"{cpu} {machine}".lower()
    is_snapdragon = any(token in text for token in ("snapdragon", "qualcomm", "x elite"))
    providers = []
    try:
        import onnxruntime as ort
        providers = list(ort.get_available_providers())
    except Exception:
        pass
    qnn = "QNNExecutionProvider" in providers
    return {
        "machine": machine,
        "cpu": cpu,
        "is_arm64": is_arm64,
        "is_snapdragon": is_snapdragon,
        "qnn_available": qnn,
        "providers": providers,
    }


def status_text() -> str:
    info = platform_info()
    if info["is_snapdragon"]:
        return "Snapdragon device detected · QNN/NPU ready" if info["qnn_available"] else "Snapdragon device detected · CPU/local fallback"
    return "Non-Snapdragon host · portable local path"


def create_onnx_session(model_path: str, providers: Optional[Sequence[str]] = None):
    """Create an ONNX Runtime session, preferring Qualcomm QNN when available."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    available = set(ort.get_available_providers())
    requested = list(providers or [])
    if not requested and "QNNExecutionProvider" in available:
        requested = ["QNNExecutionProvider", "CPUExecutionProvider"]

    if "QNNExecutionProvider" in requested and "QNNExecutionProvider" in available:
        try:
            return ort.InferenceSession(
                model_path,
                sess_options=options,
                providers=["QNNExecutionProvider", "CPUExecutionProvider"],
                provider_options=[
                    {
                        "backend_path": "QnnHtp.dll",
                        "htp_performance_mode": "burst",
                        "htp_graph_finalization_optimization_mode": "3",
                    },
                    {},
                ],
            )
        except Exception:
            pass

    return ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])


def qnn_model_asset_available(path: Optional[str]) -> bool:
    if not path:
        return False
    return os.path.exists(path) and os.path.isfile(path)
