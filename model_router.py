"""ArkaAI adaptive local AI runtime.

Snapdragon-first design:
  1) Qualcomm GenieX local server on Snapdragon Windows ARM64.
  2) Ollama fallback on non-Snapdragon systems or when GenieX is unavailable.

The GUI depends only on AdaptiveModelRouter, so the underlying AI runtime/model
can be replaced without rewriting the application.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = os.path.join(BASE_DIR, ".arkaai_model_plan.json")

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
GENIEX_URL = os.getenv("ARKAAI_GENIEX_URL", "http://127.0.0.1:18181/v1").rstrip("/")
GENIEX_MODEL_OVERRIDE = os.getenv("ARKAAI_GENIEX_MODEL", "").strip()
GENIEX_FAST_MODEL_OVERRIDE = os.getenv("ARKAAI_GENIEX_FAST_MODEL", "").strip()
GENIEX_VLM_MODEL_OVERRIDE = os.getenv("ARKAAI_GENIEX_VLM_MODEL", "").strip()
FORCE_MODEL = os.getenv("ARKAAI_MODEL_OVERRIDE", "").strip()
FORCE_FAST = os.getenv("ARKAAI_FAST_MODEL_OVERRIDE", "").strip()

# Snapdragon-first model choices.  The competition path uses Qualcomm
# GenieX + Qualcomm AI Hub precompiled bundles.  We keep a stable interface so
# the underlying model can be replaced without changing app.py.
#
# Quality: Qwen3-8B is the higher-capacity text tier that Qualcomm lists for
# Snapdragon X Elite / X Plus / X2 Elite compute.
# Fast: Qwen3-4B is lighter and has a published Snapdragon X2 Elite profile.
# Vision: Qwen3-VL-8B, then Qwen3-VL-4B, when a visual question is detected.
GENIEX_QUALITY_LADDER = [
    {"model": "ai-hub-models/Qwen3-8B", "size_gb": 8.0, "ram_gb": 16.0},
    {"model": "ai-hub-models/Qwen3-4B", "size_gb": 4.5, "ram_gb": 8.0},
]
GENIEX_FAST_LADDER = [
    {"model": "ai-hub-models/Qwen3-4B", "size_gb": 4.5, "ram_gb": 8.0},
]
GENIEX_VLM_LADDER = [
    {"model": "ai-hub-models/Qwen3-VL-8B-Instruct", "size_gb": 9.0, "ram_gb": 18.0},
    {"model": "ai-hub-models/Qwen3-VL-4B-Instruct", "size_gb": 5.0, "ram_gb": 10.0},
]

# Generic-PC fallback only. Snapdragon machines should use GenieX first.
OLLAMA_PRIMARY_LADDER = [
    {"model": "qwen3.5:122b", "size_gb": 81.0, "ram_gb": 128.0, "vram_gb": 64.0},
    {"model": "qwen3.5:35b", "size_gb": 24.0, "ram_gb": 64.0, "vram_gb": 20.0},
    {"model": "qwen3.5:27b", "size_gb": 17.0, "ram_gb": 48.0, "vram_gb": 12.0},
    {"model": "qwen3.5:9b", "size_gb": 6.6, "ram_gb": 24.0, "vram_gb": 6.0},
    {"model": "qwen3.5:4b", "size_gb": 3.4, "ram_gb": 16.0, "vram_gb": 4.0},
    {"model": "qwen3.5:2b", "size_gb": 2.7, "ram_gb": 8.0, "vram_gb": 2.0},
]
OLLAMA_FAST_LADDER = [
    {"model": "qwen3.5:9b", "size_gb": 6.6, "ram_gb": 24.0, "vram_gb": 6.0},
    {"model": "qwen3.5:4b", "size_gb": 3.4, "ram_gb": 16.0, "vram_gb": 4.0},
    {"model": "qwen3.5:2b", "size_gb": 2.7, "ram_gb": 8.0, "vram_gb": 2.0},
]

# Semantic embeddings are handled by a small local ONNX model in
# semantic_embeddings.py, so the main assistant does not depend on Ollama.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class HardwareProfile:
    ram_gb: float = 0.0
    vram_gb: float = 0.0
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    cpu_name: str = ""
    machine: str = ""
    os_name: str = ""
    is_arm64: bool = False
    is_snapdragon: bool = False
    qnn_available: bool = False
    geniex_available: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _run(cmd: List[str], timeout: int = 15) -> str:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, timeout=timeout, check=False)
        return p.stdout.strip()
    except Exception:
        return ""


def _ram_gb() -> float:
    if os.name == "nt":
        out = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        try:
            return int(out) / (1024 ** 3)
        except Exception:
            pass
    try:
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / (1024 ** 3)
    except Exception:
        return 0.0


def _cpu_name() -> str:
    if os.name == "nt":
        out = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"])
        if out:
            return out.strip()
    return platform.processor() or platform.machine()


def _windows_gpus() -> List[tuple[str, float]]:
    if os.name != "nt":
        return []
    ps = "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Csv -NoTypeInformation"
    out = _run(["powershell", "-NoProfile", "-Command", ps], timeout=12)
    result: List[tuple[str, float]] = []
    for line in out.splitlines()[1:]:
        try:
            name, ram = [x.strip('"') for x in line.split('","', 1)]
            if ram and ram.isdigit():
                result.append((name, int(ram) / (1024 ** 3)))
        except Exception:
            continue
    return result


def _nvidia_gpus() -> List[tuple[str, float]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    out = _run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], timeout=12)
    result: List[tuple[str, float]] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            result.append((parts[0], float(parts[1]) / 1024.0))
        except ValueError:
            continue
    return result


def _qnn_available() -> bool:
    try:
        import onnxruntime as ort
        return "QNNExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _find_geniex() -> Optional[str]:
    exe = shutil.which("geniex") or shutil.which("geniex.exe")
    if exe:
        return exe
    local_candidates = [
        os.path.join(BASE_DIR, ".runtime", "geniex", "geniex.exe"),
        os.path.join(BASE_DIR, ".runtime", "geniex", "bin", "geniex.exe"),
    ]
    for path in local_candidates:
        if os.path.exists(path):
            return path
    return None


def detect_hardware() -> HardwareProfile:
    machine = (platform.machine() or "").upper()
    cpu = _cpu_name()
    combined = f"{cpu} {machine}".lower()
    cards = _nvidia_gpus() or _windows_gpus()
    geniex = _find_geniex()
    return HardwareProfile(
        ram_gb=round(_ram_gb(), 1),
        vram_gb=round(sum(v for _, v in cards), 1),
        gpu_count=len(cards),
        gpu_names=[n for n, _ in cards],
        cpu_name=cpu,
        machine=machine,
        os_name=platform.platform(),
        is_arm64=("ARM64" in machine or "AARCH64" in machine),
        is_snapdragon=("snapdragon" in combined or "qualcomm" in combined or "x elite" in combined or "x2e" in combined),
        qnn_available=_qnn_available(),
        geniex_available=bool(geniex),
    )


def _http_json(url: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 60,
               method: Optional[str] = None) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method=method or ("POST" if payload is not None else "GET"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def geniex_models() -> List[str]:
    try:
        data = _http_json(f"{GENIEX_URL}/models", timeout=5)
        return [str(item.get("id", "")) for item in data.get("data", []) if item.get("id")]
    except Exception:
        return []


def ollama_models() -> List[str]:
    try:
        data = _http_json(f"{OLLAMA_URL}/api/tags", timeout=6)
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def installed_models() -> List[str]:
    """Compatibility helper used by the GUI/status display."""
    hw = detect_hardware()
    if hw.is_snapdragon and hw.geniex_available:
        return geniex_models()
    return ollama_models()


def _contains(names: List[str], wanted: str) -> bool:
    return wanted in names or any(n.split("@", 1)[0] == wanted for n in names)


def _effective_capacity(profile: HardwareProfile) -> float:
    if profile.is_snapdragon or profile.vram_gb <= 0:
        return max(0.0, profile.ram_gb * 0.72 - 3.0)
    return max(0.0, min(profile.vram_gb * 0.95, profile.ram_gb * 0.72))


def _fits(profile: HardwareProfile, entry: Dict[str, Any]) -> bool:
    return _effective_capacity(profile) >= float(entry.get("size_gb", 0.0)) * 1.18


def recommend(profile: Optional[HardwareProfile] = None) -> Dict[str, Any]:
    profile = profile or detect_hardware()
    if profile.is_snapdragon and profile.geniex_available:
        quality = next((dict(x) for x in GENIEX_QUALITY_LADDER if _fits(profile, x)), dict(GENIEX_QUALITY_LADDER[-1]))
        fast = next((dict(x) for x in GENIEX_FAST_LADDER if _fits(profile, x)), dict(GENIEX_FAST_LADDER[-1]))
        if GENIEX_MODEL_OVERRIDE:
            quality = {"model": GENIEX_MODEL_OVERRIDE, "forced": True}
        if GENIEX_FAST_MODEL_OVERRIDE:
            fast = {"model": GENIEX_FAST_MODEL_OVERRIDE, "forced": True}
        vlm = next((dict(x) for x in GENIEX_VLM_LADDER if _fits(profile, x)), dict(GENIEX_VLM_LADDER[-1]))
        if GENIEX_VLM_MODEL_OVERRIDE:
            vlm = {"model": GENIEX_VLM_MODEL_OVERRIDE, "forced": True}
        return {
            "runtime_kind": "geniex",
            "hardware": profile.as_dict(),
            "quality": quality,
            "fast": fast,
            "vlm": vlm,
            "embedding": {"model": EMBED_MODEL},
            "runtime": {"fast": fast["model"], "quality": quality["model"], "vlm": vlm["model"], "embedding": EMBED_MODEL},
            "strategy": "snapdragon-first-geniex-npu-gpu-cpu",
        }

    # Generic fallback: keep the adaptive Ollama path.
    fitted = [dict(x) for x in OLLAMA_PRIMARY_LADDER if _fits(profile, x)]
    primary = fitted[0] if fitted else dict(OLLAMA_PRIMARY_LADDER[-1])
    if FORCE_MODEL:
        primary = {"model": FORCE_MODEL, "forced": True}
    fast = next((dict(x) for x in OLLAMA_FAST_LADDER if _fits(profile, x) and x["model"] != primary["model"]), dict(OLLAMA_FAST_LADDER[-1]))
    if FORCE_FAST:
        fast = {"model": FORCE_FAST, "forced": True}
    return {
        "runtime_kind": "ollama",
        "hardware": profile.as_dict(),
        "quality": primary,
        "fast": fast,
        "embedding": {"model": EMBED_MODEL},
        "runtime": {"fast": fast["model"], "quality": primary["model"], "embedding": EMBED_MODEL},
        "strategy": "generic-adaptive-local-fallback",
    }


def _image_data_url(path: str) -> Optional[str]:
    try:
        import base64, mimetypes
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def _openai_messages(messages: List[Dict[str, Any]], image_paths: Optional[List[str]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content")
        if isinstance(content, list):
            converted = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_text":
                    converted.append({"type": "text", "text": part.get("text", "")})
                elif isinstance(part, dict) and part.get("type") == "input_image":
                    url = part.get("image_url") or part.get("image_data")
                    if isinstance(url, str):
                        converted.append({"type": "image_url", "image_url": {"url": url}})
            m["content"] = converted or content
        prepared.append(m)
    if image_paths:
        parts = []
        for path in image_paths[:4]:
            url = _image_data_url(path)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        if parts and prepared:
            last = dict(prepared[-1])
            existing = last.get("content")
            if isinstance(existing, str):
                existing = [{"type": "text", "text": existing}]
            elif not isinstance(existing, list):
                existing = []
            last["content"] = existing + parts
            prepared[-1] = last
    return prepared


def _geniex_request(messages: List[Dict[str, Any]], model: str, temperature: float = 0.3,
                    num_predict: Optional[int] = None, image_paths: Optional[List[str]] = None,
                    timeout: int = 900) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": _openai_messages(messages, image_paths),
        "temperature": temperature,
        "stream": False,
    }
    # GenieX qairt currently has a known max_tokens quirk on some releases;
    # keep the field optional and let the runtime's model configuration decide
    # the safe generation ceiling rather than relying on a silently ignored cap.
    if num_predict and False:
        payload["max_tokens"] = num_predict
    data = _http_json(f"{GENIEX_URL}/chat/completions", payload, timeout=timeout)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("GenieX returned no choices.")
    message = choices[0].get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        raise RuntimeError("GenieX returned an empty response.")
    return text


def _geniex_stream(messages: List[Dict[str, Any]], model: str, temperature: float = 0.3,
                   num_predict: Optional[int] = None, image_paths: Optional[List[str]] = None,
                   timeout: int = 900) -> Iterable[str]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": _openai_messages(messages, image_paths),
        "temperature": temperature,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{GENIEX_URL}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if payload_text == "[DONE]":
                break
            try:
                obj = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content") or ""
                if chunk:
                    yield chunk



def _ollama_json(path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 300) -> Dict[str, Any]:
    return _http_json(f"{OLLAMA_URL}{path}", payload, timeout=timeout)


def _ollama_request(messages: List[Dict[str, Any]], model: str, temperature: float = 0.3,
                    num_ctx: int = 8192, num_predict: Optional[int] = None, timeout: int = 900) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "keep_alive": "15m",
        "options": {"temperature": temperature, "top_p": 0.92, "num_ctx": num_ctx},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    data = _ollama_json("/api/chat", payload, timeout=timeout)
    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        raise RuntimeError("The local AI returned an empty response.")
    return text


def _ollama_stream(messages: List[Dict[str, Any]], model: str, temperature: float = 0.3,
                   num_ctx: int = 8192, num_predict: Optional[int] = None, timeout: int = 900) -> Iterable[str]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "15m",
        "options": {"temperature": temperature, "top_p": 0.92, "num_ctx": num_ctx},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):
                raise RuntimeError(obj["error"])
            chunk = ((obj.get("message") or {}).get("content") or "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break


class AdaptiveModelRouter:
    """Stable backend interface for chat, semantic memory and Study Studio."""
    def __init__(self) -> None:
        self.profile = detect_hardware()
        self.plan = recommend(self.profile)
        self.runtime_kind = self.plan["runtime_kind"]
        self.fast_model = self.plan["runtime"]["fast"]
        self.quality_model = self.plan["runtime"]["quality"]
        self.deep_model = self.quality_model
        self.vlm_model = self.plan.get("runtime", {}).get("vlm") or self.plan.get("vlm", {}).get("model", self.quality_model)
        self.primary_model = self.quality_model
        self.embed_model = EMBED_MODEL
        self._ensure_geniex_server()

    def refresh(self) -> Dict[str, Any]:
        self.profile = detect_hardware()
        self.plan = recommend(self.profile)
        self.runtime_kind = self.plan["runtime_kind"]
        self.fast_model = self.plan["runtime"]["fast"]
        self.quality_model = self.plan["runtime"]["quality"]
        self.deep_model = self.quality_model
        self.vlm_model = self.plan.get("runtime", {}).get("vlm") or self.plan.get("vlm", {}).get("model", self.quality_model)
        self.primary_model = self.quality_model
        self.embed_model = EMBED_MODEL
        self._ensure_geniex_server()
        return self.plan

    def _ensure_geniex_server(self) -> None:
        if self.runtime_kind != "geniex":
            return
        try:
            self._geniex_server_ready()
            return
        except Exception:
            pass
        exe = _find_geniex()
        if not exe:
            return
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            subprocess.Popen([exe, "serve", "--host", "127.0.0.1"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=creationflags)
        except Exception:
            return
        for _ in range(20):
            try:
                if self._geniex_server_ready():
                    return
            except Exception:
                pass
            time.sleep(0.5)

    def _geniex_server_ready(self) -> bool:
        try:
            _http_json(f"{GENIEX_URL}/models", timeout=2)
            return True
        except Exception:
            return False

    def pick(self, mode: str = "fast", multimodal: bool = False) -> str:
        if multimodal and self.runtime_kind == "geniex":
            return self.vlm_model
        return self.quality_model if mode == "deep" else self.fast_model

    def status(self):
        p = self.profile
        if self.runtime_kind == "geniex":
            names = geniex_models()
            server_ok = bool(names) or self._geniex_server_ready()
            fast_ok = _contains(names, self.fast_model)
            quality_ok = _contains(names, self.quality_model)
            ok = fast_ok or quality_ok
            accel = "Snapdragon · GenieX"
            if p.qnn_available:
                accel += " · QNN/NPU available"
            detail = f"{accel} · fast {self.fast_model} · quality {self.quality_model} · vision {self.vlm_model}"
            if p.ram_gb:
                detail += f" · {p.ram_gb:.0f} GB RAM"
            return server_ok, ok, detail, {
                "runtime": "GenieX",
                "fast_installed": fast_ok,
                "quality_installed": quality_ok,
                "vision_installed": _contains(names, self.vlm_model),
                "embedding": self.embed_model,
                "plan": self.plan,
            }

        names = ollama_models()
        fast_ok = _contains(names, self.fast_model)
        quality_ok = _contains(names, self.quality_model)
        ok = fast_ok or quality_ok
        detail = f"Generic local · Ollama fallback · fast {self.fast_model} · quality {self.quality_model}"
        if p.gpu_names:
            detail += f" · {p.vram_gb:.0f} GB GPU memory"
        elif p.ram_gb:
            detail += f" · {p.ram_gb:.0f} GB RAM"
        return True, ok, detail, {
            "runtime": "Ollama fallback",
            "fast_installed": fast_ok,
            "quality_installed": quality_ok,
            "embedding": self.embed_model,
            "plan": self.plan,
        }

    def chat(self, messages: List[Dict[str, Any]], mode: str = "fast", temperature: float = 0.3,
             num_ctx: int = 8192, num_predict: Optional[int] = None,
             image_paths: Optional[List[str]] = None, timeout: int = 900) -> str:
        model = self.pick(mode, bool(image_paths))
        if self.runtime_kind == "geniex":
            text = _geniex_request(messages, model, temperature=temperature,
                                  num_predict=num_predict, image_paths=image_paths, timeout=timeout)
            if not text:
                raise RuntimeError("GenieX returned an empty response.")
            return text
        return _ollama_request(messages, model, temperature=temperature,
                               num_ctx=num_ctx, num_predict=num_predict, timeout=timeout)

    def stream(self, messages: List[Dict[str, Any]], mode: str = "fast", temperature: float = 0.3,
               num_ctx: int = 8192, num_predict: Optional[int] = None,
               image_paths: Optional[List[str]] = None, timeout: int = 900) -> Iterable[str]:
        model = self.pick(mode, bool(image_paths))
        if self.runtime_kind == "geniex":
            yield from _geniex_stream(messages, model, temperature=temperature,
                                      num_predict=num_predict, image_paths=image_paths, timeout=timeout)
        else:
            yield from _ollama_stream(messages, model, temperature=temperature,
                                      num_ctx=num_ctx, num_predict=num_predict, timeout=timeout)

    def embed(self, texts: List[str], timeout: int = 120):
        # Embeddings are intentionally independent from the chat runtime.
        # This keeps semantic memory local even when the main LLM uses GenieX.
        try:
            from semantic_embeddings import LocalSemanticEmbedder
            return LocalSemanticEmbedder.shared().encode(texts)
        except Exception as exc:
            print("Local semantic embedding unavailable:", exc)
            return None


def probe_runtime(model: str, runtime_kind: str = "ollama", timeout: int = 180) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        if runtime_kind == "geniex":
            text = _geniex_request([{"role": "user", "content": "Reply with exactly READY."}], model,
                                temperature=0.0, num_predict=4, timeout=timeout)
        else:
            text = _ollama_request([{"role": "user", "content": "Reply with exactly READY."}], model,
                                temperature=0.0, num_ctx=512, num_predict=4, timeout=timeout)
        elapsed = time.perf_counter() - start
        return {"ok": bool(text), "seconds": round(elapsed, 3)}
    except Exception as exc:
        return {"ok": False, "seconds": round(time.perf_counter() - start, 3), "error": str(exc)}
