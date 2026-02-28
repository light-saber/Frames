"""
Frames — entry point.

Startup sequence:
  1. Find a free port (8000-8999)
  2. Launch ollama serve (if not already running)
  3. Poll until ollama is ready (max 30s)
  4. Pull qwen2.5vl:3b if absent
  5. Start FastAPI/uvicorn in a background thread
  6. Poll until FastAPI is ready
  7. Open pywebview window
  8. On window close, terminate subprocesses
"""
from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
import webview

from server.models import OLLAMA_BASE, OLLAMA_MODEL

OLLAMA_PORT = 11434
OLLAMA_CANDIDATES = [
    "/opt/homebrew/bin/ollama",
    "/usr/local/bin/ollama",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def find_free_port(start: int = 8000, end: int = 8999) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free port found")


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def find_ollama() -> str | None:
    for path in OLLAMA_CANDIDATES:
        if Path(path).exists():
            return path
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def wait_for_url(url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def ollama_has_model(model: str) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=5) as r:
            data = json.loads(r.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(model in m for m in models)
    except Exception:
        return False


# ── Startup ────────────────────────────────────────────────────────────────────


def start_ollama() -> subprocess.Popen | None:
    if is_port_open(OLLAMA_PORT):
        return None  # already running
    ollama_bin = find_ollama()
    if not ollama_bin:
        return None
    return subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def ensure_model(ollama_bin: str | None):
    if ollama_bin is None:
        return
    if not ollama_has_model(OLLAMA_MODEL):
        subprocess.run([ollama_bin, "pull", OLLAMA_MODEL], check=False)


def start_server(port: int):
    from server.api import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    port = find_free_port()
    server_url = f"http://127.0.0.1:{port}"

    ollama_proc = start_ollama()
    ollama_bin = find_ollama()

    if ollama_proc is not None:
        wait_for_url(OLLAMA_BASE, timeout=30)

    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()
    wait_for_url(f"{server_url}/api/status", timeout=20)

    # Pull model in background so it doesn't block the window opening
    threading.Thread(
        target=ensure_model, args=(ollama_bin,), daemon=True
    ).start()

    webview.create_window(
        "Frames",
        server_url,
        width=1400,
        height=900,
        min_size=(900, 600),
        background_color="#0f0f0f",
    )
    webview.start()

    if ollama_proc is not None:
        ollama_proc.terminate()


if __name__ == "__main__":
    main()
