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

import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn
import webview

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5vl:3b"
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
    # Try PATH
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
            import json
            data = json.loads(r.read())
            models = [m.get("name", "") for m in data.get("models", [])]
            return any(model in m for m in models)
    except Exception:
        return False


# ── Startup ────────────────────────────────────────────────────────────────────


def start_ollama() -> subprocess.Popen | None:
    # Already running?
    if is_port_open(11434):
        return None

    ollama_bin = find_ollama()
    if not ollama_bin:
        return None

    proc = subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def ensure_model(ollama_bin: str | None):
    if ollama_bin is None:
        return
    if ollama_has_model(OLLAMA_MODEL):
        return
    subprocess.run([ollama_bin, "pull", OLLAMA_MODEL], check=False)


def start_server(port: int):
    from server.api import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ── Loading splash (shown before server is ready) ──────────────────────────────

SPLASH_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: #0f0f0f;
  color: #e8e2d9;
  font-family: 'DM Mono', 'Courier New', monospace;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100vh;
  flex-direction: column;
  gap: 18px;
}
h1 {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 3rem;
  font-weight: normal;
  color: #e8e2d9;
}
p { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: #6b6560; }
.spinner {
  width: 32px; height: 32px;
  border: 2px solid #2a2a2a;
  border-top-color: #c9a84c;
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
  <div class="spinner"></div>
  <h1>Frames</h1>
  <p>Starting up…</p>
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    port = find_free_port()
    server_url = f"http://127.0.0.1:{port}"

    # Start ollama
    ollama_proc = start_ollama()
    ollama_bin = find_ollama()

    if ollama_proc is not None:
        # Wait for ollama to be ready
        wait_for_url(OLLAMA_BASE, timeout=30)

    # Pull model if needed (non-blocking: do after window opens)
    def pull_model_bg():
        if is_port_open(11434):
            ensure_model(ollama_bin)

    # Start FastAPI in background thread
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    # Wait for FastAPI
    wait_for_url(f"{server_url}/api/status", timeout=20)

    # Kick off model pull in background
    threading.Thread(target=pull_model_bg, daemon=True).start()

    # Open window
    window = webview.create_window(
        "Frames",
        server_url,
        width=1400,
        height=900,
        min_size=(900, 600),
        background_color="#0f0f0f",
    )
    webview.start()

    # Cleanup
    if ollama_proc is not None:
        ollama_proc.terminate()


if __name__ == "__main__":
    main()
