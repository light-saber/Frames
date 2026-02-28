from __future__ import annotations

import asyncio
import concurrent.futures
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analysis import check_ollama_alive, run_analysis
from .export import export_photos
from .models import (
    DEFAULT_COLOR_SETTINGS,
    OLLAMA_MODEL,
    RAW_EXTENSIONS,
    THUMB_DIR,
    _state,
    init_state,
    save_session,
)

# Resolve frontend dir — works both in development and inside PyInstaller bundle
if getattr(sys, "frozen", False):
    FRONTEND_DIR = Path(sys._MEIPASS) / "frontend"
else:
    FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="Frames")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

init_state()


# ── Pydantic request bodies ────────────────────────────────────────────────────


class FolderRequest(BaseModel):
    path: str


class StatusPatch(BaseModel):
    status: str  # keep | reject | pending


class SettingsPatch(BaseModel):
    settings: dict


class ThresholdRequest(BaseModel):
    threshold: int
    mode: str = "pending_only"  # "pending_only" | "all"


# ── SSE helper ────────────────────────────────────────────────────────────────


async def _run_in_executor_sse(fn, *args):
    """Run fn(*args) in a thread, yielding SSE data lines from a shared queue.

    fn must accept a progress_callback(done, total, item, error) as its last argument.
    Yields formatted SSE strings; sends a final ``{"done": true}`` when complete.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_callback(*cb_args):
        asyncio.run_coroutine_threadsafe(queue.put(cb_args), loop)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = loop.run_in_executor(executor, fn, *args, progress_callback)

    while True:
        try:
            cb_args = await asyncio.wait_for(queue.get(), timeout=0.2)
            yield f"data: {json.dumps(cb_args)}\n\n"
        except asyncio.TimeoutError:
            if future.done():
                while not queue.empty():
                    cb_args = queue.get_nowait()
                    yield f"data: {json.dumps(cb_args)}\n\n"
                yield 'data: {"done": true}\n\n'
                break


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text()


@app.get("/api/status")
async def status():
    return {"ok": True}


@app.get("/api/ollama-status")
async def ollama_status():
    alive = check_ollama_alive()
    _state["ollama_available"] = alive
    return {"alive": alive, "model": OLLAMA_MODEL}


@app.get("/api/photos")
async def get_photos():
    return JSONResponse([asdict(a) for a in _state["analyses"]])


@app.get("/api/photo/{filename}/thumbnail")
async def get_thumbnail(filename: str):
    thumb_path = THUMB_DIR / f"{Path(filename).stem}.jpg"
    if not thumb_path.exists():
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(str(thumb_path), media_type="image/jpeg")


@app.post("/api/folder")
async def set_folder(req: FolderRequest):
    p = Path(req.path)
    if not p.is_dir():
        raise HTTPException(400, f"Not a directory: {req.path}")
    _state["folder"] = req.path
    raw_files = sorted([f.name for f in p.iterdir() if f.suffix.lower() in RAW_EXTENSIONS])
    return {"path": req.path, "raw_count": len(raw_files), "files": raw_files}


@app.post("/api/folder-dialog")
async def folder_dialog():
    try:
        result = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "Select folder")'],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            path = result.stdout.strip().rstrip("/")
            _state["folder"] = path
            return {"path": path}
    except Exception:
        pass
    return {"path": ""}


@app.get("/api/analyze")
async def analyze_sse():
    folder = _state.get("folder", "")
    use_ai = _state.get("use_ai", False)
    if not folder or not Path(folder).is_dir():
        raise HTTPException(400, "No valid folder set")

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_callback(done, total, photo, error):
        event_data = {"done": done, "total": total}
        if photo:
            event_data["photo"] = asdict(photo)
        if error:
            event_data["error"] = error
        asyncio.run_coroutine_threadsafe(queue.put(event_data), loop)

    async def event_stream():
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(executor, run_analysis, folder, use_ai, progress_callback)
        async for chunk in _drain_queue(queue, future):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/export")
async def export_sse(export_folder: str):
    if not Path(export_folder).is_dir():
        raise HTTPException(400, f"Invalid export folder: {export_folder}")
    _state["export_folder"] = export_folder
    analyses = list(_state["analyses"])
    settings = dict(_state["color_settings"])

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def progress_callback(done, total, filename, error):
        event_data = {"done": done, "total": total, "filename": filename}
        if error:
            event_data["error"] = error
        asyncio.run_coroutine_threadsafe(queue.put(event_data), loop)

    async def event_stream():
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = loop.run_in_executor(
            executor, export_photos, analyses, export_folder, settings, progress_callback
        )
        async for chunk in _drain_queue(queue, future):
            yield chunk

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _drain_queue(queue: asyncio.Queue, future):
    """Yield SSE data lines from queue until future completes."""
    while True:
        try:
            data = await asyncio.wait_for(queue.get(), timeout=0.2)
            yield f"data: {json.dumps(data)}\n\n"
        except asyncio.TimeoutError:
            if future.done():
                while not queue.empty():
                    yield f"data: {json.dumps(queue.get_nowait())}\n\n"
                yield 'data: {"done": true}\n\n'
                break


@app.patch("/api/photo/{filename}/status")
async def update_status(filename: str, body: StatusPatch):
    if body.status not in ("keep", "reject", "pending"):
        raise HTTPException(400, "Invalid status")
    for a in _state["analyses"]:
        if a.filename == filename:
            a.status = body.status
            save_session()
            return {"filename": filename, "status": body.status}
    raise HTTPException(404, "Photo not found")


@app.get("/api/settings")
async def get_settings():
    return _state["color_settings"]


@app.patch("/api/settings")
async def update_settings(body: SettingsPatch):
    _state["color_settings"].update(body.settings)
    return _state["color_settings"]


@app.post("/api/settings/reset")
async def reset_settings():
    _state["color_settings"] = DEFAULT_COLOR_SETTINGS.copy()
    return _state["color_settings"]


@app.post("/api/batch/threshold")
async def apply_threshold(req: ThresholdRequest):
    targets = _state["analyses"] if req.mode == "all" else [
        a for a in _state["analyses"] if a.status == "pending"
    ]
    for a in targets:
        a.status = "keep" if a.overall_score >= req.threshold else "reject"
    save_session()
    return {"updated": len(targets)}


@app.post("/api/batch/keep-all")
async def keep_all():
    for a in _state["analyses"]:
        a.status = "keep"
    save_session()
    return {"updated": len(_state["analyses"])}


@app.post("/api/batch/reset")
async def reset_all():
    for a in _state["analyses"]:
        a.status = "pending"
    save_session()
    return {"updated": len(_state["analyses"])}


@app.post("/api/use-ai")
async def set_use_ai(body: dict):
    _state["use_ai"] = bool(body.get("enabled", False))
    return {"use_ai": _state["use_ai"]}
