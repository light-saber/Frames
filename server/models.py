from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SESSION_FILE = Path.home() / ".frames_session.json"

DEFAULT_COLOR_SETTINGS = {
    "brightness": 0.0,
    "contrast": 1.05,
    "saturation_boost": 1.1,
    "highlight_recovery": 0.12,
    "shadow_lift": 0.03,
    "sharpening": 0.3,
}

RAW_EXTENSIONS = {".nef", ".nrw", ".raw", ".cr2", ".cr3", ".arw", ".dng"}
THUMB_DIR = Path("/tmp/frames_thumbs")
OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5vl:3b"


@dataclass
class PhotoAnalysis:
    path: str
    filename: str
    sharpness: float
    exposure: float
    saturation: float
    is_duplicate: bool
    duplicate_of: Optional[str]
    overall_score: float
    status: str  # 'pending' | 'keep' | 'reject'
    thumbnail_path: str
    ai_score: Optional[float] = field(default=None)
    ai_reason: Optional[str] = field(default=None)


# Module-level session state
_state: dict = {
    "analyses": [],
    "analyzed": False,
    "color_settings": DEFAULT_COLOR_SETTINGS.copy(),
    "ollama_available": False,
    "folder": "",
    "export_folder": "",
    "use_ai": False,
    "sort_by": "score_desc",
}


def get_state() -> dict:
    return _state


def save_session():
    try:
        SESSION_FILE.write_text(json.dumps([asdict(a) for a in _state["analyses"]]))
    except Exception:
        pass


def load_session() -> list[PhotoAnalysis]:
    try:
        data = json.loads(SESSION_FILE.read_text())
        return [PhotoAnalysis(**d) for d in data]
    except Exception:
        return []


def init_state():
    loaded = load_session()
    _state["analyses"] = loaded
    _state["analyzed"] = bool(loaded)
    _state["color_settings"] = DEFAULT_COLOR_SETTINGS.copy()
