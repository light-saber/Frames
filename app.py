from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rawpy
import streamlit as st
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────

THUMB_DIR = Path("/tmp/frames_thumbs")
RAW_EXTENSIONS = {".nef", ".nrw", ".raw", ".cr2", ".cr3", ".arw", ".dng"}
SESSION_FILE = Path.home() / ".frames_session.json"

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5vl:3b"

DEFAULT_COLOR_SETTINGS = {
    "brightness": 0.0,
    "contrast": 1.05,
    "saturation_boost": 1.1,
    "highlight_recovery": 0.12,
    "shadow_lift": 0.03,
    "sharpening": 0.3,
}

SORT_OPTIONS = {
    "score_desc": "Score ↓",
    "score_asc":  "Score ↑",
    "name_asc":   "Name A–Z",
    "name_desc":  "Name Z–A",
}

EMPTY_STATES = {
    "pending":   ("Nothing left to decide",  "All photos have been marked keep or reject."),
    "keep":      ("No photos kept yet",       "Mark photos with ✓ Keep to see them here."),
    "reject":    ("No photos rejected",       "Mark photos with ✗ Reject to see them here."),
    "duplicate": ("No duplicates found",      "The perceptual hash analysis found no near-identical shots."),
    "all":       ("No photos found",          "Check that the folder contains supported RAW files."),
}

# ─── Data Model ───────────────────────────────────────────────────────────────


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


# ─── Session Persistence ──────────────────────────────────────────────────────


def save_session():
    try:
        SESSION_FILE.write_text(json.dumps([asdict(a) for a in st.session_state.analyses]))
    except Exception:
        pass


def load_session() -> list[PhotoAnalysis]:
    try:
        data = json.loads(SESSION_FILE.read_text())
        return [PhotoAnalysis(**d) for d in data]
    except Exception:
        return []


# ─── Session State ────────────────────────────────────────────────────────────


def init_state():
    # Apply browse-button picks BEFORE any widgets are instantiated
    if "_folder_pending" in st.session_state:
        st.session_state.folder = st.session_state.pop("_folder_pending")
    if "_export_pending" in st.session_state:
        st.session_state.export_folder = st.session_state.pop("_export_pending")

    if "analyses" not in st.session_state:
        loaded = load_session()
        st.session_state.analyses = loaded
        st.session_state.analyzed = bool(loaded)
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "color_settings" not in st.session_state:
        st.session_state.color_settings = DEFAULT_COLOR_SETTINGS.copy()
    if "export_done" not in st.session_state:
        st.session_state.export_done = False
    if "ollama_available" not in st.session_state:
        st.session_state.ollama_available = False
    if "ollama_checked" not in st.session_state:
        st.session_state.ollama_checked = False
    if "sort_by" not in st.session_state:
        st.session_state.sort_by = "score_desc"


# ─── RAW Loading ──────────────────────────────────────────────────────────────


def load_rgb(path: str) -> tuple[np.ndarray, bool]:
    """Load a RAW file as an 8-bit RGB numpy array.

    Returns (rgb_array, used_embedded_jpeg).
    Falls back to the embedded full-resolution JPEG when LibRaw cannot decode
    the RAW data (e.g. Nikon Zf, which was released after LibRaw 0.22.0).
    The embedded JPEG is the same pixel dimensions as the RAW sensor output.
    """
    try:
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)
        return rgb, False
    except (
        rawpy.LibRawFileUnsupportedError,
        rawpy.LibRawNotImplementedError,
        rawpy.LibRawDataError,
    ):
        with rawpy.imread(path) as raw:
            thumb = raw.extract_thumb()
        img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
        return np.array(img), True


# ─── Finder Integration ───────────────────────────────────────────────────────


def pick_folder_dialog() -> str:
    """Open a native macOS folder picker via AppleScript. Returns POSIX path or ''."""
    try:
        result = subprocess.run(
            [
                "osascript", "-e",
                'POSIX path of (choose folder with prompt "Select folder")',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return result.stdout.strip().rstrip("/")
    except Exception:
        pass
    return ""


# ─── Analysis Pipeline ────────────────────────────────────────────────────────


def perceptual_hash(gray_uint8: np.ndarray) -> str:
    """Compute a simple perceptual hash for duplicate detection."""
    small = cv2.resize(gray_uint8, (16, 16), interpolation=cv2.INTER_AREA)
    median = np.median(small)
    bits = (small > median).flatten()
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def hamming_distance(h1: str, h2: str) -> int:
    b1 = bytes.fromhex(h1)
    b2 = bytes.fromhex(h2)
    return sum(bin(a ^ b).count("1") for a, b in zip(b1, b2))


def check_ollama_alive() -> bool:
    """Return True if Ollama is reachable at OLLAMA_BASE."""
    try:
        with urllib.request.urlopen(OLLAMA_BASE, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def score_photo_with_ai(thumbnail_path: str) -> tuple[Optional[float], Optional[str]]:
    """Send thumbnail to Qwen2.5-VL via Ollama and return (ai_score, reason) or (None, None)."""
    try:
        with open(thumbnail_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        prompt = (
            "You are a professional photography critic evaluating a photo for technical quality.\n"
            "Analyse this image and return ONLY a JSON object with exactly these keys:\n"
            '  "composition": <integer 0-100>,\n'
            '  "lighting": <integer 0-100>,\n'
            '  "subject_clarity": <integer 0-100>,\n'
            '  "overall": <integer 0-100>,\n'
            '  "reason": "<one sentence max 20 words>"\n'
            "Do not include any text before or after the JSON.\n"
            "Score 0=very poor, 50=average, 100=excellent."
        )

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 80,
                "num_ctx": 512,
            },
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read())

        response_text = raw.get("response", "").strip()

        # Strip markdown fences if present
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            response_text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        data = json.loads(response_text)
        overall = float(data["overall"])
        composition = float(data["composition"])
        lighting = float(data["lighting"])
        subject_clarity = float(data["subject_clarity"])
        reason = str(data.get("reason", ""))

        ai_score = overall * 0.50 + composition * 0.20 + lighting * 0.20 + subject_clarity * 0.10
        return float(ai_score), reason

    except Exception:
        return None, None


def analyse_photo(path: str, existing_hashes: dict[str, str], use_ai: bool = False) -> PhotoAnalysis:
    filename = Path(path).name
    stem = Path(path).stem

    # Step 1: Load RAW (falls back to embedded JPEG if camera not supported by LibRaw)
    rgb, _from_jpeg = load_rgb(path)

    # Step 2: Thumbnail
    pil_img = Image.fromarray(rgb)
    thumb = pil_img.copy()
    thumb.thumbnail((480, 320), Image.LANCZOS)
    thumb_path = str(THUMB_DIR / f"{stem}.jpg")
    thumb.save(thumb_path, "JPEG", quality=85)

    h, w = rgb.shape[:2]

    # Step 3: Sharpness
    center = rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    gray_center = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_center, cv2.CV_64F)
    lap_var = laplacian.var()
    sharpness = float(min(100.0, lap_var / 5.0))

    # Step 4: Exposure
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    overexposed = np.mean(gray > 240)
    underexposed = np.mean(gray < 15)
    brightness_score = 100 - abs(mean_brightness - 128) / 1.28
    clipping_penalty = (overexposed + underexposed) * 200
    exposure = float(max(0.0, min(100.0, brightness_score - clipping_penalty)))

    # Step 5: Saturation
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mean_sat = hsv[:, :, 1].mean()
    saturation = float(mean_sat / 2.55)

    # Step 6: AI scoring (optional)
    ai_score, ai_reason = (None, None)
    if use_ai:
        ai_score, ai_reason = score_photo_with_ai(thumb_path)

    # Step 7: Overall score — weights shift when AI available
    sat_fitness = float(max(0, min(100, 100 - abs(saturation - 42) * 2)))
    if ai_score is not None:
        overall_score = float(sharpness * 0.35 + exposure * 0.20 + sat_fitness * 0.10 + ai_score * 0.35)
    else:
        overall_score = float(sharpness * 0.55 + exposure * 0.30 + sat_fitness * 0.15)

    # Step 8: Duplicate detection
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    phash = perceptual_hash(gray_full)
    is_duplicate = False
    duplicate_of = None
    for other_filename, other_hash in existing_hashes.items():
        if hamming_distance(phash, other_hash) < 10:
            is_duplicate = True
            duplicate_of = other_filename
            break
    existing_hashes[filename] = phash

    return PhotoAnalysis(
        path=path,
        filename=filename,
        sharpness=sharpness,
        exposure=exposure,
        saturation=saturation,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        overall_score=overall_score,
        status="pending",
        thumbnail_path=thumb_path,
        ai_score=ai_score,
        ai_reason=ai_reason,
    )


def run_analysis(folder: str, use_ai: bool = False):
    folder_path = Path(folder)
    if not folder or not folder_path.is_dir():
        st.error(f"Invalid folder path: {folder}")
        return

    raw_files = sorted([
        f for f in folder_path.iterdir()
        if f.suffix.lower() in RAW_EXTENSIONS
    ])

    if not raw_files:
        st.warning("No supported RAW files found in this folder.")
        return

    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    results: list[PhotoAnalysis] = []
    hashes: dict[str, str] = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, raw_file in enumerate(raw_files):
        suffix = " + AI…" if use_ai else "…"
        status_text.text(f"Analysing {raw_file.name} ({i + 1}/{len(raw_files)}){suffix}")
        try:
            analysis = analyse_photo(str(raw_file), hashes, use_ai=use_ai)
            results.append(analysis)
        except Exception as e:
            st.warning(f"Could not read {raw_file.name}: {e}")
        progress_bar.progress((i + 1) / len(raw_files))

    status_text.text("Analysis complete!")

    # Sort by overall score descending
    results.sort(key=lambda x: x.overall_score, reverse=True)

    # Auto-flag duplicates
    for r in results:
        if r.is_duplicate:
            r.status = "reject"

    st.session_state.analyses = results
    st.session_state.analyzed = True
    save_session()


# ─── Colour Correction Pipeline ───────────────────────────────────────────────


def apply_color_correction(rgb: np.ndarray, settings: dict) -> np.ndarray:
    """Apply colour correction to a uint8 RGB array. Returns uint8."""
    img = rgb.astype(np.float32) / 255.0

    brightness = settings["brightness"]
    contrast = settings["contrast"]
    sat_factor = settings["saturation_boost"]
    hr = settings["highlight_recovery"]
    sl = settings["shadow_lift"]
    amount = settings["sharpening"]

    # 1. Highlight recovery
    img = np.where(img > (1 - hr), (1 - hr) + (img - (1 - hr)) * 0.3, img)

    # 2. Shadow lift
    img = img + sl * (1 - img) * (1 - img)

    # 3. Brightness + contrast
    img = np.clip((img + brightness - 0.5) * contrast + 0.5, 0, 1)

    # 4. Saturation (via HSV)
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # 5. Sharpening (unsharp mask)
    img_uint8 = (img * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (0, 0), 2)
    img_uint8 = cv2.addWeighted(img_uint8, 1 + amount, blurred, -amount, 0)

    return img_uint8


def export_photos(export_folder: str, settings: dict):
    export_path = Path(export_folder)
    if not export_folder or not export_path.is_dir():
        st.error(f"Invalid export folder: {export_folder}")
        return

    to_export = [a for a in st.session_state.analyses if a.status == "keep"]
    if not to_export:
        st.warning("No photos marked as 'keep' to export.")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, analysis in enumerate(to_export):
        status_text.text(f"Exporting {analysis.filename} ({i + 1}/{len(to_export)})…")
        try:
            rgb, _from_jpeg = load_rgb(analysis.path)
            corrected = apply_color_correction(rgb, settings)
            stem = Path(analysis.path).stem
            out_path = export_path / f"{stem}_edited.tiff"
            Image.fromarray(corrected).save(str(out_path), "TIFF")
        except Exception as e:
            st.warning(f"Could not export {analysis.filename}: {e}")
        progress_bar.progress((i + 1) / len(to_export))

    status_text.text(f"Export complete! {len(to_export)} photos saved to {export_folder}")
    st.session_state.export_done = True


# ─── CSS / Styling ────────────────────────────────────────────────────────────

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg-app:        #0f0f0f;
    --bg-card:       #181818;
    --bg-surface:    #1c1c1c;
    --border-subtle: #242424;
    --border-card:   #2a2a2a;
    --text-primary:  #e8e2d9;
    --text-muted:    #6b6560;
    --text-dimmed:   #3d3836;
    --accent-gold:   #c9a84c;
    --green-keep:    #4ade80;
    --red-reject:    #f87171;
    --blue-exposure: #60a5fa;
    --purple-dupe:   #a78bfa;
    --amber-pending: #fbbf24;
}

/* DM Mono everywhere */
html, body, button, input, select, textarea,
[data-testid="stApp"], [data-testid="stSidebar"],
.stMarkdown, label, p, li {
    font-family: 'DM Mono', monospace !important;
}
/* Target spans but not Streamlit icon spans (Material Symbols) */
span:not([data-testid="stExpanderToggleIcon"]):not(.material-symbols-rounded):not(.stIconMaterial) {
    font-family: 'DM Mono', monospace !important;
}
/* Restore Material Symbols font for icons */
[data-testid="stExpanderToggleIcon"],
.material-symbols-rounded,
.stIconMaterial {
    font-family: 'Material Symbols Rounded' !important;
}

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

/* Buttons */
.stButton > button {
    background-color: transparent !important;
    border: 1px solid rgba(240,236,232,0.18) !important;
    font-family: 'DM Mono', monospace !important;
    border-radius: 6px !important;
    transition: border-color 0.15s, color 0.15s;
}
.stButton > button:hover {
    border-color: var(--accent-gold) !important;
    color: var(--accent-gold) !important;
}

/* Tighter column padding */
[data-testid="column"] { padding: 0 4px !important; }

/* Sidebar padding */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

/* ── Photo card ─────────────────────────────────────────────────────────── */

.photo-card {
    background: var(--bg-surface);
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 4px;
    border: 2px solid var(--border-card);
}

/* Thumbnail wrapper with hover overlay */
.thumb-wrap {
    position: relative;
    border-radius: 4px;
    overflow: hidden;
}
.thumb-wrap img {
    width: 100%;
    display: block;
}

/* Score badge — absolute overlay, top-right */
.score-overlay {
    position: absolute;
    top: 6px;
    right: 6px;
    background: rgba(0,0,0,0.55);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    border-radius: 4px;
    padding: 2px 7px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    line-height: 1.5;
}

/* Metrics panel — revealed on hover */
.card-metrics {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, rgba(0,0,0,0.82));
    padding: 20px 8px 8px 8px;
    opacity: 0;
    transition: opacity 0.18s ease;
}
.thumb-wrap:hover .card-metrics { opacity: 1; }

.metric-row {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 3px;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    color: rgba(255,255,255,0.65);
    width: 14px;
    text-align: right;
    flex-shrink: 0;
}
.metric-bar-bg {
    flex: 1;
    height: 3px;
    background: rgba(255,255,255,0.15);
    border-radius: 2px;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 2px;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    color: rgba(255,255,255,0.65);
    width: 22px;
    text-align: right;
    flex-shrink: 0;
}

/* Card footer */
.card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 2px 2px 2px;
}
.filename {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.dup-tag {
    background: #2a1a3a;
    color: var(--purple-dupe);
    border: 1px solid #5b21b6;
    border-radius: 20px;
    padding: 1px 7px;
    font-size: 0.6rem;
    font-family: 'DM Mono', monospace;
    flex-shrink: 0;
    margin-left: 4px;
}

/* ── Stats strip ────────────────────────────────────────────────────────── */

.stats-strip {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    padding: 6px 0 10px 0;
    border-bottom: 1px solid var(--border-subtle);
}
.stat-item-num {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: var(--text-primary);
}
.stat-item-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-left: 3px;
    margin-right: 14px;
}

/* ── Feature cards (welcome screen) ────────────────────────────────────── */

.feature-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-card);
    border-radius: 8px;
    padding: 18px 20px;
    height: 100%;
}
.feature-icon { font-size: 1.3rem; margin-bottom: 8px; }
.feature-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: var(--text-primary);
    margin-bottom: 5px;
}
.feature-desc {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Empty states ───────────────────────────────────────────────────────── */

.empty-state {
    text-align: center;
    padding: 64px 20px;
}
.empty-icon {
    font-size: 2rem;
    color: var(--text-dimmed);
    margin-bottom: 14px;
}
.empty-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.empty-desc {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    color: var(--text-dimmed);
}

/* ── Tab overrides ──────────────────────────────────────────────────────── */

[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent-gold) !important;
}
[data-testid="stTabs"] [role="tablist"] {
    border-bottom-color: var(--border-subtle) !important;
}

</style>
"""

# ─── Helper rendering ─────────────────────────────────────────────────────────


def render_step_indicator(current_step: int):
    """Render the 4-step workflow indicator in the sidebar."""
    steps = [(1, "Load"), (2, "Analyse"), (3, "Cull"), (4, "Export")]

    items = []
    for i, (num, label) in enumerate(steps):
        if num < current_step:
            dot = "✓"
            dot_color = "var(--green-keep)"
            lbl_color = "var(--text-muted)"
        elif num == current_step:
            dot = "●"
            dot_color = "var(--accent-gold)"
            lbl_color = "var(--accent-gold)"
        else:
            dot = "○"
            dot_color = "var(--text-dimmed)"
            lbl_color = "var(--text-dimmed)"

        item = (
            f'<div style="display:flex;flex-direction:column;align-items:center;flex:1;">'
            f'<span style="font-size:0.75rem;color:{dot_color};">{dot}</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.5rem;'
            f'text-transform:uppercase;letter-spacing:0.06em;color:{lbl_color};'
            f'margin-top:3px;">{label}</span>'
            f'</div>'
        )
        items.append(item)
        if i < len(steps) - 1:
            items.append(
                '<div style="height:1px;width:12px;flex:none;background:var(--border-subtle);'
                'margin-bottom:14px;"></div>'
            )

    html = (
        '<div style="display:flex;align-items:center;'
        'padding:10px 0 14px 0;margin-bottom:4px;">'
        + "".join(items)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def get_current_step() -> int:
    folder = st.session_state.get("folder", "")
    folder_set = bool(folder) and Path(folder).is_dir()
    analyzed = st.session_state.get("analyzed", False)
    has_keeps = any(a.status == "keep" for a in st.session_state.get("analyses", []))

    if not folder_set:
        return 1
    if not analyzed:
        return 2
    if not has_keeps:
        return 3
    return 4


def render_stats_strip(analyses: list[PhotoAnalysis]) -> str:
    """Return HTML for the compact stats strip."""
    total = len(analyses)
    kept = sum(1 for a in analyses if a.status == "keep")
    rejected = sum(1 for a in analyses if a.status == "reject")
    pending = sum(1 for a in analyses if a.status == "pending")
    dupes = sum(1 for a in analyses if a.is_duplicate)
    avg_score = sum(a.overall_score for a in analyses) / total if total else 0

    def item(num, label):
        return (
            f'<span class="stat-item-num">{num}</span>'
            f'<span class="stat-item-lbl">{label}</span>'
        )

    return (
        '<div class="stats-strip">'
        + item(total, "photos")
        + item(kept, "kept")
        + item(rejected, "rejected")
        + item(pending, "pending")
        + item(dupes, "dupes")
        + item(f"{avg_score:.0f}", "avg score")
        + "</div>"
    )


def render_empty_state(filter_key: str) -> str:
    """Return HTML for an empty-state panel."""
    title, desc = EMPTY_STATES.get(filter_key, ("No photos", ""))
    return (
        '<div class="empty-state">'
        '<div class="empty-icon">◻</div>'
        f'<div class="empty-title">{title}</div>'
        f'<div class="empty-desc">{desc}</div>'
        "</div>"
    )


def score_badge_overlay(score: float) -> str:
    """Return the color for the score overlay badge."""
    if score >= 70:
        return "var(--green-keep)"
    elif score >= 45:
        return "var(--amber-pending)"
    return "var(--red-reject)"


def sort_analyses(analyses: list[PhotoAnalysis], sort_by: str) -> list[PhotoAnalysis]:
    if sort_by == "score_desc":
        return sorted(analyses, key=lambda a: a.overall_score, reverse=True)
    elif sort_by == "score_asc":
        return sorted(analyses, key=lambda a: a.overall_score)
    elif sort_by == "name_asc":
        return sorted(analyses, key=lambda a: a.filename.lower())
    elif sort_by == "name_desc":
        return sorted(analyses, key=lambda a: a.filename.lower(), reverse=True)
    return list(analyses)


def _metric_row(label: str, value: float, color: str) -> str:
    pct = min(100, max(0, value))
    return (
        '<div class="metric-row">'
        f'<span class="metric-label">{label}</span>'
        '<div class="metric-bar-bg">'
        f'<div class="metric-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>'
        "</div>"
        f'<span class="metric-val">{pct:.0f}</span>'
        "</div>"
    )


def render_photo_card(analysis: PhotoAnalysis, col):
    with col:
        border_colors = {
            "keep":    "var(--green-keep)",
            "reject":  "var(--red-reject)",
            "pending": "var(--border-card)",
        }
        border_color = border_colors.get(analysis.status, "var(--border-card)")

        fname_short = (
            analysis.filename[:22] + "…"
            if len(analysis.filename) > 22
            else analysis.filename
        )
        dup_html = '<span class="dup-tag">dup</span>' if analysis.is_duplicate else ""

        score_color = score_badge_overlay(analysis.overall_score)

        # Build thumbnail with overlay badge and hover metrics
        thumb_html = ""
        if os.path.exists(analysis.thumbnail_path):
            with open(analysis.thumbnail_path, "rb") as f:
                thumb_b64 = base64.b64encode(f.read()).decode()

            metrics = _metric_row("S", analysis.sharpness, "var(--accent-gold)")
            metrics += _metric_row("E", analysis.exposure, "var(--blue-exposure)")
            if analysis.ai_score is not None:
                metrics += _metric_row("AI", analysis.ai_score, "var(--purple-dupe)")

            thumb_html = (
                '<div class="thumb-wrap">'
                f'<img src="data:image/jpeg;base64,{thumb_b64}" style="width:100%;display:block;">'
                f'<div class="score-overlay" style="color:{score_color};">'
                f'{analysis.overall_score:.0f}</div>'
                f'<div class="card-metrics">{metrics}</div>'
                "</div>"
            )

        st.markdown(
            f'<div class="photo-card" style="border-color:{border_color};">'
            f"{thumb_html}"
            f'<div class="card-footer">'
            f'<span class="filename">{fname_short}</span>'
            f"{dup_html}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            keep_label = "↩ kept" if analysis.status == "keep" else "✓ Keep"
            if st.button(keep_label, key=f"keep_{analysis.filename}", use_container_width=True):
                analysis.status = "pending" if analysis.status == "keep" else "keep"
                save_session()
                st.rerun()
        with btn_col2:
            rej_label = "↩ unrej" if analysis.status == "reject" else "✗ Reject"
            if st.button(rej_label, key=f"reject_{analysis.filename}", use_container_width=True):
                analysis.status = "pending" if analysis.status == "reject" else "reject"
                save_session()
                st.rerun()


# ─── Main App ─────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Frames",
        page_icon="🎞",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    if not st.session_state.ollama_checked:
        st.session_state.ollama_available = check_ollama_alive()
        st.session_state.ollama_checked = True

    analyses = st.session_state.analyses
    folder = st.session_state.get("folder", "")
    folder_set = bool(folder) and Path(folder).is_dir()
    analyzed = st.session_state.get("analyzed", False)
    has_keeps = any(a.status == "keep" for a in analyses)
    current_step = get_current_step()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # Header
        st.markdown(
            '<h1 style="font-family:\'DM Serif Display\',serif;font-size:2rem;margin-bottom:0;">'
            "Frames</h1>"
            '<p style="color:var(--text-muted);font-size:0.7rem;margin-top:2px;'
            'font-family:\'DM Mono\',monospace;text-transform:uppercase;letter-spacing:0.08em;">'
            "Nikon RAW Curator</p>",
            unsafe_allow_html=True,
        )

        # Step indicator
        render_step_indicator(current_step)

        st.divider()

        # ── Load section (always visible) ────────────────────────────────────
        st.markdown("**RAW folder**")
        inp_col, browse_col = st.columns([5, 1])
        with inp_col:
            st.text_input(
                "RAW folder path",
                key="folder",
                placeholder="/Volumes/SD/DCIM/...",
                label_visibility="collapsed",
            )
        with browse_col:
            if st.button("📂", key="browse_raw", help="Open Finder folder picker"):
                picked = pick_folder_dialog()
                if picked:
                    st.session_state._folder_pending = picked
                    st.rerun()

        analyse_label = "↺ Re-analyse" if analyzed else "▶ Analyse Photos"
        if st.button(
            analyse_label,
            type="primary",
            use_container_width=True,
            disabled=not folder_set,
        ):
            run_analysis(st.session_state.folder, use_ai=st.session_state.get("use_ai", False))

        # ── AI Scoring expander (after folder set) ───────────────────────────
        if folder_set:
            with st.expander("AI Scoring", expanded=st.session_state.ollama_available):
                dot_color = "var(--green-keep)" if st.session_state.ollama_available else "var(--red-reject)"
                dot_label = "Ollama running" if st.session_state.ollama_available else "Ollama not detected"
                status_col, recheck_col = st.columns([3, 1])
                with status_col:
                    st.markdown(
                        f'<span style="color:{dot_color};font-size:0.7rem;">● {dot_label}</span>',
                        unsafe_allow_html=True,
                    )
                with recheck_col:
                    if st.button("↺", help="Recheck Ollama connection"):
                        st.session_state.ollama_available = check_ollama_alive()
                        st.rerun()
                st.checkbox(
                    "Enable AI scoring (Qwen2.5-VL 3B)",
                    key="use_ai",
                    disabled=not st.session_state.ollama_available,
                    help="Requires: ollama pull qwen2.5vl:3b. Adds ~5–15s per photo.",
                )

        # ── Batch actions + Color Settings (after analysis) ──────────────────
        if analyzed:
            st.divider()

            st.markdown("**Auto-select**")
            threshold = st.slider("Keep photos scoring ≥", 0, 100, 65, step=5)
            if st.button("Apply threshold", use_container_width=True):
                for a in analyses:
                    if a.status == "pending":
                        a.status = "keep" if a.overall_score >= threshold else "reject"
                save_session()
                st.rerun()

            col_keep, col_reset = st.columns(2)
            with col_keep:
                if st.button("✓ Keep all", use_container_width=True):
                    for a in analyses:
                        a.status = "keep"
                    save_session()
                    st.rerun()
            with col_reset:
                if st.button("✗ Reset", use_container_width=True):
                    for a in analyses:
                        a.status = "pending"
                    save_session()
                    st.rerun()

            with st.expander("Color Settings", expanded=False):
                if st.button("Reset to defaults", use_container_width=True):
                    st.session_state.color_settings = DEFAULT_COLOR_SETTINGS.copy()
                    st.rerun()
                cs = st.session_state.color_settings
                cs["brightness"] = st.slider("Brightness", -0.3, 0.3, float(cs["brightness"]), step=0.01)
                cs["contrast"] = st.slider("Contrast", 0.7, 1.5, float(cs["contrast"]), step=0.01)
                cs["saturation_boost"] = st.slider("Saturation", 0.5, 1.8, float(cs["saturation_boost"]), step=0.05)
                cs["highlight_recovery"] = st.slider("Highlight recovery", 0.0, 0.3, float(cs["highlight_recovery"]), step=0.01)
                cs["shadow_lift"] = st.slider("Shadow lift", 0.0, 0.15, float(cs["shadow_lift"]), step=0.005)
                cs["sharpening"] = st.slider("Sharpening", 0.0, 1.0, float(cs["sharpening"]), step=0.05)

        # ── Export section (after analysis + has keeps) ──────────────────────
        if analyzed and has_keeps:
            st.divider()

            st.markdown("**Export folder**")
            exp_col, exp_browse_col = st.columns([5, 1])
            with exp_col:
                st.text_input(
                    "Export folder path",
                    placeholder="/path/to/export",
                    key="export_folder",
                    label_visibility="collapsed",
                )
            with exp_browse_col:
                if st.button("📂", key="browse_export", help="Open Finder folder picker"):
                    picked = pick_folder_dialog()
                    if picked:
                        st.session_state._export_pending = picked
                        st.rerun()

            if st.button("⬇ Export kept photos", type="primary", use_container_width=True):
                export_photos(
                    st.session_state.get("export_folder", ""),
                    st.session_state.color_settings,
                )

    # ── Welcome screen ────────────────────────────────────────────────────────
    if not analyzed:
        st.markdown(
            '<div style="padding:40px 0 28px 0;">'
            '<h1 style="font-family:\'DM Serif Display\',serif;font-size:3.5rem;'
            'color:var(--text-primary);margin-bottom:6px;line-height:1;">Frames</h1>'
            '<p style="font-family:\'DM Mono\',monospace;font-size:0.7rem;'
            'text-transform:uppercase;letter-spacing:0.12em;color:var(--text-muted);'
            'margin-bottom:0;">RAW photo culling for Nikon shooters</p>'
            "</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        for col, icon, title, desc in [
            (c1, "◈", "Sharpness Analysis",
             "Laplacian variance on the central crop.<br>Detects blur and focus issues automatically."),
            (c2, "◑", "Exposure Scoring",
             "Measures brightness distribution and<br>penalises blown highlights or crushed shadows."),
            (c3, "◫", "Duplicate Detection",
             "Perceptual hashing (16×16) with Hamming<br>distance — auto-rejects near-identical shots."),
        ]:
            with col:
                st.markdown(
                    f'<div class="feature-card">'
                    f'<div class="feature-icon">{icon}</div>'
                    f'<div class="feature-title">{title}</div>'
                    f'<div class="feature-desc">{desc}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            '<div style="border:1px dashed var(--border-card);border-radius:8px;'
            'padding:20px;text-align:center;color:var(--text-muted);'
            'font-family:\'DM Mono\',monospace;font-size:0.7rem;margin-top:24px;">'
            "Select a RAW folder in the sidebar to begin."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # ── Stats strip + sort control ────────────────────────────────────────────
    strip_col, sort_col = st.columns([7, 1])
    with strip_col:
        st.markdown(render_stats_strip(analyses), unsafe_allow_html=True)
    with sort_col:
        st.selectbox(
            "Sort",
            options=list(SORT_OPTIONS.keys()),
            format_func=lambda k: SORT_OPTIONS[k],
            key="sort_by",
            label_visibility="collapsed",
        )

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── Filter tabs ───────────────────────────────────────────────────────────
    n_all     = len(analyses)
    n_pending = sum(1 for a in analyses if a.status == "pending")
    n_keep    = sum(1 for a in analyses if a.status == "keep")
    n_reject  = sum(1 for a in analyses if a.status == "reject")
    n_dupes   = sum(1 for a in analyses if a.is_duplicate)

    tabs = st.tabs([
        f"All ({n_all})",
        f"Pending ({n_pending})",
        f"Keep ({n_keep})",
        f"Reject ({n_reject})",
        f"Dupes ({n_dupes})",
    ])

    filter_keys = ["all", "pending", "keep", "reject", "duplicate"]
    current_sort = st.session_state.get("sort_by", "score_desc")

    for tab, fk in zip(tabs, filter_keys):
        with tab:
            if fk == "all":
                visible = list(analyses)
            elif fk == "duplicate":
                visible = [a for a in analyses if a.is_duplicate]
            else:
                visible = [a for a in analyses if a.status == fk]

            visible = sort_analyses(visible, current_sort)

            if not visible:
                st.markdown(render_empty_state(fk), unsafe_allow_html=True)
            else:
                cols = st.columns(4)
                for i, analysis in enumerate(visible):
                    render_photo_card(analysis, cols[i % 4])


if __name__ == "__main__":
    main()
