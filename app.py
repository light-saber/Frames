from __future__ import annotations

import base64
import io
import os
import subprocess
from dataclasses import dataclass
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
DEFAULT_COLOR_SETTINGS = {
    "brightness": 0.0,
    "contrast": 1.05,
    "saturation_boost": 1.1,
    "highlight_recovery": 0.12,
    "shadow_lift": 0.03,
    "sharpening": 0.3,
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


# ─── Session State ────────────────────────────────────────────────────────────


def init_state():
    # Apply browse-button picks BEFORE any widgets are instantiated
    if "_folder_pending" in st.session_state:
        st.session_state.folder = st.session_state.pop("_folder_pending")
    if "_export_pending" in st.session_state:
        st.session_state.export_folder = st.session_state.pop("_export_pending")

    if "analyses" not in st.session_state:
        st.session_state.analyses = []
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    if "filter" not in st.session_state:
        st.session_state.filter = "all"
    if "color_settings" not in st.session_state:
        st.session_state.color_settings = DEFAULT_COLOR_SETTINGS.copy()
    if "export_done" not in st.session_state:
        st.session_state.export_done = False


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


def analyse_photo(path: str, existing_hashes: dict[str, str]) -> PhotoAnalysis:
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

    # Step 6: Overall score
    sat_fitness = float(max(0, min(100, 100 - abs(saturation - 42) * 2)))
    overall_score = float(sharpness * 0.55 + exposure * 0.30 + sat_fitness * 0.15)

    # Step 7: Duplicate detection
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
    )


def run_analysis(folder: str):
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
        status_text.text(f"Analysing {raw_file.name} ({i + 1}/{len(raw_files)})…")
        try:
            analysis = analyse_photo(str(raw_file), hashes)
            results.append(analysis)
        except Exception as e:
            st.warning(f"Could not read {raw_file.name}: {e}")
        progress_bar.progress((i + 1) / len(raw_files))

    status_text.text("Analysis complete!")

    # Step 8: Sort by overall score descending
    results.sort(key=lambda x: x.overall_score, reverse=True)

    # Step 9: Auto-flag duplicates
    for r in results:
        if r.is_duplicate:
            r.status = "reject"

    st.session_state.analyses = results
    st.session_state.analyzed = True


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

/* DM Mono everywhere — config.toml handles base colors */
html, body, button, input, select, textarea,
[data-testid="stApp"], [data-testid="stSidebar"],
.stMarkdown, label, p, span, li {
    font-family: 'DM Mono', monospace !important;
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
    border-color: #c9a84c !important;
    color: #c9a84c !important;
}

/* ── Custom HTML components ─────────────────────────────────────────────── */

.photo-card {
    background: #1c2028;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 4px;
    border: 2px solid #2d3240;
}

.filename {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #a09b96;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin: 4px 0;
}

.score-high {
    background: #1a3a2a; color: #4ade80; border: 1px solid #166534;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.7rem; font-family: 'DM Mono', monospace;
}
.score-mid {
    background: #3a2e1a; color: #fbbf24; border: 1px solid #92400e;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.7rem; font-family: 'DM Mono', monospace;
}
.score-low {
    background: #3a1a1a; color: #f87171; border: 1px solid #991b1b;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.7rem; font-family: 'DM Mono', monospace;
}

.dup-tag {
    background: #2a1a3a; color: #a78bfa; border: 1px solid #5b21b6;
    border-radius: 20px; padding: 2px 8px;
    font-size: 0.65rem; font-family: 'DM Mono', monospace; margin-left: 4px;
}

.stat-card {
    background: #1c2028;
    border: 1px solid #2d3240;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.stat-label {
    font-size: 0.65rem; color: #a09b96;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.08em;
}
.stat-value {
    font-size: 1.6rem;
    font-family: 'DM Serif Display', serif;
    line-height: 1.2;
}

.progress-bar-wrap {
    margin: 3px 0; display: flex; align-items: center;
    gap: 6px; font-size: 0.6rem; color: #a09b96;
    font-family: 'DM Mono', monospace;
}
.progress-bar-bg {
    flex: 1; height: 4px; background: #2d3240;
    border-radius: 2px; overflow: hidden;
}
.progress-bar-fill-gold { height: 100%; background: #c9a84c; border-radius: 2px; }
.progress-bar-fill-blue { height: 100%; background: #60a5fa; border-radius: 2px; }

</style>
"""

# ─── Helper rendering ─────────────────────────────────────────────────────────


def score_badge(score: float) -> str:
    if score >= 70:
        cls = "score-high"
    elif score >= 45:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="{cls}">{score:.0f}</span>'


def progress_bar_html(value: float, color_class: str, label: str) -> str:
    pct = min(100, max(0, value))
    return (
        f'<div class="progress-bar-wrap">'
        f'<span>{label}</span>'
        f'<div class="progress-bar-bg">'
        f'<div class="{color_class}" style="width:{pct:.0f}%"></div>'
        f'</div>'
        f'<span>{pct:.0f}</span>'
        f'</div>'
    )


def render_photo_card(analysis: PhotoAnalysis, col):
    with col:
        border_colors = {"keep": "#4ade80", "reject": "#f87171", "pending": "#444"}
        border_color = border_colors.get(analysis.status, "#444")

        badge = score_badge(analysis.overall_score)
        dup_html = '<span class="dup-tag">dup</span>' if analysis.is_duplicate else ""
        fname_short = (
            analysis.filename[:28] + "…"
            if len(analysis.filename) > 28
            else analysis.filename
        )
        sharp_bar = progress_bar_html(analysis.sharpness, "progress-bar-fill-gold", "S")
        exp_bar = progress_bar_html(analysis.exposure, "progress-bar-fill-blue", "E")

        # Embed thumbnail as base64 for clean card border
        thumb_html = ""
        if os.path.exists(analysis.thumbnail_path):
            with open(analysis.thumbnail_path, "rb") as f:
                thumb_b64 = base64.b64encode(f.read()).decode()
            thumb_html = (
                f'<img src="data:image/jpeg;base64,{thumb_b64}" '
                f'style="width:100%;border-radius:4px;display:block;margin-bottom:6px;">'
            )

        st.markdown(
            f'<div class="photo-card" style="border-color:{border_color};">'
            f"{thumb_html}"
            f'<div class="filename">{fname_short}</div>'
            f'<div style="margin:4px 0;">{badge}{dup_html}</div>'
            f"{sharp_bar}{exp_bar}"
            f"</div>",
            unsafe_allow_html=True,
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            keep_label = "↩" if analysis.status == "keep" else "✓"
            if st.button(keep_label, key=f"keep_{analysis.filename}", use_container_width=True):
                analysis.status = "pending" if analysis.status == "keep" else "keep"
                st.rerun()
        with btn_col2:
            rej_label = "↩" if analysis.status == "reject" else "✗"
            if st.button(rej_label, key=f"reject_{analysis.filename}", use_container_width=True):
                analysis.status = "pending" if analysis.status == "reject" else "reject"
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

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            '<h1 style="font-family:\'DM Serif Display\',serif;font-size:2rem;margin-bottom:0;">'
            "Frames</h1>"
            '<p style="color:#6b6560;font-size:0.75rem;margin-top:2px;">Nikon RAW Curator</p>',
            unsafe_allow_html=True,
        )

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

        if st.button("▶ Analyse Photos", type="primary", use_container_width=True):
            run_analysis(st.session_state.folder)

        st.divider()

        filter_options = ["all", "pending", "keep", "reject", "duplicate"]
        filter_choice = st.radio(
            "Filter",
            options=filter_options,
            index=filter_options.index(st.session_state.filter),
        )
        st.session_state.filter = filter_choice

        st.divider()

        st.markdown("**Auto-select**")
        threshold = st.slider("Keep photos scoring ≥", 0, 100, 65, step=5)
        if st.button("Apply threshold", use_container_width=True):
            for a in st.session_state.analyses:
                if a.status == "pending":
                    a.status = "keep" if a.overall_score >= threshold else "reject"
            st.rerun()

        col_keep, col_reset = st.columns(2)
        with col_keep:
            if st.button("✓ Keep all", use_container_width=True):
                for a in st.session_state.analyses:
                    a.status = "keep"
                st.rerun()
        with col_reset:
            if st.button("✗ Reset", use_container_width=True):
                for a in st.session_state.analyses:
                    a.status = "pending"
                st.rerun()

        st.divider()

        st.markdown("**Color Settings**")
        cs = st.session_state.color_settings
        cs["brightness"] = st.slider("Brightness", -0.3, 0.3, float(cs["brightness"]), step=0.01)
        cs["contrast"] = st.slider("Contrast", 0.7, 1.5, float(cs["contrast"]), step=0.01)
        cs["saturation_boost"] = st.slider("Saturation", 0.5, 1.8, float(cs["saturation_boost"]), step=0.05)
        cs["highlight_recovery"] = st.slider("Highlight recovery", 0.0, 0.3, float(cs["highlight_recovery"]), step=0.01)
        cs["shadow_lift"] = st.slider("Shadow lift", 0.0, 0.15, float(cs["shadow_lift"]), step=0.005)
        cs["sharpening"] = st.slider("Sharpening", 0.0, 1.0, float(cs["sharpening"]), step=0.05)

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
            export_photos(st.session_state.get("export_folder", ""), cs)

    # ── Main area ─────────────────────────────────────────────────────────────
    if not st.session_state.analyzed:
        st.markdown(
            '<h1 style="font-family:\'DM Serif Display\',serif;font-size:3rem;color:#e8e2d9;">Frames</h1>',
            unsafe_allow_html=True,
        )
        st.markdown("A local photo culling and colour correction tool for Nikon RAW files.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(
                "**Sharpness Analysis**  \n"
                "Laplacian variance on the central crop — detects blur and focus issues."
            )
        with c2:
            st.info(
                "**Exposure Scoring**  \n"
                "Measures brightness distribution and penalises blown highlights or crushed shadows."
            )
        with c3:
            st.info(
                "**Duplicate Detection**  \n"
                "Perceptual hashing (16×16) with Hamming distance — automatically rejects near-identical shots."
            )

        st.markdown("---")
        st.info("Enter a RAW folder path in the sidebar and click **▶ Analyse Photos** to begin.")
        return

    # ── Stats bar ─────────────────────────────────────────────────────────────
    analyses = st.session_state.analyses
    total = len(analyses)
    kept = sum(1 for a in analyses if a.status == "keep")
    rejected = sum(1 for a in analyses if a.status == "reject")
    pending = sum(1 for a in analyses if a.status == "pending")
    dupes = sum(1 for a in analyses if a.is_duplicate)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, color in [
        (c1, "Total", total, "#e8e2d9"),
        (c2, "Keep", kept, "#4ade80"),
        (c3, "Reject", rejected, "#f87171"),
        (c4, "Pending", pending, "#fbbf24"),
        (c5, "Dupes", dupes, "#a78bfa"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-label">{label}</div>'
                f'<div class="stat-value" style="color:{color};">{value}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grid view ─────────────────────────────────────────────────────────────
    filt = st.session_state.filter
    if filt == "all":
        visible = analyses
    elif filt == "duplicate":
        visible = [a for a in analyses if a.is_duplicate]
    else:
        visible = [a for a in analyses if a.status == filt]

    if not visible:
        st.markdown(
            f'<p style="color:#6b6560;">No photos match filter: <em>{filt}</em></p>',
            unsafe_allow_html=True,
        )
        return

    visible = sorted(visible, key=lambda a: a.overall_score, reverse=True)

    cols = st.columns(4)
    for i, analysis in enumerate(visible):
        render_photo_card(analysis, cols[i % 4])


if __name__ == "__main__":
    main()
