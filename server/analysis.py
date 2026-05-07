from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import rawpy
from PIL import Image

from .models import (
    OLLAMA_BASE,
    OLLAMA_MODEL,
    RAW_EXTENSIONS,
    THUMB_DIR,
    PhotoAnalysis,
    _state,
    save_session,
)


def load_rgb(path: str) -> tuple[np.ndarray, bool]:
    """Load a RAW file as 8-bit RGB. Falls back to embedded JPEG if LibRaw can't decode."""
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


def perceptual_hash(gray_uint8: np.ndarray) -> str:
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
    try:
        with urllib.request.urlopen(OLLAMA_BASE, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def score_photo_with_ai(
    thumbnail_path: str,
) -> tuple[Optional[float], Optional[str], Optional[float], Optional[float], Optional[float], Optional[bool], Optional[str]]:
    """Returns (ai_score, reason, composition, lighting, subject_clarity, usable, error_or_None)."""

    def _call() -> dict:
        with open(thumbnail_path, "rb") as f:
            img = Image.open(f).copy()
        img.thumbnail((384, 256), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        prompt = (
            "You are a photography expert scoring RAW photos for technical quality.\n"
            "Score honestly — a technically strong photo DESERVES a high score.\n"
            "\n"
            "WHAT TO SCORE:\n"
            "  subject_clarity — Is the intended subject sharp and in focus?\n"
            "  lighting        — Is exposure balanced? Any blown or crushed areas?\n"
            "  composition     — Clear subject, sensible framing, no major obstructions?\n"
            "  overall         — Holistic technical quality.\n"
            "\n"
            "SCORE GUIDE:\n"
            "  80–100  Excellent. Sharp subject, good exposure, clear composition.\n"
            "  60–79   Good. One minor flaw but generally strong.\n"
            "  40–59   Average. Noticeable issues but usable.\n"
            "  20–39   Poor. Significant flaw limits usefulness.\n"
            "   0–19   Unusable. Severe blur, wrong focus, or unrecoverable exposure.\n"
            "\n"
            "RULES:\n"
            "  - A sharp, well-exposed photo with a clear subject MUST score at least 65.\n"
            "  - Do not penalise bokeh backgrounds, artistic style, or tight crops.\n"
            "  - Set usable=false ONLY when the subject itself is blurry, the exposure\n"
            "    is completely unrecoverable, or there is no identifiable subject.\n"
            "\n"
            "Return ONLY this JSON, no other text:\n"
            "{\n"
            '  "composition": <integer 0-100>,\n'
            '  "lighting": <integer 0-100>,\n'
            '  "subject_clarity": <integer 0-100>,\n'
            '  "overall": <integer 0-100>,\n'
            '  "usable": <true or false>,\n'
            '  "reason": "<one specific observation, 10-20 words>"\n'
            "}"
        )

        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 200, "num_ctx": 2048},
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = json.loads(resp.read())

        response_text = raw.get("response", "").strip()
        match = re.search(r"\{[^{}]+\}", response_text, re.DOTALL)
        if match:
            response_text = match.group(0)
        return json.loads(response_text)

    last_error: str = "unknown error"
    for _ in range(2):
        try:
            data = _call()
            overall = float(data["overall"])
            composition = float(data["composition"])
            lighting = float(data["lighting"])
            subject_clarity = float(data["subject_clarity"])
            reason = str(data.get("reason", ""))
            usable = bool(data.get("usable", True))
            ai_score = overall * 0.50 + composition * 0.20 + lighting * 0.20 + subject_clarity * 0.10
            return float(ai_score), reason, composition, lighting, subject_clarity, usable, None
        except Exception as e:
            last_error = str(e)

    return None, None, None, None, None, None, last_error


def analyse_photo(path: str, existing_hashes: dict[str, str]) -> PhotoAnalysis:
    filename = Path(path).name
    stem = Path(path).stem

    rgb, _from_jpeg = load_rgb(path)

    pil_img = Image.fromarray(rgb)
    thumb = pil_img.copy()
    thumb.thumbnail((768, 512), Image.LANCZOS)
    thumb_path = str(THUMB_DIR / f"{stem}.jpg")
    thumb.save(thumb_path, "JPEG", quality=85)

    h, w = rgb.shape[:2]

    # Sharpness
    center = rgb[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    gray_center = cv2.cvtColor(center, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray_center, cv2.CV_64F)
    sharpness = float(min(100.0, laplacian.var() / 5.0))

    # Exposure
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    overexposed = np.mean(gray > 240)
    underexposed = np.mean(gray < 15)
    brightness_score = 100 - abs(mean_brightness - 128) / 1.28
    clipping_penalty = (overexposed + underexposed) * 200
    exposure = float(max(0.0, min(100.0, brightness_score - clipping_penalty)))

    # Saturation
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = float(hsv[:, :, 1].mean() / 2.55)

    # Technical-only score; AI pass runs separately via run_ai_scoring()
    ai_score = ai_reason = ai_composition = ai_lighting = ai_subject_clarity = ai_usable = ai_error = None
    sat_fitness = float(max(0, min(100, 100 - abs(saturation - 42) * 2)))
    overall_score = float(sharpness * 0.55 + exposure * 0.30 + sat_fitness * 0.15)

    # Duplicate detection
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
        ai_composition=ai_composition,
        ai_lighting=ai_lighting,
        ai_subject_clarity=ai_subject_clarity,
        ai_usable=ai_usable,
        ai_error=ai_error,
    )


def run_analysis(
    folder: str,
    progress_callback: Optional[Callable[[int, int, PhotoAnalysis | None, str | None], None]] = None,
) -> list[PhotoAnalysis]:
    """
    Analyse all RAW files in folder.
    progress_callback(done, total, photo_or_None, error_or_None)
    """
    folder_path = Path(folder)
    raw_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in RAW_EXTENSIONS])

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    results: list[PhotoAnalysis] = []
    hashes: dict[str, str] = {}
    total = len(raw_files)

    for i, raw_file in enumerate(raw_files):
        try:
            analysis = analyse_photo(str(raw_file), hashes)
            results.append(analysis)
            if progress_callback:
                progress_callback(i + 1, total, analysis, None)
        except Exception as e:
            if progress_callback:
                progress_callback(i + 1, total, None, f"Could not read {raw_file.name}: {e}")

    results.sort(key=lambda x: x.overall_score, reverse=True)
    for r in results:
        if r.is_duplicate:
            r.status = "reject"

    _state["analyses"] = results
    _state["analyzed"] = True
    save_session()
    return results


def run_ai_scoring(
    analyses: list[PhotoAnalysis],
    progress_callback: Optional[Callable[[int, int, PhotoAnalysis | None, str | None], None]] = None,
) -> None:
    """Second-pass AI scoring over already-analysed photos. Updates scores in-place."""
    total = len(analyses)
    for i, photo in enumerate(analyses):
        try:
            ai_score, ai_reason, ai_composition, ai_lighting, ai_subject_clarity, ai_usable, ai_error = score_photo_with_ai(photo.thumbnail_path)
            photo.ai_score = ai_score
            photo.ai_reason = ai_reason
            photo.ai_composition = ai_composition
            photo.ai_lighting = ai_lighting
            photo.ai_subject_clarity = ai_subject_clarity
            photo.ai_usable = ai_usable
            photo.ai_error = ai_error

            if ai_score is not None:
                sat_fitness = float(max(0, min(100, 100 - abs(photo.saturation - 42) * 2)))
                overall = float(photo.sharpness * 0.35 + photo.exposure * 0.20 + sat_fitness * 0.10 + ai_score * 0.35)
                if ai_usable is False:
                    overall = min(overall, 35.0)
                photo.overall_score = overall
        except Exception as e:
            if progress_callback:
                progress_callback(i + 1, total, None, f"AI scoring failed for {photo.filename}: {e}")
            continue

        if progress_callback:
            progress_callback(i + 1, total, photo, None)

    _state["analyses"].sort(key=lambda x: x.overall_score, reverse=True)
    save_session()
