from __future__ import annotations

import base64
import io
import json
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


def score_photo_with_ai(thumbnail_path: str) -> tuple[Optional[float], Optional[str]]:
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
            "options": {"temperature": 0.1, "num_predict": 80, "num_ctx": 512},
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
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            response_text = "\n".join(l for l in lines if not l.startswith("```")).strip()

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

    rgb, _from_jpeg = load_rgb(path)

    pil_img = Image.fromarray(rgb)
    thumb = pil_img.copy()
    thumb.thumbnail((480, 320), Image.LANCZOS)
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

    # AI scoring
    ai_score, ai_reason = (None, None)
    if use_ai:
        ai_score, ai_reason = score_photo_with_ai(thumb_path)

    # Overall score
    sat_fitness = float(max(0, min(100, 100 - abs(saturation - 42) * 2)))
    if ai_score is not None:
        overall_score = float(sharpness * 0.35 + exposure * 0.20 + sat_fitness * 0.10 + ai_score * 0.35)
    else:
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
    )


def run_analysis(
    folder: str,
    use_ai: bool = False,
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
            analysis = analyse_photo(str(raw_file), hashes, use_ai=use_ai)
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
