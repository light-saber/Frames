from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from .analysis import load_rgb
from .models import PhotoAnalysis


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


def export_photos(
    analyses: list[PhotoAnalysis],
    export_folder: str,
    settings: dict,
    progress_callback: Optional[Callable[[int, int, str, str | None], None]] = None,
) -> int:
    """Export kept photos as TIFFs. Returns count exported."""
    export_path = Path(export_folder)
    to_export = [a for a in analyses if a.status == "keep"]
    total = len(to_export)

    for i, analysis in enumerate(to_export):
        try:
            rgb, _ = load_rgb(analysis.path)
            corrected = apply_color_correction(rgb, settings)
            stem = Path(analysis.path).stem
            out_path = export_path / f"{stem}_edited.tiff"
            Image.fromarray(corrected).save(str(out_path), "TIFF")
            if progress_callback:
                progress_callback(i + 1, total, analysis.filename, None)
        except Exception as e:
            if progress_callback:
                progress_callback(i + 1, total, analysis.filename, str(e))

    return total
