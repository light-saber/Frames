"""
Generate frames_icon.icns — film-frame icon for the Frames macOS app.

Draw at 3× scale (3072px), resize down to 1024px with LANCZOS for smooth
anti-aliased edges. No aggdraw required; PIL.ImageDraw is sufficient.

Usage: .venv/bin/python create_icon.py
"""

import os
import shutil
import subprocess
from PIL import Image, ImageDraw

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#111111"
STRIP_BG    = "#0d0d0d"
GOLD        = "#c9a84c"
HOLE_FILL   = "#080808"

# ── Scale ────────────────────────────────────────────────────────────────────
SUPER = 3          # supersampling multiplier
S     = 1024 * SUPER  # canvas size (3072 × 3072)

# ── Geometry helpers (all values in 1024-px space, multiplied by SUPER) ──────
def px(n): return round(n * SUPER)


def rounded_rect(draw, x0, y0, x1, y1, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle using pieslice + rectangle primitives."""
    r = radius
    draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
    draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
    # Four corners
    draw.pieslice([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=fill)
    draw.pieslice([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=fill)
    draw.pieslice([x0, y1 - 2*r, x0 + 2*r, y1], 90,  180, fill=fill)
    draw.pieslice([x1 - 2*r, y1 - 2*r, x1, y1], 0,   90,  fill=fill)

    if outline and width:
        # Draw outline as a slightly inset arc + lines
        lw = width
        draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=lw)
        draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=lw)
        draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90,  180, fill=outline, width=lw)
        draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0,    90, fill=outline, width=lw)
        draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=lw)
        draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=lw)
        draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=lw)
        draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=lw)


def generate_1024() -> Image.Image:
    # ── Canvas ────────────────────────────────────────────────────────────────
    img  = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # ── 1. Rounded-square background ──────────────────────────────────────────
    PAD  = px(40)        # gap between canvas edge and rounded square
    R_BG = px(190)       # corner radius at 1024px → 570 at 3×
    rounded_rect(draw, PAD, PAD, S - PAD, S - PAD, R_BG, fill=BG)

    # ── 2. Film-frame geometry ─────────────────────────────────────────────────
    # The "frame" rectangle sits centred, with strips top and bottom
    FRAME_X0  = px(100)
    FRAME_X1  = px(924)
    FRAME_Y0  = px(130)
    FRAME_Y1  = px(894)

    STRIP_H   = px(110)          # height of sprocket strip
    STRIP_Y0T = FRAME_Y0         # top strip top
    STRIP_Y1T = FRAME_Y0 + STRIP_H  # top strip bottom
    STRIP_Y0B = FRAME_Y1 - STRIP_H  # bottom strip top
    STRIP_Y1B = FRAME_Y1         # bottom strip bottom

    # Top + bottom darkened strips
    draw.rectangle([FRAME_X0, STRIP_Y0T, FRAME_X1, STRIP_Y1T], fill=STRIP_BG)
    draw.rectangle([FRAME_X0, STRIP_Y0B, FRAME_X1, STRIP_Y1B], fill=STRIP_BG)

    # ── 3. Sprocket holes ─────────────────────────────────────────────────────
    HOLE_W    = px(64)
    HOLE_H    = px(46)
    HOLE_R    = px(10)
    BORDER_W  = px(5)
    N_HOLES   = 4

    frame_w   = FRAME_X1 - FRAME_X0
    # Distribute holes evenly across the strip width with equal margins
    spacing   = frame_w / N_HOLES
    first_cx  = FRAME_X0 + spacing / 2

    for strip_cy in (STRIP_Y0T + STRIP_H / 2, STRIP_Y0B + STRIP_H / 2):
        for i in range(N_HOLES):
            cx = first_cx + i * spacing
            hx0 = round(cx - HOLE_W / 2)
            hx1 = round(cx + HOLE_W / 2)
            hy0 = round(strip_cy - HOLE_H / 2)
            hy1 = round(strip_cy + HOLE_H / 2)
            # Black fill
            rounded_rect(draw, hx0, hy0, hx1, hy1, HOLE_R, fill=HOLE_FILL)
            # Gold border
            rounded_rect(draw, hx0, hy0, hx1, hy1, HOLE_R,
                         outline=GOLD, width=BORDER_W)

    # ── 4. Gold film-frame rectangle outline ──────────────────────────────────
    OUTLINE_W = px(28)
    FRAME_R   = px(6)        # slight rounding on frame corners

    # Draw as a thick rounded-rect outline by layering two filled rects
    ow = OUTLINE_W
    # outer = frame boundary
    rounded_rect(draw,
                 FRAME_X0, FRAME_Y0, FRAME_X1, FRAME_Y1,
                 FRAME_R, fill=GOLD)
    # inner = cut out the middle (background colour)
    rounded_rect(draw,
                 FRAME_X0 + ow, FRAME_Y0 + ow,
                 FRAME_X1 - ow, FRAME_Y1 - ow,
                 max(FRAME_R - ow // 2, 0), fill=BG)
    # Re-draw strips on top (so the frame outline sits over the strip background)
    draw.rectangle([FRAME_X0 + ow, STRIP_Y0T, FRAME_X1 - ow, STRIP_Y1T], fill=STRIP_BG)
    draw.rectangle([FRAME_X0 + ow, STRIP_Y0B, FRAME_X1 - ow, STRIP_Y1B], fill=STRIP_BG)

    # Re-draw sprocket holes on top of the re-drawn strips
    for strip_cy in (STRIP_Y0T + STRIP_H / 2, STRIP_Y0B + STRIP_H / 2):
        for i in range(N_HOLES):
            cx = first_cx + i * spacing
            hx0 = round(cx - HOLE_W / 2)
            hx1 = round(cx + HOLE_W / 2)
            hy0 = round(strip_cy - HOLE_H / 2)
            hy1 = round(strip_cy + HOLE_H / 2)
            rounded_rect(draw, hx0, hy0, hx1, hy1, HOLE_R, fill=HOLE_FILL)
            rounded_rect(draw, hx0, hy0, hx1, hy1, HOLE_R,
                         outline=GOLD, width=BORDER_W)

    # ── 5. Downsample to 1024×1024 ────────────────────────────────────────────
    return img.resize((1024, 1024), Image.LANCZOS)


# ── Iconset sizes (macOS naming convention) ───────────────────────────────────
ICONSET_SIZES = [
    # (logical_size, scale, filename)
    (16,   1, "icon_16x16.png"),
    (16,   2, "icon_16x16@2x.png"),
    (32,   1, "icon_32x32.png"),
    (32,   2, "icon_32x32@2x.png"),
    (64,   1, "icon_64x64.png"),
    (64,   2, "icon_64x64@2x.png"),
    (128,  1, "icon_128x128.png"),
    (128,  2, "icon_128x128@2x.png"),
    (256,  1, "icon_256x256.png"),
    (256,  2, "icon_256x256@2x.png"),
    (512,  1, "icon_512x512.png"),
    (512,  2, "icon_512x512@2x.png"),
    (1024, 1, "icon_512x512@2x.png"),   # same file, 1024px
]


def build_iconset(base: Image.Image, iconset_dir: str):
    os.makedirs(iconset_dir, exist_ok=True)
    seen = set()
    for logical, scale, fname in ICONSET_SIZES:
        pixel_size = logical * scale
        if fname in seen:
            continue
        seen.add(fname)
        resized = base.resize((pixel_size, pixel_size), Image.LANCZOS)
        resized.save(os.path.join(iconset_dir, fname), "PNG")
        print(f"  {pixel_size:4d}px → {fname}")


def main():
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    iconset   = os.path.join(base_dir, "frames_icon.iconset")
    icns_path = os.path.join(base_dir, "frames_icon.icns")

    print("Generating 3072×3072 master …")
    master = generate_1024()   # already downsampled to 1024

    # We need the 1024px master; regenerate at 1024 directly (already done above)
    print("Writing iconset …")
    build_iconset(master, iconset)

    print("Running iconutil …")
    result = subprocess.run(
        ["iconutil", "-c", "icns", iconset, "-o", icns_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("iconutil stderr:", result.stderr)
        raise RuntimeError("iconutil failed")

    shutil.rmtree(iconset)
    size_kb = os.path.getsize(icns_path) / 1024
    print(f"\n✓ frames_icon.icns ({size_kb:.0f} KB) written to {icns_path}")


if __name__ == "__main__":
    main()
