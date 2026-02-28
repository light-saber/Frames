# Frames

A native macOS photo culling and colour correction tool for Nikon RAW files. Point it at a folder of NEFs, score every shot for sharpness, exposure, and duplicates, then export your selects as colour-corrected TIFFs — fully offline, no browser required.

---

## Features

- **Sharpness scoring** — Laplacian variance on the central crop; detects blur and missed focus
- **Exposure scoring** — Measures brightness distribution and penalises blown highlights and crushed shadows
- **Saturation fitness** — Flags over-saturated or grey shots
- **Duplicate detection** — Perceptual hashing (16×16) with Hamming distance; auto-rejects near-identical frames
- **AI scoring (optional)** — Qwen2.5-VL 3B via Ollama scores composition, lighting, and subject clarity; blends into the overall score when enabled
- **Colour correction pipeline** — Highlight recovery, shadow lift, brightness/contrast, saturation boost, unsharp mask sharpening — applied on export, not preview
- **Finder folder picker** — Native macOS folder dialog via the 📂 button
- **TIFF export** — Colour-corrected TIFFs named `{original}_edited.tiff`
- **Session persistence** — Analysis results and keep/reject decisions survive app restarts
- **Fully local** — No internet connection, no cloud, no account

---

## Requirements

- macOS 12 or later
- Python 3.9 or later
- Supported RAW formats: `.nef` `.nrw` `.raw` `.cr2` `.cr3` `.arw` `.dng`

> **Nikon Zf note:** The Nikon Zf (released September 2023) is not yet in LibRaw 0.22.0's camera database. Frames automatically falls back to the full-resolution embedded JPEG (6048×4032) inside the NEF — same pixel dimensions as the sensor, fully usable for culling and export.

---

## Setup

Run the one-time setup script from the project folder:

```bash
bash setup.sh
```

This will:
1. Verify Python 3.9+ is present
2. Create a `.venv` virtual environment
3. Install all Python dependencies (FastAPI, uvicorn, pywebview, rawpy, OpenCV, Pillow, NumPy)

---

## Launch (dev)

```bash
source .venv/bin/activate && python main.py
```

The app opens in a native macOS window via WebKit. No browser needed.

---

## Build a standalone .app

```bash
bash build.sh
```

Produces `dist/Frames.app`. To install system-wide:

```bash
cp -r dist/Frames.app /Applications/
```

Then launch from Finder or Spotlight like any other Mac app.

---

## Workflow

1. **Select folder** — Paste the path to your RAW files, or click 📂 to open a native Finder folder picker
2. **Analyse** — Click **▶ Analyse Photos**. Progress streams live as each file is processed. Expect roughly 10–15 min for 200 photos on Intel Mac; significantly faster on Apple Silicon
3. **Review scores** — Photos are sorted best-first in a 4-column grid. Border colour indicates status: green = keep, red = reject, grey = pending
4. **Auto-cull** — Use **Apply threshold** to bulk-mark photos above/below a score (default 65)
5. **Fine-tune** — Click **✓ Keep** or **✗ Reject** on any card. Use the filter tabs to focus on a subset
6. **Adjust colour** — Tweak the Color Settings sliders in the sidebar (applied at export time, not to the preview)
7. **Set export folder** — Enter the destination path, or click 📂 to pick with Finder
8. **Export** — Click **⬇ Export kept photos**. Each kept photo is colour-corrected and saved as `{name}_edited.tiff`

---

## Score Breakdown

| Score | Badge | Meaning |
|------:|-------|---------|
| 70–100 | Green | Sharp, well-exposed, good colour — likely a keeper |
| 45–69 | Amber | Acceptable — review manually |
| 0–44 | Red | Blurry, badly exposed, or flat colour |

Weights without AI: **Sharpness 55%** · **Exposure 30%** · **Saturation fitness 15%**

Weights with AI: **Sharpness 35%** · **AI score 35%** · **Exposure 20%** · **Saturation fitness 10%**

---

## AI Scoring (Optional)

Frames can use **Qwen2.5-VL 3B** running locally via [Ollama](https://ollama.com) to score each photo on composition, lighting, and subject clarity.

**Setup:**

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen2.5vl:3b
```

When Frames launches it automatically starts `ollama serve` if it isn't already running, and pulls `qwen2.5vl:3b` if the model is absent. The sidebar shows a green **● Ollama running** indicator once it's ready. Enable **AI scoring** before clicking **▶ Analyse Photos**. Each photo adds roughly 5–15 seconds of inference time.

---

## Colour Correction Defaults

| Setting | Default | Range |
|---------|---------|-------|
| Brightness | 0.0 | −0.3 → 0.3 |
| Contrast | 1.05 | 0.7 → 1.5 |
| Saturation | 1.1× | 0.5 → 1.8 |
| Highlight recovery | 0.12 | 0.0 → 0.3 |
| Shadow lift | 0.03 | 0.0 → 0.15 |
| Sharpening | 0.3 | 0.0 → 1.0 |

All adjustments are applied in sequence on a `float32` [0, 1] image and exported as TIFF.

---

## Known Limitations

- **Colour science** — Nikon Picture Control profiles (Vivid, Portrait, etc.) are not replicated. Export colours will differ from NX Studio's rendering
- **High-ISO noise** — Heavy noise can reduce sharpness scores due to the Laplacian variance method picking up texture as signal
- **Performance** — First run on 200 photos takes ~10–15 min on Intel Mac; much faster on M-series
- **Output format** — Exports are standard TIFFs, not NX Studio native files. Compatible with Lightroom, Affinity Photo, and any TIFF editor

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Window | [pywebview](https://pywebview.flowrl.com) (WKWebView) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org) |
| UI | Vanilla JS + HTML/CSS (dark/gold theme) |
| RAW decoding | [rawpy](https://pypi.org/project/rawpy/) + LibRaw |
| Image processing | [OpenCV](https://pypi.org/project/opencv-python-headless/), [Pillow](https://python-pillow.org) |
| Numerics | [NumPy](https://numpy.org) |
| Folder picker | macOS `osascript` (AppleScript) |
| AI scoring | [Ollama](https://ollama.com) + Qwen2.5-VL 3B (optional) |
| Distribution | [PyInstaller](https://pyinstaller.org) |

---

## Acknowledgements

- **[rawpy](https://github.com/letmaik/rawpy)** — MIT — Python wrapper for LibRaw
- **[LibRaw](https://github.com/LibRaw/LibRaw)** — LGPL 2.1 / CDDL 1.0 — RAW image decoding library
- **[OpenCV](https://github.com/opencv/opencv)** — Apache 2.0 — image processing (sharpness, exposure, colour transforms)
- **[Pillow](https://github.com/python-pillow/Pillow)** — HPND — image I/O and resizing
- **[NumPy](https://github.com/numpy/numpy)** — BSD 3-Clause — numerical array operations
- **[FastAPI](https://github.com/tiangolo/fastapi)** — MIT — modern Python web framework
- **[pywebview](https://github.com/r0x0r/pywebview)** — BSD 3-Clause — native window wrapper for web content
- **[DM Serif Display](https://fonts.google.com/specimen/DM+Serif+Display)** & **[DM Mono](https://fonts.google.com/specimen/DM+Mono)** — SIL Open Font License — typefaces via Google Fonts
