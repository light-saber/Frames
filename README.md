# Frames

A native macOS photo culling and colour correction tool for RAW files. Point it at a folder of RAW photos, score every shot for sharpness, exposure, and duplicates, review them in a clean 3-column grid, then export your selects as colour-corrected TIFFs or JPEGs — fully offline, no browser required.

---

## Features

- **Sharpness scoring** — Laplacian variance on the central crop; detects blur and missed focus
- **Exposure scoring** — Measures brightness distribution and penalises blown highlights and crushed shadows
- **Saturation fitness** — Flags over-saturated or grey shots
- **Duplicate detection** — Perceptual hashing with Hamming distance; auto-rejects near-identical burst frames
- **AI scoring (optional)** — Qwen2.5-VL 3B via Ollama scores composition, lighting, and subject clarity; blends into the overall score when enabled
- **AI detail panel** — Click any photo to open a side panel showing the score breakdown, AI verdict, plain-language descriptions, and individual AI sub-scores
- **Colour correction pipeline** — Highlight recovery, shadow lift, brightness/contrast, saturation boost, unsharp mask sharpening — applied on export, not preview
- **Auto colour enhance** — One-click adaptive enhancement: grey-world white balance, CLAHE contrast, saturation boost (×1.15), and a gentle S-curve tonemap
- **TIFF or JPEG export** — Choose output format; JPEG quality is configurable from 60–100
- **Finder folder picker** — Native macOS folder dialog via the 📂 button
- **Session persistence** — Analysis results and keep/reject decisions survive app restarts
- **Fully local** — No internet connection, no cloud, no account

---

## Requirements

- macOS 12 Monterey or later
- Python 3.11 or later (3.14 recommended — install via Homebrew)
- [Homebrew](https://brew.sh) (recommended for Python and Ollama)
- Supported RAW formats: `.nef` `.nrw` `.raw` `.cr2` `.cr3` `.arw` `.dng`

> **Nikon Zf note:** The Nikon Zf is supported in LibRaw 0.22.0 (bundled with rawpy 0.26+). Standard NEF files decode natively. The exception is **HE/HE\* lossless compressed** NEFs — LibRaw does not yet decode those formats, so Frames falls back to the full-resolution embedded JPEG (6048×4032) inside the file, which has the same pixel dimensions and is fully usable for culling and export.

---

## Installation

### Option A — Build the app (recommended)

The `build.sh` script handles everything: Python version check, virtual environment, dependencies, and PyInstaller bundle.

```bash
git clone https://github.com/light-saber/Frames.git
cd Frames
bash build.sh
```

This produces `dist/Frames.app`. Install it:

```bash
cp -r dist/Frames.app /Applications/
open /Applications/Frames.app
```

### Option B — Run in dev mode (no build step)

```bash
git clone https://github.com/light-saber/Frames.git
cd Frames
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
python main.py
```

The app opens in a native macOS window via WebKit. Useful for development and testing.

---

## Workflow

1. **Select folder** — Paste the path to your RAW files, or click 📂 to open a native Finder folder picker
2. **Analyse** — Click **▶ Analyse Photos**. Progress streams live as each file is processed. Expect roughly 10–15 min for 200 photos on Intel Mac; significantly faster on Apple Silicon
3. **Review** — Photos are sorted best-first in a 3-column grid. Border colour indicates status: green = keep, red = reject, grey = pending
4. **Inspect** — Click any photo thumbnail to open the **detail panel**: see the full score breakdown, AI verdict, plain-language sharpness/exposure description, and individual AI sub-scores
5. **Auto-cull** — Use **Apply threshold** to bulk-mark photos above/below a score (default 65)
6. **Fine-tune** — Click **✓ Keep** or **✗ Reject** on any card, or use the buttons in the detail panel. The green/red highlight shows the active state. Use the filter tabs to focus on a subset
7. **Adjust colour** — Tweak the Color Settings sliders in the sidebar (applied at export time, not to the preview)
8. **Set export options** — Toggle **✨ Auto enhance colours** for adaptive per-photo enhancement. Choose **TIFF** or **JPEG** (set quality 60–100). Enter the destination path or click 📂
9. **Export** — Click **⬇ Export kept photos**. Files are saved as `{name}_edited.tiff` or `{name}_edited.jpg`

---

## Score Breakdown

| Score | Colour | Meaning |
|------:|--------|---------|
| 70–100 | Green | Sharp, well-exposed, good colour — likely a keeper |
| 45–69 | Amber | Acceptable — review manually |
| 0–44 | Red | Blurry, badly exposed, or flat colour |

Weights without AI: **Sharpness 55%** · **Exposure 30%** · **Saturation fitness 15%**

Weights with AI: **Sharpness 35%** · **AI score 35%** · **Exposure 20%** · **Saturation fitness 10%**

---

## AI Scoring (Optional)

Frames can use **Qwen2.5-VL 3B** running locally via [Ollama](https://ollama.com) to score each photo on composition, lighting, and subject clarity.

### Setup

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen2.5vl:3b
```

When Frames launches it automatically starts `ollama serve` if it isn't already running, and pulls `qwen2.5vl:3b` if the model is absent. The sidebar shows a green **● Ollama running** indicator once it's ready. Enable **AI scoring** before clicking **▶ Analyse Photos**. Each photo adds roughly 5–15 seconds of inference time.

### What the AI evaluates

| Sub-score | What it measures |
|-----------|-----------------|
| Composition (C) | Clear subject, sensible framing, no major obstructions |
| Lighting (L) | Balanced exposure, no blown highlights or crushed shadows |
| Subject Focus (F) | Primary subject sharp and in focus |

The AI also produces a one-sentence reason describing the key technical observation, visible in the detail panel.

### Usability flag

If the AI determines a photo is technically unusable (motion blur on the subject, wrong focus plane, unrecoverable exposure, or no identifiable subject), it sets an **AI reject** flag. The photo's overall score is hard-capped at 35 — well below the default 65 auto-select threshold — and a red **AI reject** badge appears on the card.

### Photo detail panel

Click any photo thumbnail to open the detail panel:

- **Overall score** — large colour-coded number with a plain verdict (Excellent / Good / Average / Below average / Poor)
- **AI verdict** — red warning box if flagged as not usable; purple note with the AI's observation otherwise
- **Technical breakdown** — Sharpness and Exposure bars with plain-language descriptions ("Tack sharp", "Blurry or out of focus", "Well exposed", "Exposure issues")
- **AI sub-scores** — Composition, Lighting, and Subject Focus bars when AI scoring was run
- **Keep / Reject buttons** — stay in sync with the grid

Close the panel with the **✕** button or press **Escape**.

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

All adjustments are applied in sequence on a `float32` [0, 1] image. When **Auto enhance** is on, it runs first (grey-world WB → CLAHE → saturation → S-curve), then manual corrections are applied on top.

---

## Known Limitations

- **Colour science** — Nikon Picture Control profiles (Vivid, Portrait, etc.) are not replicated. Export colours will differ from NX Studio's rendering
- **High-ISO noise** — Heavy noise can reduce sharpness scores due to the Laplacian variance method picking up texture as signal
- **AI model size** — Qwen2.5-VL 3B is a small vision model. Scores are a useful signal but not a substitute for human review
- **Performance** — First run on 200 photos takes ~10–15 min on Intel Mac; much faster on M-series. AI scoring adds ~5–15 s per photo
- **Output format** — Exports are standard TIFFs or JPEGs, not NX Studio native files. Compatible with Lightroom, Affinity Photo, and any TIFF/JPEG editor

---

## Tech Stack

| Layer | Library | Version |
|-------|---------|---------|
| Window | [pywebview](https://pywebview.flowrl.com) (WKWebView on macOS) | ≥ 4.4 |
| Backend | [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org) | ≥ 0.111 / ≥ 0.29 |
| Request validation | [Pydantic](https://docs.pydantic.dev) (via FastAPI) | v2 |
| Streaming | SSE (EventSource) over FastAPI `StreamingResponse`; `asyncio` + `concurrent.futures` for non-blocking analysis | stdlib |
| UI | Vanilla JS + HTML/CSS (no framework, dark/gold theme) | — |
| RAW decoding | [rawpy](https://pypi.org/project/rawpy/) + LibRaw 0.22 | ≥ 0.26 |
| Image processing | [OpenCV](https://pypi.org/project/opencv-python-headless/) (headless), [Pillow](https://python-pillow.org) | ≥ 4.9 / ≥ 10.0 |
| Numerics | [NumPy](https://numpy.org) | ≥ 1.26 |
| Folder picker | macOS `osascript` (AppleScript via `subprocess`) | — |
| AI scoring | [Ollama](https://ollama.com) HTTP API + `qwen2.5vl:3b` (optional second pass) | — |
| Distribution | [PyInstaller](https://pyinstaller.org) | ≥ 6.5 |

---

## Acknowledgements

- **[rawpy](https://github.com/letmaik/rawpy)** — MIT — Python wrapper for LibRaw
- **[LibRaw](https://github.com/LibRaw/LibRaw)** — LGPL 2.1 / CDDL 1.0 — RAW image decoding library
- **[OpenCV](https://github.com/opencv/opencv)** — Apache 2.0 — image processing (sharpness, exposure, colour transforms)
- **[Pillow](https://github.com/python-pillow/Pillow)** — HPND — image I/O and resizing
- **[NumPy](https://github.com/numpy/numpy)** — BSD 3-Clause — numerical array operations
- **[FastAPI](https://github.com/tiangolo/fastapi)** — MIT — modern Python web framework
- **[Pydantic](https://github.com/pydantic/pydantic)** — MIT — request body validation
- **[pywebview](https://github.com/r0x0r/pywebview)** — BSD 3-Clause — native window wrapper for web content
- **[DM Serif Display](https://fonts.google.com/specimen/DM+Serif+Display)** & **[DM Mono](https://fonts.google.com/specimen/DM+Mono)** — SIL Open Font License — typefaces via Google Fonts
