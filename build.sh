#!/usr/bin/env bash
set -euo pipefail

echo "==> Frames build script"
echo ""

# ── Check Python 3.11+ ────────────────────────────────────────────────────────
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
  echo "✗ python3 not found. Install via Homebrew: brew install python@3.11"
  exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  echo "✗ Python 3.11+ required (found $PY_VERSION)"
  echo "  Install: brew install python@3.11"
  exit 1
fi

echo "✓ Python $PY_VERSION"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "==> Creating virtual environment…"
  "$PYTHON" -m venv .venv
fi

echo "==> Installing dependencies…"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# ── PyInstaller build ─────────────────────────────────────────────────────────
echo "==> Building Frames.app with PyInstaller…"
.venv/bin/pyinstaller Frames.spec --clean --noconfirm

echo ""
echo "✓ Build complete: dist/Frames.app"
echo ""
echo "  Run now:    open dist/Frames.app"
echo "  Install:    cp -r dist/Frames.app /Applications/"
