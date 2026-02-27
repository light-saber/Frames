#!/usr/bin/env bash
set -e

echo "=== Frames Setup ==="
echo ""

# Check Python 3.9+
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null || true)
if [ -z "$PYTHON" ]; then
    echo "Error: Python 3 not found. Please install Python 3.9+."
    exit 1
fi

PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
PY_VERSION="$PY_MAJOR.$PY_MINOR"

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]; }; then
    echo "Error: Python 3.9+ required. Found Python $PY_VERSION."
    exit 1
fi

echo "✓ Python $PY_VERSION found at $PYTHON"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
    echo "✓ Virtual environment created."
else
    echo "✓ Virtual environment already exists."
fi

# Activate and install dependencies
source .venv/bin/activate
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To launch Frames:"
echo ""
echo "  source .venv/bin/activate && streamlit run app.py"
echo ""
