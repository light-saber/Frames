#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
exec streamlit run app.py --server.headless true
