"""Streamlit Cloud entrypoint delegating to app/streamlit_app.py."""

import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

app_path = ROOT / "app" / "streamlit_app.py"
runpy.run_path(str(app_path), run_name="__main__")
