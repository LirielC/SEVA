"""Hugging Face Spaces entrypoint delegating to app/streamlit_app.py."""

import sys
from pathlib import Path

# Ensure src/ is in python path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Run the streamlit application script directly
app_path = ROOT / "app" / "streamlit_app.py"
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(app_path), "exec")
    exec(code, globals())
