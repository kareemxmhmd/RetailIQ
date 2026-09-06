"""Alias entrypoint pointing to streamlit_app.py."""
import runpy
from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "streamlit_app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
