"""Root entrypoint for Streamlit Community Cloud and local deployments."""
import runpy
import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' resolves seamlessly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Execute the main Streamlit application
APP_PATH = ROOT_DIR / "src" / "ui" / "app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
