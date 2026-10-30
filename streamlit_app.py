"""Main entrypoint for MedTwin AI Streamlit frontend.

Can be run directly via:
    streamlit run streamlit_app.py
or
    streamlit run app/frontend/app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute app/frontend/app.py
import runpy

app_path = root_dir / "app" / "frontend" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
