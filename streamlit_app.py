"""Main entrypoint for MedTwin AI Streamlit frontend.

Can be run directly via:
    streamlit run streamlit_app.py
or
    streamlit run app/frontend/app.py
"""

import sys
import socket
import threading
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def is_backend_running(host: str = "127.0.0.1", port: int = 8000) -> bool:
    """Checks if the FastAPI backend is already listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_backend_server():
    """Starts FastAPI in a background daemon thread for single-container cloud hosting."""
    try:
        from app.db.session import init_db
        init_db()
    except Exception:
        pass

    try:
        import uvicorn
        from app.api.main import app as fastapi_app
        uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")
    except Exception as e:
        print(f"Backend startup notice: {e}")


# If FastAPI is not running on port 8000, automatically launch it in background
if not is_backend_running():
    backend_thread = threading.Thread(target=start_backend_server, daemon=True)
    backend_thread.start()
    for _ in range(12):
        if is_backend_running():
            break
        time.sleep(0.3)

# Execute app/frontend/app.py
import runpy

app_path = root_dir / "app" / "frontend" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
