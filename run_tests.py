"""Convenience script to run the MedTwin AI automated test suite.

Usage:
    python run_tests.py              # Run all unit and integration tests
    python run_tests.py --api        # Run API integration tests only
    python run_tests.py --safety     # Run clinical safety and guardrails tests only
    python run_tests.py --voice      # Run Whisper audio & voice service tests only
    python run_tests.py --coverage   # Run with test coverage report
"""

import sys
import subprocess
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent
    pytest_bin = root_dir / ".venv" / "Scripts" / "pytest.exe"
    
    # Fallback to system python -m pytest if venv path is not found
    if pytest_bin.exists():
        cmd = [str(pytest_bin), "-v"]
    else:
        cmd = [sys.executable, "-m", "pytest", "-v"]

    # Argument forwarding
    if "--api" in sys.argv:
        cmd.extend(["tests/test_api_integration.py"])
    elif "--safety" in sys.argv:
        cmd.extend(["tests/test_clinical_safety.py", "tests/test_guardrails.py"])
    elif "--voice" in sys.argv:
        cmd.extend(["tests/test_voice_service.py"])
    elif "--chat" in sys.argv:
        cmd.extend(["tests/test_chat_pipeline.py"])
    elif "--rag" in sys.argv:
        cmd.extend(["tests/test_rag.py"])
    elif "--twin" in sys.argv:
        cmd.extend(["tests/test_digital_twin.py"])
    elif "--coverage" in sys.argv:
        cmd.extend(["--cov=app", "--cov-report=term-missing"])
    else:
        # Default: run all tests in tests/
        cmd.extend(["tests/"])

    print("=" * 65)
    print("🩺 MedTwin AI Automated Test Suite Runner")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 65)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
