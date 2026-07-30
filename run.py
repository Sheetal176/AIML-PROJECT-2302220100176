import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir / "AIML-Project-RollNo-12345"

    # Virtual environment checks
    venv_python = base_dir / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = base_dir / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    app_path = project_dir / "app.py"

    print("=" * 65)
    print(" 🚀 Student Placement Prediction - Streamlit Site Launcher")
    print("=" * 65)
    print(f"📁 Project Directory  : {project_dir}")
    print(f"🐍 Python Executable  : {venv_python}")
    print(f"⚡ Streamlit App      : {app_path}")
    print("-" * 65)

    cmd = [
        str(venv_python),
        "-m",
        "streamlit",
        "run",
        str(app_path)
    ]

    print("🔹 Starting Streamlit Dashboard...")
    try:
        proc = subprocess.Popen(cmd, cwd=str(project_dir))
        time.sleep(3.5)
        
        # Open in browser
        try:
            webbrowser.open("http://localhost:8501")
        except Exception:
            pass
            
        proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stop request received. Terminating Streamlit server...")
    except Exception as e:
        print(f"⚠️ Error starting Streamlit: {e}")

if __name__ == "__main__":
    main()
