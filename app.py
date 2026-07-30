import sys
from pathlib import Path

# Add project directory to sys.path
root_dir = Path(__file__).resolve().parent
project_dir = root_dir / "AIML-Project-RollNo-12345" if (root_dir / "AIML-Project-RollNo-12345").exists() else root_dir
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

from backend.main import main

if __name__ == "__main__":
    main()
