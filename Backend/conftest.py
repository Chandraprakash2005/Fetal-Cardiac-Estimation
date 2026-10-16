"""conftest.py

Ensures `import fcac` works when running pytest against a checkout that hasn't been
`pip install -e .`'d yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
