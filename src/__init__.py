from __future__ import annotations

import sys
from pathlib import Path

# Allow the historical top-level imports such as "from data import ..."
# to work when the project is executed as the package "src".
_SRC_ROOT = Path(__file__).resolve().parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
