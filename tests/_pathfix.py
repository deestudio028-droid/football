"""Import-path bootstrap shared by every test module in this directory.

The sandbox running this project's test suite has no internet access to
install pytest, so tests are written against the standard-library
`unittest` module instead (no third-party test framework dependency).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
