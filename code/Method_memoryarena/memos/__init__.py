import os
import sys


_LOCAL_MEMOS_SRC = os.path.join(os.path.dirname(__file__), "src")
if _LOCAL_MEMOS_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_MEMOS_SRC)

from .main import run_memos