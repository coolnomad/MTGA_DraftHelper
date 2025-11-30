from pathlib import Path
import sys

# ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_canonical import build_canonical_tables


if __name__ == "__main__":
    build_canonical_tables()
