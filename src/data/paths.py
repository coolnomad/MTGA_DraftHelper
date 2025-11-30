from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

# ensure dirs exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
