import pandas as pd
from .paths import DATA_PROCESSED


def _load(name: str) -> pd.DataFrame:
    path = DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. run build_canonical_tables() first.")
    return pd.read_parquet(path)


def load_drafts() -> pd.DataFrame:
    return _load("drafts.parquet")


def load_games() -> pd.DataFrame:
    return _load("games.parquet")


def load_decks() -> pd.DataFrame:
    return _load("decks.parquet")
