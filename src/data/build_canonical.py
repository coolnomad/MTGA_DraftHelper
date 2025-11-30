import pandas as pd
from pathlib import Path
from .paths import DATA_RAW, DATA_PROCESSED


# ---------------------------
# helpers
# ---------------------------

def _read_raw_files(pattern: str) -> pd.DataFrame:
    """
    read everything matching pattern from data/raw/
    supports csv and csv.gz
    """
    files = list(DATA_RAW.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files found matching {pattern} in {DATA_RAW}")

    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f, low_memory=False))
    return pd.concat(dfs, ignore_index=True)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    make columns consistent across expansions:
    - lowercase
    - replace spaces with underscores
    - ensure numeric-ish columns are numeric
    """
    df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"))

    # typical 17lands fields we want as numeric, but we won't force Int64
    numeric_cols = [
        "event_match_wins", "event_match_losses",
        "pack_number", "pick_number",
        "user_n_games_bucket",
        "user_game_win_rate_bucket",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------
# draft table
# ---------------------------

def build_drafts_table() -> Path:
    """
    build drafts.parquet
    combines all draft logs from 17lands
    """
    df = _read_raw_files("draft*.csv*")
    df = _standardize_columns(df)

    out = DATA_PROCESSED / "drafts.parquet"
    df.to_parquet(out, index=False)
    return out


# ---------------------------
# game table
# ---------------------------

def build_games_table() -> Path:
    """
    build games.parquet
    """
    df = _read_raw_files("game*.csv*")
    df = _standardize_columns(df)

    out = DATA_PROCESSED / "games.parquet"
    df.to_parquet(out, index=False)
    return out


# ---------------------------
# deck table (stub for now)
# ---------------------------

def build_decks_table() -> Path:
    """
    build decks.parquet with one row per completed draft run.

    completed run:
      - event_match_wins == 7  OR  event_match_losses == 3

    per draft_id:
      - aggregate run outcome from drafts.parquet
      - aggregate deck_* columns from games.parquet (mean across games)
      - aggregate meta + user stats from games.parquet (first row per draft)
      - compute:
          - run_wr
          - n_games
          - deck_size_avg
    """
    games_path = DATA_PROCESSED / "games.parquet"
    drafts_path = DATA_PROCESSED / "drafts.parquet"
    if not games_path.exists() or not drafts_path.exists():
        raise FileNotFoundError("build drafts and games before building decks")

    games = pd.read_parquet(games_path)
    drafts = pd.read_parquet(drafts_path)

    # ----- run-level outcome from drafts -----
    run_agg = (
        drafts.groupby("draft_id", as_index=False)
              .agg(
                  event_match_wins=("event_match_wins", "max"),
                  event_match_losses=("event_match_losses", "max"),
              )
    )

    total_matches = run_agg["event_match_wins"] + run_agg["event_match_losses"]
    run_agg["run_wr"] = run_agg["event_match_wins"] / total_matches.replace(0, pd.NA)

    # only 7-0 / x-3 runs
    complete_mask = (run_agg["event_match_wins"] == 7) | (run_agg["event_match_losses"] == 3)
    run_complete = run_agg[complete_mask].copy()

    # restrict games to complete runs
    games_complete = games.merge(
        run_complete[["draft_id"]],
        on="draft_id",
        how="inner",
    )

    # ----- deck composition: mean deck_* across games -----
    deck_cols = [c for c in games_complete.columns if c.startswith("deck_")]
    if not deck_cols:
        raise ValueError("no deck_* columns found in games table")

    deck_means = (
        games_complete
        .groupby("draft_id")[deck_cols]
        .mean()
    )

    # average deck size
    deck_means["deck_size_avg"] = deck_means[deck_cols].sum(axis=1)

    # ----- meta + user stats -----
    base_meta_cols = [
        "expansion",
        "event_type",
        "rank",
        "main_colors",
        "splash_colors",
    ]
    # any column containing 'user' in its name
    user_cols = [c for c in games_complete.columns if "user" in c.lower()]

    meta_cols = []
    for c in base_meta_cols + user_cols:
        if c in games_complete.columns and c not in meta_cols:
            meta_cols.append(c)

    meta = (
        games_complete
        .groupby("draft_id")[meta_cols]
        .first()
    )

    # number of games played
    n_games = games_complete.groupby("draft_id").size().rename("n_games")

    # ----- combine everything -----
    decks = (
        deck_means
        .join(meta, how="left")
        .join(n_games, how="left")
        .reset_index()  # bring draft_id back as column
    )

    decks = decks.merge(
        run_complete[["draft_id", "event_match_wins", "event_match_losses", "run_wr"]],
        on="draft_id",
        how="left",
    )

    # filter to decks with avg size >= 40 cards
    decks = decks[decks["deck_size_avg"] >= 40].reset_index(drop=True)

    out = DATA_PROCESSED / "decks.parquet"
    decks.to_parquet(out, index=False)
    return out


# ---------------------------
# master builder
# ---------------------------

def build_canonical_tables() -> None:
    print("building drafts...")
    d_path = build_drafts_table()
    print(f"drafts → {d_path}")

    print("building games...")
    g_path = build_games_table()
    print(f"games → {g_path}")

    print("building decks...")
    deck_path = build_decks_table()
    print(f"decks → {deck_path}")
