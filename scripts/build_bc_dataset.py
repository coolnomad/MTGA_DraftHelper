"""
Build behavior cloning dataset from drafts.parquet.

Output: data/processed/bc_dataset.parquet with columns:
- draft_id
- pack_number
- pick_number
- rank
- user_n_games_bucket
- user_game_win_rate_bucket
- pack_card_ids: list of card names in pack
- pool_counts: dict card->count before pick
- human_pick: chosen card name
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
INPUT_PATH = DATA_PROCESSED / "drafts.parquet"
OUTPUT_PATH = DATA_PROCESSED / "bc_dataset.parquet"


def build_bc_dataset():
    use_cols = [
        "draft_id",
        "pack_number",
        "pick_number",
        "rank",
        "user_n_games_bucket",
        "user_game_win_rate_bucket",
        "pick",
        "expansion",
        "event_type",
    ]
    # load to find pack_card columns
    df_head = pd.read_parquet(INPUT_PATH, columns=None, engine="pyarrow")
    pack_cols = [c for c in df_head.columns if c.startswith("pack_card_")]
    use_cols.extend(pack_cols)

    df = pd.read_parquet(INPUT_PATH, columns=use_cols)
    df = df[(df["expansion"] == "FIN") & (df["event_type"] == "PremierDraft")].copy()
    df.drop(columns=["expansion", "event_type"], inplace=True)

    rows = []
    pack_names = [c.replace("pack_card_", "") for c in pack_cols]

    def process_group(g: pd.DataFrame):
        pool = defaultdict(int)
        g = g.sort_values(["pack_number", "pick_number"])
        pack_values = g[pack_cols].to_numpy()
        for i, (_, row) in enumerate(g.iterrows()):
            present = pack_values[i] > 0
            pack_cards = [name for name, flag in zip(pack_names, present) if flag]
            rows.append(
                {
                    "draft_id": row["draft_id"],
                    "pack_number": int(row["pack_number"]),
                    "pick_number": int(row["pick_number"]),
                    "rank": row["rank"],
                    "user_n_games_bucket": row["user_n_games_bucket"],
                    "user_game_win_rate_bucket": row["user_game_win_rate_bucket"],
                    "pack_card_ids": pack_cards,
                    "pool_counts": dict(pool),
                    "human_pick": row["pick"],
                }
            )
            pool[row["pick"]] += 1

    for _, grp in df.groupby("draft_id"):
        process_group(grp)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUTPUT_PATH, index=False)
    return out_df


if __name__ == "__main__":
    out = build_bc_dataset()
    print("wrote", len(out), "rows to", OUTPUT_PATH)
