"""
Build bc_dataset parquet from raw 17Lands draft CSV.gz (draft_data_public.*.csv.gz).

The raw file has one row per pick with wide one-hot columns:
- pack_card_<CardName>
- pool_<CardName>

This script:
1) Reads the CSV.gz.
2) Extracts pack_card_ids (slugified names where pack_card_* == 1).
3) Extracts pool_counts (slugified names with counts > 0).
4) Adds metadata columns: draft_id, pack_number, pick_number, rank, user_n_games_bucket, user_game_win_rate_bucket, human_pick (slug).
5) Writes a compact bc_dataset.parquet.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "draft_data_public.FIN.PremierDraft.csv.gz"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "bc_dataset_full.parquet"


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def build_bc_dataset(input_path: Path, output_path: Path, max_rows: int | None = None) -> Path:
    df = pd.read_csv(input_path, nrows=max_rows)
    pack_cols = [c for c in df.columns if c.startswith("pack_card_")]
    pool_cols = [c for c in df.columns if c.startswith("pool_")]

    records: List[Dict] = []
    for _, row in df.iterrows():
        pack_cards = []
        for c in pack_cols:
            val = row[c]
            if pd.isna(val) or val == 0:
                continue
            name = c[len("pack_card_") :]
            pack_cards.append(slugify(name))
        pool_counts = {}
        for c in pool_cols:
            val = row[c]
            if pd.isna(val) or val == 0:
                continue
            name = slugify(c[len("pool_") :])
            pool_counts[name] = int(val)
        record = {
            "draft_id": row["draft_id"],
            "pack_number": int(row["pack_number"]),
            "pick_number": int(row["pick_number"]),
            "rank": row.get("rank"),
            "user_n_games_bucket": row.get("user_n_games_bucket"),
            "user_game_win_rate_bucket": row.get("user_game_win_rate_bucket"),
            "pack_card_ids": pack_cards,
            "pool_counts": pool_counts,
            "human_pick": slugify(row["pick"]),
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to draft_data_public*.csv.gz")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to write bc_dataset parquet")
    parser.add_argument("--max_rows", type=int, default=None, help="Optional cap on rows")
    args = parser.parse_args()

    out = build_bc_dataset(Path(args.input), Path(args.output), args.max_rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
