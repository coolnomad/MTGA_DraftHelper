"""
Convert bc_dataset.parquet into tensors/features for behavior cloning.
Outputs: data/processed/bc_tensors.parquet with columns:
- draft_id, pack_number, pick_number, skill_bucket, human_pick
- state_vec: list[float]
- card_features: list of {card_id, feats...} (names for now)

Note: This is a minimal feature pass; replace with richer encoding using cards.parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import argparse
import re

import pandas as pd
import numpy as np

from state_encoding.encoder import encode_state, encode_card

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED / "bc_dataset.parquet"
OUTPUT_PATH = PROCESSED / "bc_tensors.parquet"


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def build_bc_tensors(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH, max_rows: Optional[int] = None):
    df = pd.read_parquet(input_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=1337)
    rows = []
    for _, row in df.iterrows():
        state_vec = encode_state(
            row["pool_counts"],
            pack_no=row["pack_number"],
            pick_no=row["pick_number"],
            skill_bucket=row.get("rank"),
        )
        card_feats = [(card, encode_card(card)) for card in row["pack_card_ids"]]
        human_pick = row["human_pick"]
        human_pick_slug = _slugify(human_pick)
        rows.append(
            {
                "draft_id": row["draft_id"],
                "pack_number": row["pack_number"],
                "pick_number": row["pick_number"],
                "skill_bucket": row.get("rank"),
                "human_pick": human_pick_slug,
                "state_vec": state_vec.tolist(),
                "pack_cards": [_slugify(c) for c, _ in card_feats],
                "card_features": [f.tolist() for _, f in card_feats],
            }
        )
    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--max_rows", type=int, default=None, help="Optional sample cap.")
    args = parser.parse_args()
    out = build_bc_tensors(Path(args.input), Path(args.output), args.max_rows)
    print("wrote", len(out), "rows to", args.output)
