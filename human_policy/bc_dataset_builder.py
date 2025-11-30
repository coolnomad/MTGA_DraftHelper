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
from typing import Dict, List

import pandas as pd
import numpy as np

from state_encoding.encoder import encode_state, encode_card

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed"
INPUT_PATH = PROCESSED / "bc_dataset.parquet"
OUTPUT_PATH = PROCESSED / "bc_tensors.parquet"


def build_bc_tensors():
    df = pd.read_parquet(INPUT_PATH)
    rows = []
    for _, row in df.iterrows():
        state_vec = encode_state(
            row["pool_counts"],
            pack_no=row["pack_number"],
            pick_no=row["pick_number"],
            skill_bucket=row.get("rank"),
        )
        card_feats = [(card, encode_card(card)) for card in row["pack_card_ids"]]
        rows.append(
            {
                "draft_id": row["draft_id"],
                "pack_number": row["pack_number"],
                "pick_number": row["pick_number"],
                "skill_bucket": row.get("rank"),
                "human_pick": row["human_pick"],
                "state_vec": state_vec.tolist(),
                "pack_cards": [c for c, _ in card_feats],
                "card_features": [f.tolist() for _, f in card_feats],
            }
        )
    out = pd.DataFrame(rows)
    out.to_parquet(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    out = build_bc_tensors()
    print("wrote", len(out), "rows to", OUTPUT_PATH)
