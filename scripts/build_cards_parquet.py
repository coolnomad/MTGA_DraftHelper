"""
Build cards.parquet from FIN.json and FCA.json with core card metadata.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_PATH = REPO_ROOT / "data" / "processed" / "cards.parquet"


def load_set(fname: str) -> pd.DataFrame:
    path = RAW_DIR / fname
    data = json.loads(path.read_text(encoding="utf-8"))
    cards = data.get("data", {}).get("cards", [])
    # pull out illustration id if present
    for c in cards:
        ident = c.get("identifiers", {}) or {}
        if "scryfallIllustrationId" in ident:
            c["illustration_id"] = ident.get("scryfallIllustrationId")
    df = pd.DataFrame(cards)
    df["set"] = fname.replace(".json", "")
    return df


def main():
    dfs = []
    for fname in ["FIN.json", "FCA.json"]:
        path = RAW_DIR / fname
        if path.exists():
            dfs.append(load_set(fname))
    cards = pd.concat(dfs, ignore_index=True)
    cols = {
        "name": "name",
        "colorIdentity": "color_identity",
        "colors": "colors",
        "manaCost": "mana_cost",
        "manaValue": "cmc",
        "rarity": "rarity",
        "type": "type_line",
        "types": "types",
        "supertypes": "supertypes",
        "subtypes": "subtypes",
        "set": "set",
        "illustration_id": "illustration_id",
    }
    cards_out = cards[[c for c in cols if c in cards.columns]].rename(columns=cols)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cards_out.to_parquet(OUT_PATH, index=False)
    print("wrote", len(cards_out), "cards to", OUT_PATH)


if __name__ == "__main__":
    main()
