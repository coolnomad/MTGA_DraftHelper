"""
Helpers to log self-play draft data to parquet.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def log_picks(logs: List[Dict[str, Any]], output: Path) -> Path:
    """
    logs: list of dicts with keys:
      seat, pack_number, pick_number, pack_cards, pool_counts, chosen_card, policy
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_parquet(output, index=False)
    return output


__all__ = ["log_picks"]
