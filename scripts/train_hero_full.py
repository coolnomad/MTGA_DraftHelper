"""
Train hero/state-value model on full bc_dataset with OOF deck bump/effect labels merged.
Drops drafts without OOF labels.
"""
from pathlib import Path
import json
import pandas as pd
from hero_bot.train_state_value import train_state_value


def main():
    input_path = Path("data/processed/bc_dataset.parquet")
    oof_path = Path("reports/deck_effect_oof.parquet")
    if not input_path.exists() or not oof_path.exists():
        raise FileNotFoundError("Missing input parquet or OOF labels.")
    df = pd.read_parquet(input_path)
    oof = pd.read_parquet(oof_path)[["draft_id", "deck_bump_oof", "deck_effect_oof"]]
    df = df.merge(oof, on="draft_id", how="inner")
    metrics = train_state_value(
        df_override=df,
        target_column="deck_bump_oof",
        oof_target_column="deck_bump_oof",
        oof_labels=None,
        max_rows=None,  # use all merged rows
        n_jobs=-1,
        seed=1337,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
