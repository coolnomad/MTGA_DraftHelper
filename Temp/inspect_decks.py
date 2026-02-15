import pandas as pd
from pathlib import Path
for p in [Path('data/processed/decks.parquet'), Path('data/processed/decks_with_preds.parquet')]:
    if p.exists():
        df=pd.read_parquet(p)
        print(p, 'rows', len(df))
        print('columns', df.columns.tolist())
