import pandas as pd
from pathlib import Path
paths = [Path('reports/replay_hero.parquet'), Path('data/processed/hero_pod_picks.parquet'), Path('data/processed/pod_picks.parquet')]
for p in paths:
    if p.exists():
        df = pd.read_parquet(p)
        print(p, 'rows', len(df))
        print('columns', df.columns.tolist())
        print(df.head(3))
