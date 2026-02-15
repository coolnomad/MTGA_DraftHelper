import pandas as pd
from pathlib import Path
path = Path('reports/tournament_hero.parquet')
df = pd.read_parquet(path)
print('rows', len(df))
print(df.head())
print('\ncolumns', df.columns.tolist())
