import pandas as pd
cards=pd.read_csv("data/cards.csv")
print(cards.columns.tolist())
print(cards.head())
