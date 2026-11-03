import pandas as pd

df = pd.read_csv("data/raw/eplusout.csv")

print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.info())