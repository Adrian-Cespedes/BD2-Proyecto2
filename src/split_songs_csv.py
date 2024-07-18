import pandas as pd

df = pd.read_csv("../data/spotify_millsongdata.csv")
print(df.size)
print(df.shape[0])
print(len(df))

sizes = [1000, 2000, 4000, 8000, 16000, 32000, 57650]

for n in sizes:
    df[:n].to_csv(f"../data/spotify_millsongdata_{n}.csv", index=False)
