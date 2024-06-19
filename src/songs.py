import pandas as pd

df = pd.read_csv("../data/spotify_millsongdata.csv")
print(df.size)

sizes = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 230600]

for n in sizes:
    df[:n].to_csv(f"../data/spotify_millsongdata_{n}.csv", index=False)

# print(df[df["artist"] == "Ed Sheeran"])
