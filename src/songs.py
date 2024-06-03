import pandas as pd

df = pd.read_csv("../data/spotify_millsongdata.csv")
print(df.size)

# save to .csv just 1000 rows
# df[:1000].to_csv("../data/spotify_millsongdata_1000.csv", index=False)

print(df[df["artist"] == "Ed Sheeran"])
