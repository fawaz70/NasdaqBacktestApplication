
import os
from pathlib import Path
import databento as db
import pandas as pd

API_KEY = os.environ.get("DATABENTO_API_KEY")

if not API_KEY:
    raise RuntimeError("DATABENTO_API_KEY not found.")

client = db.Historical(API_KEY)

DATASET = "GLBX.MDP3"
SYMBOL = "NQ.v.0"
START = "2026-01-01"
END = "2026-10-15"

Path("data").mkdir(exist_ok=True)

print("Downloading NQ 1-minute historical data...")

data = client.timeseries.get_range(
    dataset=DATASET,
    symbols=SYMBOL,
    stype_in="continuous",
    schema="ohlcv-1m",
    start=START,
    end=END,
)

df = data.to_df()

# Preserve UTC timestamps and contract metadata.
df = df.reset_index()

# Save raw downloaded data.
df.to_csv("data/NQ_2026_full.csv", index=False)

print("\nDownload complete!")
print("Total candles:", len(df))
print("Columns:", df.columns.tolist())
print("First timestamp:", df.iloc[0, 0])
print("Last timestamp:", df.iloc[-1, 0])
print("Saved to: data/NQ_2026_full.csv")
