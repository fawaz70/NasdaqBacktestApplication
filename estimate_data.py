
import os
import databento as db

api_key = os.environ.get("DATABENTO_API_KEY")

if not api_key:
    raise RuntimeError("Databento API key not found.")

client = db.Historical(api_key)

# NQ continuous front-month futures
# Full electronic session, not just regular hours
dataset = "GLBX.MDP3"
symbol = "NQ.v.0"

start = "2026-01-01"
end = "2026-10-15"

print("Checking historical data cost...")

cost = client.metadata.get_cost(
    dataset=dataset,
    symbols=symbol,
    stype_in="continuous",
    schema="ohlcv-1m",
    start=start,
    end=end,
)

print(f"\nEstimated cost: ${cost:.2f} USD")
