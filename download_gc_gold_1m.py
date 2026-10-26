"""Download CME COMEX Gold (GC) continuous futures 1-minute OHLCV from Databento.

Requires a Databento account with GLBX.MDP3 historical access.
Usage:
  pip install databento pandas
  set DATABENTO_API_KEY=YOUR_KEY             # Windows CMD
  $env:DATABENTO_API_KEY='YOUR_KEY'           # PowerShell
  python download_gc_gold_1m.py --start 2026-01-01 --end 2026-10-16

End date is exclusive. GC.v.0 is volume-based front continuous contract.
"""
import argparse
import os
from pathlib import Path
import databento as db
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-01-01')
    ap.add_argument('--end', default='2026-10-16', help='Exclusive UTC end date')
    ap.add_argument('--symbol', default='GC.v.0', help='Databento continuous GC symbol')
    ap.add_argument('--dataset', default='GLBX.MDP3')
    ap.add_argument('--output', default='GC_2026_full.csv')
    args = ap.parse_args()
    key = os.getenv('DATABENTO_API_KEY')
    if not key:
        raise SystemExit('Set DATABENTO_API_KEY in your environment before running.')
    client = db.Historical(key)
    # OHLCV-1m is aggregated exchange trades, timestamps are UTC minute starts.
    data = client.timeseries.get_range(
        dataset=args.dataset,
        symbols=[args.symbol],
        stype_in='continuous',
        schema='ohlcv-1m',
        start=args.start,
        end=args.end,
    )
    df = data.to_df().reset_index()
    if df.empty:
        raise SystemExit('No GC data returned; check entitlement, symbol and date range.')
    needed = ['ts_event', 'instrument_id', 'open', 'high', 'low', 'close']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f'Unexpected Databento response; missing columns: {missing}; got: {list(df.columns)}')
    df = df.sort_values('ts_event').drop_duplicates('ts_event')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f'Saved {len(df):,} bars to {output.resolve()}')
    print(f'UTC range: {df.ts_event.min()} through {df.ts_event.max()}')
    print(f'Instrument IDs: {df.instrument_id.nunique()} (continuous contract rolls)')
    print('IMPORTANT: GC has $100/point/contract, 0.10 tick, and MGC has $10/point/contract.')
    print('Do not run the NQ backtester unchanged on GC; adapt tick, sizing, fees and session behavior.')

if __name__ == '__main__':
    main()
