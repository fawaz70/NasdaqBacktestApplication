"""MNQ backtest: 15m/30m/1h FVG, rolling 15m TS, 1m IFVG; compare 1R/1.5R/2R.

No OTE. Three independent simulations, each with its own position management.
Uses NQ prices as MNQ proxy ($2/point), targets approximately $200 modeled stop risk ($180-$220); skips setups exceeding $220 with one MNQ.
Rejects TS sweeps deeper than 40 points beyond the previous 15m candle extreme. After a STOP, blocks that HTF FVG for the rest of the New York calendar day. Adds 20:00-22:00 NY entries to 09:45-15:00 NY entries.
Requires pandas numpy. Run: python backtest.py --data data/NQ_2026_full.csv
Limitations: 1m OHLC cannot always resolve intraminute sequence; stop-first
when both targets touched; one open trade at a time per RR scenario.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from math import floor
from pathlib import Path
import pandas as pd
import numpy as np

TZ = 'America/New_York'
TFS = (15, 30, 60)
TICK = 0.25
MNQ_POINT_VALUE = 2.0
@dataclass
class FVG:
    id: int
    tf: int
    direction: int
    bottom: float
    top: float
    created: pd.Timestamp
    instrument_id: int
    active: bool = True

def load_data(path):
    df = pd.read_csv(path)
    required = {'ts_event', 'open', 'high', 'low', 'close', 'instrument_id'}
    if not required.issubset(df.columns):
        raise ValueError('Missing columns: ' + str(sorted(required - set(df.columns))))
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True).dt.tz_convert(TZ)
    df = df.sort_values('ts_event').drop_duplicates('ts_event').set_index('ts_event')
    for col in ('open', 'high', 'low', 'close'):
        df[col] = pd.to_numeric(df[col], errors='raise')
    df['instrument_id'] = df['instrument_id'].astype('int64')
    return df


def htf_events(df):
    """Publish each candle and confirmed pivot only when the required candle closes."""
    events = defaultdict(list)
    for tf in TFS:
        resampled = df.resample(f'{tf}min', origin='start_day', closed='left', label='left')
        bars = resampled.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                              'instrument_id': ['first', 'nunique', 'count']})
        bars.columns = ['open', 'high', 'low', 'close', 'contract', 'ncontracts', 'count']
        valid = (bars['count'] == tf) & (bars['ncontracts'] == 1)
        consecutive = bars.index.to_series().diff().eq(pd.Timedelta(minutes=tf))
        eligible = valid.copy()
        for shift in (1, 2):
            eligible &= (valid.shift(shift, fill_value=False) &
                         bars['contract'].eq(bars['contract'].shift(shift)))
        eligible &= consecutive & consecutive.shift(1, fill_value=False)
        bull = eligible & bars['low'].gt(bars['high'].shift(2))
        bear = eligible & bars['high'].lt(bars['low'].shift(2))
        for start, bar in bars[valid].iterrows():
            close_at = start + pd.Timedelta(minutes=tf)
            cid = int(bar['contract'])
            events[close_at].append(('htf_close', tf, float(bar['close']), cid))
            if bull.at[start]:
                old = start - pd.Timedelta(minutes=2 * tf)
                events[close_at].append(('new_fvg', tf, 1, float(bars.at[old, 'high']),
                                         float(bar['low']), cid))
            elif bear.at[start]:
                old = start - pd.Timedelta(minutes=2 * tf)
                events[close_at].append(('new_fvg', tf, -1, float(bar['high']),
                                         float(bars.at[old, 'low']), cid))
    return events


def simulate(df, *, start_date, end_date, rr=1.0, events=None,
             risk_budget=200.0, stop_buffer=2.0, slip_ticks=1,
             commission_side=1.25, max_trade_days=15, risk_min=180.0, risk_max=220.0,
             max_sweep_points=40.0):
    if events is None:
        events = htf_events(df)
    active_gaps = []
    gaps_by_tf_contract = defaultdict(list)
    next_gap_id = 0
    trades = []
    trade = None
    prev_time = prev_contract = None
    recent = deque(maxlen=2)
    latest_ifvg_source = {1: None, -1: None}
    current_bucket = None
    bucket = previous_15 = None
    armed = []    # refreshed every completed 15m candle; each may become a new reference
    sweep = None
    stats = defaultdict(int)
    failed_fvg_day = {}  # FVG id -> NY calendar date of most recent stop-out
    start_date = pd.Timestamp(start_date).date()
    end_date = pd.Timestamp(end_date).date()
    minute = pd.Timedelta(minutes=1)
    fifteen = pd.Timedelta(minutes=15)

    def unresolved(reason):
        nonlocal trade
        if trade is not None:
            trade.update(exit_time='', exit_price=np.nan, gross_usd=np.nan,
                         fees_usd=np.nan, net_usd=np.nan, realized_R=np.nan,
                         result='unresolved', exit_reason=reason)
            trades.append(trade)
            trade = None

    def close_trade(price, at, reason):
        nonlocal trade
        if reason.startswith("STOP"):
            failed_fvg_day[trade["htf_fvg_id"]] = at.date()
            stats["stopped_fvg_lockouts"] += 1
        gross = (price - trade['entry']) * trade['direction'] * MNQ_POINT_VALUE * trade['contracts']
        fees = commission_side * 2 * trade['contracts']
        net = gross - fees
        trade.update(exit_time=at.isoformat(), exit_price=price,
                     gross_usd=round(gross, 2), fees_usd=round(fees, 2),
                     net_usd=round(net, 2),
                     realized_R=round(net / trade['planned_risk_usd'], 5),
                     result='win' if net > 0 else 'loss' if net < 0 else 'flat',
                     exit_reason=reason)
        trades.append(trade)
        trade = None

    for n, r in enumerate(df.itertuples(index=True, name='Bar'), 1):
        t = r.Index
        end = t + minute
        hi, lo, cl = float(r.high), float(r.low), float(r.close)
        cid = int(r.instrument_id)
        continuous = (prev_time is not None and t - prev_time == minute and cid == prev_contract)
        if not continuous:
            if prev_time is not None:
                stats['discontinuities'] += 1
            if prev_contract is not None and prev_contract != cid:
                stats['rollovers'] += 1
                active_gaps.clear()
                gaps_by_tf_contract.clear()
            unresolved('UNRESOLVED_DATA_GAP_OR_ROLLOVER')
            recent.clear()
            latest_ifvg_source = {1: None, -1: None}
            previous_15 = None
            current_bucket = None
            bucket = None
            armed = []
            sweep = None

        new_bucket = t.floor('15min')
        if new_bucket != current_bucket:
            if bucket is not None and bucket['count'] == 15 and bucket['complete']:
                previous_15 = bucket.copy()
                # Only gaps that were available prior to the reference candle.
                candidates = [g for g in active_gaps if g.active and g.instrument_id == bucket['cid']
                              and g.created <= bucket['start']
                              and bucket['high'] >= g.bottom and bucket['low'] <= g.top]
                stats['reference_fvg_touches'] += len(candidates)
                armed = candidates
            else:
                previous_15 = None
                armed = []
            current_bucket = new_bucket
            bucket = {'start': new_bucket, 'high': hi, 'low': lo, 'count': 0,
                      'complete': True, 'cid': cid}
            sweep = {'low_seen': False, 'high_seen': False,
                     'low_return': False, 'high_return': False,
                     'low_extreme': float('inf'), 'high_extreme': -float('inf'),
                     'low_return_time': None, 'high_return_time': None}
        if bucket['cid'] != cid:
            bucket['complete'] = False
        bucket['count'] += 1
        bucket['high'] = max(bucket['high'], hi)
        bucket['low'] = min(bucket['low'], lo)

        # Do not assess stops/targets on an entry-confirmation minute.
        if trade is not None:
            if end - pd.Timestamp(trade['entry_time']) > pd.Timedelta(days=max_trade_days):
                unresolved('UNRESOLVED_MAX_HOLD')
            else:
                stop_hit = lo <= trade['stop'] if trade['direction'] == 1 else hi >= trade['stop']
                target_hit = hi >= trade['target'] if trade['direction'] == 1 else lo <= trade['target']
                if stop_hit and target_hit:
                    stats['both_hit_same_minute'] += 1
                    close_trade(trade['stop'] - trade['direction'] * slip_ticks * TICK,
                                end, 'STOP_BOTH_HIT_CONSERVATIVE')
                elif stop_hit:
                    close_trade(trade['stop'] - trade['direction'] * slip_ticks * TICK,
                                end, 'STOP')
                elif target_hit:
                    close_trade(trade['target'] - trade['direction'] * slip_ticks * TICK,
                                end, 'TARGET')

        # All events at end are now available, never before their confirmation.
        for event in events.get(end, ()):
            kind = event[0]
            if kind == 'htf_close':
                _, tf, price, event_cid = event
                for g in gaps_by_tf_contract.get((tf, event_cid), ()):
                    if g.active and ((g.direction == 1 and price < g.bottom) or
                                     (g.direction == -1 and price > g.top)):
                        g.active = False
            elif kind == 'new_fvg':
                _, tf, direction, bottom, top, event_cid = event
                next_gap_id += 1
                g = FVG(next_gap_id, tf, direction, bottom, top, end, event_cid)
                active_gaps.append(g)
                gaps_by_tf_contract[(tf, event_cid)].append(g)
                stats['htf_fvgs_created'] += 1
        if n % 2000 == 0:
            active_gaps = [g for g in active_gaps if g.active]
            for key, values in list(gaps_by_tf_contract.items()):
                live = [g for g in values if g.active]
                if live:
                    gaps_by_tf_contract[key] = live
                else:
                    del gaps_by_tf_contract[key]

        # Each completed 15-minute candle touching a valid FVG becomes the new
        # reference. If its next candle expires without entry, repeat the test
        # using that just-completed candle, provided it also touched the FVG.
        # Thus the next TS always sweeps the immediately preceding 15m candle.
        if previous_15 is not None and armed and previous_15['start'] + fifteen == current_bucket:
            if lo < previous_15['low']:
                sweep['low_seen'] = True
                sweep['low_extreme'] = min(sweep['low_extreme'], lo)
            if hi > previous_15['high']:
                sweep['high_seen'] = True
                sweep['high_extreme'] = max(sweep['high_extreme'], hi)
            if sweep['low_seen'] and cl >= previous_15['low']:
                if not sweep['low_return']:
                    sweep['low_return_time'] = end
                sweep['low_return'] = True
            if sweep['high_seen'] and cl <= previous_15['high']:
                if not sweep['high_return']:
                    sweep['high_return_time'] = end
                sweep['high_return'] = True
        else:
            armed = []

        # Detect the newest opposite 1m FVG (3 consecutive, same-contract minutes).
        if len(recent) == 2:
            old, middle = recent[0], recent[1]
            if (old['t'] + 2 * minute == t and old['cid'] == cid == middle['cid']):
                if lo > old['high']:
                    latest_ifvg_source[-1] = {'bottom': old['high'], 'top': lo, 'created': end}
                if hi < old['low']:
                    latest_ifvg_source[1] = {'bottom': hi, 'top': old['low'], 'created': end}

        if trade is None and armed:
            clock = end.hour * 60 + end.minute
            in_ny = 9 * 60 + 45 <= clock <= 15 * 60
            in_asia = 20 * 60 <= clock <= 22 * 60
            if (in_ny or in_asia) and start_date <= end.date() <= end_date:
                for direction in (1, -1):
                    if not (sweep['low_return'] if direction == 1 else sweep['high_return']):
                        continue
                    f = latest_ifvg_source[direction]
                    if f is None or f['created'] >= end:
                        continue
                    # A 1m FVG that existed BEFORE the 15m sweep returned
                    # inside its reference candle cannot be used. We wait for
                    # a NEW 1m FVG formed strictly after that return.
                    returned_at = sweep['low_return_time'] if direction == 1 else sweep['high_return_time']
                    if returned_at is None or f['created'] <= returned_at:
                        continue
                    if not (cl > f['top'] if direction == 1 else cl < f['bottom']):
                        continue
                    options = [g for g in armed if g.active and g.direction == direction
                               and g.instrument_id == cid]
                    if not options:
                        continue
                    extreme = sweep['low_extreme'] if direction == 1 else sweep['high_extreme']
                    # Sweep depth is measured BEYOND the immediately previous 15m
                    # candle's low (bullish) or high (bearish), not FVG distance.
                    sweep_depth = ((previous_15['low'] - extreme) if direction == 1
                                   else (extreme - previous_15['high']))
                    if sweep_depth > max_sweep_points:
                        stats['deep_sweep_skips'] += 1
                        continue
                    # Distance from the TS extreme to the nearest FVG boundary.
                    # A bullish sweep far BELOW the FVG or a bearish sweep far
                    # ABOVE the FVG is not a tight FVG-driven Turtle Soup.
                    def zone_distance(g):
                        if direction == 1:
                            return max(0.0, g.bottom - extreme)
                        return max(0.0, extreme - g.top)
                    # A deep sweep is valid if it returned inside the prior
                    # 15m candle and a fresh IFVG confirms afterward.
                    # Do not reject solely because of FVG-to-sweep distance.
                    options = [g for g in options if failed_fvg_day.get(g.id) != end.date()]
                    if not options:
                        stats['failed_fvg_same_day_skips'] += 1
                        continue
                    gap = max(options, key=lambda g: (g.created, g.tf, g.id))
                    extreme = sweep['low_extreme'] if direction == 1 else sweep['high_extreme']
                    entry = cl + direction * slip_ticks * TICK
                    stop = extreme - direction * stop_buffer
                    distance = (entry - stop) * direction
                    if distance <= 0:
                        stats['bad_stop_skips'] += 1
                        continue
                    # Budget includes an additional STOP slippage tick and both commissions.
                    planned_loss_per_mnq = ((distance + slip_ticks * TICK) * MNQ_POINT_VALUE
                                            + 2 * commission_side)
                    # Choose whole contracts closest to target risk, enforcing a
                    # hard maximum of $220 (including modeled exit costs).
                    # If one MNQ exceeds $220, skip: stop stays at TS extreme.
                    if planned_loss_per_mnq > risk_max:
                        stats['over_max_risk_skips'] += 1
                        continue
                    cap = max(1, floor((risk_max + 1e-9) / planned_loss_per_mnq))
                    contracts = min(range(1, cap + 1),
                                    key=lambda k: (abs(k * planned_loss_per_mnq - risk_budget), k))
                    if contracts * planned_loss_per_mnq < risk_min:
                        stats['below_risk_band_entries'] += 1
                    target = entry + direction * distance * rr
                    planned = planned_loss_per_mnq * contracts
                    trade = {
                        'direction': direction, 'entry_time': end.isoformat(), 'entry': entry,
                        'stop': stop, 'target': target, 'risk_points': distance, 'rr_target': rr,
                        'contracts': contracts, 'planned_risk_usd': round(planned, 2),
                        'risk_budget_usd': risk_budget, 'in_180_220_band': risk_min <= planned <= risk_max,
                        'htf_fvg_to_ts_points': round(zone_distance(gap), 2),
                        'ts_sweep_depth_points': round(sweep_depth, 2),
                        'entry_session': 'ASIA_20_22' if in_asia else 'NY_0945_1500',
                        'htf_tf': gap.tf, 'htf_fvg_id': gap.id,
                        'htf_fvg_created': gap.created.isoformat(),
                        'htf_fvg_bottom': gap.bottom, 'htf_fvg_top': gap.top,
                        'reference_candle_start': previous_15['start'].isoformat(),
                        'ts_candle_start': current_bucket.isoformat(),
                        'ts_extreme_at_entry': extreme,
                        'ts_return_time': returned_at.isoformat(),
                        'ifvg_created': f['created'].isoformat(),
                        'ifvg_bottom': f['bottom'], 'ifvg_top': f['top'], 'instrument_id': cid,
                    }
                    stats['entries'] += 1
                    latest_ifvg_source[direction] = None
                    break

        # Inverted IFVG source must not be recycled, even if no TS qualified.
        for direction in (1, -1):
            f = latest_ifvg_source[direction]
            if f and f['created'] < end and (cl > f['top'] if direction == 1 else cl < f['bottom']):
                latest_ifvg_source[direction] = None
        recent.append({'t': t, 'high': hi, 'low': lo, 'cid': cid})
        prev_time, prev_contract = t, cid
        if n % 75000 == 0:
            print(f'Processed {n:,}/{len(df):,} minute bars; entries: {stats["entries"]}', flush=True)

    unresolved('UNRESOLVED_END_OF_DATA')
    return pd.DataFrame(trades), dict(stats)


def summarize(trades):
    columns = ['htf_tf', 'completed_trades', 'wins', 'losses', 'win_rate_pct',
               'net_usd', 'expectancy_usd', 'avg_realized_R', 'profit_factor',
               'max_drawdown_usd', 'unresolved_trades', 'total_mnq_contracts']
    if trades.empty:
        return pd.DataFrame(columns=columns)
    completed = trades[trades['result'].isin(['win', 'loss', 'flat'])].copy()
    report = []
    for label, segment in [('ALL', completed)] + [(str(tf), completed[completed.htf_tf == tf]) for tf in TFS]:
        if segment.empty:
            continue
        pnl = segment['net_usd'].astype(float)
        pnl = pnl.iloc[np.argsort(pd.to_datetime(segment['exit_time'], utc=True).to_numpy())]
        equity = pnl.cumsum()
        dd = (equity - equity.cummax().clip(lower=0)).min()
        gains = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        all_in_category = trades if label == 'ALL' else trades[trades.htf_tf == int(label)]
        report.append({
            'htf_tf': label, 'completed_trades': len(segment),
            'wins': int((pnl > 0).sum()), 'losses': int((pnl < 0).sum()),
            'win_rate_pct': round(100 * (pnl > 0).mean(), 2),
            'net_usd': round(pnl.sum(), 2), 'expectancy_usd': round(pnl.mean(), 2),
            'avg_realized_R': round(segment['realized_R'].astype(float).mean(), 4),
            'profit_factor': round(gains / losses, 4) if losses else np.inf,
            'max_drawdown_usd': round(float(dd), 2),
            'unresolved_trades': int((all_in_category.result == 'unresolved').sum()),
            'total_mnq_contracts': int(segment['contracts'].sum())
        })
    return pd.DataFrame(report, columns=columns)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='data/NQ_2026_full.csv')
    ap.add_argument('--start', default='2026-01-01')
    ap.add_argument('--end', default='2026-12-31')
    ap.add_argument('--risk', type=float, default=200.0)
    ap.add_argument('--stop-buffer', type=float, default=2.0)
    ap.add_argument('--risk-min', type=float, default=180.0)
    ap.add_argument('--risk-max', type=float, default=220.0)
    ap.add_argument('--slippage-ticks', type=int, default=1)
    ap.add_argument('--commission-side', type=float, default=1.25)
    ap.add_argument('--max-hold-days', type=int, default=15)
    ap.add_argument('--max-sweep-points', type=float, default=40.0)
    ap.add_argument('--output', default='results/mnq_asia_sweep40_lockout_3rr')
    args = ap.parse_args()
    if (args.risk <= 0 or args.stop_buffer <= 0 or args.slippage_ticks < 0 or args.commission_side < 0
            or args.risk_min <= 0 or args.risk_min > args.risk or args.risk > args.risk_max
            or args.max_sweep_points <= 0
            ):
        ap.error('Invalid risk, buffer, slippage or commission')
    print('Reading NQ candles for MNQ simulation...', flush=True)
    df = load_data(args.data)
    print(f'Loaded {len(df):,} bars; building 15m/30m/1h FVG events...', flush=True)
    events = htf_events(df)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_trades, all_summaries, diagnostics = [], [], []
    for rr in (1.0, 1.5, 2.0):
        print(f'Running {rr:g}R scenario...', flush=True)
        trades, stats = simulate(df, start_date=args.start, end_date=args.end,
                                 rr=rr, events=events, risk_budget=args.risk,
                                 stop_buffer=args.stop_buffer, slip_ticks=args.slippage_ticks,
                                 commission_side=args.commission_side, max_trade_days=args.max_hold_days,
                                 risk_min=args.risk_min, risk_max=args.risk_max,
                                 max_sweep_points=args.max_sweep_points)
        summary = summarize(trades)
        trades.insert(0, 'scenario_rr', rr)
        summary.insert(0, 'scenario_rr', rr)
        all_trades.append(trades)
        all_summaries.append(summary)
        diagnostics.append({'scenario_rr': rr, **stats})
        trades.to_csv(out / f'trades_{rr:g}R.csv', index=False)
        print(summary.to_string(index=False) if not summary.empty else 'No completed trades.', flush=True)
    pd.concat(all_trades, ignore_index=True).to_csv(out / 'trades_all_rr.csv', index=False)
    pd.concat(all_summaries, ignore_index=True).to_csv(out / 'summary_all_rr.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out / 'diagnostics.csv', index=False)
    print('Saved results to', out, flush=True)
    print('Different RR targets can change subsequent entries (one position per scenario).')


if __name__ == '__main__':
    main()
