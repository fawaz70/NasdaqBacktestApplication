"""Experimental NQ/MNQ 1m FVG retracement + CISD backtest.

Explicit operational definitions (adjust after chart review):
* No persistent bias. Setup's breakout candle must CLOSE beyond the
  previous NY calendar-day high (long) or low (short), plus prior 5m range.
  Day resets at midnight New York time.
* Breakout starts on a 1m CLOSE through the current day's prior HOD/LOD.
  Collect consecutive directional FVGs throughout the breakout leg, ending
  on the first opposite-color candle. Choose the middle FVG by creation order
  (for even counts, choose the earlier of the two middle FVGs).
  Do not require a gap on the exact HOD/LOD break candle.
* Retracement = subsequent candle touches the 1m FVG zone.
* CISD proxy = after retracement, a 1m candle CLOSES beyond the OPEN of the
  FIRST opposing-color 1m candle seen since retracement. The reference stays fixed throughout that pullback. This is a testable proxy, not
  a universally agreed definition of CISD.
* Stop = 2-point buffer past first of 3 FVG candles only; skip if 1 MNQ
  would exceed $220 planned risk. No OTE filter or annotation.
* Flag if retracement from FVG touch through pre-CISD bars overlaps a
  fully confirmed 5m or 15m FVG, irrespective of direction.
* Entry NY 09:35-11:00 or 20:00-22:00 America/New_York.
* Max 2 entries per session date, separately for each RR scenario.
* 1/1.5/2 R independently; one position at a time per scenario.
* NQ data, MNQ $2/point, target $200 planned risk, cap $220.
* Slippage one tick each side; commissions $1.25/side/contract.
* Same-bar stop and target: stop first. End-of-data positions unresolved.\n* 1.5R/2R: paired shadow BE simulation at +1R, with original stop\n  outcome preserved to classify saved losses vs cut winners. Stop-first\n  intrabar assumption; BE exit incurs slippage and commissions.
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd

TZ='America/New_York'
POINT_VALUE=2.
TICK=.25

def load(path):
    d=pd.read_csv(path, usecols=['ts_event','instrument_id','open','high','low','close'])
    d['ts_event']=pd.to_datetime(d.ts_event, utc=True).dt.tz_convert(TZ)
    return d.sort_values('ts_event').drop_duplicates('ts_event').set_index('ts_event')

def htf_events(d):
    ev=defaultdict(list)
    for mins in (5,15):
        # Higher-timeframe bars anchored to midnight NY.
        b=d.resample(f'{mins}min',origin='start_day',closed='left',label='left').agg(
            {'open':'first','high':'max','low':'min','close':'last',
             'instrument_id':['first','nunique','count']})
        b.columns=['open','high','low','close','cid','nunique','count']
        valid=(b['count']==mins)&(b['nunique']==1)
        for i in range(2,len(b)):
            if not all(valid.iloc[j] for j in (i-2,i-1,i)): continue
            if len({b.cid.iloc[j] for j in (i-2,i-1,i)})!=1: continue
            a,c=b.iloc[i-2],b.iloc[i]
            if c.low>a.high: direction=1; bottom=a.high; top=c.low
            elif c.high<a.low: direction=-1; bottom=c.high; top=a.low
            else: continue
            ev[b.index[i]+pd.Timedelta(minutes=mins)].append(
                (mins,direction,float(bottom),float(top),int(c.cid)))
    return ev

def summary(trades):
    x=pd.DataFrame(trades)
    out=[]
    for (rr,session),g in x.groupby(['rr','session'],sort=False):
        completed=g[g.result.isin(['win','loss','flat'])].sort_values('exit_time')
        wins=completed[completed.net_usd>0]
        losses=completed[completed.net_usd<0]
        pnl=float(completed.net_usd.sum())
        eq=completed.net_usd.cumsum()
        dd=(eq-eq.cummax().clip(lower=0)).min() if len(eq) else 0
        out.append(dict(rr=rr,session=session,completed=len(completed),
            wins=len(wins),losses=len(losses),
            win_rate_pct=round(100*len(wins)/len(completed),2) if len(completed) else 0,
            net_usd=round(pnl,2),max_drawdown_usd=round(float(dd),2),
            profit_factor=round(float(wins.net_usd.sum()/abs(losses.net_usd.sum())),3)
              if len(losses) else np.nan,
            tap_5m=int(completed.tapped_5m_fvg.sum()) if len(completed) else 0,
            tap_15m=int(completed.tapped_15m_fvg.sum()) if len(completed) else 0))
    return pd.DataFrame(out)

def run(d, events, args):
    alltrades=[]
    for rr in (1.,1.5,2.):
        hod=None
        lod=None
        day=None
        active_htf=[]
        session_counts=defaultdict(int)
        prev_t=None
        prev_cid=None
        recent=deque(maxlen=max(20,args.breakout_lookback+4))
        pending=None
        breakout=None
        trade=None
        diagnostics=defaultdict(int)
        def finish(t,price,reason):
            nonlocal trade
            direction=trade['direction']
            gross=(price-trade['entry'])*direction*POINT_VALUE*trade['contracts']
            fees=2*args.commission_side*trade['contracts']
            net=round(gross-fees,2)
            trade.update(exit_time=t.isoformat(),exit_price=price,net_usd=net,
                         result='win' if net>0 else 'loss' if net<0 else 'flat',
                         exit_reason=reason)
            alltrades.append(trade)
            trade=None
        for r in d.itertuples():
            t=r.Index
            end=t+pd.Timedelta(minutes=1)
            cid=int(r.instrument_id)
            cont=prev_t is not None and t-prev_t==pd.Timedelta(minutes=1) and cid==prev_cid
            if not cont:
                recent.clear()
                pending=None
                breakout=None
                hod=None
                lod=None
                active_htf=[]
                if trade is not None:
                    trade.update(result='unresolved',exit_reason='DATA_GAP_OR_ROLLOVER',
                                 exit_time='',exit_price=np.nan,net_usd=np.nan)
                    alltrades.append(trade)
                    trade=None
            prev_t=t
            prev_cid=cid
            if day!=t.date():
                day=t.date()
                hod=None
                lod=None
                pending=None
                breakout=None
            # Confirmed HTF FVGs become visible at their bar close.
            for mins, direction, bottom, top, gcid in events.get(t, []):
                if gcid == cid:
                    active_htf.append((mins, direction, bottom, top, t.date(), t.hour*60+t.minute))
            # Detect breakout against the HOD/LOD from BEFORE this minute.
            broke_high = hod is not None and r.close > hod
            broke_low = lod is not None and r.close < lod
            hod=max(hod,r.high) if hod is not None else float(r.high)
            lod=min(lod,r.low) if lod is not None else float(r.low)
            if trade is not None:
                # No stop/target evaluation on entry candle.
                if t>trade['entry_bar']:
                    if trade['direction']==1:
                        if r.low<=trade['stop']:
                            finish(end,trade['stop']-args.slippage_ticks*TICK,'stop')
                        elif r.high>=trade['target']:
                            finish(end,trade['target']-args.slippage_ticks*TICK,'target')
                    else:
                        if r.high>=trade['stop']:
                            finish(end,trade['stop']+args.slippage_ticks*TICK,'stop')
                        elif r.low<=trade['target']:
                            finish(end,trade['target']+args.slippage_ticks*TICK,'target')
            # Collect a multi-candle breakout leg. Its first opposite-color
            # candle ends the impulse and locks the middle FVG for retracement.
            # The first opposing candle can itself touch that chosen FVG.
            prior=list(recent)
            if breakout is not None:
                b=breakout
                b['impulse_low']=min(b['impulse_low'],float(r.low))
                b['impulse_high']=max(b['impulse_high'],float(r.high))
                if (end-b['start']).total_seconds()/60>args.breakout_max_minutes:
                    diagnostics['breakout_expired']+=1
                    breakout=None
                else:
                    if len(prior)>=2:
                        first=prior[-2]  # FVG candle 1; candle 3 is r
                        if b['direction']==1 and r.low>first['high']:
                            b['gaps'].append(dict(direction=1,formed=end,
                                bottom=float(first['high']),top=float(r.low),
                                first_low=float(first['low']),first_high=float(first['high'])))
                        elif b['direction']==-1 and r.high<first['low']:
                            b['gaps'].append(dict(direction=-1,formed=end,
                                bottom=float(r.high),top=float(first['low']),
                                first_low=float(first['low']),first_high=float(first['high'])))
                    opposing=(r.close<r.open if b['direction']==1 else r.close>r.open)
                    if opposing:
                        diagnostics['breakout_legs']+=1
                        diagnostics[f'legs_with_{min(len(b["gaps"]),5)}_plus_fvgs']+=1
                        if b['gaps']:
                            idx=(len(b['gaps'])-1)//2
                            pending=dict(b['gaps'][idx], phase='wait_retrace',
                                fvg_count=len(b['gaps']),fvg_selected_index=idx+1,
                                breakout_start=b['start'],
                                impulse_low=b['impulse_low'],impulse_high=b['impulse_high'])
                            diagnostics['selected_fvgs']+=1
                        breakout=None
            if pending and (end-pending['formed']).total_seconds()/60>args.setup_expiry:
                diagnostics['setup_expired']+=1
                pending=None
            # Setup progresses on subsequent bars (not FVG creation bar).
            if pending is not None and end>=pending['formed']:
                p=pending
                if p['phase']=='wait_retrace':
                    touch=(r.low<=p['top'] and r.high>=p['bottom'])
                    if touch:
                        p['phase']='wait_cisd'
                        p['retrace_time']=end
                        p['tapped_5m_fvg']=False
                        p['tapped_15m_fvg']=False
                        p['opposing_open']=(float(r.open) if
                            ((p['direction']==1 and r.close<r.open) or
                             (p['direction']==-1 and r.close>r.open)) else None)
                        p['pullback_low']=float(r.low)
                        p['pullback_high']=float(r.high)
                        for tf, gap_dir, bottom, top, formed_day, formed_min in active_htf:
                            session_start=570 if 570<=t.hour*60+t.minute<660 else 1200 if 1200<=t.hour*60+t.minute<1320 else None
                            if session_start is not None and formed_day==t.date() and formed_min>=session_start and gap_dir==p['direction'] and r.low<=top and r.high>=bottom:
                                p[f'tapped_{tf}m_fvg']=True
                elif p['phase']=='wait_cisd':
                    p['pullback_low']=min(p['pullback_low'],float(r.low))
                    p['pullback_high']=max(p['pullback_high'],float(r.high))
                    for tf, gap_dir, bottom, top, formed_day, formed_min in active_htf:
                        session_start=570 if 570<=t.hour*60+t.minute<660 else 1200 if 1200<=t.hour*60+t.minute<1320 else None
                        if session_start is not None and formed_day==t.date() and formed_min>=session_start and gap_dir==p['direction'] and r.low<=top and r.high>=bottom:
                            p[f'tapped_{tf}m_fvg']=True
                    # Require a fresh opposing candle AFTER retrace, then displacement close.
                    if p['opposing_open'] is None and ((p['direction']==1 and r.close<r.open) or (p['direction']==-1 and r.close>r.open)):
                        p['opposing_open']=float(r.open)
                    elif p['opposing_open'] is not None:
                        confirmed=(r.close>p['opposing_open'] if p['direction']==1
                                   else r.close<p['opposing_open'])
                        session=('NY' if 9*60+35<=t.hour*60+t.minute<11*60
                                 else 'ASIA' if 20*60<=t.hour*60+t.minute<22*60 else None)
                        # Asian session 20:00-22:00 is keyed to its starting date.
                        session_key=(session, t.date())
                        if confirmed and session and trade is None and session_counts[session_key]<args.max_trades_per_session:
                            direction=p['direction']
                            entry=float(r.close)+direction*args.slippage_ticks*TICK
                            stop=(p['first_low']-args.stop_buffer if direction==1
                                  else p['first_high']+args.stop_buffer)
                            dist=direction*(entry-stop)
                            per=dist*POINT_VALUE+2*args.commission_side
                            if dist>0 and per<=args.risk_max:
                                n=max(1,int(args.risk_max//per))
                                n=min(range(1,n+1),key=lambda k:(abs(k*per-args.risk),k))
                                target=entry+direction*rr*dist
                                session_counts[session_key]+=1
                                # Full breakout impulse: initial FVG's first-candle
                                # extreme through the highest/lowest breakout-leg extreme.
                                impulse_range=p['impulse_high']-p['impulse_low']
                                if direction==1:
                                    retracement_fraction=(p['impulse_high']-p['pullback_low'])/impulse_range if impulse_range>0 else np.nan
                                else:
                                    retracement_fraction=(p['pullback_high']-p['impulse_low'])/impulse_range if impulse_range>0 else np.nan
                                trade=dict(rr=rr,session=session,
                                    entry_time=end.isoformat(),entry_bar=t,
                                    direction=direction,entry=entry,stop=stop,target=target,
                                    contracts=n,planned_risk_usd=round(n*per,2),
                                    stop_type='fvg_first_candle',
                                    breakout_impulse_low=p['impulse_low'],
                                    breakout_impulse_high=p['impulse_high'],
                                    breakout_retrace_fraction=round(float(retracement_fraction),4),
                                    reached_50pct_breakout=bool(retracement_fraction>=0.5),
                                    fvg_count_in_breakout=p['fvg_count'],
                                    selected_fvg_number=p['fvg_selected_index'],
                                    breakout_hod_lod='HOD' if direction==1 else 'LOD',
                                    tapped_5m_fvg=p['tapped_5m_fvg'],
                                    tapped_15m_fvg=p['tapped_15m_fvg'],
                                    fvg_bottom=p['bottom'],fvg_top=p['top'],
                                    fvg_formed=p['formed'].isoformat(),
                                    retrace_time=p['retrace_time'].isoformat(),
                                    first_fvg_candle_low=p['first_low'],first_fvg_candle_high=p['first_high'])
                            pending=None
            # Initiate a leg when an actual new daily extreme closes through.
            # Subsequent FVGs belong to that leg until the first counter candle.
            if breakout is None and pending is None and len(recent)>=2:
                direction=1 if broke_high else -1 if broke_low else 0
                if direction:
                    breakout=dict(direction=direction,start=end,gaps=[],impulse_low=float(r.low),impulse_high=float(r.high))
                    diagnostics['hod_lod_breakouts']+=1
                    # Include the gap printed by the breakout candle itself.
                    a=prior[-2]
                    if direction==1 and r.low>a['high']:
                        breakout['gaps'].append(dict(direction=direction,formed=end,
                            bottom=float(a['high']),top=float(r.low),
                            first_low=float(a['low']),first_high=float(a['high'])))
                    elif direction==-1 and r.high<a['low']:
                        breakout['gaps'].append(dict(direction=direction,formed=end,
                            bottom=float(r.high),top=float(a['low']),
                            first_low=float(a['low']),first_high=float(a['high'])))
            recent.append(dict(high=float(r.high),low=float(r.low),open=float(r.open),close=float(r.close)))
        if trade is not None:
            trade.update(result='unresolved',exit_reason='END_OF_DATA',exit_time='',
                         exit_price=np.nan,net_usd=np.nan)
            alltrades.append(trade)
        alltrades.append(dict(_diagnostic=True,rr=rr,diagnostics=dict(diagnostics)))
    # Counterfactual BE simulation on the identical entry (1.5R / 2R).
    # Intrabar order is unknown: conservatively process stop before 1R trigger.
    # If BE was armed before a candle, process BE stop before target.
    index_by_time={t:i for i,t in enumerate(d.index)}
    rows=list(d.itertuples())
    for tr in alltrades:
        if tr.get('_diagnostic') or tr.get('rr')==1.:
            continue
        tr['be_activated']=False
        tr['be_activation_time']=''
        tr['be_exit_reason']=''
        tr['be_exit_time']=''
        tr['be_net_usd']=np.nan
        tr['be_would_original_win']=False
        tr['be_would_original_lose']=False
        tr['be_valid_saved_loss']=False
        tr['be_cost_winner']=False
        start=index_by_time.get(tr['entry_bar'])
        if start is None: continue
        side=tr['direction']
        risk=side*(tr['entry']-tr['stop'])
        trigger=tr['entry']+side*risk
        armed=False
        for j in range(start+1,len(rows)):
            r=rows[j]
            prev=rows[j-1]
            if (r.Index-prev.Index!=pd.Timedelta(minutes=1)
                    or r.instrument_id!=prev.instrument_id):
                tr['be_exit_reason']='unresolved_gap'
                break
            # Execution price includes one tick of exit slippage.
            stop_price=tr['entry'] if armed else tr['stop']
            hit_stop=(r.low<=stop_price if side==1 else r.high>=stop_price)
            hit_target=(r.high>=tr['target'] if side==1 else r.low<=tr['target'])
            if hit_stop:
                px=stop_price-side*args.slippage_ticks*TICK
                tr['be_exit_reason']='breakeven' if armed else 'stop'
            elif hit_target:
                px=tr['target']-side*args.slippage_ticks*TICK
                tr['be_exit_reason']='target'
            elif not armed and (r.high>=trigger if side==1 else r.low<=trigger):
                armed=True
                tr['be_activated']=True
                tr['be_activation_time']=(r.Index+pd.Timedelta(minutes=1)).isoformat()
                continue
            else:
                continue
            tr['be_exit_time']=(r.Index+pd.Timedelta(minutes=1)).isoformat()
            tr['be_net_usd']=round((px-tr['entry'])*side*POINT_VALUE*tr['contracts']
                                  -2*args.commission_side*tr['contracts'],2)
            break
        if tr['be_exit_reason']=='breakeven' and tr.get('result') in ('win','loss','flat'):
            tr['be_would_original_win']=tr['net_usd']>0
            tr['be_would_original_lose']=tr['net_usd']<0
            tr['be_valid_saved_loss']=tr['net_usd']<0
            tr['be_cost_winner']=tr['net_usd']>0
    diagnostics_rows=[x for x in alltrades if x.get('_diagnostic')]
    run.last_diagnostics=diagnostics_rows
    alltrades=[x for x in alltrades if not x.get('_diagnostic')]
    return pd.DataFrame(alltrades,columns=['rr','session','result','net_usd','exit_time','entry_bar',
        'tapped_5m_fvg','tapped_15m_fvg'] if not alltrades else None)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/NQ_2026_full.csv')
    ap.add_argument('--output',default='results/mnq_hod_lod_middle_fvg_2trades')
    ap.add_argument('--risk',type=float,default=200)
    ap.add_argument('--risk-max',type=float,default=220)
    ap.add_argument('--stop-buffer',type=float,default=2)
    ap.add_argument('--slippage-ticks',type=int,default=1)
    ap.add_argument('--commission-side',type=float,default=1.25)
    ap.add_argument('--breakout-lookback',type=int,default=5)
    ap.add_argument('--setup-expiry',type=int,default=45)
    ap.add_argument('--breakout-max-minutes',type=int,default=30)
    ap.add_argument('--max-trades-per-session',type=int,default=2)
    args=ap.parse_args()
    d=load(args.data)
    trades=run(d,htf_events(d),args)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for rr in (1.,1.5,2.):
        label=f'{rr:g}R'
        trades[trades.rr==rr].drop(columns=['entry_bar']).to_csv(out/f'trades_{label}.csv',index=False)
    trades.drop(columns=['entry_bar']).to_csv(out/'trades_all_rr.csv',index=False)
    diag=[]
    for entry in run.last_diagnostics:
        for k,v in entry['diagnostics'].items():
            diag.append(dict(rr=entry['rr'],event=k,count=v))
    pd.DataFrame(diag).to_csv(out/'diagnostics.csv',index=False)
    s=summary(trades)
    s.to_csv(out/'summary.csv',index=False)
    completed=trades[trades.result.isin(['win','loss','flat'])]
    if not completed.empty:
        taps=(completed.groupby(['rr','session','tapped_5m_fvg','tapped_15m_fvg'])
              .agg(trades=('net_usd','size'),wins=('net_usd',lambda x:(x>0).sum()),
                   net_usd=('net_usd','sum')).reset_index())
        taps['win_rate_pct']=(100*taps.wins/taps.trades).round(2)
        taps.to_csv(out/'fvg_tap_analysis.csv',index=False)
        print('\n5m/15m FVG retracement touch breakdown:')
        print(taps.to_string(index=False))
    if not completed.empty:
        halves=(completed.groupby(['rr','session','reached_50pct_breakout'])
              .agg(trades=('net_usd','size'),wins=('net_usd',lambda x:(x>0).sum()),
                   net_usd=('net_usd','sum')).reset_index())
        halves['win_rate_pct']=(100*halves.wins/halves.trades).round(2)
        halves.to_csv(out/'breakout_50pct_analysis.csv',index=False)
        print('\n50% full-breakout retracement breakdown:')
        print(halves.to_string(index=False))
    # Paired original-stop vs BE outcomes for the same trades.
    pairs=trades[(trades.rr>1)&trades.net_usd.notna()&trades.be_net_usd.notna()].copy()
    if not pairs.empty:
        pairs['pnl_difference_be_minus_original']=pairs.be_net_usd-pairs.net_usd
        pairs['original_result']=pairs.result
        pairs['be_result']=np.where(pairs.be_net_usd>0,'win',
                             np.where(pairs.be_net_usd<0,'loss','flat'))
        pairs.drop(columns=['entry_bar']).to_csv(out/'be_trade_comparison.csv',index=False)
        bsum=pairs.groupby(['rr','session']).agg(
            trades=('net_usd','size'),
            original_pnl=('net_usd','sum'),
            be_pnl=('be_net_usd','sum'),
            be_triggered=('be_activated','sum'),
            be_exits=('be_exit_reason',lambda x:(x=='breakeven').sum()),
            be_saved_losses=('be_valid_saved_loss','sum'),
            be_cut_winners=('be_cost_winner','sum'),
            pnl_change=('pnl_difference_be_minus_original','sum')).reset_index()
        bsum.to_csv(out/'be_summary.csv',index=False)
        print('\\nBE paired counterfactual results:')
        print(bsum.to_string(index=False))
    print(s.to_string(index=False))
    print('Files:',out.resolve())
if __name__=='__main__': main()
