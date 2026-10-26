"""HOD/LOD breakout -> middle 1-minute FVG -> retracement touch entry with candle-3 extreme close guard (no CISD).

Entry: bar must touch selected FVG and CLOSE without inverting it; configurable close range; third FVG
candle extreme in the breakout direction; after first opposing-color candle.
Target: nearest confirmed swing extreme, prior NY cash-session H/L, or Asia H/L
AHEAD of entry, excluding the FVG's first-candle swing. If unavailable or closer
than 1.5R, extend target to 1.5R.
Stop: 2 NQ pts beyond first candle of selected FVG. Up to two trades/session.
NY 09:40-10:30 ET; Asia 20:30-22:00 ET.
Default BE moves stop to entry after +1.25R, effective NEXT 1-minute bar;
paired no-BE counterfactual also reported for same entries. One tick of slip each
side, $1.25 commission/side/MNQ, ~$200 planned risk cap $220.
Swing pivots confirmed by left/right completed 1m bars; no lookahead.
1-minute same-bar stop/target ambiguity handled conservatively (stop first).
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
    for session,g in x.groupby('session',sort=False):
        rr=float(g.rr.mean())
        completed=g[g.result.isin(['win','loss','flat'])].sort_values('exit_time')
        wins=completed[completed.net_usd>0]
        losses=completed[completed.net_usd<0]
        pnl=float(completed.net_usd.sum())
        eq=completed.net_usd.cumsum()
        dd=(eq-eq.cummax().clip(lower=0)).min() if len(eq) else 0
        out.append(dict(rr=round(rr,3),session=session,completed=len(completed),
            wins=len(wins),losses=len(losses),
            win_rate_pct=round(100*len(wins)/len(completed),2) if len(completed) else 0,
            net_usd=round(pnl,2),max_drawdown_usd=round(float(dd),2),
            profit_factor=round(float(wins.net_usd.sum()/abs(losses.net_usd.sum())),3)
              if len(losses) else np.nan,
            tap_5m=int(completed.tapped_5m_fvg.sum()) if len(completed) else 0,
            tap_15m=int(completed.tapped_15m_fvg.sum()) if len(completed) else 0))
    return pd.DataFrame(out)

def prior_ny_session_levels(d):
    """Map each NY-local date to previous available 09:30–16:00 ET cash-session H/L."""
    ny=d.between_time('09:30','15:59')
    daily=ny.groupby(ny.index.date).agg({'high':'max','low':'min','instrument_id':'nunique'})
    daily=daily[daily.instrument_id==1]
    dates=list(daily.index)
    lookup={}
    for day in sorted(set(d.index.date)):
        prev=[z for z in dates if z<day]
        if prev:
            row=daily.loc[prev[-1]]
            lookup[day]=(float(row.high),float(row.low))
    return lookup

def run(d, events, args):
    alltrades=[]
    prior_ny=prior_ny_session_levels(d)
    for rr in (1.5,):
        hod=None
        lod=None
        day=None
        active_htf=[]
        session_counts=defaultdict(int)
        session_first_win=defaultdict(bool)
        prev_t=None
        prev_cid=None
        recent=deque(maxlen=max(60,args.swing_lookback+args.pivot_left+args.pivot_right+5))
        pivots_high=deque(maxlen=1000)
        pivots_low=deque(maxlen=1000)
        prev_day_levels=None
        running_day_high=-np.inf
        running_day_low=np.inf
        asia_hi=-np.inf
        asia_lo=np.inf
        asia_date=None
        last_asia=None
        pending=None
        breakout=None
        trade=None
        diagnostics=defaultdict(int)
        def finish(t,price,reason):
            nonlocal trade
            direction=trade['direction']
            gross=(price-trade['entry'])*direction*POINT_VALUE*trade['remaining_contracts']
            fees=2*args.commission_side*trade['remaining_contracts']
            net=round(gross-fees+trade['partial_net_usd'],2)
            trade.update(exit_time=t.isoformat(),exit_price=price,net_usd=net,
                         full_exit_reason=reason,
                         result='win' if net>0 else 'loss' if net<0 else 'flat',
                         exit_reason=reason)
            if args.stop_after_first_win and session_counts[(trade['session'],trade['entry_bar'].date())]==1 and net>0:
                session_first_win[(trade['session'],trade['entry_bar'].date())]=True
            alltrades.append(trade)
            trade=None
        for r in d.itertuples():
            t=r.Index
            end=t+pd.Timedelta(minutes=1)
            cid=int(r.instrument_id)
            cont=prev_t is not None and t-prev_t==pd.Timedelta(minutes=1) and cid==prev_cid
            if not cont:
                recent.clear()
                pivots_high.clear();pivots_low.clear()
                prev_day_levels=None;last_asia=None;asia_date=None
                running_day_high=-np.inf;running_day_low=np.inf
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
                prev_day_levels=prior_ny.get(t.date())
                running_day_high=-np.inf;running_day_low=np.inf
                day=t.date()
                prev_day_levels=prior_ny.get(t.date())
                hod=None
                lod=None
                pending=None
                breakout=None
            minute=t.hour*60+t.minute
            # Capture Asia's 20:00-midnight highs/lows before current candle;
            # yesterday's completed Asian range is the reference during NY.
            if 1200<=minute<1440:
                if asia_date!=t.date():
                    asia_date=t.date();asia_hi=-np.inf;asia_lo=np.inf
                asia_levels=(asia_hi,asia_lo) if np.isfinite(asia_hi) else last_asia
            else:
                if asia_date is not None:
                    if np.isfinite(asia_hi): last_asia=(asia_hi,asia_lo)
                    asia_date=None
                asia_levels=last_asia
            # Confirm a pivot only once the right-side bars already closed.
            if len(recent)>=args.pivot_left+args.pivot_right+1:
                hist=list(recent)
                j=len(hist)-1-args.pivot_right
                candidate=hist[j]
                neighbors=hist[j-args.pivot_left:j]+hist[j+1:j+1+args.pivot_right]
                if len(neighbors)==args.pivot_left+args.pivot_right:
                    if all(candidate['high']>z['high'] for z in neighbors):
                        pivots_high.append((candidate['t'],candidate['high']))
                    if all(candidate['low']<z['low'] for z in neighbors):
                        pivots_low.append((candidate['t'],candidate['low']))
            while pivots_high and (t-pivots_high[0][0]).total_seconds()/60>args.swing_lookback:
                pivots_high.popleft()
            while pivots_low and (t-pivots_low[0][0]).total_seconds()/60>args.swing_lookback:
                pivots_low.popleft()
            # Record newly confirmed swing pivot for MSS; only completed bars.
            if trade is not None and len(recent)>=args.pivot_left+args.pivot_right+1:
                hist=list(recent)
                j=len(hist)-1-args.pivot_right
                candidate=hist[j]
                neighbors=hist[j-args.pivot_left:j]+hist[j+1:j+1+args.pivot_right]
                if (candidate['t']>trade['entry_bar'] and len(neighbors)==args.pivot_left+args.pivot_right):
                    if trade['direction']==1 and all(candidate['high']>z['high'] for z in neighbors):
                        trade['mss_swing_level']=candidate['high']
                        trade['mss_swing_pivot_time']=candidate['t']
                    elif trade['direction']==-1 and all(candidate['low']<z['low'] for z in neighbors):
                        trade['mss_swing_level']=candidate['low']
                        trade['mss_swing_pivot_time']=candidate['t']
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
                    side=trade['direction']
                    live_stop=trade['entry'] if trade['be_armed'] else trade['stop']
                    hit_stop=(r.low<=live_stop if side==1 else r.high>=live_stop)
                    hit_target=(r.high>=trade['target'] if side==1 else r.low<=trade['target'])
                    # Conservative: if stop and target touch in same minute, stop wins.
                    # Conservative bar ordering: stop takes priority over the partial.
                    # If target and +1.25R both hit, partial is executed first
                    # (on the way to the >=1.5R target), then remaining contracts exit.
                    if (not hit_stop and args.partial_at_rr and not trade['partial_taken']
                            and trade['contracts']>1 and
                            (r.high>=trade['entry']+side*args.partial_rr*trade['initial_risk_points']
                             if side==1 else r.low<=trade['entry']+side*args.partial_rr*trade['initial_risk_points'])):
                        k=trade['contracts']//2
                        px=trade['entry']+side*args.partial_rr*trade['initial_risk_points']-side*args.slippage_ticks*TICK
                        trade['partial_net_usd']=round((px-trade['entry'])*side*POINT_VALUE*k-2*args.commission_side*k,2)
                        trade['partial_contracts']=k
                        trade['remaining_contracts']-=k
                        trade['partial_taken']=True
                        trade['partial_time']=end.isoformat()
                        if not trade['be_armed']:
                            trade['be_armed']=True
                            trade['be_activation_time']=end.isoformat()
                    if hit_stop:
                        finish(end,live_stop-side*args.slippage_ticks*TICK,
                               'breakeven' if trade['be_armed'] else 'stop')
                    elif hit_target:
                        finish(end,trade['target']-side*args.slippage_ticks*TICK,'target')
                    elif args.be_mode!='none' and not trade['be_armed']:
                        # Opposing-color candle is the first one AFTER entry;
                        # breakout must CLOSE beyond its extreme on a LATER bar.
                        if trade['opposing_level'] is None:
                            opposing=(r.close<r.open if side==1 else r.close>r.open)
                            if opposing:
                                trade['opposing_level']=float(r.high if side==1 else r.low)
                                trade['opposing_time']=t
                        swing=trade['mss_swing_level']
                        opp=trade['opposing_level']
                        if args.be_mode=='rr':
                            trigger=trade['entry']+side*args.be_trigger_rr*trade['initial_risk_points']
                            confirmed=(r.high>=trigger if side==1 else r.low<=trigger)
                        elif args.be_mode=='swing':
                            confirmed=(swing is not None and side*(float(r.close)-swing)>0)
                        elif args.be_mode=='opposing':
                            confirmed=(opp is not None and t>trade['opposing_time'] and side*(float(r.close)-opp)>0)
                        elif args.be_mode in ('opposing_or_rr','opposing_and_rr'):
                            trigger=trade['entry']+side*args.be_trigger_rr*trade['initial_risk_points']
                            rr_hit=(r.high>=trigger if side==1 else r.low<=trigger)
                            opp_hit=(opp is not None and t>trade['opposing_time'] and side*(float(r.close)-opp)>0)
                            # Track either condition across bars, rather than requiring
                            # both conditions to occur on the same minute.
                            if rr_hit: trade['rr_seen']=True
                            if opp_hit: trade['opposing_seen']=True
                            if args.be_mode=='opposing_or_rr':
                                confirmed=trade['rr_seen'] or trade['opposing_seen']
                            else:
                                confirmed=trade['rr_seen'] and trade['opposing_seen']
                        elif args.be_mode=='cisd_wick':
                            # First opposing candle is the structural reference. A later wick
                            # through its high/low triggers BE, regardless of close.
                            confirmed=(opp is not None and t>trade['opposing_time'] and
                                       (float(r.high)>opp if side==1 else float(r.low)<opp))
                        elif args.be_mode=='swing_1r':
                            confirmed=(swing is not None and side*(float(r.close)-swing)>0
                                and side*(float(r.close)-trade['entry'])>=trade['initial_risk_points'])
                        else: raise ValueError(args.be_mode)
                        if confirmed:
                            trade['be_armed']=True
                            trade['be_activation_time']=end.isoformat()
                            trade['be_trigger_level']=swing if args.be_mode in ('swing','swing_1r') else opp if args.be_mode=='opposing' else trigger if args.be_mode=='rr' else float(r.close)
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
                                first_low=float(first['low']),first_high=float(first['high']),
                                third_low=float(r.low),third_high=float(r.high)))
                        elif b['direction']==-1 and r.high<first['low']:
                            b['gaps'].append(dict(direction=-1,formed=end,
                                bottom=float(r.high),top=float(first['low']),
                                first_low=float(first['low']),first_high=float(first['high']),
                                third_low=float(r.low),third_high=float(r.high)))
                    opposing=(r.close<r.open if b['direction']==1 else r.close>r.open)
                    if opposing:
                        diagnostics['breakout_legs']+=1
                        diagnostics[f'legs_with_{min(len(b["gaps"]),5)}_plus_fvgs']+=1
                        if b['gaps']:
                            idx=(len(b['gaps'])-1)//2
                            pending=dict(b['gaps'][idx], phase='wait_retrace',
                                fvg_count=len(b['gaps']),fvg_selected_index=idx+1,
                                breakout_start=b['start'],
                                impulse_low=b['impulse_low'],impulse_high=b['impulse_high'],
                                retrace_tap5=False,retrace_tap15=False)
                            diagnostics['selected_fvgs']+=1
                        breakout=None
            # A NEW HOD/LOD close while awaiting a retest supersedes the old
            # setup. FVGs created during an ACTIVE breakout remain grouped.
            if pending is not None and (broke_high or broke_low):
                diagnostics['pending_cancelled_new_breakout']+=1
                pending=None
            if pending and args.fvg_age_max is not None and (end-pending['formed']).total_seconds()/60>args.fvg_age_max:
                diagnostics['setup_expired']+=1
                pending=None
            # FVG retracement entry: no CISD requirement.
            if pending is not None and end>pending['formed']:
                p=pending
                age=(end-p['formed']).total_seconds()/60
                # Track 5m/15m taps during the whole retracement, not just
                # the final entry candle. Same-direction, session-formed FVGs.
                for tf,gap_dir,bottom,top,formed_day,formed_min in active_htf:
                    session_start=570 if 570<=minute<960 else 1200
                    if (formed_day==t.date() and formed_min>=session_start
                            and gap_dir==p['direction']
                            and r.low<=top and r.high>=bottom):
                        if tf==5: p['retrace_tap5']=True
                        if tf==15: p['retrace_tap15']=True
                direction=p['direction']
                # A close THROUGH the opposite FVG edge invalidates the
                # original gap permanently; do not allow a later reclaim to
                # be traded as if the original FVG had held.
                if (direction==1 and float(r.close)<p['bottom']) or (direction==-1 and float(r.close)>p['top']):
                    diagnostics['pending_invalidated_ifvg_close']+=1
                    pending=None
                    # Keep the candle in the ongoing breakout-state updates.
                    # No retracement entry is eligible from this candle.
                    touched=False
                    allowed=False
                else:
                    touched=(r.low<=p['top'] and r.high>=p['bottom'])
                    if args.entry_mode=='inside':
                        allowed=(p['bottom']<=float(r.close)<=p['top'])
                    elif args.entry_mode=='near':
                        allowed=(p['bottom']<=float(r.close)<=p['top']+args.close_above_points) if direction==1 else (p['bottom']-args.close_above_points<=float(r.close)<=p['top'])
                    elif args.entry_mode=='expanded':
                        allowed=(p['bottom']<=float(r.close)<=p['third_high']) if direction==1 else (p['third_low']<=float(r.close)<=p['top'])
                    else:
                        raise ValueError(args.entry_mode)
                # The FVG must HOLD on a closing basis: a bullish gap
                # inverts if price closes BELOW its bottom; a bearish gap
                # inverts if price closes ABOVE its top. Also enforce the
                # third FVG candle's outer extreme as requested.
                if touched and allowed and age<args.fvg_age_min:
                    diagnostics['retest_before_min_age']+=1
                    # This is an early retest, not an entry. Keep waiting.
                elif touched and allowed:
                    diagnostics['fvg_touch_with_allowed_close']+=1
                    session=('NY' if args.ny_start<=minute<args.ny_end
                             else 'ASIA' if args.asia_start<=minute<args.asia_end else None)
                    session_key=(session,t.date())
                    if session and trade is None and session_counts[session_key]<args.max_trades_per_session and not session_first_win[session_key]:
                        entry=float(r.close)+direction*args.slippage_ticks*TICK
                        stop=p['first_low']-args.stop_buffer if direction==1 else p['first_high']+args.stop_buffer
                        dist=direction*(entry-stop)
                        per=dist*POINT_VALUE+2*args.commission_side
                        if dist>0 and per<=args.risk_max:
                            n=max(1,int(args.risk_max//per))
                            n=min(range(1,n+1),key=lambda k:(abs(k*per-args.risk),k))
                            # All liquidity candidate levels are already confirmed BEFORE entry;
                            # exclude any pivot occurring on FVG first candle timestamp.
                            liq=[]
                            first_ts=p['formed']-pd.Timedelta(minutes=3)
                            for pivot_ts,px in (pivots_high if direction==1 else pivots_low):
                                if pivot_ts==first_ts: continue
                                if direction*(px-entry)>0:
                                    liq.append((direction*(px-entry),'swing',px))
                            for label,levels in [('previous_day',prev_day_levels),('asia_session',asia_levels)]:
                                if levels is None: continue
                                px=levels[0] if direction==1 else levels[1]
                                if np.isfinite(px) and direction*(px-entry)>0:
                                    liq.append((direction*(px-entry),label,px))
                            liq.sort(key=lambda z:z[0])
                            nearest=liq[0] if liq else None
                            if (session=='ASIA' and args.asia_fixed_target):
                                chosen=None
                            elif args.target_policy=='fixed':
                                chosen=None
                            elif args.target_policy=='nearest':
                                chosen=nearest
                            elif args.target_policy=='session_day':
                                candidate=[z for z in liq if z[1] in ('previous_day','asia_session')]
                                chosen=candidate[0] if candidate else nearest
                            elif args.target_policy=='farther_session_day':
                                candidate=[z for z in liq if z[1] in ('previous_day','asia_session')]
                                chosen=candidate[-1] if candidate else nearest
                            elif args.target_policy=='swing':
                                candidate=[z for z in liq if z[1]=='swing']
                                chosen=candidate[0] if candidate else nearest
                            else: raise ValueError(args.target_policy)
                            # Take the chosen liquidity level when it offers
                            # >=1.5R; otherwise use the fixed minimum 1.5R.
                            # No qualifying liquidity level => fixed 1.5R.
                            target_points=max(args.min_rr*dist, chosen[0] if chosen else 0.0)
                            target=entry+direction*target_points
                            rr=target_points/dist
                            session_counts[session_key]+=1
                            impulse_range=p['impulse_high']-p['impulse_low']
                            fraction=((p['impulse_high']-r.low) if direction==1 else (r.high-p['impulse_low']))/impulse_range if impulse_range>0 else np.nan
                            tap5=bool(p.get('retrace_tap5',False))
                            tap15=bool(p.get('retrace_tap15',False))
                            trade=dict(rr=round(rr,4),session=session,entry_time=end.isoformat(),entry_bar=t,
                                remaining_contracts=n,partial_contracts=0,partial_net_usd=0.0,partial_taken=False,partial_time='',
                                direction=direction,entry=entry,stop=stop,target=target,
                                initial_risk_points=dist,be_trigger_rr=args.be_trigger_rr,
                                be_armed=False,be_activation_time='',contracts=n,
                                mss_swing_level=None,mss_swing_pivot_time=None,
                                opposing_level=None,opposing_time=None,be_trigger_level=np.nan,rr_seen=False,opposing_seen=False,
                                be_mode=args.be_mode,
                                planned_risk_usd=round(n*per,2),stop_type='fvg_first_candle',
                                target_source=(chosen[1] if chosen and chosen[0]>=args.min_rr*dist else 'min_rr_extended'),
                                liquidity_level=chosen[2] if chosen else np.nan,
                                nearest_liquidity_source=nearest[1] if nearest else 'none',
                                nearest_liquidity_rr=round(nearest[0]/dist,3) if nearest else np.nan,
                                available_liquidity_count=len(liq),
                                target_policy=args.target_policy,
                                fvg_age_minutes=age,
                                trade_number=session_counts[session_key],
                                target_extended=(chosen is None or chosen[0]<args.min_rr*dist),
                                target_rr=round(rr,4),breakout_impulse_low=p['impulse_low'],
                                breakout_impulse_high=p['impulse_high'],
                                breakout_retrace_fraction=round(float(fraction),4),
                                reached_50pct_breakout=bool(fraction>=.5),
                                fvg_count_in_breakout=p['fvg_count'],selected_fvg_number=p['fvg_selected_index'],
                                breakout_hod_lod='HOD' if direction==1 else 'LOD',
                                tapped_5m_fvg=tap5,tapped_15m_fvg=tap15,
                                fvg_bottom=p['bottom'],fvg_top=p['top'],
                                fvg_formed=p['formed'].isoformat(),retrace_time=end.isoformat(),
                                first_fvg_candle_low=p['first_low'],first_fvg_candle_high=p['first_high'],
                                third_fvg_candle_low=p['third_low'],third_fvg_candle_high=p['third_high'])
                            diagnostics['entries']+=1
                        else:
                            diagnostics['risk_or_stop_rejected']+=1
                    else:
                        diagnostics['session_or_capacity_rejected']+=1
                    pending=None
                elif touched:
                    diagnostics['fvg_touch_close_outside_allowance']+=1
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
                            first_low=float(a['low']),first_high=float(a['high']),
                            third_low=float(r.low),third_high=float(r.high)))
                    elif direction==-1 and r.high<a['low']:
                        breakout['gaps'].append(dict(direction=direction,formed=end,
                            bottom=float(r.high),top=float(a['low']),
                            first_low=float(a['low']),first_high=float(a['high']),
                            third_low=float(r.low),third_high=float(r.high)))
            running_day_high=max(running_day_high,float(r.high))
            running_day_low=min(running_day_low,float(r.low))
            if 1200<=minute<1440:
                asia_hi=max(asia_hi,float(r.high));asia_lo=min(asia_lo,float(r.low))
            recent.append(dict(t=t,high=float(r.high),low=float(r.low),open=float(r.open),close=float(r.close)))
        if trade is not None:
            trade.update(result='unresolved',exit_reason='END_OF_DATA',exit_time='',
                         exit_price=np.nan,net_usd=np.nan)
            alltrades.append(trade)
        alltrades.append(dict(_diagnostic=True,rr=rr,diagnostics=dict(diagnostics)))
    # Original-stop counterfactual on EXACTLY the same entries.
    # This does not change which subsequent entries the live BE strategy takes.
    index_by_time={t:i for i,t in enumerate(d.index)}
    rows=list(d.itertuples())
    for tr in alltrades:
        if tr.get('_diagnostic'):
            continue
        tr['original_exit_reason']=''
        tr['original_exit_time']=''
        tr['original_net_usd']=np.nan
        tr['be_saved_loss']=False
        tr['be_cut_winner']=False
        start=index_by_time.get(tr['entry_bar'])
        if start is None:
            continue
        side=tr['direction']
        for j in range(start+1,len(rows)):
            r=rows[j]
            prev=rows[j-1]
            if (r.Index-prev.Index!=pd.Timedelta(minutes=1)
                    or r.instrument_id!=prev.instrument_id):
                tr['original_exit_reason']='unresolved_gap'
                break
            hit_stop=(r.low<=tr['stop'] if side==1 else r.high>=tr['stop'])
            hit_target=(r.high>=tr['target'] if side==1 else r.low<=tr['target'])
            if hit_stop:
                px=tr['stop']-side*args.slippage_ticks*TICK
                tr['original_exit_reason']='stop'
            elif hit_target:
                px=tr['target']-side*args.slippage_ticks*TICK
                tr['original_exit_reason']='target'
            else:
                continue
            tr['original_exit_time']=(r.Index+pd.Timedelta(minutes=1)).isoformat()
            tr['original_net_usd']=round(
                (px-tr['entry'])*side*POINT_VALUE*tr['contracts']
                -2*args.commission_side*tr['contracts'],2)
            break
        if tr.get('exit_reason')=='breakeven' and pd.notna(tr['original_net_usd']):
            tr['be_saved_loss']=tr['original_net_usd']<0
            tr['be_cut_winner']=tr['original_net_usd']>0
    diagnostics_rows=[x for x in alltrades if x.get('_diagnostic')]
    run.last_diagnostics=diagnostics_rows
    alltrades=[x for x in alltrades if not x.get('_diagnostic')]
    return pd.DataFrame(alltrades,columns=['rr','session','result','net_usd','exit_time','entry_bar',
        'tapped_5m_fvg','tapped_15m_fvg'] if not alltrades else None)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/NQ_2026_full.csv')
    ap.add_argument('--output',default='results/fvg_touch_holds_no_ifvg')
    ap.add_argument('--risk',type=float,default=200)
    ap.add_argument('--risk-max',type=float,default=220)
    ap.add_argument('--stop-buffer',type=float,default=2)
    ap.add_argument('--slippage-ticks',type=int,default=1)
    ap.add_argument('--commission-side',type=float,default=1.25)
    ap.add_argument('--breakout-lookback',type=int,default=5)
    ap.add_argument('--pivot-left',type=int,default=2)
    ap.add_argument('--pivot-right',type=int,default=2)
    ap.add_argument('--swing-lookback',type=int,default=120)
    ap.add_argument('--near-points',type=float,default=2.0)
    ap.add_argument('--min-rr',type=float,default=1.5)
    ap.add_argument('--fvg-age-min',type=int,default=0)
    ap.add_argument('--fvg-age-max',type=int,default=12,help='Expiry minutes; use -1 for no expiry')
    ap.add_argument('--entry-mode',choices=['inside','near','expanded'],default='expanded')
    ap.add_argument('--close-above-points',type=float,default=2.0)
    ap.add_argument('--breakout-max-minutes',type=int,default=30)
    ap.add_argument('--max-trades-per-session',type=int,default=2)
    ap.add_argument('--stop-after-first-win',action='store_true',default=True,help='After first positive-net exit of a session, skip any second trade that session')
    ap.add_argument('--target-policy',choices=['nearest','fixed','session_day','farther_session_day','swing'],default='nearest')
    ap.add_argument('--ny-start',type=int,default=9*60+40,help='NY entry start in minutes after midnight ET')
    ap.add_argument('--ny-end',type=int,default=10*60+30)
    ap.add_argument('--asia-start',type=int,default=20*60+30)
    ap.add_argument('--asia-end',type=int,default=22*60)
    ap.add_argument('--asia-fixed-target',action='store_true',help='Use fixed 1.5R for Asia; NY retains nearest liquidity with 1.5R floor')
    ap.add_argument('--partial-at-rr',action='store_true',help='Take floor(contracts/2) off at +1.25R, after stop-first bar handling')
    ap.add_argument('--partial-rr',type=float,default=1.25)
    ap.add_argument('--be-trigger-rr',type=float,default=1.25)
    ap.add_argument('--be-mode',choices=['rr','swing','opposing','opposing_or_rr','opposing_and_rr','swing_1r','cisd_wick','none'],default='opposing_or_rr')
    ap.add_argument('--no-be',dest='use_be',action='store_false',help='Disable BE to compare the same session filters with original stop')
    ap.set_defaults(use_be=True)
    args=ap.parse_args()
    if args.fvg_age_max == -1: args.fvg_age_max=None
    if not args.use_be: args.be_mode='none'
    d=load(args.data)
    trades=run(d,htf_events(d),args)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    trades.drop(columns=['entry_bar']).to_csv(out/'trades_variable_target.csv',index=False)
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
        taps=(completed.groupby(['session','tapped_5m_fvg','tapped_15m_fvg'])
              .agg(trades=('net_usd','size'),wins=('net_usd',lambda x:(x>0).sum()),
                   net_usd=('net_usd','sum')).reset_index())
        taps['win_rate_pct']=(100*taps.wins/taps.trades).round(2)
        taps.to_csv(out/'fvg_tap_analysis.csv',index=False)
        print('\n5m/15m FVG retracement touch breakdown:')
        print(taps.to_string(index=False))
    if not completed.empty:
        halves=(completed.groupby(['session','reached_50pct_breakout'])
              .agg(trades=('net_usd','size'),wins=('net_usd',lambda x:(x>0).sum()),
                   net_usd=('net_usd','sum')).reset_index())
        halves['win_rate_pct']=(100*halves.wins/halves.trades).round(2)
        halves.to_csv(out/'breakout_50pct_analysis.csv',index=False)
        print('\n50% full-breakout retracement breakdown:')
        print(halves.to_string(index=False))
    # Live BE outcomes and paired no-BE counterfactual on the same entry set.
    pairs=trades[trades.net_usd.notna() & trades.original_net_usd.notna()].copy()
    if not pairs.empty:
        pairs['be_minus_original_usd']=pairs.net_usd-pairs.original_net_usd
        pairs.drop(columns=['entry_bar']).to_csv(out/'be_trade_comparison.csv',index=False)
        bsum=pairs.groupby(['session']).agg(
            trades=('net_usd','size'),
            original_stop_pnl=('original_net_usd','sum'),
            be_strategy_pnl=('net_usd','sum'),
            be_activated=('be_armed','sum'),
            be_exits=('exit_reason',lambda x:(x=='breakeven').sum()),
            saved_losers=('be_saved_loss','sum'),
            cut_winners=('be_cut_winner','sum'),
            be_impact_usd=('be_minus_original_usd','sum')).reset_index()
        bsum.to_csv(out/'be_summary.csv',index=False)
        print('\\n+1.25R BE vs original stop (paired entries):')
        print(bsum.to_string(index=False))
    print(s.to_string(index=False))
    print('Files:',out.resolve())



# --- Unified NQ (MNQ execution) / GC (MGC execution) runner ---
def dual_main():
    import argparse
    from types import SimpleNamespace
    import sys
    ap=argparse.ArgumentParser(description="Backtest NQ/MNQ and Gold GC/MGC, same setup, market-specific contract sizing")
    ap.add_argument('--nq',default='NQ_2026_full.csv',help='Path to NQ Databento CSV')
    ap.add_argument('--gc',default='GC_2026_full.csv',help='Path to Gold Databento CSV')
    ap.add_argument('--market',choices=['both','nq','gc'],default='both')
    ap.add_argument('--output',default='results_nq_gc_2026')
    ap.add_argument('--gold-stop-buffer',type=float,default=2.0,help='GC dollars beyond FVG candle 1; initial assumption, not volatility-normalized')
    ap.add_argument('--gold-close-points',type=float,default=2.0,help='GC dollars above/below gap for entry; initial assumption')
    ap.add_argument('--gold-commission-side',type=float,default=1.25,help='USD / side per MGC contract; update to broker fee')
    ap.add_argument('--nq-commission-side',type=float,default=1.25,help='USD / side per MNQ contract')
    ap.add_argument('--risk',type=float,default=200.)
    ap.add_argument('--risk-max',type=float,default=220.)
    ap.add_argument('--asia-fixed-1p5',action='store_true',help='Use exact fixed 1.5R in Asia; default uses liquidity target at least 1.5R')
    ap.add_argument('--partial',action='store_true',help='Take half off at 1.25R; not recommended by past NQ tests')
    ap.add_argument('--expiry',type=int,default=12)
    ap.add_argument('--close-points',type=float,default=2.,help='NQ allowance above/below FVG gap')
    ap.add_argument('--be-mode',default='opposing_or_rr',choices=['opposing_or_rr','opposing','rr','none'])
    a=ap.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    all_summaries=[]
    for market in (['NQ','GC'] if a.market=='both' else [a.market.upper()]):
        file=Path(a.nq if market=='NQ' else a.gc)
        if not file.is_file():
            sys.exit(f'ERROR: {market} file not found: {file.resolve()}\nRun from your NQ Backtester folder, or specify --{market.lower()} with full path.')
        global POINT_VALUE, TICK
        POINT_VALUE=2.0 if market=='NQ' else 10.0  # MNQ and MGC execution; GC provides signals only
        TICK=0.25 if market=='NQ' else 0.10
        params=SimpleNamespace(
            risk=a.risk,risk_max=a.risk_max,stop_buffer=2.0 if market=='NQ' else a.gold_stop_buffer,
            slippage_ticks=1,commission_side=a.nq_commission_side if market=='NQ' else a.gold_commission_side,
            breakout_lookback=5,pivot_left=2,pivot_right=2,swing_lookback=120,
            near_points=2.,min_rr=1.5,fvg_age_min=0,fvg_age_max=a.expiry if a.expiry>=0 else None,
            entry_mode='near',close_above_points=a.close_points if market=='NQ' else a.gold_close_points,
            breakout_max_minutes=30,max_trades_per_session=2,stop_after_first_win=True,
            target_policy='nearest',ny_start=9*60+40,ny_end=10*60+30,
            asia_start=20*60+30,asia_end=22*60,asia_fixed_target=a.asia_fixed_1p5,
            partial_at_rr=a.partial,partial_rr=1.25,be_trigger_rr=1.25,be_mode=a.be_mode,use_be=(a.be_mode!='none'))
        print(f'\n===== {market}: loading {file} | execution {"MNQ" if market=="NQ" else "MGC"} =====',flush=True)
        df=load(file)
        print(f'{len(df):,} bars, {df.index.min()} through {df.index.max()} | tick {TICK} | $/point {POINT_VALUE}',flush=True)
        tr=run(df,htf_events(df),params)
        folder=root/market;folder.mkdir(parents=True,exist_ok=True)
        tr.drop(columns=['entry_bar'],errors='ignore').to_csv(folder/'trades.csv',index=False)
        if tr.empty:
            print(f'{market}: no trades found. Check dataset coverage and stop sizing.',flush=True)
            continue
        sm=summary(tr)
        sm.insert(0,'market',market)
        sm.to_csv(folder/'summary.csv',index=False)
        detail=[]
        for sess,g in tr.groupby('session'):
            detail.append(dict(market=market,session=sess,trades=len(g),
                targets=int((g.exit_reason=='target').sum()),
                breakevens=int((g.exit_reason=='breakeven').sum()),
                stops=int((g.exit_reason=='stop').sum()),
                net_usd=round(float(g.net_usd.sum()),2)))
        pd.DataFrame(detail).to_csv(folder/'exit_breakdown.csv',index=False)
        all_summaries.extend(sm.to_dict('records'))
        print(sm.to_string(index=False),flush=True)
        print(pd.DataFrame(detail).to_string(index=False),flush=True)
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(root/'combined_summary.csv',index=False)
        print(f'\nFINISHED. Share {root.resolve() / "combined_summary.csv"} and both markets\' trades.csv files.',flush=True)

if __name__=='__main__': dual_main()
