"""Signal research: does anything the surveillance rules see predict what happens next?

An event study over the official daily runs. For every alert, measure the
price return from the first full minute *after the pipeline detected it* to
5, 15 and 60 minutes later, and compare with control minutes: the same
symbol, the same hour of day, on other days, away from any alert. The
control is what "nothing special happening at this time of day" looks like.

Questions:
  Q1 size      after a volume_burst / one_sided_flow alert, are later moves bigger than normal?
  Q2 direction does the flow's direction (buy- vs sell-initiated) predict the move's direction?
  Q3 shocks    after a price_shock, does the move continue (momentum) or revert?

Discipline, the part an algo developer would insist on:
  - no look-ahead: returns start after detection, never at the event itself
  - discovery vs holdout: the first 2/3 of days to look, the last 1/3 to check
  - bootstrap confidence intervals, and sample sizes printed next to every number
  - a cost check: an effect smaller than a round trip of fees (~0.2%) isn't tradeable,
    which matters for judging it even though this project never trades

  python scripts/signal_research.py            # uses whatever get_connection() points at
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ALLOWED_LATENESS_S, get_connection  # noqa: E402

HORIZONS = [5, 15, 60]
CONTROLS_PER_EVENT = 20
ROUND_TRIP_COST_PCT = 0.2
RNG = np.random.default_rng(7)


def load(con):
    runs = con.execute("""
        SELECT trade_date, arg_max(run_id, finished_at) AS run_id FROM ops.daily_runs
        WHERE status = 'completed' GROUP BY 1 ORDER BY 1
    """).df()
    ids = runs["run_id"].tolist()
    bars = con.execute("""
        SELECT symbol, bar_start, close::DOUBLE AS close FROM silver.bars_1m
        WHERE list_contains(?, run_id) AND allowed_lateness_s = ? ORDER BY symbol, bar_start
    """, [ids, ALLOWED_LATENESS_S]).df()
    alerts = con.execute("""
        SELECT a.run_id, a.symbol, a.rule, a.fired_at, a.detected_at, a.features, a.incident_no
        FROM ops.alerts a WHERE list_contains(?, a.run_id) AND a.allowed_lateness_s = ?
    """, [ids, ALLOWED_LATENESS_S]).df()
    alerts = alerts.merge(runs, on="run_id")
    return runs, bars, alerts


def forward_returns(bars: pd.DataFrame, symbol: str, starts: pd.Series) -> pd.DataFrame:
    """Return (%) from the close of minute `start` to `start + h` minutes, for each horizon."""
    s = bars[bars["symbol"] == symbol].set_index("bar_start")["close"]
    base = s.reindex(starts.values).to_numpy()
    out = {}
    for h in HORIZONS:
        later = s.reindex((starts + pd.Timedelta(minutes=h)).values).to_numpy()
        out[f"ret_{h}m"] = (later / base - 1) * 100
    return pd.DataFrame(out, index=starts.index)


def entry_minute(detected_at: pd.Series) -> pd.Series:
    # The first full minute after the pipeline knew: the earliest anyone could act.
    return detected_at.dt.floor("min") + pd.Timedelta(minutes=1)


def controls(bars, alerts, symbol, entries: pd.Series) -> pd.DataFrame:
    """For each event minute: random minutes, same symbol and hour of day, other days, no alert nearby."""
    s = bars[bars["symbol"] == symbol]["bar_start"]
    busy = alerts[alerts["symbol"] == symbol]["detected_at"].dt.floor("min")
    busy_set = set()
    for t in busy:
        busy_set.update(t + pd.Timedelta(minutes=m) for m in range(-60, 61))
    pool = s[~s.isin(busy_set)]
    by_hour = {h: g.to_numpy() for h, g in pool.groupby(pool.dt.hour)}
    picks = []
    for t in entries:
        candidates = by_hour.get(t.hour, np.array([]))
        candidates = candidates[pd.to_datetime(candidates).date != t.date()] if len(candidates) else candidates
        if len(candidates):
            picks.extend(RNG.choice(candidates, size=min(CONTROLS_PER_EVENT, len(candidates)), replace=False))
    return pd.Series(pd.to_datetime(picks))


def bootstrap_ci(x: np.ndarray, stat=np.mean, n=2000) -> tuple:
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return (np.nan, np.nan)
    draws = [stat(RNG.choice(x, size=len(x), replace=True)) for _ in range(n)]
    return tuple(np.percentile(draws, [2.5, 97.5]))


def event_table(bars, alerts, rule: str) -> pd.DataFrame:
    """One row per alert: entry minute, forward returns, and its direction signal."""
    ev = alerts[alerts["rule"] == rule].copy()
    if ev.empty:
        return ev
    ev["entry"] = entry_minute(ev["detected_at"])
    feats = ev["features"].map(json.loads)
    ev["ret_60_at_alert"] = feats.map(lambda f: f.get("ret_60"))
    ev["sell_share"] = feats.map(lambda f: f.get("sell_share_120"))
    parts = [pd.concat([g, forward_returns(bars, sym, g["entry"])], axis=1) for sym, g in ev.groupby("symbol")]
    return pd.concat(parts)


def fmt(lo_hi):
    return f"[{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"


def main() -> None:
    con = get_connection(read_only=True)
    runs, bars, alerts = load(con)
    days = sorted(runs["trade_date"])
    split = days[int(len(days) * 2 / 3)]
    print(f"{len(days)} days ({days[0]} to {days[-1]}), {len(alerts)} alerts. "
          f"Discovery: before {split}; holdout: {split} onwards.\n")

    # ---------------- Q1: are moves after context alerts bigger than normal?
    print("Q1  Size of later moves: mean |return| after the alert vs control minutes (same hour, other days)")
    print(f"{'rule':<16}{'set':<11}{'n':>5}" + "".join(f"{f'|ret| {h}m ev / ctl':>22}" for h in HORIZONS))
    for rule in ("volume_burst", "one_sided_flow", "price_shock"):
        ev = event_table(bars, alerts, rule)
        if ev.empty:
            continue
        for label, part in (("discovery", ev[ev["trade_date"] < split]), ("holdout", ev[ev["trade_date"] >= split])):
            cells = []
            for h in HORIZONS:
                e = np.abs(part[f"ret_{h}m"].to_numpy())
                ctl = pd.concat([forward_returns(bars, sym, controls(bars, alerts, sym, g["entry"]).rename("entry"))
                                 for sym, g in part.groupby("symbol")]) if len(part) else pd.DataFrame()
                c = np.abs(ctl[f"ret_{h}m"].to_numpy()) if len(ctl) else np.array([np.nan])
                cells.append(f"{np.nanmean(e):.3f} / {np.nanmean(c):.3f}%")
            print(f"{rule:<16}{label:<11}{len(part):>5}" + "".join(f"{c:>22}" for c in cells))
    print()

    # ---------------- Q2: does flow direction predict move direction?
    print("Q2  Direction: after one-sided flow, does the price keep going the flow's way?")
    print("    signal = +1 when buyers dominated (sell share < 50%), -1 when sellers did; "
          "'with signal' = mean of signal x return")
    ev = event_table(bars, alerts, "one_sided_flow")
    if not ev.empty:
        ev["signal"] = np.where(ev["sell_share"] < 0.5, 1, -1)
        for label, part in (("discovery", ev[ev["trade_date"] < split]), ("holdout", ev[ev["trade_date"] >= split])):
            for h in HORIZONS:
                x = (part["signal"] * part[f"ret_{h}m"]).to_numpy()
                hit = np.nanmean(x > 0) * 100 if len(x) else np.nan
                print(f"    {label:<10} {h:>2}m  n={np.sum(~np.isnan(x)):>4}  with signal {np.nanmean(x):+.3f}%  "
                      f"95% CI {fmt(bootstrap_ci(x))}  hit rate {hit:.0f}%")
    print()

    # ---------------- Q3: after a shock, momentum or reversal?
    print("Q3  After a price shock: continue (positive) or revert (negative)? Return in the shock's direction")
    ev = event_table(bars, alerts, "price_shock")
    if not ev.empty:
        ev["direction"] = np.sign(ev["ret_60_at_alert"])
        for label, part in (("discovery", ev[ev["trade_date"] < split]), ("holdout", ev[ev["trade_date"] >= split])):
            for h in HORIZONS:
                x = (part["direction"] * part[f"ret_{h}m"]).to_numpy()
                hit = np.nanmean(x > 0) * 100 if len(x) else np.nan
                print(f"    {label:<10} {h:>2}m  n={np.sum(~np.isnan(x)):>4}  in shock direction {np.nanmean(x):+.3f}%  "
                      f"95% CI {fmt(bootstrap_ci(x))}  continues {hit:.0f}% of the time")
    print(f"\nCost check: a round trip costs roughly {ROUND_TRIP_COST_PCT}%. Any 'with signal' mean below that, or "
          "a confidence interval that includes 0, is not an edge.")


if __name__ == "__main__":
    main()
