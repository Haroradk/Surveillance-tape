"""
Surveillance tape dashboard: the daily surveillance briefing for BTC and ETH.

Reads only what the daily job publishes - gold (briefings), ops (incidents,
the audit log) and silver bars for the chart - always via the *official* run
per day in ops.daily_runs. Nothing here writes to the warehouse.
Run with: streamlit run app.py
"""

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Streamlit Cloud exposes secrets via st.secrets, not os.environ; bridge them
# before `import config`, which reads os.environ at import time (same as the
# weather dashboards).
try:
    for key in ("MOTHERDUCK_TOKEN", "MOTHERDUCK_DATABASE"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # no secrets.toml locally - .env covers local dev

import config

st.set_page_config(page_title="Surveillance tape", page_icon="\U0001F4C8", layout="wide")

PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}
SYMBOL_COLORS = {"BTCUSDT": "#00412D", "ETHUSDT": "#4B1932"}
STATUS_LABEL = {"written": "Briefing written", "quiet": "Quiet day", "awaiting_analyst": "Awaiting analyst"}
PATTERN_LABEL = {
    "liquidation_cascade": "Liquidation cascade", "large_single_trade": "Large single trade",
    "broad_selling": "Broad selling", "broad_buying": "Broad buying", "short_squeeze": "Short squeeze",
    "volatility_burst": "Volatility burst", "unclear": "Unclear",
}
LATENESS = config.ALLOWED_LATENESS_S


@st.cache_resource
def connection():
    return config.get_connection(read_only=True)


@st.cache_data(ttl=3600)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return connection().execute(sql, list(params)).df()


# ---------------------------------------------------------------- data

days = query("""
    SELECT trade_date, arg_max(run_id, finished_at) AS run_id, arg_max(incidents, finished_at) AS incidents
    FROM ops.daily_runs WHERE status = 'completed'
    GROUP BY 1 ORDER BY 1 DESC
""")
if days.empty:
    st.warning("No processed days yet - the daily job hasn't run.")
    st.stop()

st.sidebar.title("Surveillance tape")
st.sidebar.caption("Daily market surveillance for BTC and ETH on Binance spot. "
                   "A learning project: it detects and explains moves, and never suggests trades.")
labels = {d: f"{pd.Timestamp(d):%a %d %b %Y} · {n} incident{'s' if n != 1 else ''}"
          for d, n in zip(days["trade_date"], days["incidents"])}
day = st.sidebar.radio("Day", list(labels), format_func=labels.get)
run_id = days.loc[days["trade_date"] == day, "run_id"].iloc[0]
st.sidebar.markdown("[How it works (GitHub)](https://github.com/Haroradk/Surveillance-tape)")

briefing = query("""
    SELECT status, model, headline, summary, data_notes, evidence FROM gold.daily_briefings
    WHERE trade_date = ? ORDER BY created_at DESC LIMIT 1
""", (day,))
incidents = query("""
    SELECT i.incident_no, i.symbol, i.opened_at, i.last_active_at, i.price_at_open, i.price_low, i.price_high,
           b.title, b.scope, b.pattern, b.narrative, b.evidence
    FROM ops.incidents i
    LEFT JOIN gold.incident_briefs b ON b.run_id = i.run_id AND b.incident_no = i.incident_no
    WHERE i.run_id = ? AND i.allowed_lateness_s = ?
    ORDER BY i.opened_at
""", (run_id, LATENESS))
bars = query("""
    SELECT symbol, bar_start, close::DOUBLE AS close, volume::DOUBLE AS volume
    FROM silver.bars_1m WHERE run_id = ? AND allowed_lateness_s = ? ORDER BY bar_start
""", (run_id, LATENESS))

# ---------------------------------------------------------------- briefing

tab_day, tab_week = st.tabs(["Daily briefing", "Pipeline"])

with tab_day:
    if briefing.empty:
        st.info("No briefing for this day yet.")
        evidence = {"markets": []}
    else:
        b = briefing.iloc[0]
        evidence = json.loads(b["evidence"])
        st.caption(f"{pd.Timestamp(day):%A %d %B %Y} (UTC) · {STATUS_LABEL.get(b['status'], b['status'])}"
                   + (f" · written by {b['model']}" if b["model"] else ""))
        st.header(b["headline"])
        st.write(b["summary"])
        if b["data_notes"]:
            st.caption(f"Data notes: {b['data_notes']}")

    cols = st.columns(max(len(evidence["markets"]), 1))
    for col, m in zip(cols, evidence["markets"]):
        col.metric(m["symbol"].replace("USDT", " / USDT"), f"{m['close']:,.2f}", f"{m['day_return_pct']:+.2f}% on the day")
        col.caption(f"High-low range {m['high_low_range_pct']:.2f}% · volume {m['volume_vs_normal_day_x']:.2f}x a normal day")

    # Price per coin, one panel each (very different price levels), incidents shaded.
    symbols = [s for s in SYMBOL_COLORS if s in set(bars["symbol"])]
    fig = make_subplots(rows=len(symbols), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[s.replace("USDT", "") for s in symbols])
    for row, symbol in enumerate(symbols, start=1):
        s = bars[bars["symbol"] == symbol]
        fig.add_trace(go.Scatter(x=s["bar_start"], y=s["close"], mode="lines", name=symbol,
                                 line=dict(color=SYMBOL_COLORS[symbol], width=1.4),
                                 hovertemplate="%{x|%H:%M} UTC<br>%{y:,.2f}<extra></extra>"), row=row, col=1)
        for _, inc in incidents[incidents["symbol"] == symbol].iterrows():
            fig.add_vrect(x0=inc["opened_at"] - pd.Timedelta(seconds=60), x1=inc["last_active_at"],
                          fillcolor="#C0392B", opacity=0.18, line_width=0, row=row, col=1,
                          annotation_text=f"#{inc['incident_no']}", annotation_position="top left",
                          annotation_font_size=11)
    fig.update_layout(height=230 * max(len(symbols), 1), showlegend=False, margin=dict(t=30, b=10, l=10, r=10),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption("Shaded: incidents, from just before the price shock until the last shock. "
               "Price is the 1-minute close, built from every trade.")

    st.subheader(f"Incidents ({len(incidents)})")
    if incidents.empty:
        st.write("None. The only rule that opens incidents, a price shock over 10x the normal "
                 "1-minute move for that hour of day, never fired.")
    for _, inc in incidents.iterrows():
        ev = json.loads(inc["evidence"]) if isinstance(inc["evidence"], str) else {}
        title = inc["title"] or f"{inc['symbol']} incident (not written up yet)"
        tags = " · ".join(t for t in (
            "Market-wide" if inc["scope"] == "market_wide" else ("Single asset" if inc["scope"] else None),
            PATTERN_LABEL.get(inc["pattern"]) if inc["pattern"] else None) if t)
        with st.expander(f"#{inc['incident_no']}  {inc['opened_at']:%H:%M} UTC · {title}", expanded=len(incidents) <= 3):
            if tags:
                st.caption(tags)
            if inc["narrative"]:
                st.write(inc["narrative"])
            if ev:
                c = st.columns(4)
                c[0].metric("Move at the end", f"{ev.get('move_at_end_pct', 0):+.2f}%",
                            f"{ev['move_60m_after_pct']:+.2f}% after 1 h" if ev.get("move_60m_after_pct") is not None else None,
                            delta_color="off")
                c[1].metric("Biggest 60 s move vs normal", f"{ev.get('max_move_vs_normal_x', 0):.0f}x")
                c[2].metric("Sell-initiated volume", f"{ev.get('sell_initiated_volume_pct', 0):.0f}%")
                c[3].metric("Largest trade, share of volume", f"{ev['largest_trade_pct_of_volume']:.1f}%"
                            if ev.get("largest_trade_pct_of_volume") is not None else "n/a")
                timeline = ", ".join(f"{a['rule']} ({a['seconds_from_open']:+d} s)" for a in ev.get("alert_timeline") or [])
                st.caption(f"Alerts, relative to the price shock: {timeline}")
                if ev.get("overlapping_incidents_other_symbol"):
                    st.caption(f"{ev['other_symbol']} moved {ev['other_symbol_move_pct']:+.2f}% in the same window "
                               f"(its incident {', '.join('#' + str(n) for n in ev['overlapping_incidents_other_symbol'])}).")
                with st.popover("All evidence"):
                    st.json(ev)

    st.caption("Explanations are written by an LLM from measurements taken by fixed SQL queries, and describe what "
               "happened, not why in the news. Nothing here is investment advice.")

# ---------------------------------------------------------------- pipeline

with tab_week:
    st.subheader("Incidents per day")
    per_day = days.sort_values("trade_date")
    fig = go.Figure(go.Bar(x=per_day["trade_date"], y=per_day["incidents"], marker_color="#00412D",
                           hovertemplate="%{x|%a %d %b}: %{y}<extra></extra>"))
    fig.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10), yaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.subheader("Daily job runs")
    st.caption("The audit log (ops.daily_runs): every run, whether it reconciled with Binance's own 1-minute "
               "candles, how many raw rows retention purged, and the git commit it ran.")
    runs = query("""
        SELECT trade_date AS day, status, trades, alerts, incidents,
               minutes_matching || ' / ' || minutes_checked AS minutes_matching_binance,
               bronze_rows_purged AS raw_rows_purged, code_version AS commit, finished_at
        FROM ops.daily_runs ORDER BY started_at DESC LIMIT 50
    """)
    st.dataframe(runs, use_container_width=True, hide_index=True)

    st.subheader("LLM calls")
    st.caption("One briefing call per day with incidents, at most 3 a day; quiet days need none.")
    st.dataframe(query("""
        SELECT called_at, trade_date AS day, model, status, prompt_chars, output_chars
        FROM ops.llm_calls ORDER BY called_at DESC LIMIT 30
    """), use_container_width=True, hide_index=True)
