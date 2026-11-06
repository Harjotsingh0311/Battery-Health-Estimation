"""Cell Observatory: Phase 9 dashboard for the capacity-free SOH monitor.

Replays the frozen monitor's per-cycle records (outputs/phase8b/replay/*.parquet).
It is a replay of lab data, one update per cycle, not a live feed.

Run from the project root:  uv run streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------- paths
ROOT = Path(os.environ.get("SOH_ROOT", Path(__file__).resolve().parents[1]))
OUT = ROOT / "outputs"
REPLAY_DIR = OUT / "phase8b" / "replay"
PER_CELL_CSV = OUT / "phase8b" / "s6_replay_per_cell.csv"
LATENCY_CSV = OUT / "phase8b" / "s9_latency.csv"
VERIFY_JSON = OUT / "phase8b" / "verification_report.json"
CONFIG_JSON = OUT / "phase7b" / "monitor_config.json"
MODEL_ONNX = OUT / "phase8a" / "models" / "M0_W240.onnx"

# Frozen artefacts (hashes taken from the phase 7b, 8a and 8b reports)
EXPECTED_MONITOR_SHA = "a9531dde7d933954a1d942112cd147256fe7138fe9c4cb6e17cc12b257ee4b53"
HASHES = {
    "src/health_monitor.py (v4)": EXPECTED_MONITOR_SHA,
    "monitor_config.json": "638b6c3a6ad31c699fb77a22eb9fae72b571d6dc21ee65a73b41e849d7acb4b2",
    "monitor_config_addendum.json": "d842e5f4c3186889ce8292e0654e1a0ca29d4434bcdeaf1bd3c7131479479632",
    "M0_W240.onnx": "4fe196d3f7787a4f2dcd5f20939447518afb032eb8dc724c018f891630945771",
    "onnx_scorer.py": "7f56bd5b5df08f7a37bc6104a140da2411b5290d25434a03aa8971599ec139c7",
}
BAR_MIN = 15.0  # scale of the timing bar; early-life discharges last about 14.6 min, late-life about 11.6 min (Phase 2, 4 A)

C = dict(bg="#0b1220", panel="#111a2e", line="#1e2a44", text="#e5e7eb", muted="#94a3b8",
         teal="#2dd4bf", amber="#fbbf24", orange="#fb923c", red="#f87171", violet="#a78bfa", blue="#60a5fa")

# Story beats: shown when the replay reaches them. Numbers are from the Phase 7b2 / 8b / 8c reviews.
NOTES: dict[str, list[tuple[int, str, str]]] = {
    "b3c38": [
        (1321, "A step the truth does not have",
         "Cycles 1321 to 1329: average cell temperature rises from 35.6 to 37.0 °C and the estimate steps up by about 0.8 pp "
         "while true SOH is smooth. Online and offline pipelines agree, so this is model behaviour. M0 has no temperature "
         "input. Correlation, not proof."),
    ],
    "b1c36": [
        (489, "A dip near cycle 490",
         "Cycles 489 to 493: average temperature falls from 34.4 to 28.9 °C with an internal-resistance jump, and the estimate "
         "dips to about 93 pp against a true 94.7 to 94.9. Both pipelines show it. Two episodes in two cells, not a calibrated sensitivity."),
    ],
    "b1c5": [
        (158, "One cycle the cause of which is unresolved",
         "At cycle 158 the online window and the stored curve differ by 143 mAh near 2.54 V, and the online estimate is 92.3 pp "
         "against a true 98.0. One cycle in 10,744 compared. Listed as a limitation."),
    ],
    "b1c18": [
        (8, "The reference problem",
         "Early cycles 8 to 40 have charge/discharge capacity gaps. The offline pipeline used cycles 54 to 58 as the reference; "
         "the causal monitor (v4) picks 41 to 45. This cell has the largest online/offline difference."),
    ],
    "b3c28": [
        (60, "A cell that reads too healthy",
         "Over-estimated throughout, and its end-of-life watch never fires. Over-estimated cells are the ones the lower-bound "
         "alert misses."),
    ],
    "b2c1": [
        (20, "The shortest-lived cell in the set",
         "Life 148 cycles, below the training range in development folds. It is a known hard case."),
    ],
}

# --------------------------------------------------------------------------- page + styling
st.set_page_config(page_title="Cell Observatory", page_icon="🔋", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {border: 1px solid #1e2a44; border-radius: 18px; padding: 20px 26px; margin-bottom: 14px;
  background: radial-gradient(900px 220px at 85% -40%, rgba(45,212,191,.20), transparent 60%),
              radial-gradient(700px 200px at 0% 120%, rgba(167,139,250,.16), transparent 60%), #0f172a;}
.hero h1 {margin: 0; font-size: 1.9rem; font-weight: 700;}
.hero p {margin: 4px 0 0 0; color: #94a3b8; font-size: .95rem;}
.chips {display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;}
.chip {border:1px solid #1e2a44; background:#0b1220; color:#cbd5e1; border-radius:999px; padding:3px 11px; font-size:.78rem;}
.chip b {color:#2dd4bf; font-weight:600;}
.card {border:1px solid #1e2a44; border-radius:16px; padding:14px 16px; background:#111a2e; margin-bottom:12px;}
.card h4 {margin:0 0 6px 0; font-size:.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.08em; font-weight:600;}
.big {font-size:2.5rem; font-weight:700; line-height:1.05; font-variant-numeric: tabular-nums;}
.sub {color:#94a3b8; font-size:.82rem;}
.row {display:flex; gap:14px; align-items:center;}
.batt {width:74px; flex:none;}
.batt .cap {width:26px; height:7px; background:#475569; margin:0 auto; border-radius:3px 3px 0 0;}
.batt .body {position:relative; height:128px; border:3px solid #475569; border-radius:10px; overflow:hidden; background:#0b1220;}
.batt .fill {position:absolute; bottom:0; left:0; right:0; transition: height .25s ease, background .25s ease;}
.batt .tick {position:absolute; left:0; right:0; border-top:1px dashed rgba(226,232,240,.45);}
.batt .tick span {position:absolute; right:3px; top:-13px; font-size:.62rem; color:#cbd5e1;}
.lamps {display:flex; gap:10px; margin-top:8px;}
.lamp {flex:1; border:1px solid #1e2a44; border-radius:12px; padding:8px 10px; background:#0b1220;}
.lamp .dot {display:inline-block; width:11px; height:11px; border-radius:50%; background:#334155; margin-right:7px; vertical-align:-1px;}
.lamp.on.warn {border-color:#fbbf24;} .lamp.on.warn .dot {background:#fbbf24; box-shadow:0 0 12px #fbbf24;}
.lamp.on.eol {border-color:#a78bfa;} .lamp.on.eol .dot {background:#a78bfa; box-shadow:0 0 12px #a78bfa;}
.lamp small {display:block; color:#94a3b8; margin-top:2px;}
.bar {position:relative; height:12px; border-radius:99px; background:#0b1220; border:1px solid #1e2a44; margin:8px 0 4px 0;}
.bar .mark {position:absolute; top:-4px; width:4px; height:18px; border-radius:2px; background:#2dd4bf; box-shadow:0 0 10px #2dd4bf;}
.bar .zone {position:absolute; top:0; bottom:0; border-radius:99px; background:rgba(45,212,191,.18);}
.status {display:inline-block; padding:2px 10px; border-radius:99px; font-size:.75rem; font-weight:600; border:1px solid #334155;}
.s-OK {color:#2dd4bf; border-color:#2dd4bf55;} .s-STALE, .s-INVALID_WINDOW {color:#fbbf24; border-color:#fbbf2466;}
.s-NO_REFERENCE {color:#94a3b8;}
.beat {border-left:3px solid #a78bfa; background:#151c33; border-radius:0 12px 12px 0; padding:10px 14px; margin-top:8px;}
.beat b {color:#c4b5fd;}
.tick-row {font-size:.84rem; padding:5px 0; border-bottom:1px solid #1e2a44; color:#cbd5e1;}
.tick-row i {font-style:normal; color:#94a3b8; margin-right:8px; font-variant-numeric: tabular-nums;}
.grid {display:grid; grid-template-columns: repeat(auto-fill, minmax(235px, 1fr)); gap:12px;}
.tile {border:1px solid #1e2a44; border-radius:14px; padding:10px 12px; background:#111a2e;}
.tag {display:inline-block; font-size:.62rem; padding:0 7px; margin-left:6px; border-radius:99px; border:1px solid #334155; color:#94a3b8; vertical-align:1px; font-weight:600;}
.tag.test {color:#2dd4bf; border-color:#2dd4bf88;}
.tile.warn {border-color:#fbbf2488;} .tile.eol {border-color:#a78bfa99;} .tile.ended {opacity:.62;}
.tile .t {display:flex; justify-content:space-between; align-items:baseline;}
.tile .t b {font-size:.95rem;} .tile .t span {font-size:.7rem; color:#94a3b8;}
.tile .v {font-size:1.6rem; font-weight:700; font-variant-numeric: tabular-nums;}
.pipe {display:flex; flex-wrap:wrap; gap:8px; align-items:stretch; margin:6px 0 14px 0;}
.pipe div {flex:1 1 150px; border:1px solid #1e2a44; background:#111a2e; border-radius:14px; padding:10px 12px; font-size:.84rem;}
.pipe div b {display:block; color:#2dd4bf; font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:3px;}
.verdict {border-radius:16px; padding:14px 18px; border:1px solid #1e2a44; background:#111a2e;}
.verdict .k {font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#94a3b8;}
.verdict .p {font-size:1.5rem; font-weight:700;}
.pass {color:#2dd4bf;} .fail {color:#f87171;}
code, .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
def cell_key(cid: str) -> tuple[int, int]:
    m = re.match(r"b(\d)c(\d+)", cid)
    return (int(m.group(1)), int(m.group(2))) if m else (9, 0)


def is_test(cid: str) -> bool:
    """Batch 3 is the locked, held-out test batch; batches 1 and 2 trained the deployed model."""
    return cell_key(cid)[0] == 3


def split_tag(cid: str) -> str:
    return ('<span class="tag test">held-out test</span>' if is_test(cid)
            else '<span class="tag">in training data</span>')


def find_key(obj, key):
    """First value stored under `key` anywhere inside a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def _stamp(paths) -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns) for p in paths)


@st.cache_data(show_spinner="Loading replay files…")
def _load(files: tuple[str, ...], stamp: tuple):
    data = {}
    for f in files:
        df = pd.read_parquet(f).sort_values("cycle", kind="stable").reset_index(drop=True)
        for col in ("raw", "ema", "lo", "hi", "lower_bound", "soh_nom", "minutes_into_discharge", "temp_delta_c"):
            df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
        for col in ("warn85", "eol_watch", "stale"):
            df[col] = df[col].fillna(False).astype(bool) if col in df else False
        df["status"] = df["status"].astype(str)
        data[Path(f).stem] = df
    return dict(sorted(data.items(), key=lambda kv: cell_key(kv[0])))


@st.cache_data
def _per_cell(path: str, stamp: float):
    df = pd.read_csv(path)
    return df.set_index("cell_id") if "cell_id" in df else pd.DataFrame()


def load_everything():
    files = sorted(REPLAY_DIR.glob("*.parquet"))
    if not files:
        return None, None
    data = _load(tuple(str(f) for f in files), _stamp(files))
    per = _per_cell(str(PER_CELL_CSV), PER_CELL_CSV.stat().st_mtime) if PER_CELL_CSV.exists() else pd.DataFrame()
    return data, per


def first_cycle(df: pd.DataFrame, col: str):
    s = df.loc[df[col], "cycle"]
    return int(s.min()) if len(s) else None


def life_of(cid: str, df: pd.DataFrame, per: pd.DataFrame):
    if cid in per.index and "life" in per.columns and pd.notna(per.loc[cid, "life"]):
        return int(per.loc[cid, "life"])
    return int(df.cycle.max())


def events_for(df: pd.DataFrame) -> list[tuple[int, str, str]]:
    ev = []
    ok = df[(df.status == "OK") & df.raw.notna()]
    if len(ok):
        ev.append((int(ok.cycle.iloc[0]), "start", "First estimate issued (reference cycles built, scoring begins)"))
    for c in df.loc[df.status == "NO_REFERENCE", "cycle"].head(1):
        ev.append((int(c), "info", "No reference curve yet: no estimate"))
    for c in df.loc[df.status == "INVALID_WINDOW", "cycle"]:
        ev.append((int(c), "warn", "Window failed validation: last estimate carried forward, marked stale"))
    for c in df.loc[df.status == "STALE", "cycle"]:
        ev.append((int(c), "warn", "Cycle missing or unusable: last estimate carried forward, marked stale"))
    w, e = first_cycle(df, "warn85"), first_cycle(df, "eol_watch")
    if w is not None:
        ev.append((w, "warn85", "85% warning fired (raw estimate ≤ 85)"))
    if e is not None:
        ev.append((e, "eol", "End-of-life watch fired (estimate minus lower-bound margin ≤ 80)"))
    return sorted(ev)


# --------------------------------------------------------------------------- small visual helpers
def soh_color(v: float) -> str:
    if not np.isfinite(v):
        return "#334155"
    return C["teal"] if v >= 90 else C["amber"] if v >= 85 else C["orange"] if v >= 80 else C["red"]


def fmt(v, nd=2, unit=""):
    return "–" if v is None or not np.isfinite(v) else f"{v:.{nd}f}{unit}"


def battery_html(v: float) -> str:
    h = float(np.clip(v, 0, 100)) if np.isfinite(v) else 0.0
    ticks = "".join(f'<div class="tick" style="bottom:{t}%"><span style="top:{-13 if t == 85 else 1}px">{t}</span></div>' for t in (85, 80))
    return (f'<div class="batt"><div class="cap"></div><div class="body">'
            f'<div class="fill" style="height:{h}%;background:{soh_color(v)}"></div>{ticks}</div></div>')


def lamp_html(kind: str, label: str, fired: int | None, cyc: int) -> str:
    on = fired is not None and fired <= cyc
    note = f"fired at cycle {fired}" if on else "armed"
    return (f'<div class="lamp {"on " + kind if on else ""}"><span class="dot"></span><b>{label}</b>'
            f"<small>{note}</small></div>")


def layout(fig, h=430, **kw):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"], size=12),
                      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"), **kw)
    fig.update_xaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
    return fig


def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


SPEEDS = {"1×": 1, "4×": 4, "12×": 12, "40×": 40}


# --------------------------------------------------------------------------- page: mission control
def page_mission(data, per):
    cells = list(data)
    ss("play", False)
    ss("cell", "b3c38" if "b3c38" in cells else cells[0])

    def first_pos(cid):
        d = data[cid]
        ok = d.index[(d.status == "OK") & d.raw.notna()]
        return int(d.cycle[ok[0]]) if len(ok) else int(d.cycle.iloc[0])

    ss("cyc", first_pos(st.session_state.cell))

    def on_cell():
        st.session_state.cyc = first_pos(st.session_state.cell)
        st.session_state.play = False

    top = st.columns([1.2, 1, 1, 1, 1.3])
    top[0].selectbox("Cell", cells, key="cell", on_change=on_cell,
                     format_func=lambda c: f"{c}  ·  batch {cell_key(c)[0]}")
    speed = top[1].select_slider("Replay speed", list(SPEEDS), value="12×")
    if top[2].button("⏸ Pause" if st.session_state.play else "▶ Play", width="stretch"):
        st.session_state.play = not st.session_state.play
        if st.session_state.play and st.session_state.cyc >= int(data[st.session_state.cell].cycle.max()):
            st.session_state.cyc = first_pos(st.session_state.cell)
        st.rerun()
    if top[3].button("↺ Restart", width="stretch"):
        st.session_state.cyc = first_pos(st.session_state.cell)
        st.session_state.play = False
        st.rerun()
    reveal = top[4].toggle("Reveal ground truth (descriptive)", key="reveal",
                           help="Overlays measured SOH (discharge capacity ÷ 1.1 Ah). The monitor never sees it. "
                                "Batch-3 labels are spent, so this is for looking, not for tuning.")

    run_every = 0.25 if st.session_state.play else None

    @st.fragment(run_every=run_every)
    def stage():
        cid = st.session_state.cell
        df = data[cid]
        cmin, cmax = int(df.cycle.min()), int(df.cycle.max())
        if st.session_state.play:
            st.session_state.cyc = min(cmax, st.session_state.cyc + SPEEDS[speed])
            if st.session_state.cyc >= cmax:
                st.session_state.play = False
                st.rerun(scope="app")
        st.session_state.cyc = int(np.clip(st.session_state.cyc, cmin, cmax))
        cyc = st.slider("Cycle replayed", cmin, cmax, key="cyc")
        st.markdown(f'<div class="sub"><b>{cid}</b> {split_tag(cid)} '
                    + ("never seen by the model during training." if is_test(cid)
                       else "was part of the training data of the deployed model, so its accuracy here is in-sample.")
                    + "</div>", unsafe_allow_html=True)
        st.caption("Press ▶ Play or drag the slider. The chart shows only what the monitor has seen up to this cycle.")

        pos = int(np.searchsorted(df.cycle.values, cyc, side="right") - 1)
        row, seen = df.iloc[pos], df.iloc[: pos + 1]
        raw = row.raw
        warn_c, eol_c = first_cycle(df, "warn85"), first_cycle(df, "eol_watch")
        # values shown are the last scored ones, so a warm-up cycle shows a dash instead of a stale number
        left, right = st.columns([1, 2.6], gap="medium")

        with left:
            st.markdown(
                f'<div class="card"><h4>Estimated state of health</h4><div class="row">{battery_html(raw)}'
                f'<div><div class="big" style="color:{soh_color(raw)}">{fmt(raw, 1)}<span style="font-size:1.1rem">%</span></div>'
                f'<div class="sub">of 1.1 Ah nominal · cycle {int(row.cycle)}</div>'
                f'<div style="margin-top:6px"><span class="status s-{row.status}">{row.status.replace("_", " ")}</span>'
                f'{"  <span class=sub>stale</span>" if row.stale else ""}</div></div></div>'
                f'<div class="lamps">{lamp_html("warn", "85% warning", warn_c, cyc)}'
                f'{lamp_html("eol", "EOL watch", eol_c, cyc)}</div></div>', unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m1.metric("Smoothed (EMA 0.3)", fmt(row.ema, 2, "%"))
            m2.metric("90% interval ±", fmt((row.hi - row.lo) / 2 if np.isfinite(row.hi) else np.nan, 2, " pp"))
            m3, m4 = st.columns(2)
            m3.metric("Lower bound", fmt(row.lower_bound, 2, "%"), help="One-sided bound that drives the end-of-life watch.")
            if reveal and np.isfinite(row.soh_nom) and np.isfinite(raw):
                m4.metric("Error vs truth", f"{raw - row.soh_nom:+.2f} pp", help="Estimate minus measured SOH. Positive means too healthy.")
            else:
                m4.metric("Temp shift", fmt(row.temp_delta_c, 1, " °C"), help="Informational only. Not used by the estimate or any alert.")

            mins = row.minutes_into_discharge
            pct = float(np.clip(mins / BAR_MIN, 0, 1)) * 100 if np.isfinite(mins) else None
            zone = 'left:0;width:%.1f%%' % pct if pct is not None else "display:none"
            mark = f'left:calc({pct:.1f}% - 2px)' if pct is not None else "display:none"
            st.markdown(
                f'<div class="card"><h4>When this estimate arrives</h4>'
                f'<div class="bar"><div class="zone" style="{zone}"></div><div class="mark" style="{mark}"></div></div>'
                f'<div class="sub">{"issued " + format(mins, ".1f") + " min into the discharge (bar: 0 to 15 min)" if pct is not None else "no window this cycle"}'
                f'. The window closes at 2.40 V, near the end of a discharge that lasts about 14.6 min early in life and about 11.6 min late. '
                f'One estimate per cycle.</div></div>',
                unsafe_allow_html=True)

        with right:
            temp_ok = df.temp_delta_c.notna().any()
            vals = pd.concat([df.raw, df.lo] + ([df.soh_nom] if reveal else [])).dropna()
            ylo = float(max(60, np.floor(vals.min() - 1.5))) if len(vals) else 75
            fig = make_subplots(rows=2 if temp_ok else 1, cols=1, shared_xaxes=True,
                                row_heights=[0.8, 0.2] if temp_ok else [1], vertical_spacing=0.04)
            fig.add_trace(go.Scatter(x=seen.cycle, y=seen.hi, mode="lines", line=dict(width=0), hoverinfo="skip",
                                     showlegend=False), 1, 1)
            fig.add_trace(go.Scatter(x=seen.cycle, y=seen.lo, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(96,165,250,.18)", name="90% interval", hoverinfo="skip"), 1, 1)
            fig.add_trace(go.Scatter(x=seen.cycle, y=seen.raw, mode="lines", name="Raw estimate",
                                     line=dict(color=C["blue"], width=1.6)), 1, 1)
            fig.add_trace(go.Scatter(x=seen.cycle, y=seen.ema, mode="lines", name="EMA 0.3",
                                     line=dict(color=C["amber"], width=2)), 1, 1)
            if reveal:
                fig.add_trace(go.Scatter(x=seen.cycle, y=seen.soh_nom, mode="lines", name="Measured SOH",
                                         line=dict(color="#f8fafc", width=1.4, dash="dot")), 1, 1)
            bad = seen[seen.status.isin(["STALE", "INVALID_WINDOW"])]
            if len(bad):
                fig.add_trace(go.Scatter(x=bad.cycle, y=[ylo + 0.4] * len(bad), mode="markers", name="Carried forward",
                                         marker=dict(symbol="line-ns", size=11, color=C["amber"], line=dict(width=1.5, color=C["amber"])),
                                         yaxis="y", hovertemplate="cycle %{x}: stale<extra></extra>"), 1, 1)
            if np.isfinite(raw):
                fig.add_trace(go.Scatter(x=[row.cycle], y=[raw], mode="markers", showlegend=False,
                                         marker=dict(size=11, color=soh_color(raw), line=dict(width=2, color="#fff"))), 1, 1)
            for y, lab in ((85, "85%"), (80, "80% end of life")):
                fig.add_hline(y=y, line=dict(color="rgba(148,163,184,.45)", width=1, dash="dash"), row=1, col=1,
                              annotation_text=lab, annotation_position="bottom right", annotation_font_color=C["muted"])
            for c0, col, lab in ((warn_c, C["amber"], "85% warning"), (eol_c, C["violet"], "EOL watch")):
                if c0 is not None and c0 <= cyc:
                    fig.add_vline(x=c0, line=dict(color=col, width=1.5), row=1, col=1)
                    fig.add_annotation(x=c0, y=1.0, yref="y domain", text=lab, showarrow=False, xanchor="left",
                                       font=dict(color=col, size=11), yanchor="top")
            if reveal:
                life = life_of(cid, df, per)
                if life <= cyc:
                    fig.add_vline(x=life, line=dict(color="#f87171", width=1.2, dash="dot"), row=1, col=1)
            if temp_ok:
                fig.add_trace(go.Scatter(x=seen.cycle, y=seen.temp_delta_c, mode="lines", name="Temp shift (°C)",
                                         line=dict(color=C["orange"], width=1.2)), 2, 1)
                fig.update_yaxes(title_text="°C", row=2, col=1)
            fig.update_yaxes(range=[ylo, 100.5], title_text="SOH (% of 1.1 Ah)", row=1, col=1)
            fig.update_xaxes(range=[cmin, cmax * 1.01], title_text="cycle" if not temp_ok else None, row=1, col=1)
            if temp_ok:
                fig.update_xaxes(range=[cmin, cmax * 1.01], title_text="cycle", row=2, col=1)
            layout(fig, 470 if temp_ok else 430)
            st.plotly_chart(fig, width="stretch", key="mission_chart")

        c1, c2 = st.columns([1.3, 1], gap="medium")
        with c1:
            beats = [n for n in NOTES.get(cid, []) if n[0] <= cyc]
            if beats:
                b = beats[-1]
                st.markdown(f'<div class="beat"><b>Story beat · cycle {b[0]} · {b[1]}</b><br>{b[2]}</div>', unsafe_allow_html=True)
            elif cid in NOTES:
                st.markdown(f'<div class="beat"><b>Something to watch for</b><br>A story beat unlocks at cycle {NOTES[cid][0][0]}.</div>',
                            unsafe_allow_html=True)
        with c2:
            evs = [e for e in events_for(df) if e[0] <= cyc][-5:][::-1]
            html = "".join(f'<div class="tick-row"><i>c{e[0]}</i>{e[2]}</div>' for e in evs) or '<div class="sub">No events yet.</div>'
            st.markdown(f'<div class="card"><h4>Event log</h4>{html}</div>', unsafe_allow_html=True)

    stage()


# --------------------------------------------------------------------------- page: fleet wall
def spark_svg(cyc, raw, clock, xmax, w=210, h=44):
    m = (cyc <= clock) & np.isfinite(raw)
    if m.sum() < 2:
        return f'<svg width="100%" viewBox="0 0 {w} {h}"></svg>'
    x, y = cyc[m] / xmax * w, h - (np.clip(raw[m], 75, 100) - 75) / 25 * h
    step = max(1, len(x) // 160)
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x[::step], y[::step]))
    guides = "".join(f'<line x1="0" x2="{w}" y1="{h - (g - 75) / 25 * h:.1f}" y2="{h - (g - 75) / 25 * h:.1f}" '
                     f'stroke="#475569" stroke-dasharray="3 3" stroke-width=".8"/>' for g in (85, 80))
    col = soh_color(float(raw[m][-1]))
    return f'<svg width="100%" viewBox="0 0 {w} {h}" preserveAspectRatio="none">{guides}<polyline fill="none" stroke="{col}" stroke-width="1.8" points="{pts}"/></svg>'


def page_fleet(data, per):
    cells = list(data)
    ss("fplay", False)
    xmax = max(int(d.cycle.max()) for d in data.values())
    ss("fclock", 400)  # opening view; Restart returns to cycle 30
    c = st.columns([1, 1, 1, 2])
    speed = c[0].select_slider("Speed", list(SPEEDS), value="12×", key="fspeed")
    if c[1].button("⏸ Pause" if st.session_state.fplay else "▶ Play", key="fbtn", width="stretch"):
        st.session_state.fplay = not st.session_state.fplay
        if st.session_state.fplay and st.session_state.fclock >= xmax:
            st.session_state.fclock = 30
        st.rerun()
    if c[2].button("↺ Restart", key="frst", width="stretch"):
        st.session_state.fclock, st.session_state.fplay = 30, False
        st.rerun()
    mode = c[3].radio("Mosaic shows", ["Estimated SOH", "Error vs truth (descriptive)"], horizontal=True)

    @st.fragment(run_every=0.25 if st.session_state.fplay else None)
    def wall():
        if st.session_state.fplay:
            st.session_state.fclock = min(xmax, st.session_state.fclock + SPEEDS[speed])
            if st.session_state.fclock >= xmax:
                st.session_state.fplay = False
                st.rerun(scope="app")
        clock = st.slider("Fleet clock (cycle)", 1, xmax, key="fclock")
        tiles, n_warn, n_eol = [], 0, 0
        for cid in cells:
            d = data[cid]
            cy, rw = d.cycle.values, d.raw.values
            pos = int(np.searchsorted(cy, clock, side="right") - 1)
            ended = clock > cy[-1]
            pos = max(pos, 0)
            r = d.iloc[min(pos, len(d) - 1)]
            valid = np.where(np.isfinite(rw[: pos + 1]))[0]
            v = float(rw[valid[-1]]) if len(valid) else np.nan
            w_on, e_on = bool(d.warn85.iloc[pos]) and clock >= cy[0], bool(d.eol_watch.iloc[pos])
            if ended:
                w_on, e_on = bool(d.warn85.iloc[-1]), bool(d.eol_watch.iloc[-1])
            n_warn += w_on
            n_eol += e_on
            state = "EOL watch" if e_on else "warning" if w_on else "record ended" if ended else "warming up" if not np.isfinite(v) else "healthy"
            if ended and state in ("EOL watch", "warning"):
                state += " · ended"
            cls = "eol" if e_on else "warn" if w_on else ""
            tiles.append(f'<div class="tile {cls} {"ended" if ended else ""}"><div class="t"><b>{cid}</b>{split_tag(cid)}<span style="margin-left:auto">{state}</span></div>'
                         f'<div class="v" style="color:{soh_color(v)}">{fmt(v, 1)}<span style="font-size:.8rem">%</span></div>'
                         f'{spark_svg(cy, rw, clock, xmax)}</div>')
        k = st.columns(4)
        k[0].metric("Cells replayed", len(cells))
        k[1].metric("85% warnings on", int(n_warn))
        k[2].metric("EOL watches on", int(n_eol))
        k[3].metric("Cycle", clock)
        st.markdown('<div class="grid">' + "".join(tiles) + "</div>", unsafe_allow_html=True)
        st.caption("Only batch-3 cells are held-out test cells. Batch-1 and batch-2 cells trained the deployed model, "
                   "so alert counts across the whole fleet are not a held-out result. See the Evidence view for those.")

    wall()

    st.markdown("### Whole-fleet mosaic")
    order = sorted(cells, key=lambda cid: life_of(cid, data[cid], per))
    lab = {cid: (f"{cid} · test" if is_test(cid) else cid) for cid in order}
    z = np.full((len(order), xmax + 1), np.nan)
    for i, cid in enumerate(order):
        d = data[cid]
        vals = d.raw.values - d.soh_nom.values if mode.startswith("Error") else d.raw.values
        z[i, d.cycle.values.astype(int)] = vals
    fig = go.Figure(go.Heatmap(
        z=z, x=np.arange(xmax + 1), y=[lab[c] for c in order], xgap=0, ygap=2,
        colorscale=([[0, C["red"]], [.5, "#0b1220"], [1, C["orange"]]] if mode.startswith("Error")
                    else [[0, "#7f1d1d"], [.35, C["orange"]], [.6, C["amber"]], [1, C["teal"]]]),
        zmin=-3 if mode.startswith("Error") else 78, zmax=3 if mode.startswith("Error") else 100,
        zmid=0 if mode.startswith("Error") else None,
        colorbar=dict(title="pp" if mode.startswith("Error") else "SOH %", thickness=10),
        hovertemplate="%{y} · cycle %{x}<br>%{z:.2f}<extra></extra>"))
    wx, wy, ex, ey = [], [], [], []
    for cid in order:
        for lst_x, lst_y, c0 in ((wx, wy, first_cycle(data[cid], "warn85")), (ex, ey, first_cycle(data[cid], "eol_watch"))):
            if c0 is not None:
                lst_x.append(c0)
                lst_y.append(lab[cid])
    fig.add_trace(go.Scatter(x=wx, y=wy, mode="markers", name="85% warning", marker=dict(symbol="diamond", size=9, color="#fff", line=dict(width=1.5, color=C["amber"]))))
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers", name="EOL watch", marker=dict(symbol="x", size=9, color=C["violet"], line=dict(width=1.5, color="#fff"))))
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(title_text="cycle")
    layout(fig, 120 + 28 * len(order))
    st.plotly_chart(fig, width="stretch", key="mosaic")
    st.caption("Rows are sorted by true life, shortest at the top. In the error view, warm tones mean the estimate is too healthy, "
               "which is the unsafe direction. Rows marked test are held-out; the others are in-sample. Truth is used only to colour this picture, never by the monitor.")


# --------------------------------------------------------------------------- page: evidence
def page_evidence():
    st.markdown("### What was tested, and what held")
    c = st.columns(2)
    c[0].markdown(
        '<div class="verdict"><div class="k">Development · Protocol D · strict gate (15% below best frozen baseline)</div>'
        '<div class="p"><span class="pass">W240 PASS</span> &nbsp; <span class="fail">W300 FAIL</span></div>'
        '<div class="sub">M0 cell-macro MAE 0.530 pp at W240 (target 0.557) and 0.836 pp at W300 (target 0.760). '
        'Stated as written and never relabelled.</div></div>', unsafe_allow_html=True)
    c[1].markdown(
        '<div class="verdict"><div class="k">Locked batch 3 · 40 cells · rule: no worse than best baseline</div>'
        '<div class="p"><span class="pass">W240 PASS</span> &nbsp; <span class="pass">W300 PASS</span></div>'
        '<div class="sub">M0 0.420 vs B4 0.567 pp at W240 (−0.147, CI −0.235 to −0.063). At W300 0.597 vs 0.742 '
        '(−0.145, CI −0.275 to −0.019), but the policy-cluster interval [−0.240, +0.025] includes zero.</div></div>',
        unsafe_allow_html=True)

    st.markdown("#### Batch 3, cell-macro MAE (pp, lower is better)")
    comp = pd.DataFrame({"model": ["B0b cycle count", "B1 elastic net", "B3 one-number window", "B4 random forest",
                                   "B4r residual forest", "M0 (deployed)"],
                         "W300": [2.657, 1.187, 2.075, 0.742, 0.692, 0.597],
                         "W240": [2.657, 1.250, 0.793, 0.567, 0.433, 0.420]})
    fig = go.Figure()
    for w, col in (("W240", C["teal"]), ("W300", C["violet"])):
        fig.add_trace(go.Bar(y=comp.model, x=comp[w], name=w, orientation="h", marker_color=col,
                             text=comp[w].map("{:.3f}".format), textposition="outside"))
    fig.update_layout(barmode="group", xaxis=dict(range=[0, 3.1]), yaxis=dict(autorange="reversed"))
    layout(fig, 360)
    st.plotly_chart(fig, width="stretch", key="comp")
    st.caption("The waveform beats cycle count and the one-number summary with intervals that exclude zero. "
               "It ties the well-built residual forest B4r: intervals include zero at both windows.")

    st.markdown("#### The evidence ladder (cell-macro MAE, pp)")
    ladder = pd.DataFrame([
        ["D reference (≈70 train cells, 6-fold)", 0.836, 0.895, 0.530, 0.655, "−0.059 (n.s.)", "−0.125 (sig)"],
        ["7a locked batch 3 (84 → 40)", 0.597, 0.742, 0.420, 0.567, "−0.145 (sig)", "−0.147 (sig)"],
        ["P primary (41 → 43)", 0.919, 0.955, 0.606, 0.756, "−0.036 (n.s.)", "−0.149 (sig)"],
        ["P secondary = batch 3 (41 → 40)", 0.697, 0.907, 0.403, 0.737, "−0.210 (sig)", "−0.334 (sig)"],
        ["Stress b1 → b2", 1.645, 1.936, 1.134, 1.091, "−0.291 (sig)", "+0.043 (n.s.)"],
        ["Stress b2 → b1", 1.764, 1.449, 1.215, 0.962, "+0.315 (B4 better)", "+0.254 (B4 better)"],
    ], columns=["Setting", "M0 W300", "B4 W300", "M0 W240", "B4 W240", "M0−B4 W300", "M0−B4 W240"])
    num = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(ladder, hide_index=True, width="stretch",
                 column_config={c: num for c in ("M0 W300", "B4 W300", "M0 W240", "B4 W240")})
    st.caption("The edge over the forest is real when train and test life ranges overlap and reverses when a model trained on "
               "short-lived cells is tested on long-lived ones. Stress errors are 2 to 3 times the Protocol D errors.")

    st.markdown("#### Alerts on batch 3 (frozen configuration, raw M0)")
    alerts = pd.DataFrame([
        ["85% warning", "0 of 40 missed", "0%", "median abs. delay 12 cycles (1.18% of life)", "median +1.04% of life (late)"],
        ["End-of-life watch (lower bound)", "7 of 40 missed (17.5%)", "0%", "median lead 9 cycles (0.9% of life)", "all 7 misses are over-estimated cells"],
    ], columns=["Alert (W240, deployed)", "Missed", "Early alarms", "Timing", "Note"])
    st.dataframe(alerts, hide_index=True, width="stretch")

    st.markdown("### Where it fails (read this before trusting a number)")
    limits = {
        "Long-lived cells read too healthy at W300": "Bias grows with cell life (Spearman +0.62 on batch 3). The four cells above 1,434 cycles are over-estimated by 1.6 to 1.95 pp. At W240 there is no such trend.",
        "Most of the error is a per-cell offset": "About 53 to 57% of the squared error is each cell's constant bias. It cannot be removed causally without a capacity anchor, which the capacity-free design excludes.",
        "The end-of-life watch misses over-estimated cells": "A cell that reads too healthy is the cell the lower-bound alert fires too late on. 7 of 40 batch-3 cells were missed at W240.",
        "Strong life-range shift reverses the edge over the forest": "Trained only on short-lived batch 2 and tested on long-lived batch 1, the random forest beats M0 at both windows.",
        "The causal monitor can pick different references than the offline pipeline": "For b1c18 the monitor (v4) picks cycles 41 to 45; the offline pipeline used 54 to 58 because early cycles carry charge/discharge gaps. A tail of larger differences remains even with stored references.",
        "Temperature episodes": "Two episodes (b3c38 around cycle 1321, b1c36 around cycle 490) coincide with temperature changes and with estimate artefacts. M0 has no temperature input. Not calibrated.",
        "One unresolved cycle": "b1c5 cycle 158: online 92.3 pp against a true 98.0. Cause not established.",
        "'Real time' means once per cycle": "The estimate arrives near the end of the discharge: typically 12.8 to 13.6 min in, and about 14 min in early in a cell's life when discharges last longer. This is a replay of lab data at constant 4C discharge, one cell type, one chemistry. No vehicle, variable-load or thermal validation.",
        "Intervals are conservative on batch 3": "Macro coverage 0.980 at W240 against a nominal 0.90, because batch-3 errors were smaller than the development errors used for calibration. Not recalibrated.",
    }
    for k, v in limits.items():
        with st.expander(k):
            st.write(v)


# --------------------------------------------------------------------------- page: under the hood
def page_hood():
    st.markdown("### One cycle, end to end")
    st.markdown(
        '<div class="pipe">'
        '<div><b>1 · Discharge</b>Constant 4 A; samples stream in. Nothing after the estimate instant is ever read.</div>'
        '<div><b>2 · Window</b>Voltage ≥ 2.40 V (grid index 733 of 1000). Running-minimum voltage, interpolate onto the grid.</div>'
        '<div><b>3 · ΔQ(V)</b>Subtract the cell\'s own reference curve (cycles 8 to 12). A 32-point subsample goes in.</div>'
        '<div><b>4 · M0</b>MLP 33→44→32→1 on ΔQ and Q0, five seeds, ONNX Runtime on CPU.</div>'
        '<div><b>5 · Post</b>Raw and EMA 0.3 estimates, 90% conformal interval, one-sided lower bound.</div>'
        '<div><b>6 · Alerts</b>Warning when raw ≤ 85. End-of-life watch when estimate minus lower-bound margin ≤ 80.</div>'
        "</div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Guard rails")
        st.markdown("- **No capacity input.** No current-cycle capacity and no lagged capacity. Early-life capacity Q0 is used.\n"
                    "- **Causal.** A truncated stream gives identical earlier records.\n"
                    "- **Bad window:** the last estimate is carried forward and marked stale. Never repaired, never interpolated from future cycles.\n"
                    "- **Duplicates and out-of-order cycles** are ignored. A late joiner gets no estimate.\n"
                    "- **Reference gates** check discharge current, capacity and charge/discharge consistency before a cycle becomes a reference.\n"
                    "- **Temperature** is an informational field only.")
    with b:
        st.markdown("#### Parity with the offline pipeline")
        st.markdown("- Window rebuilt from raw samples vs stored curve: p99 4.3 mAh.\n"
                    "- Estimate difference, online minus offline: median 0.03 pp, p99 0.17 pp on the 8b replay sample; a few isolated cycles reach several pp.\n"
                    "- Alert fire cycles differ by at most 5 cycles in 27 of 27 pairs that fire on both sides.\n"
                    "- ONNX vs PyTorch: max 1.5e-5 pp.")

    st.markdown("### Speed")
    k = st.columns(4)
    k[0].metric("Per-cell update p99", "1.8 ms", help="Full pipeline, ONNX Runtime, one thread, three passes: 1.79 to 1.83 ms.")
    k[1].metric("124-cell fleet update", "≈5.6 to 6.7 ms", help="Median over three passes, one scorer call.")
    k[2].metric("Targets", "10 ms · 50 ms", help="Proposal targets: per-cell p99 and fleet-batch latency.")
    k[3].metric("Model file", "62 KB", help="M0 W240, fp32, five seeds in one graph.")
    if LATENCY_CSV.exists():
        with st.expander("Latency table from the Phase 8b run"):
            st.dataframe(pd.read_csv(LATENCY_CSV), hide_index=True, width="stretch")

    with st.expander("Run a scorer micro-benchmark on this machine"):
        st.caption("Times only the ONNX call on random inputs of the right shape. It is not the full pipeline and not comparable "
                   "to the figures above.")
        if st.button("Run benchmark"):
            try:
                sys.path.insert(0, str(ROOT))
                from src.onnx_scorer import OnnxScorer
                sc = OnnxScorer(str(MODEL_ONNX), intra_op_threads=1)
                rng = np.random.default_rng(0)
                rows = []
                for n in (1, 124):
                    dq = rng.normal(-0.02, 0.01, (n, 32)).astype(np.float32)
                    q0 = np.full((n, 1), 1.075, np.float32)
                    for _ in range(50):
                        sc.predict(dq, q0)
                    t = []
                    for _ in range(1500):
                        t0 = time.perf_counter()
                        sc.predict(dq, q0)
                        t.append((time.perf_counter() - t0) * 1e3)
                    rows.append({"batch": n, "median ms": round(float(np.median(t)), 3), "p99 ms": round(float(np.percentile(t, 99)), 3)})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not run the benchmark: {e}")

    st.markdown("### Provenance")
    st.dataframe(pd.DataFrame({"artefact": list(HASHES), "SHA-256": list(HASHES.values())}), hide_index=True, width="stretch")
    st.caption("Monitor provenance chain: b2800940… → c3f0a536… → 0aa411d7… → a9531dde…. Batch-3 labels were read once, "
               "after the predictions were hashed. The config and the model files were never edited afterwards.")


# --------------------------------------------------------------------------- main
def main():
    data, per = load_everything()
    st.markdown(
        '<div class="hero"><h1>Cell Observatory</h1>'
        "<p>A capacity-free, causal state-of-health monitor for fast-charging lithium-ion cells. "
        "Everything here is a <b>replay of lab data</b>, one update per cycle.</p>"
        '<div class="chips"><span class="chip">Model <b>M0 @ W240</b></span><span class="chip">Inputs <b>ΔQ(V) + Q0</b></span>'
        '<span class="chip">Trained on <b>84 cells</b></span><span class="chip">Tested on <b>40 locked cells</b></span>'
        '<span class="chip">Runtime <b>ONNX · CPU</b></span></div></div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🔋 Cell Observatory")
        page = st.radio("View", ["Mission control", "Fleet wall", "Evidence", "Under the hood"], label_visibility="collapsed")
        st.divider()
        if data is None:
            st.error(f"No replay files in\n`{REPLAY_DIR}`")
        else:
            st.caption(f"{len(data)} replay cells loaded")
            sha = None
            if VERIFY_JSON.exists():
                try:
                    sha = find_key(json.loads(VERIFY_JSON.read_text(encoding="utf-8")), "health_monitor_sha256")
                except Exception:  # noqa: BLE001
                    sha = None
            if sha is None:
                st.caption("Monitor hash: not recorded in the verification report")
            elif sha == EXPECTED_MONITOR_SHA:
                st.caption(f"Monitor v4 `{sha[:8]}…` ✓ frozen build")
            else:
                st.warning(f"Replay files come from monitor `{sha[:8]}…`, not the frozen v4 build.")
        st.caption("Replay of lab data (MATR, constant 4C discharge). Not a live feed.")

    if page == "Evidence":
        page_evidence()
    elif page == "Under the hood":
        page_hood()
    elif data is None:
        st.info("Run the Phase 8b notebook first; it writes the replay parquet files this view reads.")
    elif page == "Mission control":
        page_mission(data, per)
    else:
        page_fleet(data, per)


main()