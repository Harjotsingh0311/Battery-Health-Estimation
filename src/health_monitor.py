# health_monitor.py : streaming SOH monitor for the MATR capacity-free nowcast (phase 8b, monitor v4).
# Dependencies: numpy, json, time, math, pathlib, and onnxruntime through src.onnx_scorer only.
# This module must never import torch or any src.soh_deep_* module.
import json
import math
import time
from pathlib import Path

import numpy as np

from src.onnx_scorer import OnnxScorer

NAN = float("nan")
VGRID = np.linspace(3.5, 2.0, 1000)          # voltage grid Vdlin (3.5 V down to 2.0 V)
WINDOW_LEN = {"W240": 734, "W300": 334}      # window = grid indices 0..n-1, closes at VGRID[n-1]
START_I = -1.0                               # discharge starts at the first sample with I <= -1.0 A
REF_START = 8                                # references use cycles with cycle_number >= 8
N_REF = 5                                    # number of reference cycles
SCORE_GAP = 8                                # scoring starts at ref_last + 8
REF_JOIN_MAX = 8                             # a cell first seen after this cycle cannot build a reference
REF_DEADLINE = 60                            # references must complete by this cycle number
WARN_PP = 85.0                               # warn85: raw <= 85.0
EOL_PP = 80.0                                # eol_watch: lower_bound <= 80.0
THRESH = {"first_abs": 0.01, "mono_tol": 5e-3, "lo": -0.02, "hi": 1.2, "max_step": 0.05}
REF_QD_RANGE = (0.5, 1.3)                    # Ah, summary discharge capacity
REF_END_TOL = 0.046                     # Ah, raw end capacity vs summary capacity (calibrated on clean cycles)
REF_I_NOM = -4.0                         # A, nominal discharge current (calibrated on clean cycles)
REF_I_TOL = 0.05                         # A, |median discharge current up to tau - nominal|
REF_I_BAND = 0.1                             # A, plateau band around the median current
REF_I_MIN_FRAC = 0.92                # minimum share of samples up to tau inside that band
REF_QC_TOL = 0.012                       # Ah, |summary QDischarge - summary QCharge| of a reference candidate; None = gate off
REF_NS_RANGE = None                   # (min, max) discharge samples of a reference candidate (whole discharge run); None = gate off
DEFAULT_RECIPE = {"dup": "first", "offset": "interp0"}
REC_KEYS = ("raw", "ema", "lo", "hi", "lower_bound")


# ---------------------------------------------------------------- configuration tables
_REQ = {"pred_lo", "pred_hi", "q2_pooled", "ql_pooled"}


def _tokens(path):
    out = set()
    for p in path:
        s = "".join(ch if ch.isalnum() else " " for ch in str(p)).lower()
        out.update(s.split())
    return out


def _walk(obj, path, found):
    if isinstance(obj, dict):
        if _REQ <= set(obj) and all(isinstance(obj[k], list) for k in _REQ):
            found.append((path, {k: list(obj[k]) for k in _REQ}))
            return
        for k, v in obj.items():
            _walk(v, path + [str(k)], found)
    elif isinstance(obj, list):
        if obj and all(isinstance(x, dict) and _REQ <= set(x) for x in obj):
            found.append((path, {k: [x[k] for x in obj] for k in _REQ}))
            return
        for j, v in enumerate(obj):
            _walk(v, path + [str(j)], found)


def _alpha_cands(obj, path, out):
    # every numeric value stored under a key that contains "alpha", with its path
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "alpha" in str(k).lower() and isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append((path + [str(k)], float(v)))
            elif isinstance(v, (dict, list)):
                _alpha_cands(v, path + [str(k)], out)
    elif isinstance(obj, list):
        for j, v in enumerate(obj):
            _alpha_cands(v, path + [str(j)], out)


def _num(x, default):
    return default if x is None else float(x)


def load_tables(config_paths, window, series="raw"):
    # Finds the conformal bin table for (window, series) in one or more JSON config files.
    # Raises ValueError (listing the candidates) if zero or several different tables match.
    if isinstance(config_paths, (str, Path)):
        config_paths = [config_paths]
    found, acands, cfgs = [], [], []
    for cp in config_paths:
        cfg = json.loads(Path(cp).read_text(encoding="utf-8"))
        cfgs.append((Path(cp).name, cfg))
        _walk(cfg, [Path(cp).name], found)
        _alpha_cands(cfg, [Path(cp).name], acands)
    pick = None
    for nm, cfg in cfgs:
        e = cfg.get("ema") if isinstance(cfg, dict) else None
        ch = e.get("chosen") if isinstance(e, dict) else None
        v = ch.get(window) if isinstance(ch, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            pick = ([nm, "ema", "chosen", window], float(v))
            break
    if pick is None:
        for p, v in acands:
            tk = _tokens(p)
            if tk & {"ema", "ewma", "ewm"} and not tk & {"mae", "oof", "dev"}:
                pick = (p, v)
                break
    if pick is None:
        for p, v in acands:
            if abs(v - 0.3) < 1e-12:
                pick = (p, v)
                break
    alpha = 0.3 if pick is None else pick[1]
    alpha_src = "default 0.3 (no EMA alpha found in the config files)" if pick is None else " / ".join(pick[0])
    other = "ema" if series == "raw" else "raw"
    cands = []
    for p, t in found:
        tok = _tokens(p)
        if window.lower() in tok and series in tok and other not in tok:
            cands.append((p, t))
    uniq = []
    for p, t in cands:
        if not any(t == u[1] for u in uniq):
            uniq.append((p, t))
    if len(uniq) != 1:
        listing = [" / ".join(p) for p, _ in (cands if cands else found)]
        raise ValueError("expected exactly one conformal table for %s/%s, found %d distinct; candidates: %s"
                         % (window, series, len(uniq), listing))
    path, t = uniq[0]
    hi = np.array([_num(x, np.inf) for x in t["pred_hi"]], dtype=float)
    lo = np.array([_num(x, -np.inf) for x in t["pred_lo"]], dtype=float)
    order = np.argsort(hi, kind="stable")
    tab = {"lo": lo[order], "hi": hi[order],
           "q2": np.array([float(x) for x in t["q2_pooled"]], dtype=float)[order],
           "ql": np.array([float(x) for x in t["ql_pooled"]], dtype=float)[order],
           "alpha": float(alpha), "alpha_src": alpha_src,
           "alpha_candidates": [(" / ".join(p), v) for p, v in acands], "path": " / ".join(path)}
    if not np.all(np.diff(tab["hi"]) >= 0):
        raise ValueError("pred_hi is not ascending in table " + tab["path"])
    return tab


def bin_index(tab, x, conv="hi_ge"):
    # hi_ge: first bin whose pred_hi >= x ; hi_gt: first bin whose pred_hi > x ; lo: last bin whose pred_lo <= x.
    x = np.asarray(x, dtype=float)
    n = len(tab["hi"])
    if conv == "hi_ge":
        i = np.searchsorted(tab["hi"], x, side="left")
    elif conv == "hi_gt":
        i = np.searchsorted(tab["hi"], x, side="right")
    elif conv == "lo":
        i = np.searchsorted(tab["lo"], x, side="right") - 1
    else:
        raise ValueError("unknown bin convention " + str(conv))
    return np.clip(i, 0, n - 1)


# ---------------------------------------------------------------- window construction
def compute_window(vmin, q, n, recipe=None):
    # vmin: running minimum of V up to tau (non-increasing); q: cumulative discharge capacity (Ah).
    r = recipe or DEFAULT_RECIPE
    vmin = np.asarray(vmin, dtype=float)
    q = np.asarray(q, dtype=float)
    m = np.ones(len(vmin), dtype=bool)
    if r["dup"] == "first":
        m[1:] = vmin[1:] < vmin[:-1]
    elif r["dup"] == "last":
        m[:-1] = vmin[1:] < vmin[:-1]
    xp = vmin[m][::-1]
    fp = q[m][::-1]
    w = np.interp(VGRID[:n], xp, fp)
    off = w[0] if r["offset"] == "interp0" else q[0]
    return w - off


class WindowBuilder:
    # Accepts discharge samples one at a time; emits the window at tau using only samples up to tau.
    def __init__(self, window="W240", recipe=None):
        self.n = WINDOW_LEN[window]
        self.vtau = float(VGRID[self.n - 1])
        self.recipe = dict(recipe or DEFAULT_RECIPE)
        self.reset()

    def reset(self):
        self.started = False
        self.ended = False
        self.done = False
        self.vm = float("inf")
        self._v = []
        self._q = []
        self.window = None
        self.t_start = None
        self.t_tau = None
        self.end_q = None
        self.cur = []
        self.bad = 0
        self.n_pushed = 0
        self.win_s = 0.0
        self.max_uptick = 0.0                      # largest upward voltage step (V) among samples up to and including tau; information only
        self._lastv = None
        self.tsum = 0.0                            # temperature sum and count over the samples up to tau; information only
        self.tn = 0

    def push(self, v, q, i=None, t=None, T=None):
        # returns True exactly once: when tau is reached and the window has been built
        if self.ended:
            return False
        if not self.started:
            if i is not None and not (i <= START_I):
                return False
            self.started = True
            self.t_start = t
        elif i is not None and i > START_I:
            self.ended = True
            return False
        self.n_pushed += 1
        if i is not None and not self.done:
            self.cur.append(i)
        if not (math.isfinite(v) and math.isfinite(q)):
            if not self.done:
                self.bad += 1
            return False
        if not self.done:
            if self._lastv is not None and v - self._lastv > self.max_uptick:
                self.max_uptick = v - self._lastv
            self._lastv = v
            if T is not None and math.isfinite(T):
                self.tsum += T
                self.tn += 1
        self.end_q = q
        if v < self.vm:
            self.vm = v
        if self.done:
            return False
        self._v.append(self.vm)
        self._q.append(q)
        if self.vm <= self.vtau:
            self.done = True
            self.t_tau = t
            t0 = time.perf_counter()
            self.window = compute_window(self._v, self._q, self.n, self.recipe)
            self.win_s = time.perf_counter() - t0
            return True
        return False

    def mean_temp(self):
        # mean temperature (deg C) over the discharge samples up to and including tau; NaN if none were supplied
        return self.tsum / self.tn if self.tn > 0 else NAN

    def process(self, V, Qd, I=None, t=None, stop_at_tau=False, T=None):
        # batch function: same code path (push) as the per-sample interface
        Vl = np.asarray(V, dtype=float).tolist()
        Ql = np.asarray(Qd, dtype=float).tolist()
        n = len(Vl)
        Il = [None] * n if I is None else np.asarray(I, dtype=float).tolist()
        Tl = [None] * n if t is None else np.asarray(t, dtype=float).tolist()
        Cl = [None] * n if T is None else np.asarray(T, dtype=float).tolist()
        push = self.push
        for v, q, i, tt, cc in zip(Vl, Ql, Il, Tl, Cl):
            hit = push(v, q, i, tt, cc)
            if self.ended or (hit and stop_at_tau):
                break
        return self.window


def validate_window(w, n=None):
    # serving-side causal check; never repairs. Returns (ok, reason).
    w = np.asarray(w, dtype=float)
    if w.ndim != 1 or (n is not None and w.size != n):
        return False, "bad shape %s" % (w.shape,)
    if not np.all(np.isfinite(w)):
        return False, "non-finite values"
    if abs(w[0]) > THRESH["first_abs"]:
        return False, "first value %.4f Ah not near zero" % w[0]
    d = np.diff(w)
    if d.min() < -THRESH["mono_tol"]:
        return False, "non-monotone (min step %.4f Ah)" % d.min()
    if w.min() < THRESH["lo"] or w.max() > THRESH["hi"]:
        return False, "value range [%.3f, %.3f] Ah" % (w.min(), w.max())
    if np.abs(d).max() > THRESH["max_step"]:
        return False, "step %.4f Ah between neighbouring grid points" % np.abs(d).max()
    return True, ""


# ---------------------------------------------------------------- per-cell state
class CellState:
    _FIELDS = ("cell_id", "ref_cycles", "ref_q", "ref_last", "q0", "score_start", "no_ref", "first_cycle",
               "last_cycle", "n_ingested", "ema", "last", "stale", "warn85_cycle", "eol_cycle", "temp_ref_vals", "temp_ref")

    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.ref_cycles = []
        self.ref_windows = []
        self.ref_q = []
        self.qref = None
        self.q0 = None
        self.ref_last = None
        self.score_start = None
        self.no_ref = False
        self.first_cycle = None
        self.last_cycle = None
        self.n_ingested = 0
        self.ema = None
        self.last = None
        self.stale = False
        self.warn85_cycle = None
        self.eol_cycle = None
        self.temp_ref_vals = []                    # mean temperature of each accepted reference candidate
        self.temp_ref = None                       # median of those means; None until the references are complete

    def to_dict(self):
        d = {k: getattr(self, k) for k in self._FIELDS}
        d["ref_cycles"] = [int(c) for c in self.ref_cycles]
        d["ref_q"] = [float(x) for x in self.ref_q]
        d["temp_ref_vals"] = [float(x) for x in self.temp_ref_vals]
        d["temp_ref"] = None if self.temp_ref is None else float(self.temp_ref)
        d["ref_windows"] = [[float(x) for x in w] for w in self.ref_windows]
        d["qref"] = None if self.qref is None else [float(x) for x in np.asarray(self.qref, dtype=float)]
        d["last"] = None if self.last is None else {k: float(v) for k, v in self.last.items()}
        return d

    @classmethod
    def from_dict(cls, d):
        st = cls(d["cell_id"])
        for k in cls._FIELDS:
            if k in d:
                setattr(st, k, d[k])
        st.ref_windows = [list(w) for w in d["ref_windows"]]
        st.qref = None if d["qref"] is None else np.asarray(d["qref"], dtype=float)
        return st


# ---------------------------------------------------------------- the monitor
class HealthMonitor:
    def __init__(self, model_dir, config_path, window="W240", variant="M0", grid_path=None, recipe=None,
                 bin_convention="hi_ge", intra_op_threads=1):
        if window not in WINDOW_LEN:
            raise ValueError("window must be one of %s" % list(WINDOW_LEN))
        if variant not in ("M0", "M0idx"):
            raise ValueError("variant must be M0 or M0idx")
        self.window, self.variant = window, variant
        self.experimental = variant != "M0"       # M0idx (log1p cycle input) is experimental
        self.n = WINDOW_LEN[window]
        model_dir = Path(model_dir)
        self.scorer = OnnxScorer(str(model_dir / ("%s_%s.onnx" % (variant, window))), intra_op_threads=intra_op_threads)
        gp = Path(grid_path) if grid_path else model_dir.parents[1] / "phase7a" / "models" / ("M0_%s.npz" % window)
        with np.load(gp, allow_pickle=False) as z:
            self.grid = np.asarray(z["grid"]).astype(np.int64)
        if self.grid.size != 32 or self.grid[0] != 0 or np.any(np.diff(self.grid) <= 0) or self.grid[-1] >= self.n:
            raise ValueError("subsample grid must be 32 increasing indices in 0..%d starting at 0" % (self.n - 1))
        self.tab = load_tables(config_path, window, "raw")
        self.alpha = float(self.tab["alpha"])
        self.recipe = dict(recipe or DEFAULT_RECIPE)
        self.bin_convention = bin_convention
        self.cells = {}
        self._open = {}
        self.orphans = []
        self.last_timing = {}

    # ---- state helpers
    def _get(self, cell_id):
        st = self.cells.get(cell_id)
        if st is None:
            st = CellState(cell_id)
            self.cells[cell_id] = st
        return st

    def state_dict(self, cell_id):
        return self.cells[cell_id].to_dict()

    def load_state(self, d):
        st = CellState.from_dict(d)
        self.cells[st.cell_id] = st
        return st

    def set_reference(self, cell_id, qref, q0, ref_last, ref_cycles=None):
        st = self._get(cell_id)
        st.qref = np.asarray(qref, dtype=float)
        st.q0 = float(q0)
        st.ref_last = int(ref_last)
        st.score_start = int(ref_last) + SCORE_GAP
        st.ref_cycles = [] if ref_cycles is None else [int(c) for c in ref_cycles]
        st.first_cycle = 1
        st.no_ref = False
        return st

    # ---- records
    def _blank(self, st, cycle, status, stale=False, carry=False):
        rec = {"cell_id": st.cell_id, "cycle": int(cycle), "status": status,
               "raw": NAN, "ema": NAN, "lo": NAN, "hi": NAN, "lower_bound": NAN,
               "warn85": st.warn85_cycle is not None, "eol_watch": st.eol_cycle is not None,
               "stale": bool(stale), "latency_ms": 0.0, "minutes_into_discharge": NAN, "max_v_uptick": NAN, "temp_delta_c": NAN}
        if carry and st.last is not None:
            for k in REC_KEYS:
                rec[k] = st.last[k]
        return rec

    def _post(self, st, cycle, raw):
        raw = float(raw)
        st.ema = raw if st.ema is None else self.alpha * raw + (1.0 - self.alpha) * st.ema
        i = int(bin_index(self.tab, raw, self.bin_convention))
        half = float(self.tab["q2"][i])
        lb = raw - float(self.tab["ql"][i])
        if st.warn85_cycle is None and raw <= WARN_PP:
            st.warn85_cycle = int(cycle)
        if st.eol_cycle is None and lb <= EOL_PP:
            st.eol_cycle = int(cycle)
        st.stale = False
        rec = self._blank(st, cycle, "OK")
        rec.update({"raw": raw, "ema": float(st.ema), "lo": raw - half, "hi": raw + half, "lower_bound": lb})
        st.last = {k: rec[k] for k in REC_KEYS}
        return rec

    def _predict(self, X, q0, cycles):
        X = np.ascontiguousarray(X, dtype=np.float32)
        q = np.asarray(q0, dtype=np.float32).reshape(-1)
        if self.variant == "M0idx":
            lc = np.log1p(np.asarray(cycles, dtype=np.float64)).astype(np.float32)
            out = self.scorer.predict(X, q, log_cycle=lc)
        else:
            out = self.scorer.predict(X, q)
        return np.asarray(out, dtype=np.float64).ravel()

    # ---- cycle life cycle: begin / finish
    def _begin(self, cell_id, cycle):
        st = self._get(cell_id)
        cycle = int(cycle)
        if st.last_cycle is not None and cycle <= st.last_cycle:
            status = "DUPLICATE" if cycle == st.last_cycle else "OUT_OF_ORDER"
            rec = self._blank(st, cycle, status, stale=st.stale, carry=True)
            rec["gap_records"] = []
            return {"st": st, "cycle": cycle, "mode": "ignore", "rec": rec, "gap": [], "builder": None,
                    "t0": time.perf_counter(), "done": False}
        gap = []
        if st.first_cycle is None:
            st.first_cycle = cycle
            if cycle > REF_JOIN_MAX:
                st.no_ref = True
        elif st.last_cycle is not None and cycle > st.last_cycle + 1 and st.last is not None:
            for c in range(st.last_cycle + 1, cycle):
                gap.append(self._blank(st, c, "STALE", stale=True, carry=True))
            st.stale = True
        if st.no_ref:
            mode = "noref"
        elif cycle < REF_START:
            mode = "warm"
        elif st.qref is None:
            if cycle > REF_DEADLINE:
                st.no_ref = True
                mode = "noref"
            else:
                mode = "ref"
        elif cycle < st.score_start:
            mode = "warm"
        else:
            mode = "score"
        st.last_cycle = cycle
        st.n_ingested += 1
        b = WindowBuilder(self.window, self.recipe) if mode in ("ref", "score") else None
        return {"st": st, "cycle": cycle, "mode": mode, "rec": None, "gap": gap, "builder": b,
                "t0": time.perf_counter(), "done": False}

    def _ref_ok(self, b, qd_total, qc_total=None):
        if b.window is None:
            return False, "tau not reached"
        if b.bad:
            return False, "%d non-finite samples" % b.bad
        ok, why = validate_window(b.window, self.n)
        if not ok:
            return False, why
        if qd_total is None or not math.isfinite(qd_total) or not (REF_QD_RANGE[0] <= qd_total <= REF_QD_RANGE[1]):
            return False, "summary capacity %s outside %s" % (qd_total, REF_QD_RANGE)
        if b.end_q is None or abs(b.end_q - qd_total) > REF_END_TOL:
            return False, "raw end capacity %s differs from summary %s" % (b.end_q, qd_total)
        if REF_QC_TOL is not None and qc_total is not None and math.isfinite(qc_total):
            if abs(qd_total - qc_total) > REF_QC_TOL:
                return False, "charge/discharge gap %.4f Ah above %.4f Ah (QD %.4f, QC %.4f)" % (abs(qd_total - qc_total), REF_QC_TOL, qd_total, qc_total)
        if REF_NS_RANGE is not None and not (REF_NS_RANGE[0] <= b.n_pushed <= REF_NS_RANGE[1]):
            return False, "discharge sample count %d outside %s" % (b.n_pushed, (REF_NS_RANGE,))
        if b.cur:
            cur = np.asarray(b.cur, dtype=float)
            med = float(np.median(cur))
            if abs(med - REF_I_NOM) > REF_I_TOL:
                return False, "median discharge current %.3f A differs from %.2f A" % (med, REF_I_NOM)
            frac = float(np.mean(np.abs(cur - med) <= REF_I_BAND))
            if frac < REF_I_MIN_FRAC:
                return False, "only %.2f of samples within %.2f A of the median current" % (frac, REF_I_BAND)
        return True, ""

    def _finish_ref(self, ctx, qd_total, qc_total=None):
        st, b, cycle = ctx["st"], ctx["builder"], ctx["cycle"]
        ok, why = self._ref_ok(b, qd_total, qc_total)
        rec = self._blank(st, cycle, "WARMUP")
        if not ok:
            rec["note"] = "reference candidate rejected: " + why
            return rec
        _skipped = []
        if not b.cur:
            _skipped.append("current gate skipped (no current samples supplied)")
        if REF_QC_TOL is not None and (qc_total is None or not math.isfinite(qc_total)):
            _skipped.append("charge gate skipped (no summary charge capacity supplied)")
        if _skipped:
            rec["note"] = "; ".join(_skipped)
        st.ref_cycles.append(int(cycle))
        st.ref_windows.append([float(x) for x in b.window])
        st.ref_q.append(float(qd_total))
        st.temp_ref_vals.append(float(b.mean_temp()))
        if len(st.ref_cycles) == N_REF:
            st.qref = np.median(np.asarray(st.ref_windows, dtype=float), axis=0)
            st.q0 = float(np.median(st.ref_q))
            tv = np.asarray(st.temp_ref_vals, dtype=float)
            st.temp_ref = float(np.median(tv[np.isfinite(tv)])) if np.isfinite(tv).any() else None
            st.ref_last = int(cycle)
            st.score_start = int(cycle) + SCORE_GAP
            st.ref_windows = []
        return rec

    def _finish_score(self, ctx):
        st, b, cycle = ctx["st"], ctx["builder"], ctx["cycle"]
        t_a = time.perf_counter()
        if b.window is None:
            ok, why = False, "tau not reached"
        elif b.bad:
            ok, why = False, "%d non-finite samples before tau" % b.bad
        else:
            ok, why = validate_window(b.window, self.n)
        t_b = time.perf_counter()
        if not ok:
            rec = self._blank(st, cycle, "INVALID_WINDOW", stale=True, carry=True)
            rec["note"] = why
            rec["max_v_uptick"] = float(b.max_uptick)
            st.stale = True
            return rec
        x = (b.window - st.qref)[self.grid].astype(np.float32)
        t_c = time.perf_counter()
        raw = self._predict(x[None, :], [st.q0], [cycle])[0]
        t_d = time.perf_counter()
        rec = self._post(st, cycle, raw)
        rec["max_v_uptick"] = float(b.max_uptick)
        mt = b.mean_temp()
        if st.temp_ref is not None and math.isfinite(mt):
            rec["temp_delta_c"] = float(mt - st.temp_ref)
        t_e = time.perf_counter()
        if b.t_start is not None and b.t_tau is not None:
            rec["minutes_into_discharge"] = float(b.t_tau - b.t_start)
        self.last_timing = {"ingest_s": (t_a - ctx["t0"]) - b.win_s, "window_s": b.win_s + (t_b - t_a),
                            "subsample_s": t_c - t_b, "onnx_s": t_d - t_c, "post_s": t_e - t_d,
                            "n_pushed": b.n_pushed}
        return rec

    def _finish(self, ctx, qd_total=None, qc_total=None):
        st, mode, cycle = ctx["st"], ctx["mode"], ctx["cycle"]
        if mode == "ignore":
            rec = ctx["rec"]
        elif mode == "noref":
            rec = self._blank(st, cycle, "NO_REFERENCE")
        elif mode == "warm":
            rec = self._blank(st, cycle, "WARMUP")
        elif mode == "ref":
            rec = self._finish_ref(ctx, qd_total, qc_total)
        else:
            rec = self._finish_score(ctx)
        rec["gap_records"] = ctx["gap"]
        rec["latency_ms"] = (time.perf_counter() - ctx["t0"]) * 1000.0
        ctx["done"] = True
        return rec

    # ---- public interface
    def ingest_cycle(self, cell_id, cycle_number, V, Qd, I=None, qd_total=None, t=None, qc_total=None, T=None):
        # V, Qd (and I, t) are the samples of one cycle; I=None means the arrays hold the discharge only.
        # qd_total is the cycler-summary discharge capacity of that cycle (Ah), used for reference checks.
        # T (optional) is the temperature (deg C) per sample; it only feeds the informational temp_delta_c field and never a gate or an alert.
        # qc_total is the cycler-summary charge capacity (Ah); reference candidates whose |qd_total - qc_total| exceeds
        # REF_QC_TOL are rejected. If qc_total is None the charge gate is skipped (and noted on an accepted candidate).
        ctx = self._begin(cell_id, cycle_number)
        if ctx["mode"] == "ignore":
            return self._finish(ctx)
        b = ctx["builder"]
        if b is not None:
            b.process(V, Qd, I, t, stop_at_tau=(ctx["mode"] == "score"), T=T)
        return self._finish(ctx, qd_total, qc_total)

    def ingest_sample(self, cell_id, cycle_number, V, Qd, I=None, t=None, T=None):
        # per-sample path. Returns a record at tau (scoring cycles) or at the first sample (non-processing
        # cycles); otherwise None. Reference candidates are finalised by end_cycle(cell_id, qd_total).
        cyc = int(cycle_number)
        ctx = self._open.get(cell_id)
        if ctx is not None and ctx["cycle"] != cyc:
            rec = self.end_cycle(cell_id)
            if rec is not None:
                self.orphans.append(rec)
            ctx = None
        if ctx is None:
            ctx = self._begin(cell_id, cyc)
            self._open[cell_id] = ctx
            if ctx["mode"] in ("ignore", "noref", "warm"):
                return self._finish(ctx)
        if ctx["done"]:
            return None
        hit = ctx["builder"].push(V, Qd, I, t, T)
        if hit and ctx["mode"] == "score":
            return self._finish(ctx)
        return None

    def end_cycle(self, cell_id, qd_total=None, qc_total=None):
        ctx = self._open.pop(cell_id, None)
        if ctx is None or ctx["done"]:
            return None
        return self._finish(ctx, qd_total, qc_total)

    def score_features(self, cell_id, cycle, dq32, q0):
        # post-processing only (scorer, EMA, interval, alerts) from stored features
        st = self._get(cell_id)
        raw = self._predict(np.asarray(dq32, dtype=np.float32)[None, :], [q0], [cycle])[0]
        st.last_cycle = int(cycle)
        st.n_ingested += 1
        return self._post(st, int(cycle), raw)

    def update_fleet(self, items):
        # items: dicts with cell_id, cycle, window (completed window, length n) and optionally q0.
        # One scorer call for all valid cells. Qref comes from each cell's reference state.
        t0 = time.perf_counter()
        recs = [None] * len(items)
        queue = []
        for j, it in enumerate(items):
            ctx = self._begin(it["cell_id"], it["cycle"])
            if ctx["mode"] == "ref":
                ctx["mode"] = "warm"
            if ctx["mode"] != "score":
                recs[j] = self._finish(ctx)
                continue
            st = ctx["st"]
            w = np.asarray(it["window"], dtype=float)
            ok, why = validate_window(w, self.n)
            if not ok:
                rec = self._blank(st, ctx["cycle"], "INVALID_WINDOW", stale=True, carry=True)
                rec["note"] = why
                st.stale = True
                rec["gap_records"] = ctx["gap"]
                ctx["done"] = True
                recs[j] = rec
                continue
            q0 = it.get("q0", st.q0)
            queue.append((j, ctx, (w - st.qref)[self.grid], q0 if q0 is not None else st.q0))
        if queue:
            X = np.stack([x for _, _, x, _ in queue]).astype(np.float32)
            Q = np.array([q for _, _, _, q in queue], dtype=np.float32)
            C = [ctx["cycle"] for _, ctx, _, _ in queue]
            raws = self._predict(X, Q, C)
            for (j, ctx, _, _), raw in zip(queue, raws):
                rec = self._post(ctx["st"], ctx["cycle"], raw)
                rec["gap_records"] = ctx["gap"]
                recs[j] = rec
        dt = (time.perf_counter() - t0) * 1000.0
        for r in recs:
            r["latency_ms"] = dt / max(len(items), 1)
        return recs

    def update_window(self, cell_id, cycle_number, window, q0=None):
        it = {"cell_id": cell_id, "cycle": cycle_number, "window": window}
        if q0 is not None:
            it["q0"] = q0
        return self.update_fleet([it])[0]
