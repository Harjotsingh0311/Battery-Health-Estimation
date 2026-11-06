"""Shared constants and helpers for the battery SOH nowcast project (written by Phase 3)."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
NOMINAL_AH = 1.1
EOL_AH = 0.88
REF_FIRST_CYCLE = 8
REF_N = 5
SCORE_GAP = 8
MIN_SCORE_CYCLE = 20
CANARY_R2 = 0.98
LATE_WINDOW_L = 0.60
V_TOP = 3.5
V_STEP = 0.0015015
N_GRID = 1000
KNOT_VOLTAGES = tuple(round(3.5 - 0.1 * i, 1) for i in range(16))


def get_root(cwd=None):
    root = Path(cwd) if cwd is not None else Path.cwd()
    return root.parent if root.name == "notebooks" else root


def paths(root=None):
    root = get_root(root) if root is None or isinstance(root, (str, Path)) else root
    proc = root / "data/processed"
    return {
        "ROOT": root, "PROC": proc, "SPL": proc / "splits",
        "P2": root / "outputs/phase2", "OUT": root / "outputs/phase3",
        "FIG": root / "outputs/phase3/figures", "SRC": root / "src",
    }


def load_tables(root=None, mmap=True):
    """Return (cycles, qdlin, t_knots, registry); qdlin is memory-mapped float32."""
    p = paths(root)["PROC"]
    cycles = pd.read_parquet(p / "cycles_table.parquet")
    qdlin = np.load(p / "qdlin_f32.npy", mmap_mode="r" if mmap else None)
    t_knots = np.load(p / "t_knots_f32.npy")
    registry = pd.read_csv(p / "cell_registry_final.csv")
    return cycles, qdlin, t_knots, registry


def grid_index(voltage):
    """Grid index of voltage V on the 1000-point 3.5 -> 2.0 V grid (int or int array)."""
    v = np.asarray(voltage, dtype=float)
    idx = np.rint((V_TOP - v) / V_STEP).astype(int)
    if np.any(idx < 0) or np.any(idx > N_GRID - 1):
        raise ValueError(f"voltage outside the 3.5 -> 2.0 V grid: {voltage}")
    return int(idx) if idx.ndim == 0 else idx


def window_slice(open_v, close_v):
    """Slice covering grid points from open_v to close_v inclusive."""
    i_open, i_close = grid_index(open_v), grid_index(close_v)
    if i_close <= i_open:
        raise ValueError("close voltage must be below open voltage")
    return slice(i_open, i_close + 1)


# role in {primary, ablation, oracle, banned, label}; available_at says when the value exists
ALLOWLIST = {
    "dq_window":   {"role": "primary",  "available_at": "window close", "note": "build_delta_q(Qdlin, Qref, i_open, i_close)"},
    "q0":          {"role": "primary",  "available_at": "after reference cycle", "note": "median qd_sum over reference cycles"},
    "qref":        {"role": "primary",  "available_at": "after reference cycle", "note": "median Qdlin over reference cycles"},
    "cycle_number": {"role": "ablation", "available_at": "window close", "note": "index ablation"},
    "ir":          {"role": "ablation", "available_at": "end of charge", "note": "internal resistance; ablation only"},
    "chargetime":  {"role": "ablation", "available_at": "end of charge", "note": "ablation only"},
    "tavg":        {"role": "ablation", "available_at": "end of discharge", "note": "cycle summary; not available at window close"},
    "tmin":        {"role": "ablation", "available_at": "end of discharge", "note": "cycle summary; not available at window close"},
    "tmax":        {"role": "ablation", "available_at": "end of discharge", "note": "cycle summary; not available at window close"},
    "qd_sum":      {"role": "label",    "available_at": "end of discharge", "note": "summary discharge capacity; SOH label source"},
    "qd_end":      {"role": "label",    "available_at": "end of discharge", "note": "end-of-discharge capacity; label cross-check"},
    "soh_nom":     {"role": "label",    "available_at": "end of discharge", "note": "qd_sum / 1.1"},
    "soh_rel":     {"role": "label",    "available_at": "end of discharge", "note": "qd_sum / q0"},
    "soh_nom_repaired": {"role": "label", "available_at": "end of discharge", "note": "offline training use only"},
    "qc_sum":      {"role": "banned",   "available_at": "end of charge", "note": "charge capacity is a near-label proxy (baseline B5)"},
    "qc_end":      {"role": "banned",   "available_at": "end of charge", "note": "charge capacity is a near-label proxy (baseline B5)"},
    "qd_lag1":     {"role": "oracle",   "available_at": "end of previous discharge", "note": "reference baseline only; not capacity-free"},
    "window_capacity": {"role": "oracle", "available_at": "window close", "note": "Qdlin[close]-Qdlin[open]; scalar canary"},
    "cell_id":     {"role": "banned",   "available_at": "any", "note": "identifier"},
    "batch":       {"role": "banned",   "available_at": "any", "note": "batch identity leaks the test split"},
    "policy_norm": {"role": "banned",   "available_at": "any", "note": "grouping key only"},
}


def assert_allowed(columns, allowlist=None, roles=("primary", "ablation")):
    """Raise ValueError on any unknown column or any column whose role is not in `roles`."""
    al = ALLOWLIST if allowlist is None else allowlist
    cols = list(columns)
    unknown = [c for c in cols if c not in al]
    wrong = [c for c in cols if c in al and al[c]["role"] not in roles]
    if unknown or wrong:
        raise ValueError(f"disallowed input columns: unknown={unknown} wrong_role={wrong}")
    return True


def build_delta_q(qdlin_row, qref, i_open, i_close, n_points=None):
    """Window delta-Q: (Q(V) - Q(V_open)) - (Qref(V) - Qref(V_open)) on grid [i_open, i_close].

    Works on a single 1000-vector or an (n, 1000) block. Uses only values at or before the
    window close and the per-cell reference curve. n_points optionally subsamples uniformly.
    """
    q = np.asarray(qdlin_row, dtype=np.float32)[..., i_open:i_close + 1]
    r = np.asarray(qref, dtype=np.float32)[..., i_open:i_close + 1]
    dq = (q - q[..., :1]) - (r - r[..., :1])
    if n_points is not None:
        idx = np.rint(np.linspace(0, dq.shape[-1] - 1, n_points)).astype(int)
        dq = dq[..., idx]
    return dq.astype(np.float32)


def shuffle_labels_within_cell(df, seed, label_col="soh_nom", cell_col="cell_id"):
    """Phase 4 canary helper: permute labels among the rows of each cell (breaks input-label link)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    arr = out[label_col].to_numpy().copy()
    for _, idx in out.groupby(cell_col).indices.items():
        idx = np.asarray(idx)
        arr[idx] = rng.permutation(arr[idx])
    out[label_col] = arr
    return out

def window_valid(qdlin_row, i_open, i_close, max_step_ah=0.05):
    # Serving-time validity of a Qdlin slice [i_open, i_close]; uses no label.
    import numpy as _np
    q = _np.asarray(qdlin_row, dtype=_np.float64)
    if q.ndim != 1 or not (0 <= i_open < i_close < q.size):
        return False
    if (not _np.isfinite(q[0])) or abs(q[0]) > 0.01:
        return False
    s = q[i_open:i_close + 1]
    if not _np.all(_np.isfinite(s)):
        return False
    d = _np.diff(s)
    if d.size and (d.min() < -5e-3 or d.max() > max_step_ah):
        return False
    if s.min() < -0.01 or s.max() > 1.3:
        return False
    return True
