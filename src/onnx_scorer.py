"""ONNX Runtime scorer for the frozen phase-7a M0 SOH models (phase 8a).

Raw inputs only: the ONNX graph contains the five ensemble members, the feature scalers and the ensemble mean.

    scorer = OnnxScorer("outputs/phase8a/models/M0_W240.onnx")
    soh_pp = scorer.predict(dq32, q0)                     # M0
    soh_pp = scorer.predict(dq32, q0, log_cycle)          # M0+idx models (log1p(cycle_number))

dq32 is DeltaQ at sub_idx(L) of the window (float32, N x 32), q0 is Q0 per row, soh_pp is SOH in percentage points.
"""
import hashlib
import re
from pathlib import Path

import numpy as np


class OnnxScorer:
    def __init__(self, model_path, intra_op_threads=1):
        self.path = Path(model_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.path}")
        stem = self.path.stem
        m = re.fullmatch(r"(M0|M0idx)_(W\d+)", stem)
        if m is None:
            raise ValueError(f"cannot read model and window from file name {self.path.name!r}; expected M0_W240.onnx style")
        self.name, self.window = stem, m.group(2)
        self.model = m.group(1)
        self.uses_log_cycle = self.model == "M0idx"
        self._threads = int(intra_op_threads)
        self._sha = None
        self._sess = None

    @property
    def sha256(self):
        if self._sha is None:
            self._sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return self._sha

    @property
    def session(self):
        if self._sess is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = self._threads
            so.inter_op_num_threads = 1
            so.log_severity_level = 3
            sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
            names = [i.name for i in sess.get_inputs()]
            want = ["dq32", "q0"] + (["log_cycle"] if self.uses_log_cycle else [])
            if names != want:
                raise RuntimeError(f"{self.path.name}: graph inputs {names} differ from the expected {want}")
            meta = sess.get_modelmeta().custom_metadata_map
            if meta.get("window") not in (None, self.window):
                raise RuntimeError(f"{self.path.name}: metadata window {meta.get('window')!r} differs from file name window {self.window!r}")
            self._sess = sess
        return self._sess

    def _col(self, name, a, n):
        a = np.asarray(a)
        if a.dtype != np.float32:
            raise TypeError(f"{name} must be float32, got {a.dtype}")
        if a.shape == (n,):
            a = a.reshape(n, 1)
        if a.shape != (n, 1):
            raise ValueError(f"{name} must have shape ({n},) or ({n}, 1), got {a.shape}")
        return self._finite(name, a)

    @staticmethod
    def _finite(name, a):
        bad = ~np.isfinite(a)
        if bad.any():
            rows = np.unique(np.nonzero(bad)[0])
            raise ValueError(f"{name} contains {int(bad.sum())} NaN/inf value(s); first bad row index {int(rows[0])}")
        return np.ascontiguousarray(a)

    def predict(self, dq32, q0, log_cycle=None):
        dq32 = np.asarray(dq32)
        if dq32.dtype != np.float32:
            raise TypeError(f"dq32 must be float32, got {dq32.dtype}")
        if dq32.ndim != 2 or dq32.shape[1] != 32 or dq32.shape[0] < 1:
            raise ValueError(f"dq32 must have shape (N, 32) with N >= 1, got {dq32.shape}")
        n = dq32.shape[0]
        feed = {"dq32": self._finite("dq32", dq32), "q0": self._col("q0", q0, n)}
        if self.uses_log_cycle:
            if log_cycle is None:
                raise ValueError(f"{self.name} needs log_cycle (log1p of the cycle number)")
            feed["log_cycle"] = self._col("log_cycle", log_cycle, n)
        elif log_cycle is not None:
            raise ValueError(f"{self.name} takes no log_cycle input")
        out = self.session.run(["soh_pp"], feed)[0]
        return np.ascontiguousarray(out.reshape(-1), dtype=np.float32)

    def __repr__(self):
        return f"OnnxScorer(name={self.name!r}, window={self.window!r}, sha256={self.sha256[:12]}...)"
