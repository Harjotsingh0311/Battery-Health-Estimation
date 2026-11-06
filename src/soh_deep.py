"""soh_deep: small 1D-CNN (M1) and MLP (M0) for capacity-free SOH nowcasting.

Phase 5a module. Contents:
  * M1, M0            ordinary single models (no pretrained weights anywhere).
  * BatchedM1/M0      G independent members trained at once (grouped conv, per-member BN,
                      per-member linear layers via einsum). Initialised by stacking G ordinary
                      single models, so member g starts identical to the single model of seed g.
  * Ensemble          wrapper with two packings: "batched" or "sequential" (list of singles).
  * fit               parallel multi-model trainer (own rows / weights / seed / head per member).
  * make_train_rows   strided, cell-balanced training-row recipe.

fp32 only. No weights are ever loaded from or saved to disk.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NOMINAL_AH = 1.1


# --------------------------------------------------------------------------- helpers
def _rnd(x: float, lo: int = 2) -> int:
    return max(lo, int(round(x)))


def m1_widths(c: float = 1.0):
    """Channel widths (conv1, conv2, conv3, head) for width multiplier c."""
    return _rnd(8 * c), _rnd(16 * c), _rnd(32 * c), _rnd(16 * c)


def sub_idx(L: int, n: int = 32) -> np.ndarray:
    """n points spread over a window of length L (np.linspace, as the Phase 4 M0/B4 subsample)."""
    return np.round(np.linspace(0, L - 1, n)).astype(np.int64)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# --------------------------------------------------------------------------- single models
class M1(nn.Module):
    """Conv1d(1->a,k9,s2)-BN-ReLU-Conv1d(a->b,k5,s2)-BN-ReLU-Conv1d(b->d,k3,s2)-BN-ReLU-GAP
    -> concat q0 -> Linear(d+1->h)-ReLU-Linear(h->1). 2,977 parameters at c=1."""

    def __init__(self, c: float = 1.0):
        super().__init__()
        a, b, d, h = m1_widths(c)
        self.c1, self.b1 = nn.Conv1d(1, a, 9, 2), nn.BatchNorm1d(a)
        self.c2, self.b2 = nn.Conv1d(a, b, 5, 2), nn.BatchNorm1d(b)
        self.c3, self.b3 = nn.Conv1d(b, d, 3, 2), nn.BatchNorm1d(d)
        self.f1, self.f2 = nn.Linear(d + 1, h), nn.Linear(h, 1)

    def forward(self, x, q):
        """x: (B, L) scaled delta-Q; q: (B,) standardized q0 -> (B,) network output z."""
        h = x.unsqueeze(1)
        h = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(h)))
        h = F.relu(self.b3(self.c3(h)))
        h = torch.cat([h.mean(-1), q.unsqueeze(1)], 1)
        return self.f2(F.relu(self.f1(h))).squeeze(1)


class M0(nn.Module):
    """MLP on a 32-point subsample of delta-Q + q0; hidden (44, 32); 2,969 parameters."""

    def __init__(self, L: int, n_pts: int = 32, hidden=(44, 32)):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(sub_idx(L, n_pts)))
        self.l1 = nn.Linear(n_pts + 1, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], 1)

    def forward(self, x, q):
        h = torch.cat([x[:, self.idx], q.unsqueeze(1)], 1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(h))))).squeeze(1)


# --------------------------------------------------------------------------- batched models
def _cat_param(singles, getter):
    return nn.Parameter(torch.cat([getter(s).detach().clone() for s in singles], 0))


def _lin_w(singles, getter):  # (G, in, out)
    return nn.Parameter(torch.stack([getter(s).detach().t().contiguous() for s in singles], 0))


def _lin_b(singles, getter):  # (G, out)
    return nn.Parameter(torch.stack([getter(s).detach().clone() for s in singles], 0))


def _lin(h, w, b):
    return torch.einsum("gbi,gio->gbo", h, w) + b[:, None, :]


class BatchedM1(nn.Module):
    """G independent M1 members: grouped Conv1d (groups=G), BN over G*C channels (so per-member
    statistics), per-member linear layers as einsum with weights (G, in, out)."""

    def __init__(self, singles):
        super().__init__()
        self.G = len(singles)
        s0 = singles[0]
        self.a, self.b, self.d = s0.c1.out_channels, s0.c2.out_channels, s0.c3.out_channels
        for n in ("c1", "c2", "c3"):
            setattr(self, n + "_weight", _cat_param(singles, lambda s, n=n: getattr(s, n).weight))
            setattr(self, n + "_bias", _cat_param(singles, lambda s, n=n: getattr(s, n).bias))
        for n in ("b1", "b2", "b3"):
            setattr(self, n + "_weight", _cat_param(singles, lambda s, n=n: getattr(s, n).weight))
            setattr(self, n + "_bias", _cat_param(singles, lambda s, n=n: getattr(s, n).bias))
            self.register_buffer(n + "_rm", torch.cat([getattr(s, n).running_mean.clone() for s in singles]))
            self.register_buffer(n + "_rv", torch.cat([getattr(s, n).running_var.clone() for s in singles]))
        for n in ("f1", "f2"):
            setattr(self, n + "_weight", _lin_w(singles, lambda s, n=n: getattr(s, n).weight))
            setattr(self, n + "_bias", _lin_b(singles, lambda s, n=n: getattr(s, n).bias))

    def _bn(self, h, n):
        return F.batch_norm(h, getattr(self, n + "_rm"), getattr(self, n + "_rv"),
                            getattr(self, n + "_weight"), getattr(self, n + "_bias"),
                            self.training, 0.1, 1e-5)

    def forward(self, x, q):
        """x: (G, B, L); q: (G, B) -> (G, B)."""
        G, B, _ = x.shape
        h = x.permute(1, 0, 2).contiguous()
        for n, k in (("1", 0), ("2", 1), ("3", 2)):
            h = F.conv1d(h, getattr(self, "c" + n + "_weight"), getattr(self, "c" + n + "_bias"),
                         stride=2, groups=G)
            h = F.relu(self._bn(h, "b" + n))
        h = h.mean(-1).view(B, G, self.d).permute(1, 0, 2)
        h = torch.cat([h, q.unsqueeze(-1)], -1)
        h = F.relu(_lin(h, self.f1_weight, self.f1_bias))
        return _lin(h, self.f2_weight, self.f2_bias).squeeze(-1)

    def member_state(self, g: int) -> dict:
        """Parameters and BN buffers of member g, named like the single model's state_dict."""
        out = {}
        for n, ch in (("c1", self.a), ("c2", self.b), ("c3", self.d)):
            out[n + ".weight"] = getattr(self, n + "_weight")[g * ch:(g + 1) * ch]
            out[n + ".bias"] = getattr(self, n + "_bias")[g * ch:(g + 1) * ch]
        for n, ch in (("b1", self.a), ("b2", self.b), ("b3", self.d)):
            for k, suf in (("weight", "_weight"), ("bias", "_bias"), ("running_mean", "_rm"), ("running_var", "_rv")):
                out[n + "." + k] = getattr(self, n + suf)[g * ch:(g + 1) * ch]
        for n in ("f1", "f2"):
            out[n + ".weight"] = getattr(self, n + "_weight")[g].t()
            out[n + ".bias"] = getattr(self, n + "_bias")[g]
        return out


class BatchedM0(nn.Module):
    """G independent M0 members (einsum linear layers, weights (G, in, out))."""

    def __init__(self, singles):
        super().__init__()
        self.G = len(singles)
        self.register_buffer("idx", singles[0].idx.clone())
        for n in ("l1", "l2", "l3"):
            setattr(self, n + "_weight", _lin_w(singles, lambda s, n=n: getattr(s, n).weight))
            setattr(self, n + "_bias", _lin_b(singles, lambda s, n=n: getattr(s, n).bias))

    def forward(self, x, q):
        h = torch.cat([x[..., self.idx], q.unsqueeze(-1)], -1)
        h = F.relu(_lin(h, self.l1_weight, self.l1_bias))
        h = F.relu(_lin(h, self.l2_weight, self.l2_bias))
        return _lin(h, self.l3_weight, self.l3_bias).squeeze(-1)

    def member_state(self, g: int) -> dict:
        out = {}
        for n in ("l1", "l2", "l3"):
            out[n + ".weight"] = getattr(self, n + "_weight")[g].t()
            out[n + ".bias"] = getattr(self, n + "_bias")[g]
        return out


# --------------------------------------------------------------------------- ensemble wrapper
class Ensemble:
    """G members behind one interface. packing="batched": one Batched* module.
    packing="sequential": list of ordinary single models run one after another."""

    def __init__(self, kind: str, seeds, L: int, c: float = 1.0, packing: str = "batched", device="cuda"):
        assert kind in ("M1", "M0") and packing in ("batched", "sequential")
        self.kind, self.packing, self.G = kind, packing, len(seeds)
        singles = []
        for s in seeds:  # CPU init so every packing starts from identical weights
            torch.manual_seed(int(s))
            singles.append(M1(c) if kind == "M1" else M0(L))
        if packing == "batched":
            self.net = (BatchedM1 if kind == "M1" else BatchedM0)(singles).to(device)
            self.nets = None
        else:
            self.nets = nn.ModuleList(singles).to(device)
            self.net = None

    def parameters(self):
        return list((self.net if self.net is not None else self.nets).parameters())

    def train(self, mode: bool = True):
        (self.net if self.net is not None else self.nets).train(mode)

    def eval(self):
        self.train(False)

    def __call__(self, x, q):
        if self.net is not None:
            return self.net(x, q)
        return torch.stack([m(x[g], q[g]) for g, m in enumerate(self.nets)], 0)

    def clip_(self, clip: float):
        """Clip every member's gradient norm to `clip` (members stay independent)."""
        if self.net is not None:
            G, ps = self.G, [p for p in self.net.parameters() if p.grad is not None]
            sq = torch.zeros(G, device=ps[0].grad.device)
            for p in ps:
                sq += p.grad.reshape(G, -1).pow(2).sum(1)
            scale = (clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)
            for p in ps:
                p.grad.view(G, -1).mul_(scale[:, None])
        else:
            for m in self.nets:
                torch.nn.utils.clip_grad_norm_(m.parameters(), clip)

    def member_state(self, g: int) -> dict:
        if self.net is not None:
            return self.net.member_state(g)
        return {k: v for k, v in self.nets[g].state_dict().items() if not k.endswith("num_batches_tracked") and k != "idx"}


# --------------------------------------------------------------------------- data + loss
class GData:
    """Tensors on the compute device. X (N, L) raw delta-Q; q0 (N,) Ah; y (N,) SOH in pp; cell (N,)."""

    def __init__(self, X, q0, y, cell, device="cuda"):
        self.X = torch.as_tensor(X, dtype=torch.float32, device=device)
        self.q0 = torch.as_tensor(q0, dtype=torch.float32, device=device)
        self.y = torch.as_tensor(y, dtype=torch.float32, device=device)
        self.cell = torch.as_tensor(cell, dtype=torch.long, device=device)
        self.cell_np = np.asarray(cell)
        self.L = int(self.X.shape[1])


def weighted_huber(pred, y, w, delta: float = 1.0):
    """Per-member cell-balanced weighted Huber. pred, y, w: (G, B) -> (G,)."""
    hub = F.huber_loss(pred, y, reduction="none", delta=delta)
    return (w * hub).sum(1) / w.sum(1).clamp_min(1e-12)


def make_train_rows(rows_by_cell: dict, stride: int, seed: int):
    """Every `stride`-th row per cell with a random per-cell offset; weights 1/(rows of that cell)
    so every cell has total weight 1. rows_by_cell: {cell_int: sorted global row indices}."""
    rows, w = [], []
    for cell, r in rows_by_cell.items():
        r = np.asarray(r)
        off = int(np.random.default_rng([int(seed), int(cell)]).integers(0, stride))
        sel = r[off::stride]
        if len(sel) == 0:
            sel = r[:1]
        rows.append(sel)
        w.append(np.full(len(sel), 1.0 / len(sel)))
    return np.concatenate(rows).astype(np.int64), np.concatenate(w).astype(np.float32)


def cell_macro(err: np.ndarray, cells: np.ndarray):
    """Cell-macro mean of abs errors; returns (macro, {cell: mae})."""
    u, inv = np.unique(cells, return_inverse=True)
    m = np.bincount(inv, weights=err) / np.bincount(inv)
    return float(m.mean()), {int(k): float(v) for k, v in zip(u, m)}


def compute_scalers(X_np, q0_np, y_np, cell_np, rows):
    """Fold-specific scalers from `rows` (all scored rows of the training cells) only.
    s: one scalar std of delta-Q (no per-point centering); q0: mean/sd over training cells;
    y: mean/sd over rows (pp)."""
    X, y = X_np[rows], y_np[rows]
    u, first = np.unique(cell_np[rows], return_index=True)
    q = q0_np[rows][first]
    return dict(s=float(X.std(dtype=np.float64)), q_mean=float(q.mean()), q_sd=float(q.std()),
                y_mean=float(y.mean()), y_sd=float(y.std()))


# --------------------------------------------------------------------------- trainer
def _member_tensors(members, dev):
    G = len(members)
    n = [len(m["train_rows"]) for m in members]
    maxn = max(n)
    TR = torch.zeros(G, maxn, dtype=torch.long)
    TW = torch.zeros(G, maxn)
    for g, m in enumerate(members):
        TR[g, :n[g]] = torch.as_tensor(m["train_rows"])
        TR[g, n[g]:] = int(m["train_rows"][0])
        TW[g, :n[g]] = torch.as_tensor(m["train_w"])
    f = lambda k: torch.tensor([m["scale"][k] for m in members], dtype=torch.float32, device=dev)
    sc = dict(s=f("s"), qm=f("q_mean"), qs=f("q_sd"), ym=f("y_mean"), ys=f("y_sd"))
    sc["res"] = torch.tensor([m["head"] == "res" for m in members], device=dev)
    return TR.to(dev), TW.to(dev), torch.tensor(n, device=dev), sc


def _base(gd, rows, sc, nominal):
    """Head offset in pp: abs -> train mean of y; res -> 100*q0/nominal (per row)."""
    return torch.where(sc["res"][:, None], 100.0 * gd.q0[rows] / nominal, sc["ym"][:, None].expand_as(rows).float())


def _inputs(gd, rows, sc):
    x = gd.X[rows] / sc["s"][:, None, None]
    q = (gd.q0[rows] - sc["qm"][:, None]) / sc["qs"][:, None]
    return x, q


class _EvalSet:
    def __init__(self, rows_list, gd, dev):
        self.G = len(rows_list)
        self.lens = [len(r) for r in rows_list]
        maxv = max(self.lens)
        R = torch.zeros(self.G, maxv, dtype=torch.long)
        for g, r in enumerate(rows_list):
            R[g, :len(r)] = torch.as_tensor(np.asarray(r))
        self.R = R.to(dev)
        self.rows = [np.asarray(r) for r in rows_list]
        self.cells = [gd.cell_np[r] for r in self.rows]


@torch.no_grad()
def predict(ens: Ensemble, gd: GData, es: _EvalSet, sc, nominal=NOMINAL_AH, chunk=2048):
    """Predictions in pp for each member's rows; returns list of numpy arrays."""
    ens.eval()
    out = torch.empty(es.G, es.R.shape[1], device=es.R.device)
    for st in range(0, es.R.shape[1], chunk):
        rows = es.R[:, st:st + chunk]
        x, q = _inputs(gd, rows, sc)
        z = ens(x, q)
        out[:, st:st + chunk] = _base(gd, rows, sc, nominal) + sc["ys"][:, None] * z
    out = out.cpu().numpy()
    return [out[g, :es.lens[g]] for g in range(es.G)]


def _score(preds, es, y_np):
    macro, per_cell = [], []
    for g in range(es.G):
        e = np.abs(preds[g] - y_np[es.rows[g]])
        m, d = cell_macro(e, es.cells[g])
        macro.append(m)
        per_cell.append(d)
    return np.array(macro), per_cell


def fit(kind, gd: GData, members, E: int, lr: float, *, c: float = 1.0, packing: str = "batched",
        ckpts=None, batch: int = 256, clip: float = 1.0, wd: float = 1e-4, delta: float = 1.0,
        noise: float = 0.02, seed: int = 42, nominal: float = NOMINAL_AH, return_preds: bool = False,
        device="cuda", eval_train: bool = True):
    """Train G independent members at once.

    members: list of dicts with train_rows, train_w, val_rows, seed, head ("abs"|"res"), scale
    (s, q_mean, q_sd, y_mean, y_sd). Each iteration every member draws `batch` rows from its own
    training rows (uniformly; loss weights are the cell-balanced row weights). One epoch is
    ceil(median training rows / batch) iterations. AdamW, cosine lr to 0 per iteration, Gaussian noise
    `noise` on the scaled input in training only, per-member gradient clipping, no early stopping.
    Evaluation at `ckpts` (epoch numbers) on ALL rows of each member's validation cells.
    """
    dev = torch.device(device)
    G = len(members)
    ens = Ensemble(kind, [m["seed"] for m in members], gd.L, c=c, packing=packing, device=dev)
    TR, TW, n_t, sc = _member_tensors(members, dev)
    ipe = int(math.ceil(float(np.median([len(m["train_rows"]) for m in members])) / batch))
    T = E * ipe
    ckpts = sorted(set(ckpts if ckpts is not None else [E]))
    opt = torch.optim.AdamW(ens.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed) * 7919 + int(sum(m["seed"] for m in members)) * 31 + G)
    es_val = _EvalSet([m["val_rows"] for m in members], gd, dev) if ckpts else None
    es_tr = _EvalSet([m["train_rows"] for m in members], gd, dev) if (ckpts and eval_train) else None
    y_np = gd.y.cpu().numpy()
    log, preds = [], None
    for it in range(T):
        cur = 0.5 * lr * (1.0 + math.cos(math.pi * it / T))
        for pg in opt.param_groups:
            pg["lr"] = cur
        ens.train()
        pos = (torch.rand(G, batch, device=dev, generator=gen) * n_t[:, None]).long().minimum(n_t[:, None] - 1)
        rows, w = TR.gather(1, pos), TW.gather(1, pos)
        x, q = _inputs(gd, rows, sc)
        if noise > 0:
            x = x + noise * torch.randn(x.shape, device=dev, generator=gen)
        z = ens(x, q)
        pred = _base(gd, rows, sc, nominal) + sc["ys"][:, None] * z
        loss = weighted_huber(pred, gd.y[rows], w, delta).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        ens.clip_(clip)
        opt.step()
        done = (it + 1) // ipe if (it + 1) % ipe == 0 else None
        if done is not None and done in ckpts:
            pv = predict(ens, gd, es_val, sc, nominal)
            vm, vc = _score(pv, es_val, y_np)
            rec = dict(epoch=done, val_mae=vm, val_cell=vc)
            if es_tr is not None:
                rec["train_mae"], _ = _score(predict(ens, gd, es_tr, sc, nominal), es_tr, y_np)
            log.append(rec)
            preds = pv
    out = dict(log=log, ipe=ipe, iters=T, final=log[-1] if log else None)
    if return_preds:
        out["preds"] = preds
    out["ens"] = ens
    return out
