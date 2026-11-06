"""soh_deep_v2: deep models and batched trainer for Phase 5a2.

OWNED by notebook 05a2_deep_pilot_fix.ipynb. The notebook overwrites this file
whenever its SHA-256 differs from the embedded source. Do not edit by hand.
"""
import math
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODULE_HASH_OK = True
KINDS = ("M0", "M1", "M1p")


def self_hash():
    """SHA-256 of the source file this module was loaded from."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sub_idx(L):
    """32 grid points used by M0 and by the tree comparators."""
    return np.round(np.linspace(0, L - 1, 32)).astype(np.int64)


# ---------------------------------------------------------------- single models
class SingleM0(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(sub_idx(L)), persistent=False)
        self.fc1 = nn.Linear(33, 44)
        self.fc2 = nn.Linear(44, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, d, q):
        x = torch.cat([d[:, self.idx], q[:, None]], 1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h).squeeze(1)


class SingleM1(nn.Module):
    def __init__(self, L, pos=False):
        super().__init__()
        self.pos = bool(pos)
        c = 2 if pos else 1
        self.conv1 = nn.Conv1d(c, 8, 9, 2, 4)
        self.bn1 = nn.BatchNorm1d(8)
        self.conv2 = nn.Conv1d(8, 16, 5, 2, 2)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 32, 3, 2, 1)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc1 = nn.Linear(33, 16)
        self.fc2 = nn.Linear(16, 1)
        if pos:
            self.register_buffer("posch", torch.linspace(-1, 1, L), persistent=False)

    def forward(self, d, q):
        x = d[:, None, :]
        if self.pos:
            p = self.posch.view(1, 1, -1).expand(d.shape[0], 1, -1)
            x = torch.cat([x, p], 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(-1)
        h = torch.cat([x, q[:, None]], 1)
        h = F.relu(self.fc1(h))
        return self.fc2(h).squeeze(1)


def make_single(kind, L, seed, dtype=torch.float32):
    torch.manual_seed(int(seed))
    if kind == "M0":
        m = SingleM0(L)
    elif kind == "M1":
        m = SingleM1(L, False)
    elif kind == "M1p":
        m = SingleM1(L, True)
    else:
        raise ValueError("unknown model kind %r" % (kind,))
    return m.to(dtype)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# -------------------------------------------------------------- batched models
class BatchedM0(nn.Module):
    """G independent M0 members; per-member linear layers via einsum."""

    def __init__(self, singles, L):
        super().__init__()
        self.G = len(singles)
        self.register_buffer("idx", torch.as_tensor(sub_idx(L)), persistent=False)

        def W(name):
            return nn.Parameter(torch.stack(
                [getattr(s, name).weight.detach().t().clone() for s in singles]))

        def b(name):
            return nn.Parameter(torch.stack(
                [getattr(s, name).bias.detach().clone() for s in singles]))

        self.W1, self.b1 = W("fc1"), b("fc1")
        self.W2, self.b2 = W("fc2"), b("fc2")
        self.W3, self.b3 = W("fc3"), b("fc3")

    def forward(self, d, q):
        x = torch.cat([d[:, :, self.idx], q[:, :, None]], 2)
        h = F.relu(torch.einsum("gbi,gio->gbo", x, self.W1) + self.b1[:, None, :])
        h = F.relu(torch.einsum("gbi,gio->gbo", h, self.W2) + self.b2[:, None, :])
        return (torch.einsum("gbi,gio->gbo", h, self.W3) + self.b3[:, None, :]).squeeze(2)

    def member_params(self, g):
        return [self.W1[g].t(), self.b1[g], self.W2[g].t(), self.b2[g],
                self.W3[g].t(), self.b3[g]]


class BatchedM1(nn.Module):
    """G independent M1 / M1p members: grouped Conv1d, per-member BN channels."""

    def __init__(self, singles, L, pos=False):
        super().__init__()
        G = len(singles)
        self.G, self.pos = G, bool(pos)
        c = 2 if pos else 1
        dt = next(singles[0].parameters()).dtype
        self.conv1 = nn.Conv1d(G * c, G * 8, 9, 2, 4, groups=G)
        self.bn1 = nn.BatchNorm1d(G * 8)
        self.conv2 = nn.Conv1d(G * 8, G * 16, 5, 2, 2, groups=G)
        self.bn2 = nn.BatchNorm1d(G * 16)
        self.conv3 = nn.Conv1d(G * 16, G * 32, 3, 2, 1, groups=G)
        self.bn3 = nn.BatchNorm1d(G * 32)
        self.W1 = nn.Parameter(torch.zeros(G, 33, 16))
        self.b1 = nn.Parameter(torch.zeros(G, 16))
        self.W2 = nn.Parameter(torch.zeros(G, 16, 1))
        self.b2 = nn.Parameter(torch.zeros(G, 1))
        if pos:
            self.register_buffer("posch", torch.linspace(-1, 1, L), persistent=False)
        self.to(dt)
        with torch.no_grad():
            for g, s in enumerate(singles):
                for i in ("1", "2", "3"):
                    conv, bn = getattr(self, "conv" + i), getattr(self, "bn" + i)
                    sc, sb = getattr(s, "conv" + i), getattr(s, "bn" + i)
                    o = conv.out_channels // G
                    sl = slice(g * o, (g + 1) * o)
                    conv.weight[sl] = sc.weight
                    conv.bias[sl] = sc.bias
                    bn.weight[sl] = sb.weight
                    bn.bias[sl] = sb.bias
                self.W1[g] = s.fc1.weight.t()
                self.b1[g] = s.fc1.bias
                self.W2[g] = s.fc2.weight.t()
                self.b2[g] = s.fc2.bias

    def forward(self, d, q):
        G, B, L = d.shape
        if self.pos:
            p = self.posch.view(1, 1, L).expand(G, B, L)
            x = torch.stack([d, p], 2).permute(1, 0, 2, 3).reshape(B, G * 2, L)
        else:
            x = d.permute(1, 0, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.mean(-1).reshape(B, G, 32).permute(1, 0, 2)
        h = torch.cat([x, q[:, :, None]], 2)
        h = F.relu(torch.einsum("gbi,gio->gbo", h, self.W1) + self.b1[:, None, :])
        return (torch.einsum("gbi,gio->gbo", h, self.W2) + self.b2[:, None, :]).squeeze(2)

    def member_params(self, g):
        out = []
        for i in ("1", "2", "3"):
            conv, bn = getattr(self, "conv" + i), getattr(self, "bn" + i)
            o = conv.out_channels // self.G
            sl = slice(g * o, (g + 1) * o)
            out += [conv.weight[sl], conv.bias[sl], bn.weight[sl], bn.bias[sl]]
        out += [self.W1[g].t(), self.b1[g], self.W2[g].t(), self.b2[g]]
        return out


def make_batched(kind, L, singles):
    if kind == "M0":
        return BatchedM0(singles, L)
    if kind == "M1":
        return BatchedM1(singles, L, False)
    if kind == "M1p":
        return BatchedM1(singles, L, True)
    raise ValueError("unknown model kind %r" % (kind,))


# ----------------------------------------------------------------- scaling etc.
def fit_scalers(X, q0, y, rows_list, head, nominal, device, dtype):
    """Per-member scalers fitted on that member's training rows only."""
    s, qm, qs, ym, ys = [], [], [], [], []
    for r in rows_list:
        rt = torch.as_tensor(np.asarray(r), device=device)
        s.append(float(X[rt].double().std()))
        qv = q0[rt].double()
        qm.append(float(qv.mean()))
        qs.append(float(qv.std()))
        if head == "abs":
            yv = y[rt].double()
            ym.append(float(yv.mean()))
            ys.append(float(yv.std()))
        elif head == "res":
            yv = (y[rt] - 100.0 * q0[rt] / nominal).double()
            ym.append(0.0)
            ys.append(float(yv.std()))
        else:
            raise ValueError("unknown head %r" % (head,))
    out = {}
    for name, v in (("s", s), ("qs", qs), ("ys", ys)):
        out[name] = torch.tensor([x if x > 1e-12 else 1.0 for x in v],
                                 dtype=dtype, device=device)
    out["qm"] = torch.tensor(qm, dtype=dtype, device=device)
    out["ym"] = torch.tensor(ym, dtype=dtype, device=device)
    return out


def make_inputs(X, q0, idx, sc):
    d = X[idx] / sc["s"][:, None, None]
    q = (q0[idx] - sc["qm"][:, None]) / sc["qs"][:, None]
    return d, q


def base_pp(q0, idx, sc, head, nominal):
    if head == "abs":
        return sc["ym"][:, None].expand(idx.shape)
    return 100.0 * q0[idx] / nominal


def lr_at(it, total, warm, lr):
    if it < warm:
        return lr * (it + 1) / warm
    prog = (it - warm) / max(1, total - warm)
    return 0.5 * lr * (1.0 + math.cos(math.pi * min(1.0, prog)))


def clip_per_member(params, G, clip):
    sq = None
    for p in params:
        if not p.grad.is_contiguous():
            p.grad = p.grad.contiguous()
        s = p.grad.view(G, -1).pow(2).sum(1)
        sq = s if sq is None else sq + s
    coef = (clip / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        p.grad.view(G, -1).mul_(coef[:, None])


def epoch_positions(rng, n, need):
    parts, tot = [], 0
    while tot < need:
        parts.append(rng.permutation(n))
        tot += n
    return np.concatenate(parts)[:need]


def cell_macro_mae(pred, yv, codes):
    """Mean over cells of the per-cell mean absolute error. Returns (macro, per_cell)."""
    err = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(yv, dtype=np.float64))
    k = int(codes.max()) + 1
    s = np.bincount(codes, weights=err, minlength=k)
    c = np.bincount(codes, minlength=k)
    ok = c > 0
    per = s[ok] / c[ok]
    return float(per.mean()), per


@torch.no_grad()
def predict(model, X, q0, rows_list, sc, head, nominal, device, dtype, chunk=1024):
    """Eval-mode predictions in pp for every member; rows_list may differ in length."""
    model.eval()
    G = len(rows_list)
    n = [len(r) for r in rows_list]
    M = max(n)
    pad = np.zeros((G, M), dtype=np.int64)
    for g, r in enumerate(rows_list):
        if n[g]:
            pad[g, :n[g]] = r
            pad[g, n[g]:] = r[0]
    pt = torch.from_numpy(pad).to(device)
    out = torch.empty((G, M), dtype=dtype, device=device)
    for a in range(0, M, chunk):
        idx = pt[:, a:a + chunk]
        d, q = make_inputs(X, q0, idx, sc)
        z = model(d, q)
        out[:, a:a + chunk] = base_pp(q0, idx, sc, head, nominal) + sc["ys"][:, None] * z
    out = out.cpu().numpy()
    return [out[g, :n[g]] for g in range(G)]


# --------------------------------------------------------------------- trainer
def train_members(X, q0, y, members, kind, head, lr, epochs, cfg, device,
                  dtype=torch.float32, log_every=10):
    """Train G members at once. members: dicts with seed, train_rows, train_w,
    val_rows, val_codes. X (N,L), q0 (N,), y (N,) are tensors on `device`."""
    t0 = time.perf_counter()
    G, L = len(members), X.shape[1]
    B, nominal, sigma = cfg["BATCH"], cfg["NOMINAL_AH"], cfg["NOISE_SIGMA"]
    singles = [make_single(kind, L, m["seed"], dtype) for m in members]
    model = make_batched(kind, L, singles).to(device)
    del singles
    train_rows = [np.asarray(m["train_rows"], dtype=np.int64) for m in members]
    val_rows = [np.asarray(m["val_rows"], dtype=np.int64) for m in members]
    sc = fit_scalers(X, q0, y, train_rows, head, nominal, device, dtype)
    n = [len(r) for r in train_rows]
    steps = int(math.ceil(max(n) / B))
    total = epochs * steps
    warm = max(1, int(round(cfg["WARMUP_FRAC"] * total)))
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["WD"])
    gen = torch.Generator(device=device)
    gen.manual_seed(int(members[0]["seed"]) * 7919 + G)
    rngs = [np.random.default_rng([int(m["seed"]), 991]) for m in members]
    y_cpu = y.detach().cpu().numpy().astype(np.float64)
    ep_pts, val_curve, loss_curve = [], [], []
    it = 0
    for ep in range(1, epochs + 1):
        model.train()
        rows_ep = np.empty((G, steps * B), dtype=np.int64)
        w_ep = np.empty((G, steps * B), dtype=np.float64)
        for g, m in enumerate(members):
            pos = epoch_positions(rngs[g], n[g], steps * B)
            rows_ep[g] = train_rows[g][pos]
            w_ep[g] = np.asarray(m["train_w"], dtype=np.float64)[pos]
        rows_t = torch.from_numpy(rows_ep).to(device)
        w_t = torch.from_numpy(w_ep).to(device=device, dtype=dtype)
        loss_acc = torch.zeros(G, device=device, dtype=dtype)
        for s in range(steps):
            lr_now = lr_at(it, total, warm, lr)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            idx = rows_t[:, s * B:(s + 1) * B]
            w = w_t[:, s * B:(s + 1) * B]
            d, q = make_inputs(X, q0, idx, sc)
            if sigma > 0:
                d = d + sigma * torch.randn(d.shape, device=device, dtype=dtype, generator=gen)
            z = model(d, q)
            pred = base_pp(q0, idx, sc, head, nominal) + sc["ys"][:, None] * z
            l = F.huber_loss(pred, y[idx], delta=cfg["HUBER_DELTA"], reduction="none")
            lg = (w * l).sum(1) / w.sum(1)
            opt.zero_grad(set_to_none=True)
            lg.sum().backward()
            clip_per_member(params, G, cfg["CLIP"])
            opt.step()
            loss_acc += lg.detach()
            it += 1
        if ep % log_every == 0 or ep == epochs:
            vp = predict(model, X, q0, val_rows, sc, head, nominal, device, dtype)
            vm = [cell_macro_mae(vp[g], y_cpu[val_rows[g]], members[g]["val_codes"])[0]
                  for g in range(G)]
            ep_pts.append(ep)
            val_curve.append(vm)
            loss_curve.append((loss_acc / steps).cpu().numpy().astype(np.float64).tolist())
    vp = predict(model, X, q0, val_rows, sc, head, nominal, device, dtype)
    tp = predict(model, X, q0, train_rows, sc, head, nominal, device, dtype)
    train_mae = []
    for g in range(G):
        w = np.asarray(members[g]["train_w"], dtype=np.float64)
        train_mae.append(float((w * np.abs(tp[g].astype(np.float64) - y_cpu[train_rows[g]])).sum() / w.sum()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return dict(
        val_pred=[v.astype(np.float32) for v in vp],
        train_mae=np.array(train_mae),
        curve=dict(epochs=ep_pts, val_mae=np.array(val_curve), train_loss=np.array(loss_curve)),
        seconds=time.perf_counter() - t0,
        steps_per_epoch=steps,
        n_members=G,
        nan_members=[g for g in range(G) if not np.isfinite(vp[g]).all()],
    )


# ---------------------------------------------------------------- verification
def parity_check(kind, X, q0, y, rows_list, seeds, head, cfg, device, dtype,
                 steps=20, lr=2e-3, B=64):
    """Batched members vs ordinary single models: same init, same 20 steps."""
    G, L = len(seeds), X.shape[1]
    nominal = cfg["NOMINAL_AH"]
    sing_a = [make_single(kind, L, s, dtype) for s in seeds]
    sing_b = [make_single(kind, L, s, dtype).to(device) for s in seeds]
    bm = make_batched(kind, L, sing_a).to(device)
    sc = fit_scalers(X, q0, y, rows_list, head, nominal, device, dtype)
    rng = np.random.default_rng(0)
    idx_all = np.stack([np.stack([rng.choice(np.asarray(r), B) for r in rows_list])
                        for _ in range(steps)])
    w_all = rng.uniform(0.5, 1.5, (steps, G, B))
    opt_b = torch.optim.AdamW(bm.parameters(), lr=lr, weight_decay=cfg["WD"])
    opts = [torch.optim.AdamW(s.parameters(), lr=lr, weight_decay=cfg["WD"]) for s in sing_b]
    params = list(bm.parameters())
    bm.train()
    for s in sing_b:
        s.train()
    for t in range(steps):
        idx = torch.from_numpy(idx_all[t]).to(device)
        w = torch.from_numpy(w_all[t]).to(device=device, dtype=dtype)
        d, q = make_inputs(X, q0, idx, sc)
        pred = base_pp(q0, idx, sc, head, nominal) + sc["ys"][:, None] * bm(d, q)
        l = F.huber_loss(pred, y[idx], delta=cfg["HUBER_DELTA"], reduction="none")
        lg = (w * l).sum(1) / w.sum(1)
        opt_b.zero_grad(set_to_none=True)
        lg.sum().backward()
        clip_per_member(params, G, cfg["CLIP"])
        opt_b.step()
        for g in range(G):
            ig, wg = idx[g], w[g]
            dg = X[ig] / sc["s"][g]
            qg = (q0[ig] - sc["qm"][g]) / sc["qs"][g]
            bg = sc["ym"][g] if head == "abs" else 100.0 * q0[ig] / nominal
            pg = bg + sc["ys"][g] * sing_b[g](dg, qg)
            lo = F.huber_loss(pg, y[ig], delta=cfg["HUBER_DELTA"], reduction="none")
            loss = (wg * lo).sum() / wg.sum()
            opts[g].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sing_b[g].parameters(), cfg["CLIP"])
            opts[g].step()
    max_p = 0.0
    for g in range(G):
        for a, b in zip(bm.member_params(g), sing_b[g].parameters()):
            max_p = max(max_p, float((a.detach() - b.detach()).abs().max()))
    ev_rows = [np.asarray(r)[:200] for r in rows_list]
    pb = predict(bm, X, q0, ev_rows, sc, head, nominal, device, dtype)
    max_e = 0.0
    with torch.no_grad():
        for g in range(G):
            sing_b[g].eval()
            ig = torch.as_tensor(ev_rows[g], device=device)
            dg = X[ig] / sc["s"][g]
            qg = (q0[ig] - sc["qm"][g]) / sc["qs"][g]
            bg = sc["ym"][g] if head == "abs" else 100.0 * q0[ig] / nominal
            ps = (bg + sc["ys"][g] * sing_b[g](dg, qg)).cpu().numpy()
            max_e = max(max_e, float(np.abs(ps - pb[g]).max()))
    return dict(kind=kind, dtype=str(dtype), max_param_diff=max_p, max_eval_diff=max_e)


def same_seed_identical(kind, L, seed):
    a = make_single(kind, L, seed)
    b = make_single(kind, L, seed)
    return all(torch.equal(p, r) for p, r in zip(a.parameters(), b.parameters()))
