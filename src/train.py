"""Train the cross-probe JEPA on simulated probe rollouts and save behavior embeddings for every asset.

Hyperparameters are frozen at the values chosen on the original 276-asset development set; nothing here is tuned on
the held-out probes or on the test assets. Two training sets are produced per seed:
  all  - self-supervised on the training probes of every asset (no labels, held-out probes never seen)
  dev  - trained on the 276 development assets only, then used to embed every asset (inductive check)
"""
import argparse
import copy
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STEPS, BATCH, K_CTX, LR, WD = 4000, 96, 4, 1e-3, 0.05
W_VAR, W_COV, EMA = 1.0, 0.04, 0.996
D_PROBE, D_STATE = 128, 64


def load_rollouts(sim_dir, names=None):
    paths = sorted(p for p in glob.glob(f"{sim_dir}/*.npz"))
    if names is not None:
        keep = set(names)
        paths = [p for p in paths if os.path.basename(p)[:-4] in keep]
    out_names, train = [], []
    for p in paths:
        out_names.append(os.path.basename(p)[:-4])
        train.append(np.load(p)["train"])
    return out_names, np.stack(train).astype(np.float32)  # A, P, R, T, C


class Enc(nn.Module):
    def __init__(self, c, n_probe, d=D_PROBE):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c, 64, 5, 2, 2), nn.GELU(), nn.Conv1d(64, 128, 5, 2, 2), nn.GELU(),
            nn.Conv1d(128, 128, 5, 2, 2), nn.GELU(), nn.Conv1d(128, 128, 3, 1, 1), nn.GELU())
        self.probe = nn.Embedding(n_probe, 256)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d))

    def forward(self, x, pid):
        h = self.conv(x.transpose(1, 2))
        h = torch.cat([h.mean(-1), h.amax(-1)], -1) + self.probe(pid)
        return self.head(h)


class Agg(nn.Module):
    def __init__(self, d=D_PROBE, s=D_STATE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, s))

    def forward(self, z, mask):
        z = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return self.net(z)


class Pred(nn.Module):
    def __init__(self, n_probe, s=D_STATE, d=D_PROBE):
        super().__init__()
        self.probe = nn.Embedding(n_probe, 64)
        self.net = nn.Sequential(nn.Linear(s + 64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, d))

    def forward(self, s, pid):
        return self.net(torch.cat([s, self.probe(pid)], -1))


def vicreg(z):
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + 1e-4)
    cov = (z.T @ z) / (z.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return F.relu(1 - std).mean(), (off ** 2).sum() / z.shape[1]


def train_jepa(Xn, seed, log=print, steps=STEPS):
    A, P, R, T, C = Xn.shape
    STEPS_ = steps
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, agg, pred = Enc(C, P), Agg(), Pred(P)
    tgt = copy.deepcopy(enc)
    for p_ in tgt.parameters():
        p_.requires_grad = False
    params = list(enc.parameters()) + list(agg.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    X = torch.tensor(Xn)
    B, K = BATCH, K_CTX
    for step in range(STEPS_):
        for g in opt.param_groups:
            g["lr"] = LR * 0.5 * (1 + math.cos(math.pi * step / STEPS_))
        a = rng.integers(0, A, B)
        ctx_p = np.stack([rng.permutation(P)[: K + 1] for _ in range(B)])
        tgt_p = ctx_p[:, -1]
        ctx_p = ctx_p[:, :K]
        ctx_r = rng.integers(0, R, (B, K))
        tgt_r = rng.integers(0, R, B)
        k_used = rng.integers(1, K + 1, B)
        mask = torch.tensor((np.arange(K)[None] < k_used[:, None]).astype(np.float32))
        zc_flat = enc(X[a[:, None], ctx_p, ctx_r].reshape(B * K, T, C), torch.tensor(ctx_p.reshape(-1)))
        s_ = agg(zc_flat.reshape(B, K, -1), mask)
        zp = pred(s_, torch.tensor(tgt_p))
        with torch.no_grad():
            zt = tgt(X[a, tgt_p, tgt_r], torch.tensor(tgt_p))
            zt = (zt - zt.mean(0)) / (zt.std(0) + 1e-4)
        pred_loss = F.mse_loss(zp, zt)
        v1, c1 = vicreg(zc_flat)
        v2, c2 = vicreg(s_)
        loss = pred_loss + W_VAR * (v1 + v2) + W_COV * (c1 + c2)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        mom = EMA + (1 - EMA) * step / STEPS_
        with torch.no_grad():
            for pt, pe in zip(tgt.parameters(), enc.parameters()):
                pt.mul_(mom).add_(pe.detach(), alpha=1 - mom)
        if step % 1000 == 0 or step == STEPS_ - 1:
            log(f"   seed {seed} step {step} pred {pred_loss.item():.4f} var {v1.item():.3f}/{v2.item():.3f} cov {c1.item():.3f}")
    enc.eval()
    agg.eval()
    return enc, agg


@torch.no_grad()
def embed(enc, agg, Xn, chunk=256):
    A, P, R, T, C = Xn.shape
    out = []
    pids = torch.arange(P).repeat_interleave(R)
    for i in range(0, A, chunk):
        x = torch.tensor(Xn[i:i + chunk])
        n = x.shape[0]
        z = enc(x.reshape(n * P * R, T, C), pids.repeat(n)).reshape(n, P * R, -1)
        out.append(agg(z, torch.ones(n, P * R)).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sim")
    ap.add_argument("out")
    ap.add_argument("--dev", help="file listing the development assets")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=STEPS, help="only for smoke tests; results use the frozen default")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    names, train = load_rollouts(args.sim)
    A, P, R, T, C = train.shape
    print(f"assets={A} probes={P} repeats={R} frames={T} channels={C}")
    dev = set(l.strip() for l in open(args.dev)) if args.dev else set()
    dev_idx = np.array([i for i, n in enumerate(names) if n in dev])
    saved = {"names": np.array(names)}
    for fit in (["all", "dev"] if len(dev_idx) else ["all"]):
        ref = train if fit == "all" else train[dev_idx]
        flat = ref.reshape(-1, T, C)
        mu, sd = flat.mean((0, 1)), flat.std((0, 1)) + 1e-6
        Xn_all = np.clip((train - mu) / sd, -6, 6)
        Xn_fit = Xn_all if fit == "all" else Xn_all[dev_idx]
        print(f"== fit on {fit}: {len(Xn_fit)} assets")
        for seed in [int(s) for s in args.seeds.split(",")]:
            enc, agg = train_jepa(Xn_fit, seed, steps=args.steps)
            saved[f"jepa_{fit}_s{seed}"] = embed(enc, agg, Xn_all)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **saved)
    print("saved", args.out, sorted(saved))
