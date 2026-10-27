"""Held-out-probe evaluation with a frozen protocol.

Question: if you search the library from an asset, do its k nearest neighbors predict how that asset reacts to
probes no embedding ever saw (a new push height/direction/speed, a diagonal ramp, a higher tilted drop)?

Protocol
  queries     test assets only (everything except the 276-asset development set used to design the model)
  index       every simulated asset, minus the query itself, its product family (same name up to the _a01-style
              suffix) and any asset with identical geometry (recolored duplicates)
  score       R^2 of the k-NN mean prediction for 11 standardized held-out outcomes, averaged over outcomes
  uncertainty 2,000 bootstrap resamples of the queries; paired differences use the same resamples
  engines     neighbors come from MuJoCo rollouts; the same neighbors are also scored against PhysX (SDF collision,
              NVIDIA's PhysX variant) outcomes to test that the result is not a MuJoCo artifact
"""
import argparse
import collections
import glob
import json
import math
import os
import re

import numpy as np
from scipy.spatial import ConvexHull

OUTCOMES = ["push max tilt", "push final tilt", "push displacement", "push toppled", "ramp slip angle", "ramp tip angle",
            "ramp final tilt", "drop final tilt", "drop displacement", "drop bounces", "drop settle time"]


def held_vec(o):
    pu, ra, dr = o["held_out"]
    return [pu["max_tilt"] / 180, pu["final_tilt"] / 180, min(pu["disp"], 3), pu["toppled"],
            ra["slip_angle"] / 50, ra["tip_angle"] / 50, ra["final_tilt"] / 180,
            dr["final_tilt"] / 180, min(dr["disp"], 3), min(dr.get("bounces", 0), 4), dr["settle"]]


def standardize(M):
    M = np.asarray(M, dtype=np.float64)
    return (M - M.mean(0)) / (M.std(0) + 1e-9)


def family(n):
    return re.sub(r"_[a-z]\d\d$", "", n)


# ---------------------------------------------------------------- descriptors that need no simulation
def meta_vec(a):
    dims = np.sort(np.asarray(a["dims"]))[::-1]
    S, H = dims[0], a["dims"][2]
    foot = min(a["dims"][0], a["dims"][1])
    com_h = (a["com"][2] - a["bmin"][2]) / max(H, 1e-6)
    return [math.log(a["mass"]), math.log(S), dims[1] / S, dims[2] / S, H / S, foot / max(H, 1e-6),
            com_h, a["friction"], a["restitution"], math.log(max(a["bulk_density"], 1e-3))]


def stability_vec(a):
    """What an engineer could compute without simulating: static tip angles around the support polygon of the
    authored resting pose, the slip angle atan(mu), center-of-mass height, proportions and radii of gyration."""
    h = np.asarray(a["hull"], float)
    com = np.asarray(a["com"], float)
    S, H = max(a["dims"]), a["dims"][2]
    zc = com[2] - h[:, 2].min()
    base = h[h[:, 2] <= h[:, 2].min() + 0.02 * H][:, :2]
    tips = []
    for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        u = np.array([math.cos(th), math.sin(th)])
        reach = (base @ u).max() - com[:2] @ u
        tips.append(math.degrees(math.atan2(max(reach, 0.0), max(zc, 1e-6))))
    tips = np.sort(tips)
    w = np.linalg.eigvalsh(np.asarray(a["inertia"]))
    k = np.sqrt(np.maximum(w, 0) / a["mass"]) / S
    dims = np.sort(np.asarray(a["dims"]))[::-1] / S
    try:
        fill = ConvexHull(h).volume / max(np.prod(a["dims"]), 1e-12)
    except Exception:
        fill = 0.0
    return [tips[0], tips[4], tips[8], tips[-1], math.degrees(math.atan(a["friction"])), zc / S, H / S,
            dims[1], dims[2], *k, fill, a["restitution"]]


def words_vec(names):
    toks = [[t.lower() for part in family(n).split("_") for t in re.findall(r"[A-Z]?[a-z]+|\d+", part)] for n in names]
    vocab = sorted({t for ts in toks for t in ts})
    ix = {t: i for i, t in enumerate(vocab)}
    df = collections.Counter(t for ts in toks for t in set(ts))
    M = np.zeros((len(names), len(vocab)))
    for r, ts in enumerate(toks):
        for t in ts:
            M[r, ix[t]] += math.log(len(names) / df[t])
    return M


def stats_vec(o):
    by = {}
    for pi, d in o["train_out"]:
        by.setdefault(pi, []).append([d["max_tilt"] / 180, d["final_tilt"] / 180, min(d["disp"], 3), min(d["max_speed"], 10),
                                      d["settle"], d["toppled"], d.get("slip_angle", 0) / 50, d.get("tip_angle", 0) / 50,
                                      min(d.get("bounces", 0), 4)])
    return np.concatenate([np.mean(by[k], 0) for k in sorted(by)])


# ---------------------------------------------------------------- evaluation core
def geometry_groups(meta, names, tol=0.02):
    """Union-find over assets whose bounding dims and hull volume match within tol (recolored duplicates)."""
    D = np.array([meta[n]["dims"] for n in names])
    V = np.array([meta[n]["hull_volume"] for n in names])
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(names)):
        close = np.where((np.abs(D - D[i]) <= tol * D[i].max()).all(1) & (np.abs(V - V[i]) <= tol * V[i]))[0]
        for j in close:
            parent[find(j)] = find(i)
    return np.array([find(i) for i in range(len(names))])


def neighbor_preds(emb, Y, queries, excl, k):
    Z = standardize(emb)
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    Sm = Z[queries] @ Z.T
    Sm[excl] = -np.inf
    nn = np.argpartition(-Sm, k, axis=1)[:, :k]
    return Y[nn].mean(1), nn


def r2_from(err2, Yq, idx=None):
    if idx is not None:
        err2, Yq = err2[idx], Yq[idx]
    sst = ((Yq - Yq.mean(0)) ** 2).sum(0)
    return 1 - err2.sum(0) / np.maximum(sst, 1e-12)


def evaluate(embs, Y, queries, excl, k, boot_idx, binary):
    res, err2s = {}, {}
    Yq = Y[queries]
    for name, E in embs.items():
        seeds = E if isinstance(E, list) else [E]
        per_seed, per_out = [], []
        for e in seeds:
            pred, nn = neighbor_preds(e, Y, queries, excl, k)
            err2 = (pred - Yq) ** 2
            per_seed.append(err2)
            per_out.append(r2_from(err2, Yq))
        err2 = np.mean(per_seed, 0)  # seed-averaged squared error per query
        err2s[name] = err2
        boots = np.array([r2_from(err2, Yq, b).mean() for b in boot_idx])
        r = {"R2": float(np.mean([p.mean() for p in per_out])), "R2 seed std": float(np.std([p.mean() for p in per_out])),
             "R2 95% CI": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
             "R2 per outcome": {o: float(v) for o, v in zip(OUTCOMES, np.mean(per_out, 0))}}
        pred, nn = neighbor_preds(seeds[0], Y, queries, excl, k)
        for bname, bvec in binary.items():
            r[f"{bname} accuracy"] = float(((bvec[nn].mean(1) > 0.5) == bvec[queries]).mean())
        res[name] = r
    return res, err2s


def paired(err2s, Yq, a, b, boot_idx):
    d = np.array([r2_from(err2s[a], Yq, i).mean() - r2_from(err2s[b], Yq, i).mean() for i in boot_idx])
    full = r2_from(err2s[a], Yq).mean() - r2_from(err2s[b], Yq).mean()
    return {"difference": float(full), "95% CI": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "fraction of resamples where first is better": float((d > 0).mean())}


def load_outs(sim_dir):
    out = {}
    for p in glob.glob(f"{sim_dir}/*.json"):
        if not os.path.basename(p).startswith("_"):
            j = json.load(open(p))
            out[j["name"]] = j
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--appearance", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--px_parts", help="PhysX sim dir with the same per-part hulls as MuJoCo (engine-only check)")
    ap.add_argument("--px_sdf", help="PhysX sim dir with SDF collision, NVIDIA's PhysX variant")
    ap.add_argument("--px_sample", help="the fixed random sample the SDF run covers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    E = np.load(args.emb)
    app = np.load(args.appearance)
    outs = load_outs(args.sim)
    pxs = {}
    if args.px_parts:
        pxs["PhysX, same hulls"] = load_outs(args.px_parts)
    if args.px_sdf:
        smp = set(open(args.px_sample).read().split()) if args.px_sample else None
        pxs["PhysX, SDF collision (NVIDIA's PhysX variant)"] = {n: o for n, o in load_outs(args.px_sdf).items() if smp is None or n in smp}
    a_idx = {n: i for i, n in enumerate(app["names"])}
    e_idx = {n: i for i, n in enumerate(E["names"])}
    names = sorted(n for n in outs if n in a_idx and n in e_idx)
    meta = {n: json.load(open(f"{args.assets}/{n}.json")) for n in names}
    dev = set(l.strip() for l in open(args.dev))
    A = len(names)
    queries = np.array([i for i, n in enumerate(names) if n not in dev])
    print(f"assets {A}  test queries {len(queries)}  dev {A - len(queries)}")

    fam = np.array([family(n) for n in names])
    geo = geometry_groups(meta, names)
    excl = (fam[queries][:, None] == fam[None]) | (geo[queries][:, None] == geo[None])
    excl[np.arange(len(queries)), queries] = True
    excl_self = np.zeros_like(excl)
    excl_self[np.arange(len(queries)), queries] = True
    print(f"excluded per query: median {np.median(excl.sum(1)):.0f}, max {excl.sum(1).max()} "
          f"(family or identical geometry); {len(set(geo))} geometry groups")

    take = lambda key: E[key][[e_idx[n] for n in names]]
    seeds = sorted({k.split("_s")[1] for k in E.files if k.startswith("jepa_all_s")})
    embs = {
        "JEPA behavior embedding": [take(f"jepa_all_s{s}") for s in seeds],
        "JEPA trained on dev assets only": [take(f"jepa_dev_s{s}") for s in seeds if f"jepa_dev_s{s}" in E.files],
        "Rollout statistics (hand-built)": standardize([stats_vec(outs[n]) for n in names]),
        "Static stability (no simulation)": standardize([stability_vec(meta[n]) for n in names]),
        "Physics metadata": standardize([meta_vec(meta[n]) for n in names]),
        "Appearance (SigLIP2)": app["emb"][[a_idx[n] for n in names]],
        "Category words (TF-IDF of names)": words_vec(names),
    }
    embs["Appearance + physics metadata"] = np.hstack([standardize(embs["Appearance (SigLIP2)"]), embs["Physics metadata"]])
    embs["Random"] = [np.random.default_rng(s).normal(size=(A, 64)) for s in range(3)]
    if not embs["JEPA trained on dev assets only"]:
        del embs["JEPA trained on dev assets only"]

    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(queries), len(queries)) for _ in range(args.boot)]
    Y = standardize([held_vec(outs[n]) for n in names])
    binary = {"topples under new push": np.array([outs[n]["held_out"][0]["toppled"] for n in names]),
              "tips before slipping on new ramp": np.array([float(outs[n]["held_out"][1]["tip_angle"] < outs[n]["held_out"][1]["slip_angle"] + 1) for n in names])}
    report = {"protocol": __doc__.strip(), "n_assets": A, "n_test_queries": int(len(queries)), "outcomes": OUTCOMES,
              "chance": {k: float(max(v[queries].mean(), 1 - v[queries].mean())) for k, v in binary.items()}}

    main, err2s = evaluate(embs, Y, queries, excl, 5, boot_idx, binary)
    report["main (MuJoCo outcomes, k=5, family + duplicate exclusion)"] = main
    Yq = Y[queries]
    J = "JEPA behavior embedding"
    report["paired differences"] = {f"{J} minus {b}": paired(err2s, Yq, J, b, boot_idx)
                                    for b in embs if b != J}
    report["paired differences"]["Rollout statistics (hand-built) minus Appearance (SigLIP2)"] = paired(
        err2s, Yq, "Rollout statistics (hand-built)", "Appearance (SigLIP2)", boot_idx)
    report["paired differences"]["Static stability (no simulation) minus Appearance (SigLIP2)"] = paired(
        err2s, Yq, "Static stability (no simulation)", "Appearance (SigLIP2)", boot_idx)

    few = {k: embs[k] for k in (J, "Rollout statistics (hand-built)", "Static stability (no simulation)", "Appearance (SigLIP2)")}
    report["k sensitivity"] = {f"k={k}": {n: r["R2"] for n, r in evaluate(few, Y, queries, excl, k, boot_idx[:200], binary)[0].items()}
                               for k in (1, 10, 20)}
    report["without duplicate exclusion (self only)"] = {n: r["R2"] for n, r in evaluate(few, Y, queries, excl_self, 5, boot_idx[:200], binary)[0].items()}

    # noise ceiling: how well an asset's own jittered repeat predicts its outcomes on the same training probe
    rep = collections.defaultdict(list)
    for n in names:
        by = collections.defaultdict(list)
        for pi, d in outs[n]["train_out"]:
            by[pi].append([d["max_tilt"] / 180, d["final_tilt"] / 180, min(d["disp"], 3), d["toppled"]])
        for pi, v in by.items():
            if len(v) > 1:
                rep["r0"].append(v[0])
                rep["r1"].append(v[1])
    if rep:
        r0, r1 = np.array(rep["r0"]), np.array(rep["r1"])
        report["repeat consistency R2 (same asset, same probe, small yaw/speed jitter)"] = float(
            (1 - ((r0 - r1) ** 2).sum(0) / ((r1 - r1.mean(0)) ** 2).sum(0)).mean())

    for label, px in pxs.items():
        sub = np.array([i for i, n in enumerate(names) if n in px])
        if len(sub) < 50:
            continue
        sn = [names[i] for i in sub]
        q_sub = np.array([j for j, i in enumerate(sub) if names[i] not in dev])
        ex_sub = (fam[sub][q_sub][:, None] == fam[sub][None]) | (geo[sub][q_sub][:, None] == geo[sub][None])
        ex_sub[np.arange(len(q_sub)), q_sub] = True
        Ypx = standardize([held_vec(px[n]) for n in sn])
        Ymj = standardize([held_vec(outs[n]) for n in sn])
        pick = lambda e: [x[sub] for x in e] if isinstance(e, list) else e[sub]
        keys = (J, "Rollout statistics (hand-built)", "Static stability (no simulation)", "Physics metadata",
                "Appearance (SigLIP2)", "Category words (TF-IDF of names)", "Random")
        b_sub = [rng.integers(0, len(q_sub), len(q_sub)) for _ in range(500)]
        cross, cerr = evaluate({k: pick(embs[k]) for k in keys}, Ypx, q_sub, ex_sub, 5, b_sub, {})
        same, _ = evaluate({k: pick(embs[k]) for k in (J, "Appearance (SigLIP2)")}, Ymj, q_sub, ex_sub, 5, b_sub[:100], {})
        Hm = np.array([held_vec(outs[n]) for n in sn])
        Hp = np.array([held_vec(px[n]) for n in sn])
        corr = {o: float(np.corrcoef(Hm[:, j], Hp[:, j])[0, 1]) for j, o in enumerate(OUTCOMES) if Hm[:, j].std() > 0 and Hp[:, j].std() > 0}
        first = lambda o: [d for pi, d in o["train_out"] if True][::max(1, len(o["train_out"]) // 10)][:10] + o["held_out"]
        tm = np.array([[d["toppled"] for d in first(outs[n])] for n in sn])
        tp = np.array([[d["toppled"] for d in first(px[n])] for n in sn])
        mxm = np.array([[d["max_tilt"] for d in first(outs[n])] for n in sn])
        mxp = np.array([[d["max_tilt"] for d in first(px[n])] for n in sn])
        clear = ~(((mxm > 25) & (mxm < 65)) | ((mxp > 25) & (mxp < 65)))
        report[f"cross-engine: {label}"] = {
            "n_assets": int(len(sub)), "n_test_queries": int(len(q_sub)),
            "MuJoCo neighbors scored on PhysX outcomes": cross,
            "same queries scored on MuJoCo outcomes (reference)": {k: v["R2"] for k, v in same.items()},
            "JEPA minus appearance on PhysX outcomes": paired(cerr, Ypx[q_sub], J, "Appearance (SigLIP2)", b_sub),
            "held-out outcome correlation (MuJoCo vs PhysX)": corr,
            "topple/no-topple agreement, all probe runs": float((tm == tp).mean()),
            "topple agreement away from the tipping point (max tilt outside 25-65 deg in both)": float((tm == tp)[clear].mean()),
            "share of probe runs near the tipping point": float(1 - clear.mean()),
            "rest tilt > 10 deg in MuJoCo / PhysX / both": [int(sum(outs[n]["rest_tilt"] > 10 for n in sn)),
                                                           int(sum(px[n]["rest_tilt"] > 10 for n in sn)),
                                                           int(sum(outs[n]["rest_tilt"] > 10 and px[n]["rest_tilt"] > 10 for n in sn))],
        }
        print(f"\n== cross-engine: {label} ({len(sub)} assets, {len(q_sub)} test queries) ==")
        for k, r in cross.items():
            print(f"{k:38s} {r['R2']:+.3f}  [{r['R2 95% CI'][0]:+.3f}, {r['R2 95% CI'][1]:+.3f}]")
        rep_ = report[f"cross-engine: {label}"]
        print({k: v for k, v in rep_.items() if k.startswith(("topple", "share", "rest", "JEPA minus"))})
    json.dump(report, open(args.out, "w"), indent=1)

    print(f"\n== main: R2 on {len(queries)} test queries (k=5, family/duplicate exclusion, 95% bootstrap CI) ==")
    for n, r in main.items():
        print(f"{n:38s} {r['R2']:+.3f}  [{r['R2 95% CI'][0]:+.3f}, {r['R2 95% CI'][1]:+.3f}]  seed std {r['R2 seed std']:.3f}  "
              + "  ".join(f"{k.split(' accuracy')[0][:18]} {v:.2f}" for k, v in r.items() if k.endswith("accuracy")))
    print("chance:", report["chance"])
    for k, v in report["paired differences"].items():
        print(f"  {k:80s} {v['difference']:+.3f} [{v['95% CI'][0]:+.3f}, {v['95% CI'][1]:+.3f}]")
    for k in ("k sensitivity", "without duplicate exclusion (self only)"):
        print(k, json.dumps(report[k], indent=None)[:600])
    if "repeat consistency R2 (same asset, same probe, small yaw/speed jitter)" in report:
        print("repeat consistency:", round(report["repeat consistency R2 (same asset, same probe, small yaw/speed jitter)"], 3))
