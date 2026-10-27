"""Pack the full dataset for the web demo.

out/site/data.json        index: one row per asset (facts, behavior summary, neighbors, map position, 72px thumbnail)
out/site/d/tr_{k}.json    probe playback tracks, 60 assets per shard, loaded when an asset is shown
out/site/d/clay_{k}.json  untextured meshes for assets without a textured model, 60 per shard
out/site/mt/{id}.txt      textured models (base64 GLB) where textures were fetched (the original 276)
Neighbors exclude the asset itself and recolored copies of the same mesh, in both the behavior and appearance rows.
"""
import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import geometry_groups  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--assets", required=True)
ap.add_argument("--emb", required=True)
ap.add_argument("--appearance", required=True)
ap.add_argument("--findings", required=True)
ap.add_argument("--eval", required=True)
ap.add_argument("--qa", required=True)
ap.add_argument("--site", required=True)
ap.add_argument("--extra_sims", nargs="*", default=[], help="other sim dirs counted in the probe-run total")
ap.add_argument("--method", default="jepa", choices=["jepa", "stats"], help="which behavior embedding drives the demo")
args = ap.parse_args()

PROBES = {"push_high": 4, "push_low": 0, "ramp": 6, "drop": 8}
NFR, SHARD = 60, 60
find = json.load(open(args.findings))
summ = find["summaries"]
E = np.load(args.emb)
app = np.load(args.appearance)
aidx = {n: i for i, n in enumerate(app["names"])}
eidx = {n: i for i, n in enumerate(E["names"])}
names = sorted(n for n in summ if n in aidx and n in eidx)
A = len(names)
KEEP = ("name", "dims", "mass", "com", "friction", "restitution", "flags", "thumb", "hull_volume", "category")
meta = {}
for n in names:  # keep the light fields; heavy visual meshes are read per asset when needed
    rec = json.load(open(f"{args.assets}/{n}.json"))
    meta[n] = {k: rec[k] for k in KEEP}
    del rec
qa = json.load(open(args.qa))
evalr = json.load(open(args.eval))


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def standardize(M):
    M = np.asarray(M, dtype=np.float64)
    return (M - M.mean(0)) / (M.std(0) + 1e-9)


def cos(M):
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ M.T


from evaluate import stats_vec  # noqa: E402
seeds = sorted(k for k in E.files if k.startswith("jepa_all_s"))
Zj = [standardize(E[k][[eidx[n] for n in names]]) for k in seeds]
Sim = {"jepa": np.mean([cos(z) for z in Zj], 0)}
Zst = standardize([stats_vec(json.load(open(f"{args.sim}/{n}.json"))) for n in names])
Sim["stats"] = cos(Zst)
Zmap = Zst if args.method == "stats" else np.mean(Zj, 0)
Jb = Sim[args.method]
Jalt = Sim["jepa" if args.method == "stats" else "stats"]
Ap = cos(app["emb"][[aidx[n] for n in names]])
geo = geometry_groups(meta, names)
dup = geo[:, None] == geo[None]
Jb_n, Ap_n, Ja_n = Jb.copy(), Ap.copy(), Jalt.copy()
for M in (Jb_n, Ap_n, Ja_n):
    M[dup] = -np.inf
xy = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(Zmap)
xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))


def thumb(path, size=72):
    if not path or not os.path.exists(path):
        return None, [150, 150, 150]
    im = Image.open(path).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
    im.thumbnail((2 * size, 2 * size))  # flood fill runs in Python, so work at a small size
    W, H = im.size
    for corner in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        if max(im.getpixel(corner)) < 60:
            ImageDraw.floodfill(im, corner, (255, 255, 255), thresh=48)
    px = np.asarray(im.resize((64, 64))).reshape(-1, 3).astype(float)
    fg = px[(px < 235).any(1)]
    col = (fg.mean(0) if len(fg) > 20 else px.mean(0)) * 0.85 + 255 * 0.15 * 0.6
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), [int(c) for c in np.clip(col, 0, 255)]


def clay_mesh(a, target=1500):
    """Visual mesh for assets without a textured model, decimated to about 1,500 faces (the hull if that fails)."""
    v = np.asarray(a["vis_v"], np.float64)
    f = np.asarray(a["vis_f"], np.int64)
    if len(f) > 1.3 * target:
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(v, f.astype(np.int32)))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target, preservetopology=False, qualitythr=0.3, planarquadric=True)
            m = ms.current_mesh()
            v, f = np.asarray(m.vertex_matrix()), np.asarray(m.face_matrix()).astype(np.int64)
        except Exception:
            pass
    if len(f) > 2 * target:
        from scipy.spatial import ConvexHull
        h = np.asarray(a["hull"], np.float64)
        hull = ConvexHull(h)
        idx = {j: i for i, j in enumerate(hull.vertices)}
        v, f = h[hull.vertices], np.array([[idx[j] for j in t] for t in hull.simplices])
    return v, f


QA_LABEL = {"com_at_base": "center of mass authored at the mesh origin, which sits at the object's base",
            "com_outside": "center of mass authored outside the object",
            "inertia_impossible": "inertia too large for the object's size",
            "unstable_at_rest": "not stable in its authored pose (both engines)",
            "no_friction": "no friction authored",
            "density": "implausible bulk density"}
os.makedirs(f"{args.site}/d", exist_ok=True)
assets, tr_shards, clay_shards = [], {}, {}
for i, n in enumerate(names):
    a, s = meta[n], summ[n]
    S = max(a["dims"])
    z = np.load(f"{args.sim}/{n}.npz")
    play = z["play"]
    idx = np.linspace(0, play.shape[1] - 1, NFR).round().astype(int)
    tracks = {}
    for key, pi in PROBES.items():
        pl = play[pi, idx]
        cols = [np.clip(pl[:, 0:3] / S * 8000, -32767, 32767), pl[:, 3:7] * 32767]
        if key.startswith("push"):
            push = pl[:, 7:10].copy()
            push[push[:, 2] < -5] = np.nan
            cols.append(np.clip(np.nan_to_num(push / S * 8000, nan=-32768), -32768, 32767))
        if key == "ramp":
            ang = np.degrees(2 * np.arccos(np.clip(pl[:, 10], -1, 1)))
            cols.append(ang[:, None] * 100)
        tracks[key] = b64(np.round(np.concatenate(cols, 1)).astype("<i2"))
    k = i // SHARD
    tr_shards.setdefault(k, {})[n] = tracks
    th, col = thumb(a.get("thumb"))
    glb = f"mt/{n}.txt" if os.path.exists(f"{args.site}/mt/{n}.txt") else None
    if not glb:
        full = json.load(open(f"{args.assets}/{n}.json"))
        v, f = clay_mesh(full)
        del full
        lo, hi = v.min(0), v.max(0)
        sc = np.maximum(hi - lo, 1e-9)
        q = np.round((v - lo) / sc * 65535 - 32768).astype("<i2")
        clay_shards.setdefault(k, {})[n] = {"v": b64(q), "f": b64(f.astype("<u2" if len(v) < 65536 else "<u4")),
                                            "w": 2 if len(v) < 65536 else 4, "lo": lo.round(6).tolist(), "sc": sc.round(6).tolist()}
    notes = [QA_LABEL[x] for x in qa["assets"].get(n, {})]
    merged = next((x for x in a["flags"] if "rigid parts merged" in x), None)
    if merged:
        notes.append(merged)
    beh = [int(j) for j in np.argsort(-Jb_n[i])[:8]]
    appn = [int(j) for j in np.argsort(-Ap_n[i])[:8]]
    alt = [int(j) for j in np.argsort(-Ja_n[i])[:8]]
    assets.append({
        "id": n, "name": s["pretty"], "cat": a["category"], "S": round(S, 4), "mass": round(a["mass"], 4),
        "mu": round(a["friction"], 3), "e": round(a["restitution"], 3), "r": round(max(0.03 * S, 0.003) / S, 4),
        "glb": glb, "sh": k, "col": col, "thumb": th,
        "beh": beh, "behSim": [round(float(Jb[i, j]), 3) for j in beh],
        "app": appn, "appSim": [round(float(Ap[i, j]), 3) for j in appn],
        "alt": alt, "altSim": [round(float(Jalt[i, j]), 3) for j in alt],
        "xy": [round(float(xy[i, 0]), 4), round(float(xy[i, 1]), 4)],
        "does": s["describe"],
        "arch": "rolls" if s["rolls"] > 0.5 else ("tippy" if s["tips_low"] > 0.5 else ("tips high" if s["tips_high"] > 0.5 else "slides")),
        "notes": notes, "qa": list(qa["assets"].get(n, {})),
        "sum": {"tl": round(s["tips_low"], 2), "th": round(s["tips_high"], 2), "ro": round(s["rolls"], 2),
                "up": round(s["lands_upright"], 2), "sl": round(min(s["slip_angle"], 50), 1), "tp": round(min(s["tip_angle"], 50), 1)},
        "comBug": bool(s["com_at_base_bug"]),
    })
for k, d in tr_shards.items():
    json.dump(d, open(f"{args.site}/d/tr_{k}.json", "w"), separators=(",", ":"))
for k, d in clay_shards.items():
    json.dump(d, open(f"{args.site}/d/clay_{k}.json", "w"), separators=(",", ":"))

idx_of = {n: i for i, n in enumerate(names)}
pairs_diff = [{**p, "a": idx_of[p["a"]], "b": idx_of[p["b"]], "bug": [idx_of[x] for x in p["suspect_data_bug"]]}
              for p in find["look_alike_behave_different"] if p["a"] in idx_of and p["b"] in idx_of]
pairs_same = [{**p, "a": idx_of[p["a"]], "b": idx_of[p["b"]]} for p in find["behave_alike_look_different"] if p["a"] in idx_of and p["b"] in idx_of]
main = evalr["main (MuJoCo outcomes, k=5, family + duplicate exclusion)"]
ev = {"n_assets": evalr["n_assets"], "n_test": evalr["n_test_queries"], "chance": evalr["chance"],
      "main": {k: {"R2": v["R2"], "ci": v["R2 95% CI"]} for k, v in main.items()},
      "paired": evalr["paired differences"], "k": evalr["k sensitivity"], "nodup": evalr["without duplicate exclusion (self only)"],
      "repeat": evalr.get("repeat consistency R2 (same asset, same probe, small yaw/speed jitter)"),
      "cross": {k: {"n": v["n_assets"], "nq": v["n_test_queries"],
                    "R2": {m: {"R2": r["R2"], "ci": r["R2 95% CI"]} for m, r in v["MuJoCo neighbors scored on PhysX outcomes"].items()},
                    "agree": v["topple/no-topple agreement, all probe runs"],
                    "agree_clear": v["topple agreement away from the tipping point (max tilt outside 25-65 deg in both)"],
                    "near": v["share of probe runs near the tipping point"]}
                for k, v in evalr.items() if k.startswith("cross-engine")}}
import glob  # noqa: E402
runs = 0
for d in [args.sim, *args.extra_sims]:
    for p in glob.glob(f"{d}/*.json"):
        if not os.path.basename(p).startswith("_"):
            j = json.load(open(p))
            runs += len(j["train_out"]) + len(j["held_out"])
data = {"runs": runs, "assets": assets, "lookAlikeBehaveDifferent": pairs_diff, "behaveAlikeLookDifferent": pairs_same, "eval": ev,
        "qa": {"counts": qa["counts"], "n": qa["n_assets_checked"]}, "nframes": NFR, "probes": list(PROBES), "shard": SHARD,
        "method": args.method}
json.dump(data, open(f"{args.site}/data.json", "w"), separators=(",", ":"))
sz = os.path.getsize(f"{args.site}/data.json")
print(f"assets {A}, data.json {sz / 1e6:.1f} MB, {len(tr_shards)} track shards, {len(clay_shards)} clay shards, "
      f"{sum(1 for a in assets if a['glb'])} textured")
