"""Find look-alike/behave-differently pairs, behave-alike/look-different pairs, and per-asset behavior summaries.

Neighbor similarity is the cosine similarity of the JEPA embedding averaged over training seeds. Behave-alike pairs
never come from the same product family or from recolored copies of one mesh. Data-quality flags come from qa.py.
"""
import argparse
import json
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from evaluate import family, geometry_groups  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True)
ap.add_argument("--assets", required=True)
ap.add_argument("--emb", required=True)
ap.add_argument("--appearance", required=True)
ap.add_argument("--qa", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--method", default="stats", choices=["jepa", "stats"], help="behavior similarity used for pairs")
args = ap.parse_args()
SIM, ASSETS = args.sim, args.assets
E = np.load(args.emb)
app = np.load(args.appearance)
aidx = {n: i for i, n in enumerate(app["names"])}
import os  # noqa: E402
names = [n for n in E["names"] if n in aidx and os.path.exists(f"{SIM}/{n}.json")]
eidx = {n: i for i, n in enumerate(E["names"])}
A = len(names)
qa = json.load(open(args.qa))["assets"]


def standardize(M):
    M = np.asarray(M, dtype=np.float64)
    return (M - M.mean(0)) / (M.std(0) + 1e-9)


def cos(M):
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M @ M.T


if args.method == "jepa":
    seed_keys = sorted(k for k in E.files if k.startswith("jepa_all_s"))
    Jb = np.mean([cos(standardize(E[k][[eidx[n] for n in names]])) for k in seed_keys], 0)
else:
    from evaluate import stats_vec
    Jb = cos(standardize([stats_vec(json.load(open(f"{SIM}/{n}.json"))) for n in names]))
Ap = cos(app["emb"][[aidx[n] for n in names]])
meta = {n: json.load(open(f"{ASSETS}/{n}.json")) for n in names}
sim = {n: json.load(open(f"{SIM}/{n}.json")) for n in names}
FAM = np.array([family(n) for n in names])
GEO = geometry_groups(meta, names)
same_obj = (FAM[:, None] == FAM[None]) | (GEO[:, None] == GEO[None])


def pretty(n):
    """NVIDIA's Wikidata class label, unless it is a long title (some labels are paper titles); then the asset id."""
    lab = meta[n]["labels"].get("wikidata_class")
    lab = (lab[0] if isinstance(lab, list) else lab) if lab else None
    if lab and len(lab) <= 32:
        return lab
    parts = family(n).split("_")
    words = lambda p: [w.lower() for w in re.findall(r"[A-Z]{2,}(?![a-z])|[A-Z]?[a-z]+|\d+", p)]
    return " ".join(w for p in (parts[1:] or parts) for w in words(p))


def summary(n):
    o = sim[n]["train_out"]
    by = {}
    for pi, d in o:
        by.setdefault(pi, []).append(d)
    m = lambda pis, k: float(np.mean([d[k] for pi in pis for d in by[pi]]))
    # push outcomes count only runs where the bar touched the object (0 if it never did at that height)
    hit = lambda pis: [d for pi in pis for d in by[pi] if d.get("touched", 1)]
    mp = lambda pis, k: float(np.mean([d[k] for d in hit(pis)])) if hit(pis) else 0.0
    s = {
        "tips_low": mp([0, 1], "toppled"), "tips_mid": mp([2, 3], "toppled"), "tips_high": mp([4, 5], "toppled"),
        "slip_angle": m([6, 7], "slip_angle"), "tip_angle": m([6, 7], "tip_angle"),
        "lands_upright": 1 - m([8, 9], "toppled"), "bounces": m([8, 9], "bounces"),
        "rolls": float(np.mean([(d["max_tilt"] > 120 and d["disp"] > 0.6) for d in hit([0, 1])])) if hit([0, 1]) else 0.0,
        "rest_tilt": sim[n]["rest_tilt"],
        "high_push_unreached": not hit([4, 5]),
    }
    if m([6, 7], "tip_angle") < 6:
        s["rolls"] = max(s["rolls"], 1.0)
    a = meta[n]
    H = a["dims"][2]
    s["com_ratio"] = (a["com"][2] - a["bmin"][2]) / max(H, 1e-6)
    s["com_at_base_bug"] = "com_at_base" in qa.get(n, {})
    s["base_ratio"] = min(a["dims"][0], a["dims"][1]) / max(H, 1e-6)
    s["friction"] = a["friction"]
    s["mass"] = a["mass"]
    s["size"] = max(a["dims"])
    return s


S = {n: summary(n) for n in names}


def describe(n):
    s = S[n]
    bits = []
    if s["rest_tilt"] > 45:
        bits.append("falls over at rest")
    if s["rolls"] > 0.5:
        bits.append("rolls when nudged")
    elif s["tips_low"] > 0.5:
        bits.append("tips even when pushed low")
    elif s["tips_high"] > 0.5:
        bits.append("tips when pushed high, slides when pushed low")
    elif s["high_push_unreached"]:
        bits.append("slides when pushed low; too low-lying for the high push")
    else:
        bits.append("slides instead of tipping")
    if s["tip_angle"] < s["slip_angle"] + 1 and s["tip_angle"] < 50:
        bits.append(f"tips over on a {s['tip_angle']:.0f}° ramp")
    elif s["slip_angle"] < 50:
        bits.append(f"slides off a {s['slip_angle']:.0f}° ramp")
    else:
        bits.append("holds on a 45° ramp")
    bits.append("lands upright" if s["lands_upright"] > 0.5 else "tumbles when dropped")
    return bits


def why(a, b):
    sa, sb = S[a], S[b]
    reasons = []
    if abs(sa["friction"] - sb["friction"]) > 0.15:
        reasons.append(f"friction {sa['friction']:.2f} vs {sb['friction']:.2f}")
    if abs(sa["com_ratio"] - sb["com_ratio"]) > 0.08:
        reasons.append(f"center of mass at {sa['com_ratio']:.0%} vs {sb['com_ratio']:.0%} of height")
    if abs(np.log(sa["base_ratio"] / sb["base_ratio"])) > 0.25:
        reasons.append(f"base width/height {sa['base_ratio']:.2f} vs {sb['base_ratio']:.2f}")
    if abs(np.log(sa["mass"] / sb["mass"])) > 0.7:
        reasons.append(f"mass {sa['mass']:.2f} kg vs {sb['mass']:.2f} kg")
    return reasons


def behav_dist(a, b):
    keys = ["tips_low", "tips_mid", "tips_high", "rolls", "lands_upright"]
    d = sum(abs(S[a][k] - S[b][k]) for k in keys)
    d += abs(min(S[a]["slip_angle"], 50) - min(S[b]["slip_angle"], 50)) / 25
    d += abs(min(S[a]["tip_angle"], 50) - min(S[b]["tip_angle"], 50)) / 25
    return d


# look alike, behave differently
cand = []
for i in range(A):
    order = np.argsort(-Ap[i])
    for j in order[1:4]:
        if i < j or i not in np.argsort(-Ap[j])[1:4]:
            cand.append((behav_dist(names[i], names[j]) * (1 - Jb[i, j]), Ap[i, j], names[i], names[j]))
cand.sort(reverse=True)
look_alike, seen = [], set()
for score, asim, a, b in cand:
    if a in seen or b in seen or a.split("_")[0] != b.split("_")[0]:
        continue
    same_mesh = bool(GEO[names.index(a)] == GEO[names.index(b)])
    if describe(a)[:2] == describe(b)[:2]:
        continue
    bug = [x for x in (a, b) if S[x]["com_at_base_bug"]]
    look_alike.append({"a": a, "b": b, "appearance_sim": float(asim), "behavior_sim": float(Jb[names.index(a), names.index(b)]),
                       "a_does": describe(a), "b_does": describe(b), "why": why(a, b), "suspect_data_bug": bug,
                       "same_mesh": same_mesh})
    seen |= {a, b}
    if len(look_alike) == 8:
        break

# behave alike, look different
app_q = np.quantile(Ap[np.triu_indices(A, 1)], 0.3)
cand2 = []
for i in range(A):
    for j in np.argsort(-Jb[i])[1:4]:
        a_, b_ = names[i], names[j]
        distinctive = max(S[a_]["tips_high"], S[a_]["rolls"], 1 - S[a_]["lands_upright"]) > 0.5
        bugs = S[a_]["com_at_base_bug"] or S[b_]["com_at_base_bug"]
        if Ap[i, j] < app_q and a_.split("_")[0] != b_.split("_")[0] and not same_obj[i, j] and distinctive and not bugs:
            cand2.append((Jb[i, j], a_, b_))
cand2.sort(reverse=True)
behave_alike, seen = [], set()
for s_, a, b in cand2:
    if a in seen or b in seen:
        continue
    behave_alike.append({"a": a, "b": b, "behavior_sim": float(s_), "appearance_sim": float(Ap[names.index(a), names.index(b)]),
                         "both_do": describe(a), "b_does": describe(b)})
    seen |= {a, b}
    if len(behave_alike) == 8:
        break

# data quality, from qa.py (authored-value checks plus rest stability confirmed in both engines)
LABEL = {"com_at_base": "center of mass authored at the mesh origin, which sits at the object's base",
         "com_outside": "center of mass authored outside the object",
         "inertia_impossible": "inertia too large for the object's size",
         "unstable_at_rest": "not stable in its authored pose (in both MuJoCo and PhysX)",
         "no_friction": "no friction authored",
         "density": "implausible bulk density"}
quality = [{"name": n, "notes": [LABEL[k] for k in qa[n]], "evidence": qa[n]} for n in names if n in qa]

out = {"look_alike_behave_different": look_alike, "behave_alike_look_different": behave_alike, "quality": quality,
       "com_at_base": [n for n in names if S[n]["com_at_base_bug"]],
       "summaries": {n: {**{k: (float(v) if not isinstance(v, bool) else v) for k, v in S[n].items()}, "describe": describe(n), "pretty": pretty(n)} for n in names}}
print("COM-at-base suspects:", out["com_at_base"])
json.dump(out, open(args.out, "w"), indent=1)
print("== look alike, behave differently ==")
for p in look_alike:
    print(f"  {pretty(p['a'])} [{p['a']}] vs {pretty(p['b'])} [{p['b']}]  app={p['appearance_sim']:.2f} beh={p['behavior_sim']:.2f}")
    print(f"     A: {p['a_does']}\n     B: {p['b_does']}\n     why: {p['why']}")
print("== behave alike, look different ==")
for p in behave_alike:
    print(f"  {pretty(p['a'])} [{p['a']}] ~ {pretty(p['b'])} [{p['b']}]  beh={p['behavior_sim']:.2f} app={p['appearance_sim']:.2f}  {p['both_do']} | {p['b_does']}")
print(f"== quality notes: {len(quality)} assets ==")
from collections import Counter
print(Counter(x.split(' (')[0].split(' ')[0] + ' ' + x.split(' ')[1] for q in quality for x in q['notes']))
