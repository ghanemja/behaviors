"""Physics data-quality checks on the authored SimReady values, each with the evidence needed to verify it.

Errors (physically impossible as authored):
  com_at_base        centerOfMass authored at the mesh origin, and that origin sits in the bottom 10% of the object
  com_outside        center of mass lies outside the convex hull of the collision geometry (by more than 2% of size)
  inertia_impossible a principal radius of gyration exceeds 1.1x the farthest collision point from the center of mass
Warnings (suspicious, not impossible):
  unstable_at_rest   tips more than 10 deg from the authored pose when released, in BOTH MuJoCo and PhysX (SDF)
  no_friction        no physics material sets friction, so engines fall back to their own defaults
  density            mass / hull volume below 15 or above 25,000 kg/m^3
Every check reads the authored values; the PhysX and MuJoCo variants of SimReady author identical mass properties
(verified per asset), so a flag applies to both.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial import ConvexHull


def com_outside(h, com, S):
    hull = ConvexHull(h)
    # signed distance to each facet plane (positive = outside)
    d = (hull.equations[:, :3] @ com + hull.equations[:, 3]).max()
    return d > 0.02 * S, float(d / S)


def check(a, rest_mj=None, rest_px=None):
    h = np.asarray(a["hull"], float)
    com = np.asarray(a["com"], float)
    S, H, m = max(a["dims"]), a["dims"][2], a["mass"]
    flags = {}
    com_h = (com[2] - a["bmin"][2]) / max(H, 1e-6)
    if np.linalg.norm(com) < 1e-5 and com_h < 0.1 and H > 0.02:
        flags["com_at_base"] = {"authored_com": com.round(6).tolist(), "com_height_fraction": round(com_h, 3)}
    out, d = com_outside(h, com, S)
    if out:
        flags["com_outside"] = {"distance_outside_as_fraction_of_size": round(d, 3), "authored_com": com.round(4).tolist()}
    w = np.linalg.eigvalsh(np.asarray(a["inertia"]))
    k = np.sqrt(np.maximum(w, 0) / m)
    rmax = np.linalg.norm(h - com, axis=1).max()
    if k.max() > 1.1 * rmax:
        flags["inertia_impossible"] = {"max_radius_of_gyration_m": round(float(k.max()), 4), "farthest_point_m": round(float(rmax), 4),
                                       "ratio": round(float(k.max() / rmax), 2)}
    if rest_mj is not None and rest_px is not None and rest_mj > 10 and rest_px > 10:
        flags["unstable_at_rest"] = {"rest_tilt_mujoco_deg": round(rest_mj, 1), "rest_tilt_physx_deg": round(rest_px, 1)}
    if any(f.startswith("no friction authored") for f in a["flags"]):
        flags["no_friction"] = {}
    bulk = a["bulk_density"]
    if bulk < 15 or bulk > 25000:
        flags["density"] = {"mass_kg": round(m, 4), "hull_volume_m3": float(f"{a['hull_volume']:.4g}"), "kg_per_m3": round(bulk, 1)}
    return flags


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--assets_px", required=True)
    ap.add_argument("--sim_mj", required=True)
    ap.add_argument("--sim_px", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, mismatch = {}, []
    for p in sorted(glob.glob(f"{args.assets}/*.json")):
        if os.path.basename(p).startswith("_"):
            continue
        a = json.load(open(p))
        n = a["name"]
        pxp = f"{args.assets_px}/{n}.json"
        if os.path.exists(pxp):
            b = json.load(open(pxp))
            if abs(a["mass"] - b["mass"]) > 1e-6 * a["mass"] or np.linalg.norm(np.subtract(a["com"], b["com"])) > 1e-4 * max(a["dims"]):
                mismatch.append(n)
        rm = f"{args.sim_mj}/{n}.json"
        rp = f"{args.sim_px}/{n}.json"
        rest_mj = json.load(open(rm))["rest_tilt"] if os.path.exists(rm) else None
        rest_px = json.load(open(rp))["rest_tilt"] if os.path.exists(rp) else None
        f = check(a, rest_mj, rest_px)
        if f:
            rows[n] = f
    counts = {}
    for f in rows.values():
        for k in f:
            counts[k] = counts.get(k, 0) + 1
    n_assets = len([p for p in glob.glob(f"{args.assets}/*.json") if not os.path.basename(p).startswith("_")])
    json.dump({"n_assets_checked": n_assets, "counts": counts, "variant_mass_mismatches": mismatch, "assets": rows,
               "definitions": __doc__.strip()}, open(args.out, "w"), indent=1)
    print(f"checked {n_assets} assets; flags: {counts}; MuJoCo/PhysX variant mass-property mismatches: {len(mismatch)}")
