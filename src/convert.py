"""Convert downloaded SimReady assets into compact rigid-body records for simulation and the web demo."""
import glob
import json
import os
import sys
import traceback

import fast_simplification
import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

RAW, OUT = sys.argv[1], sys.argv[2]
MAX_BODIES = 4
VARIANT = os.environ.get("SR_VARIANT", "MuJoCo")  # which engine variant's physics payload to read
PART_FACES = 20000  # cap per collision part saved for the simulators
os.makedirs(OUT, exist_ok=True)


def mat_np(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)])


def xform_points(M, pts):
    p = np.c_[pts, np.ones(len(pts))] @ M
    return p[:, :3]


def triangulate(counts, idx):
    tris, k = [], 0
    for c in counts:
        for t in range(1, c - 1):
            tris.append((idx[k], idx[k + t], idx[k + t + 1]))
        k += c
    return np.array(tris, dtype=np.int64).reshape(-1, 3)


def quat_to_R(q):
    w, (x, y, z) = q.GetReal(), q.GetImaginary()
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def authored(attr):
    return attr and attr.HasAuthoredValue()


def phys_material(prim):
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if not mat or not mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        return None
    api = UsdPhysics.MaterialAPI(mat.GetPrim())
    get = lambda a: a.Get() if authored(a) else None
    return {"dyn": get(api.GetDynamicFrictionAttr()), "stat": get(api.GetStaticFrictionAttr()),
            "rest": get(api.GetRestitutionAttr()), "density": get(api.GetDensityAttr()), "name": mat.GetPrim().GetName()}


def validation_failures(asset_dir):
    fails, profile = set(), None
    for p in glob.glob(f"{asset_dir}/simready_usd/payloads/Physics/physics.usda"):
        layer = Sdf.Layer.FindOrOpen(p)
        meta = dict(layer.customLayerData or {}).get("SimReady_Metadata", {})
        val = meta.get("validation", {})
        profile = val.get("profile")
        dates = sorted(val.get("validated_features", {}).keys())
        if dates:
            for feat, d in val["validated_features"][dates[-1]].items():
                if not d.get("passed", True):
                    fails.add(feat)
    return sorted(fails), profile


def license_sha(asset_dir):
    import hashlib
    p = f"{asset_dir}/.metadata/LICENSE.txt"
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:12] if os.path.exists(p) else None


def owner_body(prim):
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


def hull_props(pts, density):
    try:
        h = trimesh.convex.convex_hull(pts)
        if h.volume > 1e-12:
            h.density = density
            return h.mass, np.array(h.center_mass), np.array(h.moment_inertia), h.volume
    except Exception:
        pass
    c = pts.mean(0)
    ext = np.maximum(pts.max(0) - pts.min(0), 1e-4)
    vol = float(np.prod(ext))
    m = density * vol
    I = m / 12 * np.diag([ext[1] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[2] ** 2, ext[0] ** 2 + ext[1] ** 2])
    return m, c, I, vol


def body_mass_props(body, parts, M_body):
    mat = next((pt["mat"] for pt in parts if pt["mat"]), None) or {}
    pts = np.vstack([pt["pts"] for pt in parts])
    source = "authored"
    for pt in parts:
        if pt["mass"] is None:
            source = "density" if (pt["density"] or (mat.get("density") or 0) > 0) else "default"
    if source == "authored" and all(pt["com"] is not None and pt["I"] is not None for pt in parts):
        m = sum(pt["mass"] for pt in parts)
        com = sum(pt["mass"] * pt["com"] for pt in parts) / m
        I = np.zeros((3, 3))
        for pt in parts:
            d = pt["com"] - com
            I += pt["I"] + pt["mass"] * (d @ d * np.eye(3) - np.outer(d, d))
    elif source == "authored":
        m = sum(pt["mass"] for pt in parts)
        _, com, I1, vol = hull_props(pts, 1.0)
        I = I1 * (m / vol)
        source = "authored-mass-only"
    else:
        density = next((pt["density"] for pt in parts if pt["density"]), None) or mat.get("density") or 1000.0
        m, com, I, _ = hull_props(pts, density)
    # USD Physics precedence: mass properties authored on the rigid body override those derived from its colliders.
    # centerOfMass, diagonalInertia and principalAxes are in the body's local frame.
    bm_api = UsdPhysics.MassAPI(body)
    if body.HasAPI(UsdPhysics.MassAPI):
        if authored(bm_api.GetMassAttr()) and (bm_api.GetMassAttr().Get() or 0) > 0:
            bm = float(bm_api.GetMassAttr().Get())
            I = I * (bm / m)
            m = bm
            source = "authored-body"
        c = bm_api.GetCenterOfMassAttr().Get() if authored(bm_api.GetCenterOfMassAttr()) else None
        if c is not None and all(np.isfinite(c)):
            com = xform_points(M_body, np.array(c, dtype=np.float64)[None])[0]
            source += "+com"
        dI = bm_api.GetDiagonalInertiaAttr().Get() if authored(bm_api.GetDiagonalInertiaAttr()) else None
        if dI is not None and np.all(np.asarray(dI) > 0):
            q = bm_api.GetPrincipalAxesAttr().Get() if authored(bm_api.GetPrincipalAxesAttr()) else Gf.Quatf(1, 0, 0, 0)
            R = quat_to_R(q)
            A = M_body[:3, :3].T
            I = A @ (R @ np.diag(np.asarray(dI, dtype=np.float64)) @ R.T) @ A.T
            source += "+inertia"
    return m, np.asarray(com, float), np.asarray(I, float), source, mat


def convert(asset):
    adir = f"{RAW}/{asset}"
    roots = glob.glob(f"{adir}/simready_usd/*.usd")
    if not roots:
        return None, "no root usd"
    stage = Usd.Stage.Open(roots[0])
    root = stage.GetDefaultPrim()
    vsets = root.GetVariantSets()
    for name in ("PhysX", "Newton", "MuJoCo"):
        if vsets.HasVariantSet(name):
            vsets.GetVariantSet(name).SetVariantSelection("Enabled" if name == VARIANT else "Disabled")
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    up = UsdGeom.GetStageUpAxis(stage)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in prims if p.IsA(UsdPhysics.Joint)]
    if not bodies:
        return None, "no rigid body"
    if len(bodies) > MAX_BODIES:
        return None, f"{len(bodies)} rigid bodies"
    Mref_inv = mat_np(xc.GetLocalToWorldTransform(root).GetInverse())
    to_z = np.eye(4)
    if up == "Y":
        to_z = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    S = np.diag([mpu, mpu, mpu, 1.0])

    vis_v, vis_f, col_v, col_parts = [], [], [], []
    parts_by_body = {str(b.GetPath()): [] for b in bodies}
    nv = 0
    for p in prims:
        if not p.IsA(UsdGeom.Mesh):
            continue
        body = owner_body(p)
        if body is None:
            continue
        purpose = UsdGeom.Imageable(p).ComputePurpose()
        mesh = UsdGeom.Mesh(p)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        M = mat_np(xc.GetLocalToWorldTransform(p)) @ Mref_inv @ S @ to_z
        v = xform_points(M, np.array(pts, dtype=np.float64))
        f = triangulate(list(counts), list(idx))
        is_col = p.HasAPI(UsdPhysics.CollisionAPI)
        if purpose in ("default", "render") or not is_col:
            vis_v.append(v)
            vis_f.append(f + nv)
            nv += len(v)
        if not is_col:
            continue
        col_v.append(v)
        approx = UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() if p.HasAPI(UsdPhysics.MeshCollisionAPI) else "none"
        col_parts.append((v, f, approx or "none", str(body.GetPath())))
        part = {"pts": v, "mass": None, "com": None, "I": None, "density": None}
        if p.HasAPI(UsdPhysics.MassAPI):
            mapi = UsdPhysics.MassAPI(p)
            m = mapi.GetMassAttr().Get() if authored(mapi.GetMassAttr()) else None
            if m and m > 0:
                part["mass"] = float(m)
            d = mapi.GetDensityAttr().Get() if authored(mapi.GetDensityAttr()) else None
            if d and d > 0:
                part["density"] = float(d)
            if authored(mapi.GetCenterOfMassAttr()):
                c = np.array(mapi.GetCenterOfMassAttr().Get(), dtype=np.float64)
                part["com"] = xform_points(M, c[None])[0]
            if authored(mapi.GetDiagonalInertiaAttr()):
                dI = np.array(mapi.GetDiagonalInertiaAttr().Get(), dtype=np.float64)
                q = mapi.GetPrincipalAxesAttr().Get() if authored(mapi.GetPrincipalAxesAttr()) else Gf.Quatf(1, 0, 0, 0)
                R = quat_to_R(q)
                A = M[:3, :3].T
                part["I"] = A @ (R @ np.diag(dI) @ R.T) @ A.T
        part["mat"] = phys_material(p) or phys_material(body)
        parts_by_body[str(body.GetPath())].append(part)
    if not col_v:
        return None, "no collision meshes"
    allc = np.vstack(col_v)
    try:
        hull = trimesh.convex.convex_hull(allc)
    except Exception as e:
        return None, f"hull failed {e}"
    if hull.volume <= 1e-9:
        return None, "degenerate hull"
    bmin, bmax = allc.min(0), allc.max(0)
    dims = bmax - bmin

    props = [body_mass_props(b, parts_by_body[str(b.GetPath())], mat_np(xc.GetLocalToWorldTransform(b)) @ Mref_inv @ S @ to_z)
             for b in bodies if parts_by_body[str(b.GetPath())]]
    total_m = sum(p[0] for p in props)
    com = sum(p[0] * p[1] for p in props) / total_m
    I = np.zeros((3, 3))
    for m_b, c_b, I_b, _, _ in props:
        d = c_b - com
        I += I_b + m_b * (d @ d * np.eye(3) - np.outer(d, d))
    sources = sorted({p[3] for p in props})
    mass_source = "+".join(sources)
    mat = next((p[4] for p in props if p[4] and p[4].get("dyn") is not None), None) or next((p[4] for p in props if p[4]), {})
    evals = np.linalg.eigvalsh(I)
    if evals.min() <= 0 or evals[0] + evals[1] < evals[2] * 0.999:
        hm = hull.copy()
        hm.density = total_m / hull.volume
        I = np.array(hm.moment_inertia)
        mass_source += "+hull-inertia"

    friction = mat.get("dyn") if mat.get("dyn") is not None else mat.get("stat")
    restitution = mat.get("rest")
    flags = []
    if len(props) > 1:
        flags.append(f"{len(props)} rigid parts merged ({len(joints)} joints frozen)")
    if friction is None:
        flags.append("no friction authored (default 0.5)")
        friction = 0.5
    if restitution is None:
        restitution = 0.0
    if abs(total_m - 1.0) < 2e-3:
        flags.append("mass is exactly 1 kg")
    bulk = total_m / hull.volume
    if bulk < 15 or bulk > 25000:
        flags.append(f"implausible bulk density {bulk:.0f} kg/m3")
    if "default" in mass_source:
        flags.append("mass from default density")

    vv = np.vstack(vis_v) if vis_v else hull.vertices
    vf = np.vstack(vis_f) if vis_f else hull.faces
    if len(vf) > 1500:
        try:
            vv2, vf2 = fast_simplification.simplify(vv.astype(np.float32), vf.astype(np.int32), target_reduction=1 - 1500 / len(vf))
            if len(vf2) > 50:
                vv, vf = vv2, vf2
        except Exception:
            pass
    body_ids = {str(b.GetPath()): i for i, b in enumerate(bodies)}
    pv, pf, meta = [], [], []
    for v, f, approx, bpath in col_parts:
        if len(f) > PART_FACES:
            try:
                v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - PART_FACES / len(f))
                if len(f2) > 50:
                    v, f = v2, f2
            except Exception:
                pass
        meta.append((sum(len(x) for x in pv), len(v), sum(len(x) for x in pf), len(f), body_ids[bpath]))
        pv.append(np.asarray(v, np.float32))
        pf.append(np.asarray(f, np.int32))
    np.savez_compressed(f"{OUT}/{asset}.parts.npz", v=np.vstack(pv), f=np.vstack(pf), meta=np.array(meta, np.int64),
                        approx=np.array([c[2] for c in col_parts]))
    fails, profile = validation_failures(adir)
    labels = {}
    for a in root.GetAttributes():
        val = a.Get()
        if "semantics" in a.GetName() and val is not None:
            labels[a.GetName().split(":")[-1]] = [str(x) for x in val] if hasattr(val, "__iter__") and not isinstance(val, str) else str(val)
    thumb = glob.glob(f"{adir}/simready_usd/.thumbs/*_thumbnail.png")
    return {
        "name": asset,
        "category": asset.split("_")[0],
        "labels": labels,
        "dims": dims.tolist(), "bmin": bmin.tolist(),
        "mass": total_m, "com": np.asarray(com).tolist(), "inertia": I.tolist(),
        "friction": float(friction), "restitution": float(restitution),
        "material": mat.get("name"), "mass_source": mass_source, "flags": flags,
        "bulk_density": bulk, "hull_volume": hull.volume,
        "validation_fails": fails, "profile": profile,
        "hull": np.round(hull.vertices, 5).tolist(),
        "vis_v": np.round(np.asarray(vv), 4).tolist(), "vis_f": np.asarray(vf).astype(int).tolist(),
        "thumb": thumb[0] if thumb else None,
        "n_parts": len(props), "n_joints": len(joints), "variant": VARIANT,
        "n_col_parts": len(col_parts), "approx": sorted({c[2] for c in col_parts}),
        "license_sha": license_sha(adir),
    }, None


assets = sorted(d for d in os.listdir(RAW) if os.path.isdir(f"{RAW}/{d}") and not d.startswith("."))
if len(sys.argv) > 3:
    assets = [a for a in assets if a in sys.argv[3:]]
ok, skipped = 0, {}
for a in assets:
    try:
        rec, why = convert(a)
    except Exception as e:
        rec, why = None, f"error {type(e).__name__}: {e}"
        traceback.print_exc(limit=1)
    if rec:
        json.dump(rec, open(f"{OUT}/{a}.json", "w"))
        ok += 1
    else:
        skipped[a] = why
print(f"converted {ok}, skipped {len(skipped)}")
reasons = {}
for a, w in skipped.items():
    reasons.setdefault(w.split(":")[0][:40], []).append(a)
for w, lst in sorted(reasons.items(), key=lambda x: -len(x[1])):
    print(f"  {len(lst):4d}  {w}   e.g. {lst[:3]}")
prev = json.load(open(f"{OUT}/_skipped.json")) if os.path.exists(f"{OUT}/_skipped.json") else {}
prev.update(skipped)
json.dump(prev, open(f"{OUT}/_skipped.json", "w"), indent=1)
