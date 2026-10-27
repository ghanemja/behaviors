"""Scan collision approximations and authored PhysX attributes per physics variant."""
import collections, glob, json, sys
from pxr import Usd, UsdGeom, UsdPhysics

RAW = sys.argv[1]
assets = [l.strip() for l in open(sys.argv[2]) if l.strip()]
stats = {v: collections.Counter() for v in ("MuJoCo", "PhysX", "Newton")}
extra = collections.Counter()
for a in assets:
    roots = glob.glob(f"{RAW}/{a}/simready_usd/*.usd")
    if not roots:
        continue
    for var in stats:
        st = Usd.Stage.Open(roots[0])
        vs = st.GetDefaultPrim().GetVariantSets()
        for n in stats:
            if vs.HasVariantSet(n):
                vs.GetVariantSet(n).SetVariantSelection("Enabled" if n == var else "Disabled")
        approx = collections.Counter()
        for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.IsA(UsdGeom.Gprim):
                if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    approx[UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() or "none"] += 1
                else:
                    approx[p.GetTypeName()] += 1
                if var == "PhysX":
                    for at in p.GetAttributes():
                        if at.HasAuthoredValue() and at.GetName().startswith(("physx", "physxSDF", "physxConvex")):
                            extra[at.GetName()] += 1
        stats[var]["+".join(sorted(approx)) or "none"] += 1
for v, c in stats.items():
    print(v, c.most_common(8))
print("authored PhysX attrs:", extra.most_common(15))
