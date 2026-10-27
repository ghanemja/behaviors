import sys

from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(sys.argv[1])
root = stage.GetDefaultPrim()
print("defaultPrim:", root.GetPath(), "metersPerUnit:", UsdGeom.GetStageMetersPerUnit(stage), "up:", UsdGeom.GetStageUpAxis(stage))
for vs_name in root.GetVariantSets().GetNames():
    vs = root.GetVariantSets().GetVariantSet(vs_name)
    print("variantSet", vs_name, vs.GetVariantNames(), "selected:", vs.GetVariantSelection())
if len(sys.argv) > 2:
    for kv in sys.argv[2:]:
        k, v = kv.split("=")
        root.GetVariantSets().GetVariantSet(k).SetVariantSelection(v)
for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    schemas = prim.GetAppliedSchemas()
    attrs = {}
    for a in prim.GetAttributes():
        n = a.GetName()
        if any(n.startswith(p) for p in ("physics:", "physxRigidBody", "mjc:", "physxCollision", "material:binding")) or n in ("extent",):
            v = a.Get()
            if v is not None:
                attrs[n] = v if not hasattr(v, "__len__") or len(str(v)) < 80 else str(v)[:80]
    rels = {r.GetName(): [str(t) for t in r.GetTargets()] for r in prim.GetRelationships() if "binding" in r.GetName() or r.GetName().startswith(("physics", "mjc"))}
    extra = ""
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get()
        extra = f" mesh pts={len(pts) if pts else 0} faces={len(m.GetFaceVertexCountsAttr().Get() or [])}"
    print(f"{prim.GetPath()} [{prim.GetTypeName()}] {schemas}{extra}")
    for k, v in attrs.items():
        print(f"     {k} = {v}")
    for k, v in rels.items():
        print(f"     rel {k} -> {v}")
