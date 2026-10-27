import os
import sys

from pxr import Usd, UsdGeom, UsdShade

RAW = sys.argv[1]


def src_of(inp):
    if not inp:
        return None
    conns = inp.GetConnectedSources()[0] if inp.HasConnectedSource() else []
    return conns[0].source if conns else None


def asset_attr_path(attr):
    v = attr.Get()
    if v is None:
        return None
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    layer = stack[0].layer.realPath if stack else ""
    return os.path.normpath(os.path.join(os.path.dirname(layer), v.path)) if v.path else None


for asset in sys.argv[2:]:
    root = [f for f in os.listdir(f"{RAW}/{asset}/simready_usd") if f.endswith(".usd")][0]
    st = Usd.Stage.Open(f"{RAW}/{asset}/simready_usd/{root}")
    print("==", asset)
    for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(p)
        pv = UsdGeom.PrimvarsAPI(p)
        uvs = [(x.GetPrimvarName(), x.GetInterpolation(), x.IsIndexed()) for x in pv.GetPrimvars() if x.GetTypeName().role == "TextureCoordinate" or "st" in str(x.GetPrimvarName())]
        subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
        mat, rel = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        print(f"  mesh {p.GetName()} purpose={UsdGeom.Imageable(p).ComputePurpose()} uvs={uvs} subsets={len(subsets)} mat={mat.GetPrim().GetPath() if mat else None}")
        if not mat:
            continue
        for sh in Usd.PrimRange(mat.GetPrim()):
            if not sh.IsA(UsdShade.Shader):
                continue
            s = UsdShade.Shader(sh)
            sid = s.GetShaderId()
            impl = s.GetImplementationSourceAttr().Get()
            mdl = s.GetSourceAsset("mdl")
            ins = {}
            for i in s.GetInputs():
                n = i.GetBaseName()
                if i.HasConnectedSource():
                    ins[n] = "<-" + str(src_of(i).GetPath().name)
                else:
                    v = i.Get()
                    if v is not None and not isinstance(v, (list,)):
                        if "file" in n or "texture" in n:
                            ins[n] = asset_attr_path(i.GetAttr())
                        else:
                            ins[n] = str(v)[:40]
            print(f"     shader {sh.GetName()} id={sid} impl={impl} mdl={mdl.path if mdl else ''}")
            for k, v in list(ins.items())[:14]:
                print(f"         {k} = {v}")
