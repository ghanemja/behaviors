"""Build one textured, decimated GLB per asset (same body frame as the simulation) for the web viewer."""
import glob
import io
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pymeshlab
import trimesh
from huggingface_hub import hf_hub_download
from PIL import Image
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

RAW, ASSETS, TREE, OUT = sys.argv[1:5]
ONLY = sys.argv[5:]
REPO = "nvidia/simready-assets"
TRI_BUDGET = 12000
os.makedirs(OUT, exist_ok=True)
tree = json.load(open(TREE))
tree_paths = {p for files in tree.values() for p, _ in files}


def mat_np(m):
    return np.array([[m[i][j] for j in range(4)] for i in range(4)])


def owner_body(prim):
    p = prim
    while p and p.IsValid():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            return p
        p = p.GetParent()
    return None


def connected(inp):
    if inp and inp.HasConnectedSource():
        srcs = inp.GetConnectedSources()[0]
        if srcs:
            return UsdShade.Shader(srcs[0].source.GetPrim())
    return None


def tex_file(shader):
    inp = shader.GetInput("file")
    if not inp:
        return None
    attr = inp.GetAttr()
    v = attr.Get()
    if v is None or not v.path:
        return None
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    base = os.path.dirname(stack[0].layer.realPath) if stack else ""
    local = os.path.normpath(os.path.join(base, v.path))
    return os.path.relpath(local, RAW)


def uv_xform(tex_shader):
    st = connected(tex_shader.GetInput("st"))
    if st and st.GetShaderId() == "UsdTransform2d":
        g = lambda n, d: (st.GetInput(n).Get() if st.GetInput(n) and st.GetInput(n).Get() is not None else d)
        return np.array(g("scale", (1, 1)), float), float(g("rotation", 0.0)), np.array(g("translation", (0, 0)), float)
    return None


def material_info(mat):
    info = {"albedo": None, "orm": None, "normal": None, "color": [0.7, 0.7, 0.7], "tint": [1, 1, 1],
            "rough": 0.6, "metal": 0.0, "glass": False, "xf": None, "name": mat.GetPrim().GetName() if mat else "none"}
    if not mat:
        return info
    for sh in Usd.PrimRange(mat.GetPrim()):
        if sh.IsA(UsdShade.Shader):
            src = UsdShade.Shader(sh).GetSourceAsset("mdl")
            if src and "Glass" in src.path:
                info["glass"] = True
    surf = mat.ComputeSurfaceSource()[0]
    if not surf or surf.GetShaderId() != "UsdPreviewSurface":
        return info
    def const(name):
        i = surf.GetInput(name)
        return i.Get() if i and not i.HasConnectedSource() and i.Get() is not None else None
    t = connected(surf.GetInput("diffuseColor"))
    if t and t.GetShaderId() == "UsdUVTexture":
        info["albedo"] = tex_file(t)
        info["xf"] = uv_xform(t)
        sc = t.GetInput("scale")
        if sc and sc.Get() is not None:
            info["tint"] = list(sc.Get())[:3]
    elif const("diffuseColor") is not None:
        info["color"] = list(const("diffuseColor"))
    for key, inp in (("orm", "roughness"), ("normal", "normal")):
        s = connected(surf.GetInput(inp))
        if s and s.GetShaderId() == "UsdUVTexture":
            info[key] = tex_file(s)
    if const("roughness") is not None:
        info["rough"] = float(const("roughness"))
    if const("metallic") is not None:
        info["metal"] = float(const("metallic"))
    return info


def best_variant(rel):
    if rel is None:
        return None
    d, b = os.path.dirname(rel), os.path.basename(rel)
    for cand in (f"{d}/.thumbs/512x512/{b}.png", f"{d}/.thumbs/256x256/{b}.png", rel):
        if cand in tree_paths:
            return cand
    return None


def extract(asset):
    roots = glob.glob(f"{RAW}/{asset}/simready_usd/*.usd")
    st = Usd.Stage.Open(roots[0])
    root = st.GetDefaultPrim()
    vs = root.GetVariantSets()
    for k, v in (("PhysX", "Disabled"), ("Newton", "Disabled"), ("MuJoCo", "Enabled")):
        if vs.HasVariantSet(k):
            vs.GetVariantSet(k).SetVariantSelection(v)
    mpu = UsdGeom.GetStageMetersPerUnit(st) or 1.0
    to_z = np.eye(4)
    if UsdGeom.GetStageUpAxis(st) == "Y":
        to_z = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    S = np.diag([mpu, mpu, mpu, 1.0])
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    Mref_inv = mat_np(xc.GetLocalToWorldTransform(root).GetInverse())
    parts = []
    for p in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or owner_body(p) is None:
            continue
        purpose = UsdGeom.Imageable(p).ComputePurpose()
        if purpose not in ("default", "render"):
            continue
        mesh = UsdGeom.Mesh(p)
        pts, counts, idx = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        M = mat_np(xc.GetLocalToWorldTransform(p)) @ Mref_inv @ S @ to_z
        V = (np.c_[np.array(pts, float), np.ones(len(pts))] @ M)[:, :3]
        counts, idx = np.array(counts), np.array(idx)
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        tri_c, tri_face = [], []
        for fi, (s0, c) in enumerate(zip(starts, counts)):
            for t in range(1, c - 1):
                tri_c.append((s0, s0 + t, s0 + t + 1))
                tri_face.append(fi)
        tri_c, tri_face = np.array(tri_c), np.array(tri_face)
        F = idx[tri_c]
        if np.linalg.det(M[:3, :3]) < 0:
            F = F[:, ::-1]
            tri_c = tri_c[:, ::-1]
        uvc = None
        pv = UsdGeom.PrimvarsAPI(p)
        st_pv = pv.GetPrimvar("st") if pv.HasPrimvar("st") else next((x for x in pv.GetPrimvars() if x.GetTypeName().role == "TextureCoordinate"), None)
        if st_pv:
            vals = st_pv.ComputeFlattened()
            if vals is not None and len(vals):
                vals = np.array(vals, float)
                interp = st_pv.GetInterpolation()
                if interp == "faceVarying" and len(vals) == len(idx):
                    uvc = vals[tri_c]
                elif interp in ("vertex", "varying") and len(vals) == len(pts):
                    uvc = vals[F]
                elif interp == "uniform" and len(vals) == len(counts):
                    uvc = np.repeat(vals[tri_face][:, None], 3, 1)
        subsets = [s for s in UsdGeom.Subset.GetAllGeomSubsets(mesh) if s.GetFamilyNameAttr().Get() == "materialBind"]
        groups = []
        if subsets:
            for s in subsets:
                fids = set(s.GetIndicesAttr().Get() or [])
                sel = np.array([f in fids for f in tri_face])
                mat = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
                groups.append((sel, mat))
        else:
            groups.append((np.ones(len(F), bool), UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]))
        for sel, mat in groups:
            if not sel.any():
                continue
            info = material_info(mat)
            u = uvc[sel] if uvc is not None else None
            if u is not None and info["xf"] is not None:
                sc, rot, tr = info["xf"]
                u = u * sc
                c, s_ = math.cos(math.radians(rot)), math.sin(math.radians(rot))
                u = np.stack([u[..., 0] * c - u[..., 1] * s_, u[..., 0] * s_ + u[..., 1] * c], -1) + tr
            parts.append({"V": V, "F": F[sel], "UV": u, "mat": info})
    return parts


def crease_normals(V, F, angle=38):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fu = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-20)
    inc = [[] for _ in range(len(V))]
    for f, tri in enumerate(F):
        for v in tri:
            inc[v].append(f)
    cosT = math.cos(math.radians(angle))
    N = np.zeros((len(F), 3, 3))
    for f, tri in enumerate(F):
        for k, v in enumerate(tri):
            nb = np.array(inc[v])
            ok = nb[(fu[nb] @ fu[f]) > cosT]
            n = fn[ok].sum(0)
            N[f, k] = n / (np.linalg.norm(n) + 1e-20)
    return N


def decimate(V, F, UV, target):
    if len(F) <= target:
        return V, F, UV
    try:
        ms = pymeshlab.MeshSet()
        if UV is not None:
            tmp = os.path.join(OUT, "_tmp")
            os.makedirs(tmp, exist_ok=True)
            Image.new("RGB", (2, 2), (200, 200, 200)).save(f"{tmp}/t.png")
            open(f"{tmp}/m.mtl", "w").write("newmtl m\nKd 1 1 1\nmap_Kd t.png\n")
            cuv, inv = np.unique(np.round(UV.reshape(-1, 2), 6), axis=0, return_inverse=True)
            vt_idx = inv.reshape(-1, 3) + 1
            with open(f"{tmp}/m.obj", "w") as fh:
                fh.write("mtllib m.mtl\n")
                fh.write("".join(f"v {a:.7f} {b:.7f} {c:.7f}\n" for a, b, c in V))
                fh.write("".join(f"vt {a:.6f} {b:.6f}\n" for a, b in cuv))
                fh.write("usemtl m\n")
                fh.write("".join(f"f {f0+1}/{t0} {f1+1}/{t1} {f2+1}/{t2}\n" for (f0, f1, f2), (t0, t1, t2) in zip(F, vt_idx)))
            ms.load_new_mesh(f"{tmp}/m.obj")
        else:
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32)))
        if UV is not None:
            ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=int(target), qualitythr=0.4, extratcoordw=1.0,
                                                                     preserveboundary=True, optimalplacement=True, preservenormal=True)
        else:
            ms.meshing_remove_duplicate_vertices()
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target), qualitythr=0.4, preserveboundary=True,
                                                        preservenormal=True, optimalplacement=True, planarquadric=True)
        cm = ms.current_mesh()
        V2, F2 = cm.vertex_matrix(), cm.face_matrix()
        UV2 = cm.wedge_tex_coord_matrix().reshape(-1, 3, 2) if UV is not None else None
        if len(F2) < 8:
            raise ValueError("degenerate")
        return V2, F2, UV2
    except Exception as e:
        print("   decimate fallback:", type(e).__name__, str(e)[:80])
        return V, F, UV


_img_cache = {}


def load_img(rel, size, mode, quality):
    key = (rel, size, mode)
    if rel is None:
        return None
    if key in _img_cache:
        return _img_cache[key]
    path = f"{RAW}/{rel}"
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert(mode)
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    out = Image.open(io.BytesIO(buf.getvalue()))
    _img_cache[key] = out
    return out


def build(asset):
    parts = extract(asset)
    if not parts:
        return None
    total = sum(len(p["F"]) for p in parts)
    scene = trimesh.Scene()
    stats = {"tris_in": int(total), "tris_out": 0, "textured": 0, "parts": len(parts)}
    for k, p in enumerate(parts):
        target = max(150, int(TRI_BUDGET * len(p["F"]) / total)) if total > TRI_BUDGET else len(p["F"])
        used = np.unique(p["F"])
        remap = -np.ones(len(p["V"]), int)
        remap[used] = np.arange(len(used))
        V, F, UV = decimate(p["V"][used], remap[p["F"]], p["UV"], target)
        N = crease_normals(V, F)
        cV = V[F].reshape(-1, 3)
        cN = N.reshape(-1, 3)
        cUV = UV.reshape(-1, 2) if UV is not None else np.zeros((len(cV), 2))
        key = np.c_[F.reshape(-1), np.round(cUV * 1e4), np.round(cN * 500)]
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        Vn, Nn, UVn = cV[first], cN[first], cUV[first]
        Fn = inv.reshape(-1, 3)
        info = p["mat"]
        alb = load_img(best_variant(info["albedo"]), 512, "RGB", 84) if UV is not None else None
        orm = load_img(best_variant(info["orm"]), 256, "RGB", 85) if alb is not None else None
        nrm = load_img(best_variant(info["normal"]), 512, "RGB", 88) if alb is not None else None
        base = info["tint"] if alb is not None else info["color"]
        alpha = 0.35 if info["glass"] else 1.0
        if info["glass"]:
            base = [0.86, 0.92, 0.96]
        mat = PBRMaterial(name=f"m{k}", baseColorFactor=[int(255 * min(max(c, 0), 1)) for c in base] + [int(255 * alpha)],
                          baseColorTexture=alb, metallicRoughnessTexture=orm, normalTexture=nrm,
                          metallicFactor=1.0 if orm is not None else info["metal"], roughnessFactor=1.0 if orm is not None else info["rough"],
                          alphaMode="BLEND" if info["glass"] else "OPAQUE", doubleSided=bool(info["glass"]))
        if alb is not None:
            stats["textured"] += 1
        m = trimesh.Trimesh(vertices=Vn, faces=Fn, vertex_normals=Nn, visual=TextureVisuals(uv=UVn, material=mat), process=False)
        scene.add_geometry(m, node_name=f"part{k}")
        stats["tris_out"] += int(len(Fn))
    glb = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    open(f"{OUT}/{asset}.glb", "wb").write(glb)
    stats["bytes"] = len(glb)
    return stats


if __name__ == "__main__":
    assets = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{ASSETS}/*.json") if not os.path.basename(p).startswith("_"))
    sim_dir = next((d for d in (f"{ASSETS}/../sim_mj", f"{ASSETS}/../sim") if os.path.isdir(d)), None)
    sims = {os.path.basename(p)[:-4] for p in glob.glob(f"{sim_dir}/*.npz")} if sim_dir else set(assets)
    assets = [a for a in assets if a in sims]
    if ONLY:
        assets = [a for a in assets if a in ONLY]
    print("scanning", len(assets), "assets for textures")
    need = set()
    cache = {}
    for a in assets:
        try:
            cache[a] = extract(a)
        except Exception as e:
            print("extract fail", a, e)
            continue
        for p in cache[a]:
            for key in ("albedo", "orm", "normal"):
                v = best_variant(p["mat"][key])
                if v and not os.path.exists(f"{RAW}/{v}"):
                    need.add(v)
    print("texture files to fetch:", len(need))

    def fetch(rel):
        for _ in range(3):
            try:
                hf_hub_download(REPO, rel, repo_type="dataset", local_dir=RAW)
                return None
            except Exception as e:
                err = e
        return rel
    with ThreadPoolExecutor(8) as ex:
        bad = [r for r in ex.map(fetch, sorted(need)) if r]
    print("fetch failures:", len(bad))
    report = {}
    for a in assets:
        try:
            report[a] = build(a)
        except Exception as e:
            print("build fail", a, type(e).__name__, str(e)[:120])
    json.dump(report, open(f"{OUT}/_report.json", "w"), indent=0)
    ok = [r for r in report.values() if r]
    print(f"built {len(ok)}; tris median {int(np.median([r['tris_out'] for r in ok]))}; "
          f"textured parts {sum(r['textured'] for r in ok)}/{sum(r['parts'] for r in ok)}; total {sum(r['bytes'] for r in ok)/1e6:.1f} MB; "
          f"median {np.median([r['bytes'] for r in ok])/1e3:.0f} KB, max {max(r['bytes'] for r in ok)/1e6:.2f} MB")
