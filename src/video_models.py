"""Textured models for video objects that the demo site does not ship.

Downloads each asset's USD files from Hugging Face (as full_pipeline.py does), builds a GLB with build_glb.py and
stores it base64-encoded as <out>/mt/<id>.txt, where reel.html looks for it.
Usage: python src/video_models.py video/models.txt video
"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f"{B}/data/raw/nv"
REPO = "nvidia/simready-assets"
ids = open(sys.argv[1]).read().split()
MT = f"{sys.argv[2]}/mt"
os.makedirs(MT, exist_ok=True)
tree = json.load(open(f"{B}/data/meta/nv_tree.json"))


def needed(p):
    if "/Textures/" in p or "/source_assets/" in p or "/.thumbs/" in p or p.endswith(".mdl"):
        return False
    if "/payloads/Newton/" in p or p.endswith("instances_newton.usda"):
        return False
    return p.endswith((".usd", ".usda", ".usdc", ".json"))


def fetch(p):
    for k in range(6):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=RAW)
            return None
        except Exception as e:
            err = e
            time.sleep(min(30, 2 ** k))
    return f"{p}: {err}"


todo = [a for a in ids if not os.path.exists(f"{MT}/{a}.txt")]
files = [p for a in todo if not os.path.isdir(f"{RAW}/{a}/simready_usd") for p, _ in tree[a] if needed(p)]
with ThreadPoolExecutor(8) as ex:
    bad = [r for r in ex.map(fetch, files) if r]
if bad:
    print("download failures:", bad[:5])
with tempfile.TemporaryDirectory() as glb:
    subprocess.run([sys.executable, f"{B}/src/build_glb.py", RAW, f"{B}/data/v2/assets_mj", f"{B}/data/meta/nv_tree.json", glb, *todo], check=False)
    for a in todo:
        if os.path.exists(f"{glb}/{a}.glb"):
            open(f"{MT}/{a}.txt", "w").write(base64.b64encode(open(f"{glb}/{a}.glb", "rb").read()).decode())
print("models ready:", sum(os.path.exists(f"{MT}/{a}.txt") for a in ids), "of", len(ids))
