import json
import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

REPO = "nvidia/simready-assets"
tree_path, out_dir, list_path = sys.argv[1], sys.argv[2], sys.argv[3]
tree = json.load(open(tree_path))
assets = [a.strip() for a in open(list_path) if a.strip()]


def needed(path):
    if "/Textures/" in path or "/source_assets/" in path or path.endswith(".mdl"):
        return False
    if "/.thumbs/" in path:
        return path.endswith("_thumbnail.png")
    if "/payloads/Newton/" in path or "/payloads/PhysX/" in path or path.endswith(("instances_newton.usda", "instances_physx.usda")):
        return False
    return path.endswith((".usd", ".usda", ".usdc", ".json"))


jobs = [p for a in assets for p, _ in tree[a] if needed(p)]
print("files to fetch:", len(jobs))


def fetch(p):
    for attempt in range(4):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=out_dir)
            return None
        except Exception as e:
            err = e
    return f"{p}: {type(err).__name__}"


with ThreadPoolExecutor(8) as ex:
    errs = [e for e in ex.map(fetch, jobs) if e]
print("errors:", len(errs))
for e in errs[:10]:
    print(" ", e)
