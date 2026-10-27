import json
import sys

from huggingface_hub import hf_hub_download

REPO = "nvidia/simready-assets"
tree_path, out_dir, asset = sys.argv[1], sys.argv[2], sys.argv[3]
tree = json.load(open(tree_path))
files = tree[asset]
for path, size in files:
    if "/Textures/" in path or "/.thumbs/256" in path or "/.thumbs/512" in path:
        continue
    print(f"  {size:>12}  {path}")

wanted = [p for p, s in files if (
    p.endswith(("LICENSE.txt", "packaging.json", "root_usds.json"))
    or "/payloads/" in p and p.endswith(".usda") and s < 200_000
    or p.endswith("_thumbnail.png")
    or (p.endswith((".usd", ".usda")) and "/payloads/" not in p and s < 200_000)
)]
for p in wanted:
    hf_hub_download(REPO, p, repo_type="dataset", local_dir=out_dir)
print("downloaded", len(wanted), "small files to", out_dir)
