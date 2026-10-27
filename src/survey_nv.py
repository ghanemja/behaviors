import collections
import json
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO = "nvidia/simready-assets"
api = HfApi()
out_dir = sys.argv[1]

readme = hf_hub_download(REPO, "README.md", repo_type="dataset", local_dir=out_dir) if False else None
info = api.dataset_info(REPO)
print("card license:", (info.card_data or {}).get("license") if info.card_data else None)
print("tags:", info.tags[:20])

files = [f for f in api.list_repo_tree(REPO, repo_type="dataset", recursive=True) if hasattr(f, "size")]
by_asset = collections.defaultdict(list)
for f in files:
    by_asset[f.path.split("/")[0]].append(f)
print("assets:", len(by_asset))

root_files = [f.path for f in files if "/" not in f.path]
print("root files:", root_files[:20])

prefix = collections.Counter(a.rsplit("_", 1)[0] for a in by_asset)
print("distinct base names:", len(prefix))
print(sorted(prefix)[:400])

summary = {}
for a, fs in by_asset.items():
    summary[a] = [(f.path, f.size) for f in fs]
with open(f"{out_dir}/nv_tree.json", "w") as fh:
    json.dump(summary, fh)

for a in list(by_asset)[100:103]:
    print("==", a)
    for f in by_asset[a]:
        print("   ", f.path, f.size)
