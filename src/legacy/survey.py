import collections
import sys

from huggingface_hub import HfApi

api = HfApi()
for repo in sys.argv[1:]:
    print(f"== {repo} ==")
    try:
        files = [f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(f, "size")]
    except Exception as e:
        print("  error:", type(e).__name__, str(e)[:200])
        continue
    ext = collections.Counter(f.path.rsplit(".", 1)[-1].lower() if "." in f.path.split("/")[-1] else "" for f in files)
    total = sum(f.size or 0 for f in files)
    usd = [f for f in files if f.path.lower().endswith((".usd", ".usda", ".usdc", ".usdz"))]
    tops = collections.Counter(f.path.split("/")[0] for f in files)
    print(f"  files={len(files)} total={total/1e9:.2f} GB usd={len(usd)} usd_bytes={sum(f.size or 0 for f in usd)/1e9:.2f} GB")
    print("  ext:", dict(ext.most_common(10)))
    print("  top-level dirs:", len(tops), list(tops)[:12])
    for f in usd[:6]:
        print("   ", f.path, f.size)
