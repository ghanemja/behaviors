import collections
import json
import re
import sys

tree = json.load(open(sys.argv[1]))


def needed(path):
    if "/Textures/" in path or "/source_assets/" in path or path.endswith(".mdl"):
        return False
    if "/.thumbs/" in path:
        return path.endswith("_thumbnail.png")
    return path.endswith((".usd", ".usda", ".usdc", ".json", ".txt"))


rows = []
for asset, files in tree.items():
    if asset.startswith("."):
        continue
    keep = [(p, s) for p, s in files if needed(p)]
    geo = sum(s for p, s in keep if p.endswith((".usd", ".usdc")) and "/payloads/" in p)
    has_mj = any("/payloads/MuJoCo/enabled.usda" in p for p, _ in files)
    has_thumb = any(p.endswith("_thumbnail.png") for p, _ in files)
    base = re.sub(r"_[a-z]\d\d$", "", asset)
    rows.append((asset, base, sum(s for _, s in keep), geo, has_mj, has_thumb))

print("assets:", len(rows), " with MuJoCo payload:", sum(r[4] for r in rows), " with thumbnail:", sum(r[5] for r in rows))
sizes = sorted(r[2] for r in rows)
print("needed bytes per asset: median %.1f MB, p90 %.1f MB, total %.1f GB" % (
    sizes[len(sizes) // 2] / 1e6, sizes[int(len(sizes) * 0.9)] / 1e6, sum(sizes) / 1e9))
cats = collections.Counter(r[1].split("_")[0] for r in rows)
print("top-level categories:", cats.most_common(80))
json.dump(rows, open(sys.argv[2], "w"))
