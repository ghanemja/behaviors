"""Local data file for the video renderer: the demo index with every asset's tracks inlined."""
import glob
import json
import shutil
import sys

site, out = sys.argv[1], sys.argv[2]
d = json.load(open(f"{site}/data.json"))
tracks = {}
for p in glob.glob(f"{site}/d/tr_*.json"):
    tracks.update(json.load(open(p)))
for a in d["assets"]:
    a["tracks"] = tracks[a["id"]]
json.dump(d, open(f"{out}/data.json", "w"), separators=(",", ":"))
n = 0
for p in glob.glob(f"{site}/mt/*.txt"):
    shutil.copy(p, f"{out}/mt/")
    n += 1
print("assets", len(d["assets"]), "models", n)
