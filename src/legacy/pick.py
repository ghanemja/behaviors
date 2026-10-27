import collections
import json
import random
import sys

rows = json.load(open(sys.argv[1]))
CAPS = {
    "food": 30, "kitchenware": 26, "box": 14, "handTool": 16, "consumable": 12, "paint": 10, "pallet": 5,
    "container": 10, "crate": 8, "cleaning": 10, "drum": 6, "pail": 5, "bucket": 5, "bottle": 6, "trashCan": 6,
    "toy": 9, "dishware": 9, "utensil": 6, "utensils": 4, "sign": 8, "lamp": 5, "safety": 8, "fireProtection": 5,
    "ppe": 6, "tape": 5, "stationery": 8, "chair": 5, "ladder": 3, "powerTool": 8, "storage": 6, "medicine": 5,
    "misc": 6, "decor": 6, "device": 5, "instrument": 6, "testingTool": 5, "serviceTool": 6, "equipment": 5,
}
MAX_BYTES = 12_000_000
rng = random.Random(7)
by_cat = collections.defaultdict(lambda: collections.defaultdict(list))
for asset, base, nbytes, geo, has_mj, has_thumb in rows:
    cat = base.split("_")[0]
    if cat in CAPS and has_mj and has_thumb and nbytes <= MAX_BYTES:
        by_cat[cat][base].append(asset)

picked = []
for cat, cap in CAPS.items():
    bases = list(by_cat[cat].items())
    rng.shuffle(bases)
    for lst in bases:
        lst[1].sort()
    chosen = []
    depth = 0
    while len(chosen) < cap and any(len(v) > depth for _, v in bases):
        for _, variants in bases:
            if depth < len(variants) and len(chosen) < cap:
                chosen.append(variants[depth])
        depth += 1
    picked += chosen
print(len(picked), "assets picked")
print(collections.Counter(a.split("_")[0] for a in picked))
open(sys.argv[2], "w").write("\n".join(picked) + "\n")
