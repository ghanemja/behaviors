"""Embed each asset's NVIDIA-provided thumbnail with SigLIP2 (the model family NVIDIA's USD Search uses)."""
import glob
import json
import os
import sys

import numpy as np
import open_clip
import torch
from PIL import Image

ASSETS, OUT = sys.argv[1], sys.argv[2]
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-SigLIP2", pretrained="webli")
model.eval()
names, embs = [], []
for p in sorted(glob.glob(f"{ASSETS}/*.json")):
    if os.path.basename(p).startswith("_"):
        continue
    a = json.load(open(p))
    if not a.get("thumb"):
        continue
    im = Image.open(a["thumb"]).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    with torch.no_grad():
        e = model.encode_image(preprocess(im)[None])
    names.append(a["name"])
    embs.append(torch.nn.functional.normalize(e, dim=-1)[0].numpy())
os.makedirs(OUT, exist_ok=True)
np.savez(f"{OUT}/appearance.npz", names=np.array(names), emb=np.stack(embs))
print("embedded", len(names))
