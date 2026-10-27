"""Stream the whole SimReady dataset in batches: download what the simulators need, convert each asset under the
MuJoCo and PhysX physics variants, keep the thumbnail, then delete the raw files so the run fits on disk."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
REPO = "nvidia/simready-assets"
RAW = f"{B}/data/raw/stream"
OUT = {"MuJoCo": f"{B}/data/v2/assets_mj", "PhysX": f"{B}/data/v2/assets_px"}
THUMBS = f"{B}/data/v2/thumbs"
BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 100
for d in (RAW, THUMBS, *OUT.values()):
    os.makedirs(d, exist_ok=True)

tree = json.load(open(f"{B}/data/meta/nv_tree.json"))


def needed(p):
    if "/Textures/" in p or "/source_assets/" in p or p.endswith(".mdl"):
        return False
    if "/.thumbs/" in p:
        return p.endswith("_thumbnail.png")
    if "/payloads/Newton/" in p or p.endswith("instances_newton.usda"):
        return False
    return p.endswith((".usd", ".usda", ".usdc", ".json"))


def fetch(p):
    err = None
    for attempt in range(6):
        try:
            hf_hub_download(REPO, p, repo_type="dataset", local_dir=RAW)
            return None
        except Exception as e:
            err = e
            time.sleep(min(60, 2 ** attempt) if "429" in str(e) else 1)
    return f"{p}: {type(err).__name__} {str(err)[:80]}"


def converted():
    seen = set()
    for d in OUT.values():
        seen |= {os.path.basename(p)[:-5] for p in glob.glob(f"{d}/*.json") if not os.path.basename(p).startswith("_")}
        sk = f"{d}/_skipped.json"
        if os.path.exists(sk):
            seen |= set(json.load(open(sk)))
    return seen


def finish(asset, raw_dir):
    th = glob.glob(f"{raw_dir}/{asset}/simready_usd/.thumbs/*_thumbnail.png")
    dst = f"{THUMBS}/{asset}.png"
    if th:
        shutil.copy(th[0], dst)
    for d in OUT.values():
        p = f"{d}/{asset}.json"
        if os.path.exists(p):
            rec = json.load(open(p))
            rec["thumb"] = dst if th else None
            json.dump(rec, open(p, "w"))


assets = sorted(a for a in tree if not a.startswith("."))
todo = [a for a in assets if a not in converted()]
print(f"{len(todo)} assets to process in batches of {BATCH}", flush=True)
t_start = time.time()
for b0 in range(0, len(todo), BATCH):
    batch = todo[b0:b0 + BATCH]
    jobs = [p for a in batch for p, _ in tree[a] if needed(p)]
    t0 = time.time()
    with ThreadPoolExecutor(32) as ex:
        errs = [e for e in ex.map(fetch, jobs) if e]
    t1 = time.time()
    procs = [subprocess.Popen([PY, f"{B}/src/convert.py", RAW, OUT[v], *batch], env={**os.environ, "SR_VARIANT": v},
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for v in OUT]
    logs = [p.communicate()[0] for p in procs]
    for a in batch:
        finish(a, RAW)
        shutil.rmtree(f"{RAW}/{a}", ignore_errors=True)
    shutil.rmtree(f"{RAW}/.cache", ignore_errors=True)
    summary = [l.strip().splitlines()[-1] if l.strip() else "?" for l in logs]
    done = b0 + len(batch)
    eta = (time.time() - t_start) / done * (len(todo) - done) / 60
    print(f"batch {b0 // BATCH + 1}: {len(batch)} assets, {len(jobs)} files, {len(errs)} fetch errors, "
          f"download {t1 - t0:.0f}s, convert {time.time() - t1:.0f}s | MJ {summary[0]} | PX {summary[1]} | "
          f"{done}/{len(todo)} done, ETA {eta:.0f} min", flush=True)
    for e in errs[:3]:
        print("   ", e, flush=True)
print("all done", flush=True)
