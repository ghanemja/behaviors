"""Keep simulations caught up with the streaming converter: every pass, simulate any newly converted asset in
MuJoCo (per-part hulls, 3 repeats) and PhysX (SDF collision, 1 repeat). Exits once the pipeline has finished and
everything converted has been simulated."""
import glob
import os
import subprocess
import sys
import time

B = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG = sys.argv[1]
WHICH = int(sys.argv[2]) if len(sys.argv) > 2 else None  # run only one entry of RUNS
SDF_SAMPLE = set(open(f"{B}/data/v2/px_sdf_sample.txt").read().split())  # fixed random 600 of all 2,576 assets
RUNS = [("assets_mj", "sim_mj", ["--engine", "mujoco", "--geom", "parts", "--workers", "2"], None),
        ("assets_px", "sim_px_parts", ["--engine", "physx", "--geom", "parts", "--repeats", "1", "--workers", "2"], None),
        ("assets_px", "sim_px_sdf", ["--engine", "physx", "--geom", "sdf", "--repeats", "1", "--workers", "2", "--shuffle"], SDF_SAMPLE)]


def pending(assets, sim):
    have = {os.path.basename(p)[:-5] for p in glob.glob(f"{B}/data/v2/{sim}/*.json")}
    fails = set()
    fp = f"{B}/data/v2/{sim}/_fails.json"
    if os.path.exists(fp):
        import json
        fails = set(json.load(open(fp)))
    return [os.path.basename(p)[:-5] for p in glob.glob(f"{B}/data/v2/{assets}/*.json")
            if not os.path.basename(p).startswith("_") and os.path.basename(p)[:-5] not in have | fails
            and os.path.exists(p.replace(".json", ".parts.npz"))]


while True:
    finished = "all done" in open(LOG).read()
    runs = RUNS if WHICH is None else [RUNS[WHICH]]
    work = [(a, s, extra, [n for n in pending(a, s) if keep is None or n in keep]) for a, s, extra, keep in runs]
    if finished and not any(w[3] for w in work):
        print("caught up", flush=True)
        break
    procs = []
    for a, s, extra, todo in work:
        if todo:
            procs.append(subprocess.Popen([PY, f"{B}/src/sim.py", f"{B}/data/v2/{a}", f"{B}/data/v2/{s}", *extra, "--only", *todo],
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    for p in procs:
        out = p.communicate()[0]
        print(time.strftime("%H:%M"), [l for l in out.splitlines() if l.startswith("simulated")], flush=True)
    if not procs:
        time.sleep(60)
