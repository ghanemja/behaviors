#!/usr/bin/env bash
# Reproduce every number in the write-up from scratch on a CPU-only machine (about 6-8 hours on 4 cores).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p data/meta data/v2/results

# 0. File listing of nvidia/simready-assets (Hugging Face) -> data/meta/nv_tree.json
$PY src/survey_nv.py data/meta

# 1. Stream all 2,576 assets in batches: download what the simulators need, convert each under the MuJoCo and PhysX
#    physics variants (data/v2/assets_mj, data/v2/assets_px), keep thumbnails, delete raw files.
$PY src/full_pipeline.py 100

# 2. Physics sanity: boxes with textbook answers must behave correctly in both engines.
$PY src/sanity.py data/v2/sanity.json

# 3. Probe battery. MuJoCo with per-part convex hulls (NVIDIA's MuJoCo variant), 3 jittered repeats per probe.
$PY src/sim.py data/v2/assets_mj data/v2/sim_mj --engine mujoco --geom parts
#    PhysX 5 with the same hulls (engine-only comparison), every asset, 1 repeat.
$PY src/sim.py data/v2/assets_px data/v2/sim_px_parts --engine physx --geom parts --repeats 1
#    PhysX 5 with SDF collision (NVIDIA's PhysX variant) on a fixed random sample of 600 assets.
$PY src/sim.py data/v2/assets_px data/v2/sim_px_sdf --engine physx --geom sdf --repeats 1 --shuffle \
    --only $(cat data/v2/px_sdf_sample.txt)

# 4. Embeddings: frozen-hyperparameter JEPA (3 seeds; trained on all assets, and on the 276 dev assets only),
#    and the SigLIP2 appearance baseline.
$PY src/train.py data/v2/sim_mj data/v2/results/emb.npz --dev data/v2/dev_assets.txt --seeds 0,1,2
$PY src/appearance.py data/v2/assets_mj data/v2/results

# 5. Evaluation on test assets (everything except the dev set), data-quality checks, findings, demo export.
$PY src/evaluate.py --sim data/v2/sim_mj --assets data/v2/assets_mj --emb data/v2/results/emb.npz \
    --appearance data/v2/results/appearance.npz --dev data/v2/dev_assets.txt \
    --px_parts data/v2/sim_px_parts --px_sdf data/v2/sim_px_sdf --px_sample data/v2/px_sdf_sample.txt \
    --out data/v2/results/eval.json
$PY src/qa.py --assets data/v2/assets_mj --assets_px data/v2/assets_px --sim_mj data/v2/sim_mj \
    --sim_px data/v2/sim_px_parts --out data/v2/results/qa.json
$PY src/analysis.py --sim data/v2/sim_mj --assets data/v2/assets_mj --emb data/v2/results/emb.npz \
    --appearance data/v2/results/appearance.npz --qa data/v2/results/qa.json --out data/v2/results/findings.json
