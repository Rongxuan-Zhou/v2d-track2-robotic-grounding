# Maniphys Lab: V2D Challenge Track 2 submission container

This container reproduces the three counted Track 2 submissions of team Maniphys Lab. Each tier has one counted file. The same file is selected as the Final Submission on all four metric competitions of its tier.

| Label | Track | Submitted file | Kaggle submission IDs (AUC, SP-SR, MP-SR, MPPE) | Code commit |
|---|---|---|---|---|
| tier1 | track_2_tier1 | submitted/track2_tier1_t1x1102.csv | 57920552, 57920554, 57920555, 57920556 | b59f6a6cb3f97fc53ebdb28e2e8496534478c108 |
| tier2 | track_2_tier2 | submitted/track2_tier2_select_b1014.csv | see submission.tier2.json | c4f248dc60f39f4c7b03c535fbace6ae6b422bc9 |
| tier3 | track_2_tier3 | submitted/track2_tier3_t3m1029a.csv | 57783386, 57783387, 57783388, 57783392 | 2175e83e0c13b9ae2391d724f3d1c80c392b73d3 |

Repository: https://github.com/Manifold-ai/maniphys_lab_v2d_challenge

## One manifest per tier

The three tiers were packed from three different commits. The `v2d_container_v1` manifest holds one code commit, so this container ships one manifest per tier:

- `submission.tier1.json` with `code_tier1/`
- `submission.tier2.json` with `code_tier2/`
- `submission.tier3.json` with `code_tier3/`

Each `code_tierN/` is a clean git checkout of that tier's commit, with its `.git` directory. `submission.json` is a copy of `submission.tier1.json`. Select a tier with the `MANIFEST` variable:

```bash
# verify (no GPU)
docker run --rm --network none -v /path/to/dataset:/data:ro -e MANIFEST=/submission/submission.tier2.json IMAGE \
    /submission/run.sh verify --dataset-root /data
# reproduce (GPU)
docker run --rm --gpus all --network none -v /path/to/dataset:/data:ro -v "$PWD/out:/output" \
    -e MANIFEST=/submission/submission.tier3.json IMAGE \
    /submission/run.sh reproduce --label tier3 --dataset-root /data --output /output
```

`/path/to/dataset` is the challenge dataset root, the directory that holds `track_2/`.

## Contents

- `weights/tierN/episode_XXXXXX.safetensors`: the 30 evaluated policy checkpoints per tier, byte for byte as evaluated (90 files in total).
- `outputs/tierN/rollouts/`: the evaluator outputs packed into each counted file.
- `frame_references/tier2`, `frame_references/tier3`: the `--frame-references` directories passed to the packer. Tier 2 uses our denoised trajectories. Tier 3 uses our monocular reconstruction, aligned at frame 0 as the organizers allowed on 10/21.
- `submitted/`: the exact CSV files uploaded to Kaggle and their upload receipts.
- `kit/`: the submission kit copy that packed and uploaded all three files (the 10/01 kit, zip sha256 0f2f9f4b...).
- `verifier_kit/`: the 11/02 kit, which provides `tools/verify_container.py`.

## Runtime

- The base image is the FlashCHORD image (`flash_chord:latest`) in which every counted episode was evaluated. The evaluator runtime is `/opt/venv` with the package at `/workspace/src/flash_chord/evaluation`.
- The kit's Python requirements are in a separate venv `/opt/kitvenv`, so the evaluator runtime is unchanged.
- Hardware: one NVIDIA GPU with at least 24 GB for `reproduce` (we evaluated on an L40S and an RTX 4090). `verify` needs no GPU.
- Run time: one episode takes minutes on one GPU, and a full tier takes a few hours with `--parallel 1`.
- The FlashCHORD simulator is not bitwise deterministic. Reproduced poses differ slightly from the submitted ones. The provenance columns must match exactly.
