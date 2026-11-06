---
license: other
pretty_name: Maniphys Lab V2D Challenge Track 2 submission
---

# Maniphys Lab: V2D Challenge Track 2 submission

This repository holds the verification container and the technical report of team Maniphys Lab for Track 2 (Robotic Grounding) of the NVIDIA Video to Data (V2D) Challenge.

| File | Contents |
|---|---|
| `maniphys-lab-v2d-submission-v2.tar.zst` | **Use this file.** The Docker image (tags `maniphys-lab-v2d-submission:v2` and `maniphys-lab-v2d-submission:final`), saved with `docker save` and compressed with `zstd`. |
| `maniphys-lab-v2d-submission-v2.tar.zst.sha256` | sha256 of the v2 tarball and the image ID. |
| `maniphys-lab-v2d-submission.tar.zst` | v1 of the image, sent on 2026-11-04. Kept for the record. Do not use it to reproduce rollouts (see "Changes in v2"). |
| `maniphys-lab-v2d-submission.tar.zst.sha256` | sha256 of the v1 tarball and the image ID. |
| `v2d-track2-technical-report.pdf` | The technical report. |

## Load and run

```bash
sha256sum -c maniphys-lab-v2d-submission-v2.tar.zst.sha256
zstd -dc maniphys-lab-v2d-submission-v2.tar.zst | docker load
# verify one tier (no GPU); /path/to/dataset holds track_2/
docker run --rm --network none -v /path/to/dataset:/data:ro \
    -e MANIFEST=/submission/submission.tier1.json maniphys-lab-v2d-submission:v2 \
    /submission/run.sh verify --dataset-root /data
# reproduce one tier (GPU)
docker run --rm --gpus all --network none -v /path/to/dataset:/data:ro -v "$PWD/out:/output" \
    -e MANIFEST=/submission/submission.tier1.json maniphys-lab-v2d-submission:v2 \
    /submission/run.sh reproduce --label tier1 --dataset-root /data --output /output
```

The container covers three counted files, one per tier: Tier 1 `t1x1102`, Tier 2 `b1014` and Tier 3 `t3m1029a`. Each file was packed from its own code commit, so the container has one manifest per tier. `/submission/README.md` inside the image explains the layout, the manifests and the `reproduce` command.

## Changes in v2

In v1, `reproduce` failed: each checkpoint loads its reference motion from `/repo/runs/motion`, and v1 did not contain that directory. v2 adds one image layer on top of v1 (`FROM` the v1 image `sha256:d870d3ab...`) with the files that the evaluator reads from `/repo/runs/motion`. These are 540 files:

- the reference motion parquet of each of the 90 checkpoints (`task.parquet` in the checkpoint's resolved config),
- the object mesh files that each parquet points to (`textured_mesh.urdf`, `textured_mesh.obj`, `material.mtl`, `material_0.png`),
- the support surface (`reconstructed_stage/<sequence>_support.usda`) of each episode that has scene support turned on.

Nothing under `/submission` or `/workspace` changes. The checkpoints, manifests, stored outputs and submitted files are the same bytes as in v1.

Checks on v2:

- `run.sh verify --dataset-root` passes for all three tiers (11 pass, 0 fail, 0 skip each), as for v1.
- Inside the image, the evaluator's own `load_reference` and `support_usda_for_reference` resolve every file for all 90 checkpoints.
- On an RTX 4090, the evaluator ran one episode per tier to completion with only these files mounted at `/repo/runs/motion` (`--network none`): Tier 1 episode 1, Tier 2 episode 9, Tier 3 episode 7.

Code: https://github.com/Manifold-ai/maniphys_lab_v2d_challenge
