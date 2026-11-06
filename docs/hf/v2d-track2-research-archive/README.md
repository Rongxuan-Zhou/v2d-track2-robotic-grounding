---
license: cc-by-4.0
tags:
- archive
- v2d-challenge
- robotics
- humanoid
- dexterous-manipulation
pretty_name: V2D Track 2 research archive (Maniphys Lab)
---

# v2d-track2-research-archive

All experiment data of the Maniphys Lab entry to NVIDIA [Video to Data (V2D) Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) Track 2 (Robotic Grounding). We organized and uploaded it after the 2026-11-04 leaderboard freeze, for later research. The Track 2 task: the input is third-person 4D human-object interaction data of different quality. On the fixed Unitree G1 + Dex3 and the official simulator, train one robot policy for each episode. Each of the three input tiers has its own score.

- Tier 1: clean human and object trajectories.
- Tier 2: corrupted trajectories. You must denoise them before training.
- Tier 3: only the monocular third-person video (`observation.images.exo_camera`). The trajectories must be reconstructed from the video.

The three tiers have separate scores, and you cannot compare scores between tiers. The Tier 2 and Tier 3 pipelines do not read the clean Tier 1 trajectories. The only exception is the official answer of 10/21: Tier 3 can use Tier 1 frame 0 to compute the alignment. For Tier 3 ep12 and ep27, the choice between symmetric branches used Tier 1 frame 0, and this choice also entered the training reference. Section 5 of the technical report states this openly.

This repository holds only data and results. The code is in a private repository and is not public.

## Relation to other repositories

| Repository | Contents |
|---|---|
| This repository | All artifacts under `runs/` of each host: checkpoints, training logs, evaluation results, exported rollouts, motion data, Tier 3 reconstruction artifacts, submission CSVs, diagnostic output |
| [`Rongxuan-Zhou/maniphys-lab-v2d-track2-submission`](https://huggingface.co/datasets/Rongxuan-Zhou/maniphys-lab-v2d-track2-submission) | The submission container (docker image tar.zst, 8.4 GB) and the technical report PDF. The container holds the 90 checkpoints of the three Final Submissions and the evaluation environment, and it can reproduce the counted scores |
| [`Rongxuan-Zhou/v2d-track2-archive`](https://huggingface.co/Rongxuan-Zhou/v2d-track2-archive) | 1987 checkpoints (`state_*`, `policy_*`, about 27.5 GB), archived from mf4090 in 5 batches from 10/01 to 10/12 to free disk space (then deleted on mf4090). This repository does not store them again |

The challenge data is the official data that V2D Challenge released. The challenge rules state that it is used and shared under CC BY 4.0. This repository does not upload the raw data again. `INDEX/<host>/excluded.tsv` records the sha256 of each file of that data on each host. You can use it to check whether you have the same data.

## Results

The leaderboard froze at 2026-11-04 15:00 MST. The table below gives the hand-selected Final Submission of each tier:

| Tier | Submission file | Code commit |
|---|---|---|
| Tier 1 | `t1x1102` | `b59f6a6c` |
| Tier 2 | `b1014` | `c4f248dc` |
| Tier 3 | `t3m1029a` | `2175e83e` |

For the method, the changes of each recipe, the episode selection rules and the limitations, see the technical report (the PDF in the submission repository).

## Hosts and directories

| Prefix | Host | GPU | Contents |
|---|---|---|---|
| `mf4090/` | mf4090 | RTX 4090 24 GB | `repo/runs/` (Tier 1/2/3 training and evaluation), `v2d/envs/trajopt_pkgs.tar` (third-party Python packages for trajopt), `v2d/work/`, download and setup records under `home/`, `meta/` |
| `ml40/` | ml40 | L40S 48 GB | `repo/runs/`, `v2d/t3recon/` (the artifacts of each Tier 3 version reconstructed from video. The three Python environments `env.tar` for reconstruction contain member paths, so they are in the private repository), `v2d/container/` (build directory and logs of the submission container), `v2d/tier3_kfix*`, `ops/`, `meta/`. ml40 has been unreachable since the night of the 11/04 freeze. The data comes from its data disk, mounted on another CPU host |
| `mac/` | Local Mac | None | `repo/runs/`: submission CSVs and receipts, logs and evaluation results copied back from the training hosts (`runs/ml40/`, `runs/mf4090/`), Tier 3 world frame and visualization artifacts, `container.tar` (the staging directory of the submission container) |

Archive size of each host (counted from `INDEX/<host>/manifest.jsonl` after the upload completed on 11/05. File counts do not include `INDEX/`). After the upload of each host, we checked the size and sha256 of each file, then downloaded 5 random files and computed them again:

| Host | This repository | Private repository | Excluded (official data, caches and others) | Check |
|---|---|---|---|---|
| `mac/` | 8794 files, 2.36 GB | 13 files, 0.14 GB | 0.99 GB | Passed |
| `mf4090/` | 27298 files, 84.47 GB | 497 files, 0.78 GB | 6.07 GB | Passed |
| `ml40/` | 42561 files, 157.53 GB | 25547 files, 3.82 GB | 23.77 GB | Passed (in the first check, the recorded sha256 of 1 checkpoint was wrong. The content was correct. It passed after a new commit) |

The path in the repository is `<host>/<prefix>/<relative path on the host>`. `<host>/repo/` is the root of the code repository copy on the training host. Thus `mf4090/repo/runs/flash_chord/<run>/` is `runs/flash_chord/<run>/` on mf4090.

The main directories under `runs/`:

| Directory | Contents |
|---|---|
| `flash_chord/<run>/` | One training run: `policy_<step>.safetensors` (policy parameters, for export and evaluation), `state_<step>.safetensors` (resume state, must pair with the policy of the same step), `hydra/` (training configuration), `run_record.jsonl`, `eval_*_w4096.json` (evaluation in 4096 worlds), `export_8c778f9/` (rollouts exported with the fixed evaluator package 8c778f9), `wandb/` (local W&B records), `DONE`, `ARCHIVED.json` (which files went to `v2d-track2-archive`) |
| `exports/` | Records of the export chain |
| `submission/` | Submission CSVs, upload receipts, evaluator JSON, Tier 2/3 frame references |
| `motion/`, `motion_src/` | Reference motion of each tier and version (retargeted human and object trajectories) |
| `tier3/`, `t3world/` | Tier 3 reconstruction artifacts |
| `logs/` | Logs of the training chains, exports and evaluations |
| `diag/`, `diag2/`, `trajopt/`, `recovery/` | Intermediate results of diagnostics and Tier 2 denoising |
| `hf_src/`, `hf_pull/` | Files moved from other hosts through the HF transfer. For example, `hf_src/l40n_env2_1102` is the environment record of l40n |
| `<host>/` (only on the Mac) | Copies of logs and evaluation results copied back from the training hosts to the Mac. The ml40 copy is the main source while ml40 is unreachable |

Small directories with many files are packed as tar files (for example `mac/repo/runs/container.tar`, `mf4090/repo/runs/stage.tar`). In `INDEX/<host>/manifest.jsonl`, the `packed_from` field gives the original directory, and `members` is the file count.

## Run names

`t<tier>-ep<episode>-w<parallel worlds>-cap<buffer>-<recipe>-[s<seed>]-<host>`, for example `t1-ep0-w2048-cap4m-r3o-s43-mf4090`.

- Names that start with `smoke-` are smoke runs before a chain starts (4,096,000 steps). A full run trains 349,999,104 steps. Its last checkpoint is `policy_349999104.safetensors`.
- `cap4m`: 4 M replay buffer (mf4090 with RTX 4090 24 GB). `cap10m`: 10 M (ml40 is an L40S with 48 GB. The runs on l40n also use 10 M).
- The default seed is 42. `s43` and `s44` are other seeds.

| Tag | Change relative to the official recipe |
|---|---|
| No tag | Official recipe |
| `r3a` | Object position reward width 0.2 m → 0.05 m. Object orientation termination threshold 1.5 rad → 0.6 rad |
| `r3o` | `r3a` plus a 13th reward term w·exp(−θ²/σ²). θ is the geodesic angle between the simulated and the reference object orientation, w = 1.0, σ = 0.4 rad |
| `r3o-nv` | Official recipe plus the same 13th term |
| `c10` | The hand contact windows in the reference start 10 frames earlier, and gaps of 15 frames or fewer are filled (Tier 1 only) |
| `fc20` | `objective.force_closure_weight` 5 → 20 |
| `pt08` | `termination.palm_position_threshold` 0.15 m → 0.08 m |
| `p185` | Official recipe, trained again with the box geometry fix |

An environment variable sets the 13th reward term during training. The configuration stored in the checkpoint still has the 12 official reward names, so the official evaluator can read it directly.

## Reproduction

- Training and evaluation both run in the `flash_chord:latest` image (FlashCHORD v0.4.0 plus the upstream box geometry fix #185. Evaluation and export use the fixed evaluator package 8c778f9). The submission container is built on this image. Thus, for a reproduction environment, use the container in the submission repository. For the image Id and build history, see `mf4090/meta/docker_*.txt` and `ml40/meta/docker_*.txt`.
- To read a policy: `policy_<step>.safetensors` is a standard safetensors file. `safetensors.torch.load_file` reads all tensors. The network structure and the hyperparameters are in `hydra/` of the same run.
- `ml40/v2d/container/c1104/weights_<host>.tsv` gives the run and the checkpoint that each episode of the three Final Submissions uses. Its columns are tier, episode, sha256, path in the container and path on the host.

## INDEX

Each host has one directory `INDEX/<host>/` (host information is in `<host>/meta/`: driver, disks, images):

| File | Contents |
|---|---|
| `manifest.jsonl` | Each file of this host in this repository: path in the repository, size, sha256 |
| `symlinks.tsv` | The symbolic links on the host (path, target). HF does not store symbolic links. After the download, restore them from this table |
| `excluded.tsv` | Files that were not uploaded: path, size, sha256, reason. The sha256 is computed only for the official data, the copies of the official toolkit, third-party model weights and the submission container tar.zst. It is empty for caches and staging files |
| `held.tsv` | Files that the credential scan stopped and that were not uploaded: path, size, name of the matched rule |

To restore the symbolic links (run this in the download directory):

```bash
while IFS=$'\t' read -r path target; do mkdir -p "$(dirname "$path")"; ln -sfn "$target" "$path"; done < INDEX/mf4090/symlinks.tsv
```

## Known gaps

- l40n (the extra training host in the late phase) went to a teammate from 11/03 and has been unreachable since the freeze. Its files are not included. The evaluation readings of its runs are in `results/l40n/` of the private repository. The environment record is in `mf4090/repo/runs/hf_src/l40n_env2_1102/`.
- The docker image for training is not uploaded separately. The submission container is built on `flash_chord:latest`, and all image layers are in the tar.zst of the submission repository.
- The files that the credential scan stopped are not uploaded. `INDEX/<host>/held.tsv` lists the paths and the matched rules.
- The code, the experiment records (`agent-doc/`) and the analysis tables (`results/`) are not in this repository.
