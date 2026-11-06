# Per-Episode Policy Selection with an Object-Orientation Reward for Video-to-Robot Grounding

Code and documents of team Maniphys Lab for Track 2 (Robotic Grounding) of the NVIDIA [Video to Data (V2D) Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).

**[Project page](https://rongxuan-zhou.github.io/v2d-track2-robotic-grounding/)** ·
[Technical report (PDF)](https://huggingface.co/datasets/Rongxuan-Zhou/maniphys-lab-v2d-track2-submission/resolve/main/v2d-track2-technical-report.pdf) ·
[Submission container](https://huggingface.co/datasets/Rongxuan-Zhou/maniphys-lab-v2d-track2-submission) ·
[Data archive](https://huggingface.co/datasets/Rongxuan-Zhou/v2d-track2-research-archive)

Rongxuan Zhou<sup>1</sup>, Yuquan Chen<sup>2</sup>, Lixin Ruan<sup>2</sup>, Qingyuan Shan<sup>2</sup>, Yudong Tao<sup>2</sup>
<sup>1</sup>Northeastern University, <sup>2</sup>Maniphys Lab

## Summary

Track 2 asks for one simulated G1 + Dex3 humanoid policy per human-object interaction episode, from clean multi-view trajectories (Tier 1), corrupted trajectories (Tier 2) or a single exocentric video (Tier 3). We train one FlashCHORD policy per episode and tier, add a reward term on the object orientation, and keep per episode the policy that tracks its own tier's reference best in 4096 simulated worlds, under rules fixed in advance.

Public leaderboard scores of the three counted files (scores of different tiers are not comparable):

| Tier | ADD-AUC | SP-SR | MP-SR | MPPE (cm) |
|---|---|---|---|---|
| 1 | 0.63535 | 0.86666 | 0.70000 | 8.46753 |
| 2 | 0.50890 | 0.73333 | 0.53333 | 15.32195 |
| 3 | 0.28173 | 0.46666 | 0.10000 | 36.60061 |

## What this repository holds

| Directory | Contents |
|---|---|
| [`stages/`](stages/) | The four pipeline stages: `recovery` (Tier 2 denoising, Tier 3 run and check scripts), `retarget`, `policy` (training chains and queues), `evaluate` (export, per-episode selection, packing) |
| [`libs/v2d_infra/`](libs/v2d_infra/) | Shared layer: paths, the tier-isolation guard, SOMA forward kinematics, ADD-AUC |
| [`tools/`](tools/) | Archive and transfer tools for Hugging Face, and remote status scripts |
| [`scripts/dev/`](scripts/dev/) | Wrappers to run commands on the training hosts and to record outputs |
| [`deploy/submission_container/`](deploy/submission_container/) | The submission container: Dockerfiles, manifests and `run.sh` |
| [`third_party/`](third_party/) | Patches and sha256 lists of the FlashCHORD versions we used |
| [`docs/`](docs/) | [Engineering handbook](docs/engineering/v2d-track2-engineering-handbook.en.md), [technical report](docs/report/v2d-track2-technical-report.md), stage contracts, Hugging Face READMEs |
| [`project-page/`](project-page/) | Source of the project page |

Each directory README has a file index. Many code comments are in Chinese.

## Relation to the submitted code

This repository is a snapshot of our research repository, taken after the leaderboard froze on 2026-11-04. It is not the submission itself, and it does not change the submission. The exact code of each counted file is in the [submission container](https://huggingface.co/datasets/Rongxuan-Zhou/maniphys-lab-v2d-track2-submission): one checkout per tier at the commit recorded in that tier's manifest.

The snapshot differs from the research repository in these points:

- It has no git history.
- Host addresses are removed.
- Internal planning notes, command-output records and the team progress page are not included. Some comments and documents still refer to these paths (`agent-doc/`, `results/`, `docs/site/`).
- The Tier 3 reconstruction code (CARI4D pipeline) is not included.

Data, checkpoints and logs are in the [data archive](https://huggingface.co/datasets/Rongxuan-Zhou/v2d-track2-research-archive). The human videos come from the challenge dataset [nvidia/video_to_data_challenge](https://huggingface.co/datasets/nvidia/video_to_data_challenge) (CC BY 4.0).

## Citation

```bibtex
@techreport{maniphys2026v2dtrack2,
  title       = {Per-Episode Policy Selection with an Object-Orientation Reward
                 for Video-to-Robot Grounding},
  author      = {Zhou, Rongxuan and Chen, Yuquan and Ruan, Lixin and Shan, Qingyuan and Tao, Yudong},
  institution = {maniphysAI lab},
  type        = {Technical Report, NVIDIA V2D Challenge Track 2},
  year        = {2026}
}
```
