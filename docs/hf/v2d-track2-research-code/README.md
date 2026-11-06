---
tags:
- archive
- v2d-challenge
---

# v2d-track2-research-code (private)

The code and the modified dependencies of the Maniphys Lab entry to NVIDIA V2D Challenge Track 2, organized after the 2026-11-04 leaderboard freeze. This content is not public. The data and the results are in the public repository [`Rongxuan-Zhou/v2d-track2-research-archive`](https://huggingface.co/datasets/Rongxuan-Zhou/v2d-track2-research-archive). The two repositories have the same directory structure: `<host>/<prefix>/<relative path on the host>`.

## Contents

| Path | Contents |
|---|---|
| `mac/code/maniphys_lab_v2d_challenge.bundle` | The git bundle of the code repository (`git bundle create --all`, all branches and tags), including the `agent-doc/` experiment records and `results/` |
| `mac/code/HEAD.txt`, `git_status_short.txt` | The HEAD and the `git status --short` of the Mac working tree at packing time |
| `mac/code/worktree_vs_HEAD.patch` | The uncommitted changes of the working tree relative to HEAD (`git diff HEAD --binary`) |
| `mac/code/untracked_not_ignored.tar` | Files that are untracked and that `.gitignore` does not ignore |
| `<host>/v2d/fc_src/` | The FlashCHORD copies on the training hosts: `fc_8c778f9.tar.zst` is the fixed evaluator package, `<modified version>.vs_fc_8c778f9.patch` is the difference of each modified version relative to it, `README.txt` gives the restore steps, `*.files.sha256` checks the restore result |
| Other files under `<host>/…` | Copies of the scripts (`*.py`, `*.sh` and others) in `runs/` and other directories of each host, and the `code/` and `code_tier*/` directories. `*.code.tar` is the code part of a packed directory |
| `INDEX/<host>/` | The `manifest.jsonl` of this private copy, and the full `symlinks.tsv`, `excluded.tsv`, `held.tsv`, `member_ref.tsv` (the copy in the public repository removes the lines that contain member user names) |

Non-code files whose logs or indexes contain member user names are also here. `INDEX/<host>/member_ref.tsv` lists them.

## Restore the code repository

```bash
git clone mac/code/maniphys_lab_v2d_challenge.bundle maniphys_lab_v2d_challenge
cd maniphys_lab_v2d_challenge
git checkout "$(cat ../mac/code/HEAD.txt)"        # HEAD at packing time (track2/p1-training)
git apply --binary ../mac/code/worktree_vs_HEAD.patch
tar -xf ../mac/code/untracked_not_ignored.tar
git remote set-url origin https://github.com/Manifold-ai/maniphys_lab_v2d_challenge.git   # original repository (private)
```

The commits of the three Final Submissions: Tier 1 `b59f6a6cb3f97fc53ebdb28e2e8496534478c108`, Tier 2 `c4f248dc60f39f4c7b03c535fbace6ae6b422bc9`, Tier 3 `2175e83e0c13b9ae2391d724f3d1c80c392b73d3`.

## Restore the modified FlashCHORD versions

See `<host>/v2d/fc_src/README.txt`. At packing time, we did these steps once on the host. The file sha256 list of each modified version was the same as the original directory.
