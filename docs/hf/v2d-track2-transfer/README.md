---
tags:
- v2d-challenge
- transfer
---

# v2d-track2-transfer

The transfer repository between the training hosts of NVIDIA V2D Challenge Track 2 (G1 + Dex3, FlashCHORD FlashSAC training). The two training hosts have no direct connection,
and a relay through the local computer is too slow. Thus the sending host first uploads the intermediate data here. Then the receiving host downloads it and checks each file against the sha256 list.
The repository is private and is only for transfer. It is not a data release. The content comes from the challenge data (reconstruction, conversion and retargeting results), and it is not public.

## Contents

| Tag | Upload date (UTC-6) | Sender → receiver | Contents | Files | Size |
|---|---|---|---|---|---|
| `smoke_1013` | 2026-10-13 | mf4090 → ml40 | Smoke test: one text file and one LFS file, to test the path of upload, download through the proxy and redirect to the CDN | 2 | about 64 KB |
| `t3hm_1013a` | 2026-10-13 | mf4090 → ml40 | The tier directory `runs/tier3/root_hm` of the final Tier 3 delivery version (best_hm), the conversion result `runs/motion_src/tier3_hm`, the retargeting result `runs/motion/tier3_hm`. 29 episodes (without ep28). Status: transferred and checked (ml40 received 523/523, the sha256 lines of the two ends are the same, moved into place) | 523 | 2.40 GB |
| `t3dyn4_1014a` | 2026-10-14 | mf4090 → ml40 | The tier directory `runs/tier3/root_dyn4` of the Tier 3 dynamics-fix version best_dyn4 (ep3, 12, 15, 16), the conversion result `runs/motion_src/tier3_dyn4`, the retargeting result `runs/motion/tier3_dyn4`. ml40 trains only ep12 of these. Status: transferred and checked (ml40 received 74/74, the sha256 lines of the two ends are the same, moved into place) | 74 | about 328 MB |
| `t3dyn4s_1014a` | 2026-10-14 | mf4090 → ml40 | `runs/tier3/root_dyn4s`, `runs/motion_src/tier3_dyn4s`, `runs/motion/tier3_dyn4s` of ep12 of best_dyn4s (best_dyn4 plus temporal smoothing of the human). Status: transferred and checked (ml40 received 20/20, the sha256 lines of the two ends are the same, moved into place) | 20 | about 118 MB |
| `t3kfix_5_1016` | 2026-10-16 | ml40 → mf4090 | The parquet, metadata and deployed version of the new k-fix reference (`tier3_kfix_F`, expressed again with the calibrated intrinsics) for ep0, 18, 24, 26, 29, under `<tag>/pack/`. On the receiving host, `stages/recovery/tier3_stage_hf.sh` checks each file against the hash table, then merges it into `runs/tier3/root_kfix`. Status: transferred and checked (11/11) | 11 | 4.8 MB |
| `t3v2s_20_1016` | 2026-10-16 | ml40 → mf4090 | The v2 reference of ep20 plus the delivered box tabletop (`<tag>/pack/`), merged into `runs/tier3/root_v2s`. Status: transferred and checked (3/3) | 3 | 0.7 MB |
| `t3gb3_11_1017` | 2026-10-17 | ml40 → mf4090 | The parquet, metadata, support surfaces and deployed version of the gb3 delivery version of a teammate (calibrated intrinsics, `tier3_cari4d_gb_best_dr2s_dyn7`) for ep0, 6, 7, 8, 18, 20, 21, 22, 24, 26, 29 (`<tag>/pack/`), merged into `runs/tier3/root_gb3`. Status: transferred and checked (34/34) | 34 | 10.2 MB |
| `t3gb4_11_1019` | 2026-10-19 | ml40 → mf4090 | The parquet, metadata, support surfaces and deployed version of the gb4 delivery version of a teammate (calibrated intrinsics, human solved again with PoseInversion, `tier3_cari4d_gb4_best_dr2s_dyn7`) for ep0, 6, 7, 8, 18, 20, 21, 22, 24, 26, 29 (`<tag>/pack/`), merged into `runs/tier3/root_gb4`. Status: transferred and checked (34/34) | 34 | 11.8 MB |
| `t12replay_1020` | 2026-10-20 | mf4090 → ml40 | The reference replay checkpoints of Tier 2 ep4, ep25, ep1 (`smoke-t2-ep{4,25,1}-w2048-cap4m-r3a-sd-mf4090/policy_409600.safetensors`, zero residual), for the fall diagnosis on ml40 (plan `agent-doc/plan-t12replay-2026-10-20.md`). Status: waiting for transfer | 3 | about 15 MB |
| `t3gb3_1017a` | 2026-10-17 | mf4090 → ml40 | The tier directory `runs/tier3/root_gb3` of gb3, the conversion result `runs/motion_src/tier3_gb3`, the retargeting result `runs/motion/tier3_gb3`. 11 episodes (ep0, 6, 7, 8, 18, 20, 21, 22, 24, 26, 29). Status: transferred and checked (ml40 received 199/199, the sha256 lines of the two ends are the same, moved into place) | 199 | 850.5 MB |
| `l40n_env_1102` | 2026-11-02 | mf4090 → l40n | The training environment of the third training host (L40), packed as 3 tar files: `robotic_grounding/flash_chord` and `robotic_grounding/source` of the official toolkit 33129dd (`toolkit.tar`), the FlashCHORD copies `v040_p185_nfg_oo` and `v040_p185_nfg` (`fc_src.tar`), the reference trajectories `runs/motion/tier2` and `tier3` (`motion.tar`), plus `TARS.sha256`. Script: `deploy/l40n/stage_env_tars.sh`. Status: upload in progress | 4 | 3.68 GB |

## Directory structure

```
<tag>/
  MANIFEST.sha256          # sha256 of all data files under this tag (sha256sum format, paths relative to the code repository root on the training host)
  runs/...                 # the same relative paths as on the training host
```

The tags that `tier3_stage_hf.sh` transfers (`t3kfix_5_1016`, `t3v2s_20_1016`, `t3gb3_11_1017`, `t3gb4_11_1019`) contain `pack/`: the files that the sender unpacked from its package (`DEPLOYED`, `data/`, `meta/`, `support_surface/`). The receiver checks them against the hash table before it merges them into the tier directory.

## Download and check

The receiving host runs this in the code repository root (the copy of `hf_transfer.py` on the training host is in `runs/tools/`. The credential comes in only through the environment variable `HF_TOKEN`, not on the command line):

```bash
mkdir runs/.relay_tier3_hm.partial
python3 runs/tools/hf_transfer.py pull --repo Rongxuan-Zhou/v2d-track2-transfer --tag t3hm_1013a --dest runs/.relay_tier3_hm.partial
```

`pull` uses only the Python standard library. It first downloads `MANIFEST.sha256`, then downloads the files one by one and computes the sha256 during the download. If a hash is different from the list, it waits a few seconds and downloads again, at most 3 times. If the hash is still different, it stops.
The token goes only to huggingface.co itself. A redirect to the CDN does not carry it.
When all files match, the last line is `HF_PULL_OK <file count>`. After that, the receiver also compares the full sha256 list line by line with the list that the sender makes at that time.
It checks the episode count, the tier and the support surfaces, and only then moves the directory to its final location (`stages/recovery/tier3_relay.sh`).

## Relation to the code repository

- Upload and download: `tools/hf_transfer.py` (`push` on the sender, `pull` on the receiver).
- The full relay and check: `stages/recovery/tier3_relay.sh`. Plan and criteria: Section 8 of `agent-doc/plan-d-t3hm-train-2026-10-12.md`.
- The record of each relay is under `results/t3hm/` (`relay_ml40_hf.txt` and others). For the two tags of 10/14, see Section 3, step 5 of `agent-doc/plan-t3dyn-pilot-2026-10-14.md`. Their records are under `results/t3dyn/`.
- For the two tags of 10/16, see option 4 of `agent-doc/plan-t3-kfix-2026-10-16.md`. Records: `results/t3kfix/stage_hf_mf4090.txt`, `stage_hf_v2s_mf4090.txt`. For the gb3 tag of 10/17, see method 2 of `agent-doc/plan-t3-gb3-switch-2026-10-17.md`. Its records are under `results/t3gb3/`.

## Retention

After the receiver passes the check and the applicable training is complete, a separate decision sets whether to delete the copies here.
