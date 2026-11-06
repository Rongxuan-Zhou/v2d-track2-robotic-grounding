---
tags:
- archive
- v2d-challenge
---

# v2d-track2-archive

The checkpoint archive of NVIDIA V2D Challenge Track 2 (G1 + Dex3, FlashCHORD FlashSAC training). These files come from the training host mf4090.
The training host no longer needs them, or does not need them for now. We archive them here to free disk space on the training host. The repository is public since 2026-10-11.
The training data is the challenge data that V2D Challenge released. The challenge rules state that the publicly released challenge data is used and shared under CC BY 4.0.

## Contents

| Batch | Archive date (UTC-6) | Contents | Files | Size | Local copy on the training host |
|---|---|---|---|---|---|
| 1 | 2026-10-01 | `state_<step>.safetensors`: all states of the smoke runs, and the states of the full runs except the last one. 38 runs in total | 39 | 6.24 GB | Deleted on 10/1 |
| 2 | 2026-10-08 | `policy_<step>.safetensors`: the intermediate checkpoints of 52 runs, 26 per run (steps 10000384–260009984) | 1352 | 6.57 GB | Deleted on 10/8 |
| 3a | 2026-10-11 | Batch 3 planned 299 files (11.55 GB). At 125 committed files, the private storage quota of the account ran out. The committed files are: the last `state_<step>.safetensors` of each of 28 completed full runs, 3 states of 1 run except its last one, and 94 intermediate `policy_<step>.safetensors` checkpoints of 4 runs (steps 10000384–260009984) | 125 | 5.42 GB | Deleted on 10/11 |
| 3b | 2026-10-11 | The other 174 files of batch 3, uploaded after the repository became public: the last state of each of 34 completed full runs, and 140 intermediate policy checkpoints of 6 runs (steps 10000384–260009984) | 174 | 6.12 GB | Deleted on 10/11 |
| 4 | 2026-10-12 | 11 completed full runs (9 of Tier 3, 2 of Tier 1): for each, 26 intermediate `policy_<step>.safetensors` checkpoints (steps 10000384–260009984) and the last `state_<step>.safetensors` | 297 | 3.15 GB | Deleted on 10/13 |

- `state`: the training state for resume. A resume must pair it with the `policy` of the same step.
- `policy`: the policy network parameters, for export and evaluation. Training saves one about every 10 million steps.
- For the 52 runs of batch 2, the training host keeps the latest 9 policies of each run (steps 270010368–349999104). Export uses the last one. When the evaluator reports an error,
  the export chain goes back at most 8 policies. Thus these 9 stay local. Batches 3 and 4 select policies by the same rule, and the latest 9 are not archived.
- From batch 3, the last state of each completed full run is also archived. The training chain does not resume a completed run, and this state is necessary only for a resume.
  The policy of the same step stays on the training host, so export and evaluation are not affected.
- The number in a file name is the training step. The run directory name gives the tier (`t1`, `t2`, `t3`), the episode (`ep18`), the number of parallel worlds (`w2048`),
  the replay buffer limit (`cap4m`) and the training host (`mf4090`). For the meaning of the other suffixes, see the experiment records in the code repository.

## Directory structure

```
README.md
.gitattributes
mf4090/                                   # source training host
  runs/flash_chord/<run>/                 # same relative path as under the code repository root on the training host
    state_<step>.safetensors              # batches 1 and 3
    policy_<step>.safetensors             # batches 2 and 3
```

## Relation to the records on the training host

Before the training host deletes a local copy, it writes `ARCHIVED.json` in each run directory (append, no overwrite). Below is the verbatim text of
`runs/flash_chord/t1-ep16-w2048-cap4m-mf4090/ARCHIVED.json` (times are in Beijing time, which the training host uses):

```json
[
 {
  "time": "2026-10-01T23:52:35+0800",
  "files": [
   {
    "file": "state_340013056.safetensors",
    "step": 340013056,
    "size": 160121036,
    "sha256": "9da3ff26f01bad0590c0fc4621bcbf087453ba34bcfcef7966b33acf828a800e",
    "repo": "Rongxuan-Zhou/v2d-track2-archive",
    "path_in_repo": "mf4090/runs/flash_chord/t1-ep16-w2048-cap4m-mf4090/state_340013056.safetensors"
   }
  ]
 }
]
```

After the archive and deletion of each batch, the host appends one record. After the deletion of batch 1, the smoke runs with no states left also got `STATE_PRUNED` (a convention of the training chain).
The record time of batch 2 is 2026-10-09 01:27 Beijing time, which is 10/8 11:27 UTC-6. 19 runs are in both batch 1 and batch 2.
The record time of batch 3a is 2026-10-11 23:21 Beijing time, which is 10/11 09:21 UTC-6. Of the 28 runs of batch 3a, 4 have 1 record in `ARCHIVED.json`,
15 have 2 records and 9 have 3 records. The record time of batch 3b is 2026-10-12 11:56 Beijing time, which is 10/11 21:56 UTC-6. Of the 35 runs of batch 3b,
5 have 1 record in `ARCHIVED.json`, 20 have 2 records and 10 have 3 records.

## Download and restore

The repository is public, so a download needs no token. `HF_ENDPOINT` must be unset or equal to https://huggingface.co. The `.env` of the code repository on the training host points it to a download mirror,
and we did not check whether the mirror has the files of this repository. The training host cannot connect to huggingface.co directly and must use a proxy.

To restore to a training host, run `tools/hf_remote.sh` in the code repository root on the Mac. It reads the token and the proxy address from `.env` and sends them to the training host through stdin.
It sets `HF_ENDPOINT` to https://huggingface.co, and the command runs in the code repository root on the training host.

```bash
# Download all archived files of one run to runs/restore/ first, then move them back to the original path runs/flash_chord/<run>/
tools/hf_remote.sh 'hf download Rongxuan-Zhou/v2d-track2-archive --include "mf4090/runs/flash_chord/<run>/*" --local-dir runs/restore && mv runs/restore/mf4090/runs/flash_chord/<run>/*.safetensors runs/flash_chord/<run>/'
```

On 2026-10-08, we tested this command on mf4090 and downloaded 1 file. Its sha256 matched the list (code repository `results/ops/hf_restore_test_mf4090.txt`).

On other hosts, make sure that `HF_ENDPOINT` is unset or equal to https://huggingface.co. Then use the same `hf download` command, or use Python:

```python
from huggingface_hub import hf_hub_download

# One file. It downloads to runs/restore/mf4090/runs/flash_chord/<run>/policy_<step>.safetensors
p = hf_hub_download("Rongxuan-Zhou/v2d-track2-archive",
                    "mf4090/runs/flash_chord/<run>/policy_<step>.safetensors",
                    repo_type="model", local_dir="runs/restore")
```

## Check

After the download, compute the sha256 and compare it with the `sha256` of the file with the same name in `ARCHIVED.json`. The LFS SHA256 in the file information on the web page is the same value.

```bash
sha256sum runs/flash_chord/<run>/policy_<step>.safetensors
```

The upload had two checks. First, the size and sha256 of each file were compared with the list (batch 1 39 / 39, batch 2 1352 / 1352).
Then we downloaded a random part of the files and computed the sha256 again (batch 1: 3 files, batch 2: 1 per run, 52 in total). All matched.
The 125 files of batch 3a: all sizes and sha256 values matched (125 / 125). The spot check downloaded 33 files: the last state of each of the 28 runs,
plus 1 random file from each of the 5 runs with other files. All matched. Before the deletion of the local copies on the training host, we checked the repository again, and it was still 125 / 125.
The 174 files of batch 3b: all sizes and sha256 values matched (174 / 174). The spot check downloaded 40 files. These were the last state of each of the 34 runs, 1 random file from each of the 5 runs that also had policies,
and 1 random file from the 1 run with only policies. All matched. Before the deletion of the local copies on the training host, we checked the repository again, and it was still 174 / 174.
We also computed the sha256 of each of the 174 files on the training host again, and they matched the list.
The 297 files of batch 4: all sizes and sha256 values matched (297 / 297). The spot check downloaded 22 files: the last state and 1 random policy of each of the 11 runs. All matched.
Before the deletion of the local copies on the training host, we checked the repository two more times, and both were 297 / 297. At deletion, we also computed the sha256 of each of the 297 files on the training host again, and they matched the list.

## Related code and records

All paths below are in the code repository (private).

- `tools/hf_archive.py`: selects files (plan), uploads, checks, spot-checks, deletes, and uploads this README (readme).
- `tools/hf_archive_run.sh`: runs upload, check, spot check and deletion in sequence on the training host. If any step fails, it stops.
- `tools/hf_remote.sh`: runs a command with credentials on the training host. The credentials go only through stdin, and `HF_ENDPOINT` is set to https://huggingface.co.
- `tools/hf_archive_report.py`: summarizes the checks, the spot checks and the local state.
- Plans: `agent-doc/plan-hf-archive-2026-10-01.md` (batch 1), `agent-doc/plan-hf-archive-policies-2026-10-08.md` (batch 2),
  `agent-doc/plan-hf-archive-batch3-2026-10-10.md` (batch 3), `agent-doc/plan-hf-archive-batch4-2026-10-12.md` (batch 4).
- Records: `results/ops/hf_archive_*.txt`.
