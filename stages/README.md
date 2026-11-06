# stages

The four stages of the pipeline. Each stage is a separate package. Each tier goes through `recovery → retarget → policy → evaluate`.

| Stage | Input | Output | Contract |
|---|---|---|---|
| [`recovery/`](recovery/) | Tier 2 corrupted trajectories / Tier 3 exo video | Recovered human and object trajectories | [`recovery.md`](../docs/contracts/recovery.md) |
| [`retarget/`](retarget/) | Human and object trajectories | G1 + Dex3 reference motion | [`retarget.md`](../docs/contracts/retarget.md) |
| [`policy/`](policy/) | Reference motion | Policy checkpoints and rollouts | [`policy.md`](../docs/contracts/policy.md) |
| [`evaluate/`](evaluate/) | Checkpoints and rollouts | Submission files, locally recomputed scores | [`evaluate.md`](../docs/contracts/evaluate.md) |

The stages exchange only files and do not import each other. The Isaac side and the non-Isaac side cannot share one Python environment.
The file handoff also lets you run each stage again on its own and replace the implementation of each stage on its own.
