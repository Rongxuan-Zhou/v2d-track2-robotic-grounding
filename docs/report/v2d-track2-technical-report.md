# Maniphys Lab: Technical Report for V2D Challenge Track 2 (Robotic Grounding)

Team: Maniphys Lab. Date: 4 November 2026. Track 2, Tiers 1, 2 and 3.

## 1. Overview

The counted submissions score ADD-AUC 0.63535 on Tier 1, 0.50890 on Tier 2 and 0.28173 on Tier 3 on the public leaderboard (Section 6). On Tier 3, the board ADD-AUC rose from 0.10649 (18 October) to 0.28173 (29 October) through later reconstruction versions, frame-0 alignment and per-episode replacements together.

Track 2 asks for one simulated robot policy per human-object interaction episode. The robot is the fixed G1 humanoid with Dex3 hands, and the simulator and trainer are the official FlashCHORD release. The benchmark has 30 episodes. Each tier gives a different input for the same episodes:

- Tier 1: clean multi-view human and object trajectories.
- Tier 2: the same trajectories with synthetic corruption.
- Tier 3: only the exocentric monocular video (`observation.images.exo_camera`); the team reconstructs the human and object motion.

All three tiers are scored against the clean multi-view reference with four metrics (ADD-AUC, SP-SR, MP-SR, MPPE). The three tiers are scored separately, and each tier has one submission file that counts on all four of its boards.

Our entry trains one FlashCHORD policy per episode and per tier, evaluates it with the official evaluator, and picks per episode the policy that tracks its own tier's reference best under fixed rules. The data flow is the same for every tier: recovery, retargeting, policy training, evaluation and packing. The stages exchange only files. Tier 1 skips the recovery stage.

This report describes the method that produced the three counted files:

| Tier | Counted file | Code commit |
|---|---|---|
| 1 | `track2_tier1_t1x1102.csv` | `b59f6a6cb3f97fc53ebdb28e2e8496534478c108` |
| 2 | `track2_tier2_select_b1014.csv` | `c4f248dc60f39f4c7b03c535fbace6ae6b422bc9` |
| 3 | `track2_tier3_t3m1029a.csv` | `2175e83e0c13b9ae2391d724f3d1c80c392b73d3` |

Later uploads to the AUC boards only (Tier 2 `t2a1102`, Tier 3 `t3m1104c`) used the orientation reward term of Section 2.2 on more episodes; they are not the counted files, because the final submission must be the same file on all four boards of a tier.

## 2. Common pipeline

### 2.1 Retargeting

Human motion goes to the G1 + Dex3 with the official `soma_to_g1` retargeter (v0.4.0). A thin wrapper in our repository calls it; the official source is not modified. Retargeting runs frame by frame without interpolation. The object trajectory is used only for the contact labels. The output is one reference per episode (robot joint targets, palm poses, object pose and contact labels).

### 2.2 Per-episode policy training

Each policy is trained by FlashCHORD with the official FlashSAC configuration, and the reference is played back frame by frame, as in the official scoring. The official recipe trains for 349,999,104 environment steps with 2048 parallel worlds and a nine-stage curriculum in which the object assist force decays to zero from stage 7. Its reward has twelve terms. The only object term is on position (width σ = 0.2 m). The object terminations are 0.10 m in position and 1.5 rad in orientation. Training uses FlashCHORD v0.4.0 with the upstream box-geometry fix #185; without it, the v0.4.0 importer builds each box object as a solid cube with 1 m edges. Evaluation and export use the pinned evaluator runtime (8c778f9), which includes this fix.

The counted files use these recipe variants. The tag in brackets is the tag in the run name.

| Variant | Change from the official recipe |
|---|---|
| Official (no tag) | None. |
| Variant A (`r3a`) | Object position reward width 0.2 m to 0.05 m; object orientation termination 1.5 rad to 0.6 rad. |
| Orientation term (`r3o`) | Variant A plus a 13th reward term w·exp(−θ²/σ²), where θ is the geodesic angle between the simulated and the reference object orientation; w = 1.0, σ = 0.4 rad, in all curriculum stages. |
| Orientation term on the official recipe (`r3o-nv`) | Official recipe plus the same 13th term. |
| Earlier contact labels (`c10`, Tier 1 only) | Hand contact windows in the reference start 10 frames earlier, and gaps of up to 15 frames are filled. |
| Grasp closure (`fc20`, Tier 1 ep9) | `objective.force_closure_weight` 5 to 20. |
| Palm reset (`pt08`, Tier 1 ep25) | `termination.palm_position_threshold` 0.15 m to 0.08 m. |
| Box geometry (`p185`, Tier 2 ep28) | Official recipe, retrained on the corrected box geometry. |

The orientation term is set through two environment variables at training time. The configuration that the checkpoint stores keeps the twelve official reward names, so the official evaluator reads every checkpoint unchanged.

The replay buffer holds 4 M transitions on the RTX 4090 (24 GB) host (`cap4m`) and 10 M on the L40S (48 GB) host (`cap10m`). The default seed is 42; `s43` and `s44` mark other seeds. Every training chain first runs a 4,096,000-step smoke run.

### 2.3 Evaluation

The submitted rollout of each episode is the official evaluator export: one world, deterministic actions, from one checkpoint. To choose between candidates we also evaluate each checkpoint in 4096 worlds against the reference of its own tier (Tier 1: the clean reference; Tier 2: our denoised reference; Tier 3: our reconstruction). Tier 2 and Tier 3 selections never use Tier 1 trajectories. Every evaluation world starts from the same state, so worlds differ only through GPU non-determinism; the 4096-world rates estimate how likely the single exported world is to succeed.

### 2.4 Per-episode selection

The policies of an episode are compared one candidate against the listed run of that episode, with rules written down before the results:

- Tier 1 (the rules of the November lists `t1x1101` and `t1x1102`): a candidate replaces the listed run if (a) MP-SR rises by at least 0.10 and MPPE does not rise by more than 1 cm, or (b) MP-SR changes by less than 0.10, SP-SR rises by at least 0.10 and MPPE is as in (a), or (c) ADD-AUC rises by at least 0.03 and MPPE falls by at least 0.5 cm, with neither success rate falling by more than 0.02. All values are 4096-world values against the clean Tier 1 reference. The single exported world is also checked before a list is packed.
- Tier 2: per episode the run with the highest 4096-world MP-SR against the Tier 2 reference, then the highest ADD-AUC, then the listed run. Section 4 gives the change for lost episodes in `b1014`.
- Tier 3 (the rule of the replacements in `t3m1029a`): a candidate replaces the listed run if it passes the tracking gate (4096-world SP-SR at least 0.5 and MPPE at most 10 cm against its own reconstruction), has a lower MPPE than the listed run and an SP-SR that is not lower.

### 2.5 Packing

The selected exports are packed with the official packer (`tools/pack_submission.py`) and uploaded with the official uploader (`submit/v2d_submit.py`) from the kit copy `update_0f2f9f4b`. Each counted file has 22,989 rows. Tier 2 and Tier 3 are packed with `--frame-references`, which give our policy input in the released Track 2 world; the packer checks that each episode's frame reference and evaluator reference differ by one rigid transform. The official packer reads the clean reference to validate rows and to print a local score. For Tier 2 and Tier 3 our wrapper drops that local score line before it is shown or stored, and no selection uses it.

## 3. Tier 1 (`t1x1102`)

The object-orientation reward term gave the largest single-episode gains on Tier 1. In 4096-world evaluation against the clean reference, it raised the ADD-AUC of ep24 from 0.580 to 0.948 and of ep7 from 0.622 to 0.915, and on ep17 `r3o-nv` raised SP-SR from 0 to 0.932. Ten of the 30 episodes of `t1x1102` use the term.

Tier 1 trains directly on the retargeted clean trajectories. The list was built in steps over earlier submissions. `t1x1101` is `t1x1018` with seven episodes replaced under the rules of Section 2.4. `t1x1102` is `t1x1101` with five more replacements: ep8 and ep24 (4 M buffer, `r3o`), ep6 (4 M, `r3o`, seed 43), ep26 (10 M, `r3o-nv`) and ep9 (10 M, `fc20`). The other 25 episodes are byte for byte the exports in `t1x1101`.

By recipe, the 30 episodes of `t1x1102` are: 8 `r3o`, 2 `r3o-nv`, 15 variant A (4 of them with `c10`), 3 official, 1 `fc20` and 1 `pt08`. Fifteen use the 4 M buffer and fifteen the 10 M buffer. All are 349,999,104-step checkpoints.

Two single-change fixes target lost episodes. The `fc20` change lowered the 4096-world MPPE of ep9 from 40.5 cm to 17.0 cm, and `pt08` lowered that of ep25 from 114.1 cm to 68.7 cm.

The leaderboard scores of `t1x1102` equal the local recomputation of the packer.

## 4. Tier 2 (`b1014`)

`b1014` lowered the Tier 2 board MPPE from 16.54028 cm to 15.32195 cm over `b1013` by changing the policy of four lost episodes, with the same SP-SR and MP-SR. Its input is our own denoising of the Tier 2 trajectories, with no Tier 1 data.

**Tier isolation.** The Tier 2 pipeline never reads Tier 1 data. The denoiser and every Tier 2 analysis script call a guard (`tier_guard.forbid_tier1()`) at start-up; after it, any read under the Tier 1 directory raises an error. The 4096-world selection scores are against our own Tier 2 reference.

**Denoiser (v5.4).** The denoiser works on the object trajectory of the Tier 2 input. Frames where the object is not visible first carry the last valid pose. The denoiser then holds static segments on a support surface (three degrees of freedom), bridges motion segments with Brownian-bridge interpolation, corrects flips and height, removes penetration of the ground by the smallest rotation about the object origin, and smooths in time. All thresholds and the rest criterion of v5.4 are derived from Tier 2 data only (`cleanroom_params.json`). The human motion of the Tier 2 input goes to retargeting as it is; its root drift is not corrected.

**List.** `b1013` applies the Tier 2 rule of Section 2.4 to all trained runs. In some lost episodes every candidate has a 4096-world MP-SR of 0, so the ADD-AUC tie-break picks among failures. `b1014` changes only these episodes: among candidates that all fail, it takes the run with the lower MPPE of the exported world against our Tier 2 reference, with neither success rate worse. This replaced four episodes (ep1, ep4, ep9, ep28). On the board, MPPE fell from 16.54028 cm (`b1013`) to 15.32195 cm, SP-SR and MP-SR stayed the same, and ADD-AUC changed from 0.51379 to 0.50890.

By recipe, `b1014` has 24 variant A runs (10 M buffer, one with seed 43), 5 official-recipe runs (4 M buffer: ep9, ep11, ep17, ep23, ep25) and 1 `p185` run (ep28, 4 M). It uses no orientation term. All are 349,999,104-step checkpoints.

## 5. Tier 3 (`t3m1029a`)

Frame-0 alignment and seven per-episode replacements on the latest reconstruction moved the Tier 3 board SP-SR from 0.26666 to 0.46666 and MPPE from 54.98984 cm to 36.60061 cm (from `gb1018e22` to `t3m1029a`).

**Reconstruction.** Tier 3 starts from the exo video only. The pipeline segments the person and the object with Grounding DINO and SAM 2, and reconstructs them with CARI4D (MoGe-2 depth, FoundationPose object tracking, contact refinement) and the SOMA body model. Object meshes and the ground plane are the released files, which are byte-identical across tiers. The reconstruction went through several versions; the counted file uses runs trained on five of them:

| Version | What it adds |
|---|---|
| `hm` | The first full CARI4D reconstruction (`best_hm`), before the intrinsics calibration of `gb3`. |
| `v2` | An earlier reconstruction, used for ep29 only. |
| `gb3` | Calibrated intrinsics (focal 866.3 px, principal point 701.8, 531.1 px), ground correction, depth correction (`dr2s`) and a dynamics pass (`dyn7`) that locks untouched static segments and removes pose jumps. |
| `gb4` | gb3 with the object refit by a Kabsch fit to the mapped surface and the human re-solved with the SOMA pose inversion. |
| `gb5` | gb4 with object poses re-selected among FoundationPose candidates for temporal consistency on 13 episodes. ep15 and ep27 also get a fix of a held segment (`hr`). |

The world frame of our reconstruction is our own estimate: the main camera is the world camera, plus the ground correction and a scale fit.

**Frame-0 alignment.** No camera calibration is released for Tier 3. In the official FAQ (October 2026) the organizers answered: "Using the first frame of Tier 1 data to compute alignment is fine, provided the Tier 1 caption isn't used in Tier 3 otherwise." We compute one rigid transform per episode that maps our frame 0 to the Tier 1 frame-0 pose. A fixed rule picks how: if the human pose is usable and the object yaw agrees with the human yaw within 15°, the full object pose is matched when the object tilt is at most 3° and only its yaw otherwise; if they disagree, the human yaw is matched; without a usable human pose, only the position is matched. The transform is applied to the frame references given to the packer. A single script reads Tier 1 frame 0, and no later Tier 1 frame is read. The alignment needs no retraining.

**Symmetric objects (ep12, ep27).** For two episodes the reconstruction cannot separate two orientation branches of a symmetric (ep27) or nearly symmetric (ep12) object, because shape, silhouette and continuity are equal for both. For these two episodes we picked the branch that agrees with the frame-0 alignment anchor and retrained on that reference. The frame-0 information therefore enters the training reference of these two episodes, not only the alignment transform.

**List.** `t3m1022aa` is the earlier list `t3m1022a` with frame-0 alignment only (no retraining). `t3m1029a` is `t3m1022aa` with seven episodes replaced under the Tier 3 rule of Section 2.4: ep5, ep10 and ep14 (gb5, seed 43), ep23 (gb5), ep15 (gb5 with held fix), ep12 and ep27 (gb5 with branch choice). The frame references of these seven were rebuilt and aligned with the same rule. By reconstruction version, the 30 runs are 13 `hm`, 8 `gb4`, 1 `gb3`, 1 `v2` and 7 `gb5`; all use variant A, none the orientation term. Three episodes (ep9, ep11, ep28) have no policy that tracks their reference; they hold placeholder checkpoints at 4,096,000 steps so that the file covers all 30 episodes. The other 27 are 349,999,104-step checkpoints.

On the board, Tier 3 moved from ADD-AUC 0.10649 / SP-SR 0.26666 / MP-SR 0 / MPPE 54.98984 cm (`gb1018e22`, 18 October) to 0.22590 / 0.33333 / 0.10000 / 49.93762 cm (`t3m1022aa`), and with the seven replacements to the scores in Section 6. `t3m1022aa` differs from `gb1018e22` both in its runs and in the alignment, so this report does not separate the two effects.

## 6. Results and limitations

| Tier | File | ADD-AUC | SP-SR | MP-SR | MPPE (cm) |
|---|---|---|---|---|---|
| 1 | `t1x1102` | 0.63535 | 0.86666 | 0.70000 | 8.46753 |
| 2 | `b1014` | 0.50890 | 0.73333 | 0.53333 | 15.32195 |
| 3 | `t3m1029a` | 0.28173 | 0.46666 | 0.10000 | 36.60061 |

Public leaderboard scores of the counted files. Scores of different tiers are not comparable.

Limitations:

- **Lost episodes.** In some episodes the object leaves the reference soon after first contact, mostly in orientation, in all worlds. In our records of 3 November these are ep1, 2, 4, 9, 11, 25 and 28 on Tier 1, ep1, 4, 9, 11, 17, 25 and 28 on Tier 2, and at least ep1, 2, 3, 4, 13, 16, 17, 19 and 25 on Tier 3. On Tier 3 the known causes include reference objects that turn before any contact (ep3, ep4), objects that start on high surfaces and fall in simulation (ep16, ep25) and a stool without collision (ep1). The per-episode changes in Section 3 did not transfer between episodes.
- **Run-to-run variation.** The score depends on one exported world. Worlds differ only through GPU non-determinism, so a policy with a high 4096-world success rate can still fail in the export: on Tier 1, ep20 and ep21 had 4096-world MP-SR 0.918 and 0.951 and failed in the exported world. A reproduced file can therefore differ from the leaderboard file in such episodes.
- **Own-reference scores.** Tier 2 and Tier 3 selection uses scores against our own reference, which overstate board gains; most on Tier 3, where the reconstruction differs from the clean reference.
- **Simulation only.** Each policy is specific to one episode, and all results are in simulation.

## 7. Reproducibility

The container follows the kit's `CONTAINER_SUBMISSION.md`. Under `/submission` it holds:

- `code_tier1/`, `code_tier2/`, `code_tier3/`: this repository with `.git`, each checked out at its tier's commit, with no modified tracked files. The `v2d_container_v1` manifest holds one code commit, so the container ships one manifest per tier (`submission.tier1.json`, `submission.tier2.json`, `submission.tier3.json`), selected with the `MANIFEST` variable; `submission.json` is the Tier 1 manifest.
- `kit/`: the kit copy `update_0f2f9f4b` that packed and uploaded all three files; `verifier_kit/`: the kit copy that provides `tools/verify_container.py`.
- `weights/<label>/episode_XXXXXX.safetensors`: one checkpoint per episode, the exact files that were evaluated (the `checkpoint_sha256` of every row was checked against the stored checkpoints).
- The FlashCHORD runtime at the pinned evaluator commit 8c778f9 in `/workspace`, where it was mounted during evaluation, on top of the FlashCHORD image the episodes were evaluated in.
- `outputs/<label>/rollouts/`: the evaluator exports packed into each file; for Tier 2 and Tier 3 also the frame references passed with `--frame-references`.
- `submitted/`: the three uploaded CSV files and their upload receipts; `submission.json`; `README.md`.
- `run.sh reproduce` rolls out every episode's policy with the evaluator and packs a file; `run.sh verify` compares the hashes and the reproduced file with the submitted one.

Provenance recorded in the three files (the same values in all three): `eval_script_sha256` 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1, `eval_code_sha256` b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1, `packer_script_sha256` 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be, `uploader_script_sha256` fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718.

| Tier | Kaggle submission IDs (AUC, SP-SR, MP-SR, MPPE) | CSV sha256 | Code commit |
|---|---|---|---|
| 1 | 57920552, 57920554, 57920555, 57920556 | ccabcc5183a3e80590ac46ef1ba8d89d4925b30d003a8d10e86bbb499084ac85 | https://github.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/b59f6a6cb3f97fc53ebdb28e2e8496534478c108 |
| 2 | 57197028, 57197030, 57197033, 57197036 | 3f89908ae1937a9e2514c1c97291631dcf44b1c63df96fda76c075c131dd33be | https://github.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/c4f248dc60f39f4c7b03c535fbace6ae6b422bc9 |
| 3 | 57783386, 57783387, 57783388, 57783392 | 37a43a779aab20bba0664dc102b1597808c3a5d14926c1d129058a4c1a1bef7d | https://github.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/2175e83e0c13b9ae2391d724f3d1c80c392b73d3 |
