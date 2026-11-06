# Contracts

Contracts define the function. The implementation does not. One contract file covers one contract surface: `docs/contracts/<surface>.md`. The file header states
the boundary. If a project part does not exist yet, we write its contract first.

## What a contract looks like

Each contract file has four tables, one for each of the four layers. Each row is one contract entry:

| ID | Topic | Delivers | Verification | Review | Status |
|---|---|---|---|---|---|
| DATA-I001 | Tier isolation | (Example) The read entry point of Tier 2 / 3 refuses to open a Tier 1 directory | `libs/v2d_infra/tests/test_tier_isolation.py`<br>- test_a | Pending review | Draft |

- **ID**: `<surface abbreviation>-<layer><sequence number>`. The layer is `U` (User-facing) / `C` (Caller) / `D` (Data) /
  `I` (Internal). After an ID is assigned, it does not change and is not used again. An entry that no longer holds gets the mark `Deprecated` and stays in the table.
- **Topic**: an optional group label. Entries with the same topic are next to each other.
- **Delivers**: what becomes true after release, in one sentence that you can decide as true or false. "Works well" is not a delivery. "The locally recomputed Tier 1
  MPPE differs from the output of the official script by 1e-6 or less" is.
- **Verification**: the file path plus the test names, one per line. Use `<br>` for line breaks in the cell. Write a manual verification as
  `Manual: <steps> | Verification scope: <path>`.
- **Review**: `Pending review` / `Confirmed`. **Only a person can move an entry to Confirmed.** When the contract text or its test changes,
  the person or agent that made the change must set the entry back to Pending review.
- **Status**: `Draft` → `Implemented` → `Verified`. The other values are `Not implemented`, `Not applicable`, `Superseded` and `Deprecated`.
  When the status is `Verified`, the Verification column must have a reference that resolves.

The contract files are in Chinese and use the Chinese forms of these Review and Status values.

The Review and Status columns are independent. A passing test proves only that the code matches the intent of the person who wrote the test. It does not prove that the intent is correct.

## Rules

- Without verification, an entry is a wish, not a contract. Make it observable, write the manual verification steps, or delete it.
- Each test states the contract ID that it guards in its docstring. If a test does not point back to a contract ID, delete the test.
- One contract entry states one thing that you can decide.
- Write the non-goals explicitly at the end of the contract file.
- The contract gate (`tools/contract_review`) does not exist yet. Until it exists, people check the rules above by hand.

## The four layers

| Layer | Scope |
|---|---|
| **User-facing** | Inputs, outputs, failure modes and edge cases that a user (a person or an agent) can observe |
| **Caller** | The external program interface: commands, function signatures and semantics, error classes |
| **Data** | The schemas of written and handed-off files, coordinate frames, frame indices, invariants |
| **Internal** | Order and lifecycle assumptions between components, resource ownership |

## Current contracts

All tables are empty. We fill them after the applicable ADR is decided.

| Contract | Coverage |
|---|---|
| [`data.md`](data.md) | Dataset reading, tier addressing and isolation, trajectory schema (`libs/v2d_infra`) |
| [`recovery.md`](recovery.md) | Tier 2 / 3 trajectory recovery (`stages/recovery`) |
| [`retarget.md`](retarget.md) | Retargeting of human motion to G1 + Dex3 (`stages/retarget`) |
| [`policy.md`](policy.md) | Policy training and rollout (`stages/policy`) |
| [`evaluate.md`](evaluate.md) | Official evaluator wrapper, local recomputation, submission files (`stages/evaluate`) |
