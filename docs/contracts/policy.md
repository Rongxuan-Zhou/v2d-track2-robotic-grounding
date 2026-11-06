# 契约：策略训练与回放（policy）

边界：`stages/policy`：在 Isaac Lab 里训练跟随参考运动的策略，产出 checkpoint 与回放。

**契约 ID**：`POLICY-<层><序号>`，层用 U / C / D / I。目前只有 Internal 层的非有限防护条目（训练用的补丁副本
`fc_src/v040_p185_nfg`，ADR-00003 变更记录 2026-10-02；设计与闸见 `agent-doc/plan-ep7-eval-and-guards-2026-10-02.md` 的 B 部分），其余待填。

## User-facing

| ID | 主题 | 交付 | 验证 | 审阅 | 状态 |
|---|---|---|---|---|---|

## Caller

| ID | 主题 | 交付 | 验证 | 审阅 | 状态 |
|---|---|---|---|---|---|

## Data

| ID | 主题 | 交付 | 验证 | 审阅 | 状态 |
|---|---|---|---|---|---|

## Internal

| ID | 主题 | 交付 | 验证 | 审阅 | 状态 |
|---|---|---|---|---|---|
| POLICY-I001 | 非有限防护 | 世界内任一刚体位姿含 NaN / inf、或平移绝对值超过 1e3 m 时，ReconBody 终止判定当步把这个世界判终止，原因记骨盆、骨盆位置误差记 −1、其余误差记 0，不受冻结期屏蔽 | `stages/policy/nfg_test_kernels.py`<br>- test_k1<br>`stages/policy/nfg_env_probe.py`（GPU）<br>- T2 | 待审 | 已实现 |
| POLICY-I002 | 非有限防护 | 世界全部有限时，打了防护的终止判定 12 个输出与上游 v0.4.0 逐位相同 | `stages/policy/nfg_test_kernels.py`<br>- test_k1<br>`stages/policy/nfg_env_probe.py`（GPU）<br>- K1 | 待审 | 已实现 |
| POLICY-I003 | 非有限防护 | 重置时，被重置世界的 `qacc_warmstart`（与 `act`，若有）只要一行里有非有限值就整行清零，有限的行与没被重置的世界逐位不动；被重置的坏世界之后保持有限 | `stages/policy/nfg_test_kernels.py`<br>- test_k2<br>`stages/policy/nfg_env_probe.py`（GPU）<br>- T2<br>- T2n | 待审 | 已实现 |
| POLICY-I004 | 非有限防护 | 交给 `consume_transition` 的 transition 全部有限、观测存得进回放的存储精度；坏的世界步是“终止 1、其余 0”的失败样本，重置后的观测与上下文只在它们自己非有限时才换 | `stages/policy/nfg_test_learner.py`<br>- test_bad<br>- test_env_flag<br>- test_consume | 待审 | 已实现 |
| POLICY-I005 | 非有限防护 | 没有坏的世界步时，清洗函数返回的每个数组与输入逐位相同 | `stages/policy/nfg_test_learner.py`<br>- test_clean | 待审 | 已实现 |
| POLICY-I006 | 非有限防护 | 有坏世界步的日志窗口打一行 `NONFINITE_GUARD_EVENT`；只有学习器判出的世界步累计超过 100、或同一世界连续 3 步坏时，打一行 `NONFINITE_GUARD_STOP` 后停训 | `stages/policy/nfg_test_learner.py`<br>- test_report<br>`stages/policy/nfg_train_inject.py`（GPU）<br>- T6 | 待审 | 已实现 |
| POLICY-I007 | r3o 物体朝向项 | 训练副本（`v040_p185_nfg_oo`）的 ReconBody 目标函数第 13 项 = exp(−θ²/σ²)，θ 与物体朝向终止用同一个量（同一刚体、`command.body_target_quat_w`、同一帧）；权重与 σ 只来自 `V2D_OBJ_ORI_WEIGHT`、`V2D_OBJ_ORI_STD`（默认 0、0.4），非法值报错；类里 `wp.launch` 的实参与 kernel 形参逐位对应 | `stages/policy/oo_test_kernels.py`<br>- O1<br>- O2<br>- O6<br>- O7 | 待审 | 已实现 |
| POLICY-I008 | r3o 物体朝向项 | 第 13 项权重 0 时，目标函数 kernel 的前 12 项与分数与上游 v0.4.0 逐位相同（CPU 上测；GPU 上重编译的 kernel、学习器按 13 项重算奖励的归约顺序不在此列） | `stages/policy/oo_test_kernels.py`<br>- O1 | 待审 | 已实现 |
| POLICY-I009 | r3o 物体朝向项 | checkpoint 存的解析后配置保持上游的样子：课程各段 `objective_weights` 只有上游 12 个名字，没有 `object_orientation_weight`、`object_orientation_std` 键，没有 `V2D_OBJ_ORI`；上游 8c778f9 评测器能读、能评 4096 世界、能导出 1 局；训练时课程没写的第 13 项由 `set_weights` 与 `weights_for` 补上环境变量的权重 | `stages/policy/oo_test_kernels.py`<br>- O3<br>- O4<br>- O5<br>`stages/policy/r3o_g1.sh`（GPU） | 待审 | 已实现 |
| POLICY-I010 | r3o 物体朝向项 | run 名带 `-r3o-`、设了 `V2D_OBJ_ORI_WEIGHT`、训练副本打了物体朝向补丁，三者同时具备才训，缺一样 `fc_run.sh` 就拒绝；`arm_var_chain.sh` 对 r3o 段与其他段的队列行双向核对（规则在 `stages/policy/r3o_lines.awk`）：r3o 段每行权重 1.0、σ 每行自带且两个变量各正好一次，run 名带 `-sgDD-` 时 σ 必须等于 D.D，不带时 σ 是 0.4；普通副本的段一行都不许带这两个变量或 `-r3o-`（10/28 第 2 版：σ 从固定 0.4 改成与 run 名标签一致） | `stages/policy/test_fc_run_r3o_guard.py`<br>- test_guard<br>`stages/policy/test_r3o_lines.sh` | 待审 | 已实现 |

## 非目标

待填。
