# ft_chain.sh 的段配置：t2ft（计划 agent-doc/plan-t2ft-2026-10-26.md）。与 10/26 14:05 跑的那版行为相同。
SQ=stages/policy/queues/mf4090_t2ft_smoke.txt; FQ=stages/policy/queues/mf4090_t2ft.txt; SSTEP=354095104
START_SHA=(
  [t2-ep25-w2048-cap4m-r3a-ot3-mf4090/state_349999104.safetensors]=96774d00cb802665ba5b3c8e8a82392a17aed65df0039bcfd986093ce84effe7
  [t2-ep4-w2048-cap4m-r3a-ot3-mf4090/state_349999104.safetensors]=3d145509c2446b48bae3ef0da783c59fa30ca4e4a9197e9ea9e5f4b8b0d2761b
)
SMOKE_START_POLICY=t2-ep25-w2048-cap4m-r3a-ot3-mf4090/policy_349999104.safetensors
start_of() { case "$1" in *ep25*) echo t2-ep25-w2048-cap4m-r3a-ot3-mf4090/state_349999104.safetensors ;; *) echo t2-ep4-w2048-cap4m-r3a-ot3-mf4090/state_349999104.safetensors ;; esac; }
steps_of() { case "$1" in smoke-*) echo 354095104 ;; *) echo 419999744 ;; esac; }
p8_of() { case "$1" in *-ftf-*) echo 1.0 ;; *) echo 0.5 ;; esac; }
ot_of() { echo 0.3; }
