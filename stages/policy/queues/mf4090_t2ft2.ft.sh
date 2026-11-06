# ft_chain.sh 的段配置：t2ft2（计划 agent-doc/plan-t2ft2-2026-10-26.md）。不跑冒烟；先评 K 的 6 份中间 checkpoint 与起点 run 的最后 4 份（C），再训 A1、A2、B。
FQ=stages/policy/queues/mf4090_t2ft2.txt
K=t2-ep25-w2048-cap4m-r3a-ot3-ftk-mf4090
S0=t2-ep25-w2048-cap4m-r3a-ot3-mf4090
# K 第 350013440 步那份没更新过、与起点相同，不评（复核 1）；起点 run 最后 4 份作同一谱系的波动参照（复核 2）
EVAL_PRE=($K:360013824 $K:370014208 $K:380014592 $K:390014976 $K:400015360 $K:410015744 $S0:310011904 $S0:320012288 $S0:330012672 $S0:340013056)
START_SHA=(
  [t2-ep25-w2048-cap4m-r3a-ot3-mf4090/state_349999104.safetensors]=96774d00cb802665ba5b3c8e8a82392a17aed65df0039bcfd986093ce84effe7
  [t2-ep25-w2048-cap4m-r3a-c5-mf4090/state_349999104.safetensors]=134456b272174b0e08122ea71b5e0fcadc2a07fa7f82a2a8cd9c969a7da79b8f
  [$K/state_419999744.safetensors]=1e68d788f91b5de78aaf67beac80c1e7b6f26e1d5fe92d8c637b49d568969675
)
start_of() { case "$1" in *-c5-ftk-*) echo t2-ep25-w2048-cap4m-r3a-c5-mf4090/state_349999104.safetensors ;;
                         *-ftk2-*) echo "$K/state_419999744.safetensors" ;;
                         *) echo t2-ep25-w2048-cap4m-r3a-ot3-mf4090/state_349999104.safetensors ;; esac; }
steps_of() { case "$1" in *-ftk2-*) echo 490000384 ;; *) echo 419999744 ;; esac; }
p8_of() { echo 0.5; }
ot_of() { case "$1" in *-c5-*) echo 0.1 ;; *) echo 0.3 ;; esac; }
extra_of() { case "$1" in *-ftks43-*) echo "reset.seed=43 env.reset.seed=43 training.seed=43" ;; *) echo "reset.seed=42 env.reset.seed=42 training.seed=42" ;; esac; }
