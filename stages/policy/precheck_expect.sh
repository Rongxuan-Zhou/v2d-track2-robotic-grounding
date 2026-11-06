#!/usr/bin/env bash
# 开链前预跑训练链的“正向核对”（与 stages/policy/day2_train_chain.sh 第 1 步相同的断言），不开训练、不占 GPU。训练机上、仓库根执行。
# 用途：一段队列要等别的链跑完才自动开跑（例如批量的 B 段），它开头的核对要是不过，整条链就停、GPU 空着没人管；开跑前先在这里核一遍。
# 每条 run（冒烟与正式）：队列里有这一行、runs/cfgcheck/<run>.yaml 存在（fc_config_check.sh 生成）、正式步数 349999104，
# 再在容器里用 assert_cfg.py 断言 parquet 的档位目录与 sequence_id、回放（逐帧一臂不得有容差键）、scene.support、buffer、步数、世界数、种子、motion_speed。
# 档位取自 run 名的 t<档位>- 前缀（冒烟与正式必须一致），parquet 必须在 /repo/runs/motion/tier<档位>/ 下：只核 sequence_id 时，
# t2 的 run 指到 tier1 的同名序列也会过（计划 agent-doc/plan-t2-batch-2026-10-01.md）。队列行里有 V2D_FC_SRC= 也算不过（除非 V2D_ALLOW_FC_SRC=1）。
# 档位目录取自队列行的 parquet：tier1、tier2、tier3，或 Tier 3 校正版的 tier3_<名>（libs/v2d_infra/paths.py 的 tier_dirname；
# 计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md）。目录里的档位数字必须与 run 名一致、序列必须与期望表一致；带 _<名> 的目录
# run 名里要有 -<名>-（例如 t3-ep8-w2048-cap4m-r3a-v2-mf4090 对 tier3_v2），免得两版参考对错 run。断言用的就是这个目录。
# 10/16 起 Tier 1、2 也许带后缀的目录（如 tier1_c10：只改训练用的接触标记，计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md）；
# 档位数字相等这一条照旧，Tier 2 的 run 指不到 tier1_*。
# 训练链 day2_train_chain.sh 的第 1 步就是调用本脚本，两处断言只在这里维护一份。
# 用法：stages/policy/precheck_expect.sh <期望表> <冒烟队列> <正式队列>；全过退出码 0，最后一行 PRECHECK_OK（n 条 run）。
set -uo pipefail
cd "$(dirname "$0")/../.."
EXPECT=${1:?期望表}; SQ=${2:?冒烟队列}; FQ=${3:?正式队列}
bad=0
qline() { grep -v '^[[:space:]]*#' "$1" | awk -v r="$2" '{for (i = 1; i <= NF; i++) if ($i == r) {print; exit}}'; }
tier_of() { [[ $1 =~ ^(smoke-)?t([123])- ]] && echo "${BASH_REMATCH[2]}"; }
mapfile -t ROWS < <(tail -n +2 "$EXPECT" | grep -v '^[[:space:]]*$')
[ "${#ROWS[@]}" -gt 0 ] || { echo "期望表 $EXPECT 没有数据行"; exit 1; }
if [ "${V2D_ALLOW_FC_SRC:-0}" != 1 ] && awk '!/^[[:space:]]*#/ && /V2D_FC_SRC=/ {f = 1} END {exit !f}' "$SQ" "$FQ"; then
  echo "FAIL 队列行里有 V2D_FC_SRC=（会把补丁源码副本挂成 /workspace）"; bad=$((bad + 1))
fi
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r run smoke seq pb src support pos ori cap seed xev <<< "$row"
  T="$(tier_of "$run")"
  if [ -z "$T" ] || [ "$(tier_of "$smoke")" != "$T" ]; then echo "FAIL $run / ${smoke}：名字没有 t<档位>- 前缀或两者档位不一致"; bad=$((bad + 1)); continue; fi
  for r in "$smoke" "$run"; do
    q="$( [ "$r" = "$smoke" ] && echo "$SQ" || echo "$FQ")"
    line="$(qline "$q" "$r")"
    [ -n "$line" ] || { echo "FAIL ${r}：队列 $q 里没有这一行"; bad=$((bad + 1)); continue; }
    y=runs/cfgcheck/$r.yaml
    [ -f "$y" ] || { echo "FAIL ${r}：缺 ${y}（先跑 fc_config_check.sh）"; bad=$((bad + 1)); continue; }
    read -r -a qt <<< "$line"; j=0
    while [[ ${qt[$j]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do j=$((j + 1)); done
    steps=${qt[$((j + 3))]}
    pq=${qt[$((j + 1))]}
    if [[ $pq =~ ^/repo/runs/motion/(tier[123](_[a-z0-9]+)?)/whole_body/soma/sequence_id=([^/]+)/robot_name=g1/data\.parquet$ ]]; then
      dir=${BASH_REMATCH[1]}; tag=${BASH_REMATCH[2]}; qseq=${BASH_REMATCH[3]}
    else
      echo "FAIL ${r}：队列行的 parquet 不在 /repo/runs/motion/<档位目录>/ 下：$pq"; bad=$((bad + 1)); continue
    fi
    if [ "${dir:4:1}" != "$T" ] || [ "$qseq" != "$seq" ] || { [ -n "$tag" ] && [[ $run != *"-${tag#_}-"* ]]; }; then
      echo "FAIL ${r}：队列行 parquet 的目录 ${dir}、序列 $qseq 与 run 名的 t${T}、期望表的序列 $seq 不符（目录的档位数字要与 run 名一致；带后缀的目录，run 名要含 -${tag#_}-）"
      bad=$((bad + 1)); continue
    fi
    if [ "$r" = "$run" ] && [ "$steps" != 349999104 ]; then echo "FAIL ${r}：正式步数 $steps"; bad=$((bad + 1)); continue; fi
    if [ "$pb" = real ]; then play=(task.source_frame_playback=false "embodiment.config.frame_position_tolerance=$pos" "embodiment.config.frame_orientation_tolerance=$ori")
    else play=(task.source_frame_playback=true '!embodiment.config.frame_position_tolerance' '!embodiment.config.frame_orientation_tolerance'); fi
    out="$(scripts/dev/in-flash-chord "python /repo/stages/policy/assert_cfg.py /repo/$y 'task.parquet~=/repo/runs/motion/$dir/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet' ${play[*]} scene.support=$support training.replay.capacity=$cap training.total_environment_steps=$steps scene.world_count=2048 reset.seed=$seed training.seed=$seed task.motion_speed=1.0" 2>&1)"; rc=$?
    echo "$out" | tail -1
    [ "$rc" = 0 ] || { echo "FAIL ${r}：断言不过"; bad=$((bad + 1)); }
  done
done
if [ "$bad" = 0 ]; then echo "PRECHECK_OK（${#ROWS[@]} 条 run）"; else echo "PRECHECK_FAIL（$bad 处不过）"; exit 1; fi
