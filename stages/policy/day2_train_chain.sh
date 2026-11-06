#!/usr/bin/env bash
# Day 2 训练链（计划 agent-doc/plan-day2-2026-09-28.md 第 2 版）：按期望表逐条冒烟、再逐条正式训练；单条失败不连累其余。训练机上执行，放 tmux。
#   0. 来历：开头记下 .v2d-source-commit，每条 run 前比对，变了就停整条链（同步会改写它、也会换掉链后面才调用的脚本）；GPU 上有别的进程就停
#   1. 正向核对：调 stages/policy/precheck_expect.sh（断言只在那里维护一份）。期望表里每条 run（冒烟与正式）的展开配置
#      （runs/cfgcheck/<run>.yaml，由 fc_config_check.sh 预先生成）逐键断言（assert_cfg.py）：parquet 的档位目录与 sequence_id、回放、
#      palm 容差（逐帧一臂不得有）、scene.support、buffer、步数、世界数、种子。任何一条不符就停整条链
#   2. 冒烟：逐条 fc_run（DONE 跳过；有残留 state_* 没有 DONE 记 SKIPPED，不续训）→ smoke_check → 256 世界 diag_rollout
#      → 用评测器顶点算不动基线（经 record 落成 day2/bar_<冒烟 run>）→ 帧数核对。任何一步不过，这条的正式训练记 SKIPPED，继续下一条
#   3. 正式：冒烟通过的 run 逐条 fc_run（DONE 跳过；残留 state 没 DONE 记 SKIPPED；失败记 FAILED，继续下一条）
# Tier 2 批量起（计划 agent-doc/plan-t2-batch-2026-10-01.md）另加：
#   - 档位：从 run 名的 t<档位>- 前缀取档位，正向核对断言 parquet 在 runs/motion/tier<档位>/（Tier 3 校正版是 tier3_<名>/）下（只核对 sequence_id 时，
#     t2 的 run 指到 tier1 的同名序列也会过）；名字没有档位前缀就停整条链（在 precheck_expect.sh 里）
#   - 正式训练经 fc_run_watched.sh：带发散看守，看守止损记“发散”（退出码 9），训完但最新 policy 非有限也记“发散”（退出码 10）
#   - 磁盘：每条冒烟与正式开跑前，仓库所在分区的可用空间不到 V2D_MIN_FREE_MB（默认 2048）就停整条链
#   - 冒烟的 256 世界 rollout 里非有限世界的个数照打（只报数，不作判据）
#   - 运行时：环境或队列行里出现 V2D_FC_SRC（换挂补丁源码副本）就停，除非 V2D_ALLOW_FC_SRC=1
#   - V2D_PRUNE_SMOKE_STATE=1：冒烟全部判过、基线算完之后删掉它的 state（只留 policy，步数记进 STATE_PRUNED），省磁盘；
#     冒烟不续训，正式训练也不从冒烟的 state 起步，smoke_check.sh 重判时认 STATE_PRUNED
# 期望表（TSV，第一行是表头）：run smoke sequence_id playback(frame|real) src_frames support(true|false) pos_tol ori_tol capacity seed xeval(yes|no)
#   期望帧数：逐帧 = src_frames；真实时长 = int(round((src_frames − 1) / 30 × 50)) + 1（同上游 resample_times）。pos_tol / ori_tol 两种回放的行都填，
#   逐帧一臂训练时不用（断言它没有容差键），评测链的交叉评测按它放宽。xeval 只给评测链用。
# 让卡（计划 agent-doc/plan-t2-scaleout-2026-10-09.md 第 5 节，Fable 5.1 审查 M1）：
#   - V2D_GPU_WAIT_S（默认 0，即不等）：每条冒烟与正式开跑前见到 GPU 上有别的进程，每 V2D_POLL_S（默认 60）秒看一次，最多等这么久；
#     空了接着跑，等不到就停整条链。同 export_chain.sh 的 wait_gpu
#   - 停链标记 runs/.chain-stop-<日志名>：每条冒烟与正式开跑前见到它，剩下的都不开（正在训的那条不受影响），照常写结束行，
#     评测链与导出门接着处理已完成的 run。fc_run.sh 的 STOP 只管已经起了容器的 run，没起的会被当成残留清掉
# 测试：stages/policy/test_day2_chain_waits.py。
# 用法（仓库根）：stages/policy/day2_train_chain.sh <期望表> <冒烟队列> <正式队列> <冒烟显存上限 MiB> <日志名>
# 日志 runs/logs/<日志名>.log，最后一行 DAY2_TRAIN_DONE（完成 x、失败 y、跳过 z、发散 v）或 DAY2_TRAIN_FAILED（链本身出错）。
set -uo pipefail
cd "$(dirname "$0")/../.."
EXPECT=${1:?期望表}; SQ=${2:?冒烟队列}; FQ=${3:?正式队列}; PEAK=${4:?冒烟显存上限}; NAME=${5:?日志名}
mkdir -p runs/logs results/day2
exec >> "runs/logs/$NAME.log" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
die() { say "$*"; say DAY2_TRAIN_FAILED; exit 1; }
DAY1_BASE11=e2934912858820b8; DAY1_FREEZE=df0cf35711a68fb3   # 两台机器 Day 1 run_record.jsonl 里的镜像指纹
SRC0="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
say "start $NAME, source $SRC0, 期望表 $EXPECT"
MARK="runs/.chain-active-$NAME"   # scripts/dev/remote 见到 runs/.chain-active-* 就拒绝同步（同步会改写来历、换掉链后面才调用的脚本）
echo "$NAME $SRC0 $(date '+%F %T %Z')" > "$MARK"; trap 'rm -f "$MARK"' EXIT
case "$SRC0" in *-dirty*|*-syncing*|unknown) die "来历 $SRC0 不干净，先提交再同步" ;; esac
check_src() { [ "$(cat .v2d-source-commit 2>/dev/null || echo unknown)" = "$SRC0" ] || die "来历变了（开头 $SRC0，现在 $(cat .v2d-source-commit 2>/dev/null)），停下"; }
GPU_WAIT_S="${V2D_GPU_WAIT_S:-0}"
[[ $GPU_WAIT_S =~ ^[0-9]+$ ]] || die "V2D_GPU_WAIT_S 只能是秒数：$GPU_WAIT_S"
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -; }
gpu_check() {  # GPU 上有别的进程：最多等 GPU_WAIT_S 秒（0 就不等），空了接着跑，等不到就停整条链
  local o t0=$SECONDS
  o="$(gpu_pids)"; [ -n "$o" ] || return 0
  [ "$GPU_WAIT_S" -gt 0 ] || die "GPU 上有别的进程（$o），停下"
  say "GPU 上有别的进程（${o}），每 ${V2D_POLL_S:-60} 秒看一次，最多等 ${GPU_WAIT_S} 秒"
  while [ -n "$(gpu_pids)" ]; do
    [ $((SECONDS - t0)) -lt "$GPU_WAIT_S" ] || die "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，停下"
    sleep "${V2D_POLL_S:-60}"
  done
  say "GPU 空了，接着跑"
}
STOP_MARK="runs/.chain-stop-$NAME"
stopped() {  # 见到停链标记：记一次，返回 0（调用处 break）
  [ -e "$STOP_MARK" ] || return 1
  [ -n "${STOPPED:-}" ] || say "见到 ${STOP_MARK}：剩下的冒烟与正式训练都不开；已完成的照常交给评测链"
  STOPPED=1
}
qline() { grep -v '^[[:space:]]*#' "$1" | awk -v r="$2" '{for (i = 1; i <= NF; i++) if ($i == r) {print; exit}}'; }
run_q() {  # run_q <脚本> <队列行>：带行首的环境变量调 fc_run.sh（冒烟）或 fc_run_watched.sh（正式），返回它的退出码
  local tok envs=() i=0
  read -r -a tok <<< "$2"
  while (( i < ${#tok[@]} )) && [[ ${tok[$i]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs+=("${tok[$i]}"); i=$((i + 1)); done
  env "${envs[@]}" bash "stages/policy/$1" "${tok[@]:$i}" < /dev/null
}
tier_of() { [[ $1 =~ ^(smoke-)?t([123])- ]] && echo "${BASH_REMATCH[2]}"; }
disk_check() {  # 仓库所在分区的可用空间（MB）不到 V2D_MIN_FREE_MB 就停整条链
  local need="${V2D_MIN_FREE_MB:-2048}" avail
  avail="$(df -Pk . | awk 'NR == 2 {print int($4 / 1024)}')"
  [ "${avail:-0}" -ge "$need" ] || die "磁盘只剩 ${avail} MB（要求至少 ${need} MB），停下"
}
exp_frames() { python3 -c 'import sys; n = int(sys.argv[2]); print(n if sys.argv[1] == "frame" else int(round((n - 1) / 30.0 * 50.0)) + 1)' "$1" "$2"; }
leftover() { [ ! -f "runs/flash_chord/$1/DONE" ] && compgen -G "runs/flash_chord/$1/state_*.safetensors" > /dev/null; }

mapfile -t ROWS < <(tail -n +2 "$EXPECT" | grep -v '^[[:space:]]*$')
[ "${#ROWS[@]}" -gt 0 ] || die "期望表 $EXPECT 没有数据行"
gpu_check; disk_check
if [ "${V2D_ALLOW_FC_SRC:-0}" != 1 ]; then
  [ -z "${V2D_FC_SRC:-}" ] || die "环境里设了 V2D_FC_SRC=${V2D_FC_SRC}（会把补丁源码副本挂成 /workspace），停下"
  ! awk '!/^[[:space:]]*#/ && /V2D_FC_SRC=/ {f = 1} END {exit !f}' "$SQ" "$FQ" || die "队列行里有 V2D_FC_SRC=，停下"   # 单进程，不经管道（pipefail 下 SIGPIPE 会盖掉匹配）
fi

# 1. 正向核对（断言在 precheck_expect.sh 里维护）
stages/policy/precheck_expect.sh "$EXPECT" "$SQ" "$FQ" || die "正向核对没过（见上），停下"
say "正向核对：${#ROWS[@]} 条 run 的冒烟与正式配置都与期望表一致（含档位目录）"

# 2. 冒烟
declare -A OK
nd=0; nf=0; ns=0; nv=0
for row in "${ROWS[@]}"; do
  stopped && break
  IFS=$'\t' read -r run smoke seq pb src support pos ori cap seed xev <<< "$row"; frames="$(exp_frames "$pb" "$src")"
  check_src; gpu_check; disk_check
  S=runs/flash_chord/$smoke
  if [ -f "$S/DONE" ]; then say "冒烟 $smoke 已 DONE"
  elif leftover "$smoke"; then say "冒烟 $smoke 有残留 state 没有 DONE：SKIPPED，$run 也跳过"; ns=$((ns + 1)); continue
  else
    say "冒烟 $smoke"
    run_q fc_run.sh "$(qline "$SQ" "$smoke")" || { say "冒烟 $smoke 失败：$run SKIPPED"; ns=$((ns + 1)); continue; }
  fi
  read -r -a qt <<< "$(qline "$SQ" "$smoke")"; j=0; while [[ ${qt[$j]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do j=$((j + 1)); done
  ssteps=${qt[$((j + 3))]}   # 冒烟步数取自队列行（B-rt 的冒烟加长到跨过第一个 curriculum 阶段与 checkpoint）
  stages/policy/smoke_check.sh "$smoke" "$ssteps" "$PEAK" "$DAY1_BASE11" "$DAY1_FREEZE" || { say "冒烟判据没过：$run SKIPPED"; ns=$((ns + 1)); continue; }
  mkdir -p "runs/diag/$smoke"
  V2D_GPU=1 V2D_CONTAINER_NAME="diag-$smoke" scripts/dev/in-flash-chord \
    "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$S/policy_$ssteps.safetensors /repo/runs/diag/$smoke/rollout_w256 --worlds 256" \
    || { say "$smoke 的 256 世界 diag_rollout 失败：$run SKIPPED"; ns=$((ns + 1)); continue; }
  scripts/dev/record "day2/bar_$smoke" scripts/dev/in-flash-chord "python /repo/stages/evaluate/static_bar_from_rollout.py /repo/runs/diag/$smoke/rollout_w256.npz" \
    || { say "$smoke 的不动基线没算出来：$run SKIPPED"; ns=$((ns + 1)); continue; }
  got="$(sed -n 's/^ *summary_json //p' "results/day2/bar_$smoke.txt" | python3 -c 'import json, sys; print(json.loads(sys.stdin.read())["frames"])')"
  [ "$got" = "$frames" ] || { say "$smoke 的参考是 $got 帧，期望 $frames：$run SKIPPED"; ns=$((ns + 1)); continue; }
  nfw="$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1])).get("non_finite_worlds", [])))' "runs/diag/$smoke/rollout_w256.json" 2>/dev/null || echo '?')"
  say "冒烟 $smoke 通过：$got 帧，不动基线见 results/day2/bar_$smoke.txt；256 世界 rollout 里非有限世界 ${nfw} 个（只报数）"
  if [ "${V2D_PRUNE_SMOKE_STATE:-0}" = 1 ] && st="$(python3 stages/policy/ckpt_latest.py "$S")" && [ -n "$st" ]; then
    echo "$st" > "$S/STATE_PRUNED" && rm -f "$S"/state_*.safetensors && say "冒烟 $smoke 的 state 已删（policy 留着，STATE_PRUNED 记 $st 步）"
  fi
  OK[$run]=1
done

# 3. 正式
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r run smoke seq pb src support pos ori cap seed xev <<< "$row"; frames="$(exp_frames "$pb" "$src")"
  [ -n "${OK[$run]:-}" ] || continue
  stopped && break
  check_src; gpu_check; disk_check
  if [ -f "runs/flash_chord/$run/DONE" ]; then say "$run 已 DONE"; nd=$((nd + 1)); continue; fi
  if leftover "$run"; then say "$run 有残留 state 没有 DONE：SKIPPED（续训不恢复 replay，要从 0 重来就先移走该目录）"; ns=$((ns + 1)); continue; fi
  say "正式 $run"
  run_q fc_run_watched.sh "$(qline "$FQ" "$run")"; rc=$?
  if [ "$rc" = 0 ] && [ -f "runs/flash_chord/$run/DONE" ]; then
    segs="$(grep -c '"event": "start"' "runs/flash_chord/$run/run_record.jsonl")"
    say "$run 完成（$segs 段）"; nd=$((nd + 1))
  elif [ "$rc" = 9 ]; then
    say "$run 发散：看守止损（runs/logs/$run.nanwatch.log），继续下一条"; nv=$((nv + 1))
  elif [ "$rc" = 10 ]; then
    say "$run 发散：训完但最新 policy 含非有限参数（已写 NONFINITE，评测链不评），继续下一条"; nv=$((nv + 1))
  elif [ "$rc" = 4 ]; then
    say "$run 显存不足（fc_run 退出码 4）：剩下的 run 同样的配置也放不下，全部跳过"; nf=$((nf + 1)); break
  else
    say "$run 失败（退出码 ${rc}），继续下一条"; nf=$((nf + 1))
  fi
done
[ -z "${STOPPED:-}" ] || say "停链：见到 ${STOP_MARK} 之后的 run 都没开（续跑另起一段，见计划）"
say "DAY2_TRAIN_DONE（完成 ${nd}、失败 ${nf}、跳过 ${ns}、发散 ${nv}）"
