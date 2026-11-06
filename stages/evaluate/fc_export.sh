#!/usr/bin/env bash
# 用 8c778f9 评测器导出一个 run 的一份 policy（训练机上、仓库根执行）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3、5 节。
# 命令与 kit 的 examples/run_policy_evaluation.py 逐项相同：evaluation.checkpoint、world_count（默认 1，kit 与提交的设置）、
# reset_mode=explicit、metrics_output、object_trajectories_output、hydra.run.dir、logging.mode=disabled，在运行时根目录（/workspace）里跑
# 它的 scripts/evaluate_policy.py。8c778f9 副本里这个脚本与 kit 自带的那份逐字节相同（results/submission/eval_runtime_facts.txt）。
# 用法：V2D_FC_SRC=<8c778f9 副本> stages/evaluate/fc_export.sh <run> <步数>
#   V2D_EXPORT_WORLDS=1       世界数；V2D_EXPORT_TRAJ=0 只写 metrics JSON（4096 世界的比对用：轨迹每世界每帧一行，太大）
#   V2D_EXPORT_UNGUARDED=1    跳过 NaN 守卫，只给兼容性验证里含 NaN 的反向对照用
#   V2D_ALLOW_SHARED_GPU=1    GPU 上有别的进程时也照样导出（默认拒绝，退出码 6）
#   V2D_EXPORT_SUFFIX=_<名>   输出目录与日志名后加这个后缀（下划线开头，小写字母与数字，最多 16 个）；默认空，行为不变。用途：同一份 policy
#                             再导出一次、与原来那份比 sha256（计划 agent-doc/plan-first-place-2026-10-14.md 第 2 节选项零）
#   V2D_EXPORT_DRYRUN=1       预检：守卫、run 记录、v2d_meta.json、输出路径都照常核，然后只打印将在容器里执行的命令，退出码 0；
#                             不加锁、不查 GPU、不起容器、不写任何文件（链在跑的时候也能用，计划第 7 节第 3 步）
# 规矩一（用户 10/1 采纳）：含 NaN 的 policy 不拿去导出。run 目录有 NONFINITE，或这一份 policy 有非有限参数、读不了、还没写完，
#   就拒绝，退出码 12，不起容器。
# 规矩二（同上）：提交时某条 episode 报 “evaluation must disable task failure termination”，退到同一条 run 里更早的一份有限 checkpoint。
#   本脚本只负责把这种失败认出来：评测器退出码非零且这一次追加的日志里有这句话，退出码 13；其他失败退出码 3。回退由调用方按步数往前重试。
#   V2D_EXPORT_PARQUET=/repo/runs/motion/<目录>/whole_body/soma/sequence_id=<序列>/robot_name=g1/data.parquet
#                             换参考（k-fix，计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 4）：用这份 policy 对新参考评测与导出。必须同时给
#                             V2D_EXPORT_SUFFIX；新参考的档位、序列要与 run 的相同，runs/motion_src/<目录>/<序列>/v2d_meta.json 的 episode 要与原来的
#                             相同，parquet 要在；容器里改跑 stages/evaluate/eval_playback_override.py（--playback same --parquet <新参考>，导出参数经
#                             --extra 透传），它会核评测 JSON 里的 parquet 正是新参考。默认空，行为不变。
# 运行时：V2D_FC_SRC 必须是 make_fc_src_8c778f9.sh 生成的副本（SOURCE_V2D.txt 第一行含 8c778f905c），否则退出码 2。
# 输出 runs/flash_chord/<run>/export_8c778f9/s<步数>_w<世界数>/episode_<6 位序号>.{json,parquet}；日志 runs/logs/<run>.export_8c778f9_s<步数>_w<世界数>.log。
#   episode 序号取自 runs/motion_src/<目录>/<序列>/v2d_meta.json：目录与序列取自 run_record.jsonl 第一条 start 记录的 parquet 路径
#   runs/motion/<目录>/…（目录是 tier1、tier2、tier3，或 Tier 3 校正版的 tier3_<名>，见 libs/v2d_infra/paths.py 的 tier_dirname），
#   档位取自 run 名前缀，parquet 路径里的档位必须与它一致。已有完整结果（JSON 与要求的 parquet 都在）就直接打印，不重复导出。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
die() { echo "fc_export: $*" >&2; exit "${2:-2}"; }
RUN=${1:?用法：fc_export.sh <run> <步数>}; STEP=${2:?用法：fc_export.sh <run> <步数>}
W=${V2D_EXPORT_WORLDS:-1}; TRAJ=${V2D_EXPORT_TRAJ:-1}; SUF=${V2D_EXPORT_SUFFIX:-}
[[ -z $SUF || $SUF =~ ^_[a-z0-9]{1,16}$ ]] || die "V2D_EXPORT_SUFFIX 只许空，或下划线开头、1–16 个小写字母与数字：$SUF"
# 冒烟 run（smoke-t<档位>-…）也可以导出：计划 agent-doc/plan-t2st-2026-10-19.md 用训练早期的 checkpoint（复核第 1 条）
[[ $RUN =~ ^(smoke-)?t([123])-[a-z0-9_-]{2,56}$ ]] || die "run 名非法或没有 [smoke-]t<档位>- 前缀：$RUN"
T=${BASH_REMATCH[2]}
[[ $STEP =~ ^[0-9]+$ && $W =~ ^[1-9][0-9]*$ && $TRAJ =~ ^[01]$ ]] || die "步数、世界数必须是正整数，V2D_EXPORT_TRAJ 只能是 0 或 1"
O="runs/flash_chord/$RUN"; P="$O/policy_${STEP}.safetensors"
[ -f "$P" ] || die "没有 $P"
[ -n "${V2D_FC_SRC:-}" ] && head -1 "$V2D_FC_SRC/SOURCE_V2D.txt" 2>/dev/null | grep -q 8c778f905c \
  || die "V2D_FC_SRC 必须是 make_fc_src_8c778f9.sh 生成的 8c778f9 副本（现在是 ${V2D_FC_SRC:-未设}）"

if [ "${V2D_EXPORT_UNGUARDED:-0}" != 1 ]; then
  [ -f "$O/NONFINITE" ] && die "$RUN 有 NONFINITE（训完时最新 policy 含非有限参数）：不导出" 12
  want="$(python3 -c 'import sys; print(f"{int(sys.argv[1]):,d} 步：非有限 0 /")' "$STEP")"
  nf="$(python3 stages/policy/ckpt_nonfinite.py "$RUN" --steps "$STEP" --fail-on-nonfinite 2>&1)" && grep -qF "$want" <<< "$nf" \
    || die "$RUN 第 ${STEP} 步的 policy 含非有限参数、读不了或还没写完：不导出（$(grep -F ' 步：' <<< "$nf" | head -1 | sed 's/^ *//')）" 12
fi

read -r PT DN SEQ <<< "$(python3 - "$O/run_record.jsonl" "$RUN" <<'PY'
import json, re, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("event") == "start":
        m = re.fullmatch(r"/repo/runs/motion/(tier[123](?:_([a-z0-9]+))?)/whole_body/soma/sequence_id=([^/]+)/robot_name=g1/data\.parquet",
                         r.get("parquet", ""))
        # 带后缀的目录（tier3_<名>，10/16 起也有 tier1_<名>）：run 名要含 -<名>-，同 precheck_expect.sh
        ok = m and (not m.group(2) or f"-{m.group(2)}-" in sys.argv[2])
        print(f"{m.group(1)[4]} {m.group(1)} {m.group(3)}" if ok else "? ? ?")
        break
PY
)"
[ "${PT:-?}" = "$T" ] || die "$RUN 的 run 记录里 parquet 的档位是 ${PT:-?}，与 run 名的 t$T 不一致（或读不出）"
META="runs/motion_src/$DN/$SEQ/v2d_meta.json"
EP="$(python3 - "$META" "$T" "$SEQ" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert str(m["tier"]) == sys.argv[2] and m["sequence_id"] == sys.argv[3], (m["tier"], m["sequence_id"])
print(int(m["episode"]))
PY
)" || die "读不了 ${META}，或其中的档位、序列与 run 不符"
NEWP=${V2D_EXPORT_PARQUET:-}
if [ -n "$NEWP" ]; then   # 换参考：先核再用
  [ -n "$SUF" ] || die "V2D_EXPORT_PARQUET 要和 V2D_EXPORT_SUFFIX 一起用（换参考的导出不能与原来的放在同一个目录）"
  [[ $NEWP =~ ^/repo/runs/motion/(tier[12]|tier3(_[a-z0-9]+)?)/whole_body/soma/sequence_id=([^/]+)/robot_name=g1/data\.parquet$ ]] \
    || die "V2D_EXPORT_PARQUET 不是 /repo/runs/motion/<目录>/whole_body/soma/sequence_id=<序列>/robot_name=g1/data.parquet：$NEWP"
  NDN=${BASH_REMATCH[1]}; NSEQ=${BASH_REMATCH[3]}
  [ "${NDN:4:1}" = "$T" ] && [ "$NSEQ" = "$SEQ" ] || die "新参考的档位或序列（${NDN}、${NSEQ}）与 run 的（t${T}、${SEQ}）不同"
  NEP="$(python3 - "runs/motion_src/$NDN/$NSEQ/v2d_meta.json" "$T" "$NSEQ" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert str(m["tier"]) == sys.argv[2] and m["sequence_id"] == sys.argv[3], (m["tier"], m["sequence_id"])
print(int(m["episode"]))
PY
)" || die "读不了 runs/motion_src/${NDN}/${NSEQ}/v2d_meta.json，或其中的档位、序列与 run 不符"
  [ "$NEP" = "$EP" ] || die "新参考的 episode 是 ${NEP}，原来的是 ${EP}"
  [ -f "${NEWP#/repo/}" ] || die "新参考的 parquet 不在：${NEWP#/repo/}"
fi
OUTD="$O/export_8c778f9/s${STEP}_w${W}${SUF}"
J="$OUTD/episode_$(printf %06d "$EP").json"; Q="${J%.json}.parquet"
LOG="runs/logs/$RUN.export_8c778f9_s${STEP}_w${W}${SUF}.log"
# 摘要只供人看：它出错（例如 JSON 缺了它认识的键）不改变导出的结论
summary() { python3 stages/policy/eval_summary.py "$J" || echo "fc_export: 摘要打印失败（导出本身的结论不变）"; }
ARGS=("evaluation.checkpoint=/repo/$P" "evaluation.world_count=$W" "evaluation.reset_mode=explicit" "evaluation.metrics_output=/repo/$J")
[ "$TRAJ" = 1 ] && ARGS+=("evaluation.object_trajectories_output=/repo/$Q")
ARGS+=("hydra.run.dir=/repo/$OUTD/hydra/episode_$(printf %06d "$EP")" "logging.mode=disabled")
CMD="cd /workspace && python scripts/evaluate_policy.py $(printf '%q ' "${ARGS[@]}")"
if [ -n "$NEWP" ]; then   # 换参考：经 eval_playback_override.py（恒等回放、只换 parquet），导出参数透传
  XA=(--extra evaluation.reset_mode=explicit)
  [ "$TRAJ" = 1 ] && XA+=(--extra "evaluation.object_trajectories_output=/repo/$Q")
  XA+=(--extra logging.mode=disabled)
  CMD="cd /workspace && python /repo/stages/evaluate/eval_playback_override.py $(printf '%q ' "/repo/$P" "/repo/$J" --playback same --worlds "$W" --parquet "$NEWP" "${XA[@]}")"
fi
if [ "${V2D_EXPORT_DRYRUN:-0}" = 1 ]; then
  echo "fc_export: DRYRUN $RUN policy_$STEP episode $EP world_count=$W traj=$TRAJ${NEWP:+ 新参考 $NEWP}"; echo "  $CMD"; exit 0
fi
if [ -s "$J" ] && { [ "$TRAJ" = 0 ] || [ -s "$Q" ]; }; then echo "fc_export: 已有结果 $J"; summary; exit 0; fi

exec 9>"$O/.export.lock"; flock -n 9 || die "$O 正在被另一个 fc_export 导出"
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
[ -z "$others" ] || [ "${V2D_ALLOW_SHARED_GPU:-0}" = 1 ] || die "GPU 上有别的进程（PID ${others//$'\n'/,}），拒绝导出" 6
mkdir -p "$OUTD" runs/logs
rm -f "$J" "$Q"
n0="$(wc -l < "$LOG" 2>/dev/null || echo 0)"   # 日志按 run、步数、世界数累积：报错分类只看这一次追加的部分
n0="$(tr -d " " <<< "$n0")"
echo "=== $(date '+%F %T %Z') fc_export $RUN policy_$STEP episode $EP world_count=$W traj=$TRAJ ===" | tee -a "$LOG"
V2D_GPU=1 V2D_CONTAINER_NAME="fc-export-$RUN" scripts/dev/in-flash-chord "$CMD" >> "$LOG" 2>&1 9>&-
rc=$?
NEW="$(tail -n +"$((n0 + 1))" "$LOG")"
grep -a -F 'in-flash-chord: /workspace ← ' <<< "$NEW" | tail -1   # 这一次挂的是哪份源码
if [ "$rc" != 0 ]; then
  # 失败的这一次可能已写出 JSON（例如换参考时包装层核对不过、判作废，返回 2）：删掉，免得下次运行当成“已有结果”跳过
  rm -f "$J" "$Q"
  grep -q 'must disable task failure termination' <<< "$NEW" \
    && die "评测器报 must disable task failure termination（退出码 ${rc}）：按规矩二退到同一 run 更早的有限 checkpoint。见 $LOG" 13
  die "导出失败（退出码 ${rc}），见 $LOG" 3
fi
[ -s "$J" ] && { [ "$TRAJ" = 0 ] || [ -s "$Q" ]; } || die "评测器退出码 0，但缺 $J 或 ${Q}，见 $LOG" 3
echo "fc_export: $RUN policy_${STEP}（episode ${EP}，$W 世界）→ $J$([ "$TRAJ" = 1 ] && echo " 与 $Q")"; summary
exit 0
