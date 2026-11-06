#!/usr/bin/env bash
# 评一份没训完的 checkpoint（ep7 方盒两条 run 发散前最后一份有限的 policy），核候选条件，再跑一次取证 rollout。
# 训练机上、仓库根执行，放 tmux。计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 A 部分（第 3 版，按审查修改）。
# 用法：V2D_EVAL_SRC=<8c778f9 副本> V2D_ROLLOUT_SRC=<v0.4.0 + #185 副本> stages/evaluate/ep7_ckpt_eval.sh <run> <步数> <期望帧数> <日志名>
#       V2D_EP7_EVAL_DRYRUN=1：只做第 0 步的检查并打印将要执行的命令，不写日志、不占 GPU。
# 0. 拒绝（退出码 2，不写日志）：日志 runs/logs/<日志名>.log 已存在；policy 文件不在；这一步的评测 JSON 或取证 npz 已存在（不沿用旧结果、不覆盖旧证据）；
#    两份源码副本过不了 fc_src_check.py（8c778f9 副本对 flash_chord-8c778f9.sha256，#185 副本对 flash_chord-p185.sha256）；
#    ckpt_nonfinite.py 没有给出这一步“非有限 0”的那一行（规矩一：含 NaN 的 policy 不评、不导出；读不了的也拒绝）。
#    开头 unset V2D_EVAL_WORLDS、V2D_ALLOW_SHARED_GPU：世界数固定 4096，fc_eval 自己的 GPU 检查不许关。
# 1. 写长链标记 runs/.chain-active-<日志名>；日志开头记下本脚本、blowup_rollout.py、diag_palms.py 的 md5。
# 2. GPU 上有进程就每分钟看一次，最多等 V2D_GPU_WAIT_S 秒（默认 7200），等不到就停（EP7_EVAL_FAILED：什么都还没跑）。
# 3. 评测：V2D_FC_SRC=<8c778f9 副本> stages/policy/fc_eval.sh <run> <步数>（4096 世界，与方盒评测链同一运行时），经 record 落成
#    results/ep7/eval_<run>_s<步数>.txt。fc_eval 退出码 6（GPU 被别的进程占了）：等 GPU 空了重跑，等不到就停。
#    评测完（不论成败）把 fc_eval 自己的日志里的要点行摘到 results/ep7/evallog_<run>_s<步数>.txt（不占 GPU）：挂的是哪份源码、是否走满期望帧数、
#    评测器报了几个非有限世界、有没有报 must disable task failure termination。评测的结局分三种：成功；失败（终止报错）；失败（其他，等于没评成）。
# 4. 候选条件 A-2（评测成功才判）：评测 JSON 的 world_count 为 4096、non_finite_worlds 是空列表、step_count 等于期望帧数、
#    checkpoint_sha256 等于这份 policy 文件的 sha256。逐项打印，落到 results/ep7/a2_<run>_s<步数>.txt。
# 5. 取证 rollout（不论评测结果都跑）：再等一次 GPU，用 <v0.4.0 + #185 副本> 跑 stages/evaluate/blowup_rollout.py（4096 世界，逐步抄物体与手掌位置，
#    不清零）→ runs/ep7/<run>/blowup_s<步数>_w4096.npz，记录 results/ep7/blowup_<run>_s<步数>.txt。评分阶段的终止报错会连同 Traceback 打进输出，
#    所以对它设 V2D_RECORD_TRACEBACK_OK=1（脚本自己的退出码就是结论）。等 GPU 超时：记“取证 没跑（GPU 被占）”，不算整条链失败。
# 6. 日志最后一行 EP7_EVAL_DONE（评测 …；评测器记的非有限世界 …；A-2 …；取证 …）。
set -uo pipefail
cd "$(dirname "$0")/../.."
unset V2D_EVAL_WORLDS V2D_ALLOW_SHARED_GPU
RUN=${1:?用法见文件头}; STEP=${2:?用法见文件头}; FRAMES=${3:?用法见文件头}; NAME=${4:?用法见文件头}
EVAL_SRC=${V2D_EVAL_SRC:?要给 V2D_EVAL_SRC（8c778f9 副本）}; ROLLOUT_SRC=${V2D_ROLLOUT_SRC:?要给 V2D_ROLLOUT_SRC（v0.4.0 + #185 副本）}
LOG="runs/logs/$NAME.log"; MARK="runs/.chain-active-$NAME"; GPU_WAIT_S="${V2D_GPU_WAIT_S:-7200}"
P="runs/flash_chord/$RUN/policy_$STEP.safetensors"; E="runs/flash_chord/$RUN/eval_${STEP}_w4096.json"
ELOG="runs/logs/$RUN.eval_${STEP}_w4096.log"; D="runs/ep7/$RUN"; NPZ="$D/blowup_s${STEP}_w4096.npz"
die() { echo "ep7_ckpt_eval: $*" >&2; exit 2; }
[[ $RUN =~ ^t[12]-ep[0-9]+-[a-z0-9_-]+$ && $STEP =~ ^[0-9]+$ && $FRAMES =~ ^[1-9][0-9]*$ && $NAME =~ ^[a-z0-9_]+$ ]] || die "参数不对：$RUN $STEP $FRAMES $NAME"
[[ $GPU_WAIT_S =~ ^[0-9]+$ ]] || die "V2D_GPU_WAIT_S 必须是非负整数"
[ ! -e "$LOG" ] || die "$LOG 已存在，拒绝（确认后改名或删掉再起）"
[ -f "$P" ] || die "没有 $P"
[ ! -e "$E" ] || die "$E 已存在：不沿用旧的评测结果，确认后移走再起"
[ ! -e "$NPZ" ] || die "$NPZ 已存在：不覆盖旧的取证结果，确认后移走再起"
python3 stages/policy/fc_src_check.py "$EVAL_SRC" third_party/patches/flash_chord-8c778f9.sha256 > /dev/null || die "$EVAL_SRC 不是核对得过的 8c778f9 副本"
python3 stages/policy/fc_src_check.py "$ROLLOUT_SRC" third_party/patches/flash_chord-p185.sha256 > /dev/null || die "$ROLLOUT_SRC 不是核对得过的 v0.4.0 + #185 副本"
scan="$(python3 stages/policy/ckpt_nonfinite.py --steps "$STEP" "$RUN" 2>&1 || true)"
grep -qE '步：非有限 0 / ' <<< "$scan" || die "ckpt_nonfinite.py 没有给出 policy_$STEP “非有限 0”的那一行，不评"
EVAL_CMD=(scripts/dev/record "ep7/eval_${RUN}_s${STEP}" stages/policy/fc_eval.sh "$RUN" "$STEP")
ROLL_CMD="cd /workspace && python /repo/stages/evaluate/blowup_rollout.py /repo/$P /repo/${NPZ%.npz} --worlds 4096"
if [ "${V2D_EP7_EVAL_DRYRUN:-0}" = 1 ]; then
  echo "DRYRUN ${NAME}：policy $P 有限；期望帧数 ${FRAMES}；评测运行时 $(head -1 "$EVAL_SRC/SOURCE_V2D.txt")；取证运行时 $(head -1 "$ROLLOUT_SRC/SOURCE_V2D.txt")"
  echo "DRYRUN GPU 最多等 ${GPU_WAIT_S} 秒"
  echo "DRYRUN 评测：V2D_FC_SRC=$EVAL_SRC ${EVAL_CMD[*]}  → ${E}"
  echo "DRYRUN 取证（评测之后总是跑）：V2D_FC_SRC=$ROLLOUT_SRC in-flash-chord \"$ROLL_CMD\""
  echo "EP7_EVAL_DRYRUN_OK"; exit 0
fi
mkdir -p runs/logs "$D" results/ep7
touch "$MARK"; trap 'rm -f "$MARK"' EXIT
say() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$LOG"; }
stop() { say "$*"; say "EP7_EVAL_FAILED"; exit 1; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -; }
wait_gpu() {  # GPU 上有进程就每分钟看一次，最多 GPU_WAIT_S 秒；等到返回 0，超时返回 1
  local t0=$SECONDS
  [ -z "$(gpu_pids)" ] && return 0
  say "GPU 上有别的进程（$(gpu_pids)），每 ${V2D_POLL_S:-60} 秒看一次"
  while [ -n "$(gpu_pids)" ]; do
    [ $((SECONDS - t0)) -lt "$GPU_WAIT_S" ] || return 1
    sleep "${V2D_POLL_S:-60}"
  done
  say "GPU 空了"
}
md5of() { { md5sum "$1" 2>/dev/null || true; } | cut -c1-12; }
say "start ${NAME}：${RUN} policy_${STEP}，期望 ${FRAMES} 帧；评测运行时 $(head -1 "$EVAL_SRC/SOURCE_V2D.txt")；来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
say "脚本 md5：ep7_ckpt_eval.sh $(md5of stages/evaluate/ep7_ckpt_eval.sh)，blowup_rollout.py $(md5of stages/evaluate/blowup_rollout.py)，diag_palms.py $(md5of stages/evaluate/diag_palms.py)"

# 3. 评测
wait_gpu || stop "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，评测没跑"
say "评测 ${RUN} policy_${STEP}（4096 世界）"
while :; do
  rc=0
  V2D_FC_SRC="$EVAL_SRC" "${EVAL_CMD[@]}" >> "$LOG" 2>&1 < /dev/null || rc=$?
  [ "$rc" = 6 ] || break
  rm -f "results/ep7/eval_${RUN}_s${STEP}.txt.failed"
  say "fc_eval 报 GPU 上有别的进程（退出码 6），等它空出来再评"
  wait_gpu || stop "评测时 GPU 被别的进程占了 ${GPU_WAIT_S} 秒，评测没跑成"
done
if [ -f "$ELOG" ]; then
  scripts/dev/record "ep7/evallog_${RUN}_s${STEP}" grep -a -n -E \
    "in-flash-chord: /workspace|=== .* fc_eval |evaluation step ${FRAMES}/${FRAMES}\$|worlds recorded non-finite|Error executing job|^[A-Za-z]+Error: " "$ELOG" >> "$LOG" 2>&1 < /dev/null || true
fi
ev_nf="$(grep -a -oE '[0-9]+ worlds recorded non-finite' "$ELOG" 2>/dev/null | tail -1 | cut -d' ' -f1 || true)"
if [ "$rc" = 0 ] && [ -s "$E" ]; then
  eval_state=成功
elif grep -a -q 'must disable task failure termination' "$ELOG" 2>/dev/null; then
  eval_state="失败（终止报错）"
else
  eval_state="失败（其他，等于没评成，退出码 ${rc}）"
fi

# 4. 候选条件 A-2
a2=没判; nf=未知
if [ "$eval_state" = 成功 ]; then
  a2=不满足
  if scripts/dev/record "ep7/a2_${RUN}_s${STEP}" python3 - "$E" "$P" "$FRAMES" >> "$LOG" 2>&1 <<'PY'
import hashlib, json, sys
e = json.load(open(sys.argv[1]))
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
nf = e.get("non_finite_worlds")
checks = [
    ("world_count 为 4096", e.get("world_count") == 4096, e.get("world_count")),
    ("non_finite_worlds 是空列表", isinstance(nf, list) and len(nf) == 0, f"{len(nf)} 个" if isinstance(nf, list) else repr(nf)),
    (f"step_count 等于期望帧数 {sys.argv[3]}", e.get("step_count") == int(sys.argv[3]), e.get("step_count")),
    ("checkpoint_sha256 等于 policy 文件", e.get("checkpoint_sha256") == sha, f"{str(e.get('checkpoint_sha256'))[:16]}… 对 {sha[:16]}…"),
]
for name, ok, value in checks:
    print(f"  {'满足' if ok else '不满足'}  {name}：{value}")
ok = all(c[1] for c in checks)
print("A2_OK" if ok else "A2_FAIL")
sys.exit(0 if ok else 1)
PY
  then a2=满足; fi
  nf="$(python3 -c 'import json, sys
v = json.load(open(sys.argv[1])).get("non_finite_worlds")
print(len(v) if isinstance(v, list) else "未知")' "$E" 2>/dev/null || echo 未知)"
elif [ -n "$ev_nf" ]; then
  nf="$ev_nf"
fi
say "评测${eval_state}；评测器记的非有限世界 ${nf}；A-2 ${a2}"

# 5. 取证 rollout
roll_state="没跑（GPU 被占）"
if wait_gpu; then
  say "取证 rollout（4096 世界，物体与手掌位置原样存）"
  if V2D_RECORD_TRACEBACK_OK=1 V2D_FC_SRC="$ROLLOUT_SRC" V2D_GPU=1 V2D_CONTAINER_NAME="ep7-blowup-$RUN" scripts/dev/record "ep7/blowup_${RUN}_s${STEP}" \
       scripts/dev/in-flash-chord "$ROLL_CMD" >> "$LOG" 2>&1 < /dev/null && [ -s "$NPZ" ]; then
    bad="$(grep -a -oE 'BLOWUP_ROLLOUT_OK bad_worlds=[0-9]+' "results/ep7/blowup_${RUN}_s${STEP}.txt" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
    roll_state="完成（崩掉的世界 ${bad:-未知}）"
  else
    roll_state=失败
  fi
  rm -rf "${NPZ%.npz}_hydra"
fi
say "EP7_EVAL_DONE（评测 ${eval_state}；评测器记的非有限世界 ${nf}；A-2 ${a2}；取证 ${roll_state}）"
