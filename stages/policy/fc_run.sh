#!/usr/bin/env bash
# FlashCHORD 训练的宿主机入口（训练机上执行，放在 tmux 里）：断点续训、W&B、运行记录、自动重试。
#
# 用法：stages/policy/fc_run.sh <run> <parquet（容器内路径）> <world_count> <total_steps> [hydra 覆盖...]
# 可选环境变量：
#   V2D_WANDB=1|0             是否记到 W&B（默认 1）；V2D_WANDB_TAG=tier1 作为 W&B tag
#   V2D_WANDB_ENTITY / V2D_WANDB_PROJECT  默认 zhou-rongx-northeastern-university / v2d-track2（自测用 v2d-selftest）
#   V2D_WANDB_UPLOAD_CKPT=1|0 是否把 policy checkpoint 传成 W&B artifact（默认 1）
#   V2D_WANDB_PROXY=1|0       W&B 是否走 .env 里的 V2D_PROXY（默认 0；10/30 起训练机直连 W&B 返回 403）
#   V2D_GPU_SAMPLE_MS=2000    GPU 采样间隔；V2D_GPU_APPS=1 另记按进程的显存（.apps）
#   V2D_MAX_RETRY=2           意外退出后从最新 checkpoint 重试的次数；V2D_RETRY_BACKOFF_S=30 每次重试前等 30×次数 秒
#   V2D_KEEP_STATES=2         训练中与结束后保留的 state_* 份数（≥1；policy_* 全留）
#   V2D_ALLOW_SHARED_GPU=1    GPU 上有别的进程时也照样启动（默认拒绝）
#   V2D_FORCE_RESUME=1        续训时即使 parquet/参数/镜像与本 lineage 起点不同也续（默认拒绝）
#   V2D_FORCE_RESTART=1       目录里有 checkpoint 但没有一份完整时，仍从 0 开始（默认拒绝）
#   V2D_PRINT_CMD=1           只打印将在容器里执行的命令后退出（不起容器、不写记录；配置核对用）
#   TF_CPP_MIN_LOG_LEVEL、XLA_* 会被 in-flash-chord 转发进容器
#
# 设计：
#   - 容器后台运行（名字 fc-<run>，标签 v2d.owner=<用户>），本脚本 docker logs -f 实时收日志、docker wait 取退出码，
#     删容器前把这一段的完整日志存到 runs/logs/<run>.s<段>.log，成功与否只从这个文件判定。
#     本脚本（或 tmux）死掉不影响训练；再运行本脚本会接管还在跑的容器，或回收已结束容器的退出码。
#   - 断点续训：同一条命令重跑即可。从步数最大的完整 state_N 续；续训前与本 lineage 起点（最近一次从 0 开始的记录）
#     比对 parquet sha256、覆盖参数、world_count、总步数、镜像 ID，不一致就拒绝。
#   - W&B：每一段一个独立 run，id = <run>-s<段号>，group = <run>（同一 id 续写会让 wandb 丢掉与上一段重叠的步数）。
#   - 退出分类：成功 → 剪枝、写 DONE；STOP 标记 → 3；显存不足 → 4；正在接管/运行中的容器被 stop/kill（137/143）→ 5；
#     其他非零、以及回收到的已退出容器（例如机器重启）→ 退避后自动重试，最多 V2D_MAX_RETRY 次。
#   - 有意停止用 stages/policy/fc_stop.sh <run>；Ctrl-C 或关掉 tmux 只结束本脚本，不停训练。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
die() { echo "fc_run: $*" >&2; exit "${2:-2}"; }
[ $# -ge 4 ] || die "用法：fc_run.sh <run> <parquet> <world_count> <total_steps> [覆盖...]"
RUN=$1 PARQ=$2 W=$3 STEPS=$4; shift 4
OVR_ARR=("$@"); OVR="$*"
[[ $RUN =~ ^[a-z0-9][a-z0-9_-]{2,59}$ ]] || die "run 名只能用小写字母、数字、- _（3–60 字符）：$RUN"
[[ $W =~ ^[0-9]+$ && $STEPS =~ ^[0-9]+$ && $W -gt 0 ]] || die "world_count / total_steps 必须是正整数"
(( STEPS % W == 0 )) || die "total_steps 必须能被 world_count 整除"
KEEP="${V2D_KEEP_STATES:-2}"; [[ $KEEP =~ ^[1-9][0-9]*$ ]] || die "V2D_KEEP_STATES 必须 ≥ 1"
# r3o（计划 agent-doc/plan-r3o-2026-10-15.md 2.1，代码审查第 7 条）：run 名带 -r3o-、设了 V2D_OBJ_ORI_WEIGHT、训练副本打了物体朝向补丁
# （SOURCE_V2D.txt 里有 object-orientation），三样要么都有，要么都没有。否则手动起的 r3o run 会在普通副本上按权重 0 训、却叫 r3o。
HAS_OO=0
[ -n "${V2D_FC_SRC:-}" ] && grep -q object-orientation "$V2D_FC_SRC/SOURCE_V2D.txt" 2>/dev/null && HAS_OO=1
if [[ $RUN == *-r3o-* ]] || [ -n "${V2D_OBJ_ORI_WEIGHT:-}" ] || [ "$HAS_OO" = 1 ]; then
  [[ $RUN == *-r3o-* ]] && [ -n "${V2D_OBJ_ORI_WEIGHT:-}" ] && [ "$HAS_OO" = 1 ] \
    || die "r3o 的 run 名（-r3o-）、V2D_OBJ_ORI_WEIGHT、打了物体朝向补丁的副本要同时具备：run ${RUN}，V2D_OBJ_ORI_WEIGHT=${V2D_OBJ_ORI_WEIGHT:-未设}，V2D_FC_SRC=${V2D_FC_SRC:-未设}"
fi
ME="$(id -un)"

OUT="runs/flash_chord/$RUN"; LOGDIR=runs/logs
mkdir -p "$OUT" "$LOGDIR" runs/cache
LOG="$LOGDIR/$RUN.log"; GPULOG="$LOGDIR/$RUN.gpu"; APPLOG="$LOGDIR/$RUN.apps"; REC="$OUT/run_record.jsonl"
CNAME="fc-$RUN"
[ -f "$OUT/DONE" ] && { echo "fc_run: $RUN 已完成（$OUT/DONE），跳过"; exit 0; }
# 锁在 fd 9 上；所有后台子进程都关掉它（9>&-），否则孤儿进程会在本脚本退出后继续占着锁
exec 9>"$OUT/.lock"; flock -n 9 || die "$OUT 正被另一个 fc_run 占用"
touch "$LOG"

WB=(logging.mode=disabled)
if [ "${V2D_WANDB:-1}" = 1 ]; then
  # key 只从 600 权限的 .env 里取这一项，不把代理等其他变量带进来；值不进 argv、日志与记录
  WANDB_API_KEY="$(sed -n 's/^WANDB_API_KEY=//p' .env 2>/dev/null | tail -1)"
  [ -n "$WANDB_API_KEY" ] || die ".env 里没有 WANDB_API_KEY（或设 V2D_WANDB=0）"
  export WANDB_API_KEY WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-180}"
  # V2D_WANDB_PROXY=1：W&B 走 .env 里的 V2D_PROXY。10/30 起两台训练机直连 api.wandb.ai 返回 403（出口 IP 在 CN），
  # 经代理（出口 SG）可达；值同 key 一样只经环境变量按名字转发进容器，不进 argv、日志与记录
  if [ "${V2D_WANDB_PROXY:-0}" = 1 ]; then
    HTTPS_PROXY="$(sed -n 's/^V2D_PROXY=//p' .env 2>/dev/null | tail -1)"
    [ -n "$HTTPS_PROXY" ] || die ".env 里没有 V2D_PROXY（V2D_WANDB_PROXY=1 要用）"
    export HTTPS_PROXY https_proxy="$HTTPS_PROXY" NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
  fi
fi

host_path() {  # 容器内路径 → 宿主机路径
  case "$1" in
    /repo/*) echo "$ROOT/${1#/repo/}" ;;
    /v2d/assets/*) echo "$ROOT/third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/${1#/v2d/assets/}" ;;
    *) echo "$1" ;;
  esac
}
PARQ_HOST="$(host_path "$PARQ")"; [ -f "$PARQ_HOST" ] || die "找不到 parquet：$PARQ_HOST"
PARQ_SHA="$(sha256sum "$PARQ_HOST" | cut -d' ' -f1)"
IMG="$(docker image inspect flash_chord:latest --format '{{.Id}}' 2>/dev/null)" || die "没有 flash_chord:latest 镜像"
UPSTREAM="$(git -C third_party/video_to_data rev-parse --short=7 HEAD 2>/dev/null || echo unknown)"

record() {  # 追加一行 JSON 运行记录（不含凭据）
  python3 - "$REC" "$@" <<'PY'
import json, sys, time
path, kv = sys.argv[1], sys.argv[2:]
row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
for item in kv:
    k, _, v = item.partition("=")
    row[k] = v
with open(path, "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}
lineage_start() {  # 本 lineage 起点（最近一次 resume_from=none 的 start）里的字段；没有则输出 __NONE__
  python3 - "$REC" "$1" <<'PY'
import json, sys
out = "__NONE__"
try:
    for line in open(sys.argv[1]):
        r = json.loads(line)
        if r.get("event") == "start" and r.get("resume_from") == "none":
            out = r.get(sys.argv[2], "__NONE__")
except FileNotFoundError:
    pass
print(out)
PY
}
count_starts() { local n=0; [ -f "$REC" ] && n="$(grep -c '"event": "start"' "$REC")"; echo "${n:-0}"; }
image_fingerprint() {  # 镜像 base 层与依赖 freeze 的指纹（ADR-00003），按镜像 ID 缓存
  local c="runs/cache/image-${IMG#sha256:}.json"
  if [ ! -s "$c" ]; then
    local layers freeze
    layers="$(docker image inspect flash_chord:latest --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | head -11 | sha256sum | cut -c1-16)"
    freeze="$(V2D_GPU=0 scripts/dev/in-flash-chord "python /repo/stages/policy/deps_freeze.py" 2>/dev/null | sha256sum | cut -c1-16)"
    printf '{"base11_sha": "%s", "freeze_sha": "%s"}\n' "$layers" "$freeze" > "$c"
  fi
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['base11_sha'], d['freeze_sha'])" "$c"
}
owner_ok() {  # 同名容器必须是本用户用 in-flash-chord 起的
  local o; o="$(docker inspect -f '{{index .Config.Labels "v2d.owner"}}' "$CNAME" 2>/dev/null || true)"
  [ -z "$o" ] || [ "$o" = "$ME" ]
}

BG=()
stop_bg() { for p in "${BG[@]}"; do kill "$p" 2>/dev/null; done; for p in "${BG[@]}"; do wait "$p" 2>/dev/null; done; BG=(); }
trap 'stop_bg' EXIT
trap 'stop_bg; exit 130' HUP INT TERM
sampler() {  # 本脚本活着才采样（本脚本被 SIGKILL 时采样进程也会自己退出）
  local q=$1 out=$2 iv; iv="$(awk "BEGIN{print ${V2D_GPU_SAMPLE_MS:-2000}/1000}")"
  ( while kill -0 $$ 2>/dev/null; do nvidia-smi "$q" --format=csv,noheader 2>/dev/null; sleep "$iv"; done >> "$out" ) 9>&- &
  BG+=($!)
}
prune_states() {  # checkpoint 是原子写入（.<名字>.tmp 再改名），可见的 state_N 都完整；按数字排序只留最近 KEEP 份
  ls "$OUT"/state_*.safetensors 2>/dev/null | sed -E 's/.*state_([0-9]+)\.safetensors/\1/' | sort -n \
    | head -n -"$KEEP" | while read -r s; do rm -f "$OUT/state_${s}.safetensors"; done
}
start_bg() {
  sampler --query-gpu=timestamp,memory.used,utilization.gpu "$GPULOG"
  [ "${V2D_GPU_APPS:-0}" = 1 ] && sampler --query-compute-apps=timestamp,pid,used_memory "$APPLOG"
  ( while kill -0 $$ 2>/dev/null; do sleep 60; prune_states; done ) 9>&- & BG+=($!)
}
seg_has() { [ -f "$SEGLOG" ] && grep -qE "$1" "$SEGLOG"; }   # 只读文件、不走管道（避免 pipefail 下 SIGPIPE 误判）

# 上次 fc_stop 留下、但对应容器早已不在的 STOP：这次重跑就是要继续，清掉
if [ -f "$OUT/STOP" ] && [ -z "$(docker inspect -f '{{.State.Status}}' "$CNAME" 2>/dev/null || true)" ]; then
  rm -f "$OUT/STOP"; echo "fc_run: 清除了上次残留的 STOP 标记" | tee -a "$LOG"; record event=stale_stop_cleared
fi

attempt=0; max_retry="${V2D_MAX_RETRY:-2}"
while :; do
  attempt=$((attempt + 1)); reaped=0
  status="$(docker inspect -f '{{.State.Status}}' "$CNAME" 2>/dev/null || true)"
  [ -z "$status" ] || owner_ok || die "同名容器 $CNAME 不是 $ME 起的，不接管也不删除"
  seg="$(count_starts)"
  if [ "$status" = running ]; then
    [ "${V2D_PRINT_CMD:-0}" = 1 ] && die "容器 $CNAME 正在运行，不打印"
    echo "=== $(date '+%F %T %Z') fc_run $RUN 接管仍在运行的容器 $CNAME ===" >> "$LOG"
    record event=adopt attempt="$attempt" segment="$seg"
    docker logs -f --since "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CNAME" >> "$LOG" 2>&1 9>&- & BG+=($!)
    start_bg
    rc="$( exec 9>&-; docker wait "$CNAME" 2>/dev/null || echo 255 )"
  elif [ -n "$status" ]; then
    [ "${V2D_PRINT_CMD:-0}" = 1 ] && die "有未回收的容器 $CNAME，不打印"
    rc="$(docker inspect -f '{{.State.ExitCode}}' "$CNAME" 2>/dev/null || echo 255)"; reaped=1
    echo "=== $(date '+%F %T %Z') fc_run $RUN 回收已结束的容器 $CNAME（退出码 $rc）===" >> "$LOG"
    record event=reap attempt="$attempt" segment="$seg" rc="$rc"
  else
    # 开新段之前：有意停止优先；GPU 必须没有别人的进程；目录里残留的临时文件清掉（此时没有写入者）
    if [ -f "$OUT/STOP" ]; then rm -f "$OUT/STOP"; record event=stopped; echo "fc_run: $RUN 按 STOP 停止（未开新段）"; exit 3; fi
    others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
    [ -z "$others" ] || [ "${V2D_ALLOW_SHARED_GPU:-0}" = 1 ] || [ "${V2D_PRINT_CMD:-0}" = 1 ] \
      || die "GPU 上有别的进程（PID ${others//$'\n'/,}），拒绝启动（V2D_ALLOW_SHARED_GPU=1 可强制）" 6
    [ "${V2D_PRINT_CMD:-0}" = 1 ] || rm -f "$OUT"/.*.tmp
    LATEST="$(python3 stages/policy/ckpt_latest.py "$OUT")" || die "ckpt_latest.py 出错"
    if [ -z "$LATEST" ] && ls "$OUT"/state_*.safetensors "$OUT"/policy_*.safetensors >/dev/null 2>&1 \
       && [ "${V2D_FORCE_RESTART:-0}" != 1 ]; then
      die "$OUT 里有 checkpoint 但没有一份完整的，拒绝从 0 开始覆盖（确认后设 V2D_FORCE_RESTART=1）"
    fi
    RESUME=()
    if [ -n "$LATEST" ]; then
      if [ "${V2D_FORCE_RESUME:-0}" != 1 ]; then
        for kv in "parquet_sha256=$PARQ_SHA" "overrides=$OVR" "world_count=$W" "total_steps=$STEPS" "image=$IMG" \
                  "fc_src=${V2D_FC_SRC:-image}" "obj_ori_weight=${V2D_OBJ_ORI_WEIGHT:-}" "obj_ori_std=${V2D_OBJ_ORI_STD:-}"; do   # 后三项：r3o 审查第 6 条（旧记录里没有就跳过）
          k="${kv%%=*}"; want="$(lineage_start "$k")"
          [ "$want" = __NONE__ ] || [ "$want" = "${kv#*=}" ] || die "续训被拒：$k 与本 lineage 起点不同（起点：$want）。确认无误可设 V2D_FORCE_RESUME=1"
        done
      fi
      RESUME=(training.resume=true "training.checkpoint_path=/repo/$OUT/state_${LATEST}.safetensors")
    fi
    seg=$((seg + 1))
    if [ "${V2D_WANDB:-1}" = 1 ]; then
      export WANDB_RUN_ID="$RUN-s$seg" WANDB_TAGS="${V2D_WANDB_TAG:-untagged},$(hostname -s)"
      WB=(logging.mode=online "logging.entity=${V2D_WANDB_ENTITY:-zhou-rongx-northeastern-university}"
          "logging.project=${V2D_WANDB_PROJECT:-v2d-track2}" "logging.experiment=$RUN" "logging.run_name=$RUN-s$seg")
      [ "${V2D_WANDB_UPLOAD_CKPT:-1}" = 1 ] || WB+=(logging.upload_checkpoints=false)
    fi
    CMD="bash /repo/stages/policy/fc_train.sh $(printf '%q ' "$PARQ" "$W" "$STEPS" "/repo/$OUT" "${WB[@]}" "${RESUME[@]}" \
         "hydra.run.dir=/repo/$OUT/hydra/s$seg" "${OVR_ARR[@]}")"
    [ "${V2D_PRINT_CMD:-0}" = 1 ] && { echo "$CMD"; exit 0; }
    read -r FP_BASE FP_FREEZE <<< "$(image_fingerprint)"
    record event=start attempt="$attempt" segment="$seg" run="$RUN" host="$(hostname)" \
      gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)" image="$IMG" \
      image_base11_sha="$FP_BASE" deps_freeze_sha="$FP_FREEZE" \
      source_commit="$(cat .v2d-source-commit 2>/dev/null || echo unknown)" upstream="$UPSTREAM" \
      parquet="$PARQ" parquet_sha256="$PARQ_SHA" world_count="$W" total_steps="$STEPS" \
      overrides="$OVR" resume_from="${LATEST:-none}" wandb_id="${WANDB_RUN_ID:-disabled}" wandb_proxy="${HTTPS_PROXY:+on}" \
      fc_src="${V2D_FC_SRC:-image}" obj_ori_weight="${V2D_OBJ_ORI_WEIGHT:-}" obj_ori_std="${V2D_OBJ_ORI_STD:-}"
    echo "=== $(date '+%F %T %Z') fc_run $RUN 第 $seg 段（尝试 $attempt）resume_from=${LATEST:-none} ===" >> "$LOG"
    if ! V2D_DETACH=1 V2D_GPU=1 V2D_CONTAINER_NAME="$CNAME" scripts/dev/in-flash-chord "$CMD" >> "$LOG" 2>&1; then
      record event=abort reason=docker_run_failed; die "docker run 失败，见 $LOG" 7
    fi
    docker logs -f "$CNAME" >> "$LOG" 2>&1 9>&- & BG+=($!)
    start_bg
    rc="$( exec 9>&-; docker wait "$CNAME" 2>/dev/null || echo 255 )"
  fi
  sleep 2; stop_bg
  SEGLOG="$LOGDIR/$RUN.s${seg}.log"
  docker logs "$CNAME" >> "$SEGLOG" 2>&1          # 这一段的完整输出（判定只看它）
  docker rm -f "$CNAME" >/dev/null 2>&1
  record event=exit attempt="$attempt" segment="$seg" rc="$rc"

  if [ "$rc" = 0 ] && [ -f "$OUT/policy_${STEPS}.safetensors" ] && seg_has "completed ${STEPS} environment steps"; then
    prune_states; date '+%F %T %Z' > "$OUT/DONE"; record event=done
    echo "fc_run: $RUN 完成"; exit 0
  fi
  if [ -f "$OUT/STOP" ]; then
    rm -f "$OUT/STOP"; record event=stopped
    echo "fc_run: $RUN 已按 STOP 停止；重跑同一条命令即从断点继续"; exit 3
  fi
  if seg_has 'RESOURCE_EXHAUSTED|exceeds the base limit'; then
    record event=abort reason=oom; die "$RUN 显存不足，不自动重试（见 $SEGLOG）" 4
  fi
  if [ "$reaped" = 0 ] && { [ "$rc" = 137 ] || [ "$rc" = 143 ]; }; then
    record event=abort reason=killed rc="$rc"; die "$RUN 被外部停止（退出码 $rc），不自动重试；重跑即从断点继续" 5
  fi
  if [ "$rc" = 0 ]; then
    record event=abort reason=exit0_incomplete; die "$RUN 退出码 0 但没跑完（缺 policy_${STEPS} 或 completed 行）" 8
  fi
  (( attempt > max_retry )) && { record event=abort reason=retries_exhausted; die "$RUN 重试 $max_retry 次仍失败（退出码 $rc）"; }
  wait_s=$(( ${V2D_RETRY_BACKOFF_S:-30} * attempt ))
  echo "fc_run: $RUN 退出码 $rc，${wait_s} 秒后从最新 checkpoint 重试（期间 fc_stop 可取消）" | tee -a "$LOG"
  sleep "$wait_s"
done
