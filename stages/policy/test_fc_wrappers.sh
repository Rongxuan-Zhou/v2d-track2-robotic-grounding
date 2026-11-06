#!/usr/bin/env bash
# fc_run.sh / fc_queue.sh / fc_stop.sh / ckpt_latest.py / fc_train.sh / in-flash-chord 的行为测试
# （暂无契约 ID：运行规范待成文）。在训练机上的临时沙箱里跑：假的 docker（模拟后台容器的
# run/inspect/logs/wait/stop/rm 与标签）、假的 nvidia-smi、假训练进程；不占 GPU、不起真容器，结束后删除沙箱。
# 用法：bash stages/policy/test_fc_wrappers.sh
set -uo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd -P)"
SB="$(mktemp -d "$HOME/v2d/_fc_test.XXXXXX")"; trap 'pkill -f "$SB/" 2>/dev/null; rm -rf "$SB"' EXIT
PASS=0; FAIL=0
check() { if eval "$2"; then PASS=$((PASS + 1)); echo "  PASS  $1"; else FAIL=$((FAIL + 1)); echo "  FAIL  $1"
          [ -f "$SB/out.txt" ] && sed "s/^/        | /" "$SB/out.txt" | tail -5; fi; }

R="$SB/repo"; mkdir -p "$R/stages/policy" "$R/scripts/dev" "$R/runs/motion" "$SB/bin" "$SB/ctr"
cp "$SRC"/stages/policy/{fc_run.sh,fc_queue.sh,fc_stop.sh,ckpt_latest.py,fc_train.sh,deps_freeze.py} "$R/stages/policy/"
sed -i 's/sleep 2; stop_bg/stop_bg/' "$R/stages/policy/fc_run.sh"   # 测试里不等日志落盘
cd "$R"
KEY=wandb_v1_TESTKEY_must_never_leak
printf 'V2D_PROXY=http://user:secretproxy@h:1\nWANDB_API_KEY=%s\n' "$KEY" > .env; chmod 600 .env
echo "abc1234-test" > .v2d-source-commit
echo "parquet bytes" > runs/motion/p.parquet; PQ=/repo/runs/motion/p.parquet
export FAKE_ROOT="$R" CTR="$SB/ctr" PATH="$SB/bin:$PATH" V2D_RETRY_BACKOFF_S=0 ME="$(id -un)"

# ── 假 nvidia-smi：FAKE_GPU_BUSY=1 时报告一个别人的进程 ─────────────
cat > "$SB/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *compute-apps=pid\ *|*compute-apps=pid) [ "${FAKE_GPU_BUSY:-0}" = 1 ] && echo "4242"; exit 0 ;;
  *) echo "FakeGPU, 999.99" ;;
esac
EOF
# ── 假训练：写 checkpoint（先 policy 后 state），按 FAKE_MODE 出错 ─────
cat > "$SB/bin/fake-train" <<'EOF'
#!/usr/bin/env bash
cmd="$1"; out=$(echo "$cmd" | grep -oE '/repo/runs/flash_chord/[a-z0-9_-]+' | head -1); out="${out/\/repo/$FAKE_ROOT}"
read -r -a t <<< "$cmd"; steps=""; for ((i=1;i<${#t[@]};i++)); do [[ ${t[$i]} =~ ^[0-9]+$ && ${t[$((i-1))]} =~ ^[0-9]+$ ]] && { steps=${t[$i]}; break; }; done
start=0; r=$(echo "$cmd" | grep -oE 'state_[0-9]+\.safetensors' | head -1); [ -n "$r" ] && start=${r//[^0-9]/}
mk() { python3 - "$1" "$2" <<'PY'
import json, struct, sys
h = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}, "__metadata__": {}}).encode()
blob = struct.pack("<Q", len(h)) + h + b"\0" * 16
open(sys.argv[1], "wb").write(blob[:-5] if sys.argv[2] == "1" else blob)
PY
}
once() { [ -f "$CTR/.once" ] && return 1; touch "$CTR/.once"; }
iv=$(( steps / 4 ))
[ "${FAKE_MODE:-ok}" = crash_before_ckpt ] && once && { echo "early crash"; exit 1; }
for ((s = start + iv; s < steps; s += iv)); do
  mk "$out/policy_$s.safetensors" 0; mk "$out/state_$s.safetensors" 0; echo "training progress: $s/$steps"
  case "${FAKE_MODE:-ok}" in
    crash_once)   once && { echo "Segmentation fault"; exit 139; } ;;
    exit137_once) once && exit 137 ;;
    trunc_once)   once && { s2=$((s + iv)); mk "$out/policy_$s2.safetensors" 0; mk "$out/state_$s2.safetensors" 1; exit 139; } ;;
    oom)          echo "jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate"; exit 1 ;;
    oom_biglog)   echo "RESOURCE_EXHAUSTED: Out of memory"; yes "filler log line for pipe buffer" | head -n 200000; exit 1 ;;
    benign_crash_once) echo "W external/xla: ran out of memory trying to allocate; this is not a failure"; once && exit 1 ;;
    always_crash) echo "boom"; exit 1 ;;
    slow)         sleep 1 ;;
  esac
done
mk "$out/policy_$steps.safetensors" 0; mk "$out/state_$steps.safetensors" 0
echo "completed $steps environment steps; checkpoints: $out"
EOF
# ── 假 docker：容器 = 后台进程 + 状态目录 $CTR/<name>/{pid,log,rc,owner}；run 只记 argv ─
cat > "$SB/bin/docker" <<'EOF'
#!/usr/bin/env bash
name_of() { for a in "$@"; do :; done; echo "$a"; }
alive() { [ -f "$CTR/$1/pid" ] && kill -0 "$(cat "$CTR/$1/pid")" 2>/dev/null; }
killc() { local p; p=$(cat "$CTR/$1/pid"); pkill -TERM -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; }
case "$1" in
  image) echo "sha256:fakeimage${FAKE_IMG_SUFFIX:-}" ;;
  run) printf '%s\n' "$@" > "$CTR/run_argv"; echo "fakecontainerid" ;;
  inspect)
    n=$(name_of "$@"); [ -d "$CTR/$n" ] || exit 1
    case "$3" in
      *Status*) alive "$n" && echo running || echo exited ;;
      *ExitCode*) cat "$CTR/$n/rc" 2>/dev/null || echo 255 ;;
      *Labels*) cat "$CTR/$n/owner" 2>/dev/null ;;
    esac ;;
  logs)
    n=$(name_of "$@"); [ -d "$CTR/$n" ] || exit 1
    if [[ " $* " == *" -f "* ]]; then tail -n +1 --pid="$(cat "$CTR/$n/pid")" -f "$CTR/$n/log"; else cat "$CTR/$n/log"; fi ;;
  wait)
    n=$(name_of "$@"); [ -d "$CTR/$n" ] || { echo "Error: No such container: $n" >&2; exit 1; }
    while [ ! -f "$CTR/$n/rc" ]; do sleep 0.1; done; cat "$CTR/$n/rc" ;;
  stop)
    n=$(name_of "$@"); alive "$n" || exit 1
    killc "$n"; sleep 0.3; echo "${FAKE_STOP_RC:-137}" > "$CTR/$n/rc"; echo "$n" ;;
  rm)
    n=$(name_of "$@"); if alive "$n"; then killc "$n"; fi; rm -rf "${CTR:?}/$n" ;;
  *) echo "fake docker: unsupported $*" >&2; exit 1 ;;
esac
EOF
# ── 假 in-flash-chord：记录命令与环境（只记是否存在），V2D_DETACH=1 时后台起假训练 ─
cat > "$R/scripts/dev/in-flash-chord" <<'EOF'
#!/usr/bin/env bash
if [ "${V2D_DETACH:-0}" != 1 ]; then echo "fake freeze for $*"; exit 0; fi   # image_fingerprint 的依赖 freeze
echo "$*" >> "$CTR/cmds"
echo "run=${WANDB_RUN_ID:-} tags=${WANDB_TAGS:-} key=${WANDB_API_KEY:+set} proxy=${V2D_PROXY:+set}${HTTPS_PROXY:+https} timeout=${WANDB_INIT_TIMEOUT:-} name=${V2D_CONTAINER_NAME:-} gpu=${V2D_GPU:-} detach=${V2D_DETACH:-}" >> "$CTR/envs"
n="$V2D_CONTAINER_NAME"; [ -d "$CTR/$n" ] && { echo "Conflict. The container name \"/$n\" is already in use" >&2; exit 125; }
mkdir -p "$CTR/$n"; echo "${FAKE_OWNER:-$ME}" > "$CTR/$n/owner"
( fake-train "$*" > "$CTR/$n/log" 2>&1; echo $? > "$CTR/$n/rc" ) &
echo $! > "$CTR/$n/pid"; echo "fakecontainerid-$n"
EOF
chmod +x "$SB/bin/"* "$R/scripts/dev/in-flash-chord"

fresh() { rm -rf "$CTR"; mkdir -p "$CTR"; }
runfc() { bash stages/policy/fc_run.sh "$@" > "$SB/out.txt" 2>&1; echo $?; }
rec() { cat "runs/flash_chord/$1/run_record.jsonl"; }
ext_start() {  # 模拟“fc_run 已不在、容器还在（或已退出）”：直接用假 in-flash-chord 起容器
  mkdir -p "runs/flash_chord/$1"
  ( V2D_DETACH=1 V2D_GPU=1 V2D_CONTAINER_NAME="fc-$1" FAKE_MODE="$2" scripts/dev/in-flash-chord \
      "bash /repo/stages/policy/fc_train.sh '$PQ' 2048 409600 /repo/runs/flash_chord/$1" > /dev/null )
}

echo "== T1 正常跑完"
fresh; rc=$(FAKE_MODE=ok runfc t1-ok $PQ 2048 409600); O=runs/flash_chord/t1-ok
check "退出码 0、写 DONE" '[ "$rc" = 0 ] && [ -f $O/DONE ]'
check "最终 policy 在、policy 全留 4 份、state 只留 2 份" '[ -f $O/policy_409600.safetensors ] && [ $(ls $O/policy_*.safetensors | wc -l) = 4 ] && [ $(ls $O/state_*.safetensors | wc -l) = 2 ]'
check "容器已删除" '[ ! -d $CTR/fc-t1-ok ]'
check "W&B：id=<run>-s1、tag、key 已注入、代理没注入、init 超时 180、后台运行、占 GPU、容器名" 'grep -q "run=t1-ok-s1 tags=untagged,$(hostname -s) key=set proxy= timeout=180 name=fc-t1-ok gpu=1 detach=1" $CTR/envs'
check "W&B 覆盖：online、entity、project、group=<run>、name=<run>-s1" 'grep -q "logging.mode=online logging.entity=zhou-rongx-northeastern-university logging.project=v2d-track2 logging.experiment=t1-ok logging.run_name=t1-ok-s1" $CTR/cmds'
check "hydra.run.dir 在输出目录里" 'grep -q "hydra.run.dir=/repo/runs/flash_chord/t1-ok/hydra/s1" $CTR/cmds'
check "首次不带 resume" '! grep -q training.resume $CTR/cmds'
check "段日志完整、实时日志也有" 'grep -q "completed 409600 environment steps" runs/logs/t1-ok.s1.log && grep -q "completed 409600" runs/logs/t1-ok.log'
check "运行记录：start/exit/done，含 sha256、commit、镜像、base 层与依赖指纹" '[ $(rec t1-ok | grep -cE "\"event\": \"(start|exit|done)\"") = 3 ] && rec t1-ok | grep -q "\"source_commit\": \"abc1234-test\"" && rec t1-ok | grep -qE "parquet_sha256\": \"[0-9a-f]{64}" && rec t1-ok | grep -q "sha256:fakeimage" && rec t1-ok | grep -qE "\"deps_freeze_sha\": \"[0-9a-f]{16}\""'
check "GPU 采样有数据" '[ -s runs/logs/t1-ok.gpu ]'
check "W&B key 没出现在日志、记录、命令里" '! grep -rq "$KEY" runs $CTR/cmds'
check "代理密码没出现在任何地方" '! grep -rq secretproxy runs $CTR'

echo "== T2 已完成的再跑：跳过"
rc=$(runfc t1-ok $PQ 2048 409600); check "退出码 0 且提示跳过" '[ "$rc" = 0 ] && grep -q 跳过 $SB/out.txt'

echo "== T3 意外退出后自动续（新 W&B 段）"
fresh; rc=$(FAKE_MODE=crash_once runfc t3-crash $PQ 2048 409600); O=runs/flash_chord/t3-crash
check "退出码 0、DONE" '[ "$rc" = 0 ] && [ -f $O/DONE ]'
check "第 2 段从 state_102400 续" 'sed -n 2p $CTR/cmds | grep -q "training.resume=true training.checkpoint_path=/repo/runs/flash_chord/t3-crash/state_102400.safetensors"'
check "第 2 段 W&B id 是 -s2，group 不变" 'sed -n 2p $CTR/envs | grep -q "run=t3-crash-s2" && sed -n 2p $CTR/cmds | grep -q "logging.experiment=t3-crash logging.run_name=t3-crash-s2"'
check "两段各有独立的段日志" '[ -s runs/logs/t3-crash.s1.log ] && [ -s runs/logs/t3-crash.s2.log ]'

echo "== T4 最新 state 写到一半（policy 完整）：退回上一份"
fresh; rc=$(FAKE_MODE=trunc_once runfc t4-trunc $PQ 2048 409600)
check "续训指向 state_102400，跳过 state 半写的 204800" '[ -f runs/flash_chord/t4-trunc/policy_409600.safetensors ] && sed -n 2p $CTR/cmds | grep -q "state_102400.safetensors"'

echo "== T5 显存不足：不重试；大日志下也能判出；无害的 XLA 提示不算"
fresh; rc=$(FAKE_MODE=oom runfc t5-oom $PQ 2048 409600)
check "退出码 4、只跑 1 次、记录 oom" '[ "$rc" = 4 ] && [ $(wc -l < $CTR/cmds) = 1 ] && rec t5-oom | grep -q "\"reason\": \"oom\""'
fresh; rc=$(FAKE_MODE=oom_biglog runfc t5-big $PQ 2048 409600)
check "OOM 行在最前、后面 20 万行日志：仍判为 oom（不因 SIGPIPE 漏判）" '[ "$rc" = 4 ] && [ $(wc -l < $CTR/cmds) = 1 ]'
fresh; rc=$(FAKE_MODE=benign_crash_once runfc t5-benign $PQ 2048 409600)
check "有\"ran out of memory…not a failure\"的意外退出：当作崩溃重试并完成" '[ "$rc" = 0 ] && [ $(wc -l < $CTR/cmds) = 2 ]'

echo "== T6 重试用完"
fresh; rc=$(V2D_MAX_RETRY=2 FAKE_MODE=always_crash runfc t6-fail $PQ 2048 409600)
check "退出码 2、共 3 次、记录 retries_exhausted" '[ "$rc" = 2 ] && [ $(wc -l < $CTR/cmds) = 3 ] && rec t6-fail | grep -q retries_exhausted'

echo "== T7 fc_stop：不重试，重跑后续上"
fresh; ( FAKE_MODE=slow bash stages/policy/fc_run.sh t7-stop $PQ 2048 409600 > $SB/t7.txt 2>&1; echo $? > $SB/t7.rc ) &
sleep 1.8; bash stages/policy/fc_stop.sh t7-stop > /dev/null; wait
check "退出码 3、只跑 1 次、STOP 已清、记录 stopped" '[ "$(cat $SB/t7.rc)" = 3 ] && [ $(wc -l < $CTR/cmds) = 1 ] && [ ! -f runs/flash_chord/t7-stop/STOP ] && rec t7-stop | grep -q stopped'
rc=$(FAKE_MODE=ok runfc t7-stop $PQ 2048 409600)
check "重跑从最新 checkpoint 续并完成（第 2 段）" '[ "$rc" = 0 ] && sed -n 2p $CTR/cmds | grep -q "training.resume=true" && sed -n 2p $CTR/envs | grep -q "run=t7-stop-s2"'

echo "== T8 运行中被外部 docker stop（没有 STOP）：退出码 5，不重试"
for src in 137 143; do
  fresh; ( FAKE_STOP_RC=$src FAKE_MODE=slow bash stages/policy/fc_run.sh t8-kill-$src $PQ 2048 409600 > /dev/null 2>&1; echo $? > $SB/t8.rc ) &
  sleep 1.8; FAKE_STOP_RC=$src docker stop fc-t8-kill-$src > /dev/null; wait
  check "stop 退出码 $src → fc_run 退出码 5、只跑 1 次、记录 killed" '[ "$(cat $SB/t8.rc)" = 5 ] && [ $(wc -l < $CTR/cmds) = 1 ] && rec t8-kill-'$src' | grep -q "\"reason\": \"killed\""'
done

echo "== T9 接管：fc_run 死了、容器还在跑，再运行 fc_run 就接上"
fresh; ext_start t9-adopt slow; rc=$(FAKE_MODE=slow runfc t9-adopt $PQ 2048 409600)
check "没有另起容器、记录 adopt、段日志完整、完成" '[ "$rc" = 0 ] && [ $(wc -l < $CTR/cmds) = 1 ] && rec t9-adopt | grep -q "\"event\": \"adopt\"" && grep -q "training progress: 102400" runs/logs/t9-adopt.s0.log && [ -f runs/flash_chord/t9-adopt/DONE ]'

echo "== T10 回收：容器已崩溃退出但没被删 → 先回收再续"
fresh; ext_start t10-reap crash_once; sleep 1; rc=$(FAKE_MODE=ok runfc t10-reap $PQ 2048 409600)
check "记录 reap（139），随后从 state_102400 续并完成" '[ "$rc" = 0 ] && rec t10-reap | grep -q "\"event\": \"reap\"" && rec t10-reap | grep -q "\"rc\": \"139\"" && sed -n 2p $CTR/cmds | grep -q state_102400'

echo "== T11 回收到 137（例如机器重启）：没有 STOP → 自动续，不当作外部停止"
fresh; ext_start t11-reboot exit137_once; sleep 1; rc=$(FAKE_MODE=ok runfc t11-reboot $PQ 2048 409600)
check "reap 137 后直接开新段并完成" '[ "$rc" = 0 ] && rec t11-reboot | grep -q "\"rc\": \"137\"" && sed -n 2p $CTR/cmds | grep -q "training.resume=true"'

echo "== T12 GPU 上有别人的进程：拒绝启动"
fresh; rc=$(FAKE_GPU_BUSY=1 runfc t12-busy $PQ 2048 409600)
check "退出码 6、没起容器" '[ "$rc" = 6 ] && [ ! -s $CTR/cmds ]'

echo "== T13 续训一致性（与本 lineage 起点比）"
fresh; rc=$(FAKE_MODE=crash_once V2D_MAX_RETRY=0 runfc t13-cons $PQ 2048 409600 training.replay.capacity=1000000)
rc=$(FAKE_MODE=ok runfc t13-cons $PQ 2048 409600 training.replay.capacity=8000000)
check "覆盖参数不同 → 拒绝" '[ "$rc" = 2 ] && grep -q "续训被拒：overrides" $SB/out.txt'
rc=$(FAKE_IMG_SUFFIX=v2 FAKE_MODE=ok runfc t13-cons $PQ 2048 409600 training.replay.capacity=1000000)
check "镜像不同 → 拒绝" '[ "$rc" = 2 ] && grep -q "续训被拒：image" $SB/out.txt'
rc=$(V2D_FORCE_RESUME=1 FAKE_IMG_SUFFIX=v2 FAKE_MODE=ok runfc t13-cons $PQ 2048 409600 training.replay.capacity=1000000)
check "V2D_FORCE_RESUME=1 可强制续" '[ "$rc" = 0 ]'
fresh; rc=$(FAKE_MODE=crash_once V2D_MAX_RETRY=0 runfc t13-empty $PQ 2048 409600)
rc=$(FAKE_MODE=ok runfc t13-empty $PQ 2048 409600 training.replay.capacity=1000000)
check "起点没有覆盖参数、续训加了 → 拒绝（原因是 overrides）" '[ "$rc" = 2 ] && grep -q "续训被拒：overrides" $SB/out.txt'
rc=$(runfc t13-empty $PQ 2048 409600)
check "锁在上一次 fc_run 退出后立即释放" '! grep -q 占用 $SB/out.txt'
fresh; rc=$(FAKE_MODE=crash_before_ckpt V2D_MAX_RETRY=0 runfc t13-lin $PQ 2048 409600 training.replay.capacity=1000000)
rc=$(FAKE_MODE=crash_once V2D_MAX_RETRY=0 runfc t13-lin $PQ 2048 409600 training.replay.capacity=8000000)
rc=$(FAKE_MODE=ok runfc t13-lin $PQ 2048 409600 training.replay.capacity=8000000)
check "第一次在 checkpoint 前就挂、改参数重来 → 新 lineage 的续训不被误拒" '[ "$rc" = 0 ]'

echo "== T14 checkpoint 全部不完整：拒绝从 0 覆盖"
fresh; D=runs/flash_chord/t14-bad; mkdir -p $D; echo junk > $D/state_102400.safetensors; echo junk > $D/policy_102400.safetensors
rc=$(FAKE_MODE=ok runfc t14-bad $PQ 2048 409600)
check "退出码 2、没起容器" '[ "$rc" = 2 ] && [ ! -s $CTR/cmds ] && grep -q "没有一份完整" $SB/out.txt'
rc=$(V2D_FORCE_RESTART=1 FAKE_MODE=ok runfc t14-bad $PQ 2048 409600)
check "V2D_FORCE_RESTART=1 → 从 0 开始并完成" '[ "$rc" = 0 ] && ! grep -q training.resume $CTR/cmds'

echo "== T15 STOP：退避期间可取消；启动时残留的会被清掉"
fresh; ( V2D_RETRY_BACKOFF_S=3 V2D_MAX_RETRY=2 FAKE_MODE=always_crash bash stages/policy/fc_run.sh t15-backoff $PQ 2048 409600 > $SB/t15.txt 2>&1; echo $? > $SB/t15.rc ) &
sleep 1.5; touch runs/flash_chord/t15-backoff/STOP; wait
check "退避中放 STOP → 退出码 3、不再开新段" '[ "$(cat $SB/t15.rc)" = 3 ] && [ $(wc -l < $CTR/cmds) = 1 ]'
fresh; mkdir -p runs/flash_chord/t15-stale; touch runs/flash_chord/t15-stale/STOP
rc=$(FAKE_MODE=ok runfc t15-stale $PQ 2048 409600)
check "启动时残留的 STOP（容器早已不在）→ 清掉并正常跑完" '[ "$rc" = 0 ] && rec t15-stale | grep -q stale_stop_cleared'

echo "== T16 同名容器不是自己起的：不接管也不删除"
fresh; FAKE_OWNER=someoneelse ext_start t16-other slow; rc=$(runfc t16-other $PQ 2048 409600)
check "退出码 2、对方容器还在" '[ "$rc" = 2 ] && grep -q "不是" $SB/out.txt && [ -d $CTR/fc-t16-other ]'; docker rm -f fc-t16-other

echo "== T17 数字排序与剪枝"
D=runs/flash_chord/t17; mkdir -p $D
for s in 102400 90003456 340013056; do python3 - $D/policy_$s.safetensors $D/state_$s.safetensors <<'PY'
import json, struct, sys
h = json.dumps({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}).encode()
for p in sys.argv[1:]: open(p, "wb").write(struct.pack("<Q", len(h)) + h + b"\0" * 16)
PY
done
check "ckpt_latest 选 340013056" '[ "$(python3 stages/policy/ckpt_latest.py $D)" = 340013056 ]'
( OUT=$D KEEP=2; source <(sed -n "/^prune_states()/,/^}/p" stages/policy/fc_run.sh); prune_states )
check "剪枝后只剩 90003456 与 340013056" '[ "$(ls $D/state_*.safetensors | xargs -n1 basename | sort | tr "\n" " ")" = "state_340013056.safetensors state_90003456.safetensors " ]'
check "不存在的目录 → 无输出、退出码 0" '[ -z "$(python3 stages/policy/ckpt_latest.py $SB/none)" ]'

echo "== T18 参数校验与互斥"
check "run 名非法" '[ "$(runfc Bad_Name $PQ 2048 409600)" = 2 ]'
check "步数不整除" '[ "$(runfc t18-div $PQ 2048 409601)" = 2 ]'
check "parquet 不存在" '[ "$(runfc t18-nopq /repo/runs/motion/none.parquet 2048 409600)" = 2 ]'
check "V2D_KEEP_STATES=0 被拒" '[ "$(V2D_KEEP_STATES=0 runfc t18-keep $PQ 2048 409600)" = 2 ]'
mkdir -p runs/flash_chord/t18-lock; exec 8>runs/flash_chord/t18-lock/.lock; flock -n 8
check "另一个 fc_run 持锁时拒绝" '[ "$(runfc t18-lock $PQ 2048 409600)" = 2 ] && grep -q 占用 $SB/out.txt'
exec 8>&-
check "V2D_WANDB=1 但 .env 没有 key → 拒绝" 'mv .env .env.bak; rc=$(runfc t18-nokey $PQ 2048 409600); mv .env.bak .env; [ "$rc" = 2 ]'
check "V2D_WANDB_PROXY=1 → 容器拿到 HTTPS_PROXY、运行记录 wandb_proxy=on、代理密码不落盘" 'rc=$(V2D_WANDB_PROXY=1 runfc t19-proxy $PQ 2048 409600); [ "$rc" = 0 ] && grep "run=t19-proxy-s1" $CTR/envs | grep -q "proxy=https " && grep "\"event\": \"start\"" runs/flash_chord/t19-proxy/run_record.jsonl | grep -q "\"wandb_proxy\": \"on\"" && ! grep -rq secretproxy runs $CTR'
check "V2D_WANDB_PROXY=1 但 .env 没有 V2D_PROXY → 拒绝" 'cp .env .env.bak; sed -i "/^V2D_PROXY=/d" .env; rc=$(V2D_WANDB_PROXY=1 runfc t20-noproxy $PQ 2048 409600); mv .env.bak .env; [ "$rc" = 2 ]'

echo "== T19 队列：解析、逐行环境变量、失败即停、重跑跳过已完成"
fresh
cat > q.txt <<EOF
# 注释行
V2D_WANDB_TAG=grpA q-one $PQ 2048 409600 training.replay.capacity=1000000   # 行尾注释
V2D_WANDB=0 q-two $PQ 2048 409600
q-three /repo/runs/motion/none.parquet 2048 409600
q-four $PQ 2048 409600
EOF
FAKE_MODE=ok bash stages/policy/fc_queue.sh q.txt > $SB/q1.txt 2>&1; qrc=$?
check "第 1 项带 tag 与容量覆盖" 'sed -n 1p $CTR/envs | grep -q "tags=grpA," && sed -n 1p $CTR/cmds | grep -q "training.replay.capacity=1000000"'
check "第 2 项 V2D_WANDB=0 → logging.mode=disabled、不注入 key" 'sed -n 2p $CTR/cmds | grep -q "logging.mode=disabled" && sed -n 2p $CTR/envs | grep -q "key= "'
check "第 3 项失败后队列停下，第 4 项没跑" '[ "$qrc" != 0 ] && [ ! -d runs/flash_chord/q-four ]'
sed -i '/none.parquet/d' q.txt; FAKE_MODE=ok bash stages/policy/fc_queue.sh q.txt > $SB/q2.txt 2>&1
check "修好后重跑：前两项跳过、第 4 项完成" '[ $(grep -c 跳过 $SB/q2.txt) = 2 ] && [ -f runs/flash_chord/q-four/DONE ]'

echo "== T20 fc_train.sh 的默认值与 parquet 引号"
ft() { ( python() { echo "$@"; }; cd() { :; }; source <(sed 's/^exec //' stages/policy/fc_train.sh) "$@" ); }
check "没给 logging.mode → 补 disabled；没给 hydra.run.dir → 放进输出目录" 'ft /p 2048 409600 /o | grep -q "logging.mode=disabled hydra.run.dir=/o/hydra/"'
check "给了就不重复补" 'o=$(ft /p 2048 409600 /o logging.mode=online hydra.run.dir=/x); [ $(echo "$o" | grep -o "logging.mode=" | wc -l) = 1 ] && [ $(echo "$o" | grep -o "hydra.run.dir=" | wc -l) = 1 ]'
check "parquet 整体加单引号给 Hydra" 'ft "/a/sequence_id=x/robot_name=g1/d.parquet" 2048 409600 /o | grep -q "task.parquet='"'"'/a/sequence_id=x/robot_name=g1/d.parquet'"'"'"'

echo "== T21 真实 in-flash-chord 拼出的 docker 参数（假 docker 只记 argv）"
mkdir -p "$SB/ifcroot/scripts/dev" "$SB/ifcroot/third_party/video_to_data/robotic_grounding/flash_chord"
cp "$SRC/scripts/dev/in-flash-chord" "$SB/ifcroot/scripts/dev/in-flash-chord"; touch "$SB/ifcroot/.env"
( export WANDB_API_KEY="$KEY" WANDB_RUN_ID=x-s1 V2D_DETACH=1 V2D_GPU=1 V2D_CONTAINER_NAME=fc-x
  bash "$SB/ifcroot/scripts/dev/in-flash-chord" "bash /repo/stages/policy/fc_train.sh '/repo/a/sequence_id=q/robot_name=g1/d.parquet' 2048 409600 /repo/o" > /dev/null )
A="$CTR/run_argv"
check "后台运行、不 --rm、容器名与属主标签" 'grep -qx -- "-d" $A && ! grep -qx -- "--rm" $A && grep -qx "fc-x" $A && grep -qx "v2d.owner=$ME" $A'
check "key 只以变量名出现（-e WANDB_API_KEY），值不在 argv" 'grep -qx WANDB_API_KEY $A && ! grep -q "$KEY" $A'
check "容器里 .env 被遮住、Warp 缓存挂在 /warp-cache 并设 WARP_CACHE_PATH、占 GPU" 'grep -qx "/dev/null:/repo/.env:ro" $A && grep -q ":/warp-cache$" $A && grep -qx "WARP_CACHE_PATH=/warp-cache" $A && grep -qx -- "--gpus" $A'
check "没有任何挂载点落在 /tmp/.cache 下（否则 W&B 缓存目录不可写）" '! grep -q ":/tmp/.cache" $A'
check "最后一个参数是完整命令串（parquet 含 = 仍是一个单元）" '[ "$(tail -1 $A)" = "bash /repo/stages/policy/fc_train.sh '"'"'/repo/a/sequence_id=q/robot_name=g1/d.parquet'"'"' 2048 409600 /repo/o" ]'

echo; echo "结果：PASS $PASS  FAIL $FAIL"
[ "$FAIL" = 0 ]
