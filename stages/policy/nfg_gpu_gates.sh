#!/usr/bin/env bash
# 非有限防护的 GPU 闸（计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 第 5 版的 B.4）。训练机上、仓库根执行，耗时约 40 分钟，放 tmux 或 setsid 里跑。
# 依次：t1、t1b（未打防护的 v040_p185）、t2a、t2b、t2c、t2d、t2n（打了防护的 v040_p185_nfg）、T6（打了防护，端到端注入）、
# T4 基准（上游测试，未打防护）与 T4（打了防护）。探针在前，出错能早发现；各闸互不依赖。每一步经 scripts/dev/record 记到 results/ep7/nfg_<步>_<机器>.txt；一步不过照样往下跑（判据各自记），
# 最后打印每一步的结果与 T4 两份的对照。每一步之前核 GPU 上没有别的进程，有就停整条链。日志 runs/logs/nfg_gates_<机器>.log，最后一行 NFG_GATES_DONE。
# 用法：stages/policy/nfg_gpu_gates.sh <机器名> <fc_src 目录>（例：ml40 /lab_data/rxzhou/v2d/fc_src）
set -uo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：nfg_gpu_gates.sh <机器名> <fc_src 目录>}; SRC=${2:?fc_src 目录}
PLAIN=$SRC/v040_p185; NFG=$SRC/v040_p185_nfg
LOG=runs/logs/nfg_gates_$M.log; mkdir -p runs/logs
say() { echo "$(date '+%F %T %Z') $*" | tee -a "$LOG"; }
die() { say "$*"; say NFG_GATES_STOPPED; exit 1; }
gpu_check() { local o; o="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"; [ -z "$o" ] || die "GPU 上有别的进程（$o），停下"; }
for d in "$PLAIN" "$NFG"; do [ -f "$d/SOURCE_V2D.txt" ] || die "缺副本 $d"; done
[ -e runs/nfg_probe ] && die "runs/nfg_probe 已存在（上一次的探针结果），先挪走再跑"
[ -e "runs/nfg_t6/t6-$M" ] && die "runs/nfg_t6/t6-$M 已存在，先挪走再跑"
say "start：未打防护 $PLAIN；打了防护 $NFG；来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
declare -A RC
step() {  # step <名字> <副本> <容器里的命令> [tb]：给 tb 时输出里的 Traceback 不算失败（pytest 的失败报告里会有）
  local name=$1 src=$2 cmd=$3 tb=0
  [ "${4:-}" = tb ] && tb=1
  gpu_check
  say "== $name（$(basename "$src")）"
  V2D_RECORD_TRACEBACK_OK=$tb V2D_GPU=1 V2D_FC_SRC=$src XLA_PYTHON_CLIENT_PREALLOCATE=false \
    scripts/dev/record "ep7/nfg_${name}_$M" scripts/dev/in-flash-chord "$cmd" >> "$LOG" 2>&1
  RC[$name]=$?
  say "$name 退出码 ${RC[$name]}：$(tail -1 "results/ep7/nfg_${name}_$M.txt" 2>/dev/null || tail -1 "results/ep7/nfg_${name}_$M.txt.failed" 2>/dev/null)"
}
T4="cd /workspace && python -m pytest -q -rfEs -p no:cacheprovider tests/training/flash_sac tests/lifecycle/test_termination.py tests/envs/test_rl.py tests/envs/test_base.py"
for g in t1 t1b; do step "probe_$g" "$PLAIN" "python /repo/stages/policy/nfg_env_probe.py $g"; done
for g in t2a t2b t2c t2d t2n; do step "probe_$g" "$NFG" "python /repo/stages/policy/nfg_env_probe.py $g"; done
step t6 "$NFG" "python /repo/stages/policy/nfg_train_inject.py /repo/runs/nfg_t6/t6-$M"
step t4base "$PLAIN" "$T4" tb
step t4 "$NFG" "$T4" tb
say "== 汇总"
for k in probe_t1 probe_t1b probe_t2a probe_t2b probe_t2c probe_t2d probe_t2n t6 t4base t4; do say "$k：退出码 ${RC[$k]}"; done
for k in t4base t4; do
  f="results/ep7/nfg_${k}_$M.txt"; [ -f "$f" ] || f="$f.failed"
  say "$k 的 pytest 结果：$(grep -E '^(FAILED|ERROR) ' "$f" | sort | tr '\n' ';')$(grep -E '[0-9]+ (passed|failed|skipped|error)' "$f" | tail -1)"
done
say NFG_GATES_DONE
