#!/usr/bin/env bash
# 起变体 A 链一段的三个 tmux（训练机上、仓库根执行）。arm_r3x_chain.sh 的参数化版本，核对项相同。段名以清单名开头：
#   t2x*  Tier 2 铺开，ml40（计划 agent-doc/plan-t2-scaleout-2026-10-09.md；t2xa 15 条、t2xb 9 条，续跑与补跑另起一段）
#   t3p*  Tier 3 试跑，mf4090（计划 agent-doc/plan-t3-pilot-2026-10-09.md；t3p2 是第二轮，计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md）
#   t1x*  Tier 1 补候选，mf4090，同 t3p* 的显存上限、磁盘要求与冒烟后删 state（计划 agent-doc/plan-next-training-2026-10-12.md 第 3 节 A）
#   t1y*  Tier 1 补候选，ml40，同 t2x*（同计划第 3 节 C）
#   t1c*  Tier 1 接触标记提前，ml40，同 t1y*；参考目录是 runs/motion/tier1_c10（计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md）
#   t1o*、t2o*  r3o 试点（计划 agent-doc/plan-r3o-2026-10-15.md）：t1o* 同 t1x*（mf4090），t2o* 同 t2x*（ml40）；训练挂 v040_p185_nfg_oo、按
#         third_party/patches/flash_chord-p185-nfg-oo.sha256 核对；冒烟与正式队列每行都要带 V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4、run 名带 -r3o-；
#         其余段一行都不许带 V2D_OBJ_ORI_ 或 -r3o-（训练副本 v040_p185_nfg 不认这两个变量，带了也是白带）
#   t3pr3o*、t3qr3o*  Tier 3 关朝向终止 + 物体朝向项（计划 agent-doc/plan-t3oo-2026-10-28.md）：同 t3p* / t3q* 的机器与限制，训练副本同 t1o*；
#         σ 每行自带、run 名带 -sgDD-（σ = D.D），核对规则在 stages/policy/r3o_lines.awk（契约 POLICY-I010）
# 队列与导出清单由 stages/policy/make_var_queue.py 按段生成。起之前确认（任何一条不过就停，什么也不起）：
#   - GPU 型号对得上这份清单的机器；来历干净；三段日志、导出结果表、导出门的日志与标记、长链标记、停链标记、同名 tmux 会话都不存在；
#     GPU 上没有进程；
#     仓库所在分区的可用空间不低于这份清单的要求
#   - 训练副本（v040_p185_nfg，t1o*、t2o* 是 v040_p185_nfg_oo）与 fc_8c778f9 都在，训练副本按它的 sha256 清单核对通过
#   - 期望表至少一条 run；导出清单的行与期望表的 run 一一对应、顺序相同、档位相同
#   - 导出要读的 runs/motion_src/<目录>/<序列>/v2d_meta.json 都在，里面的档位与 episode 和导出清单一致（fc_export.sh 从这里取 episode 号）；
#     目录取自正式队列行的 parquet（runs/motion/<目录>/…），是 tier<档>，或 Tier 3 校正版的 tier3_<名>（与 fc_export.sh 相同）
#   - 正式 run 的目录里没有训练产物（目录不存在，或只有 fc_config_check.sh 经 fc_run.sh 打印模式建的空 .lock；不续训、不覆盖）；
#     冒烟的目录要么也没有产物，要么已有 DONE（续跑、补跑时冒烟已经过了，训练链会重判），有产物没 DONE 的就停
#   - 全部配置的 runs/cfgcheck/<run>.yaml 都在（fc_config_check.sh 先生成），precheck_expect.sh 全过；
#     配方断言：objective.object_position_std 与 termination.object_orientation_threshold（顶层与 env 下各一份）都是变体 A 的 0.05、0.6；r3p 段的 σ 是 0.03；
#     队列里每一行都写了同一个朝向终止值（如对照臂 oc15 的 1.5）就断言那个值
# 起三个会话（<段名> 如 t2xa）：
#   v2d-<段名>-train   训练链（环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>/v040_p185_nfg；t3p 另加冒烟判过后删 state、可用不到 1200 MB 就停）；
#                      V2D_GPU_WAIT_S=21600：GPU 被别人占着时每分钟看一次，最多等 6 小时（审查 M3）；要让卡时建 runs/.chain-stop-<段名>_train_<机器>
#   v2d-<段名>-eval    评测链：等训练链结束，用 fc_8c778f9 只评最终 checkpoint（4096 世界）；同样最多等卡 6 小时
#   v2d-<段名>-export  导出门（stages/evaluate/export_after_eval.sh）：等评测链结束，最后一行是“成功 N、失败 0”才跑导出链，
#                      用 fc_8c778f9 导出 1 个世界的轨迹，V2D_FALLBACK_MAX=0（只用最终 checkpoint，不回退）；评测链没有干净结束就一条都不导
# 用法：stages/policy/arm_var_chain.sh <段名>      V2D_ARM_DRYRUN=1：核完只打印三条命令，不起会话
set -euo pipefail
cd "$(dirname "$0")/../.."
N=${1:?用法：arm_var_chain.sh <段名>}
[[ $N =~ ^[a-z0-9_]+$ ]] || { echo "段名非法：$N" >&2; exit 2; }
case "$N" in
  t2x*) M=ml40; T=2; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;
  t3pr3o*) M=mf4090; T=3; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; COPY=v040_p185_nfg_oo ;;   # Tier 3 关朝向终止 + 物体朝向项（计划 agent-doc/plan-t3oo-2026-10-28.md）；要排在 t3p* 前面（t3po* 会吃掉 t3pocoff）
  t3qr3o*) M=ml40; T=3; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA=""; COPY=v040_p185_nfg_oo ;;   # 同上，ml40；要排在 t3q* 前面
  t3p*) M=mf4090; T=3; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;
  t1zp*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; COPY=v040_p185_nfg_oo ;;   # Tier 1 冲第一 A2，r3o（计划 agent-doc/plan-t1push-2026-10-29.md）
  t1zs*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; COPY=v040_p185_nfg_oo ;;   # r3o 铺开（计划 agent-doc/plan-t12night-2026-10-30.md），同 t1zp
  t1zt*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; COPY=v040_p185_nfg_oo ;;   # r3o 铺到六集（计划 agent-doc/plan-t1r3o-2026-10-31.md），同 t1zs
  t1zx*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; COPY=v040_p185_nfg_oo ;;   # 清单 run 的行加 r3o（make_clone_queue.py，计划 agent-doc/plan-t1r3o-2026-10-31.md 第 3 版）
  t1zw*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA=""; COPY=v040_p185_nfg_oo ;;   # r3o 用 1000 万（计划 agent-doc/plan-t1r3o-2026-10-31.md 第 2 版）
  t1hr*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA=""; COPY=v040_p185_nfg_oo; NOVAR=1 ;;   # 跟丢的集，基础配方加物体朝向项（计划 agent-doc/plan-t1hard-2026-10-31.md）
  t1hp2*|t1hp3*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; NOVAR=1 ;;   # ep1 手掌位置终止 0.05、0.08，400 万（计划 agent-doc/plan-t1ep1-2026-11-02.md）
  t1hp*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA=""; NOVAR=1 ;;   # 跟丢的集，手掌位置终止 0.08（同计划）
  t1hq*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA=""; NOVAR=1 ;;   # MPPE 榜，手掌位置终止 0.08（计划 agent-doc/plan-t1mppe-2026-11-01.md）
  t1hg*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA=""; NOVAR=1 ;;   # MPPE 榜，抓握闭合项权重 20，1000 万（计划 agent-doc/plan-t1night-2026-11-01.md）
  t1hf*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; NOVAR=1 ;;   # 跟丢的集，抓握闭合项权重 20（同计划）
  t1zj*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # 400 万换 1000 万（同计划），同 t1za
  t1za*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # Tier 1 冲第一 A1
  t1zh*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # 手掌尺度 0.05（计划 agent-doc/plan-t1palm-2026-10-30.md）
  t1zq*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;   # 同上，mf4090
  t1zi*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000
        EXTRA="" ;;   # 手掌尺度 0.1（计划 agent-doc/plan-t1palm-2026-10-30.md 第 3 版）
  t1zr*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;   # 同上，mf4090
  t1x*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;
  t1y*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;
  t3q*) M=ml40; T=3; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # D 在 ml40 上（计划 agent-doc/plan-d-t3hm-train-2026-10-12.md）
  t1o*) M=mf4090; T=1; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; COPY=v040_p185_nfg_oo ;;   # r3o（计划 agent-doc/plan-r3o-2026-10-15.md）
  t2o*) M=ml40; T=2; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA=""; COPY=v040_p185_nfg_oo ;;
  t1c*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # 接触标记提前（计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md）
  t2zo*) M=mf4090; T=2; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200"; COPY=v040_p185_nfg_oo ;;   # Tier 2 AUC 榜 r3o，400 万（计划 agent-doc/plan-t2auc2-2026-11-02.md），同 t1zs
  t2l*) M=l40n; T=2; GPU="NVIDIA L40"; SRC="$HOME/v2d/fc_src"; PEAK=44000; NEED_MB=20000; EXTRA=""; COPY=v040_p185_nfg_oo ;;   # 第三台 l40n（L40，46 GB，同 ml40 显存），Tier 2 AUC 榜 r3o，1000 万（计划 agent-doc/plan-t2auc3-2026-11-02.md）；一条 run 约 360 MB，盘 197 GB
  t2m*) M=mf4090; T=2; GPU="NVIDIA GeForce RTX 4090"; SRC="$HOME/v2d/fc_src"; PEAK=24564; NEED_MB=2500
        EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;   # Tier 2 接触标记提前（计划 agent-doc/plan-c10-extend-2026-10-16.md 路 B）
  t2f*) M=ml40; T=2; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # 参考合手（计划 agent-doc/plan-t2fc-2026-10-20.md）
  t2h*) M=ml40; T=2; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # 保底参考（计划 agent-doc/plan-t12lift-2026-10-20.md）
  t1h*) M=ml40; T=1; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # 同上，Tier 1
  t2t*) M=ml40; T=2; GPU="NVIDIA L40S"; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; NEED_MB=200000; EXTRA="" ;;   # 放宽物体位置终止（计划 agent-doc/plan-t2term-2026-10-20.md）
  *) echo "段名要以 t2x、t3pr3o、t3qr3o、t3p、t3q、t1zp、t1zs、t1zt、t1zw、t1zx、t1hr、t1hp2、t1hp3、t1hp、t1hg、t1hf、t1hq、t1zj、t1za、t1zh、t1zq、t1zi、t1zr、t1x、t1y、t1o、t2o、t1c、t2m、t2zo、t2l、t2f、t2h、t1h 或 t2t 开头" >&2; exit 2 ;;
esac
# 10/30 17:0x 起两台训练机直连 api.wandb.ai 返回 403（出口 IP 在 CN），经 V2D_PROXY（出口 SG）可达：所有段的 W&B 都走代理（fc_run.sh V2D_WANDB_PROXY）
EXTRA="${EXTRA:+$EXTRA }V2D_WANDB_PROXY=1"
COPY=${COPY:-v040_p185_nfg}; MAN=third_party/patches/flash_chord-p185-nfg.sha256
[ "$COPY" = v040_p185_nfg ] || MAN=third_party/patches/flash_chord-p185-nfg-oo.sha256
WAIT="V2D_GPU_WAIT_S=21600"
Q=stages/policy/queues/${M}_$N; LIST=stages/evaluate/exports/cand_t${T}_${N}_$M.tsv
LT=${N}_train_$M; LE=${N}_eval_$M; LX=cand_t${T}_${N}_$M
SESS=("v2d-$N-train" "v2d-$N-eval" "v2d-$N-export")
stop() { echo "arm_var_chain: $*，停" >&2; exit 1; }

[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" = "$GPU" ] || stop "这台不是 ${M}（GPU 不是 ${GPU}）"
src0="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$src0" in *-dirty*|*-syncing*|unknown) stop "来历 $src0 不干净" ;; esac
for f in "$Q.expect.tsv" "${Q}_smoke.txt" "$Q.txt" "$LIST"; do [ -f "$f" ] || stop "缺 $f"; done
for L in "$LT" "$LE" "$LX"; do
  [ ! -e "runs/logs/$L.log" ] || stop "已存在 runs/logs/$L.log"
  [ ! -e "runs/.chain-active-$L" ] || stop "已存在 runs/.chain-active-$L"
  [ ! -e "runs/.chain-stop-$L" ] || stop "已存在停链标记 runs/.chain-stop-$L"
done
[ ! -e "runs/exports/$LX.tsv" ] || stop "已存在 runs/exports/$LX.tsv"
[ ! -e "runs/logs/${LX}_gate.log" ] && [ ! -e "runs/.chain-active-${LX}-gate" ] || stop "已存在导出门的日志或标记"
for s in "${SESS[@]}"; do
  if tmux has-session -t "$s" 2>/dev/null; then stop "tmux 会话 $s 已存在"; fi
done
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
[ -z "$others" ] || stop "GPU 上有别的进程（${others//$'\n'/,}）"
free_mb="$(df -Pk . | awk 'NR == 2 {print int($4 / 1024)}')"
[ "$free_mb" -ge "$NEED_MB" ] || stop "可用空间只有 ${free_mb} MB（要求 ${NEED_MB} MB）"
[ -f "$SRC/$COPY/SOURCE_V2D.txt" ] && [ -f "$SRC/fc_8c778f9/SOURCE_V2D.txt" ] || stop "$SRC 下缺副本（$COPY 或 fc_8c778f9）"
python3 stages/policy/fc_src_check.py "$SRC/$COPY" "$MAN" | tail -1 | grep -q '^FC_SRC_OK' \
  || stop "$SRC/$COPY 按清单 $MAN 核对不过"
# r3o 行的核对在 stages/policy/r3o_lines.awk（权重 1.0；σ 每行自带，run 名带 -sgDD- 时 σ 必须等于 D.D，不带时 0.4；两个变量各正好一次；普通副本的段一行都不许带）
r3o_out="$({ grep -vhE '^[[:space:]]*(#|$)' "${Q}_smoke.txt" "$Q.txt" || true; } | awk -v oo="$([ "$COPY" = v040_p185_nfg_oo ] && echo 1 || echo 0)" -f stages/policy/r3o_lines.awk)"
r3o_bad="$(tail -1 <<< "$r3o_out")"
[ "$r3o_bad" = 0 ] || { grep '^BAD:' <<< "$r3o_out" >&2; stop "冒烟或正式队列里有 ${r3o_bad} 行的 r3o 环境变量、σ 或 run 名与训练副本 $COPY 不符"; }

runs=(); cfgs=(); seqs=()
while IFS=$'\t' read -r run smoke seq _; do runs+=("$run"); cfgs+=("$run" "$smoke"); seqs+=("$seq"); done < <(tail -n +2 "$Q.expect.tsv" | grep -v '^[[:space:]]*$')
[ "${#runs[@]}" -ge 1 ] || stop "期望表里没有 run"
mapfile -t ltier < <(grep -vE '^(#|[[:space:]]*$)' "$LIST" | cut -f1)
mapfile -t leps < <(grep -vE '^(#|[[:space:]]*$)' "$LIST" | cut -f2)
mapfile -t listed < <(grep -vE '^(#|[[:space:]]*$)' "$LIST" | cut -f3)
[ "${listed[*]}" = "${runs[*]}" ] || stop "导出清单的 run 与期望表不一致：${listed[*]}"
for i in "${!runs[@]}"; do
  [ "${ltier[$i]}" = "$T" ] || stop "导出清单第 $((i + 1)) 行的档位是 ${ltier[$i]}，不是 $T"
  pq="$(grep -v '^[[:space:]]*#' "$Q.txt" | awk -v r="${runs[$i]}" '{for (k = 1; k < NF; k++) if ($k == r) {print $(k + 1); exit}}' || true)"
  [[ $pq =~ ^/repo/runs/motion/(tier[123](_[a-z0-9]+)?)/whole_body/soma/sequence_id=([^/]+)/robot_name=g1/data\.parquet$ ]] \
    && [ "${BASH_REMATCH[3]}" = "${seqs[$i]}" ] || stop "正式队列里 ${runs[$i]} 的 parquet 读不出，或序列与期望表不符：${pq:-无}"
  m="runs/motion_src/${BASH_REMATCH[1]}/${seqs[$i]}/v2d_meta.json"
  [ -f "$m" ] || stop "缺 ${m}（导出要用）"
  python3 -c 'import json, sys; j = json.load(open(sys.argv[1])); sys.exit(0 if (str(j["tier"]), str(j["episode"])) == (sys.argv[2], sys.argv[3]) else 1)' \
    "$m" "$T" "${leps[$i]}" || stop "$m 的档位或 episode 与导出清单（Tier $T ep${leps[$i]}）不一致"
done
for r in "${cfgs[@]}"; do
  # 目录还不存在时 find 退出码非零，set -e 下会不打印就退出（10/16 凌晨 t1ob、t2ob 就停在这里）；不存在就当没有产物，往下核 cfgcheck
  left="$(find "runs/flash_chord/$r" -mindepth 1 ! -name .lock -print -quit 2>/dev/null || true)"
  if [ -n "$left" ]; then
    case "$r" in smoke-*) [ -f "runs/flash_chord/$r/DONE" ] || stop "runs/flash_chord/$r 里有产物但没有 DONE（${left}）" ;;
                 *) stop "runs/flash_chord/$r 里已有训练产物（${left}）" ;; esac
  fi
  [ -f "runs/cfgcheck/$r.yaml" ] || stop "缺 runs/cfgcheck/$r.yaml（先跑 fc_config_check.sh）"
done
stages/policy/precheck_expect.sh "$Q.expect.tsv" "${Q}_smoke.txt" "$Q.txt" | tail -1 | grep -q '^PRECHECK_OK' || stop "precheck_expect.sh 不过"
# σ 跟着段走（计划 agent-doc/plan-precision-2026-10-18.md）：每一行都是 r3p（run 名带 -r3p-、覆盖里 object_position_std=0.03）就断言 0.03；
# 一行都不是就断言变体 A 的 0.05；混在一起就停
nq="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c . || true)"
np="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c -- '-r3p-' || true)"
ns="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c 'objective.object_position_std=0.03' || true)"
if [ "$np" = 0 ] && [ "$ns" = 0 ]; then SIG=0.05
elif [ "$np" = "$nq" ] && [ "$ns" = "$nq" ]; then SIG=0.03
else stop "队列里 r3p 的行与变体 A 的行混在一起（${nq} 行里 -r3p- ${np} 行、σ 0.03 ${ns} 行）"; fi
VAR="objective.object_position_std=$SIG env.objective.object_position_std=$SIG termination.object_orientation_threshold=0.6 env.termination.object_orientation_threshold=0.6"
# 基础配方的段（NOVAR=1，计划 agent-doc/plan-t1hard-2026-10-31.md）：不加变体 A，断言基础配方的 0.2 与 1.5；队列里出现变体 A 的覆盖就停
if [ "${NOVAR:-0}" = 1 ]; then
  ! grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -qE 'objective\.object_position_std=|termination\.object_orientation_threshold=' || stop "基础配方的段里有变体 A 的覆盖"
  VAR="objective.object_position_std=0.2 env.objective.object_position_std=0.2 termination.object_orientation_threshold=1.5 env.termination.object_orientation_threshold=1.5"
fi
# 物体朝向终止（计划 agent-doc/plan-t3hand-2026-10-27.md 第 3 版 P0 的对照臂 oc15）：队列里每一行都写了同一个值就断言那个值（变体 A 的行本来就写 0.6），混在一起就停
nr="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c 'termination.object_orientation_threshold=' || true)"
rv="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -o 'termination.object_orientation_threshold=[0-9.]*' | sort -u || true)"
if [ "$nr" = "$nq" ] && [ "$(grep -c . <<< "$rv")" = 1 ]; then VAR="${VAR//termination.object_orientation_threshold=0.6/$rv}"
elif [ "$nr" != 0 ]; then stop "队列里物体朝向终止的覆盖不一致（${nq} 行里 ${nr} 行写了：${rv//$'\n'/ }）"; fi
# 物体位置终止（计划 agent-doc/plan-t2term-2026-10-20.md 复核 MINOR 4）：队列里每一行都写了同一个值就正向断言它，混在一起就停
no="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c 'termination.object_position_threshold=' || true)"
ov="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -o 'termination.object_position_threshold=[0-9.]*' | sort -u || true)"   # 没有这一项时 grep 返回 1，set -e 下要接住（10/21 t3pg5a 预演因此静默退出）
if [ "$no" = "$nq" ] && [ "$(grep -c . <<< "$ov")" = 1 ]; then VAR="$VAR $ov env.$ov"
elif [ "$no" != 0 ]; then stop "队列里物体位置终止的覆盖不一致（${nq} 行里 ${no} 行写了：${ov//$'\n'/ }）"; fi
# 手掌位置奖励的尺度（计划 agent-doc/plan-t1palm-2026-10-30.md）：队列里每一行都写了同一个值、run 名都带 -r3h- 就正向断言它（顶层与 env 下各一份），混在一起就停
nh="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c 'objective.palm_position_std=' || true)"
hv="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -o 'objective.palm_position_std=[0-9.]*' | sort -u || true)"
nhr="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -cE -- '-r3[hi]-' || true)"
case "$hv" in objective.palm_position_std=0.05) ht=-r3h- ;; objective.palm_position_std=0.1) ht=-r3i- ;; *) ht="" ;; esac   # 第 3 版：0.05 配 -r3h-、0.1 配 -r3i-
nht=0; [ -z "$ht" ] || nht="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c -- "$ht" || true)"
if [ "$nh" = "$nq" ] && [ -n "$ht" ] && [ "$nht" = "$nq" ] && [ "$nhr" = "$nq" ]; then VAR="$VAR $hv env.$hv"
elif [ "$nh" != 0 ] || [ "$nhr" != 0 ]; then stop "队列里手掌尺度的覆盖或 -r3h-/-r3i- 不一致（${nq} 行里 ${nh} 行写了覆盖、${nhr} 行带标签：${hv//$'\n'/ }）"; fi
# 抓握闭合项权重、手掌位置终止（同计划）：队列里每一行都写了同一个值就正向断言它（顶层与 env 下各一份），混在一起就停
for k in objective.force_closure_weight termination.palm_position_threshold; do
  nk="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c -- "$k=" || true)"
  kv="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -o -- "$k=[0-9.]*" | sort -u || true)"
  if [ "$nk" = "$nq" ] && [ "$(grep -c . <<< "$kv")" = 1 ]; then VAR="$VAR $kv env.$kv"
  elif [ "$nk" != 0 ]; then stop "队列里 $k 的覆盖不一致（${nq} 行里 ${nk} 行写了：${kv//$'\n'/ }）"; fi
done
# 手掌位置终止的值与 run 名标签要对上（计划 agent-doc/plan-t1mppe-2026-11-01.md 复核 NIT 6）：0.08 配 -pt08-、0.05 配 -pt05-
pv="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -o -- 'termination.palm_position_threshold=[0-9.]*' | sort -u || true)"
case "$pv" in "") pt="" ;; termination.palm_position_threshold=0.08) pt=-pt08- ;; termination.palm_position_threshold=0.05) pt=-pt05- ;; *) stop "手掌位置终止的值没有对应的 run 名标签：${pv//$'\n'/ }" ;; esac
npt="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -cE -- '-pt0[0-9]-' || true)"
if [ -n "$pt" ]; then
  nptm="$(grep -hv '^#' "$Q.txt" "${Q}_smoke.txt" | grep -c -- "$pt" || true)"
  [ "$nptm" = "$nq" ] && [ "$npt" = "$nq" ] || stop "队列里手掌位置终止 ${pv} 与 run 名标签 ${pt} 对不上（${nq} 行里 ${nptm} 行带 ${pt}）"
elif [ "$npt" != 0 ]; then stop "run 名带 -ptNN- 标签，队列里却没有手掌位置终止的覆盖"; fi
cmds=""
for r in "${cfgs[@]}"; do cmds+="python /repo/stages/policy/assert_cfg.py /repo/runs/cfgcheck/$r.yaml $VAR || bad=1; "; done
scripts/dev/in-flash-chord "bad=0; ${cmds}exit \$bad" || stop "配方断言不过（见上）"
echo "核对：$N 在 ${M}，${#runs[@]} 条 run，来历 ${src0}，可用 ${free_mb} MB，训练副本 ${COPY}，${#cfgs[@]} 份配置的 precheck 与配方断言全过"

CMD_T="V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC/$COPY $WAIT ${EXTRA:+$EXTRA }stages/policy/day2_train_chain.sh $Q.expect.tsv ${Q}_smoke.txt $Q.txt $PEAK $LT"
CMD_E="V2D_FC_SRC=$SRC/fc_8c778f9 $WAIT V2D_EVAL_STEPS=349999104 V2D_WAIT_FOR_LOG=runs/logs/$LT.log stages/evaluate/day2_eval_chain.sh $Q.expect.tsv - $LE"
CMD_X="V2D_FC_SRC=$SRC/fc_8c778f9 V2D_FALLBACK_MAX=0 stages/evaluate/export_after_eval.sh runs/logs/$LE.log $LIST $LX"
if [ "${V2D_ARM_DRYRUN:-0}" = 1 ]; then
  printf '预演，不起会话：\n  train:  %s\n  eval:   %s\n  export: %s\n' "$CMD_T" "$CMD_E" "$CMD_X"
  echo DRYRUN_OK; exit 0
fi
tmux new-session -d -s "${SESS[0]}" -c "$PWD" "$CMD_T"; sleep 2
tmux new-session -d -s "${SESS[1]}" -c "$PWD" "$CMD_E"; sleep 2
tmux new-session -d -s "${SESS[2]}" -c "$PWD" "$CMD_X"; sleep 3
# 以下几行只供人看：起来了就以会话与标记为准
set +e
tmux ls | grep -E "^v2d-$N-(train|eval|export):"
for L in "$LT" "$LE" "$LX"; do echo "${L}：$(tail -1 "runs/logs/$L.log" 2>/dev/null)"; done
ls runs/.chain-active-* 2>/dev/null
n=0; for s in "${SESS[@]}"; do tmux has-session -t "$s" 2>/dev/null && n=$((n + 1)); done
if [ "$n" = 3 ] && [ -e "runs/.chain-active-$LT" ]; then echo ARMED; else echo "没起全：会话 $n / 3" >&2; exit 1; fi
