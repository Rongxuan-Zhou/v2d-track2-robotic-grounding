#!/usr/bin/env bash
# 计划 agent-doc/plan-t3auc1-2026-11-03.md 第 1 节：把 ml40 上 1000 万臂（ep0、26、18）要的 Tier 3 gb4 参考打成 tar，经 HF 私有中转库转到 ml40。
#   stage  （mf4090 仓库根）：runs/hf_src/<标签>/ 下 motion.tar（runs/motion/tier3_gb4 里三个序列的整个目录与支撑面文件）、
#          msrc.tar（runs/motion_src/tier3_gb4 里三个序列的整个目录，导出读 v2d_meta.json）、root.tar（runs/tier3/root_gb4 整份，打包做帧参考用），
#          加 TARS.sha256。目标目录已存在就停。最后一行 STAGE_OK <标签> 或 STAGE_FAIL <原因>。
#   unpack （ml40 仓库根）：<下载目录>/runs/hf_src/<标签>/ 按 TARS.sha256 核对后解到 runs/ 下同样的相对位置；目标已存在就停（不覆盖）。
#          最后一行 UNPACK_OK 或 UNPACK_FAIL <原因>。
# 计划 agent-doc/plan-final-2026-11-03.md 第 1 节（ml40 的 t3qr3o2f）：V2D_GB4_SEQS="<序列> …" 换成别的序列；V2D_GB4_NO_ROOT=1 不打、不解 root.tar
#   （ml40 上已有 runs/tier3/root_gb4，解包时目标已存在会停）。
# 11/04 plan-t3gb6：V2D_GB4_SET=<集名> 换掉 gb4（例如 gb6dyn7），默认 gb4 不变。
# 用法：bash tools/t3auc1_gb4_stage.sh stage <标签>；bash tools/t3auc1_gb4_stage.sh unpack <标签> <下载目录>
set -uo pipefail
MODE=${1:?用法：t3auc1_gb4_stage.sh stage|unpack <标签> [<下载目录>]}; TAG=${2:?标签}
SEQS=(2026-06-04_14-16-23_tall_bar_stool_tip_up_04 2026-06-05_16-05-08_skateboard_rotate_ground_10 2026-05-06_10-00-19_blue_trash_can_drag_sideways_08)   # ep0、26、18（取自 mf4090_t3pg4.txt）
[ -n "${V2D_GB4_SEQS:-}" ] && read -r -a SEQS <<< "$V2D_GB4_SEQS"
SET=${V2D_GB4_SET:-gb4}; ROOTP=tier3/root_$SET; TARS="motion msrc root"; [ "${V2D_GB4_NO_ROOT:-0}" = 1 ] && { ROOTP=""; TARS="motion msrc"; }
fail() { echo "${MODE^^}_FAIL $*"; exit 1; }
paths_motion() { for s in "${SEQS[@]}"; do echo "motion/tier3_$SET/whole_body/soma/sequence_id=$s"; echo "motion/tier3_$SET/whole_body/reconstructed_stage/${s}_support.usda"; done; }
paths_msrc() { for s in "${SEQS[@]}"; do echo "motion_src/tier3_$SET/$s"; done; }
case "$MODE" in
  stage)
    OUT="runs/hf_src/$TAG"; [ -e "$OUT" ] && fail "$OUT 已存在"
    for p in $(paths_motion) $(paths_msrc) $ROOTP; do [ -e "runs/$p" ] || fail "缺 runs/$p"; done
    mkdir -p "$OUT" || fail mkdir
    tar -cf "$OUT/motion.tar" -C runs $(paths_motion) || fail motion
    tar -cf "$OUT/msrc.tar" -C runs $(paths_msrc) || fail msrc
    [ -z "$ROOTP" ] || tar -cf "$OUT/root.tar" -C runs "$ROOTP" || fail root
    ( cd "$OUT" && sha256sum *.tar > TARS.sha256 ) || fail sha
    ls -l "$OUT"; echo "STAGE_OK $TAG" ;;
  unpack)
    IN=${3:?下载目录}; D="$IN/runs/hf_src/$TAG"
    [ -f "$D/TARS.sha256" ] || fail "缺 $D/TARS.sha256"
    ( cd "$D" && sha256sum -c TARS.sha256 ) || fail sha256
    for p in $(paths_motion) $(paths_msrc) $ROOTP; do [ -e "runs/$p" ] && fail "runs/$p 已存在"; done
    for t in $TARS; do tar -xf "$D/$t.tar" -C runs || fail "$t"; done
    du -sh "runs/motion/tier3_$SET" "runs/motion_src/tier3_$SET" ${ROOTP:+"runs/$ROOTP"}; echo UNPACK_OK ;;
  *) fail "模式只能是 stage 或 unpack" ;;
esac
