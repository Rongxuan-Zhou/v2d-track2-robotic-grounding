#!/usr/bin/env bash
# t3fix B1（贴图选朝向）在 30 集上全量跑（ml40，CPU）。计划 agent-doc/plan-t3fix-2026-10-27.md 第 3.1 版第 1 节 B1。6 路，每路 4 线程；不进 B1 的集脚本自己跳过。
# 代码取 t3recon/dev_t3fx/code（track2/t3sil-choose 的 stages/recovery/tier3 经 rsync 部署）。日志 runs/logs 下，结果在 dev_t3fx/tex。
set -uo pipefail
C=/lab_data/rxzhou/v2d/t3recon/dev_t3fx/code/stages/recovery/tier3
R=/lab_data/rxzhou/v2d/t3recon/t3fx
PY=/lab_data/rxzhou/v2d/t3recon/env/gate/bin/python
OUT=/lab_data/rxzhou/v2d/t3recon/dev_t3fx/tex
L=/lab_data/rxzhou/v2d/maniphys_lab_v2d_challenge/runs/logs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$C"
args=("$R" "$R/runs/tier3_cari4d_gb5_best.selection.json" tier3_cari4d_gb5_best_dr2s_dyn7 "$R/runs/physics/tier3_cari4d_gb5_best_dr2s" "$OUT")
pids=()
for s in 0 1 2 3 4 5; do
  eps=$(python3 -c "print(\",\".join(str(e) for e in range(30) if e % 6 == $s))")
  "$PY" t3_fix_tex.py "${args[@]}" stride=3 only=$eps > "$L/t3fx_tex_$s.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "T3FX_TEX_DONE（失败 ${fail}）" >> "$L/t3fx_tex_0.log"
