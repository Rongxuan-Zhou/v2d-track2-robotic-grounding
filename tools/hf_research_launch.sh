#!/usr/bin/env bash
# 三台机器的 plan 参数，按机器启动 tools/hf_research_remote.sh（Mac 上、仓库根执行）。
# 计划 agent-doc/plan-hf-research-archive-2026-11-04.md 第 3 版“各机器的 plan 参数”。
# 用法：tools/hf_research_launch.sh <mf4090|ml40|mac> <prepare|upload|all>
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
M=${1:?机器}; STEPS=${2:?步骤}
case "$M" in
  mf4090)
    H=mf4090; V=/home/rxzhou/v2d; R=$V/maniphys_lab_v2d_challenge; X=$V/hf_research_extra; W=$V/hf_research; PY=python3
    A=(--src "$R/runs=repo/runs" --pack "$R/runs/stage=repo/runs/stage.tar" --src "$R/data=repo/data"
       --src "$V/envs=v2d/envs" --pack "$V/envs/trajopt_pkgs=v2d/envs/trajopt_pkgs.tar:public"
       --src "$V/work=v2d/work" --src "$V/scripts=v2d/scripts")
    for f in fetch2.sh fetch_soma_assets.sh install_soma.sh lfs_pull.log lfs_pull.sh noise_probe.log setup2.sh setup3.sh \
             setup_phase0.sh smoke.sh sweep.log sweep.sh smoke.log; do A+=(--src "$V/$f=v2d/$f"); done
    A+=(--src "/home/rxzhou/work=home/work")
    for f in dl_challenge.sh setup_hf.sh v2d_download.log; do A+=(--src "/home/rxzhou/$f=home/$f"); done
    A+=(--src "$X/fc_src=v2d/fc_src:private" --src "$X/meta=meta:private") ;;  # docker 构建历史可能带构建参数，放私有
  ml40)
    H=v2d-cpu; V=/lab_data/rxzhou/v2d; R=$V/maniphys_lab_v2d_challenge; X=$V/hf_research_extra; W=$V/hf_research
    PY=$W/py
    A=(--src "$R/runs=repo/runs" --src "$R/data=repo/data"
       --src "$V/t3recon=v2d/t3recon" --pack "$V/t3recon/env=v2d/t3recon/env.tar:public" --src "$V/container=v2d/container")
    for d in tier3_kfix tier3_kfix_r2 tier3_gb3feet t3sil t3_init_check t3_init_check.out report_build tools_bin; do
      A+=(--src "$V/$d=v2d/$d"); done
    A+=(--src "/lab_data/rxzhou/ops=ops" --src "$X/fc_src=v2d/fc_src:private" --src "$X/meta=meta:private") ;;
  mac)
    H=local; R=$ROOT; W=$HOME/v2d_hf_research/work; PY=$HOME/v2d_hf_research/venv/bin/python
    A=(--src "$R/runs=repo/runs" --pack "$R/runs/container=repo/runs/container.tar" --src "$R/data=repo/data"
       --src "$HOME/v2d_hf_research/mac_code=code:private") ;;
  *) echo "机器只能是 mf4090、ml40、mac" >&2; exit 2 ;;
esac
exec "$ROOT/tools/hf_research_remote.sh" "$H" "$M" "$W" "$PY" "$STEPS" -- "${A[@]}"
