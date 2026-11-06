#!/usr/bin/env bash
# r3o_lines.awk 的测试（契约 POLICY-I010）：正例、σ 与 -sgDD- 不符、缺变量、变量重复、普通副本带变量、t1o 风格的 0.4 行。Mac bash 3.2 可跑。
# 用法：stages/policy/test_r3o_lines.sh   最后一行 R3O_LINES_OK 或 R3O_LINES_FAIL
set -uo pipefail
cd "$(dirname "$0")/../.."
A=stages/policy/r3o_lines.awk
fail=0
chk() {  # chk <oo> <期望坏行数> <行...>
  local oo=$1 want=$2; shift 2
  local got; got="$(printf '%s\n' "$@" | awk -v oo="$oo" -f "$A" | tail -1)"
  if [ "$got" != "$want" ]; then echo "FAIL oo=$oo 期望 $want 得到 $got：$1"; fail=1; else echo "ok   oo=$oo 坏行 $got"; fi
}
G04="V2D_WANDB_TAG=t V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 t3-ep27-w2048-cap4m-r3o-gb5-ocoff-sg04-mf4090 /p 2048 1 termination.object_orientation_threshold=3.2"
G20="V2D_WANDB_TAG=t V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=2.0 t3-ep27-w2048-cap4m-r3o-gb5-ocoff-sg20-mf4090 /p 2048 1"
G10="V2D_WANDB_TAG=t V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=1.0 smoke-t3-ep15-w2048-cap10m-r3o-gb5-ocoff-sg10-ml40 /p 2048 1"
T1O="V2D_WANDB_TAG=t1o V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 t1-ep20-w2048-cap4m-r3o-mf4090 /p 2048 1"
chk 1 0 "$G04" "$G20" "$G10" "$T1O"
chk 1 1 "V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 t3-ep27-w2048-cap4m-r3o-gb5-ocoff-sg20-mf4090 /p 2048 1"   # 标签 sg20、σ 0.4
chk 1 1 "V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=2.0 t1-ep20-w2048-cap4m-r3o-mf4090 /p 2048 1"                   # 没有标签时 σ 要 0.4
chk 1 1 "V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.0 t3-ep27-w2048-cap4m-r3o-gb5-ocoff-sg00-mf4090 /p 2048 1"    # σ 0 不许
chk 1 1 "V2D_OBJ_ORI_WEIGHT=1.0 t3-ep27-w2048-cap4m-r3o-gb5-ocoff-sg04-mf4090 /p 2048 1"                        # 缺 STD
chk 1 1 "V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 t3-ep27-w2048-cap4m-r3a-gb5-ocoff-sg04-mf4090 /p 2048 1"   # run 名没有 -r3o-
chk 1 1 "V2D_OBJ_ORI_STD=0.4 V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 t3-ep27-w2048-cap4m-r3o-gb5-ocoff-sg04-mf4090 /p 2048 1"   # STD 两次
chk 1 1 "V2D_OBJ_ORI_WEIGHT=0.5 V2D_OBJ_ORI_STD=0.4 t3-ep27-w2048-cap4m-r3o-gb5-ocoff-sg04-mf4090 /p 2048 1"   # 权重不是 1.0
chk 0 0 "V2D_WANDB_TAG=t3pocoff t3-ep27-w2048-cap4m-r3a-gb5-ocoff-mf4090 /p 2048 1 termination.object_orientation_threshold=3.2"
chk 0 1 "$G04"                                                                                                  # 普通副本带了变量
chk 0 1 "V2D_WANDB_TAG=t t3-ep27-w2048-cap4m-r3o-gb5-mf4090 /p 2048 1"                                          # 普通副本 run 名带 -r3o-
# 仓库里的真实队列：r3o 段全过、普通段全过
for q in mf4090_t3pr3o ml40_t3qr3o mf4090_t1oa ml40_t2oa; do
  got="$(grep -vhE '^[[:space:]]*(#|$)' stages/policy/queues/${q}_smoke.txt stages/policy/queues/$q.txt | awk -v oo=1 -f "$A" | tail -1)"
  [ "$got" = 0 ] && echo "ok   队列 $q 坏行 0" || { echo "FAIL 队列 $q 坏行 $got"; fail=1; }
done
for q in mf4090_t3pocoff ml40_t3qcoff ml40_t3qhr13; do
  got="$(grep -vhE '^[[:space:]]*(#|$)' stages/policy/queues/${q}_smoke.txt stages/policy/queues/$q.txt | awk -v oo=0 -f "$A" | tail -1)"
  [ "$got" = 0 ] && echo "ok   队列 $q 坏行 0" || { echo "FAIL 队列 $q 坏行 $got"; fail=1; }
done
[ "$fail" = 0 ] && echo R3O_LINES_OK || echo R3O_LINES_FAIL
exit $fail
