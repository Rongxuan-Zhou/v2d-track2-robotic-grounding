#!/usr/bin/env bash
# 从 ml40 只读拷另一位用户的 Tier 3 best_hm 逐帧证据表（$T/dev/perframe/hm/ep*.csv 与 summary.json）到本仓，逐个核 sha256 与登记的快照相同
# （计划 agent-doc/plan-frame-onset-2026-10-14.md 第 3 节、第 8 节第 12 条）。不同或缺文件就失败，目标目录里只留核对过的文件。
# 另打用到的布尔列（label_left、label_right、viol_object_any、viol_rot_*、viol_root_acc、orient_outlier）的取值集。输出里不出现别人的目录名。
# Mac 上、仓库根执行（bash 3.2 可跑）：
#   scripts/dev/record t3frame/evidence_fetch_hm tools/t3_evidence_fetch.sh results/t3frame/evidence_hm_1015_1235cst results/ops/t3_perframe_hm_ml40_1014_2355.txt
set -uo pipefail
DST=${1:?目标目录}; REC=${2:?登记快照的记录}
[ -e "$DST" ] && { echo "t3_evidence_fetch: $DST 已存在，拒绝覆盖" >&2; exit 2; }
want="$(grep -E '^[0-9a-f]{64}  (ep[0-9]{2}\.csv|summary\.json)$' "$REC")"
n="$(printf '%s\n' "$want" | grep -c .)"
[ "$n" = 31 ] || { echo "t3_evidence_fetch: $REC 里登记的 sha256 是 $n 个，不是 31 个" >&2; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
root="$(ssh -n -o BatchMode=yes ml40 'ls -d /lab_data/*/v2d/tier3 | grep -v "^/lab_data/$(id -un)/" | head -1')"
[ -n "$root" ] || { echo "t3_evidence_fetch: ml40 上找不到另一位用户的 tier3 目录" >&2; exit 2; }
ssh -n -o BatchMode=yes ml40 "cd '$root/dev/perframe/hm' && tar cf - ep[0-9][0-9].csv summary.json" 2>/dev/null | tar xf - -C "$tmp" \
  || { echo "t3_evidence_fetch: 拷贝失败" >&2; exit 1; }
bad=0
while read -r sha f; do
  got="$(shasum -a 256 "$tmp/$f" 2>/dev/null | cut -d' ' -f1)"
  if [ "$got" = "$sha" ]; then echo "ok   $f $sha"; else echo "FAIL ${f} 登记 ${sha}，拷来的 ${got:-缺}"; bad=$((bad + 1)); fi
done <<< "$want"
extra="$(cd "$tmp" && ls | grep -vxF -f <(printf '%s\n' "$want" | awk '{print $2}') || true)"
[ -z "$extra" ] || { echo "FAIL 拷来的多出文件：$extra"; bad=$((bad + 1)); }
if [ "$bad" != 0 ]; then echo "T3_EVIDENCE_FETCH_FAIL $bad"; exit 1; fi
echo "== 用到的布尔列的取值（30 集合计）"
for c in 4 5 7 11 12 13 15 16 19; do
  name="$(head -1 "$tmp/ep00.csv" | cut -d, -f"$c")"
  echo "${name}：$(tail -q -n +2 "$tmp"/ep*.csv | cut -d, -f"$c" | sort | uniq -c | awk '{printf "%s×%s ", $2, $1}')"
done
mkdir -p "$DST" && cp "$tmp"/ep*.csv "$tmp"/summary.json "$DST"/
echo "T3_EVIDENCE_FETCH_OK 31 个文件 → $DST"
