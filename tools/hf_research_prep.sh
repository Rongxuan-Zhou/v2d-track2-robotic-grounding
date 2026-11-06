#!/usr/bin/env bash
# 赛后研究归档的准备步骤（机器上执行，plain ssh 即可，不需要凭据）：把 FlashCHORD 各份副本压成“一份底 + 若干 patch”，
# 再记一份机器信息，都写进 <输出目录>，之后作为 plan 的源（fc_src 走私有，meta 走公开）。只读源目录。
# 计划 agent-doc/plan-hf-research-archive-2026-11-04.md 第 3 版。
# 用法：bash hf_research_prep.sh <fc_src 目录> <输出目录> <要记大小的目录...>
# 最后一行 PREP_DONE 或 PREP_FAILED <原因>。
set -uo pipefail
FC=${1:?fc_src}; X=${2:?输出目录}; shift 2
fail() { echo "PREP_FAILED $1"; exit 1; }
mkdir -p "$X/fc_src" "$X/meta" || fail mkdir
cd "$FC" || fail "cd $FC"
[ -d fc_8c778f9 ] || fail "没有 fc_8c778f9"
listhash() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum); }
tar -C "$FC" -cf - fc_8c778f9 | zstd -q -10 -T2 -o "$X/fc_src/fc_8c778f9.tar.zst" -f || fail "tar fc_8c778f9"
listhash fc_8c778f9 > "$X/fc_src/fc_8c778f9.files.sha256"
for v in *; do
  [ -d "$v" ] && [ "$v" != fc_8c778f9 ] || continue
  git diff --no-index --binary fc_8c778f9 "$v" > "$X/fc_src/$v.vs_fc_8c778f9.patch"; rc=$?
  [ $rc -le 1 ] || fail "diff $v"
  listhash "$v" > "$X/fc_src/$v.files.sha256"
  # 自检：按 README.txt 的步骤还原一遍，文件 sha256 列表必须与原目录相同；还原的临时副本随后删掉（是本脚本自己建的）
  rm -rf "$X/check" && mkdir -p "$X/check" && cp -a fc_8c778f9 "$X/check/$v" || fail "check copy $v"
  (cd "$X/check/$v" && git apply -p2 --binary "$X/fc_src/$v.vs_fc_8c778f9.patch") || fail "check apply $v"
  cmp -s <(listhash "$X/check/$v") "$X/fc_src/$v.files.sha256" || fail "check $v（还原后与原目录不同）"
  rm -rf "$X/check"; echo "还原自检通过：$v"
done
cat > "$X/fc_src/README.txt" <<'EOF'
还原：tar --zstd -xf fc_8c778f9.tar.zst；cp -a fc_8c778f9 <改动版>；cd <改动版>；git apply -p2 --binary ../<改动版>.vs_fc_8c778f9.patch
（patch 由 git diff --no-index --binary fc_8c778f9 <改动版> 生成，路径前缀是 a/fc_8c778f9、b/<改动版>，所以用 -p2）。
还原后用 <改动版>.files.sha256 核对：cd <改动版> && sha256sum -c ../<改动版>.files.sha256
EOF
{
  echo "# $(date '+%F %T %Z') $(hostname)"; uname -a; echo; nvidia-smi -L 2>&1; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1
  echo; df -h 2>&1 | grep -v -E 'tmpfs|overlay|loop'; echo; for d in "$@"; do du -sh "$d" 2>&1; done
  echo; python3 --version 2>&1; docker --version 2>&1
} > "$X/meta/machine.txt"
if docker info >/dev/null 2>&1; then
  docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}} {{.Size}}' > "$X/meta/docker_images.txt" 2>&1
  for img in flash_chord:latest robotic-grounding:latest; do
    f="$X/meta/docker_$(tr ':/' '__' <<< "$img")"
    docker image inspect "$img" --format '{{json .Id}} {{json .RepoDigests}} {{json .Created}} {{json .Config.Labels}}' > "$f.inspect.txt" 2>&1
    docker image inspect "$img" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | cut -d= -f1 > "$f.env_names.txt"
    docker history --no-trunc --format '{{.CreatedAt}}\t{{.Size}}\t{{.CreatedBy}}' "$img" > "$f.history.txt" 2>&1
  done
fi
echo PREP_DONE
