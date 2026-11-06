"""从两份只读核对记录生成拼 Tier 3 档目录时要比对的哈希表（Mac 上跑，只用标准库）。计划 agent-doc/plan-t3-pilot-2026-10-09.md 第 3 节（Fable 5.1 审查 M1）。

输入：
  <远端记录>    tools/t3_episode_remote.sh 的输出（results/ops/t3_episodes_ml40_*.txt）：取“可读 … sha256 <hex> <相对路径>”行，
                即同组成员产物里各集 parquet 与元数据的 sha256；
  <Tier 2 记录> results/ops/t3p_mf4090_inputs_*.txt：取“<md5> <相对路径>”行，即 mf4090 上官方 Tier 2 目录里网格、地面平面与支撑面的 md5。
输出（stdout，TSV，无表头）：相对路径、算法、哈希。data/、meta/ 下的是转来的产物（sha256），mesh/、ground_plane/、support_surface/ 下的是
Tier 2 目录里的文件（md5）。同一路径出现两次且哈希不同、或一行也没取到，就报错退出。
--support-from-remote（计划 agent-doc/plan-t3-hm-p2-2026-10-12.md：Tier 3 不用官方支撑面，取交付目录里的文件）：支撑面的行改取远端记录里
  “可读 … sha256 <hex> support_surface/<文件名>.usda” 行（tools/t3_episode_remote.sh 设 V2D_T3_SUPPORT=delivery 时打的；不许子目录），
  Tier 2 记录里的 support_surface/ 行丢掉；远端的支撑面行数必须等于 parquet 行数（每集一份），否则报错。不加时输出与以前相同。
--support-record <记录>（计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md：动力学修复版 best_dyn4、best_dyn4s 的目录里没有支撑面，元数据与 best_hm
  的逐字节相同，支撑面沿用 best_hm 交付目录里的文件）：支撑面的行改取另一份远端记录（同样是 tools/t3_episode_remote.sh 设 V2D_T3_SUPPORT=delivery
  打的，例如 best_hm 的）里“可读 … sha256 <hex> support_surface/<文件名>.usda”行，只取 <远端记录> 里各集元数据 JSON 的 support_surface 指到的那几份；
  缺一份、或某集元数据没有 support_surface 就报错；Tier 2 记录里的 support_surface/ 行丢掉。与 --support-from-remote 只能给一个。
用法：python3 stages/recovery/tier3_expect.py <远端记录> <Tier 2 记录> [--support-from-remote | --support-record <记录>] > stages/recovery/expect/<标签>.tsv
测试：stages/recovery/test_tier3_expect.py。
"""
import argparse
import json
import re
import sys
from pathlib import Path

REMOTE = re.compile(r"^可读 .* sha256 ([0-9a-f]{64}) ((?:data|meta)/\S+)$")
REMOTE_SUP = re.compile(r"^可读 .* sha256 ([0-9a-f]{64}) ((?:data|meta)/\S+|support_surface/[^/\s]+\.usda)$")
TIER2 = re.compile(r"^([0-9a-f]{32}) ((?:mesh|ground_plane|support_surface)/\S+)$")
TIER2_NO_SUP = re.compile(r"^([0-9a-f]{32}) ((?:mesh|ground_plane)/\S+)$")
SUP_ONLY = re.compile(r"^可读 .* sha256 ([0-9a-f]{64}) (support_surface/[^/\s]+\.usda)$")
SUP_REL = re.compile(r"^support_surface/[A-Za-z0-9._-]+\.usda$")


def needed_supports(remote: Path) -> list:
    """<远端记录> 里各集元数据 JSON（以 { 开头的整行）的 support_surface，按出现顺序、不去重。"""
    out = []
    for line in remote.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise SystemExit(f"tier3_expect: {remote} 里这一行像元数据 JSON 但读不了：{line[:80]}")
        rel = row.get("support_surface") or ""
        if not SUP_REL.match(rel):
            raise SystemExit(f"tier3_expect: {remote} 里 ep{row.get('episode_index')} 的 support_surface 不合要求：{rel!r}")
        out.append(rel)
    return out


def collect(remote: Path, tier2: Path, support_from_remote: bool = False, support_record: Path = None) -> dict:
    if support_from_remote and support_record:
        raise SystemExit("tier3_expect: --support-from-remote 与 --support-record 只能给一个")
    out = {}
    rpat, tpat = (REMOTE_SUP, TIER2_NO_SUP) if support_from_remote else (REMOTE, TIER2_NO_SUP if support_record else TIER2)
    for path, pat, algo in ((remote, rpat, "sha256"), (tier2, tpat, "md5")):
        n = 0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():   # record 的 command 行里非 ASCII 参数被转义坏了，不是合法 UTF-8
            m = pat.match(line.strip())
            if not m:
                continue
            rel, h = m.group(2), m.group(1)
            if rel in out and out[rel] != (algo, h):
                raise SystemExit(f"tier3_expect: {rel} 出现两次且哈希不同")
            out[rel] = (algo, h)
            n += 1
        if n == 0:
            raise SystemExit(f"tier3_expect: {path} 里一行 {algo} 也没取到")
    if support_from_remote:
        n_sup = sum(r.startswith("support_surface/") for r in out)
        n_pq = sum(r.startswith("data/") and r.endswith(".parquet") for r in out)
        if n_sup != n_pq:
            raise SystemExit(f"tier3_expect: --support-from-remote 要每集一份支撑面，远端记录里支撑面 {n_sup} 行、parquet {n_pq} 行")
    if support_record:
        have = {}
        for line in support_record.read_text(encoding="utf-8", errors="replace").splitlines():
            m = SUP_ONLY.match(line.strip())
            if m:
                if m.group(2) in have and have[m.group(2)] != m.group(1):
                    raise SystemExit(f"tier3_expect: {support_record} 里 {m.group(2)} 出现两次且哈希不同")
                have[m.group(2)] = m.group(1)
        need = needed_supports(remote)
        n_pq = sum(r.startswith("data/") and r.endswith(".parquet") for r in out)
        if len(need) != n_pq or len(set(need)) != len(need):
            raise SystemExit(f"tier3_expect: --support-record 要每集一份支撑面，元数据里 {len(need)} 份（不同的 {len(set(need))} 份）、parquet {n_pq} 行")
        missing = [r for r in need if r not in have]
        if missing:
            raise SystemExit(f"tier3_expect: {support_record} 里没有这些支撑面的 sha256：{missing}")
        for rel in need:
            out[rel] = ("sha256", have[rel])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用法：tier3_expect.py <远端记录> <Tier 2 记录> [--support-from-remote | --support-record <记录>]")
    ap.add_argument("remote", type=Path)
    ap.add_argument("tier2", type=Path)
    ap.add_argument("--support-from-remote", action="store_true")
    ap.add_argument("--support-record", type=Path)
    a = ap.parse_args(argv)
    for rel, (algo, h) in sorted(collect(a.remote, a.tier2, a.support_from_remote, a.support_record).items()):
        print(f"{rel}\t{algo}\t{h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
