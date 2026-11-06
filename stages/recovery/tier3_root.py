"""从转来的 Tier 3 重建产物拼出档目录（计划 agent-doc/plan-t3-pilot-2026-10-09.md 第 3 节）。只用标准库。

输入 <incoming>：stages/recovery/tier3_pack_remote.sh 打的 tar 解开后的目录，有 data/chunk-000/episode_<N>.parquet、
meta/episode_<N>.metadata.json（同组成员的转换脚本每集写一份，字段与 Tier 2 的 episodes_metadata.jsonl 一行相同）与 DEPLOYED。
--expect <表>：stages/recovery/tier3_expect.py 从两份只读核对记录生成的哈希表（相对路径、算法、哈希）。转来的每个 parquet 与元数据（sha256）、
元数据里 mesh、ground_plane、support_surface 指向的官方 Tier 2 文件（md5）都要与表里相同，表里每一行也都要用上；
任何一处不符就退出码 1，什么也不写（Fable 5.1 审查 M1：同组成员的产物还在变，换了新版本不能悄悄进来）。
输出 <root>（默认 libs/v2d_infra/paths.py 的 TIER3，即 runs/tier3/root）：
  data/chunk-000/episode_<N>.parquet、meta/episodes_metadata.jsonl（每集一行，按 episode_index 排）、
  元数据里三个相对路径指向的文件（从官方 Tier 2 目录 --tier2 复制），以及 meta/provenance.json（每集的来源：parquet 在同组成员那边的修改时间、
  各文件哈希、哈希表的路径，以及转发时对方的部署版本——那是转发那一刻的代码版本，不一定是写出这份 parquet 的版本，文件是不是那一份以哈希为准；
  stages/retarget/challenge_to_soma.py 转 Tier 3 时把这一集的那份写进 v2d_meta.json，审查 m5）。
  规则解读未确认：Tier 3 用 Tier 2 目录里的官方网格、地面平面与支撑面（同组成员的重建也用了前两者），正式做 Tier 3 前要问组委会。
同一集已在 root 里：parquet 与元数据都相同就跳过，不同就拒绝。相对路径必须留在各自目录里（不许绝对路径与 ..）。
--support-from-incoming（计划 agent-doc/plan-t3-hm-p2-2026-10-12.md：Tier 3 不用官方支撑面）：支撑面改从 <incoming>/<相对路径> 复制
  （tier3_pack_remote.sh 设 V2D_T3_SUPPORT=delivery 时打进 tar 的交付目录里的文件），按哈希表核（表里是 B.1 记录的 sha256）；
  provenance 每集另记 support_source = incoming 与 support_sha256，tier2_md5 只有网格与地面，note 照实写来路。不加时输出与以前相同。
--support-dir <目录>（计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md：动力学修复版的目录里没有支撑面，沿用 best_hm 交付的那几份）：支撑面改从
  <目录>/<相对路径> 复制（例如 runs/tier3/root_hm，里面的支撑面是 best_hm 拼目录时按哈希表核过的交付文件），按哈希表的 sha256 核
  （stages/recovery/tier3_expect.py --support-record 生成）；provenance 记 support_source = <目录> 与 support_sha256。与 --support-from-incoming 只能给一个。
开头调 tier_guard.forbid_tier1()：本进程读不到 Tier 1 的任何文件。
用法（仓库根）：python3 stages/recovery/tier3_root.py <incoming> --expect <表> [--root <输出>] [--tier2 <官方 Tier 2 目录>]
                [--support-from-incoming | --support-dir <目录>]
测试：stages/recovery/test_tier3_root.py。
"""
import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from paths import TIER2, TIER3  # noqa: E402
from tier_guard import forbid_tier1  # noqa: E402

LINKED = ("mesh", "ground_plane", "support_surface")


def digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_rel(rel: str) -> PurePosixPath:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise SystemExit(f"tier3_root: 元数据里的相对路径不安全：{rel}")
    return p


def load_expect(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rel, algo, h = line.split("\t")
        if algo not in ("sha256", "md5") or rel in out:
            raise SystemExit(f"tier3_root: 哈希表 {path} 的这一行不对：{line}")
        out[rel] = (algo, h)
    if not out:
        raise SystemExit(f"tier3_root: 哈希表 {path} 是空的")
    return out


def deployed(incoming: Path) -> dict:
    text = (incoming / "DEPLOYED").read_text() if (incoming / "DEPLOYED").is_file() else ""
    m = re.search(r"\b[0-9a-f]{40}\b", text)
    if not m:
        raise SystemExit(f"tier3_root: {incoming}/DEPLOYED 里没有 40 位的 commit")
    return {"commit": m.group(0), "clean": "dirty=0" in text}


def plan(incoming: Path, root: Path, tier2: Path, expect: dict, support_from_incoming: bool = False, support_dir: Path = None) -> tuple:
    """核对并列出要写的文件（源 → 目标）、metadata 行与来源；不写任何东西。"""
    if support_from_incoming and support_dir is not None:
        raise SystemExit("tier3_root: --support-from-incoming 与 --support-dir 只能给一个")
    sup_src = incoming if support_from_incoming else support_dir   # None：支撑面同网格、地面，取官方 Tier 2 目录
    metas = sorted((incoming / "meta").glob("episode_*.metadata.json"))
    if not metas:
        raise SystemExit(f"tier3_root: {incoming}/meta 下没有 episode_*.metadata.json")
    used = set()

    def check(rel: str, f: Path) -> str:
        if rel not in expect:
            raise SystemExit(f"tier3_root: {rel} 不在哈希表里")
        algo, want = expect[rel]
        got = digest(f, algo)
        if got != want:
            raise SystemExit(f"tier3_root: {rel} 的 {algo} 是 {got}，哈希表里是 {want}")
        used.add(rel)
        return got

    old_rows = {}
    jl = root / "meta" / "episodes_metadata.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                old_rows[int(r["episode_index"])] = r
    dep = deployed(incoming)
    copies, rows, prov = [], dict(old_rows), {}
    for m in metas:
        row = json.loads(m.read_text())
        ep = int(row["episode_index"])
        if m.name != f"episode_{ep:06d}.metadata.json":
            raise SystemExit(f"tier3_root: {m.name} 里的 episode_index 是 {ep}")
        src_pq = incoming / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        if not src_pq.is_file():
            raise SystemExit(f"tier3_root: 缺 {src_pq}")
        mtime = datetime.datetime.fromtimestamp(src_pq.stat().st_mtime, datetime.timezone.utc).isoformat(timespec="seconds")
        info = {"deployed_at_staging": dep["commit"], "deployed_at_staging_clean": dep["clean"], "incoming": str(incoming),
                "source_parquet_mtime_utc": mtime,
                "parquet_sha256": check(f"data/chunk-000/{src_pq.name}", src_pq),
                "metadata_sha256": check(f"meta/{m.name}", m), "tier2_md5": {}}
        dst_pq = root / "data" / "chunk-000" / src_pq.name
        if ep in old_rows and (old_rows[ep] != row or not dst_pq.is_file()
                               or digest(dst_pq, "sha256") != info["parquet_sha256"]):
            raise SystemExit(f"tier3_root: ep{ep} 已在 {root} 里，且元数据或 parquet 与这次的不同；不覆盖")
        if ep not in old_rows:
            copies.append((src_pq, dst_pq))
        for k in LINKED:
            rel = safe_rel(row[k])
            from_inc = sup_src is not None and k == "support_surface"
            src = (sup_src if from_inc else tier2) / rel
            where = ("转来的目录" if support_from_incoming else f"支撑面目录 {support_dir}") if from_inc else "官方 Tier 2 目录"
            if not src.is_file():
                raise SystemExit(f"tier3_root: ep{ep} 的 {k} 在{where}里没有：{src}")
            if from_inc:
                info["support_source"] = "incoming" if support_from_incoming else str(support_dir)
                info["support_sha256"] = check(str(rel), src)
            else:
                info["tier2_md5"][k] = check(str(rel), src)
            dst = root / rel
            if dst.exists() and digest(dst, "sha256") != digest(src, "sha256"):
                raise SystemExit(f"tier3_root: {dst} 已存在且与{'这次取的支撑面' if from_inc else '官方文件'}不同")
            if not dst.exists():
                copies.append((src, dst))
        rows[ep] = row
        prov[str(ep)] = info
    unused = sorted(set(expect) - used)
    if unused:
        raise SystemExit(f"tier3_root: 哈希表里有这次没用上的行：{unused}")
    return copies, rows, prov


def main(argv=None) -> int:
    forbid_tier1()
    ap = argparse.ArgumentParser()
    ap.add_argument("incoming", type=Path)
    ap.add_argument("--expect", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=TIER3)
    ap.add_argument("--tier2", type=Path, default=TIER2)
    ap.add_argument("--support-from-incoming", action="store_true")
    ap.add_argument("--support-dir", type=Path)
    a = ap.parse_args(argv)
    copies, rows, prov = plan(a.incoming, a.root, a.tier2, load_expect(a.expect), a.support_from_incoming, a.support_dir)
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rel = dst.relative_to(a.root)
        md5 = f" md5 {digest(dst, 'md5')}" if rel.parts[0] in LINKED else ""
        print(f"写 {rel} sha256 {digest(dst, 'sha256')}{md5}")
    meta = a.root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "episodes_metadata.jsonl").write_text("".join(json.dumps(rows[ep], ensure_ascii=False) + "\n" for ep in sorted(rows)))
    pj = meta / "provenance.json"
    old = json.loads(pj.read_text()) if pj.exists() else {}
    episodes = {**old.get("episodes", {}), **{ep: dict(info, expect=str(a.expect)) for ep, info in prov.items()}}
    note = ("Tier 3 的轨迹来自同组成员的 CARI4D 重建（不是官方数据）；网格、地面取自官方 Tier 2 目录（规则解读未确认）；"
            "支撑面取自同组成员的交付目录（转来的文件，sha256 核过）" if a.support_from_incoming
            else "Tier 3 的轨迹来自同组成员的 CARI4D 重建（不是官方数据）；网格、地面取自官方 Tier 2 目录（规则解读未确认）；"
            f"支撑面取自 {a.support_dir}（同组成员上一版交付目录里的文件，sha256 核过）" if a.support_dir is not None
            else "Tier 3 的轨迹来自同组成员的 CARI4D 重建（不是官方数据）；网格、地面、支撑面取自官方 Tier 2 目录（规则解读未确认）")
    pj.write_text(json.dumps({"note": note, "episodes": episodes}, ensure_ascii=False, indent=1) + "\n")
    print(f"写 meta/episodes_metadata.jsonl（{len(rows)} 行：{', '.join(f'ep{e}' for e in sorted(rows))}）与 meta/provenance.json")
    for ep in sorted(prov, key=int):
        print(f"ep{ep} 来源：parquet 在对方的修改时间 {prov[ep]['source_parquet_mtime_utc']}，转发时对方的部署版本 {prov[ep]['deployed_at_staging']}"
              f"（干净 {prov[ep]['deployed_at_staging_clean']}），parquet sha256 {prov[ep]['parquet_sha256']}")
    print("TIER3_ROOT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
