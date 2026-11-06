"""对比两台机器上 tools/motion_inputs_remote.sh 的记录：一组要在“训练机”上训的序列，训练机上的参考 parquet 是否与基线 run 训练时用的同一份。
计划 agent-doc/plan-next-training-2026-10-12.md 第 2 版第 0 步第 7 条（审查 m6）。只用标准库，Mac 上跑。

两份记录都用同一组 <序列>=<基线 run> 参数生成：基线 run 在哪台，哪台的 run_parquet_sha256 列就有值（另一台记 -）。
对每条要核的序列：基线值取两份里不是 - 的那一个（两份都有值且不同、或都没有，算不过）；训练机上的 parquet_sha256 必须等于它；
训练机上的 v2d_meta 必须是 tier=<档>,ep=<给定的 episode>；两台的支撑面文件 md5（support_md5）必须相同（代码审查 m2：支撑面开着的集
训练时会加载它）。逐条打印，最后一行 INPUTS_OK 或 INPUTS_BAD（退出码 0 / 1）。
用法：python3 tools/compare_motion_inputs.py --train <训练机的记录> --other <另一台的记录> --tier <档> --expect <序列>=<episode>…
"""
import argparse
import sys


def rows(path: str) -> dict:
    out = {}
    head = None
    for line in open(path, encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        if c[0] == "seq":
            head = c
            continue
        if head and len(c) == len(head):
            out[c[0]] = dict(zip(head, c))
    if head is None:
        raise SystemExit(f"{path} 里没有表头 seq…")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--expect", nargs="+", required=True, help="<序列>=<episode>")
    a = ap.parse_args(argv)
    tr, ot = rows(a.train), rows(a.other)
    bad = 0
    for kv in a.expect:
        s, ep = kv.rsplit("=", 1)
        t, o = tr.get(s), ot.get(s)
        if t is None or o is None:
            print(f"BAD  {s}：两份记录里缺这一行")
            bad += 1
            continue
        refs = {x for x in (t["run_parquet_sha256"], o["run_parquet_sha256"]) if x != "-"}
        if len(refs) != 1:
            print(f"BAD  {s}：基线 run 记下的 parquet sha256 不是恰好一个（{sorted(refs) or '无'}）")
            bad += 1
            continue
        ref = refs.pop()
        meta_ok = t["v2d_meta"] == f"tier={a.tier},ep={ep}"
        sup_ok = t["support_md5"] == o["support_md5"]
        ok = t["parquet_sha256"] == ref and meta_ok and sup_ok
        bad += not ok
        print(f"{'OK  ' if ok else 'BAD '} {s}：训练机 {t['parquet_sha256'][:16]}，基线 {ref[:16]}，v2d_meta {t['v2d_meta']}（应为 tier={a.tier},ep={ep}），"
              f"支撑面 md5 两台{'相同' if sup_ok else '不同'}（{t['support_md5'][:12]}），基线 run {t['run'] if t['run_parquet_sha256'] != '-' else o['run']}")
    print("INPUTS_OK" if bad == 0 else f"INPUTS_BAD（{bad} 条不过）")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
