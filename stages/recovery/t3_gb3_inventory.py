"""列出另一位用户 gb3 交付版（tier3_cari4d_gb_best_dr2s_dyn7）每集的来源与文件指纹（ml40 宿主机上经 stdin 跑，只用标准库，只读）。
计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 的事实一节：每集 gb_best 选的是哪一种 convert，与 f830_best（我们 hm 参考的来源）选的是否相同；
集在交付目录还是 _suspect；parquet、元数据与支撑面的 sha256 前 12 位。
目录自己找：/lab_data/*/v2d/tier3 里不属于本用户的，必须恰好一个（同 tier3_pack_remote.sh）；输出只有相对路径。
用法（Mac，仓库根）：scripts/dev/record t3gb3/inventory_ml40 bash -c 'ssh -o BatchMode=yes ml40 python3 - < stages/recovery/t3_gb3_inventory.py | tools/redact_stdin.sh'
最后一行 T3_GB3_INVENTORY_DONE <集数>。
"""
import getpass
import glob
import hashlib
import json
import os
import sys

VER = "tier3_cari4d_gb_best_dr2s_dyn7"


def sha12(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:12]


def main() -> int:
    me = getpass.getuser()
    roots = [d for d in glob.glob("/lab_data/*/v2d/tier3") if not d.startswith(f"/lab_data/{me}/")]
    if len(roots) != 1:
        print(f"别的用户的 tier3 目录有 {len(roots)} 个，应当恰好一个", file=sys.stderr)
        return 1
    runs = os.path.join(roots[0], "runs")
    sel = {"gb": "tier3_cari4d_gb_best.selection.json", "f830": "tier3_cari4d_f830_best.selection.json"}
    chosen = {}
    for k, f in sel.items():
        with open(os.path.join(runs, f)) as fh:
            chosen[k] = {x["episode_index"]: (x["chosen"], x.get("convert")) for x in json.load(fh)}
    print(f"版本 runs/{VER}；选择记录 runs/{sel['gb']}、runs/{sel['f830']}")
    print("集\tgb_best 选\tconvert\tf830_best 选\t相同\t目录\tparquet\t元数据\t支撑面文件\t支撑面")
    n = 0
    for ep in range(30):
        e = f"{ep:06d}"
        where = None
        for sub in (VER, VER + "_suspect"):
            if os.path.exists(os.path.join(runs, sub, "meta", f"episode_{e}.metadata.json")):
                where = sub
                break
        g, f8 = chosen["gb"].get(ep, ("-", "-")), chosen["f830"].get(ep, ("-", "-"))
        if where is None:
            print(f"ep{ep}\t{g[0]}\t{g[1]}\t{f8[0]}\t{'是' if g[0] == f8[0] else '否'}\t缺\t-\t-\t-\t-")
            continue
        d = os.path.join(runs, where)
        meta = os.path.join(d, "meta", f"episode_{e}.metadata.json")
        with open(meta) as fh:
            sup = json.load(fh).get("support_surface") or ""
        sp = os.path.join(d, sup) if sup else ""
        print(f"ep{ep}\t{g[0]}\t{g[1]}\t{f8[0]}\t{'是' if g[0] == f8[0] else '否'}\t{'交付' if where == VER else '_suspect'}\t"
              f"{sha12(os.path.join(d, 'data', 'chunk-000', f'episode_{e}.parquet'))}\t{sha12(meta)}\t{os.path.basename(sup) or '-'}\t"
              f"{sha12(sp) if sp and os.path.exists(sp) else '缺'}")
        n += 1
    same = sum(1 for ep in range(30) if chosen["gb"].get(ep, ("?",))[0] == chosen["f830"].get(ep, ("!",))[0])
    print(f"gb_best 与 f830_best 选的相同：{same}/30")
    print(f"T3_GB3_INVENTORY_DONE {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
