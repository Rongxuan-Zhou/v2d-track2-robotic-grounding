"""一份逐世界 dump（fc_eval_dump.sh 写的 eval_<步数>_w<世界数>.npz）里物体平均位置误差的分布：多大比例的世界把物体带离参考很远。
只用标准库（自己解 npz 里的 .npy），ml40 宿主机没有 numpy 也能跑。计划 agent-doc/plan-ep25-sweep-seeds-2026-10-18.md 第 2.3 节（只报）。
读 mp_position_m（每个世界整段的物体平均位置误差，米）与 sp_success、mp_success。ep25 的量级：瓶子一直不动约 0.45 m，掉到地上的约 1.1 m
（t2-ep25-w2048-cap4m-mf4090 的导出 1.157 m，results/t1diag/export_world0_mf4090.txt）。
用法：python3 stages/evaluate/dump_knock.py <npz> [阈值 m，默认 0.8]   最后一行 DUMP_KNOCK_DONE
"""

import ast
import struct
import sys
import zipfile
from array import array

CODES = {"<f8": "d", "<f4": "f", "|b1": "B", "|u1": "B", "<i8": "q"}


def read_npy(zf: zipfile.ZipFile, name: str) -> list:
    with zf.open(name) as f:
        if f.read(6) != b"\x93NUMPY":
            raise SystemExit(f"{name} 不是 .npy")
        major = f.read(2)[0]
        hlen = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        head = ast.literal_eval(f.read(hlen).decode("latin1"))
        data = f.read()
    code = CODES.get(head["descr"])
    if code is None or head["fortran_order"] or len(head["shape"]) != 1:
        raise SystemExit(f"{name}：不支持的格式 {head}")
    if sys.byteorder != "little":
        raise SystemExit("只支持小端机器")
    a = array(code)
    a.frombytes(data[: head["shape"][0] * a.itemsize])
    if len(a) != head["shape"][0]:
        raise SystemExit(f"{name}：数据长度不对")
    return list(a)


def main():
    if len(sys.argv) not in (2, 3):
        print("用法：dump_knock.py <npz> [阈值 m]", file=sys.stderr)
        sys.exit(2)
    thr = float(sys.argv[2]) if len(sys.argv) == 3 else 0.8
    with zipfile.ZipFile(sys.argv[1]) as zf:
        pos = read_npy(zf, "mp_position_m.npy")
        sp = read_npy(zf, "sp_success.npy")
        mp = read_npy(zf, "mp_success.npy")
    n = len(pos)
    if n == 0 or len(sp) != n or len(mp) != n:
        raise SystemExit(f"世界数对不上：{n} {len(sp)} {len(mp)}")
    s = sorted(pos)
    q = lambda f: s[min(n - 1, int(f * n))]  # noqa: E731
    print(f"{sys.argv[1]}：{n} 个世界；物体平均位置误差 中位 {q(0.5):.3f} m，四分位 {q(0.25):.3f} / {q(0.75):.3f} m，最小 {s[0]:.3f}、最大 {s[-1]:.3f} m")
    print(f"平均位置误差超过 {thr:g} m 的世界 {sum(1 for x in pos if x > thr) / n:.4f}；SP 成功 {sum(1 for x in sp if x) / n:.4f}，MP 成功 {sum(1 for x in mp if x) / n:.4f}")
    print("DUMP_KNOCK_DONE")


if __name__ == "__main__":
    main()
