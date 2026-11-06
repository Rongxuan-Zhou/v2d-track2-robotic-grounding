"""列出提交容器的 90 个 checkpoint 在评测时要读的 /repo 文件，并在给定的仓库副本里核对、可选地复制出来（标准库，训练机宿主机可跑）。

起因：组委会 11/05 回信，Tier 1 checkpoint 载入时找不到 /repo/runs/motion 下的参考运动，镜像里没有这些文件。
评测器（FlashCHORD 8c778f9，evaluation/harness.py 的 _resolve_parquet）按 checkpoint 里 resolved config 的 task.parquet 绝对路径读参考；
parquet 的 object_mesh_paths 等列又按绝对路径指向物体网格（textured_mesh.obj / .urdf），可能在另一个版本目录下（例如 tier1_c10 指向 tier1）。
场景开了支撑面（scene.support）时，assets/registry.py 的 support_usda_for_parquet 按 parquet 路径找
<版本>/whole_body/reconstructed_stage/<序列>_<机器人>_support.usda，再找 <序列>_support.usda，找不到就报错（11/05 在 mf4090 实测时撞到）。
所以文件集合 = 每个 checkpoint 的 task.parquet，parquet 里出现的每个 /repo/runs/motion/... 文件所在目录的全部文件，
加上这两个支撑面候选里存在的那些。路径里有符号链接时评测器会先 resolve，这里遇到符号链接就报出来（symlink 行）。

用法：python3 motion_closure.py <weights 目录 | parquet 路径清单.txt> <仓库副本根，对应 /repo> [--copy <输出根>]
     （没有 weights 的机器上，用另一台机器 --list 打出的清单代替：python3 motion_closure.py <weights> --list）
输出（stdout，TSV）：状态（ok / missing）、/repo 下的路径、大小、md5、引用它的 checkpoint 数。--copy 时把 ok 的文件按原路径复制到 <输出根>/ 下。
退出码：有 missing 时为 1。
"""
import hashlib
import json
import re
import shutil
import struct
import sys
from collections import defaultdict
from pathlib import Path

MOTION_REF = re.compile(rb"/repo/runs/motion/[A-Za-z0-9_=./+-]+?\.(?:obj|urdf|mtl|png|jpg|jpeg|stl|parquet)")


def checkpoint_parquet(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))["__metadata__"]
    return json.loads(meta["user.resolved_config_json"])["task"]["parquet"]


def support_candidates(parquet):
    path = Path(parquet)
    seq_dir = next((q for q in path.parents if q.name.startswith("sequence_id=")), None)
    if seq_dir is None:
        return []
    robot = next((q.name.split("=", 1)[1] for q in path.parents if q.name.startswith("robot_name=")), None)
    seq = seq_dir.name.split("=", 1)[1]
    stage = seq_dir.parent.parent / "reconstructed_stage"
    names = ([f"{seq}_{robot}_support.usda"] if robot else []) + [f"{seq}_support.usda"]
    return [str(stage / n) for n in names]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    args = sys.argv[1:]
    copy_to = None
    if "--copy" in args:
        i = args.index("--copy")
        copy_to = Path(args[i + 1])
        del args[i:i + 2]
    weights = Path(args[0])
    users = defaultdict(set)
    if weights.is_file():
        for line in weights.read_text().splitlines():
            parquet, _, who = line.partition("\t")
            users[parquet] |= set(who.split(",")) if who else {"?"}
    else:
        for ckpt in sorted(weights.glob("tier*/*.safetensors")):
            users[checkpoint_parquet(ckpt)].add(str(ckpt.relative_to(weights)))
    if args[1] == "--list":
        for parquet in sorted(users):
            print(parquet + "\t" + ",".join(sorted(users[parquet])))
        return 0
    root = Path(args[1])
    needed = defaultdict(set)
    for parquet, who in list(users.items()):
        needed[parquet] |= who
        local = root / parquet[len("/repo/"):]
        for cand in support_candidates(parquet):
            if (root / cand[len("/repo/"):]).is_file():
                needed[cand] |= who
        if not local.is_file():
            continue
        if local.resolve() != local.absolute():
            print(f"symlink\t{parquet}\t-\t-\t{len(who)}")
        for ref in set(MOTION_REF.findall(local.read_bytes())):
            ref_dir = Path(ref.decode())
            if ref_dir.suffix == ".parquet":
                continue
            local_dir = root / str(ref_dir.parent)[len("/repo/"):]
            needed[str(ref_dir)] |= who
            if local_dir.is_dir():
                for f in local_dir.iterdir():
                    if f.is_file():
                        needed["/repo/" + str(f.relative_to(root))] |= who
    missing = 0
    for p in sorted(needed):
        local = root / p[len("/repo/"):]
        if not local.is_file():
            missing += 1
            print(f"missing\t{p}\t-\t-\t{len(needed[p])}")
            continue
        print(f"ok\t{p}\t{local.stat().st_size}\t{md5(local)}\t{len(needed[p])}")
        if copy_to:
            dst = copy_to / p[len("/repo/"):]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, dst)
    print(f"# checkpoints {sum(len(v) for v in users.values())}, parquets {len(users)}, files {len(needed)}, missing {missing}",
          file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
