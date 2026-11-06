"""列出一个 policy checkpoint 评测时会读到的全部输入，并逐个算 sha256（训练机上执行，只用标准库）。

用途：在另一台机器上评测某条 run 的 policy 前，证明两边的输入逐字节相同。两台各跑一次，
diff 两份输出里的 `sha` 行即可。评测器按 checkpoint 内嵌的 resolved config 重建环境，
所以清单从 config 出发：config 里所有 /repo、/workspace、/v2d 开头的路径；参考 parquet 所在的
整个 sequence 目录（网格、材质、URDF）；按加载器规则查找的 support usda
（<motion_root>/whole_body/reconstructed_stage/<seq>{_g1,}_support.usda）。

用法：python3 stages/policy/policy_inputs.py runs/flash_chord/<run>/policy_<N>.safetensors
"""
import hashlib
import json
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FC = REPO / "third_party/video_to_data/robotic_grounding/flash_chord"
MOUNTS = {"/repo": REPO, "/workspace": FC, "/v2d/assets": FC / "../source/robotic_grounding/robotic_grounding/assets"}


def host_path(p: str) -> Path:
    for prefix, root in MOUNTS.items():
        if p == prefix or p.startswith(prefix + "/"):
            return Path(str(root) + p[len(prefix):])
    return Path(p)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    policy = Path(sys.argv[1]).resolve()
    with open(policy, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        md = json.loads(f.read(n))["__metadata__"]
    cfg = json.loads(md["user.resolved_config_json"])
    print(f"policy {policy.relative_to(REPO)}  sha256 {sha256(policy)}")
    for k in ("user.environment_steps", "user.curriculum_stage", "user.voc_scale",
              "user.immediate_first_frame_probability", "user.checkpoint_set_id"):
        print(f"  {k} = {md.get(k)}")

    paths: list[tuple[str, str]] = []

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{key}.{k}" if key else k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{key}[{i}]")
        elif isinstance(x, str) and x.strip("'").startswith(("/repo", "/workspace", "/v2d")):
            paths.append((key, x.strip("'")))

    walk(cfg)
    files: dict[Path, str] = {}
    for key, p in paths:
        hp = host_path(p)
        print(f"cfg {key} = {p}  ({'file' if hp.is_file() else 'dir' if hp.is_dir() else 'MISSING'})")
        if hp.is_file():
            files[hp] = sha256(hp)
        if hp.suffix == ".parquet" and hp.is_file():
            seq_dir = hp.parents[1]                      # .../sequence_id=<seq>/robot_name=g1/data.parquet
            for f in sorted(q for q in seq_dir.rglob("*") if q.is_file()):
                files[f] = sha256(f)
            seq = seq_dir.name.split("=", 1)[1]
            stage = hp.parents[3] / "reconstructed_stage"  # .../whole_body/soma/sequence_id=.../robot_name=g1
            for s in (f"{seq}_g1_support.usda", f"{seq}_support.usda"):
                if (stage / s).is_file():
                    files[stage / s] = sha256(stage / s)
    missing = [p for _, p in paths if not host_path(p).exists()]
    for f in sorted(files):
        print(f"sha {files[f]}  {f.relative_to(REPO) if f.is_relative_to(REPO) else f}")
    print(f"{len(files)} files hashed, {len(missing)} config paths missing")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
