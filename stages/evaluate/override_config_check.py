"""核对 eval_playback_override.py 的配置覆盖有没有漏键（只用标准库；训练机宿主机上直接跑，不进容器）。

拿同一条序列、同一配方的两个 checkpoint：一个逐帧训练、一个真实时长训练。把逐帧那个内嵌的配置用 override_config 覆盖成真实时长，
再与真实时长那个内嵌的配置逐键比：除命名类的键（logging.*、output_dir、training 里由 logging 派生的名字）之外应当完全相同。
恒等覆盖只证明“不改键时结果不变”；这一步证明“改成另一种回放时该改的键都改到了”。有意料之外的差异就逐条列出、退出码 1。
用法：python3 stages/evaluate/override_config_check.py <逐帧 policy> <真实时长 policy> <源帧数> <源 fps> [--pos 0.002] [--ori 0.005]
"""
import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_playback_override import flat, override_config  # noqa: E402

NAMING = ("logging.", "output_dir", "training.run_name", "training.experiment_name", "training.wandb_entity", "training.wandb_project")


def embedded(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(json.loads(f.read(n))["__metadata__"]["user.resolved_config_json"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_policy")
    ap.add_argument("real_policy")
    ap.add_argument("src_frames", type=int)
    ap.add_argument("src_fps", type=float)
    ap.add_argument("--pos", type=float, default=0.002)
    ap.add_argument("--ori", type=float, default=0.005)
    a = ap.parse_args()
    over, changes = override_config(embedded(a.frame_policy), "real", a.pos, a.ori, a.src_frames, a.src_fps)
    real = embedded(a.real_policy)
    fo, fr = flat(over), flat(real)
    diffs = {k: (fo.get(k, "<无>"), fr.get(k, "<无>")) for k in sorted(set(fo) | set(fr)) if fo.get(k, "<无>") != fr.get(k, "<无>")}
    naming = {k: v for k, v in diffs.items() if k.startswith(NAMING)}
    other = {k: v for k, v in diffs.items() if k not in naming}
    print(f"覆盖改动的键：{changes}")
    print(f"覆盖后与真实时长 checkpoint 的配置相比：{len(diffs)} 个键不同，其中命名类 {len(naming)} 个")
    for k, (x, y) in other.items():
        print(f"  意外  {k}: 覆盖后 {x!r}，真实时长 checkpoint {y!r}")
    print("RESULT " + ("OK：该改的键都改到了" if not other else "有意料之外的差异"))
    return 0 if not other else 1


if __name__ == "__main__":
    sys.exit(main())
