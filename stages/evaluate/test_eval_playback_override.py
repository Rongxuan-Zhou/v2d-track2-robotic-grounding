"""eval_playback_override.override_config 的测试（只用标准库，Mac 上直接跑；暂无契约 ID：诊断工具不在交付链上）。

配置照两条 ep20 满配 checkpoint 里实际存的结构（task / reference / embodiment.config 这几段，外加几个无关键）：
恒等覆盖一个键都不改；逐帧 → 真实时长只改回放开关、num_frames（553 → 921）与两个容差键；真实时长 → 逐帧只改回放开关与 num_frames；
已是目标回放时不改；容差已经更宽时不收窄；motion_speed 不是 1.0 时拒绝；非法回放名拒绝。
换参考（k-fix）：只动 task.parquet（保留引号）、reference.parquet 与按新帧数重算的 num_frames；新路径不是 /repo/runs/motion/…/data.parquet 时拒绝。
用法：python3 stages/evaluate/test_eval_playback_override.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_playback_override import override_config  # noqa: E402

PARQ = "/repo/runs/motion/tier1/whole_body/soma/sequence_id=2026-06-08_16-57-34_rice_cooker_clockwise_04/robot_name=g1/data.parquet"


def ckpt(real: bool) -> dict:
    emb = {"asset_id": "g1_dex3_upstream_main", "asset_sha256": "8c7f", "replace_cylinders_with_capsules": True}
    if real:
        emb.update(frame_position_tolerance=0.002, frame_orientation_tolerance=0.005)
    return {"task": {"motion_speed": 1.0, "source_frame_playback": not real, "motion_start_frame": 0, "motion_end_frame": -1,
                     "parquet": f"'{PARQ}'"},
            "reference": {"num_frames": 921 if real else 553, "parquet": PARQ, "playback_fps": 50.0,
                          "object_articulations": {"entries": [], "version": 1}},
            "embodiment": {"_target_": "flash_chord.embodiments.g1_dex3.G1Dex3", "config": emb},
            "training": {"total_environment_steps": 349999104, "replay": {"capacity": 10000000}},
            "scene": {"world_count": 2048, "support": True}}


NEWP = "/repo/runs/motion/tier3_kfix/whole_body/soma/sequence_id=seq0/robot_name=g1/data.parquet"


def test_parquet() -> None:
    """换参考：只动 task.parquet（保留引号）、reference.parquet 与按新帧数重算的 num_frames；非法路径拒绝。"""
    c = ckpt(False)
    out, ch = override_config(c, "same", 0.002, 0.005, 553, 30.0, parquet=NEWP)
    assert out["task"]["parquet"] == f"'{NEWP}'" and out["reference"]["parquet"] == NEWP, out
    assert set(ch) == {"task.parquet", "reference.parquet"}, ch
    out, ch = override_config(c, "same", 0.002, 0.005, 560, 30.0, parquet=NEWP)
    assert out["reference"]["num_frames"] == 560 and set(ch) == {"task.parquet", "reference.parquet", "reference.num_frames"}, ch
    out, ch = override_config(ckpt(True), "same", 0.002, 0.005, 553, 30.0, parquet=NEWP)
    assert out["reference"]["num_frames"] == 921, out["reference"]["num_frames"]   # 真实时长：(553 − 1) / 30 × 50 + 1
    for bad in ("/tmp/x.parquet", "runs/motion/tier3/data.parquet", "/repo/runs/motion/tier3/x.csv"):
        try:
            override_config(c, "same", 0.002, 0.005, 553, 30.0, parquet=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法的新参考路径没被拒绝：{bad}")
    print("ok  换参考：task.parquet（保留引号）、reference.parquet、num_frames 按新帧数重算；非法路径拒绝")


def main() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  | {detail}"))
        fails += not ok

    for real in (False, True):
        c = ckpt(real)
        out, ch = override_config(c, "same", 0.002, 0.005, 553, 30.0)
        check(f"恒等覆盖（{'真实时长' if real else '逐帧'} checkpoint）：配置不变、没有改动", out == c and ch == {}, str(ch))
    c = ckpt(False)
    out, ch = override_config(c, "real", 0.002, 0.005, 553, 30.0)
    want = {"task.source_frame_playback": [True, False], "reference.num_frames": [553, 921],
            "embodiment.config.frame_position_tolerance": [None, 0.002], "embodiment.config.frame_orientation_tolerance": [None, 0.005]}
    check("逐帧 → 真实时长：只改回放、num_frames 553 → 921、补两个容差键", ch == want, str(ch))
    check("原配置对象没被改", c == ckpt(False))
    c = ckpt(True)
    out, ch = override_config(c, "frame", 0.002, 0.005, 553, 30.0)
    check("真实时长 → 逐帧：只改回放与 num_frames 921 → 553", ch == {"task.source_frame_playback": [False, True], "reference.num_frames": [921, 553]},
          str(ch))
    out, ch = override_config(ckpt(True), "real", 0.002, 0.005, 553, 30.0)
    check("已是真实时长再覆盖成真实时长：不改", ch == {}, str(ch))
    out, ch = override_config(ckpt(False), "frame", 0.002, 0.005, 553, 30.0)
    check("已是逐帧再覆盖成逐帧：不改", ch == {}, str(ch))
    out, ch = override_config(ckpt(True), "real", 0.001, 0.003, 553, 30.0)
    check("容差已经更宽（2 mm / 5e-3）时不收窄", ch == {}, str(ch))
    out, ch = override_config(ckpt(False), "real", 0.002, 0.005, 922, 30.0)
    check("922 帧的序列按真实时长是 1536 帧", ch.get("reference.num_frames") == [553, 1536], str(ch))
    bad = copy.deepcopy(ckpt(False))
    bad["task"]["motion_speed"] = 0.5
    try:
        override_config(bad, "real", 0.002, 0.005, 553, 30.0)
        check("motion_speed 0.5 时拒绝", False, "没有报错")
    except ValueError:
        check("motion_speed 0.5 时拒绝", True)
    try:
        override_config(ckpt(False), "fast", 0.002, 0.005, 553, 30.0)
        check("非法回放名拒绝", False, "没有报错")
    except ValueError:
        check("非法回放名拒绝", True)
    print(f"结果：{'全部通过' if fails == 0 else f'{fails} 项失败'}")
    return 1 if fails else 0


if __name__ == "__main__":
    test_parquet()
    sys.exit(main())
