"""day2_compare.py 的合成测试（只用标准库，Mac 上直接跑；暂无契约 ID：判定工具不在交付链上）。

每个格子造一份只含来历行与 summary_json 的假结果文件。查：真实时长 / 逐帧各自全面占优、两臂在逐帧评测下都触底时真实时长仍赢
（复审第 1 条）、真实时长 policy 按逐帧评时崩了则判逐帧（对冲起作用）、混合无偏好、SP 与 MP 合成一项、地板（≤ 3 个世界）不算票与
票数不够不下结论、步数不对 / 来历不干净 / 不是同一 checkpoint / 不是同一 parquet 判无效、不稳标记、边缘判定、种子噪声抬高阈值、
B 要两颗种子都达标。
用法：python3 stages/evaluate/test_day2_compare.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = 669
FR = {"frame": SRC, "real": int(round((SRC - 1) / 30.0 * 50.0)) + 1}
PQ = "/repo/runs/motion/tier1/whole_body/soma/sequence_id=x/robot_name=g1/data.parquet"


def write(tmp: Path, name: str, summary: dict, source: str = "abc123def456") -> str:
    p = tmp / f"{name}.txt"
    p.write_text(f"# command  : fake\n# source   : {source}\n# ---\n  summary_json {json.dumps(summary)}\n")
    return str(p)


def ev(add, sp, mp, mppe, e, sha, chord=0.5, steps=None, worlds=4096, parquet=PQ):
    return {"add_auc": add, "spider_sr_uncentered": sp, "maniptrans_sr": mp, "mppe_cm": mppe, "chord_sr": chord,
            "step_count": FR[e] if steps is None else steps, "world_count": worlds, "checkpoint_sha256": sha, "parquet": parquet}


def seq(tmp: Path, key: str, cells: dict, counts=True, bars=(0.47, 0.47), source="abc123def456", trend=None, **bad) -> dict:
    """cells: {(训练, 评测): (add, sp, mp, mppe)}"""
    b = {pb: write(tmp, f"{key}_bar_{pb}", {"add_auc": bars[i], "frames": FR[pb], "parquet": PQ}) for i, pb in enumerate(("frame", "real"))}
    evals = {}
    for (x, e), (add, sp, mp, mppe) in cells.items():
        sha = f"sha-{key}-{x}" + ("-other" if bad.get("sha") and (x, e) == ("real", "real") else "")
        steps = 999 if bad.get("steps") and (x, e) == ("frame", "frame") else None
        parquet = PQ.replace("=x/", "=y/") if bad.get("parquet") and (x, e) == ("real", "frame") else PQ
        evals[f"{x}@{e}"] = write(tmp, f"{key}_{x}_at_{e}", ev(add, sp, mp, mppe, e, sha, steps=steps, parquet=parquet),
                                  source=source if (x, e) == ("real", "real") else "abc123def456")
    s = {"key": key, "src_frames": SRC, "counts": counts, "bars": b, "evals": evals}
    if trend:
        s["trend"] = {x: [write(tmp, f"{key}_trend_{x}_{i}", ev(a, 0, 0, 3, x, "t")) for i, a in enumerate(trend[x])] for x in trend}
    return s


C = lambda rr, rf, ff, fr: {("real", "real"): rr, ("real", "frame"): rf, ("frame", "frame"): ff, ("frame", "real"): fr}  # noqa: E731
GOOD_R = C((0.60, 0.9, 0.9, 3.0), (0.58, 0.8, 0.8, 3.2), (0.43, 0.0, 0.0, 5.2), (0.44, 0.1, 0.1, 5.0))
GOOD_F = C((0.43, 0.0, 0.0, 5.2), (0.40, 0.0, 0.0, 5.5), (0.60, 0.9, 0.9, 3.0), (0.59, 0.85, 0.8, 3.1))
EVEN = C(*[(0.50, 0.5, 0.5, 4.0)] * 4)
FLOOR = C(*[(0.30, 1 / 4096, 2 / 4096, 9.0)] * 4)
# 复审第 1 条：两臂按逐帧评都触底（打平），真实时长训练按真实时长评远好 → 应判 R
BOTTOM = C((0.55, 0.85, 0.9, 3.9), (0.43, 0.0, 0.0, 5.3), (0.43, 0.0, 0.0, 5.2), (0.44, 0.1, 0.1, 5.0))
# 真实时长训练的 policy 按逐帧评时崩了（最大损失在它那边）→ 应判 F
TRAP = C((0.60, 0.9, 0.9, 3.0), (0.30, 0.0, 0.0, 8.0), (0.50, 0.5, 0.5, 4.0), (0.49, 0.45, 0.45, 4.2))
# SP 与 MP 分歧：SP 差 0.5、MP 只差 0.1 → SR 算平；gap 与 MPPE 都平 → 无偏好
SPLIT = C((0.50, 0.9, 0.6, 4.0), (0.50, 0.9, 0.6, 4.0), (0.50, 0.4, 0.5, 4.0), (0.50, 0.4, 0.5, 4.0))


def bset(tmp: Path, s43: tuple) -> dict:
    vals = {"b_rt": (0.53, 0.8, 0.85, 4.0, 0.8), "ctrl_rt": (0.547, 0.844, 0.887, 3.9, 0.81), "b_frame": (0.40, 0.0, 0.0, 5.5, 0.5),
            "ctrl_frame": (0.431, 0.0, 0.0005, 5.2, 0.79), "b_rt_s43": s43}
    return {n: {"eval": write(tmp, f"b_{n}_{id(s43)}", ev(a, sp, mp, pe, "real" if "rt" in n else "frame", "b", chord=ch)),
                "bar": write(tmp, f"b_bar_{n}_{id(s43)}", {"add_auc": 0.47, "frames": FR["real"], "parquet": PQ})}
            for n, (a, sp, mp, pe, ch) in vals.items()}


def run(tmp: Path, sequences: list, b=None):
    spec = tmp / "spec.json"
    spec.write_text(json.dumps({"sequences": sequences, **({"b": b} if b else {})}))
    p = subprocess.run([sys.executable, str(HERE / "day2_compare.py"), str(spec)], capture_output=True, text=True, check=False)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  | {detail[-900:]}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        rc, out = run(t, [seq(t, f"r{i}", GOOD_R) for i in range(3)])
        check("三条都是真实时长全面占优 → 训练用真实时长更稳", rc == 0 and "Q1 判定：训练用真实时长更稳" in out, out)
        rc, out = run(t, [seq(t, f"f{i}", GOOD_F) for i in range(3)])
        check("三条都是逐帧全面占优 → 训练用逐帧更稳", "Q1 判定：训练用逐帧更稳" in out, out)
        rc, out = run(t, [seq(t, f"b{i}", BOTTOM) for i in range(3)])
        check("两臂按逐帧评都触底、真实时长按真实时长评远好 → 真实时长（复审第 1 条的情形）", "Q1 判定：训练用真实时长更稳" in out, out)
        rc, out = run(t, [seq(t, f"t{i}", TRAP) for i in range(3)])
        check("真实时长 policy 按逐帧评时崩了 → 最大损失在它那边，判逐帧", "Q1 判定：训练用逐帧更稳" in out, out)
        rc, out = run(t, [seq(t, "s1", SPLIT), seq(t, "s2", SPLIT)])
        check("SP 赢 0.5 但 MP 只差 0.1 → SR 算平、序列无偏好", "== s1（计票）：无偏好" in out and "sr " in out, out)
        rc, out = run(t, [seq(t, "m1", GOOD_R), seq(t, "m2", GOOD_F), seq(t, "m3", EVEN)])
        check("一条 R、一条 F、一条平 → 无明确偏好", "Q1 判定：无明确偏好" in out, out)
        rc, out = run(t, [seq(t, "a1", GOOD_R), seq(t, "a2", FLOOR), seq(t, "a3", FLOOR)])
        check("两条地板（≤ 3 个世界）不算票、只剩 1 条 → 不下结论", "== a2（计票）：地板" in out and "Q1 判定：不下结论" in out, out)
        rc, out = run(t, [seq(t, "v1", GOOD_R, steps=True), seq(t, "v2", GOOD_R), seq(t, "v3", GOOD_R)])
        check("步数不对的序列判无效、不计票，其余两条照判", "== v1（计票）：无效" in out and "step_count 999" in out
              and "Q1 判定：训练用真实时长更稳" in out, out)
        rc, out = run(t, [seq(t, "d1", GOOD_R, source="abc-dirty"), seq(t, "d2", GOOD_R, sha=True), seq(t, "d3", GOOD_R, parquet=True),
                          seq(t, "d4", GOOD_R)])
        check("来历 -dirty、不是同一 checkpoint、不是同一 parquet → 无效，只剩 1 条不下结论",
              "== d1（计票）：无效" in out and "不是同一个 checkpoint" in out and "不是同一个 parquet" in out and "不下结论" in out, out)
        rc, out = run(t, [seq(t, "e20", GOOD_R, counts=False), seq(t, "x1", GOOD_F), seq(t, "x2", GOOD_F)])
        check("不计票的 ep20 不影响判定", "e20（只作参考）：R" in out and "Q1 判定：训练用逐帧更稳（计票 2 条" in out, out)
        rc, out = run(t, [seq(t, "g1", GOOD_R, trend={"real": [0.50, 0.62, 0.60]}), seq(t, "g2", GOOD_R), seq(t, "g3", GOOD_R)])
        check("中间 checkpoint 偏离两端连线 > 0.05 → 标“不稳”，不改判定", "真实时长训练不稳" in out and "训练用真实时长更稳" in out, out)
        marg = dict(GOOD_R)
        marg[("frame", "frame")] = (0.475, 0.0, 2 / 4096, 5.0)
        rc, out = run(t, [seq(t, "k1", marg), seq(t, "k2", GOOD_R)])
        check("MP-SR 只靠 2 个世界、ADD-AUC 贴着基线 → 标边缘", "逐帧训练 @ 逐帧评测" in out and "边缘" in out, out)
        rc, out = run(t, [seq(t, f"n{i}", GOOD_R) for i in range(3)], b=bset(t, (0.33, 0.8, 0.85, 6.5, 0.8)))
        check("种子噪声 gap −0.2、MPPE +2.5 → 这两项阈值抬高成平，只剩 SR 赢 → 无偏好；B 的 s43 不达标 → 不可用",
              "Q1 判定：无明确偏好" in out and "（已并入 Q1 的阈值）" in out and "mf4090 400 万 → 不可用" in out, out)
        rc, out = run(t, [seq(t, f"p{i}", GOOD_R) for i in range(3)], b=bset(t, (0.525, 0.79, 0.84, 4.1, 0.8)))
        check("两颗种子都达标 → mf4090 可用；噪声小，Q1 仍判真实时长", "mf4090 400 万 → 可用" in out and "训练用真实时长更稳" in out, out)
    print(f"结果：{'全部通过' if fails == 0 else f'{fails} 项失败'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
