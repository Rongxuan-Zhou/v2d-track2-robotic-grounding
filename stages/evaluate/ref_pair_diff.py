"""两份重定向参考（motion_v1 parquet，单行按帧存）逐帧比（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只读两份参考，不读真值）。
用途：k-fix 方案 4 第一批里，ep0 的现有策略对旧参考（tier3_v2）4096 世界 MP-SR 0.965，对新参考（tier3_kfix）0.000、骨盆失败 0.9998。
包装层已核过恒等（results/t3kfix/reexport_checks_mf4090.txt），差别只在参考本身。这里量两份参考差在哪：
  机器人根（位置、偏航）、物体根位置、物体相对机器人根的位置（策略看到的是这类相对量）、各末端到物体根的距离、关节角，
  另报第 0 帧的根高、物体高与 coord_frame、网格路径是否相同。逐帧差报中位与最大。
用法（容器内）：python /repo/stages/evaluate/ref_pair_diff.py <旧参考 parquet> <新参考 parquet>；最后一行 REF_PAIR_DIFF_DONE。
"""
import sys

import numpy as np
import pyarrow.parquet as pq


def load(p: str) -> dict:
    r = pq.read_table(p).to_pylist()[0]
    return {
        "fps": r["fps"], "frame": r["coord_frame"], "mesh": r["object_mesh_paths"], "ee": r["ee_link_names"],
        "root": np.asarray(r["robot_root_position"], float), "wxyz": np.asarray(r["robot_root_wxyz"], float),
        "q": np.asarray(r["robot_joint_positions"], float), "obj": np.asarray(r["object_root_position"], float),
        "eep": np.asarray([[p7[:3] for p7 in fr] for fr in r["ee_pose_w"]], float),
        "contact": contact_by_frame(np.asarray(r["hand_contact_active"], float), len(r["robot_root_position"])),
    }


def contact_by_frame(c: np.ndarray, frames: int) -> np.ndarray:
    # hand_contact_active 按手存（形状 手数 × 帧数）；转成 帧数 × 手数（10/16 第一版按帧读错了方向，ep0 那份记录的接触行作废）
    return c.T if c.shape[0] != frames else c


def yaw(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = wxyz.T
    return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def stat(name: str, d: np.ndarray, unit: str = "m") -> None:
    print(f"  {name}：中位 {np.median(d):.4f} {unit}、最大 {np.max(d):.4f} {unit}（第 {int(np.argmax(d))} 帧）")


def main() -> int:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    n = min(len(a["root"]), len(b["root"]))
    print(f"帧数 {len(a['root'])} / {len(b['root'])}，fps {a['fps']} / {b['fps']}，coord_frame {a['frame']} / {b['frame']}，"
          f"网格路径{'相同' if a['mesh'] == b['mesh'] else '不同'}，末端 {a['ee']}")
    print(f"第 0 帧：根 {np.round(a['root'][0], 3)} → {np.round(b['root'][0], 3)}；物体 {np.round(a['obj'][0], 3)} → {np.round(b['obj'][0], 3)}")
    stat("机器人根位置差", np.linalg.norm(a["root"][:n] - b["root"][:n], axis=1))
    dy = np.abs((yaw(a["wxyz"][:n]) - yaw(b["wxyz"][:n]) + 180) % 360 - 180)
    stat("机器人根偏航差", dy, "°")
    stat("物体根位置差", np.linalg.norm(a["obj"][:n] - b["obj"][:n], axis=1))
    rel_a, rel_b = a["obj"][:n] - a["root"][:n], b["obj"][:n] - b["root"][:n]
    stat("物体相对机器人根的位置差", np.linalg.norm(rel_a - rel_b, axis=1))
    for i, name in enumerate(a["ee"]):
        da = np.linalg.norm(a["eep"][:n, i] - a["obj"][:n], axis=1)
        db = np.linalg.norm(b["eep"][:n, i] - b["obj"][:n], axis=1)
        stat(f"{name} 到物体根的距离之差", np.abs(da - db))
    stat("关节角差（每帧取各关节最大）", np.abs(a["q"][:n] - b["q"][:n]).max(axis=1), "rad")
    ca, cb = a["contact"][:n] > 0.5, b["contact"][:n] > 0.5
    print(f"  接触帧（各手）：{ca.sum(axis=0).tolist()} → {cb.sum(axis=0).tolist()}；首个接触帧 "
          f"{[int(np.argmax(c)) if c.any() else None for c in ca.T]} → {[int(np.argmax(c)) if c.any() else None for c in cb.T]}")
    print("REF_PAIR_DIFF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
