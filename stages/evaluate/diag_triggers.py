"""按事先登记的数值条件，把诊断报告翻译成“结果 → 下一步”（只读 results/ 下的文本，只用标准库）。

条件（agent-doc/plan-day1b-2026-09-27.md 第 3 版 B3，跑之前定下）：
  T1 非操作段失分为主：主 policy 第 1、3 段（操作窗口之外）占全程 ADD-AUC 差 ≥ 50% → 查接近与抓取、reset 与 VOC；
     对照 policy 也 ≥ 50% → 归到配方层（门槛以静止段为主）
  T2 滞后：主 policy 中位净转动比 < 第 420 帧不越门槛所需比值 → 做回放速度对照（1.67 倍速 vs 真实时长）
  T3 手没跟上：主 policy 操作段（第 2 段）palm 误差中位 ≥ 7.5 cm（门槛 15 cm 的一半）→ 查策略
  T4 参考手离物体远：主序列参考接触帧上 palm 到物体距离中位（各接触手取最小）比对照序列大 ≥ 2 cm → 查重定向管线
只有全部报告的 VALIDITY 为 PASS 才判定；否则只列数字，退出码 3。

用法：python3 stages/evaluate/diag_triggers.py --main <主 policy 报告> --control <对照报告> [--extra <其他报告>...]
"""
import argparse
import json
import sys
from pathlib import Path


def summary(path: str) -> dict:
    for line in Path(path).read_text().splitlines():
        if line.strip().startswith("summary_json "):
            return json.loads(line.strip()[len("summary_json "):])
    raise SystemExit(f"{path}: 没有 summary_json 行")


def closest_contact(s: dict):  # 返回 float 或 None（不写 X | None：Mac 自带的 python3 太旧）
    d = s.get("contact_palm_mesh_dist_cm") or {}
    return min(d.values()) if d else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    m, c = summary(a.main), summary(a.control)
    extras = {p: summary(p) for p in a.extra}
    valid = m["valid"] and c["valid"] and all(e["valid"] for e in extras.values())
    for name, s in [("主", m), ("对照", c)] + [(p, e) for p, e in extras.items()]:
        print(f"{name}：VALIDITY {'PASS' if s['valid'] else 'FAIL'}；ADD-AUC {s['add_auc']:.4f} vs 不动 {s['static_add_auc']:.4f}；"
              f"非操作段占差 {s['non_manipulation_gap_share']:.0%}；中位净转比 {s['median_net_ratio']:.2f}（需 {s['required_ratio']:.2f}）；"
              f"palm 误差中位（分段）{[round(x, 1) for x in s['palm_error_median_cm']]} cm；接触帧距离 {s['contact_palm_mesh_dist_cm']}")
    t1 = m["non_manipulation_gap_share"] >= 0.5
    t1c = c["non_manipulation_gap_share"] >= 0.5
    t2 = m["median_net_ratio"] < m["required_ratio"]
    t3 = m["palm_error_median_cm"][1] >= 7.5
    dm, dc = closest_contact(m), closest_contact(c)
    t4 = dm is not None and dc is not None and dm - dc >= 2.0
    rows = [
        ("T1 非操作段失分为主", t1, "查接近与抓取阶段、reset 与 VOC 设置" + ("；对照也如此 → 归到配方层" if t1 and t1c else "")),
        ("T2 滞后（中位净转比 < 所需）", t2, "做回放速度对照（1.67 倍速 vs 真实时长）"),
        ("T3 手没跟上（操作段 palm 误差 ≥ 7.5 cm）", t3, "查策略"),
        ("T4 参考手离物体远（比对照远 ≥ 2 cm）", t4, f"查重定向管线（主 {dm} cm，对照 {dc} cm）"),
    ]
    if not valid:
        print("有报告的闸没过：只列数字，不判定")
        return 3
    for name, hit, action in rows:
        print(f"{'触发' if hit else '未触发'}  {name}" + (f" → {action}" if hit else ""))
    if not any(hit for _, hit, _ in rows):
        print("四个条件都没触发：事先没有对应的下一步，如实报告")
    return 0


if __name__ == "__main__":
    sys.exit(main())
