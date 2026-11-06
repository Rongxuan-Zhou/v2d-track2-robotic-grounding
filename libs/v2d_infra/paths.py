"""仓内路径解析。脚本不再写死绝对路径。

动机：数据集与上游 toolkit 从 ~/v2d/* 搬进仓库后，写死路径的脚本会静默地
找不到文件。路径从仓库根推导，换机器、换布局都不用改脚本。

档位隔离（见 CLAUDE.md）将来落在这一层：tier() 应当只返回调用方声明的那一档，
让 Tier 2/3 的管线拿不到 Tier 1 的干净轨迹。当前仅做路径解析，未加强制。
"""
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "challenge"
TRACK2 = DATA / "track_2"
TIER1 = TRACK2 / "tier_1_multiview_caption"
TIER2 = TRACK2 / "tier_2_synthetic_noise"
RESULTS = REPO / "results"
RUNS = REPO / "runs"          # 训练与缓存产物，gitignored，rsync 受 P 规则保护
# Tier 3 只发 exo 视频，没有官方的轨迹目录。这里不是官方数据：是从视频重建后拼成的档目录（与 Tier 2 同 schema），
# 由 stages/recovery/tier3_root.py 写（计划 agent-doc/plan-t3-pilot-2026-10-09.md）。
# 重建不止一版（同组成员 10/9 晚另写了地面、焦距校正版）。环境变量 V2D_T3_SET 选用哪一版拼成的档目录（计划
# agent-doc/plan-t3-pilot-v2-2026-10-10.md）：不设时是 10/9 试跑那版 runs/tier3/root；设成 <名>（只许小写字母与数字）时是
# runs/tier3/root_<名>，runs/motion_src、runs/motion 下的目录名也跟着变（tier_dirname）。只影响 Tier 3，旧版的文件一个不动。
T3_SET = os.environ.get("V2D_T3_SET", "")
if T3_SET and not re.fullmatch(r"[a-z0-9]+", T3_SET):
    raise ValueError(f"paths: V2D_T3_SET 只许小写字母与数字：{T3_SET!r}")
TIER3 = RUNS / "tier3" / (f"root_{T3_SET}" if T3_SET else "root")
UPSTREAM = REPO / "third_party" / "video_to_data"
SOMA_ASSETS = (UPSTREAM / "robotic_grounding/source/robotic_grounding"
               / "robotic_grounding/assets/body_models/soma")

def tier(n: int) -> Path:
    """按档位号取数据根。不要跨档合并——三档的 episode 与帧号刻意对齐，
    用途是逐帧比较，不是互相补数据。"""
    return {1: TIER1, 2: TIER2, 3: TIER3}[n]


def tier_dirname(n: int) -> str:
    """runs/motion_src、runs/motion 下这一档的目录名：tier<N>；Tier 3 设了 V2D_T3_SET 时是 tier3_<名>。"""
    if n not in (1, 2, 3):
        raise ValueError(f"paths: 没有第 {n} 档")
    return f"tier3_{T3_SET}" if n == 3 and T3_SET else f"tier{n}"
