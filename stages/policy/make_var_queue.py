"""生成变体 A 清单的期望表、冒烟队列、正式队列与导出清单，一次一段。只用标准库。
  t2x  Tier 2 铺开：其余 24 条 episode 的变体 A seed 42，ml40，buffer 1000 万（计划 agent-doc/plan-t2-scaleout-2026-10-09.md）。
       分两段（Fable 5.1 审查 m5）：t2xa 是基线失败:骨盆低于 0.65 的 15 条（与试验同一种情形），t2xb 是几乎每个世界都摔的 9 条（外推）
  t3p  Tier 3 试跑：ep8、ep24 的变体 A seed 42，mf4090，buffer 400 万，参考换成 runs/motion/tier3 下从视频重建的那份
       （计划 agent-doc/plan-t3-pilot-2026-10-09.md）；一段 t3p
  t3p2 Tier 3 试跑第二轮：同两集，参考换成 runs/motion/tier3_v2 下同组成员校正版重建的那份，run 名带 -v2；其余同 t3p
       （计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md）；一段 t3p2
  t3p3 Tier 3 校正版其余 10 集（P2′ 过门槛的 ep0、15、17、18、20、21、22、23、26、29）：参考同 t3p2，run 名同 t3p2 的写法；
       基线取 Tier 2 导出清单里这条 episode 现在交的 run（同 t2x；ep15、17、21、23 的基线在 mf4090 上）。
       分两段（计划 agent-doc/plan-t3-v2-train10-2026-10-10.md）：t3p3a 是现有磁盘放得下的 4 集（有支撑面的 ep20 与没有手接触的 ep29 在前），
       t3p3b 是归档腾出空间后的 6 集（ep17 放最后：Tier 2 变体 A 在 ep17 上每个世界都失败）
  t1x  Tier 1 变体 A 补候选，mf4090，buffer 400 万（计划 agent-doc/plan-next-training-2026-10-12.md 第 2 版第 3 节 A）：
       段 t1xa 是 10/9 逐条选后 1 世界 MP-SR 仍失败、4096 世界失败原因里骨盆比例低于 0.65 的 7 条（按骨盆比例从低到高）
  t1y  同上，ml40，buffer 1000 万（同计划第 3 节 C）：段 t1ya 是骨盆比例 0.969–1.000、不含方盒 ep1、ep28 的 6 条（同样排序，同为 1.000 的按 episode 号）
  t1xb Tier 1 追基线，mf4090，buffer 400 万（计划 agent-doc/plan-t1-chase-2026-10-13.md）：t1xa 没开的 5 条换个顺序，SP-SR 失败的排前面：
       ep3、19、13、10、14；run 名、期望表行与 t1xa 的相同（冒烟已 DONE），基线、派生规则同 t1x，W&B 标签 t1xb
  t1yb 同一计划，ml40，buffer 1000 万：1 世界 SP-SR 失败、Tier 2 变体 A 的 4096 世界 SP-SR 不低于 0.4 的 5 条，顺序 ep16、2、3、19、13；
       ep16、ep2 的 run 名与 t1ya 的相同（t1ya 没训过），基线、派生规则同 t1y，W&B 标签 t1yb
段名以清单名开头，有几个清单名都匹配时取最长的（t3p2 不落到 t3p）。续跑或补跑另起一段（审查 M2）：段名不在上面、用 --episodes 指定这一段的 episode（必须属于这份清单，顺序照清单），
例如 t2xa2；run 名不变，文件名带段名，所以日志、导出结果都是新的，每个 checkpoint 仍只导出一次。

每条都从一条基线 run（seed 42）派生：Tier 2、Tier 3 的清单用 Tier 2 基线，Tier 1 的清单用 Tier 1 基线。基线行用 make_trial_queue.py 的 baseline()
在基线所在机器的队列文件里找、核对（期望表与队列各恰好一种、冒烟与正式一致、步数、没有种子与变体的键、种子 42）。基线覆盖里只允许 scene.support=false 与
training.replay.capacity=4000000：出现别的键就报错，因为不知道它在新 run 里该不该留。
  t2x 的基线是 stages/evaluate/exports/{ml40,mf4090}.tsv 里这条 episode 现在交的 run，机器取 run 名的最后一段（如 -p185-ml40 → ml40）。
      buffer 用 ml40 默认的 1000 万：基线的 training.replay.capacity=4000000 去掉，期望表的 buffer 列写 10000000。
  t1x、t1y 的基线是 stages/evaluate/exports/t1_select_r3x.tsv（10/9 逐条选的结果，与 results/ml40/select_t1/t1_select_r3x.tsv 相同）里这条 episode
      现在交的 run，机器同样取 run 名的最后一段。t1x 一律 400 万 buffer（ml40 上训的基线加这条覆盖，同 t3p3）；t1y 一律 1000 万（同 t2x）。
  t3p、t3p2 的基线是 ml40 上的 t2-ep<N>-w2048-cap10m-ml40；t3p3 的基线同 t2x。Tier 3 都把 parquet 的 /runs/motion/tier2/ 换成清单 motion 键给的目录
      （t3p 是 /runs/motion/tier3/，t3p2、t3p3 是 /runs/motion/tier3_v2/；序列名相同），加 400 万 buffer。
命名：t2x → t2-ep<N>-w2048-cap10m-r3a-ml40；t3p → t3-ep<N>-w2048-cap4m-r3a-mf4090；t3p2、t3p3 → t3-ep<N>-w2048-cap4m-r3a-v2-mf4090；
t1x、t1xb → t1-ep<N>-w2048-cap4m-r3a-mf4090；t1y、t1yb → t1-ep<N>-w2048-cap10m-r3a-ml40。顺序照计划（t2x、t1x、t1y 按基线 4096 世界失败:骨盆从低到高，t1xb、t1yb 见上）。
输出：stages/policy/queues/<机器>_<段名>{.expect.tsv,_smoke.txt,.txt} 与 stages/evaluate/exports/cand_t<档>_<段名>_<机器>.tsv（export_chain.sh 的清单）。
W&B 标签用清单名（t2x、t3p、t3p2、t3p3、t1x、t1y、t1xb、t1yb）。任何一条核对不过就退出，什么文件都不写。
  t3pk、t3ps（计划 agent-doc/plan-t3-kfix-retrain-2026-10-16.md）：取法同 t3p3（基线取 Tier 2 导出清单，支撑面开关沿用），只换参考目录与 run 名：
       t3pk = ep0、18、29，参考 /runs/motion/tier3_kfix/，run 名 t3-ep<N>-w2048-cap4m-r3a-kfix-mf4090；
       t3ps = ep20，参考 /runs/motion/tier3_v2s/（v2 参考加交付版方块桌面），run 名 t3-ep<N>-w2048-cap4m-r3a-v2s-mf4090。
  t3p4、t3q Tier 3 最终交付版 best_hm 的 29 集（计划 agent-doc/plan-d-t3hm-train-2026-10-12.md）：参考在 /runs/motion/tier3_hm/，基线同 t3p3 取 Tier 2
       导出清单；t3p4 在 mf4090（400 万 buffer，run 名 t3-ep<N>-w2048-cap4m-r3a-hm-mf4090），t3q 在 ml40（1000 万，t3-ep<N>-w2048-cap10m-r3a-hm-ml40）。
       两份清单都带同一个 29 集的总顺序，段之间挪集只要 --episodes。支撑面按 best_hm 自己的支撑面定（tables 键，计划审查 1）：
       期望表的 support 列与 scene.support=false 覆盖都按它给，不沿用 Tier 2 基线的（ep1 的 Tier 2 基线开着支撑面，best_hm 的 ep1 只有地面）。
  t3pg、t3qg Tier 3 标定内参的 gb3 交付版（计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md）：参考在 /runs/motion/tier3_gb3/，其余取法同 t3p4、t3q
       （基线取 Tier 2 导出清单；支撑面按 gb3 自己的，有桌面的 12 集与 best_hm 相同，results/t3gb3/inventory_ml40.txt）；
       t3pg 在 mf4090（ep20、0、22、21，400 万，run 名 t3-ep<N>-w2048-cap4m-r3a-gb3-mf4090），t3qg 在 ml40（ep6、8、7、29，1000 万，
       t3-ep<N>-w2048-cap10m-r3a-gb3-ml40）。第 2 版按复核：ep0、29 在 gb3 上新训（不拿已有的 k-fix 重训套事后定的规则）。
--seed N（计划 agent-doc/plan-first-place-2026-10-14.md 第 2 节选项甲）：只许用在非预设的段（同时给 --episodes），N 是 1–9999、不是 42 的整数；
  段名必须以 _s<N> 结尾，段名以 _s<数字> 结尾时也必须给同一个 --seed（防止漏给种子时静默生成 seed 42、或生成与预设段同名的 run）。
  run 名在机器名前加 -s<N>（如 t1-ep14-w2048-cap4m-r3a-s43-mf4090，与 make_trial_queue.py 的 seed 43 命名一致），覆盖在基线覆盖之后、变体 A 两项之前
  加 reset.seed=<N> training.seed=<N>（同 make_trial_queue.py 的 SEED43），期望表的种子列写 N；其余照这份清单。不给 --seed 时一切照旧（seed 42）。
用法（仓库根）：python3 stages/policy/make_var_queue.py t2xa|t2xb|t3p|t3p2|t3p3a|t3p3b|t3pk|t3ps|t1xa|t1ya|t1xb|t1yb|t3p4a|t3p4b|t3qa|t3qb|t1oa|t1ob|t2oa|t2ob|t1c|t2m|t3pg|t3qg|t2mv|t2xp|t1xp|t2mx|t1za|t1zp|t1zh|t1zq|t1zi|t1zr|t1zj|t1zs|t1zt|t1zw|t1hr|t1hp|t1hf|t1hq|t1hq2|t1hr2|t1zs2 [--root <仓库根>]
           python3 stages/policy/make_var_queue.py <段名> --episodes 7,16 [--root <仓库根>]   （续跑、补跑）
           python3 stages/policy/make_var_queue.py t1xb_s43 --episodes 14 --seed 43   （新种子）
           python3 stages/policy/make_var_queue.py t3qs_s43 --episodes 5,10,3,14 --seed 43   （t3s43，没有预设段的清单）
           python3 stages/policy/make_var_queue.py t3qf1 --episodes 15,27   （t3far，没有预设段的清单，seed 42）
  t1c（计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md）：ml40、1000 万，跟丢的 ep25、28、11、1 的变体 A seed 42，基线取法同 t1y，
      基线若是变体 A 的 run 就取它的基线配方 run（同 t1o）；parquet 的 /runs/motion/tier1/ 换成 /runs/motion/tier1_c10/，run 名带 -c10-。
      清单里还有 ep9、ep2（计划 agent-doc/plan-c10-extend-2026-10-16.md 路 A）：make_var_queue.py t1c2 --episodes 9,2，预设段 t1c 不变。
  t2m（计划 agent-doc/plan-c10-extend-2026-10-16.md 路 B）：Tier 2 接触标记提前，参数只由 Tier 2 的数据定（提前 5 帧、补 15 帧以内的空档），
      mf4090、400 万，跟丢的 ep25、9、1 的变体 A seed 42；基线取 Tier 2 导出清单（同 t2o），parquet 的 /runs/motion/tier2/ 换成 /runs/motion/tier2_c5/，
      run 名带 -c5-。清单里还有 Tier 2 诊断登记的其余合格集 ep2、11、17（10/17 续跑）：make_var_queue.py t2m2 --episodes 2,11,17。
  t1o、t2o（计划 agent-doc/plan-r3o-2026-10-15.md）：r3o 试点 = 变体 A 加物体朝向项。队列行开头多两个环境变量
      V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4（训练副本 v040_p185_nfg_oo 读它们，不进 Hydra 配置）；run 名带 -r3o-。
      t1o（mf4090，400 万）：t1oa = 对照 ep20 与 ep25、28、11，t1ob = ep4、9、1、2；t2o（ml40，1000 万）：t2oa = 对照 ep12 与 ep25、28、1，
      t2ob = ep4、17、9、2、11。基线的取法同 t1x、t2x（Tier 1 是 t1_select_r3x.tsv，Tier 2 是 exports/ml40.tsv、mf4090.tsv 这两份较早的导出清单，
      不是 b1014；基线只用来取 parquet、期望表的列与支撑面覆盖，同一集的这几样与 b1014 选中的 run 相同）；那条 run 是变体 A 的（名字带 -r3a-）时改用去掉 -r3a- 的基线配方 run
      （变体 A 就是从它派生的，parquet 相同；make_trial_queue.baseline 不收带变体键的基线）。
测试：stages/policy/test_make_var_queue.py。
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_trial_queue import HEADER, SMOKE_STEPS, STEPS, VARIANT, baseline  # noqa: E402

ALLOWED = {"scene.support=false", "training.replay.capacity=4000000"}
# best_hm 有桌面的 12 集（results/t3hm/support_kinds_root_mf4090.txt 的 scaled_cube=yes；其余 17 集只有地面）
TABLES_HM = [2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27]
# gb3 有桌面的集与 best_hm 相同（其余集的支撑面是空的 Xform，results/t3gb3/inventory_ml40.txt）
TABLES_GB3 = TABLES_HM
TABLES_GB4 = TABLES_HM   # gb4 的支撑面：有桌面的仍是这 12 集（计划 agent-doc/plan-t3g4-2026-10-19.md 事实 2）
# t3static 的站立参考：有桌面的集 = t3m1022a 里原 run 没有 scene.support=false 的集（results/t3static/floor_mf4090.txt、results/ml40/t3static/floor_ml40.txt 打出的覆盖）
TABLES_ST = [2, 3, 9, 11, 13, 16, 17, 25, 27, 28]
TABLES_GB5 = sorted(set(TABLES_HM) | {1})   # gb5：有桌面的 12 集，加 ep1（10/21 给吸尘器补了两个倚靠用的方盒；计划 agent-doc/plan-t3g5-2026-10-21.md 第 2 版事实 3、复核 MAJOR 2）
# D 的总顺序：① ep15、17；② 没在 Tier 3 训练过的 18 集与 ep8（10/9 试跑在旧参考上失败），按集号；③ 其余 8 集重训（计划 agent-doc/plan-d-t3hm-train-2026-10-12.md 第 2 节）
ORDER_HM = [15, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 19, 21, 25, 27, 0, 18, 22, 24, 26, 20, 23, 29]
SETS = {
    "t2x": {"plan": "agent-doc/plan-t2-scaleout-2026-10-09.md", "tier": 2, "machine": "ml40", "cap": None,
            "episodes": [29, 23, 6, 18, 7, 16, 0, 21, 15, 10, 27, 3, 5, 11, 17, 9, 1, 13, 19, 2, 4, 14, 25, 28],
            "parts": {"t2xa": [29, 23, 6, 18, 7, 16, 0, 21, 15, 10, 27, 3, 5, 11, 17], "t2xb": [9, 1, 13, 19, 2, 4, 14, 25, 28]},
            "run": "t2-ep{ep}-w2048-cap10m-r3a-ml40",
            "desc": "Tier 2 铺开：其余 24 条的变体 A seed 42（ml40，buffer 1000 万，按基线 4096 世界失败:骨盆从低到高）"},
    "t3p": {"plan": "agent-doc/plan-t3-pilot-2026-10-09.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
            "episodes": [8, 24], "parts": {"t3p": [8, 24]}, "base": "t2-ep{ep}-w2048-cap10m-ml40", "motion": "tier3",
            "run": "t3-ep{ep}-w2048-cap4m-r3a-mf4090",
            "desc": "Tier 3 试跑：ep8、ep24 的变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3 下从视频重建的那份）"},
    "t3p2": {"plan": "agent-doc/plan-t3-pilot-v2-2026-10-10.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
             "episodes": [8, 24], "parts": {"t3p2": [8, 24]}, "base": "t2-ep{ep}-w2048-cap10m-ml40", "motion": "tier3_v2",
             "run": "t3-ep{ep}-w2048-cap4m-r3a-v2-mf4090",
             "desc": "Tier 3 试跑第二轮：ep8、ep24 的变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3_v2 下同组成员校正版重建的那份）"},
    "t3p3": {"plan": "agent-doc/plan-t3-v2-train10-2026-10-10.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
             "episodes": [20, 29, 0, 15, 18, 21, 22, 23, 26, 17],
             "parts": {"t3p3a": [20, 29, 0, 15], "t3p3b": [18, 21, 22, 23, 26, 17]}, "base": None, "motion": "tier3_v2",
             "run": "t3-ep{ep}-w2048-cap4m-r3a-v2-mf4090",
             "desc": "Tier 3 校正版其余 10 集：变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3_v2 下校正版重建的那份，基线取 Tier 2 导出清单）"},
    "t3pk": {"plan": "agent-doc/plan-t3-kfix-retrain-2026-10-16.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
             "episodes": [0, 18, 29], "parts": {"t3pk": [0, 18, 29]}, "base": None, "motion": "tier3_kfix",
             "run": "t3-ep{ep}-w2048-cap4m-r3a-kfix-mf4090",
             "desc": "Tier 3 k-fix 重训：现有策略跟不住新参考的 3 集，变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3_kfix，取法同 t3p3）"},
    "t3ps": {"plan": "agent-doc/plan-t3-kfix-retrain-2026-10-16.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
             "episodes": [20], "parts": {"t3ps": [20]}, "base": None, "motion": "tier3_v2s",
             "run": "t3-ep{ep}-w2048-cap4m-r3a-v2s-mf4090",
             "desc": "Tier 3 ep20 换成交付版方块桌面重训：参考是 runs/motion/tier3_v2s（v2 的参考加方块桌面），变体 A seed 42（mf4090，buffer 400 万，取法同 t3p3）"},
    "t3p4": {"plan": "agent-doc/plan-d-t3hm-train-2026-10-12.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
             "episodes": ORDER_HM, "parts": {"t3p4a": [15, 17, 1, 2, 3], "t3p4b": [12, 13, 14, 16, 19, 21, 25, 27]},
             "base": None, "motion": "tier3_hm", "tables": TABLES_HM, "run": "t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090",
             "desc": "Tier 3 最终交付版 best_hm：变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3_hm，基线取 Tier 2 导出清单，支撑面按 best_hm）"},
    "t3q": {"plan": "agent-doc/plan-d-t3hm-train-2026-10-12.md", "tier": 3, "machine": "ml40", "cap": None,
            "episodes": ORDER_HM, "parts": {"t3qa": [4, 5, 6, 7, 8, 9, 10, 11], "t3qb": [0, 18, 22, 24, 26, 20, 23, 29]},
            "base": None, "motion": "tier3_hm", "tables": TABLES_HM, "run": "t3-ep{ep}-w2048-cap10m-r3a-hm-ml40",
            "desc": "Tier 3 最终交付版 best_hm：变体 A seed 42（ml40，buffer 1000 万，参考是 runs/motion/tier3_hm，基线取 Tier 2 导出清单，支撑面按 best_hm）"},
    "t3pg": {"plan": "agent-doc/plan-t3-gb3-switch-2026-10-17.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
             "episodes": [20, 0, 22, 21], "parts": {"t3pg": [20, 0, 22, 21]},
             "base": None, "motion": "tier3_gb3", "tables": TABLES_GB3, "run": "t3-ep{ep}-w2048-cap4m-r3a-gb3-mf4090",
             "desc": "Tier 3 gb3（标定内参）：变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3_gb3，基线取 Tier 2 导出清单，支撑面按 gb3）"},
    # 10/19 晚（计划 agent-doc/plan-t3g4-2026-10-19.md）：gb4 交付（人体按 PoseInversion 重解、物体 Kabsch、地面与配准重算），跟得住的 11 集重训
    "t3pg4": {"plan": "agent-doc/plan-t3g4-2026-10-19.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
              "episodes": [0, 6, 7, 8, 20, 22, 18, 24, 26, 29, 21], "parts": {"t3pg4": [0, 6, 7, 8, 20, 22, 18, 24, 26, 29, 21]},
              "base": None, "motion": "tier3_gb4", "tables": TABLES_GB4, "run": "t3-ep{ep}-w2048-cap4m-r3a-gb4-mf4090",
              "desc": "Tier 3 gb4（标定内参、人体重解）：变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3_gb4，基线取 Tier 2 导出清单，支撑面按 gb4）"},
    # 10/21（计划 agent-doc/plan-t3g5-2026-10-21.md）：跟丢的 18 集在 gb5 上训，A 批是 Tier 2 训得成的 7 集，B 批其余 11 集
    "t3pg5": {"plan": "agent-doc/plan-t3g5-2026-10-21.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
              "episodes": [5, 10, 15, 27, 13, 19, 1, 9, 17], "parts": {"t3pg5a": [5, 10, 15, 27], "t3pg5b": [13, 19, 1, 9, 17]},
              "base": None, "motion": "tier3_gb5", "tables": TABLES_GB5, "run": "t3-ep{ep}-w2048-cap4m-r3a-gb5-mf4090",
              "desc": "Tier 3 gb5（gb4 上重选 FoundationPose 集的物体位姿）：跟丢集的变体 A seed 42（mf4090，buffer 400 万，参考是 runs/motion/tier3_gb5，基线取 Tier 2 导出清单，支撑面按 gb5）"},
    # 10/21 下午（计划 agent-doc/plan-t3static-2026-10-21.md 第 2 版）：跟丢的 15 集换成站立保底（机器人全程停在第 0 帧，零残差的第 409600 步冒烟策略）；
    # 参考 runs/motion/tier3_st 里每集从它在 t3m1022a 的那条 run 的参考版本生成（hm、v2、hm28）；有桌面的集照抄原 run（原 run 没有 scene.support=false 的 10 集）
    "t3pst": {"plan": "agent-doc/plan-t3static-2026-10-21.md", "tier": 3, "machine": "mf4090", "cap": "4000000",
              "episodes": [16, 23, 1, 2, 3, 4, 9, 11, 12, 13, 17, 19, 25, 27, 28],
              "parts": {"t3psta": [16, 23, 1], "t3pstb": [2, 3, 4, 9, 11, 12, 13, 17, 19, 25, 27, 28]},
              "base": None, "motion": "tier3_st", "tables": TABLES_ST, "run": "t3-ep{ep}-w2048-cap4m-r3a-st-mf4090",
              "smoke_extra": ["training.checkpoint.save_interval_environment_steps=409600"],
              "desc": "Tier 3 站立保底：跟丢的 15 集，参考是 runs/motion/tier3_st（机器人全程停在第 0 帧、物体轨迹照抄），变体 A seed 42（mf4090，buffer 400 万），冒烟每 409600 步存一份、只评第 409600 步；正式训练用停链标记挡住"},
    # 10/26（计划 agent-doc/plan-t3s43-2026-10-26.md 第 2 版）：gb5 上差一点的 4 集换种子；没有预设段，用 t3qs_s43 --episodes 5,10,3,14 --seed 43 生成（复核 MAJOR 1）
    "t3qs": {"plan": "agent-doc/plan-t3s43-2026-10-26.md", "tier": 3, "machine": "ml40", "cap": None,
             "episodes": [5, 10, 3, 14], "parts": {}, "base": None, "motion": "tier3_gb5", "tables": TABLES_GB5,
             "run": "t3-ep{ep}-w2048-cap10m-r3a-gb5-ml40",
             "desc": "Tier 3 gb5 差一点的 4 集（ep5、10、3、14）换种子：变体 A（ml40，buffer 1000 万，参考是 runs/motion/tier3_gb5，支撑面按 gb5），种子由 --seed 给"},
    # 10/27（计划 agent-doc/plan-t3far-2026-10-27.md）：gb5 上只在 mf4090（400 万）训过、仍跟丢的 ep15、27 挪到 ml40（1000 万）重训，seed 42；没有预设段，用 t3qf1 --episodes 15,27 生成；同计划在 mf4090 上用 t3pg5c_s43 --episodes 13,19 --seed 43
    "t3qf": {"plan": "agent-doc/plan-t3far-2026-10-27.md", "tier": 3, "machine": "ml40", "cap": None,
             "episodes": [15, 27], "parts": {}, "base": None, "motion": "tier3_gb5", "tables": TABLES_GB5,
             "run": "t3-ep{ep}-w2048-cap10m-r3a-gb5-ml40",
             "desc": "Tier 3 gb5 上只在 mf4090 训过的 ep15、27 挪到 ml40：变体 A seed 42（ml40，buffer 1000 万，参考是 runs/motion/tier3_gb5，支撑面按 gb5）"},
    "t3qg5": {"plan": "agent-doc/plan-t3g5-2026-10-21.md", "tier": 3, "machine": "ml40", "cap": None,
              "episodes": [3, 12, 16, 14, 23, 2, 4, 11, 25], "parts": {"t3qg5a": [3, 12, 16], "t3qg5b": [14, 23, 2, 4, 11, 25]},
              "base": None, "motion": "tier3_gb5", "tables": TABLES_GB5, "run": "t3-ep{ep}-w2048-cap10m-r3a-gb5-ml40",
              "desc": "Tier 3 gb5（gb4 上重选 FoundationPose 集的物体位姿）：跟丢集的变体 A seed 42（ml40，buffer 1000 万，参考是 runs/motion/tier3_gb5，基线取 Tier 2 导出清单，支撑面按 gb5）"},
    "t3qg": {"plan": "agent-doc/plan-t3-gb3-switch-2026-10-17.md", "tier": 3, "machine": "ml40", "cap": None,
             "episodes": [6, 8, 7, 29], "parts": {"t3qg": [6, 8, 7, 29]},
             "base": None, "motion": "tier3_gb3", "tables": TABLES_GB3, "run": "t3-ep{ep}-w2048-cap10m-r3a-gb3-ml40",
             "desc": "Tier 3 gb3（标定内参）：变体 A seed 42（ml40，buffer 1000 万，参考是 runs/motion/tier3_gb3，基线取 Tier 2 导出清单，支撑面按 gb3）"},
    "t1x": {"plan": "agent-doc/plan-next-training-2026-10-12.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
            "episodes": [17, 21, 10, 13, 19, 3, 14], "parts": {"t1xa": [17, 21, 10, 13, 19, 3, 14]}, "base": None,
            "run": "t1-ep{ep}-w2048-cap4m-r3a-mf4090",
            "desc": "Tier 1 补候选：骨盆比例低于 0.65 的 7 条的变体 A seed 42（mf4090，buffer 400 万，基线取 10/9 逐条选的清单）"},
    "t1y": {"plan": "agent-doc/plan-next-training-2026-10-12.md", "tier": 1, "machine": "ml40", "cap": None,
            "episodes": [2, 11, 16, 4, 9, 25], "parts": {"t1ya": [2, 11, 16, 4, 9, 25]}, "base": None,
            "run": "t1-ep{ep}-w2048-cap10m-r3a-ml40",
            "desc": "Tier 1 补候选：几乎每个世界都摔的 6 条（不含方盒）的变体 A seed 42（ml40，buffer 1000 万，基线取 10/9 逐条选的清单）"},
    "t1xb": {"plan": "agent-doc/plan-t1-chase-2026-10-13.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [3, 19, 13, 10, 14], "parts": {"t1xb": [3, 19, 13, 10, 14]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap4m-r3a-mf4090",
             "desc": "Tier 1 追基线：10/12 计划 A 没开的 5 条，SP-SR 失败的 ep3、19、13 排在前面的变体 A seed 42（mf4090，buffer 400 万，基线取 10/9 逐条选的清单）"},
    "t1yb": {"plan": "agent-doc/plan-t1-chase-2026-10-13.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [16, 2, 3, 19, 13], "parts": {"t1yb": [16, 2, 3, 19, 13]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap10m-r3a-ml40",
             "desc": "Tier 1 追基线：1 世界 SP-SR 失败、Tier 2 变体 A 的 4096 世界 SP-SR 不低于 0.4 的 5 条的变体 A seed 42（ml40，buffer 1000 万，基线取 10/9 逐条选的清单）"},
    "t1c": {"plan": "agent-doc/plan-t1-contact-anticipation-2026-10-16.md", "tier": 1, "machine": "ml40", "cap": None,
            "episodes": [25, 28, 11, 1, 9, 2], "parts": {"t1c": [25, 28, 11, 1]}, "base": None, "motion": "tier1_c10", "strip_r3a": True,
            "run": "t1-ep{ep}-w2048-cap10m-r3a-c10-ml40",
            "desc": "Tier 1 接触标记提前：跟丢的 4 条的变体 A seed 42，参考换成 runs/motion/tier1_c10（接触起点提前 10 帧、补 15 帧内的空档，参考轨迹不变；ml40，buffer 1000 万）"},
    "t2m": {"plan": "agent-doc/plan-c10-extend-2026-10-16.md", "tier": 2, "machine": "mf4090", "cap": "4000000",
            "episodes": [25, 9, 1, 2, 11, 17], "parts": {"t2m": [25, 9, 1]}, "base": None, "motion": "tier2_c5", "strip_r3a": True,
            "run": "t2-ep{ep}-w2048-cap4m-r3a-c5-mf4090",
            "desc": "Tier 2 接触标记提前（参数只由 Tier 2 的数据定）：跟丢的 ep25、9、1 的变体 A seed 42，参考换成 runs/motion/tier2_c5（接触起点提前 5 帧、补 15 帧以内的空档，参考轨迹不变；mf4090，buffer 400 万，基线取 Tier 2 导出清单）"},
    "t1o": {"plan": "agent-doc/plan-r3o-2026-10-15.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
            "episodes": [20, 25, 28, 11, 4, 9, 1, 2], "parts": {"t1oa": [20, 25, 28, 11], "t1ob": [4, 9, 1, 2]}, "base": None,
            "run": "t1-ep{ep}-w2048-cap4m-r3o-mf4090", "variant": "r3o",
            "desc": "Tier 1 r3o 试点 P1：对照 ep20 与跟丢的 7 集，变体 A 加物体朝向项 seed 42（mf4090，buffer 400 万，基线取 10/9 逐条选的清单）"},
    "t2o": {"plan": "agent-doc/plan-r3o-2026-10-15.md", "tier": 2, "machine": "ml40", "cap": None,
            "episodes": [12, 25, 28, 1, 4, 17, 9, 2, 11], "parts": {"t2oa": [12, 25, 28, 1], "t2ob": [4, 17, 9, 2, 11]}, "base": None,
            "run": "t2-ep{ep}-w2048-cap10m-r3o-ml40", "variant": "r3o",
            "desc": "Tier 2 r3o 试点 P2：对照 ep12 与跟丢的 8 集，变体 A 加物体朝向项 seed 42（ml40，buffer 1000 万，基线取 Tier 2 导出清单）"},
    # 10/18 晚（计划 agent-doc/plan-precision-2026-10-18.md）：SP 过、MP 不过的集卡在位置（时间平均 4–7 cm，MP 门槛 3 cm），朝向都在门槛内
    "t2mv": {"plan": "agent-doc/plan-precision-2026-10-18.md", "tier": 2, "machine": "mf4090", "cap": "4000000",
             "episodes": [23], "parts": {"t2mv": [23]}, "base": None,
             "run": "t2-ep{ep}-w2048-cap4m-r3a-mf4090",
             "desc": "Tier 2 ep23 补变体 A seed 42（10/9 铺开时 ml40 冒烟显存超限没训；mf4090，buffer 400 万，基线取 Tier 2 导出清单）"},
    "t2xp": {"plan": "agent-doc/plan-precision-2026-10-18.md", "tier": 2, "machine": "ml40", "cap": None,
             "episodes": [14, 13, 19, 16], "parts": {"t2xp": [14, 13, 19, 16]}, "base": None, "strip_r3a": True,
             "run": "t2-ep{ep}-w2048-cap10m-r3p-ml40", "variant": "r3p",
             "desc": "Tier 2 r3p（变体 A 把物体位置奖励的 σ 从 0.05 收到 0.03）：SP 过、MP 不过的 ep14、13、19 与对照 ep16 seed 42（ml40，buffer 1000 万，基线取 Tier 2 导出清单）"},
    "t1xp": {"plan": "agent-doc/plan-precision-2026-10-18.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [10, 29, 19], "parts": {"t1xp": [10, 29, 19]}, "base": None, "strip_r3a": True,
             "run": "t1-ep{ep}-w2048-cap4m-r3p-mf4090", "variant": "r3p",
             "desc": "Tier 1 r3p（变体 A 把物体位置奖励的 σ 从 0.05 收到 0.03）：SP 过、MP 不过的 ep10、29、19 seed 42（mf4090，buffer 400 万，基线取 10/9 逐条选的清单）"},
    # 10/19 早（计划 agent-doc/plan-t2mx-2026-10-19.md）：ep23 的变体 A 在 mf4090、400 万上修好了 MP；其余三条 SP 过、MP 不过的集只在 ml40、1000 万上训过变体 A
    "t2mx": {"plan": "agent-doc/plan-t2mx-2026-10-19.md", "tier": 2, "machine": "mf4090", "cap": "4000000",
             "episodes": [13, 19, 14], "parts": {"t2mx": [13, 19, 14]}, "base": None, "strip_r3a": True,   # ep14 放最后：正式训练显存不足（退出码 4）会跳过其后所有 run（复核 2）
             "run": "t2-ep{ep}-w2048-cap4m-r3a-mf4090",
             "desc": "Tier 2 变体 A 换到 mf4090、buffer 400 万：SP 过、MP 不过的 ep13、19、14 seed 42（ml40、1000 万的变体 A 没修好 MP；基线取 Tier 2 导出清单）"},
    # 10/19 下午（计划 agent-doc/plan-t2sd-2026-10-19.md）：409.6 万步的冒烟策略已经会走到瓶子跟前；冒烟改成每 409600 步存一份，评更早的 checkpoint
    "t2msd": {"plan": "agent-doc/plan-t2sd-2026-10-19.md", "tier": 2, "machine": "mf4090", "cap": "4000000",
              "episodes": [25], "parts": {"t2msd": [25]}, "base": None, "strip_r3a": True,
              "run": "t2-ep{ep}-w2048-cap4m-r3a-sd-mf4090", "smoke_extra": ["training.checkpoint.save_interval_environment_steps=409600"],
              "desc": "Tier 2 ep25 冒烟密存档：变体 A seed 42（mf4090，buffer 400 万），冒烟每 409600 步存一份 checkpoint；正式训练用停链标记挡住"},
    # 10/19 晚（计划 agent-doc/plan-t2rp-2026-10-19.md）：ep25 的参考回放 checkpoint（第 409600 步）比训好的少碰物体；其余全失败的集照样量一遍
    "t2mrp": {"plan": "agent-doc/plan-t2rp-2026-10-19.md", "tier": 2, "machine": "mf4090", "cap": "4000000",
              "episodes": [4, 28, 1, 9, 2, 11, 17], "parts": {"t2mrp": [4, 28, 1, 9, 2, 11, 17]}, "base": None, "strip_r3a": True,
              "run": "t2-ep{ep}-w2048-cap4m-r3a-sd-mf4090", "smoke_extra": ["training.checkpoint.save_interval_environment_steps=409600"],
              "desc": "Tier 2 全失败的 ep4、28、1、9、2、11、17 冒烟密存档：变体 A seed 42（mf4090，buffer 400 万），冒烟每 409600 步存一份 checkpoint；正式训练用停链标记挡住"},
    # 10/20 凌晨（计划 agent-doc/plan-t2fc-2026-10-20.md）：参考的 Dex3 手指全程张开；接触段合到行程的 0.8（stages/retarget/finger_close.py --mode fixed）
    "t2fc": {"plan": "agent-doc/plan-t2fc-2026-10-20.md", "tier": 2, "machine": "ml40", "cap": None,
             "episodes": [25, 4], "parts": {"t2fc": [25, 4]}, "base": None, "motion": "tier2_fc8", "strip_r3a": True,
             "run": "t2-ep{ep}-w2048-cap10m-r3a-fc8-ml40",
             "desc": "Tier 2 参考合手：举起物体全失败的 ep25、ep4 的变体 A seed 42，参考换成 runs/motion/tier2_fc8（只改接触段的 14 个 Dex3 手指关节，合到行程的 0.8；ml40，buffer 1000 万，基线取 Tier 2 导出清单）"},
    # 10/20 上午（计划 agent-doc/plan-t12lift-2026-10-20.md）：ep25 训好的策略把瓶子弄到地上（Tier 2 117.09、Tier 1 114.10 cm），物体不动时约 49.5 cm；
    # 参考的两条手臂与手指固定为第 0 帧（stages/retarget/hands_off.py），训一条不去够物体的保底策略
    "t2ho": {"plan": "agent-doc/plan-t12lift-2026-10-20.md", "tier": 2, "machine": "ml40", "cap": None,
             "episodes": [25, 28], "parts": {"t2ho": [25, 28]}, "base": None, "motion": "tier2_ho", "strip_r3a": True,
             "run": "t2-ep{ep}-w2048-cap10m-r3a-ho-ml40",
             "desc": "Tier 2 保底：ep25、ep28 的变体 A seed 42，参考换成 runs/motion/tier2_ho（两条手臂与手指固定为第 0 帧、接触标记清零，腿与根照抄；ml40，buffer 1000 万，基线取 Tier 2 导出清单）"},
    # 10/20 晚（计划 agent-doc/plan-t2term-2026-10-20.md）：举起类全失败集的训练回合多数因物体终止结束（ep25 0.736、ep28 0.737），放宽物体位置终止 0.1 → 0.3 m
    "t2tm": {"plan": "agent-doc/plan-t2term-2026-10-20.md", "tier": 2, "machine": "ml40", "cap": None,
             "episodes": [25, 28], "parts": {"t2tm": [25, 28]}, "base": None, "strip_r3a": True,
             "run": "t2-ep{ep}-w2048-cap10m-r3a-ot3-ml40", "extra": ["termination.object_position_threshold=0.3"],
             "desc": "Tier 2 放宽物体位置终止：ep25、ep28 的变体 A seed 42 加 termination.object_position_threshold=0.3（原 0.1；ml40，buffer 1000 万，基线取 Tier 2 导出清单）"},
    # 10/21 午（计划 agent-doc/plan-t2mtr-2026-10-21.md）：t2term 在 ml40 上 ep25、ep28 的 MPPE 降了 52、20 cm，单个种子；换到 mf4090、400 万重复，加外推 ep4，ep28 放最后（复核 5）
    "t2mtr": {"plan": "agent-doc/plan-t2mtr-2026-10-21.md", "tier": 2, "machine": "mf4090", "cap": "4000000",
              "episodes": [25, 4, 28], "parts": {"t2mtr": [25, 4, 28]}, "base": None, "strip_r3a": True,
              "run": "t2-ep{ep}-w2048-cap4m-r3a-ot3-mf4090", "extra": ["termination.object_position_threshold=0.3"],
              "desc": "Tier 2 放宽物体位置终止的重复：ep25、外推 ep4 与 ep28 的变体 A seed 42 加 termination.object_position_threshold=0.3（原 0.1；mf4090，buffer 400 万，基线取 Tier 2 导出清单）"},
    "t1ho": {"plan": "agent-doc/plan-t12lift-2026-10-20.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [25], "parts": {"t1ho": [25]}, "base": None, "motion": "tier1_ho", "strip_r3a": True,
             "run": "t1-ep{ep}-w2048-cap10m-r3a-ho-ml40",
             "desc": "Tier 1 保底：ep25 的变体 A seed 42，参考换成 runs/motion/tier1_ho（同 t2ho 的做法，只用 Tier 1 自己的参考；ml40，buffer 1000 万，基线取 Tier 1 现在交的清单）"},
    # 10/29（计划 agent-doc/plan-t1push-2026-10-29.md）：Tier 1 冲第一
    "t1za": {"plan": "agent-doc/plan-t1push-2026-10-29.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [25, 4, 10, 28, 11, 9, 1, 17], "parts": {"t1za": [25, 4, 10, 28, 11, 9, 1, 17]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap10m-r3a-ml40",
             "desc": "Tier 1 冲第一 A1：难集与 MP 偏弱集在原参考上的变体 A seed 42（ml40，buffer 1000 万，基线取 Tier 1 现在交的清单）"},
    "t1zp": {"plan": "agent-doc/plan-t1push-2026-10-29.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [8, 21, 17, 20], "parts": {"t1zp": [8, 21, 17]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap4m-r3o-mf4090", "variant": "r3o",
             "desc": "Tier 1 冲第一 A2：变体 A 加物体朝向项 seed 42（ep20 另起 t1zp_s43 段，seed 43；mf4090，buffer 400 万，基线取 Tier 1 现在交的清单）"},
    # 10/30（计划 agent-doc/plan-t1palm-2026-10-30.md）：难集收紧手掌位置奖励的尺度 0.2 → 0.05（r3h），其余同变体 A
    "t1zh": {"plan": "agent-doc/plan-t1palm-2026-10-30.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [25, 9, 4], "parts": {"t1zh": [25, 9, 4]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap10m-r3h-ml40", "extra": ["objective.palm_position_std=0.05"],
             "desc": "Tier 1 r3h（变体 A 把手掌位置奖励的尺度从 0.2 收到 0.05）：难集 ep25、9、4 seed 42（ml40，buffer 1000 万，基线取 t1za 同配方的 r3a）"},
    "t1zq": {"plan": "agent-doc/plan-t1palm-2026-10-30.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [20, 17], "parts": {"t1zq": [20, 17]}, "base": None, "strip_r3a": True,
             "run": "t1-ep{ep}-w2048-cap4m-r3h-mf4090", "extra": ["objective.palm_position_std=0.05"],
             "desc": "Tier 1 r3h（同 t1zh）：成功集对照 ep20 与难集 ep17 seed 42（mf4090，buffer 400 万；基线 t1-ep{20,17}-w2048-cap4m-r3a-mf4090，ep20 另有 seed 43）"},
    # 10/30 晚（计划 agent-doc/plan-t1palm-2026-10-30.md 第 3 版）：0.05 下 ep25 不变好，按登记试下一档 0.1（r3i）
    "t1zi": {"plan": "agent-doc/plan-t1palm-2026-10-30.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [25, 9], "parts": {"t1zi": [25, 9]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap10m-r3i-ml40", "extra": ["objective.palm_position_std=0.1"],
             "desc": "Tier 1 r3i（变体 A 把手掌位置奖励的尺度从 0.2 收到 0.1）：难集 ep25、9 seed 42（ml40，buffer 1000 万，基线取 t1za 同配方的 r3a）"},
    "t1zr": {"plan": "agent-doc/plan-t1palm-2026-10-30.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [20, 17], "parts": {"t1zr": [20, 17]}, "base": None, "strip_r3a": True,
             "run": "t1-ep{ep}-w2048-cap4m-r3i-mf4090", "extra": ["objective.palm_position_std=0.1"],
             "desc": "Tier 1 r3i（同 t1zi）：成功集对照 ep20 与难集 ep17 seed 42（mf4090，buffer 400 万；基线同 t1zq）"},
    # 10/30 夜（计划 agent-doc/plan-t12night-2026-10-30.md）：v3 之后两台卡接着训的 Tier 1 候选
    "t1zj": {"plan": "agent-doc/plan-t12night-2026-10-30.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [11, 29, 14, 22], "parts": {"t1zj": [29, 14, 22]}, "base": None, "strip_r3a": True,
             "run": "t1-ep{ep}-w2048-cap10m-r3a-ml40",
             "desc": "Tier 1 补候选：ep11（导出那一局 SP 成功、MP 失败）与清单里用 400 万 run、MP-SR 期望偏低的 ep29、14、22，换 buffer 1000 万的变体 A（ml40，同 t1za；10/30 夜按复核起的是 t1zj_s43 --episodes 11,29,14 --seed 43）"},
    "t1zs": {"plan": "agent-doc/plan-t12night-2026-10-30.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [18, 24], "parts": {"t1zs": [18, 24]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap4m-r3o-mf4090", "variant": "r3o",
             "desc": "Tier 1 补候选：r3o（变体 A 加物体朝向项）seed 42 铺到 ep18、24（mf4090，buffer 400 万，同 t1zp；ep18 的基线是同机同配方的 r3a seed 42，ep24 只有非变体 A 的基线）"},
    # 10/31（计划 agent-doc/plan-t1r3o-2026-10-31.md）：t1zs 两集都按规则换入，r3o 铺到清单里跟得住、MP-SR 期望偏低的集
    "t1zt": {"plan": "agent-doc/plan-t1r3o-2026-10-31.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [19, 13, 7, 3, 22, 16], "parts": {"t1zt": [19, 13], "t1zt2": [7, 3], "t1zt3": [22, 16], "t1zt4": [7, 22]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap4m-r3o-mf4090", "variant": "r3o",
             "desc": "Tier 1 r3o 铺开：seed 42，三段各两集（ep19、13；ep7、3；ep22、16），mf4090，buffer 400 万，同 t1zs；配方取 10/9 清单（t1_select_r3x.tsv）里同集的 run，换集时对照现在的规则清单"},
    # 10/31 晚（计划 agent-doc/plan-t1r3o-2026-10-31.md 第 2 版）：ep19、13、3、16 在 400 万下 r3a 也跟不住，清单用的是 1000 万 r3a；
    # r3o 改在 ml40 用 1000 万，与清单的 run 只差物体朝向项。mf4090 只留对照也是 400 万的 ep7、22（t1zt4）
    "t1zw": {"plan": "agent-doc/plan-t1r3o-2026-10-31.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [19, 13, 3, 16], "parts": {"t1zw": [19, 13], "t1zw2": [3, 16]}, "base": None,
             "run": "t1-ep{ep}-w2048-cap10m-r3o-ml40", "variant": "r3o",
             "desc": "Tier 1 r3o 用 1000 万：seed 42，两段各两集（ep19、13；ep3、16），ml40；配方取 10/9 清单里同集的 run，换集时对照现在的规则清单（1000 万 r3a）"},
    # 10/31（计划 agent-doc/plan-t1hard-2026-10-31.md）：跟丢的集，三条都是基础配方（novar，不加变体 A；变体 A 在 ep17、9、25 上都比基础配方差）：
    # t1hr = 加物体朝向项（ep17 只是朝向超线）；t1hp = 手掌位置终止 0.15 → 0.08 m、t1hf = 抓握闭合项权重 5 → 20（ep25、9 手掌贴不住）
    "t1hr": {"plan": "agent-doc/plan-t1hard-2026-10-31.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [17], "parts": {"t1hr": [17]}, "base": None, "novar": True,
             "run": "t1-ep{ep}-w2048-cap10m-r3o-nv-ml40", "variant": "r3o",
             "desc": "Tier 1 跟丢的集：基础配方加物体朝向项（-r3o-nv-，不加变体 A）seed 42，ml40，buffer 1000 万；配方取 10/9 清单里同集的 run"},
    "t1hp": {"plan": "agent-doc/plan-t1hard-2026-10-31.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [25, 9], "parts": {"t1hp": [25, 9]}, "base": None, "novar": True,
             "run": "t1-ep{ep}-w2048-cap10m-pt08-ml40", "extra": ["termination.palm_position_threshold=0.08"],
             "desc": "Tier 1 跟丢的集：基础配方把手掌位置终止从 0.15 收到 0.08 m（pt08，不加变体 A）seed 42，ml40，buffer 1000 万；配方取 10/9 清单里同集的 run"},
    "t1hf": {"plan": "agent-doc/plan-t1hard-2026-10-31.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
             "episodes": [25, 9], "parts": {"t1hf": [25, 9]}, "base": None, "novar": True,
             "run": "t1-ep{ep}-w2048-cap4m-fc20-mf4090", "extra": ["objective.force_closure_weight=20.0"],
             "desc": "Tier 1 跟丢的集：基础配方把抓握闭合项权重 5 → 20（fc20，不加变体 A）seed 42，mf4090，buffer 400 万；配方取 10/9 清单里同集的 run"},
    # 11/01（计划 agent-doc/plan-t1mppe-2026-11-01.md）：pt08 铺到 MPPE 最大的 ep4；ep25 pt08 换 seed 43 走 t1hq_s43 --episodes 25 --seed 43
    "t1hq": {"plan": "agent-doc/plan-t1mppe-2026-11-01.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [4, 25], "parts": {"t1hq": [4]}, "base": None, "novar": True,
             "run": "t1-ep{ep}-w2048-cap10m-pt08-ml40", "extra": ["termination.palm_position_threshold=0.08"],
             "desc": "Tier 1 MPPE 榜：基础配方把手掌位置终止从 0.15 收到 0.08 m（pt08，不加变体 A），ml40，buffer 1000 万；配方取 10/9 清单里同集的 run"},
    # 11/01（计划 agent-doc/plan-t1fill-2026-11-01.md）：ep25 手掌位置终止再收到 0.05；朝向项铺到没试过的 1000 万基础配方集 ep26、23
    "t1hq2": {"plan": "agent-doc/plan-t1fill-2026-11-01.md", "tier": 1, "machine": "ml40", "cap": None,
              "episodes": [25], "parts": {"t1hq2": [25]}, "base": None, "novar": True,
              "run": "t1-ep{ep}-w2048-cap10m-pt05-ml40", "extra": ["termination.palm_position_threshold=0.05"],
              "desc": "Tier 1 ep25：基础配方把手掌位置终止从 0.15 收到 0.05 m（pt05，不加变体 A）seed 42，ml40，buffer 1000 万；配方取 10/9 清单里同集的 run"},
    "t1zs2": {"plan": "agent-doc/plan-t1fill-2026-11-01.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
              "episodes": [26, 12], "parts": {"t1zs2": [26, 12]}, "base": None,
              "run": "t1-ep{ep}-w2048-cap4m-r3o-mf4090", "variant": "r3o",
              "desc": "Tier 1 AUC 榜：r3o（变体 A 加物体朝向项）seed 42 铺到没试过的 ep26、12（mf4090，buffer 400 万，同 t1zs；两集只有非变体 A 的基线）"},
    "t1hr2": {"plan": "agent-doc/plan-t1fill-2026-11-01.md", "tier": 1, "machine": "ml40", "cap": None,
              "episodes": [26, 23], "parts": {"t1hr2": [26, 23]}, "base": None, "novar": True,
              "run": "t1-ep{ep}-w2048-cap10m-r3o-nv-ml40", "variant": "r3o",
              "desc": "Tier 1 AUC 榜：基础配方加物体朝向项（-r3o-nv-，不加变体 A）seed 42，ml40，buffer 1000 万；配方取 10/9 清单里同集的 run"},
    # 11/01 夜（计划 agent-doc/plan-t1night-2026-11-01.md）：MPPE 最大的跟丢集试抓握闭合项权重 20；t1hg 在 ml40 用 1000 万（ep4、9 的清单 run 是同机同配方），
    # t1hf2 在 mf4090 用 400 万（ep1、4；ep9 换种子走 t1hf2_s43 --episodes 9 --seed 43）
    "t1hg": {"plan": "agent-doc/plan-t1night-2026-11-01.md", "tier": 1, "machine": "ml40", "cap": None,
             "episodes": [4, 9, 1], "parts": {"t1hg": [4, 9]}, "base": None, "novar": True,
             "run": "t1-ep{ep}-w2048-cap10m-fc20-ml40", "extra": ["objective.force_closure_weight=20.0"],
             "desc": "Tier 1 MPPE 榜：基础配方把抓握闭合项权重 5 → 20（fc20，不加变体 A）seed 42，ml40，buffer 1000 万；配方取 10/9 清单里同集的 run"},
    "t1hf2": {"plan": "agent-doc/plan-t1night-2026-11-01.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
              "episodes": [1, 4, 9], "parts": {"t1hf2": [1, 4]}, "base": None, "novar": True,
              "run": "t1-ep{ep}-w2048-cap4m-fc20-mf4090", "extra": ["objective.force_closure_weight=20.0"],
              "desc": "Tier 1 MPPE 榜：基础配方把抓握闭合项权重 5 → 20（fc20，不加变体 A），mf4090，buffer 400 万；配方取 10/9 清单里同集的 run"},
    # 11/02（计划 agent-doc/plan-t1ep1-2026-11-02.md）：ep1 的 1000 万 fc20 走 t1hg2 --episodes 1（seed 43 另加 --seed 43）；
    # mf4090 上 ep1 试手掌位置终止 0.05、0.08（400 万）
    "t1hp2": {"plan": "agent-doc/plan-t1ep1-2026-11-02.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
              "episodes": [1], "parts": {"t1hp2": [1]}, "base": None, "novar": True,
              "run": "t1-ep{ep}-w2048-cap4m-pt05-mf4090", "extra": ["termination.palm_position_threshold=0.05"],
              "desc": "Tier 1 MPPE 榜：ep1 基础配方把手掌位置终止从 0.15 收到 0.05 m（pt05，不加变体 A）seed 42，mf4090，buffer 400 万；配方取 10/9 清单里同集的 run"},
    "t1hp3": {"plan": "agent-doc/plan-t1ep1-2026-11-02.md", "tier": 1, "machine": "mf4090", "cap": "4000000",
              "episodes": [1], "parts": {"t1hp3": [1]}, "base": None, "novar": True,
              "run": "t1-ep{ep}-w2048-cap4m-pt08-mf4090", "extra": ["termination.palm_position_threshold=0.08"],
              "desc": "Tier 1 MPPE 榜：ep1 基础配方把手掌位置终止从 0.15 收到 0.08 m（pt08，不加变体 A）seed 42，mf4090，buffer 400 万；配方取 10/9 清单里同集的 run"},
    # 11/02（计划 agent-doc/plan-t2auc2-2026-11-02.md）：Tier 2 跟得住、自己参考上 ADD-AUC 最低的 4 集加物体朝向项（r3o），只由 Tier 2 的数据挑集
    "t2zo": {"plan": "agent-doc/plan-t2auc2-2026-11-02.md", "tier": 2, "machine": "mf4090", "cap": "4000000",
             "episodes": [10, 0, 26, 21], "parts": {"t2zo": [10, 0], "t2zo2": [26, 21]}, "base": None,
             "run": "t2-ep{ep}-w2048-cap4m-r3o-mf4090", "variant": "r3o",
             "desc": "Tier 2 AUC 榜：跟得住、ADD-AUC 最低的 ep10、0、26、21 用 r3o（变体 A 加物体朝向项）seed 42，mf4090，buffer 400 万，基线取 Tier 2 导出清单"},
    "t2o3": {"plan": "agent-doc/plan-t2auc2-2026-11-02.md", "tier": 2, "machine": "ml40", "cap": None,
             "episodes": [10, 0, 26, 21], "parts": {"t2o3": [10, 0], "t2o3b": [26, 21]}, "base": None,
             "run": "t2-ep{ep}-w2048-cap10m-r3o-ml40", "variant": "r3o",
             "desc": "Tier 2 AUC 榜：同 t2zo 的 4 集，r3o seed 42，ml40，buffer 1000 万（与清单里的 1000 万变体 A 只差物体朝向项），基线取 Tier 2 导出清单"},
    "t2l3": {"plan": "agent-doc/plan-t2auc3-2026-11-02.md", "tier": 2, "machine": "l40n", "cap": None,
             "episodes": [18, 6, 15, 8], "parts": {"t2l3": [18, 6], "t2l3b": [15, 8]}, "base": None,
             "run": "t2-ep{ep}-w2048-cap10m-r3o-l40n", "variant": "r3o",
             "desc": "Tier 2 AUC 榜：再换 4 集，r3o seed 42，第三台 l40n（L40），buffer 1000 万（配方同 t2o3），基线取 Tier 2 导出清单"},
    "t2l4": {"plan": "agent-doc/plan-final-2026-11-03.md", "tier": 2, "machine": "l40n", "cap": None,
             "episodes": [5, 22, 16, 20, 29, 27], "parts": {"t2l4": [5, 22], "t2l4b": [16, 20], "t2l4c": [29, 27]}, "base": None,
             "run": "t2-ep{ep}-w2048-cap10m-r3o-l40n", "variant": "r3o",
             "desc": "Tier 2 AUC 榜：冻结前最后一轮，余下跟得住的 6 集，配方同 t2l3（r3o seed 42，l40n，buffer 1000 万），基线取 Tier 2 导出清单"},
}
VARIANT_P = ["objective.object_position_std=0.03" if v.startswith("objective.object_position_std=") else v for v in VARIANT]   # r3p：只改 σ
R3O_ENV = "V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4"   # r3o 的物体朝向项（计划 agent-doc/plan-r3o-2026-10-15.md 2.1）
T1_CURRENT = "t1_select_r3x.tsv"   # Tier 1 现在交的 30 条（10/9 逐条选）；Tier 2 的在 ml40.tsv、mf4090.tsv
ENV_SMOKE = "V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"


def env_formal(name: str) -> str:
    return f"V2D_WANDB_TAG={name} V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1"


def current_baselines(root: Path, tier: int = 2) -> dict:
    """某一档的 episode → 现在交的 run，每条 episode 恰好一行。Tier 2：两份导出清单合起来；Tier 1：10/9 逐条选的清单。"""
    ex = root / "stages" / "evaluate" / "exports"
    files = [ex / T1_CURRENT] if tier == 1 else [ex / "ml40.tsv", ex / "mf4090.tsv"]
    out = {}
    for f in files:
        for line in f.read_text().splitlines():
            c = line.split("\t")
            if len(c) >= 4 and c[0] == str(tier):
                ep = int(c[1])
                if ep in out:
                    raise SystemExit(f"make_var_queue: Tier {tier} ep{ep} 在导出清单里不止一行（{out[ep]}、{c[2]}）")
                out[ep] = c[2]
    return out


def resolve(part: str, episodes=None) -> tuple:
    """段名 → (清单名, 这一段的 episode)。预设的段不许再给 --episodes；别的段名必须给，且只能取这份清单里的 episode。"""
    name = max((k for k in SETS if part.startswith(k)), key=len, default=None)   # 取最长的：t3p2 不落到 t3p
    if name is None or not part.replace("_", "").isalnum() or part != part.lower():
        raise SystemExit(f"make_var_queue: 段名 {part} 不以 {'、'.join(SETS)} 开头，或含小写字母、数字、下划线以外的字符")
    s = SETS[name]
    if part in s["parts"]:
        if episodes is not None:
            raise SystemExit(f"make_var_queue: {part} 是预设的段，不能再给 --episodes")
        return name, s["parts"][part]
    if not episodes:
        raise SystemExit(f"make_var_queue: {part} 不是预设的段（{'、'.join(s['parts'])}），要用 --episodes 指定")
    bad = sorted(set(episodes) - set(s["episodes"]))
    if bad or len(set(episodes)) != len(episodes):
        raise SystemExit(f"make_var_queue: --episodes 里有不属于 {name} 的 {bad}，或有重复")
    return name, [ep for ep in s["episodes"] if ep in episodes]


def make(part: str, root: Path, episodes=None, seed=None) -> dict:
    name, eps = resolve(part, episodes)
    s = SETS[name]
    tail = re.search(r"_s(\d+)$", part)
    if seed is None and tail:
        raise SystemExit(f"make_var_queue: 段名 {part} 以 _s{tail.group(1)} 结尾，要给同一个 --seed")
    if seed is not None:
        if part in s["parts"]:
            raise SystemExit(f"make_var_queue: {part} 是预设的段，不能改种子；新种子另起一段并给 --episodes")
        if seed == 42 or not 1 <= seed <= 9999:
            raise SystemExit(f"make_var_queue: --seed 要在 1–9999 之间、不是 42（不给就是 seed 42）：{seed}")
        if not tail or int(tail.group(1)) != seed:
            raise SystemExit(f"make_var_queue: 给了 --seed {seed}，段名要以 _s{seed} 结尾：{part}")
    seed = 42 if seed is None else seed
    seed_over = [] if seed == 42 else [f"reset.seed={seed}", f"training.seed={seed}"]
    qdir = root / "stages" / "policy" / "queues"
    if s["tier"] in (1, 2) or s["base"] is None:   # t2x、t3p3、t1x、t1y：基线取这条 episode 现在交的 run
        btier = 1 if s["tier"] == 1 else 2
        cur = current_baselines(root, btier)
        missing = [ep for ep in eps if ep not in cur]
        if missing:
            raise SystemExit(f"make_var_queue: 导出清单里没有 Tier {btier} 的 {missing}")
        bases = {ep: cur[ep] for ep in eps}
        if s.get("variant") == "r3o" or s.get("strip_r3a"):   # 现在交的是变体 A 的 run（如 t1-ep20-w2048-cap4m-r3a-mf4090）就取它的基线配方 run
            bases = {ep: b.replace("-r3a-", "-") for ep, b in bases.items()}
    else:
        bases = {ep: s["base"].format(ep=ep) for ep in eps}
    exp, smoke, formal_lines, cand = [HEADER], [], [], []
    for ep in eps:
        base = bases[ep]
        row, formal = baseline(qdir, base.rsplit("-", 1)[1], base)
        extra = sorted(set(formal["overrides"]) - ALLOWED)
        if extra:
            raise SystemExit(f"make_var_queue: {base} 的覆盖里有计划没说怎么处理的键：{extra}")
        if formal["worlds"] != "2048":
            raise SystemExit(f"make_var_queue: {base} 的世界数是 {formal['worlds']}，不是 2048")
        keep = [o for o in formal["overrides"] if not o.startswith("training.replay.capacity=")]
        cols = list(row[2:8])
        if "tables" in s:   # t3p4、t3q：支撑面按交付版自己的支撑面定，不沿用 Tier 2 基线的列与覆盖（计划审查 1）
            on = ep in s["tables"]
            cols[3] = "true" if on else "false"
            keep = [o for o in keep if o != "scene.support=false"] + ([] if on else ["scene.support=false"])
        over = ([f"training.replay.capacity={s['cap']}"] if s["cap"] else []) + keep + seed_over + ([] if s.get("novar") else VARIANT_P if s.get("variant") == "r3p" else VARIANT)   # novar：基础配方，不加变体 A（t1hr、t1hp、t1hf）
        over += s.get("extra", [])   # t2tm：冒烟与正式都加（计划 agent-doc/plan-t2term-2026-10-20.md）
        dup = sorted({o.split("=", 1)[0] for o in over if sum(x.split("=", 1)[0] == o.split("=", 1)[0] for x in over) > 1})
        if dup:
            raise SystemExit(f"make_var_queue: {name} ep{ep} 的覆盖里有重复的键：{dup}")
        parquet = formal["parquet"]
        if s["tier"] == 3:
            if parquet.count("/runs/motion/tier2/") != 1:
                raise SystemExit(f"make_var_queue: {base} 的 parquet 不在 /runs/motion/tier2/ 下：{parquet}")
            parquet = parquet.replace("/runs/motion/tier2/", f"/runs/motion/{s['motion']}/")
        elif s.get("motion"):   # t1c、t2m：Tier 1、2 段换参考目录（只换接触标记的那份参考，序列名相同；目录名的档位必须与清单相同）
            src_dir = f"/runs/motion/tier{s['tier']}/"
            if not re.fullmatch(rf"tier{s['tier']}_[a-z0-9]+", s["motion"]):
                raise SystemExit(f"make_var_queue: {name} 的参考目录 {s['motion']} 不是 tier{s['tier']}_<后缀>")
            if parquet.count(src_dir) != 1:
                raise SystemExit(f"make_var_queue: {base} 的 parquet 不在 {src_dir} 下：{parquet}")
            parquet = parquet.replace(src_dir, f"/runs/motion/{s['motion']}/")
        run = s["run"].format(ep=ep)
        if seed != 42:
            stem_, mach_ = run.rsplit("-", 1)
            run = f"{stem_}-s{seed}-{mach_}"
        exp.append("\t".join([run, f"smoke-{run}", *cols, s["cap"] or "10000000", str(seed), "no"]))
        tail = f"{parquet} {formal['worlds']}"
        r3o = f"{R3O_ENV} " if s.get("variant") == "r3o" else ""
        formal_lines.append(f"{env_formal(name)} {r3o}{run} {tail} {STEPS} {' '.join(over)}")
        smoke.append(f"{ENV_SMOKE} {r3o}smoke-{run} {tail} {SMOKE_STEPS} {' '.join(over + s.get('smoke_extra', []))}")
        cand.append(f"{s['tier']}\t{ep}\t{run}\t{STEPS}")
    head = (f"# {s['desc']}；这一段 {part}：{'、'.join(f'ep{e}' for e in eps)}；计划 {s['plan']}；变体 A = {' '.join(VARIANT)}；"
            + (f"本段改种子：种子 {seed}（{' '.join(seed_over)}）；" if seed_over else "")
            + (f"r3p = {' '.join(VARIANT_P)}（只改 σ）；" if s.get("variant") == "r3p" else "")
            + (f"冒烟另加 {' '.join(s['smoke_extra'])}（只加在冒烟行）；" if s.get("smoke_extra") else "")
            + (f"另加 {' '.join(s['extra'])}（冒烟与正式都加）；" if s.get("extra") else "")
            + ("本段不加变体 A（基础配方）；" if s.get("novar") else "")
            + (f"r3o = {'基础配方' if s.get('novar') else '变体 A'}加 {R3O_ENV}；训练挂 v040_p185_nfg_oo" if s.get("variant") == "r3o" else "训练挂 v040_p185_nfg")
            + "（链的环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>，不写进队列行）。make_var_queue.py 生成，不要手改。")
    stem = f"{s['machine']}_{part}"
    return {
        qdir / f"{stem}.expect.tsv": "\n".join(exp) + "\n",
        qdir / f"{stem}_smoke.txt": "\n".join([head, *smoke]) + "\n",
        qdir / f"{stem}.txt": "\n".join([head, *formal_lines]) + "\n",
        root / "stages" / "evaluate" / "exports" / f"cand_t{s['tier']}_{part}_{s['machine']}.tsv":
            "\n".join([f"# {s['desc']}，段 {part} 的候选导出（计划 {s['plan']}）：档位 episode run 步数。"
                       + (f"本段改种子：种子 {seed}（{' '.join(seed_over)}）。" if seed_over else "") + "make_var_queue.py 生成，不要手改。", *cand]) + "\n",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--episodes", type=lambda x: [int(v) for v in x.split(",") if v.strip()], default=None)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    for path, text in make(a.part, Path(a.root), a.episodes, a.seed).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"写 {path.relative_to(a.root)}（{text.count(chr(10))} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
