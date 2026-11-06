# 搭提交容器的 /submission 目录（Mac 上、仓库根执行；官方要求见 kit update_7e1fdd00 的 CONTAINER_SUBMISSION.md）。
# 三档各一份计分文件、各一个 commit；官方 submission.json 只有一个 code commit，所以每档一份 manifest
# （submission.tier1.json …，各自的 code_tierN/ 是对应 commit 的干净 checkout），submission.json 是 Tier 1 那份。
# 本脚本放好代码、kit、已交文件与回执、rollout、帧参考、manifest、run.sh、Dockerfile、README；
# checkpoint 不在 Mac 上，写 weights_<机器>.tsv（标签、集、sha256、容器内路径、训练机上的路径），由 container_weights_remote.sh 在各机器上取。
# 用法：python3 tools/container_stage.py <输出目录>（目录不能已存在）。最后一行 CONTAINER_STAGE_DONE。
import json
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
SUB = OUT / "submission"
KIT_OLD = Path("third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit")
KIT_NEW = Path("third_party/v2d_submission_kit/update_7e1fdd00/v2d_submission_kit")
REPO = "https://github.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/"
TIERS = {
    "tier1": dict(track="track_2_tier1", csv="track2_tier1_t1x1102.csv", sha="b59f6a6cb3f97fc53ebdb28e2e8496534478c108",
                  ids={"add_auc": 57920552, "spider_sr": 57920554, "maniptrans_sr": 57920555, "mppe_cm": 57920556},
                  rollouts="runs/submission/t1x1102/t1", frefs=None),
    "tier2": dict(track="track_2_tier2", csv="track2_tier2_select_b1014.csv", sha="c4f248dc60f39f4c7b03c535fbace6ae6b422bc9",
                  ids={"add_auc": None, "spider_sr": 57197030, "maniptrans_sr": 57197033, "mppe_cm": 57197036},
                  rollouts="runs/submission/select_t2_b1014/t2", frefs="runs/submission/frame_refs_t2"),
    "tier3": dict(track="track_2_tier3", csv="track2_tier3_t3m1029a.csv", sha="2175e83e0c13b9ae2391d724f3d1c80c392b73d3",
                  ids={"add_auc": 57783386, "spider_sr": 57783387, "maniptrans_sr": 57783388, "mppe_cm": 57783392},
                  rollouts="runs/submission/t3m1029a/t3", frefs="runs/submission/t3m1029a/frefs"),
}
EVAL_PKG = "/workspace/src/flash_chord/evaluation"


def kaggle_ids():
    """Tier 2 AUC 的 b1014 提交 ID 从计分清单之外取：kaggle CLI 列 tier2-auc 的提交。"""
    k = str(Path.home() / ".cache/uv/archive-v0/uvfKxZAYaAMiPZYi/bin/kaggle")
    out = subprocess.run([k, "competitions", "submissions", "v2d-challenge-track2-tier2-auc", "-v"], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if "track2_tier2_select_b1014.csv" in line:
            return int(line.split(",")[0])
    sys.exit("没找到 b1014 在 tier2-auc 上的提交 ID")


def per_episode():
    rows = []
    for line in open("results/freeze/counted_inventory.txt", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 5 and p[0].endswith(".csv") and p[1].isdigit():
            rows.append(p)
    return rows


SUB.mkdir(parents=True)
TIERS["tier2"]["ids"]["add_auc"] = kaggle_ids()
subprocess.run(["git", "bundle", "create", str(OUT / "repo.bundle"), "--all"], check=True, capture_output=True)
for label, t in TIERS.items():
    code = SUB / f"code_{label}"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(OUT / "repo.bundle"), str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "checkout", "-q", t["sha"]], check=True)
    subprocess.run(["git", "-C", str(code), "remote", "set-url", "origin", "https://github.com/Manifold-ai/maniphys_lab_v2d_challenge.git"], check=True)
    dirty = subprocess.run(["git", "-C", str(code), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    assert not dirty, (label, dirty[:200])
(OUT / "repo.bundle").unlink()
shutil.copytree(KIT_OLD, SUB / "kit")
shutil.copytree(KIT_NEW, SUB / "verifier_kit")
(SUB / "submitted").mkdir()
for label, t in TIERS.items():
    for f in Path("runs/submission/csv").glob(t["csv"] + "*"):
        shutil.copy2(f, SUB / "submitted" / f.name)
    shutil.copytree(t["rollouts"], SUB / "outputs" / label / "rollouts")
    if t["frefs"]:
        shutil.copytree(t["frefs"], SUB / "frame_references" / label)

rows = per_episode()
weights = {}
for label, t in TIERS.items():
    mine = [r for r in rows if r[0] == t["csv"]]
    assert len(mine) == 30, (label, len(mine))
    ckpts = {}
    for name, ep, sha, host, path in mine:
        rel = f"weights/{label}/episode_{int(ep):06d}.safetensors"
        ckpts[str(int(ep))] = rel
        src = path.split(" | ")[0].replace("/repo/", "", 1)
        if host == "?":  # t1x1018 的三集不在计分文件里；这里不该出现
            sys.exit(f"{label} ep{ep} 没有 checkpoint 路径")
        weights.setdefault(host, []).append((label, ep, sha, rel, src))
    entry = {"label": label, "track": t["track"], "file": f"submitted/{t['csv']}",
             "kaggle_submission_ids": t["ids"],
             "evaluator": {"script": "kit/flash_chord/scripts/evaluate_policy.py", "evaluation_package": EVAL_PKG},
             "checkpoints": ckpts, "outputs": f"outputs/{label}/rollouts"}
    if t["frefs"]:
        entry["frame_references"] = f"frame_references/{label}"
    manifest = {"schema": "v2d_container_v1", "team": "Maniphys Lab",
                "code": {"path": f"code_{label}", "commit_url": REPO + t["sha"]}, "kit": "kit", "submissions": [entry]}
    (SUB / f"submission.{label}.json").write_text(json.dumps(manifest, indent=1) + "\n")
shutil.copy2(SUB / "submission.tier1.json", SUB / "submission.json")
for host, items in weights.items():
    with open(OUT / f"weights_{host}.tsv", "w") as f:
        for it in items:
            f.write("\t".join(it) + "\n")
    print(f"{host}\t{len(items)} 个 checkpoint")
print("CONTAINER_STAGE_DONE")
