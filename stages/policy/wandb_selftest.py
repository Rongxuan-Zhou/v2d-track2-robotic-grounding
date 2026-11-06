"""W&B 端到端小测（在 FlashCHORD 容器里跑）：按训练时的方式初始化（entity/project/group/name 显式传，
id 与 tags 走环境变量），记几步指标，传一个 5 MB artifact，计时 finish()，再用 API 回读核对。
用法：WANDB_API_KEY、WANDB_RUN_ID、WANDB_TAGS 由宿主机按名字注入；python wandb_selftest.py <entity> <project> <group>"""
import os
import sys
import tempfile
import time

import wandb

entity, project, group = sys.argv[1:4]
rid = os.environ["WANDB_RUN_ID"]
t0 = time.perf_counter()
run = wandb.init(entity=entity, project=project, group=group, name=rid, config={"selftest": True},
                 mode="online", dir=tempfile.mkdtemp())
t_init = time.perf_counter() - t0
for step in (1000, 2000, 3000):
    run.log({"selftest/value": step / 1000.0}, step=step)
art = wandb.Artifact(name=f"run-{run.id}-checkpoint", type="model")
path = os.path.join(tempfile.mkdtemp(), "policy_3000.safetensors")
with open(path, "wb") as f:
    f.write(os.urandom(5 * 1024 * 1024))
art.add_file(path)
run.log_artifact(art, aliases=["latest"])
t1 = time.perf_counter()
run.finish()
t_finish = time.perf_counter() - t1
api = wandb.Api(timeout=60)
r = api.run(f"{entity}/{project}/{rid}")
# 注意：wandb 0.22.2 的 scan_history(keys=...) 回读会把每行重复一次且丢 _step（客户端问题，数据本身正确），
# 所以用不带 keys 的 scan_history 并核对 (步数, 值)
hist = [(row["_step"], row["selftest/value"]) for row in r.scan_history() if "selftest/value" in row]
arts = [a.name for a in r.logged_artifacts()]
print(f"init {t_init:.1f}s；finish（含 5 MB 上传）{t_finish:.1f}s")
print(f"id={r.id} name={r.name} group={r.group} tags={r.tags} state={r.state}")
print(f"回读指标 {hist}；artifacts {arts}")
print(f"URL {r.url}")
ok = hist == [(1000, 1), (2000, 2), (3000, 3)] and r.group == group and set(os.environ.get("WANDB_TAGS", "").split(",")) <= set(r.tags) and arts
print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
sys.exit(0 if ok else 1)
