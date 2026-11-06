"""tools/hf_archive.py 里不连网的部分的行为测试：挑文件规则与删除（暂无契约 ID：运维工具）。只用标准库，Mac 与训练机都能跑。

在临时目录里造假的 runs/flash_chord：
  - smoke-a：有 DONE、两份 state → 两份都挑；删光后写 STATE_PRUNED（取最大的 policy 步数）
  - t1-b：有 DONE、三份 state，最大的一份 state 写坏了 → 留第二大的完整一份，其余两份挑（含写坏的那份）
  - t1-c：没有 DONE → 不挑
  - t2-ep0-x：有 DONE → 名字含 t2-，不挑
  - smoke-t2-ep0-x：有 DONE → 不挑
删除：核对或抽查没通过一个都不删；演习不删；--yes 先写 ARCHIVED.json 再删；本地文件被改过的跳过；
中途断了（文件已删、STATE_PRUNED 没写）重跑能补上 STATE_PRUNED；暂存目录的硬链接一起清掉。

policy 中间份（plan --kind policy，K=9），另造一套 runs/flash_chord 与 runs/exports/a.tsv：
  - t1-p：14 份 policy、步 14 有 state，导出表里有步 2 → 挑 1、3、4、5
  - t2-ep1-p：12 份 → 名字带 t2- 也算，挑 1、2、3
  - t1-q：12 份，最新一份写坏了 → 整个 run 不动
  - t1-r：没有 DONE → 不动；t1-s：正好 9 份 → 没得挑
  - t1-u：12 份，步 2 与步 12 有 state → 步 2 的 policy 留着，挑 1、3
  - smoke-v：11 份，STATE_PRUNED 记步 1 → 挑 2
删除：有训练链在跑一个都不删；按删除时的文件重算保护（计划之后新导出的步数跳过）；ARCHIVED.json 追加不覆盖；
最新一份 policy 在删除前坏了，整个 run 跳过；导出输入清单 stages/evaluate/exports/*.tsv 里的步数也受保护；
抽查 --per-run 每条 run 挑 1 个。
最终 state（plan --kind final，第 3 批，agent-doc/plan-hf-archive-batch3-2026-10-10.md），另造一套 runs/flash_chord：
  - t1-f：DONE，state 10、20 → final 挑 state_20（state 规则另挑 state_10）；t2-ep3-f：DONE、state 30 → 名字带 t2- 也挑
  - t1-m：DONE，state 10 → 挑；smoke-f：冒烟不看；t1-g：没有 DONE；t1-h：policy_20 写坏 → 整个 run 不动；t1-k：只有 policy
  - --kind 连用（state,final）是并集；重复、不认识、空的 --kind 报错；有训练链在跑不挑
删除：有训练链在跑一个都不删；计划之后多了更新的 state、或同步数的 policy 坏了，这份就不是最终的，跳过；
同步数的 policy 留着；ARCHIVED.json 记下；正式 run 不写 STATE_PRUNED。
抽查（第 3 批审查 m2、m4）：--per-run 时有最终 state 的 run 固定挑它、再从其余文件里随机补 1 个；下载前可用空间不够、
下载后 xet 缓存涨过上限，都停在抽查，不写抽查结果（假 hub 的下载可以往 HF_HOME/xet 里写字节，模拟缓存增长）。
hubsubset（第 3 批撞上私有存储额度后加）：只挑仓库里已有、大小与 sha256 都一致的行；仓库里有同路径但不一致的文件就退出码 1；
--rest-out 写出仓库里还没有的行。流程脚本的 check 步骤：不调 upload，核对、抽查之后停在删除前；流程脚本退出时删掉 runs/hf_spotcheck。
readme 子命令与 upload 的闸：PYTHONPATH 前面放一个假的 huggingface_hub（状态存在 JSON 里）。仓库没有 README.md 时 upload 停、
不留暂存目录；readme 建私有仓库（--visibility public 遇到不存在的仓库就停，不新建公开仓库）、传 README、下载回来比 sha256，对不上报 HF_README_FAIL；有 README 后 upload 能走到上传；
仓库是公开的 upload 停。
可见性（第 3 批第 4 版，用户同意公开归档仓库后加）：whoami、readme、upload 的 --visibility 与仓库实际可见性对不上就停或报 FAIL；
visibility 子命令只碰本工具建的归档仓库（根目录有 README.md、至少一个 <机器名>/runs/flash_chord/<run>/(state|policy)_<步数>.safetensors），
--confirm 与 --repo 不同就停，不加 --yes 不改，已是目标可见性不重复改，改完读回来对不上报 HF_VISIBILITY_FAIL，旧版 huggingface_hub 用 update_repo_visibility；
流程脚本把 V2D_HF_VISIBILITY 原样传给 upload，值不对退出码 2。
hf_archive_run.sh（只在 Linux 上测，用 GNU find）：runs/tools/ 下放桩工具（upload、verify、spotcheck 是假的，其余转给真工具），
PATH 前面放假 nvidia-smi 与假 pgrep（训练机上有真训练链在跑时，真 pgrep 会让删除步骤停下；10/11 第 3 批测试实际遇到过）。verify 步骤停在删除前、什么都不删；delete 步骤缺核对结果、有训练链、GPU 上有进程都停，
正常时先重新核对再删；删除时多删一个清单外的文件，全量比对要报 HF_ARCHIVE_FILES_CHECK_FAILED。
用法：python3 tools/test_hf_archive.py
"""
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "hf_archive.py"
RUNNER = Path(__file__).resolve().parent / "hf_archive_run.sh"


def ckpt(p: Path, n: int = 4, broken: bool = False) -> None:
    data = b"\0" * (4 * n)
    head = json.dumps({"w": {"dtype": "F32", "shape": [n], "data_offsets": [0, len(data)]}}).encode()
    blob = struct.pack("<Q", len(head)) + head + data
    p.write_bytes(blob[:-3] if broken else blob)


def run(root: Path, *args: str, rc: int = 0, env: dict | None = None) -> str:
    r = subprocess.run([sys.executable, str(TOOL), *args], cwd=root, capture_output=True, text=True, timeout=60, env=env)
    assert r.returncode == rc, (args, r.returncode, r.stdout, r.stderr)
    return r.stdout + r.stderr


def build(root: Path) -> None:
    R = root / "runs/flash_chord"
    for name, done, steps, broken_last in (("smoke-a", True, (10, 20), False), ("t1-b", True, (10, 20, 30), True),
                                           ("t1-c", False, (10, 20), False), ("t2-ep0-x", True, (10, 20), False),
                                           ("smoke-t2-ep0-x", True, (10,), False)):
        d = R / name
        d.mkdir(parents=True)
        if done:
            (d / "DONE").write_text("x\n")
        for i, s in enumerate(steps):
            last = i == len(steps) - 1
            ckpt(d / f"state_{s}.safetensors", n=8 + s, broken=broken_last and last)
            ckpt(d / f"policy_{s}.safetensors", n=2)


def verified_from_plan(root: Path, manifest: str, out: str) -> list:
    rows = json.loads((root / manifest).read_text())
    v = [{**r, "repo": "u/r", "path_in_repo": f"m/runs/flash_chord/{r['run']}/{r['file']}", "hub_size": r["size"],
          "hub_sha256": r["sha256"], "match": True} for r in rows]
    (root / out).write_text(json.dumps(v))
    return v


def test_plan_rules() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build(root)
        run(root, "plan", "--out", "m.json")
        rows = json.loads((root / "m.json").read_text())
        got = sorted((r["run"], r["file"]) for r in rows)
        assert got == [("smoke-a", "state_10.safetensors"), ("smoke-a", "state_20.safetensors"),
                       ("t1-b", "state_10.safetensors"), ("t1-b", "state_30.safetensors")], got


def test_delete_gates_and_effects() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build(root)
        run(root, "plan", "--out", "m.json")
        v = verified_from_plan(root, "m.json", "v.json")
        (root / "spot_bad.json").write_text(json.dumps({"ok": False, "checked": [{"path_in_repo": v[0]["path_in_repo"]}]}))
        (root / "spot.json").write_text(json.dumps({"ok": True, "checked": [{"path_in_repo": v[0]["path_in_repo"], "match": True}]}))
        assert "一个都不删" in run(root, "delete", "--verified", "v.json", "--spotcheck", "spot_bad.json", rc=1)
        bad = [dict(r, match=False) if i == 0 else r for i, r in enumerate(v)]
        (root / "v_bad.json").write_text(json.dumps(bad))
        assert "一个都不删" in run(root, "delete", "--verified", "v_bad.json", "--spotcheck", "spot.json", rc=1)
        out = run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json")  # 演习
        assert "演习" in out and (root / "runs/flash_chord/smoke-a/state_10.safetensors").exists(), out
        stage = root / "runs/hf_stage/m/runs/flash_chord/smoke-a"  # 假装上传时留下的硬链接
        stage.mkdir(parents=True)
        os.link(root / "runs/flash_chord/smoke-a/state_10.safetensors", stage / "state_10.safetensors")
        ckpt(root / "runs/flash_chord/t1-b/state_10.safetensors", n=99)  # 本地被改过：应跳过
        out = run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json", "--yes")
        R = root / "runs/flash_chord"
        assert not (R / "smoke-a/state_10.safetensors").exists() and not (R / "smoke-a/state_20.safetensors").exists(), out
        assert (R / "smoke-a/STATE_PRUNED").read_text().strip() == "20", out
        assert (R / "t1-b/state_10.safetensors").exists() and "不一样了" in out, out
        assert not (R / "t1-b/state_30.safetensors").exists() and (R / "t1-b/state_20.safetensors").exists(), out
        arch = json.loads((R / "smoke-a/ARCHIVED.json").read_text())
        assert {f["file"] for f in arch[0]["files"]} == {"state_10.safetensors", "state_20.safetensors"}, arch
        for name in ("t1-c", "t2-ep0-x", "smoke-t2-ep0-x"):
            assert len(list((R / name).glob("state_*"))) >= 1 and not (R / name / "ARCHIVED.json").exists(), name
        assert not (root / "runs/hf_stage").exists(), "暂存目录没清掉"


def test_rerun_after_interrupted_delete_writes_state_pruned() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build(root)
        run(root, "plan", "--out", "m.json")
        verified_from_plan(root, "m.json", "v.json")
        v = json.loads((root / "v.json").read_text())
        (root / "spot.json").write_text(json.dumps({"ok": True, "checked": [{"path_in_repo": v[0]["path_in_repo"], "match": True}]}))
        R = root / "runs/flash_chord/smoke-a"
        for p in R.glob("state_*"):  # 模拟上一轮删完文件、还没写 STATE_PRUNED 就断了
            p.unlink()
        out = run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json", "--yes")
        assert (R / "STATE_PRUNED").read_text().strip() == "20", out


def build_policy(root: Path) -> None:
    R = root / "runs/flash_chord"

    def mk(name, done, pol_steps, state_steps=(), broken_last=False, state_pruned=None):
        d = R / name
        d.mkdir(parents=True)
        if done:
            (d / "DONE").write_text("x\n")
        pol_steps = list(pol_steps)
        for i, s in enumerate(pol_steps):
            ckpt(d / f"policy_{s}.safetensors", n=2 + s, broken=broken_last and i == len(pol_steps) - 1)
        for s in state_steps:
            ckpt(d / f"state_{s}.safetensors", n=8 + s)
        if state_pruned is not None:
            (d / "STATE_PRUNED").write_text(f"{state_pruned}\n")

    mk("t1-p", True, range(1, 15), state_steps=(14,))
    mk("t2-ep1-p", True, range(1, 13))
    mk("t1-q", True, range(1, 13), broken_last=True)
    mk("t1-r", False, range(1, 16))
    mk("t1-s", True, range(1, 10))
    mk("t1-u", True, range(1, 13), state_steps=(2, 12))
    mk("smoke-v", True, range(1, 12), state_pruned=1)
    E = root / "runs/exports"
    E.mkdir(parents=True)
    (E / "a.tsv").write_text("tier\tepisode\trun\tstep\tstatus\n1\t0\tt1-p\t2\tok\n1\t0\tt1-p\tnot-a-step\tok\n")


POLICY_PICK = {("t1-p", 1), ("t1-p", 3), ("t1-p", 4), ("t1-p", 5), ("t2-ep1-p", 1), ("t2-ep1-p", 2), ("t2-ep1-p", 3),
               ("t1-u", 1), ("t1-u", 3), ("smoke-v", 2)}


def spot_ok(root: Path, v: list, name: str = "spot.json") -> None:
    (root / name).write_text(json.dumps({"ok": True, "checked": [{"path_in_repo": v[0]["path_in_repo"], "match": True}]}))


def test_policy_plan_rules() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build_policy(root)
        out = run(root, "plan", "--kind", "policy", "--out", "pm.json")
        rows = json.loads((root / "pm.json").read_text())
        assert {(r["run"], r["step"]) for r in rows} == POLICY_PICK, (rows, out)
        assert all(r["kind"] == "policy" and r["keep"] == 9 and r["file"] == f"policy_{r['step']}.safetensors" for r in rows), rows
        assert "t1-q" in out and "不完整" in out, out
        (root / "runs/.chain-active-x").write_text("")
        assert "训练链" in run(root, "plan", "--kind", "policy", "--out", "pm2.json", rc=1)


def test_policy_delete() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build_policy(root)
        R = root / "runs/flash_chord"
        (R / "t1-p/ARCHIVED.json").write_text(json.dumps([{"time": "old", "files": [{"file": "state_0.safetensors"}]}]))
        run(root, "plan", "--kind", "policy", "--out", "pm.json")
        v = verified_from_plan(root, "pm.json", "v.json")
        spot_ok(root, v)
        (root / "runs/.chain-active-x").write_text("")
        assert "一个都不删" in run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json", "--yes", rc=1)
        assert (R / "t1-p/policy_1.safetensors").exists()
        (root / "runs/.chain-active-x").unlink()
        with (root / "runs/exports/a.tsv").open("a") as f:  # 计划之后又导出了步 3：删除时要跳过
            f.write("1\t0\tt1-p\t3\tok\n")
        ckpt(R / "t2-ep1-p/policy_1.safetensors", n=99)  # 本地被改过：跳过
        stage = root / "runs/hf_stage/m/runs/flash_chord/t1-u"
        stage.mkdir(parents=True)
        os.link(R / "t1-u/policy_1.safetensors", stage / "policy_1.safetensors")
        out = run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json", "--yes")
        left = {(d.name, int(p.stem.split("_")[1])) for d in R.iterdir() for p in d.glob("policy_*.safetensors")}
        deleted = {("t1-p", 1), ("t1-p", 4), ("t1-p", 5), ("t2-ep1-p", 2), ("t2-ep1-p", 3), ("t1-u", 1), ("t1-u", 3), ("smoke-v", 2)}
        assert not (deleted & left), (deleted & left, out)
        assert {("t1-p", 3), ("t1-p", 2), ("t2-ep1-p", 1), ("t1-u", 2), ("smoke-v", 1)} <= left, out
        assert "受保护" in out and "不一样了" in out, out
        for name, n in (("t1-p", 9), ("t2-ep1-p", 9), ("t1-u", 9), ("smoke-v", 9)):  # 最新 9 份都在
            steps = sorted(s for r, s in left if r == name)
            assert steps[-n:] == list(range(max(steps) - n + 1, max(steps) + 1)), (name, steps)
        arch = json.loads((R / "t1-p/ARCHIVED.json").read_text())
        assert len(arch) == 2 and arch[0]["time"] == "old", arch
        assert {f["file"] for f in arch[1]["files"]} == {"policy_1.safetensors", "policy_4.safetensors", "policy_5.safetensors"}, arch
        assert (R / "smoke-v/STATE_PRUNED").read_text().strip() == "1"
        assert not (R / "t1-p/STATE_PRUNED").exists()
        assert not (root / "runs/hf_stage").exists(), "暂存目录没清掉"


def test_policy_delete_skips_run_whose_newest_policy_broke() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build_policy(root)
        R = root / "runs/flash_chord"
        run(root, "plan", "--kind", "policy", "--out", "pm.json")
        v = verified_from_plan(root, "pm.json", "v.json")
        spot_ok(root, v)
        ckpt(R / "t1-u/policy_12.safetensors", n=14, broken=True)
        out = run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json", "--yes")
        assert (R / "t1-u/policy_1.safetensors").exists() and (R / "t1-u/policy_3.safetensors").exists(), out
        assert "跳过 t1-u" in out and not (R / "t1-u/ARCHIVED.json").exists(), out


def test_export_input_lists_protect() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build_policy(root)
        L = root / "stages/evaluate/exports"
        L.mkdir(parents=True)
        (L / "l.tsv").write_text("# 档位 episode run 步数\n1\t1\tt2-ep1-p\t3\n1\t1\tt2-ep1-p\tbad\n")
        run(root, "plan", "--kind", "policy", "--out", "pm.json")
        got = {(r["run"], r["step"]) for r in json.loads((root / "pm.json").read_text())}
        assert got == POLICY_PICK - {("t2-ep1-p", 3)}, got


def test_spot_pick() -> None:
    spec = importlib.util.spec_from_file_location("hf_archive_under_test", TOOL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows = [{"run": f"r{i % 3}", "file": f"f{i}"} for i in range(30)]
    per_run = m.spot_pick(rows, 5, True, 7)
    assert sorted(r["run"] for r in per_run) == ["r0", "r1", "r2"], per_run
    assert len(m.spot_pick(rows, 5, False, 7)) == 5 and len(m.spot_pick(rows, 99, False, 7)) == 30
    rows = ([{"run": "a", "kind": "policy", "file": f"p{i}"} for i in range(10)] + [{"run": "a", "kind": "final", "file": "s"}]
            + [{"run": "b", "kind": "policy", "file": f"q{i}"} for i in range(10)] + [{"run": "c", "kind": "final", "file": "t"}])
    for seed in range(20):
        got = m.spot_pick(rows, 5, True, seed)
        kinds = sorted((r["run"], r["kind"]) for r in got)
        assert kinds == [("a", "final"), ("a", "policy"), ("b", "policy"), ("c", "final")], (seed, got)


def build_final(root: Path) -> None:
    R = root / "runs/flash_chord"
    for name, done, st, pol, broken_pol in (("t1-f", True, (10, 20), (10, 20), None), ("t2-ep3-f", True, (30,), (30,), None),
                                             ("t1-m", True, (10,), (10,), None), ("smoke-f", True, (10,), (10,), None),
                                             ("t1-g", False, (10,), (10,), None), ("t1-h", True, (10, 20), (10, 20), 20),
                                             ("t1-k", True, (), (10, 20), None)):
        d = R / name
        d.mkdir(parents=True)
        if done:
            (d / "DONE").write_text("x\n")
        for s in st:
            ckpt(d / f"state_{s}.safetensors", n=8 + s)
        for s in pol:
            ckpt(d / f"policy_{s}.safetensors", n=2 + s, broken=s == broken_pol)


def test_final_plan_rules() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build_final(root)
        out = run(root, "plan", "--kind", "final", "--out", "fm.json")
        rows = json.loads((root / "fm.json").read_text())
        assert sorted((r["run"], r["file"], r["kind"], r["step"]) for r in rows) == [
            ("t1-f", "state_20.safetensors", "final", 20), ("t1-m", "state_10.safetensors", "final", 10),
            ("t2-ep3-f", "state_30.safetensors", "final", 30)], (rows, out)
        assert "跳过 t1-h" in out, out
        out = run(root, "plan", "--kind", "state,final", "--out", "sf.json")
        got = sorted((r["run"], r["file"], r["kind"]) for r in json.loads((root / "sf.json").read_text()))
        assert got == [("smoke-f", "state_10.safetensors", "smoke"), ("t1-f", "state_10.safetensors", "formal"),
                       ("t1-f", "state_20.safetensors", "final"), ("t1-h", "state_20.safetensors", "formal"),
                       ("t1-m", "state_10.safetensors", "final"), ("t2-ep3-f", "state_30.safetensors", "final")], (got, out)
        assert "合计 6 个文件" in out, out
        for bad in ("final,final", "final,bogus", ""):
            run(root, "plan", "--kind", bad, "--out", "x.json", rc=2)
        assert not (root / "x.json").exists()
        (root / "runs/.chain-active-x").write_text("")
        assert "训练链" in run(root, "plan", "--kind", "final", "--out", "fm2.json", rc=1)
        assert not (root / "fm2.json").exists()


def test_final_delete() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build_final(root)
        R = root / "runs/flash_chord"
        run(root, "plan", "--kind", "state,final", "--out", "sf.json")
        v = verified_from_plan(root, "sf.json", "v.json")
        spot_ok(root, v)
        (root / "runs/.chain-active-x").write_text("")
        assert "最终 state 一个都不删" in run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json", "--yes", rc=1)
        assert (R / "t1-f/state_20.safetensors").exists() and (R / "t1-f/state_10.safetensors").exists()
        (root / "runs/.chain-active-x").unlink()
        ckpt(R / "t2-ep3-f/state_40.safetensors", n=48)  # 计划之后多了更新的一对：state_30 不再是最终的
        ckpt(R / "t2-ep3-f/policy_40.safetensors", n=42)
        ckpt(R / "t1-m/policy_10.safetensors", n=12, broken=True)  # 计划之后同步数的 policy 坏了
        out = run(root, "delete", "--verified", "v.json", "--spotcheck", "spot.json", "--yes")
        assert not list((R / "t1-f").glob("state_*")), out
        assert (R / "t1-f/policy_10.safetensors").exists() and (R / "t1-f/policy_20.safetensors").exists(), out
        assert (R / "t2-ep3-f/state_30.safetensors").exists() and (R / "t1-m/state_10.safetensors").exists(), out
        assert out.count("已不是步数最大") == 2, out
        arch = json.loads((R / "t1-f/ARCHIVED.json").read_text())
        assert len(arch) == 1 and {f["file"] for f in arch[0]["files"]} == {"state_10.safetensors", "state_20.safetensors"}, arch
        for name in ("t2-ep3-f", "t1-m", "t1-g", "t1-k"):
            assert not (R / name / "ARCHIVED.json").exists(), name
        assert not (R / "t1-f/STATE_PRUNED").exists(), out
        assert not (R / "smoke-f/state_10.safetensors").exists() and (R / "smoke-f/STATE_PRUNED").read_text().strip() == "10", out
        assert (R / "t1-g/state_10.safetensors").exists() and (R / "t1-h/state_10.safetensors").exists(), out


FAKE_HUB = """import json, os
from pathlib import Path
S = Path(os.environ["FAKE_HF_STATE"])


def _load():
    return json.loads(S.read_text())


def _save(d):
    S.write_text(json.dumps(d))


class _Info:
    def __init__(self, private):
        self.private = private
        self.siblings = []


class HfApi:
    def whoami(self):
        return {"name": "u", "auth": {"accessToken": {"role": "write"}}}

    def repo_exists(self, repo_id, repo_type=None):
        return _load()["exists"]

    def file_exists(self, repo_id, filename, repo_type=None):
        return filename in _load()["files"]

    def repo_info(self, repo_id, repo_type=None, files_metadata=False):
        return _Info(_load()["private"])

    def list_repo_files(self, repo_id, repo_type=None):
        d = _load()
        return list(d["files"]) + list(d.get("tree", {}))

    def _set_private(self, private, key):
        d = _load()
        d[key] = d.get(key, 0) + 1
        if not d.get("ignore_update"):
            d["private"] = private
        _save(d)

    def update_repo_settings(self, repo_id, private=None, repo_type=None):
        self._set_private(private, "settings_calls")

    def update_repo_visibility(self, repo_id, private=False, repo_type=None):
        self._set_private(private, "visibility_calls")

    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
        d = _load()
        if not d["exists"]:
            d["exists"], d["private"] = True, private
        _save(d)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        d = _load()
        d["files"][path_in_repo] = Path(path_or_fileobj).read_text()
        _save(d)

    def list_repo_tree(self, repo_id, repo_type=None, path_in_repo=None, recursive=False, expand=False):
        class _F:
            pass
        out = []
        for path, (size, sha) in _load().get("tree", {}).items():
            f = _F()
            f.path, f.size, f.lfs = path, size, _F()
            f.lfs.sha256 = sha
            out.append(f)
        return out

    def upload_large_folder(self, **kw):
        d = _load()
        d["large_folder_calls"] = d.get("large_folder_calls", 0) + 1
        _save(d)


def hf_hub_download(repo_id, filename, repo_type=None, local_dir=None, force_download=False):
    d = _load()
    p = Path(local_dir) / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(d["files"][filename] + d.get("corrupt", ""))
    if d.get("cache_growth"):
        c = Path(os.environ["HF_HOME"]) / "xet"
        c.mkdir(parents=True, exist_ok=True)
        with (c / "chunk.bin").open("ab") as f:
            f.write(b"0" * d["cache_growth"])
    return str(p)


if _load().get("old_api"):  # 模拟没有 update_repo_settings 的旧版 huggingface_hub
    del HfApi.update_repo_settings
"""


def fake_hub(root: Path) -> dict:
    hub = root / "fakehub/huggingface_hub"
    hub.mkdir(parents=True)
    (hub / "__init__.py").write_text(FAKE_HUB)
    (root / "hub_state.json").write_text(json.dumps({"exists": False, "private": True, "files": {}}))
    env = {k: v for k, v in os.environ.items() if k != "HF_ENDPOINT"}
    return {**env, "PYTHONPATH": str(root / "fakehub"), "FAKE_HF_STATE": str(root / "hub_state.json"),
            "HF_TOKEN": "x", "https_proxy": "http://proxy.invalid:1"}


def hub_state(root: Path, **change) -> dict:
    st = json.loads((root / "hub_state.json").read_text())
    if change:
        st.update(change)
        (root / "hub_state.json").write_text(json.dumps(st))
    return st


def test_readme_and_upload_guard() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build(root)
        run(root, "plan", "--out", "m.json")
        env = fake_hub(root)
        up = ("upload", "--manifest", "m.json", "--repo", "u/r", "--prefix", "m")
        out = run(root, *up, rc=1, env=env)
        assert "没有 README.md" in out and not (root / "runs/hf_stage").exists(), out
        (root / "README.md").write_text("# r\n")
        out = run(root, "readme", "--repo", "u/r", "--file", "README.md", "--visibility", "public", rc=1, env=env)
        assert "不新建公开仓库" in out and not hub_state(root)["exists"], out  # 第 4 版审查 m3
        out = run(root, "readme", "--repo", "u/r", "--file", "README.md", env=env)
        st = hub_state(root)
        assert "HF_README_OK" in out and st["exists"] and st["private"] and st["files"]["README.md"] == "# r\n", (out, st)
        assert not (root / "runs/hf_readme_check").exists()
        hub_state(root, corrupt="x")
        assert "HF_README_FAIL" in run(root, "readme", "--repo", "u/r", "--file", "README.md", rc=1, env=env)
        hub_state(root, corrupt="")
        out = run(root, *up, env=env)
        assert "HF_UPLOAD_DONE" in out and hub_state(root).get("large_folder_calls") == 1 and (root / "runs/hf_stage/m").exists(), out
        shutil.rmtree(root / "runs/hf_stage")
        hub_state(root, private=False)
        out = run(root, *up, rc=1, env=env)
        assert "不是私有仓库" in out and not (root / "runs/hf_stage").exists(), out
        # 第 4 版：公开仓库要明确写 --visibility public；写了就能传，私有仓库配 --visibility public 反过来停
        out = run(root, *up, "--visibility", "public", env=env)
        assert "HF_UPLOAD_DONE" in out and "公开仓库" in out and hub_state(root)["large_folder_calls"] == 2, out
        shutil.rmtree(root / "runs/hf_stage")
        assert "HF_README_OK" in run(root, "readme", "--repo", "u/r", "--file", "README.md", "--visibility", "public", env=env)
        assert "不是私有仓库" in run(root, "readme", "--repo", "u/r", "--file", "README.md", rc=1, env=env)
        hub_state(root, private=True)
        out = run(root, *up, "--visibility", "public", rc=1, env=env)
        assert "不是公开仓库" in out and not (root / "runs/hf_stage").exists() and hub_state(root)["large_folder_calls"] == 2, out
        assert "不是公开仓库" in run(root, "readme", "--repo", "u/r", "--file", "README.md", "--visibility", "public", rc=1, env=env)


def test_whoami_visibility() -> None:
    """whoami 的可见性与 --visibility 对上才报 HF_WHOAMI_OK（第 3 批第 4 版）。"""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        env = fake_hub(root)
        hub_state(root, exists=True, private=True)
        assert "HF_WHOAMI_OK" in run(root, "whoami", "--repo", "u/r", env=env)
        assert "HF_WHOAMI_FAIL" in run(root, "whoami", "--repo", "u/r", "--visibility", "public", rc=1, env=env)
        hub_state(root, private=False)
        assert "HF_WHOAMI_FAIL" in run(root, "whoami", "--repo", "u/r", rc=1, env=env)
        assert "HF_WHOAMI_OK" in run(root, "whoami", "--repo", "u/r", "--visibility", "public", env=env)


def test_visibility_command() -> None:
    """visibility 子命令（第 3 批第 4 版）：只碰本工具建的归档仓库，--confirm 要与 --repo 相同，不加 --yes 不改，
    已是目标可见性不重复改，改完重新读对不上报 HF_VISIBILITY_FAIL，旧版 huggingface_hub 退回 update_repo_visibility。"""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        env = fake_hub(root)
        hub_state(root, exists=True, private=True, files={"README.md": "# r\n"},
                  tree={"runs/flash_chord/t1-a/policy_1.safetensors": [4, "x"], "m/runs/flash_chord/t1-a/b/policy_1.safetensors": [4, "x"]})
        go = ("visibility", "--repo", "u/r", "--to", "public", "--confirm", "u/r", "--yes")
        out = run(root, *go, rc=1, env=env)
        assert "不是本工具建的归档仓库" in out and hub_state(root)["private"] and not hub_state(root).get("settings_calls"), out
        hub_state(root, files={}, tree={"m/runs/flash_chord/t1-a/policy_1.safetensors": [4, "x"]})
        assert "不是本工具建的归档仓库" in run(root, *go, rc=1, env=env) and hub_state(root)["private"]
        hub_state(root, files={"README.md": "# r\n"})
        out = run(root, "visibility", "--repo", "u/r", "--to", "public", "--confirm", "u/x", "--yes", rc=1, env=env)
        assert "不同" in out and hub_state(root)["private"] and not hub_state(root).get("settings_calls"), out
        out = run(root, *go[:-1], env=env)
        assert "HF_VISIBILITY_DRYRUN" in out and hub_state(root)["private"] and not hub_state(root).get("settings_calls"), out
        out = run(root, *go, env=env)
        assert "HF_VISIBILITY_OK" in out and not hub_state(root)["private"] and hub_state(root)["settings_calls"] == 1, out
        out = run(root, *go, env=env)
        assert "不改" in out and "HF_VISIBILITY_OK" in out and hub_state(root)["settings_calls"] == 1, out
        hub_state(root, private=True, ignore_update=True)
        assert "HF_VISIBILITY_FAIL" in run(root, *go, rc=1, env=env) and hub_state(root)["private"]
        hub_state(root, ignore_update=False, old_api=True)
        out = run(root, *go, env=env)
        st = hub_state(root)
        assert "HF_VISIBILITY_OK" in out and not st["private"] and st["visibility_calls"] == 1 and st["settings_calls"] == 2, (out, st)
        hub_state(root, old_api=False)
        out = run(root, "visibility", "--repo", "u/r", "--to", "private", "--confirm", "u/r", "--yes", env=env)
        assert "HF_VISIBILITY_OK" in out and hub_state(root)["private"], out


def test_spotcheck_guards() -> None:
    import hashlib
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        env = {**fake_hub(root), "HF_HOME": str(root / "hfhome")}
        files, rows = {}, []
        for run_, kind, name in (("a", "final", "state_9.safetensors"), ("a", "policy", "policy_1.safetensors"),
                                 ("a", "policy", "policy_2.safetensors"), ("b", "final", "state_7.safetensors")):
            text = f"{run_}/{name}"
            path = f"m/runs/flash_chord/{run_}/{name}"
            files[path] = text
            rows.append({"run": run_, "kind": kind, "file": name, "size": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "repo": "u/r", "path_in_repo": path, "match": True})
        (root / "v.json").write_text(json.dumps(rows))
        hub_state(root, files=files)
        base = ("spotcheck", "--verified", "v.json", "--per-run")
        out = run(root, *base, "--min-free-mb", "1", "--max-cache-growth-mb", "100", "--out", "s.json", env=env)
        spot = json.loads((root / "s.json").read_text())
        got = sorted(c["path_in_repo"].rsplit("/", 2)[1] + "/" + c["path_in_repo"].rsplit("/", 1)[1] for c in spot["checked"])
        assert spot["ok"] and "HF_SPOTCHECK_OK" in out and len(got) == 3, (out, got)
        assert "a/state_9.safetensors" in got and "b/state_7.safetensors" in got, got
        out = run(root, *base, "--min-free-mb", str(10 ** 9), "--out", "s2.json", rc=1, env=env)
        assert "停在抽查" in out and not (root / "s2.json").exists() and not (root / "runs/hf_spotcheck").exists(), out
        hub_state(root, cache_growth=3 * 1048576)
        out = run(root, *base, "--max-cache-growth-mb", "1", "--out", "s3.json", rc=1, env=env)
        assert "xet 缓存比开始时多了" in out and not (root / "s3.json").exists() and not (root / "runs/hf_spotcheck").exists(), out


def test_hubsubset() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        build_final(root)
        run(root, "plan", "--kind", "state,final", "--out", "sf.json")
        rows = json.loads((root / "sf.json").read_text())
        env = fake_hub(root)
        tree = {f"m/runs/flash_chord/{r['run']}/{r['file']}": [r["size"], r["sha256"]] for r in rows[:3]}
        hub_state(root, tree=tree)
        out = run(root, "hubsubset", "--manifest", "sf.json", "--repo", "u/r", "--prefix", "m", "--out", "sub.json",
                  "--rest-out", "rest.json", env=env)
        sub = json.loads((root / "sub.json").read_text())
        assert sub == rows[:3] and "HF_SUBSET_OK" in out and f"仓库里没有的 {len(rows) - 3} 个" in out, out
        assert json.loads((root / "rest.json").read_text()) == rows[3:], out
        bad = dict(tree)
        k = next(iter(bad))
        bad[k] = [bad[k][0], "0" * 64]
        hub_state(root, tree=bad)
        out = run(root, "hubsubset", "--manifest", "sf.json", "--repo", "u/r", "--prefix", "m", "--out", "sub2.json", rc=1, env=env)
        assert "HF_SUBSET_MISMATCH 1" in out and len(json.loads((root / "sub2.json").read_text())) == 2, out


STUB = """import os, shutil, subprocess, sys
REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_archive_real.py")
cmd = sys.argv[1]
if cmd == "upload":
    open("runs/upload_called", "w").write(" ".join(sys.argv[1:]))
    sys.exit(0)
if cmd in ("verify", "spotcheck"):
    shutil.copy(os.environ["STUB_VERIFIED" if cmd == "verify" else "STUB_SPOT"], sys.argv[sys.argv.index("--out") + 1])
    sys.exit(0)
r = subprocess.run([sys.executable, REAL] + sys.argv[1:])
if cmd == "delete" and os.environ.get("STUB_EXTRA_DELETE"):
    os.unlink(os.environ["STUB_EXTRA_DELETE"])
sys.exit(r.returncode)
"""
RUNNER_ARGS = ["bash", str(RUNNER), "runs/pm.json", "u/r", "m", "runs/x"]


def runner_setup(root: Path) -> dict:
    build_policy(root)
    tools = root / "runs/tools"
    tools.mkdir(parents=True)
    shutil.copy(TOOL, tools / "hf_archive_real.py")
    (tools / "hf_archive.py").write_text(STUB)
    run(root, "plan", "--kind", "policy", "--out", "runs/pm.json")
    v = verified_from_plan(root, "runs/pm.json", "runs/stub_verified.json")
    spot_ok(root, v, "runs/stub_spot.json")
    b = root / "bin"
    b.mkdir()
    (b / "nvidia-smi").write_text('#!/bin/sh\nprintf "%s" "${STUB_GPU:-}"\n')
    (b / "nvidia-smi").chmod(0o755)
    (b / "pgrep").write_text('#!/bin/sh\n[ -n "${STUB_BUSY:-}" ] || exit 1\nprintf "%s\\n" "$STUB_BUSY"\n')
    (b / "pgrep").chmod(0o755)
    return {**os.environ, "PATH": f"{b}:{os.environ['PATH']}", "STUB_VERIFIED": str(root / "runs/stub_verified.json"),
            "STUB_SPOT": str(root / "runs/stub_spot.json")}


def run_runner(root: Path, env: dict, steps: str, **extra) -> subprocess.CompletedProcess:
    return subprocess.run(RUNNER_ARGS, cwd=root, env={**env, "V2D_HF_ARCHIVE_STEPS": steps, **extra},
                          capture_output=True, text=True, timeout=120)


def n_policies(root: Path) -> int:
    return len(list((root / "runs/flash_chord").glob("*/policy_*.safetensors")))


def linux_only(name: str) -> bool:
    if sys.platform.startswith("linux"):
        return True
    print(f"跳过 {name}：要 GNU find（-printf），只在训练机上跑")
    return False


def test_runner_verify_step() -> None:
    if not linux_only("test_runner_verify_step"):
        return
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        env = runner_setup(root)
        before = n_policies(root)
        r = run_runner(root, env, "verify", V2D_HF_SPOTCHECK_PER_RUN="1")
        assert r.returncode == 0 and r.stdout.rstrip().endswith("HF_ARCHIVE_VERIFIED"), r.stdout + r.stderr
        assert "抽查 每条 run 1 个" in r.stdout and n_policies(root) == before, r.stdout
        assert (root / "runs/x.verified.json").exists() and (root / "runs/x.spotcheck.json").exists()
        assert run_runner(root, env, "nope").returncode == 2
        assert run_runner(root, env, "verify", V2D_HF_SPOTCHECK_PER_RUN="2").returncode == 2
        assert (root / "runs/upload_called").read_text().endswith("--visibility private")
        (root / "runs/upload_called").unlink()
        assert run_runner(root, env, "verify", V2D_HF_VISIBILITY="nope").returncode == 2
        r = run_runner(root, env, "verify", V2D_HF_VISIBILITY="public")
        assert r.returncode == 0 and "可见性 public" in r.stdout, r.stdout + r.stderr
        assert (root / "runs/upload_called").read_text().endswith("--visibility public")
        (root / "runs/upload_called").unlink()
        r = run_runner(root, env, "check", V2D_HF_SPOTCHECK_PER_RUN="1")
        assert r.returncode == 0 and r.stdout.rstrip().endswith("HF_ARCHIVE_VERIFIED") and "没有上传" in r.stdout, r.stdout + r.stderr
        assert not (root / "runs/upload_called").exists() and n_policies(root) == before, r.stdout


def test_runner_delete_step() -> None:
    if not linux_only("test_runner_delete_step"):
        return
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        env = runner_setup(root)
        before = n_policies(root)
        (root / "runs/hf_spotcheck/m").mkdir(parents=True)  # 假装上一次抽查被打断，留下了下载到一半的文件
        (root / "runs/hf_spotcheck/m/half.safetensors").write_text("x")
        r = run_runner(root, env, "delete")
        assert r.returncode == 1 and "HF_ARCHIVE_STOPPED delete（缺核对或抽查结果）" in r.stdout, r.stdout + r.stderr
        assert not (root / "runs/hf_spotcheck").exists(), "流程脚本退出时没删 runs/hf_spotcheck"
        shutil.copy(root / "runs/stub_verified.json", root / "runs/x.verified.json")
        shutil.copy(root / "runs/stub_spot.json", root / "runs/x.spotcheck.json")
        (root / "runs/.chain-active-x").write_text("")
        r = run_runner(root, env, "delete")
        assert r.returncode == 1 and "HF_ARCHIVE_STOPPED delete（有训练链在跑）" in r.stdout, r.stdout + r.stderr
        (root / "runs/.chain-active-x").unlink()
        r = run_runner(root, env, "delete", STUB_GPU="4242")
        assert r.returncode == 1 and "GPU 上有进程：4242" in r.stdout and n_policies(root) == before, r.stdout + r.stderr
        r = run_runner(root, env, "delete", STUB_BUSY="777 bash stages/policy/fc_run_watched.sh t1-x")
        assert r.returncode == 1 and "有导出、评测或训练相关的进程：777" in r.stdout and n_policies(root) == before, r.stdout + r.stderr
        r = run_runner(root, env, "delete")
        assert r.returncode == 0 and r.stdout.rstrip().endswith("HF_ARCHIVE_DONE"), r.stdout + r.stderr
        assert "清单外消失的文件：无" in r.stdout and "其中没删的 0 个" in r.stdout, r.stdout
        assert (root / "runs/x.reverified.json").exists() and n_policies(root) == before - len(POLICY_PICK), r.stdout


def test_runner_files_check_catches_extra_deletion() -> None:
    if not linux_only("test_runner_files_check_catches_extra_deletion"):
        return
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        env = runner_setup(root)
        shutil.copy(root / "runs/stub_verified.json", root / "runs/x.verified.json")
        shutil.copy(root / "runs/stub_spot.json", root / "runs/x.spotcheck.json")
        extra = root / "runs/flash_chord/t1-s/policy_1.safetensors"
        r = run_runner(root, env, "delete", STUB_EXTRA_DELETE=str(extra))
        assert r.returncode == 1 and r.stdout.rstrip().endswith("HF_ARCHIVE_FILES_CHECK_FAILED"), r.stdout + r.stderr
        assert "t1-s/policy_1.safetensors" in r.stdout, r.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
