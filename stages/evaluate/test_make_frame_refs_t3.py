"""make_frame_refs_t3.py 的测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-r3o-2026-10-15.md 2.4.1）。要 numpy、pyarrow；装了 pandas 时
再用 kit 打包器自己的函数核四元数约定。

临时目录里搭最小仓库：真的 make_frame_refs_t3.py 与 libs/v2d_infra/tier_guard.py；假的 run_record.jsonl（host 是本机）、v2d_meta.json、Tier 3 root
的 parquet（runs/tier3/root_hm/data/chunk-000/episode_XXXXXX.parquet）与选择清单。poses.npy 用 ast 从 stages/retarget/challenge_to_soma.py 原样
抽出的 pose7_to_mat 生成（流水线写 poses.npy 用的就是它；那个模块导入 soma_fk，测试里导入不了），不可见帧照 challenge_to_soma 按前一帧补。
  1. 正常：只做本机的行，打出别的机器有几集；npz 的键、类型、形状与 kit 的 tracks/track_2/README.md 一致；位置就是 poses.npy 的平移；四元数与原来的
     wxyz 换成 xyzw 后只差符号（w ≥ 0），其中有转角 179° 绕 x、y、z 的帧（迹为负，走另外三个分支）与 w 为负的帧；不可见帧跳过、打出帧数；
     manifest 的列与 sha256 对，来源 parquet 的 sha256 是现算的；
     有 pandas 时（按 Fable 5.1 的审查重写）：评测器参考 = 在矩阵上对 poses.npy 做一个刚体变换，再用生成器的 mat_to_xyzw 转四元数；kit 的
     convert_track2_frame 要把它恰好换回我们的帧参考（位置差 < 1e-9，四元数 1 − |点积| < 1e-12）；反向对照：评测器参考的四元数按共轭（转置）
     约定算时，kit 报不是一个刚体变换；
  2. 逐条 FAIL（退出码 1，同机别的行照做）：v2d_meta 的 episode、档位、来源不对；run 记录指向的不是 Tier 3；run 记录的主机不是本机；旋转不正交；
     有 NaN；最后一行不是 [0,0,0,1]；来源 parquet 缺失、sha256 与 v2d_meta 不符、frame_index 跳帧、episode_index 不对、行数与 poses.npy 不符、
     可见帧位姿与 poses.npy 不符；失败的这一集不留 npz；
  3. 拒绝（退出码非 0）：输出目录已存在；本机的行档位不是 3（不建输出目录）；本机没有行；
  4. 档位隔离：序列名里带 tier_1_multiview_caption 时，读 v2d_meta.json 被 tier_guard 拦下（Tier1AccessError，不是读进来）。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow --with pandas python stages/evaluate/test_make_frame_refs_t3.py
     （没有 pandas 时只跑不依赖 kit 的部分）
"""
import ast
import hashlib
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit"
OUT = "runs/submission/frame_refs_t3_x"
INVISIBLE = {5: 10}   # ep5 第 10 帧不可见：来源 parquet 那一帧全零，poses.npy 按前一帧补


def _pose7_to_mat():
    src = (REPO / "stages/retarget/challenge_to_soma.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "pose7_to_mat")
    g = {"np": np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "challenge_to_soma.py", "exec"), g)
    return g["pose7_to_mat"]


POSE7_TO_MAT = _pose7_to_mat()


def wxyz(T: int, rng) -> np.ndarray:
    q = rng.normal(size=(T, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    for i, axis in enumerate(np.eye(3)):   # 转角 179°：迹为负
        a = np.radians(179.0)
        q[i] = np.concatenate([[np.cos(a / 2)], np.sin(a / 2) * axis])
    q[3] = -np.abs(q[3])   # w 为负
    q[4] = [1.0, 0.0, 0.0, 0.0]
    return q


def write_parquet(path: Path, fi, epi, p7, vis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"frame_index": pa.array(np.asarray(fi), pa.int64()), "episode_index": pa.array(np.asarray(epi), pa.int64()),
                             "observation.object.pose": pa.array(list(p7), pa.list_(pa.float64(), 7)),
                             "observation.object.visible": pa.array(np.asarray(vis, bool))}), path)


def setup(root: Path, rows=(("mf4090", 2), ("mf4090", 5), ("ml40", 7)), meta_kw=None, bad=None, seq_fmt="seq{ep}", src_kw=None) -> dict:
    """返回 {ep: (填补后的 p7，poses.npy 的矩阵)}。src_kw[ep] 可改来源 parquet：frame_index、episode_index、extra_row。"""
    for f in ("stages/evaluate/make_frame_refs_t3.py", "libs/v2d_infra/tier_guard.py"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / f, root / f)
    rng = np.random.default_rng(0)
    truth, lines = {}, ["# 注释\n"]
    for m, ep in rows:
        T, seq, run = 30 + ep, seq_fmt.format(ep=ep), f"t3-ep{ep}-w2048-cap4m-r3a-hm-{m}"
        p7 = np.concatenate([rng.normal(scale=0.5, size=(T, 3)), wxyz(T, rng)], axis=1)
        vis = np.ones(T, bool)
        raw = p7.copy()
        if ep in INVISIBLE:
            k = INVISIBLE[ep]
            vis[k] = False
            raw[k] = 0.0
            p7[k] = p7[k - 1]   # 同 challenge_to_soma 的 fill_invisible：按前一帧补
        M = POSE7_TO_MAT(p7)
        if bad and ep in bad:
            M = bad[ep](M.copy())
        kw = (src_kw or {}).get(ep, {})
        fi = kw.get("frame_index", np.arange(T))
        epi = kw.get("episode_index", np.full(T, ep))
        if kw.get("extra_row"):
            fi, epi, raw, vis = np.arange(T + 1), np.full(T + 1, ep), np.vstack([raw, raw[-1:]]), np.append(vis, True)
        sp = root / f"runs/tier3/root_hm/data/chunk-000/episode_{ep:06d}.parquet"
        write_parquet(sp, fi, epi, raw, vis)
        src = root / f"runs/motion_src/tier3_hm/{seq}"
        src.mkdir(parents=True, exist_ok=True)
        np.save(src / "poses.npy", M)
        meta = {"tier": 3, "episode": ep, "sequence_id": seq,
                "object_pose_source": {"kind": "tier3_reconstruction", "column": "observation.object.pose",
                                       "provenance": {"parquet_sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}}}
        meta.update((meta_kw or {}).get(ep, {}))
        (src / "v2d_meta.json").write_text(json.dumps(meta))
        d = root / f"runs/flash_chord/{run}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_record.jsonl").write_text(json.dumps({"event": "start", "host": socket.gethostname(),
                                                        "parquet": f"/repo/runs/motion/tier3_hm/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"}) + "\n")
        lines.append(f"3\t{ep}\t{run}\t4096000\t{m}\n")
        truth[ep] = (p7, M)
    (root / "sel.tsv").write_text("".join(lines))
    return truth


def run(root: Path, machine: str = "mf4090") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "stages/evaluate/make_frame_refs_t3.py", "sel.tsv", machine, "--out", OUT],
                          cwd=root, capture_output=True, text=True, timeout=120)


def test_normal() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        truth = setup(root)
        r = run(root)
        assert r.returncode == 0 and r.stdout.rstrip().endswith("FRAME_REFS_T3_OK 2") and "清单另有 1 集在别的机器" in r.stdout, r.stdout + r.stderr
        assert "不可见 1 帧按前一帧补" in r.stdout and "不可见 0 帧按前一帧补" in r.stdout and "RuntimeWarning" not in r.stderr, r.stdout + r.stderr
        out = root / OUT
        assert sorted(p.name for p in out.iterdir()) == ["episode_000002.npz", "episode_000005.npz", "manifest_mf4090.tsv"]
        man = (out / "manifest_mf4090.tsv").read_text().splitlines()
        assert man[0].split("\t") == ["episode", "run", "step", "tier_dir", "sequence", "frames", "poses_sha256", "source_parquet_sha256",
                                      "file", "sha256"], man[0]
        assert len(man) == 3, man
        for line in man[1:]:
            ep, run_, step, tdir, seq, n, psha, ssha, f, sha = line.split("\t")
            ep = int(ep)
            assert (run_, step, tdir, seq) == (f"t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090", "4096000", "tier3_hm", f"seq{ep}"), line
            assert ssha == hashlib.sha256((root / f"runs/tier3/root_hm/data/chunk-000/episode_{ep:06d}.parquet").read_bytes()).hexdigest()
            assert hashlib.sha256((out / f).read_bytes()).hexdigest() == sha
            assert hashlib.sha256((root / f"runs/motion_src/{tdir}/{seq}/poses.npy").read_bytes()).hexdigest() == psha
            with np.load(out / f, allow_pickle=False) as z:
                assert sorted(z.files) == ["coordinate_frame", "episode_index", "frame_index", "pose_xyzw"], z.files
                assert str(z["coordinate_frame"].item()) == "released_track2_world" and int(z["episode_index"].item()) == ep
                assert z["frame_index"].dtype == np.int64 and np.array_equal(z["frame_index"], np.arange(int(n)))
                p, (v, _) = z["pose_xyzw"], truth[ep]
                assert p.shape == (len(v), 1, 7) and p.dtype == np.float64 and int(n) == len(v)
                assert np.array_equal(p[:, 0, :3], v[:, :3])
                want = np.concatenate([v[:, 4:7], v[:, 3:4]], axis=1)
                want[want[:, 3] < 0] *= -1.0
                assert np.abs(p[:, 0, 3:] - want).max() < 1e-12 and (p[:, 0, 6] >= 0).all(), np.abs(p[:, 0, 3:] - want).max()
        kit_check(out, {ep: M for ep, (_, M) in truth.items() if ep in (2, 5)})


def _generator():
    spec = importlib.util.spec_from_file_location("make_frame_refs_t3", REPO / "stages/evaluate/make_frame_refs_t3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kit_check(out: Path, mats: dict) -> None:
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("  （没有 pandas：跳过用 kit 函数核四元数约定）")
        return
    sys.path.insert(0, str(KIT / "tools"))
    sys.path.insert(0, str(KIT))
    import pack_submission as ps
    gen = _generator()
    a, b = np.radians(30.0), np.radians(20.0)
    Rg = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]) @ \
        np.array([[1.0, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    tg = np.array([1.0, 2.0, 0.5])
    for ep, M in mats.items():
        n = len(M)
        ref = ps.input_frame_reference(out, ep, np.arange(n), 1, ["object"])
        assert ref.shape == (n, 7)
        Re = Rg @ M[:, :3, :3]   # 评测器参考：在矩阵上做刚体变换，再用生成器的函数转四元数
        ev = np.concatenate([M[:, :3, 3] @ Rg.T + tg, gen.mat_to_xyzw(Re)], axis=1)
        conv, dp, da, changed = ps.convert_track2_frame(ev, ev, ref)
        assert dp < 1e-9 and da < 1e-6 and changed, (ep, dp, da)
        assert np.abs(conv[:, :3] - ref[:, :3]).max() < 1e-9 and (1 - np.abs(np.sum(conv[:, 3:] * ref[:, 3:], axis=1))).max() < 1e-12, ep
        bad = ev.copy()   # 反向对照：四元数按共轭（转置）约定算，kit 应当拦下
        bad[:, 3:] = gen.mat_to_xyzw(np.transpose(Re, (0, 2, 1)))
        try:
            ps.convert_track2_frame(bad, bad, ref)
        except ValueError as err:
            assert "rigid transform" in str(err), err
        else:
            raise AssertionError(f"ep{ep}：共轭约定的四元数没被 kit 的刚体换算拦下")
    print("  kit 的 input_frame_reference、convert_track2_frame 核对通过：矩阵上的刚体变换恰好换回，共轭约定被拦下")


def _scale(M):
    M[:, :3, :3] *= 1.01
    return M


def _nan(M):
    M[3, 0, 1] = np.nan
    return M


def _lastrow(M):
    M[:, 3, 0] = 1e-3
    return M


def _shift(M):
    M[7, 0, 3] += 1e-6   # 可见帧的平移差 1 µm
    return M


def test_row_failures() -> None:
    skip = np.concatenate([np.arange(7), np.arange(8, 33)])   # ep2 有 32 帧：少了第 7 帧、多了第 32 帧
    cases = {
        "episode": ({"meta_kw": {2: {"episode": 3}}}, "v2d_meta.json 不符"),
        "tier": ({"meta_kw": {2: {"tier": 2}}}, "v2d_meta.json 不符"),
        "kind": ({"meta_kw": {2: {"object_pose_source": {"kind": "parquet"}}}}, "v2d_meta.json 不符"),
        "ortho": ({"bad": {2: _scale}}, "不是刚体变换"),
        "nan": ({"bad": {2: _nan}}, "非有限值"),
        "lastrow": ({"bad": {2: _lastrow}}, "不是刚体变换"),
        "pose": ({"bad": {2: _shift}}, "物体位姿不同"),
        "srcsha": ({"meta_kw": {2: {"object_pose_source": {"kind": "tier3_reconstruction", "provenance": {"parquet_sha256": "0" * 64}}}}}, "sha256"),
        "frameidx": ({"src_kw": {2: {"frame_index": skip}}}, "frame_index 不是 0..T−1"),
        "epidx": ({"src_kw": {2: {"episode_index": np.full(32, 3)}}}, "episode_index 不全等于"),
        "rows": ({"src_kw": {2: {"extra_row": True}}}, "行，poses.npy"),
        "record": ({}, "不是 Tier 3 的 motion parquet"),
        "host": ({}, "主机"),
        "nosrc": ({}, "FileNotFoundError"),
    }
    for name, (kw, word) in cases.items():
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            setup(root, **kw)
            rec = root / "runs/flash_chord/t3-ep2-w2048-cap4m-r3a-hm-mf4090/run_record.jsonl"
            if name == "record":
                rec.write_text(json.dumps({"event": "start", "host": socket.gethostname(),
                                           "parquet": "/repo/runs/motion/tier2/whole_body/soma/sequence_id=seq2/robot_name=g1/data.parquet"}) + "\n")
            if name == "host":
                rec.write_text(rec.read_text().replace(json.dumps(socket.gethostname()), json.dumps("other-host")))
            if name == "nosrc":
                (root / "runs/tier3/root_hm/data/chunk-000/episode_000002.parquet").unlink()
            r = run(root)
            assert r.returncode == 1 and r.stdout.rstrip().endswith("FRAME_REFS_T3_FAIL 1"), (name, r.stdout + r.stderr)
            assert word in r.stdout and "FAIL ep2" in r.stdout and "ok   ep5" in r.stdout, (name, r.stdout)
            assert not (root / OUT / "episode_000002.npz").exists() and (root / OUT / "episode_000005.npz").exists(), name


def test_refusals() -> None:
    with tempfile.TemporaryDirectory() as t:   # 输出目录已存在
        root = Path(t)
        setup(root)
        (root / OUT).mkdir(parents=True)
        r = run(root)
        assert r.returncode != 0 and "拒绝覆盖" in r.stderr, r.stderr
    with tempfile.TemporaryDirectory() as t:   # 本机的行档位不是 3
        root = Path(t)
        setup(root)
        s = root / "sel.tsv"
        s.write_text(s.read_text().replace("3\t2\t", "2\t2\t"))
        r = run(root)
        assert r.returncode != 0 and "档位不是 3" in r.stderr and not (root / OUT).exists(), r.stderr
    with tempfile.TemporaryDirectory() as t:   # 本机没有行
        root = Path(t)
        setup(root, rows=(("ml40", 7),))
        r = run(root)
        assert r.returncode != 0 and "没有 mf4090 的行" in r.stderr and not (root / OUT).exists(), r.stderr


def add_kfix(root: Path, eps: tuple, scale: float = 1.067) -> dict:
    """在 setup 的基础上加一套 tier3_kfix：位置是 hm 的 scale 倍（模仿 k-fix 的整体放大），朝向不变；返回 {ep: 矩阵}。"""
    out = {}
    for ep in eps:
        seq = f"seq{ep}"
        Mh = np.load(root / f"runs/motion_src/tier3_hm/{seq}/poses.npy")
        Mk = Mh.copy()
        Mk[:, :3, 3] *= scale
        p7 = np.concatenate([Mk[:, :3, 3], np.zeros((len(Mk), 4))], axis=1)
        q = _generator().mat_to_xyzw(Mk[:, :3, :3])   # xyzw
        p7[:, 3:] = q[:, [3, 0, 1, 2]]                 # parquet 里是 wxyz
        sp = root / f"runs/tier3/root_kfix/data/chunk-000/episode_{ep:06d}.parquet"
        write_parquet(sp, np.arange(len(Mk)), np.full(len(Mk), ep), p7, np.ones(len(Mk), bool))
        src = root / f"runs/motion_src/tier3_kfix/{seq}"
        src.mkdir(parents=True, exist_ok=True)
        np.save(src / "poses.npy", Mk)
        (src / "v2d_meta.json").write_text(json.dumps({"tier": 3, "episode": ep, "sequence_id": seq, "object_pose_source": {
            "kind": "tier3_reconstruction", "column": "observation.object.pose",
            "provenance": {"parquet_sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}}}))
        out[ep] = Mk
    return out


def export_dir(root: Path, eps: dict) -> Path:
    """导出结果目录：{ep: (run, 步数, 档位目录)} → episode_XXXXXX.json（只写 checkpoint 与 parquet 两个键）。"""
    d = root / "export"
    d.mkdir(exist_ok=True)
    for ep, (run, step, tdir) in eps.items():
        (d / f"episode_{ep:06d}.json").write_text(json.dumps({
            "checkpoint": f"/repo/runs/flash_chord/{run}/policy_{step}.safetensors",
            "parquet": f"/repo/runs/motion/{tdir}/whole_body/soma/sequence_id=seq{ep}/robot_name=g1/data.parquet"}))
    return d


def test_from_export() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        mk = add_kfix(root, (2, 5))
        ex = export_dir(root, {2: ("t3-ep2-w2048-cap4m-r3a-hm-mf4090", "4096000", "tier3_kfix"),
                               5: ("t3-ep5-w2048-cap4m-r3a-hm-mf4090", "4096000", "tier3_kfix")})
        r = subprocess.run([sys.executable, "stages/evaluate/make_frame_refs_t3.py", "sel.tsv", "mf4090", "--out", OUT, "--from-export", str(ex)],
                           cwd=root, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "FRAME_REFS_T3_OK 2" in r.stdout, r.stdout + r.stderr
        rows = [ln.split("\t") for ln in (root / OUT / "manifest_mf4090.tsv").read_text().splitlines()[1:]]
        assert all(rw[3] == "tier3_kfix" for rw in rows), rows
        for ep, M in mk.items():
            with np.load(root / OUT / f"episode_{ep:06d}.npz") as z:
                assert np.abs(z["pose_xyzw"][:, 0, :3] - M[:, :3, 3]).max() < 1e-9, ep
        # 反向对照：导出 JSON 的 checkpoint 步数与清单不同，这一集判失败、不留 npz
        ex2 = root / "export"
        (ex2 / "episode_000005.json").write_text(json.dumps({
            "checkpoint": "/repo/runs/flash_chord/t3-ep5-w2048-cap4m-r3a-hm-mf4090/policy_9999.safetensors",
            "parquet": "/repo/runs/motion/tier3_kfix/whole_body/soma/sequence_id=seq5/robot_name=g1/data.parquet"}))
        r = subprocess.run([sys.executable, "stages/evaluate/make_frame_refs_t3.py", "sel.tsv", "mf4090", "--out", OUT + "_b", "--from-export", str(ex2)],
                           cwd=root, capture_output=True, text=True, timeout=120)
        assert r.returncode == 1 and "FAIL ep5" in r.stdout and "checkpoint" in r.stdout, r.stdout + r.stderr
        assert not (root / (OUT + "_b") / "episode_000005.npz").exists()
    print("  5. --from-export：参考取导出 JSON 的 parquet（tier3_kfix），manifest 记新档位目录、位置是新参考的；checkpoint 步数不符的一集判失败、不留 npz")


def test_tier1_blocked() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, rows=(("mf4090", 2),), seq_fmt="x_tier_1_multiview_caption_{ep}")
        r = run(root)
        assert r.returncode == 1 and "Tier1AccessError" in r.stdout, r.stdout + r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL_OK")
