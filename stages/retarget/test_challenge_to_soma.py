"""challenge_to_soma 的测试（宿主机跑，不需要容器）。

守：转换器契约——soma_params.npz 的键与形状满足官方 _load_npz；joint_names 与 SOMA rig
逐字节一致；poses.npy 是未变换的 OpenCV 系齐次矩阵；flex 逐帧写入；不可见帧被填补；
Tier 2、3 的命令行入口装上档位隔离守卫。
契约表尚未落到 docs/contracts/（仓库处于 P0），暂无契约 ID，落表后补上。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "libs" / "v2d_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pyarrow.parquet as pq
import add_metrics as A
from challenge_to_soma import convert, fill_invisible, pose7_to_mat, tier3_provenance
from paths import tier
from soma_fk import col, joint_names

REQUIRED = {"poses", "transl", "joint_names", "identity_model_type", "identity_coeffs",
            "scale_params", "joint_orient", "unit", "keep_root"}   # read_soma._load_npz 的必填键


def test_fill_invisible():
    p = np.arange(5 * 7, dtype=float).reshape(5, 7)
    out, n = fill_invisible(p, np.array([False, True, False, True, False]))
    assert n == 3
    assert np.array_equal(out[0], p[1]) and np.array_equal(out[2], p[1]) and np.array_equal(out[4], p[3])
    print("  不可见帧：开头回填、中间和结尾沿用上一有效 ✓")


def test_tier3_provenance():
    """Tier 3：meta/provenance.json 里这一集的来源原样取出；缺文件或缺这一集就停（计划 agent-doc/plan-t3-pilot-2026-10-09.md，审查 m5）。"""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "meta").mkdir()
        for missing in (True, False):
            try:
                tier3_provenance(root, 8)
                raise AssertionError("应当拒绝")
            except SystemExit as e:
                assert ("缺" if missing else "没有 episode 8") in str(e), e
            (root / "meta" / "provenance.json").write_text(json.dumps({"episodes": {"24": {"deployed_commit": "x"}}}))
        (root / "meta" / "provenance.json").write_text(json.dumps({"episodes": {"8": {"deployed_commit": "c" * 40, "parquet_sha256": "a" * 64}}}))
        assert tier3_provenance(root, 8) == {"deployed_commit": "c" * 40, "parquet_sha256": "a" * 64}
    print("  Tier 3 来源：取得到、缺文件或缺集都拒绝 ✓")


def test_entry_guard():
    """命令行入口：--tier 2、3 先调 forbid_tier1()，之后读 Tier 1 的路径被拦；--tier 1 不拦
    （计划 agent-doc/plan-t3-delivery-p2-2026-10-12.md 审查 12）。守卫装上就撤不掉，所以在子进程里跑；
    子进程把 convert 换成探针，只看守卫，不真转换。"""
    probe = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(REPO / 'libs' / 'v2d_infra')!r}); sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})",
        "import challenge_to_soma as c",
        "from tier_guard import Tier1AccessError",
        "def probe(tier_n, episode, object_poses=None):",
        "    try:",
        "        open('/nonexistent/tier_1_multiview_caption/x')",
        "    except Tier1AccessError:",
        "        return {'guard': 'BLOCKED'}",
        "    except OSError:",
        "        return {'guard': 'OPEN'}",
        "c.convert = probe",
        "sys.exit(c.main(sys.argv[1:]))",
    ])
    for n, want in ((1, "OPEN"), (2, "BLOCKED"), (3, "BLOCKED")):
        r = subprocess.run([sys.executable, "-c", probe, "--tier", str(n), "--episode", "0"], capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(r.stdout)["guard"] == want, (n, r.stdout, r.stderr)
    print("  入口守卫：Tier 2、3 读 Tier 1 被拦，Tier 1 照常 ✓")


def test_pose_matrix_matches_independent_impl():
    t = pq.read_table(tier(1) / "data" / "chunk-000" / "episode_000020.parquet")
    p7 = col(t, "observation.object.pose")
    M = pose7_to_mat(p7)
    R_ref = A.quat_wxyz_to_matrix(p7[:, 3:7])
    e = max(np.abs(M[:, :3, :3] - R_ref).max(), np.abs(M[:, :3, 3] - p7[:, :3]).max())
    assert e < 1e-6, f"与 add_metrics 的实现差 {e}"
    assert np.allclose(M[:, 3], [0, 0, 0, 1])
    print(f"  位姿矩阵与独立实现最大差 {e:.1e} ✓")


def test_convert_episode_20():
    with tempfile.TemporaryDirectory() as td:
        info = convert(1, 20, runs_root=Path(td))
        src = Path(info["data_folder"])
        t = pq.read_table(tier(1) / "data" / "chunk-000" / "episode_000020.parquet")
        T = t.num_rows
        z = np.load(src / "soma_params.npz")
        assert REQUIRED <= set(z.files), f"缺必填键 {REQUIRED - set(z.files)}"
        assert z["poses"].shape == (T, 77, 3) and z["transl"].shape == (T, 3)
        assert [str(n) for n in z["joint_names"]] == joint_names(), "joint_names 与 rig 不一致"
        assert str(z["identity_model_type"]) == "mhr" and str(z["unit"]) == "meters"
        assert np.array_equal(z["bone_length_flexibles"],
                              col(t, "observation.human.bone_length_flexibles").astype(np.float32))
        assert z["bone_length_flexibles"].shape == (T, 6)
        P = np.load(src / "poses.npy")
        assert P.shape == (T, 4, 4) and P.dtype == np.float64
        for f in ("reconstructed_mesh/output_aligned.glb", "ground_plane.json"):
            assert (src / f).stat().st_size > 0, f"{f} 为空"
        assert Path(info["support"]).stat().st_size > 0
        m = json.loads((src / "v2d_meta.json").read_text())
        assert m["source_fps"] == 30.0 and m["episode"] == 20 and m["tier"] == 1, m
        assert Path(info["support"]).parent == Path(info["motion_root"]) / "whole_body" / "reconstructed_stage"
        print(f"  ep20 {info['sequence_id']}: {T} 帧，键/形状/关节名/flex/网格/support 全部正确 ✓")
        print(f"  不可见帧填补 {info['invisible_filled']} 帧；support 坐标系声明 {info['support_surface_frame']}")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{len(tests)} 个测试全部通过")
