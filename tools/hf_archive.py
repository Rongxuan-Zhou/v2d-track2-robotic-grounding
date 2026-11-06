"""把训练机上已经用不上的 state 与 policy 文件归档到 Hugging Face 仓库，核对无误后再删本地（训练机上、仓库根执行）。
计划 agent-doc/plan-hf-archive-2026-10-01.md（state，第 2 版，按 Fable 5.1 审查修改）与
agent-doc/plan-hf-archive-policies-2026-10-08.md（policy 中间份）与 agent-doc/plan-hf-archive-batch3-2026-10-10.md（第 3 批，加最终 state；
第 4 版：用户 10/11 同意把归档仓库改为公开）。

可见性：whoami、readme、upload 都带 --visibility private|public（默认 private），仓库的实际可见性与它不同就停（whoami 报 HF_WHOAMI_FAIL）。
默认值保持第 1–3 批“只往私有仓库传”的行为；公开仓库要明确写 --visibility public。

八个子命令，各自独立、各自落盘，任何一步没过就停在那一步：
  whoami    --repo <用户/仓库> [--visibility V]   查 token 属于谁、权限、仓库的可见性、仓库里已有多少文件（只读）。
  visibility --repo <用户/仓库> --to public|private --confirm <用户/仓库> [--yes]
                                  改仓库的可见性。--confirm 必须与 --repo 相同；仓库根目录要有 README.md，且至少有一个
                                  <机器名>/runs/flash_chord/<run>/(state|policy)_<步数>.safetensors，即本工具建的归档仓库，别的仓库一律不碰。
                                  已经是目标可见性就不改；不加 --yes 只打印将做什么（HF_VISIBILITY_DRYRUN）；改完重新读一次，
                                  对上才报 HF_VISIBILITY_OK。
  storage                         列出 token 所属账号下每个模型、数据集、Space 仓库是否私有与已用存储，再给私有与公开的合计（只读）。
                                  第 3 批上传遇到“Private storage limit reached”后加的，看额度被谁占着。
  hubsubset --manifest <清单> --repo <用户/仓库> --prefix <机器名> --out <子清单.json> [--rest-out <其余.json>]
                                  从清单里挑出已经提交到仓库、大小与 sha256 都对得上的行，另写一份清单（只读，不改仓库）；
                                  --rest-out 把仓库里还没有的行另写一份，以后接着上传用它（子清单删过之后，原清单不能再用来上传）。
                                  仓库里有同路径但大小或 sha256 不同的文件就退出码 1。第 3 批上传到一半撞上私有存储额度，
                                  只对已提交的那部分走核对、抽查、删除（计划 agent-doc/plan-hf-archive-batch3-2026-10-10.md 第 3 版）。
  readme    --repo <用户/仓库> --file <README.md> [--message <提交说明>] [--visibility V]
                                  把 README 传成仓库根的 README.md（仓库不存在时只新建私有仓库，--visibility public 就停），再下载回来比 sha256。
                                  用户 10/8 的要求：每个上传过的 HF 仓库都要有 README，所以 upload 遇到没有 README.md 的仓库就停。
  plan      --out <清单.json> [--kind state|policy|final，可用逗号连几种] [--keep 9]
                                  按规则挑文件，记下每个文件的大小与 sha256（只读）。--kind 默认 state；几种连用时清单是各种的并集，
                                  同一个文件出现两次就报错。
  upload    --manifest <清单> --repo <用户/仓库> --prefix <机器名> [--visibility V]
                                  仓库必须已经存在、可见性与 --visibility 相同、根目录有 README.md（先跑 readme），把清单里的文件硬链接进暂存目录 runs/hf_stage/<机器名>/，
                                  再用 upload_large_folder 上传（可断点续传，2 个并发）。
  verify    --manifest <清单> --repo <用户/仓库> --prefix <机器名> --out <核对.json>
                                  列出仓库里的文件，逐个比大小与 LFS sha256；全部对上才退出码 0。这个 sha256 是上传时客户端算的，
                                  只能证明“仓库里有个标着这个哈希的对象”，所以删除前还要 spotcheck。
  spotcheck --verified <核对.json> (--n 3 | --per-run) --out <抽查.json> [--min-free-mb M] [--max-cache-growth-mb C]
                                  随机挑 n 个文件（--per-run：每条 run 随机挑 1 个；这条 run 有最终 state 时固定挑它，其余文件里再随机挑 1 个）
                                  从仓库下载回来，本地重算 sha256 与清单比，比完删掉下载的副本；全部一致才退出码 0。
                                  训练链同时在跑时（第 3 批）用两道闸，不写抽查结果就停：每次下载前可用空间不到 M MB 加这个文件的大小；
                                  每次下载后 xet 缓存（HF_XET_CACHE，或 HF_HOME/xet）比开始时多出 C MB 以上。0 表示不查。
  delete    --verified <核对.json> --spotcheck <抽查.json> [--yes]
                                  核对与抽查都通过才删；删之前逐个重新核对：run 目录还有 DONE、本地大小与 sha256 没变、
                                  不是正式 run 步数最大的那份完整 state，也不是受保护的 policy（规则同 plan，删除时按当时的文件重算）。
                                  每个 run 先写 ARCHIVED.json（归档了什么、在仓库哪里；追加，不覆盖以前的记录）再删；
                                  清单里有 state 时，冒烟目录的 state 空了就写 STATE_PRUNED（同训练链的约定），不依赖这一轮有没有删到东西，
                                  中途断了重跑能补上。清单里有 policy 或最终 state 时，有训练链在跑就一个都不删。
                                  最终 state（final）删之前重查：仍是这条 run 步数最大的 state，它与同步数的 policy 都完整。
                                  最后清掉暂存目录的硬链接，空间才真正释放。不加 --yes 只打印将删什么。

挑文件的规则（plan）：只看 runs/flash_chord 下有 DONE 的 run，跳过名字里有 t2- 的（Tier 2 第 1 批正在用）。
  - 冒烟（smoke- 或 smoke_ 开头）：全部 state_*.safetensors；
  - 其他：除“步数最大、state 与同步数 policy 都完整”的那份以外的 state_*.safetensors；找不到完整的一份，整个 run 不动。
  policy_*.safetensors、日志、评测结果一律不动。
挑 policy 中间份的规则（plan --kind policy --keep K）：有训练链在跑（runs/.chain-active-* 存在）就不做。只看有 DONE 的 run
（名字里有 t2- 的也算，Tier 2 第 1 批早已跑完）。每条 run 留下这些 policy，其余的挑出来：
  - 步数最大的 K 份（K 默认 9：最后一份，加上 stages/evaluate/export_chain.sh 报终止错误时最多往前退的 8 份）；
  - 还有 state 文件的步数（state 与同步数的 policy 成对才能续训，见 stages/policy/ckpt_latest.py）；
  - runs/exports/*.tsv 里导出过的步数（任何状态的行都算），以及导出输入清单 stages/evaluate/exports/*.tsv 里的步数；
  - STATE_PRUNED 里记的步数。
  步数最大的那份 policy 不完整，整个 run 不动。state、日志、评测结果、导出产物一律不动。
挑最终 state 的规则（plan --kind final，第 3 批）：有训练链在跑就不做。只看有 DONE 的正式 run（不分档位；冒烟没有最终 state 的说法，
不看）。每条 run 挑步数最大的那份 state，条件是它与同步数的 policy 都完整；不完整就整个 run 不动。同步数的 policy 留着（导出用 policy）。
  这份 state 只在续训时用到：训练链对有 DONE 的 run 一律跳过（stages/policy/fc_run.sh 第 44 行、day2_train_chain.sh），
  而且续训不恢复 replay（day2_train_chain.sh 第 132 行），链也不续训没 DONE 的 run。删掉之后对这条 run 再跑 fc_run.sh，
  没有 DONE 时会因为“有 policy 没有完整 checkpoint”拒绝从 0 开始（fc_run.sh 第 162–164 行），不会覆盖。
凭据只经环境变量传进来（HF_TOKEN、https_proxy），不落命令行、不写日志；HF_ENDPOINT 必须没设或等于 https://huggingface.co
（服务器 .env 把它指向下载镜像，带 token 的请求不能经第三方转发）。只用标准库加 huggingface_hub。
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import struct
import sys
import time
from pathlib import Path

RUNS = Path("runs/flash_chord")
STAGE = Path("runs/hf_stage")
EXPORTS = Path("runs/exports")
EXPORT_LISTS = Path("stages/evaluate/exports")
STATE = re.compile(r"state_(\d+)\.safetensors")
POLICY = re.compile(r"policy_(\d+)\.safetensors")
VIS = ("private", "public")
ARCHIVE_PATH = re.compile(r"[A-Za-z0-9_.-]+/runs/flash_chord/[^/]+/(state|policy)_\d+\.safetensors")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def complete(p: Path) -> bool:
    """safetensors 文件头能解析、大小等于头里声明的数据末尾（同 stages/policy/ckpt_latest.py）。"""
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return False
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or 8 + n > size:
                return False
            header = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        return size == 8 + n + end
    except (OSError, ValueError, KeyError, TypeError):
        return False


def states(d: Path) -> list:
    out = []
    for p in d.glob("state_*.safetensors"):
        m = STATE.fullmatch(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def policies(d: Path) -> list:
    out = []
    for p in d.glob("policy_*.safetensors"):
        m = POLICY.fullmatch(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def chain_active() -> bool:
    return any(Path("runs").glob(".chain-active-*"))


def exported_steps() -> dict:
    """导出用到过或排着要用的 (run, 步数)：runs/exports/*.tsv（导出链的结果表，有表头，不管那一行的状态）
    与 stages/evaluate/exports/*.tsv（导出输入清单，# 开头是注释，列是 档位 episode run 步数 [主机]）。"""
    out = {}
    for t in sorted(EXPORTS.glob("*.tsv")):
        with t.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for r in rows:
            step = (r.get("step") or "").strip()
            if step.isdigit():
                out.setdefault((r.get("run") or "").strip(), set()).add(int(step))
    for t in sorted(EXPORT_LISTS.glob("*.tsv")):
        for line in t.read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            if line.startswith("#") or len(cols) < 4 or not cols[3].strip().isdigit():
                continue
            out.setdefault(cols[2].strip(), set()).add(int(cols[3].strip()))
    return out


def protected_policies(d: Path, run: str, keep: int, exported: dict):
    """这条 run 留下的 policy 步数（规则见文件头）。步数最大的那份 policy 不完整、或 STATE_PRUNED 读不懂，返回 None：整个 run 不动。"""
    pol = policies(d)
    if not pol:
        return set()
    if not complete(pol[-1][1]):
        return None
    keep_steps = {s for s, _ in pol[-keep:]} | {s for s, _ in states(d)} | exported.get(run, set())
    sp = d / "STATE_PRUNED"
    if sp.exists():
        text = sp.read_text().strip()
        if not text.isdigit():
            return None
        keep_steps.add(int(text))
    return keep_steps


def is_smoke(run: str) -> bool:
    return run.startswith("smoke-") or run.startswith("smoke_")


def keep_state(d: Path, run: str):
    """正式 run 留下的那份：步数最大、且 state 与同步数的 policy 都完整；冒烟不留（None）。找不到完整的返回 "__none__"。"""
    if is_smoke(run):
        return None
    for step, p in reversed(states(d)):
        if complete(p) and complete(d / f"policy_{step}.safetensors"):
            return p.name
    return "__none__"


def candidates() -> list:
    rows = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        run = d.name
        if "t2-" in run or not (d / "DONE").exists():
            continue
        st = states(d)
        if not st:
            continue
        keep = keep_state(d, run)
        if keep == "__none__":
            print(f"跳过 {run}：找不到完整的最终 state，整个 run 不动")
            continue
        for step, p in st:
            if p.name == keep:
                continue
            rows.append({"run": run, "kind": "smoke" if is_smoke(run) else "formal", "file": p.name, "step": step,
                         "size": p.stat().st_size, "sha256": sha256(p)})
    return rows


def final_state(d: Path):
    """步数最大的 state，且它与同步数的 policy 都完整时返回文件名；没有 state 或不完整返回 None。"""
    st = states(d)
    if not st:
        return None
    step, p = st[-1]
    return p.name if complete(p) and complete(d / f"policy_{step}.safetensors") else None


def candidates_final() -> list:
    if chain_active():
        sys.exit("有训练链在跑（runs/.chain-active-*），最终 state 归档不做")
    rows = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        run = d.name
        if is_smoke(run) or not (d / "DONE").exists() or not states(d):
            continue
        name = final_state(d)
        if name is None:
            print(f"跳过 {run}：步数最大的 state 或同步数的 policy 不完整，整个 run 不动")
            continue
        p = d / name
        rows.append({"run": run, "kind": "final", "file": name, "step": int(STATE.fullmatch(name).group(1)),
                     "size": p.stat().st_size, "sha256": sha256(p)})
    return rows


def candidates_policy(keep: int) -> list:
    if chain_active():
        sys.exit("有训练链在跑（runs/.chain-active-*），policy 归档不做")
    exported = exported_steps()
    rows = []
    for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        run = d.name
        if not (d / "DONE").exists():
            continue
        pol = policies(d)
        if len(pol) <= keep:
            continue
        prot = protected_policies(d, run, keep, exported)
        if prot is None:
            print(f"跳过 {run}：步数最大的 policy 不完整或 STATE_PRUNED 读不懂，整个 run 不动")
            continue
        picked = [(s, p) for s, p in pol if s not in prot]
        extra = sorted(s for s in prot if s < pol[-keep][0] and any(s == q for q, _ in pol))
        print(f"  {run}：{len(pol)} 份，挑 {len(picked)} 份（步 {picked[0][0] if picked else '-'}–{picked[-1][0] if picked else '-'}），"
              f"留最新 {keep} 份（步 {pol[-keep][0]}–{pol[-1][0]}）" + (f"，另外保护 {extra}" if extra else ""))
        for step, p in picked:
            rows.append({"run": run, "kind": "policy", "file": p.name, "step": step, "keep": keep,
                         "size": p.stat().st_size, "sha256": sha256(p)})
    return rows


def kinds_arg(x: str) -> list:
    ks = [k.strip() for k in x.split(",") if k.strip()]
    if not ks or len(set(ks)) != len(ks) or set(ks) - {"state", "policy", "final"}:
        raise argparse.ArgumentTypeError("--kind 取 state、policy、final 中的一种或几种，用逗号连、不重复")
    return ks


def cmd_plan(a) -> int:
    rows = []
    for kind in a.kind:
        part = {"policy": lambda: candidates_policy(a.keep), "state": candidates, "final": candidates_final}[kind]()
        total, runs = sum(r["size"] for r in part), sorted({r["run"] for r in part})
        if kind == "policy":
            print(f"候选 {len(part)} 个文件，{len(runs)} 个 run，共 {total / 1e9:.2f} GB（policy 中间份，每条 run 留最新 {a.keep} 份与受保护的步数）")
        elif kind == "state":
            print(f"候选 {len(part)} 个文件，{len(runs)} 个 run，共 {total / 1e9:.2f} GB（冒烟 "
                  f"{sum(r['size'] for r in part if r['kind'] == 'smoke') / 1e9:.2f} GB，正式 run 的旧 state "
                  f"{sum(r['size'] for r in part if r['kind'] == 'formal') / 1e9:.2f} GB）")
            for r in part:
                print(f"  {r['kind']:<6} {r['run']}/{r['file']}  {r['size']}  {r['sha256'][:16]}…")
        else:
            print(f"候选 {len(part)} 个文件，{len(runs)} 个 run，共 {total / 1e9:.2f} GB（有 DONE 的正式 run 的最终 state，同步数的 policy 留着）")
            for r in part:
                print(f"  final  {r['run']}/{r['file']}  {r['size']}  {r['sha256'][:16]}…")
        rows += part
    seen = [(r["run"], r["file"]) for r in rows]
    if len(set(seen)) != len(seen):
        sys.exit("几种挑法挑到了同一个文件，不写清单")
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    if len(a.kind) > 1:
        print(f"合计 {len(rows)} 个文件，{len({r['run'] for r in rows})} 个 run，共 {sum(r['size'] for r in rows) / 1e9:.2f} GB（{'、'.join(a.kind)}）")
    print(f"清单写到 {a.out}")
    return 0


def repo_path(prefix: str, r: dict) -> str:
    return f"{prefix}/runs/flash_chord/{r['run']}/{r['file']}"


def need_env() -> None:
    for k in ("HF_TOKEN", "https_proxy"):
        if not os.environ.get(k):
            sys.exit(f"缺环境变量 {k}（经 stdin 注入，不要写在命令行上）")
    ep = os.environ.get("HF_ENDPOINT", "")
    if ep and ep.rstrip("/") != "https://huggingface.co":
        sys.exit("HF_ENDPOINT 不是 https://huggingface.co（服务器 .env 把它指向下载镜像），带 token 的请求不能经它转发，停下")


def vis_name(private: bool) -> str:
    return "私有" if private else "公开"


def check_visibility(api, repo: str, vis: str) -> None:
    """仓库的实际可见性与 --visibility 不同就停。"""
    priv = bool(api.repo_info(repo, repo_type="model").private)
    if priv != (vis == "private"):
        sys.exit(f"{repo} 不是{vis_name(vis == 'private')}仓库（现在是{vis_name(priv)}，--visibility {vis}），停下")


def cmd_whoami(a) -> int:
    need_env()
    from huggingface_hub import HfApi
    api = HfApi()
    me = api.whoami()
    role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
    print(f"账号 {me.get('name')}，token 权限 {role}")
    info = api.repo_info(a.repo, repo_type="model", files_metadata=True)
    files = list(info.siblings or [])
    print(f"仓库 {a.repo}：{vis_name(bool(info.private))}，{len(files)} 个文件，共 {sum(f.size or 0 for f in files) / 1e9:.2f} GB"
          f"（期望{vis_name(a.visibility == 'private')}）")
    ok = bool(info.private) == (a.visibility == "private") and role in ("write", "fineGrained")
    print("HF_WHOAMI_OK" if ok else "HF_WHOAMI_FAIL")
    return 0 if ok else 1


def cmd_visibility(a) -> int:
    need_env()
    if a.confirm != a.repo:
        sys.exit(f"--confirm {a.confirm} 与 --repo {a.repo} 不同，停下")
    from huggingface_hub import HfApi
    api = HfApi()
    files = list(api.list_repo_files(a.repo, repo_type="model"))
    n_arch = sum(1 for f in files if ARCHIVE_PATH.fullmatch(f))
    if "README.md" not in files or n_arch == 0:
        sys.exit(f"{a.repo} 根目录没有 README.md，或没有 <机器名>/runs/flash_chord/<run>/(state|policy)_<步数>.safetensors"
                 f"（{n_arch} 个）：不是本工具建的归档仓库，不碰")
    want = a.to == "private"
    now = bool(api.repo_info(a.repo, repo_type="model").private)
    print(f"仓库 {a.repo}：现在{vis_name(now)}，{len(files)} 个文件，其中归档文件 {n_arch} 个；目标{vis_name(want)}")
    if now == want:
        print(f"已经是{vis_name(want)}，不改；HF_VISIBILITY_OK")
        return 0
    if not a.yes:
        print(f"将把 {a.repo} 从{vis_name(now)}改为{vis_name(want)}（演习，没改；加 --yes 才改）；HF_VISIBILITY_DRYRUN")
        return 0
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(a.repo, private=want, repo_type="model")
    else:  # 没有 update_repo_settings 的旧版 huggingface_hub
        api.update_repo_visibility(a.repo, private=want, repo_type="model")
    got = bool(api.repo_info(a.repo, repo_type="model").private)
    ok = got == want
    print(f"{time.strftime('%F %T')} 已提交修改，重新读到{vis_name(got)}；" + ("HF_VISIBILITY_OK" if ok else "HF_VISIBILITY_FAIL"))
    return 0 if ok else 1


def cmd_storage(a) -> int:
    need_env()
    from huggingface_hub import HfApi
    api = HfApi()
    me = api.whoami()
    name = me.get("name")
    print(f"账号 {name}，isPro={me.get('isPro')}")
    total = {True: 0, False: 0}
    for kind, lister in (("model", api.list_models), ("dataset", api.list_datasets), ("space", api.list_spaces)):
        try:
            repos = list(lister(author=name))
        except Exception as e:  # 某一类查不了时照样报其余几类
            print(f"  {kind}：列不出来（{type(e).__name__}: {str(e)[:200]}）")
            continue
        for r in repos:
            try:  # 列表接口不认 usedStorage（10/11 实测报 400），逐个仓库查；查不到就退回当前版本文件大小之和
                info = api.repo_info(r.id, repo_type=kind, expand=["usedStorage", "private"])
                used, src = getattr(info, "used_storage", None), "usedStorage"
            except Exception:
                info, used = None, None
            if used is None:
                info = api.repo_info(r.id, repo_type=kind, files_metadata=True)
                used, src = sum(f.size or 0 for f in (info.siblings or [])), "当前文件合计"
            priv = bool(getattr(info, "private", False))
            total[priv] += used
            print(f"  {kind:<7} {'私有' if priv else '公开'} {used / 1e9:8.2f} GB（{src}）  {r.id}")
    print(f"私有合计 {total[True] / 1e9:.2f} GB，公开合计 {total[False] / 1e9:.2f} GB；HF_STORAGE_DONE")
    return 0


def cmd_readme(a) -> int:
    need_env()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if a.visibility == "public" and not api.repo_exists(a.repo, repo_type="model"):  # 第 4 版审查 m3：仓库名写错时不多出一个公开仓库
        sys.exit(f"{a.repo} 不存在：readme 不新建公开仓库，停下")
    api.create_repo(a.repo, repo_type="model", private=a.visibility == "private", exist_ok=True)
    check_visibility(api, a.repo, a.visibility)
    want = sha256(Path(a.file))
    api.upload_file(path_or_fileobj=a.file, path_in_repo="README.md", repo_id=a.repo, repo_type="model", commit_message=a.message)
    tmp = Path("runs/hf_readme_check")
    tmp.mkdir(parents=True, exist_ok=True)
    got = sha256(Path(hf_hub_download(a.repo, "README.md", repo_type="model", local_dir=str(tmp), force_download=True)))
    rmtree(tmp)
    ok = got == want
    print(f"README.md 传到 {a.repo}；下载回来 sha256 {got[:16]}…，本地 {want[:16]}…；" + ("HF_README_OK" if ok else "HF_README_FAIL"))
    return 0 if ok else 1


def cmd_upload(a) -> int:
    need_env()
    from huggingface_hub import HfApi
    rows = json.loads(Path(a.manifest).read_text())
    api = HfApi()  # 先查仓库，再建暂存目录：查不过就停，不留硬链接
    if not api.repo_exists(a.repo, repo_type="model") or not api.file_exists(a.repo, "README.md", repo_type="model"):
        sys.exit(f"{a.repo} 不存在或根目录没有 README.md：每个上传过的 HF 仓库都要有 README（用户 10/8 的要求），先跑 readme 子命令")
    check_visibility(api, a.repo, a.visibility)
    stage = STAGE / a.prefix
    for r in rows:
        src = RUNS / r["run"] / r["file"]
        dst = stage / repo_path(a.prefix, r).split("/", 1)[1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)  # 硬链接，不多占盘
    t0 = time.time()
    print(f"{time.strftime('%F %T')} 开始上传 {len(rows)} 个文件到{vis_name(a.visibility == 'private')}仓库 {a.repo}"
          f"（{sum(r['size'] for r in rows) / 1e9:.2f} GB）", flush=True)
    api.upload_large_folder(repo_id=a.repo, folder_path=str(STAGE), repo_type="model",
                            allow_patterns=[f"{a.prefix}/**"], private=a.visibility == "private", num_workers=2)  # 训练机上只占 2 个并发
    print(f"{time.strftime('%F %T')} 上传结束，用时 {(time.time() - t0) / 60:.1f} 分钟；HF_UPLOAD_DONE", flush=True)
    return 0


def hub_files(repo: str, prefix: str) -> dict:
    """仓库里 <机器名>/ 下每个 LFS 文件的 (大小, sha256)。"""
    from huggingface_hub import HfApi
    hub = {}
    for f in HfApi().list_repo_tree(repo, repo_type="model", path_in_repo=prefix, recursive=True, expand=True):
        lfs = getattr(f, "lfs", None)
        if lfs is not None:
            hub[f.path] = (f.size, getattr(lfs, "sha256", None))
    return hub


def cmd_hubsubset(a) -> int:
    need_env()
    rows = json.loads(Path(a.manifest).read_text())
    hub = hub_files(a.repo, a.prefix)
    keep, missing, odd = [], 0, []
    for r in rows:
        size, sha = hub.get(repo_path(a.prefix, r), (None, None))
        if size is None:
            missing += 1
        elif size == r["size"] and sha == r["sha256"]:
            keep.append(r)
        else:
            odd.append(f"{r['run']}/{r['file']}（仓库 {size} / {(sha or '无')[:12]}…，清单 {r['size']} / {r['sha256'][:12]}…）")
    Path(a.out).write_text(json.dumps(keep, indent=1) + "\n")
    if a.rest_out:
        rest = [r for r in rows if repo_path(a.prefix, r) not in hub]
        Path(a.rest_out).write_text(json.dumps(rest, indent=1) + "\n")
        print(f"仓库里还没有的 {len(rest)} 行写到 {a.rest_out}（{sum(r['size'] for r in rest)} 字节）")
    kinds = sorted({r["kind"] for r in rows})
    print("已在仓库且一致：" + "；".join(f"{k} {sum(r['kind'] == k for r in keep)} / {sum(r['kind'] == k for r in rows)}" for k in kinds)
          + f"；共 {len(keep)} / {len(rows)} 个、{sum(r['size'] for r in keep)} 字节、{len({r['run'] for r in keep})} 条 run；仓库里没有的 {missing} 个")
    for o in odd:
        print(f"  不一致 {o}")
    print(f"子清单写到 {a.out}；" + ("HF_SUBSET_OK" if not odd else f"HF_SUBSET_MISMATCH {len(odd)}"))
    return 0 if not odd else 1


def cmd_verify(a) -> int:
    need_env()
    rows = json.loads(Path(a.manifest).read_text())
    hub = hub_files(a.repo, a.prefix)
    out, bad = [], 0
    for r in rows:
        path = repo_path(a.prefix, r)
        size, sha = hub.get(path, (None, None))
        ok = size == r["size"] and sha == r["sha256"]
        bad += not ok
        out.append({**r, "repo": a.repo, "path_in_repo": path, "hub_size": size, "hub_sha256": sha, "match": ok})
        print(f"  {'ok ' if ok else 'BAD'} {path}  本地 {r['size']} / {r['sha256'][:12]}…  仓库 {size} / {(sha or '无')[:12]}…")
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(rows) - bad} / {len(rows)} 个文件核对一致；" + ("HF_VERIFY_OK" if not bad else f"HF_VERIFY_FAIL {bad}"))
    return 0 if not bad else 1


def rmtree(root: Path) -> None:
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    root.rmdir()


def spot_pick(rows: list, n: int, per_run: bool, seed: int) -> list:
    """抽查挑哪些文件：per_run 时每条 run 挑它的最终 state（有的话；那是这条 run 唯一一份不可再生的 state），
    再从其余文件里随机挑 1 个；否则从全部里随机挑 n 个。"""
    rng = random.Random(seed)
    if not per_run:
        return rng.sample(rows, min(n, len(rows)))
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    pick = []
    for _, rs in sorted(by_run.items()):
        rest = [r for r in rs if r.get("kind") != "final"]
        pick += [r for r in rs if r.get("kind") == "final"] + ([rng.choice(rest)] if rest else [])
    return pick


def free_mb() -> int:
    st = os.statvfs(".")
    return st.f_bavail * st.f_frsize // 1048576


def xet_cache_dir() -> Path:
    if os.environ.get("HF_XET_CACHE"):
        return Path(os.environ["HF_XET_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface")
    return Path(home) / "xet"


def dir_mb(root: Path) -> int:
    total = 0
    for f in (root.rglob("*") if root.exists() else []):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:  # 缓存里的文件可能正被别的进程删
            pass
    return total // 1048576


def cmd_spotcheck(a) -> int:
    need_env()
    from huggingface_hub import hf_hub_download
    rows = json.loads(Path(a.verified).read_text())
    if not all(r["match"] for r in rows):
        sys.exit("核对结果里有没对上的文件，不抽查")
    seed = int(time.time())
    pick = spot_pick(rows, a.n, a.per_run, seed)
    cache = xet_cache_dir()
    cache0 = dir_mb(cache)
    print(f"抽查 {len(pick)} 个文件；xet 缓存 {cache} 开始时 {cache0} MB，可用 {free_mb()} MB"
          + (f"；每次下载前可用须不少于 {a.min_free_mb} MB 加这个文件的大小" if a.min_free_mb else "")
          + (f"；缓存比开始时多出 {a.max_cache_growth_mb} MB 以上就停" if a.max_cache_growth_mb else ""), flush=True)
    tmp = Path("runs/hf_spotcheck")
    tmp.mkdir(parents=True, exist_ok=True)
    out, bad = [], 0
    for r in pick:
        need = a.min_free_mb + r["size"] // 1048576 + 1
        if a.min_free_mb and free_mb() < need:
            rmtree(tmp)
            sys.exit(f"可用 {free_mb()} MB，不到 {need} MB（保底 {a.min_free_mb} MB 加这个文件），停在抽查，不写抽查结果、不删任何东西")
        local = Path(hf_hub_download(r["repo"], r["path_in_repo"], repo_type="model", local_dir=str(tmp)))
        got = sha256(local)
        size = local.stat().st_size
        ok = got == r["sha256"] and size == r["size"]
        bad += not ok
        out.append({"path_in_repo": r["path_in_repo"], "size": size, "sha256": got, "expected": r["sha256"], "match": ok})
        print(f"  {'ok ' if ok else 'BAD'} {r['path_in_repo']}  下载后 sha256 {got[:16]}…，清单 {r['sha256'][:16]}…", flush=True)
        local.unlink()
        grown = dir_mb(cache) - cache0
        if a.max_cache_growth_mb and grown > a.max_cache_growth_mb:
            rmtree(tmp)
            sys.exit(f"xet 缓存比开始时多了 {grown} MB，超过 {a.max_cache_growth_mb} MB，停在抽查，不写抽查结果、不删任何东西")
    rmtree(tmp)  # 下载的副本与 local_dir 里的缓存都删掉
    print(f"xet 缓存抽查后 {dir_mb(cache)} MB（开始时 {cache0} MB），可用 {free_mb()} MB")
    Path(a.out).write_text(json.dumps({"seed": seed, "checked": out, "ok": not bad}, indent=1) + "\n")
    print(f"抽查 {len(out)} 个，{len(out) - bad} 个一致（随机种子 {seed}）；" + ("HF_SPOTCHECK_OK" if not bad else f"HF_SPOTCHECK_FAIL {bad}"))
    return 0 if not bad else 1


def cmd_delete(a) -> int:
    rows = json.loads(Path(a.verified).read_text())
    if not all(r["match"] for r in rows):
        sys.exit("核对结果里有没对上的文件，一个都不删")
    spot = json.loads(Path(a.spotcheck).read_text())
    verified_paths = {r["path_in_repo"] for r in rows}
    if not spot.get("ok") or not spot.get("checked") or not all(c["path_in_repo"] in verified_paths for c in spot["checked"]):
        sys.exit("抽查没通过（或抽查的文件不在核对清单里），一个都不删")
    has_policy = any(r["kind"] == "policy" for r in rows)
    if has_policy and chain_active():
        sys.exit("有训练链在跑（runs/.chain-active-*），policy 一个都不删")
    if any(r["kind"] == "final" for r in rows) and chain_active():
        sys.exit("有训练链在跑（runs/.chain-active-*），最终 state 一个都不删")
    exported = exported_steps() if has_policy else {}
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    freed = 0
    for run, rs in sorted(by_run.items()):
        d = RUNS / run
        if not (d / "DONE").exists():
            print(f"跳过 {run}：没有 DONE")
            continue
        st_rows = [r for r in rs if r["kind"] in ("smoke", "formal")]
        pol_rows = [r for r in rs if r["kind"] == "policy"]
        keep = keep_state(d, run) if st_rows else None
        if keep == "__none__":
            print(f"跳过 {run}：找不到完整的最终 state，整个 run 不动")
            continue
        fin = final_state(d) if any(r["kind"] == "final" for r in rs) else None
        prot = protected_policies(d, run, min(r["keep"] for r in pol_rows), exported) if pol_rows else set()
        if prot is None:
            print(f"跳过 {run}：步数最大的 policy 不完整或 STATE_PRUNED 读不懂，整个 run 不动")
            continue
        todo = []
        for r in rs:
            p = d / r["file"]
            if r["kind"] == "policy" and r["step"] in prot:
                print(f"跳过 {run}/{r['file']}：按现在的文件重算，这份 policy 受保护")
                continue
            if r["kind"] in ("smoke", "formal") and r["file"] == keep:
                print(f"跳过 {run}/{r['file']}：是这个正式 run 步数最大的完整 state")
                continue
            if r["kind"] == "final" and r["file"] != fin:
                print(f"跳过 {run}/{r['file']}：按现在的文件重算，它已不是步数最大且与 policy 成对完整的那份 state")
                continue
            if not p.exists():
                print(f"跳过 {run}/{r['file']}：已不在（之前删过）")
                continue
            if p.stat().st_size != r["size"] or sha256(p) != r["sha256"]:
                print(f"跳过 {run}/{r['file']}：和清单里的不一样了")
                continue
            todo.append(r)
        if a.yes and todo:  # 先写记录再删：中途断了，记录里也有这些文件在仓库里的位置
            arch = d / "ARCHIVED.json"
            old = json.loads(arch.read_text()) if arch.exists() else []
            files = [{k: r[k] for k in ("file", "step", "size", "sha256", "repo", "path_in_repo")} for r in todo]
            arch.write_text(json.dumps(old + [{"time": time.strftime("%FT%T%z"), "files": files}], indent=1) + "\n")
        for r in todo:
            print(f"{'删' if a.yes else '将删'} {run}/{r['file']}（{r['size'] / 1e6:.0f} MB，在 {r['repo']}:{r['path_in_repo']}）")
            if a.yes:
                (d / r["file"]).unlink()
                freed += r["size"]
        if a.yes and st_rows and is_smoke(run) and not states(d) and not (d / "STATE_PRUNED").exists():
            pol = sorted(int(m.group(1)) for p in d.glob("policy_*.safetensors") if (m := POLICY.fullmatch(p.name)))
            if pol:
                (d / "STATE_PRUNED").write_text(f"{pol[-1]}\n")
                print(f"写 {run}/STATE_PRUNED（{pol[-1]} 步）")
    if a.yes and STAGE.exists():
        rmtree(STAGE)  # 硬链接也要删掉，空间才真正释放
    est = sum(r["size"] for r in rows) / 1e9
    print(f"释放 {freed / 1e9:.2f} GB；HF_DELETE_DONE" if a.yes else f"预计释放约 {est:.2f} GB（演习，没删）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("whoami")
    w.add_argument("--repo", required=True)
    w.add_argument("--visibility", choices=VIS, default="private")
    v = sub.add_parser("visibility")
    v.add_argument("--repo", required=True)
    v.add_argument("--to", choices=VIS, required=True)
    v.add_argument("--confirm", required=True, help="再写一遍仓库名，与 --repo 相同才改")
    v.add_argument("--yes", action="store_true")
    sub.add_parser("storage")
    rd = sub.add_parser("readme")
    rd.add_argument("--repo", required=True)
    rd.add_argument("--file", required=True)
    rd.add_argument("--message", default="Update README")
    rd.add_argument("--visibility", choices=VIS, default="private")
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--kind", type=kinds_arg, default=["state"], help="state、policy、final，可用逗号连几种")
    p.add_argument("--keep", type=int, default=9, help="policy：每条 run 至少留步数最大的几份")
    for name in ("upload", "verify", "hubsubset"):
        q = sub.add_parser(name)
        q.add_argument("--manifest", required=True)
        q.add_argument("--repo", required=True)
        q.add_argument("--prefix", required=True)
        if name != "upload":
            q.add_argument("--out", required=True)
        else:
            q.add_argument("--visibility", choices=VIS, default="private")
        if name == "hubsubset":
            q.add_argument("--rest-out", default=None)
    c = sub.add_parser("spotcheck")
    c.add_argument("--verified", required=True)
    c.add_argument("--n", type=int, default=3)
    c.add_argument("--per-run", action="store_true", help="每条 run 随机挑 1 个，有最终 state 时固定挑它再随机补 1 个（忽略 --n）")
    c.add_argument("--min-free-mb", type=int, default=0, help="每次下载前可用空间不到这个数加文件大小就停；0 不查")
    c.add_argument("--max-cache-growth-mb", type=int, default=0, help="xet 缓存比开始时多出这么多就停；0 不查")
    c.add_argument("--out", required=True)
    d = sub.add_parser("delete")
    d.add_argument("--verified", required=True)
    d.add_argument("--spotcheck", required=True)
    d.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan" and a.keep < 1:
        ap.error("--keep 至少是 1")
    return {"whoami": cmd_whoami, "visibility": cmd_visibility, "storage": cmd_storage, "hubsubset": cmd_hubsubset, "readme": cmd_readme, "plan": cmd_plan, "upload": cmd_upload, "verify": cmd_verify, "spotcheck": cmd_spotcheck, "delete": cmd_delete}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
