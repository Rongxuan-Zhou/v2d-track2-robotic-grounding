#!/usr/bin/env bash
# 比较两版官方 submission kit（Mac 上、仓库根执行，只读）：KIT_MANIFEST.json 里记的文件哈希逐个比，列出新增、删除、改动的文件；
# 改动的文件里，与 Track 2 提交有关的（打包器、上传器、计分代码、v2dlb、tracks.json、README、样例 CSV、评测入口与配置）打出 diff（样例 CSV 只比行数与 row_id）。
# 用法：scripts/dev/record submission/kit_diff_<旧>_<新> stages/evaluate/kit_diff.sh <旧 kit 目录> <新 kit 目录>
#   kit 目录指 third_party/v2d_submission_kit/update_<哈希>/v2d_submission_kit
set -uo pipefail
cd "$(dirname "$0")/../.."
OLD=${1:?旧 kit 目录}; NEW=${2:?新 kit 目录}
python3 - "$OLD" "$NEW" <<'EOF'
import difflib, hashlib, json, sys
from pathlib import Path
old, new = Path(sys.argv[1]), Path(sys.argv[2])

def files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}

a, b = files(old), files(new)
added, removed = sorted(set(b) - set(a)), sorted(set(a) - set(b))
changed = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print(f"旧 {old}：{len(a)} 个文件；新 {new}：{len(b)} 个文件")
print(f"新增 {len(added)}：{added}")
print(f"删除 {len(removed)}：{removed}")
print(f"改动 {len(changed)}：{changed}")
for name in ("KIT_MANIFEST.json",):
    try:
        ma = json.loads((old / name).read_text()); mb = json.loads((new / name).read_text())
        print(f"KIT_MANIFEST evaluator：旧 {json.dumps(ma.get('evaluator'), ensure_ascii=False)[:300]}")
        print(f"KIT_MANIFEST evaluator：新 {json.dumps(mb.get('evaluator'), ensure_ascii=False)[:300]}")
    except Exception as e:
        print(f"读 {name} 出错：{e}")
keys = ("tools/pack_submission.py", "submit/v2d_submit.py", "metric_code/", "v2dlb/", "config/tracks.json", "README.md", "tracks/track_2/",
        "flash_chord/scripts/evaluate_policy.py", "flash_chord/src/flash_chord/configs/evaluation/", "flash_chord/src/flash_chord/evaluation/")
for k in changed:
    if not any(k == key or k.startswith(key) for key in keys) and not k.endswith(".csv"):
        continue
    print(f"\n======== {k}")
    if k.endswith(".csv"):
        ra = (old / k).read_text().splitlines(); rb = (new / k).read_text().splitlines()
        ida = [x.split(",")[0] for x in ra]; idb = [x.split(",")[0] for x in rb]
        print(f"行数 {len(ra)} → {len(rb)}；表头 {ra[0][:120]} → {rb[0][:120]}；row_id 序列{'相同' if ida == idb else '不同'}")
        continue
    diff = difflib.unified_diff((old / k).read_text().splitlines(), (new / k).read_text().splitlines(), "旧", "新", lineterm="", n=2)
    lines = list(diff)
    print("\n".join(lines[:200]))
    if len(lines) > 200:
        print(f"…（还有 {len(lines) - 200} 行）")
print("\nKIT_DIFF_DONE")
EOF
