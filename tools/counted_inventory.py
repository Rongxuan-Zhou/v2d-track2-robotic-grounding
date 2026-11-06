# 冻结后搭提交容器用的只读清单（Mac 上、仓库根执行；官方要求见工具包 update_7e1fdd00 的 CONTAINER_SUBMISSION.md）。
# 1. 用 kaggle CLI 列 12 个 Track 2 榜上我们的提交，按榜取“计分的那份”：没有手选 Final Submission 时是分最好的那份
#    （MPPE 越小越好，其余越大越好）；同分的几份都列出来，容器要都覆盖。
# 2. 每份计分文件：本地 runs/submission/csv/ 下同名 CSV 的 sha256，以及 CSV 里五个溯源列和 commit 的取值。
# 3. 每份计分文件的每集 checkpoint_sha256，在 runs/submission/ 下所有 episode_XXXXXX.json 里找对应的 checkpoint 路径
#    （evaluator 写的 /repo/runs/flash_chord/<run>/policy_<步数>.safetensors），机器取 run 名最后一段。
#    写 runs/inventory/ckpt_<机器>.tsv（sha256、路径），给 ckpt_check_remote.sh 在各机器上逐个核对。
# 只读：不上传、不改任何提交文件。kaggle 凭据只用 CLI 自己读的 ~/.kaggle/access_token。
# 用法：python tools/counted_inventory.py <kaggle 可执行文件>
import csv
import glob
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

import pandas as pd

KAGGLE = sys.argv[1]
BOARDS = [(t, m) for t in (1, 2, 3) for m in ('auc', 'sp-sr', 'mp-sr', 'mppe')]
PROV = ['eval_script_sha256', 'eval_code_sha256', 'packer_script_sha256', 'uploader_script_sha256', 'code_commit_url']
HOSTS = ('ml40', 'mf4090', 'l40n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def listing(comp):
    out = subprocess.run([KAGGLE, 'competitions', 'submissions', comp, '-v'], capture_output=True, text=True, check=True).stdout
    lines = [l for l in out.splitlines() if not l.startswith('Warning')]
    return list(csv.DictReader(io.StringIO('\n'.join(lines))))


counted = defaultdict(list)  # 文件名 → [(榜, 提交 ID, 分数)]
print('## 1. 各榜计分的提交（没有手选时取分最好的；同分全列）')
print('榜\t计分提交 ID\t文件\t分数\t日期（UTC）\t同分份数')
for t, m in BOARDS:
    comp = f'v2d-challenge-track2-tier{t}-{m}'
    rows = [r for r in listing(comp) if r['status'].endswith('COMPLETE') and r['publicScore'] not in ('', None)]
    if not rows:
        print(f'tier{t}-{m}\t无已完成的提交')
        continue
    key = (lambda r: float(r['publicScore'])) if m == 'mppe' else (lambda r: -float(r['publicScore']))
    best = min(key(r) for r in rows)
    tied = [r for r in rows if key(r) == best]
    for r in tied:
        print(f"tier{t}-{m}\t{r['ref']}\t{r['fileName']}\t{r['publicScore']}\t{r['date'][:19]}\t{len(tied)}")
        counted[r['fileName']].append((f'tier{t}-{m}', r['ref'], r['publicScore']))

# evaluator JSON 索引：checkpoint sha256 → checkpoint 路径
index = defaultdict(set)
for p in glob.glob('runs/submission/**/episode_*.json', recursive=True):
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        continue
    if isinstance(d, dict) and d.get('checkpoint_sha256') and d.get('checkpoint'):
        index[d['checkpoint_sha256']].add(d['checkpoint'])

print('\n## 2. 计分文件与溯源列')
print('文件\t覆盖的榜\t本地 CSV sha256\t回执数\t' + '\t'.join(PROV))
need = defaultdict(set)  # 机器 → {(sha, 路径)}
missing = []
per_ep = []
for name, boards in sorted(counted.items()):
    path = f'runs/submission/csv/{name}'
    if not os.path.exists(path):
        print(f"{name}\t{','.join(b for b, _, _ in boards)}\t本地没有这份 CSV")
        continue
    df = pd.read_csv(path, usecols=['row_id', 'checkpoint_sha256'] + PROV, dtype=str)
    receipts = len(glob.glob(path + '.*.kaggle-receipt.json'))
    vals = ['/'.join(sorted(df[c].unique()))[:200] for c in PROV]
    print(f"{name}\t{','.join(b for b, _, _ in boards)}\t{sha256(path)}\t{receipts}\t" + '\t'.join(vals))
    df['ep'] = df['row_id'].str.extract(r'/e(\d+)/')[0].astype(int)
    for ep, sha in sorted(set(zip(df['ep'], df['checkpoint_sha256']))):
        paths = sorted(index.get(sha, ()))
        host = '?'
        if paths:
            run = paths[0].split('/')[-2]
            host = next((h for h in HOSTS if run.endswith('-' + h)), '?')
            for p in paths:
                need[host].add((sha, p))
        else:
            missing.append((name, ep, sha))
        per_ep.append((name, ep, sha, host, ' | '.join(paths) or '没找到 evaluator JSON'))

print('\n## 3. 每集 checkpoint（文件、集、sha256、机器、路径）')
for row in per_ep:
    print('\t'.join(map(str, row)))

os.makedirs('runs/inventory', exist_ok=True)
print('\n## 4. 各机器待核对的 checkpoint')
for host, items in sorted(need.items()):
    out = f'runs/inventory/ckpt_{host}.tsv'
    with open(out, 'w') as f:
        for sha, p in sorted(items, key=lambda x: x[1]):
            f.write(f'{sha}\t{p}\n')
    print(f'{host}\t{len(items)} 个\t{out}')
print(f'\n计分文件 {len(counted)} 份，每集 checkpoint 共 {len(per_ep)} 行，找不到路径的 {len(missing)} 行')
for name, ep, sha in missing:
    print(f'NO_JSON\t{name}\tep{ep}\t{sha}')
print('COUNTED_INVENTORY_DONE')
