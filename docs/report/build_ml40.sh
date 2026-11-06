#!/usr/bin/env bash
# 在 ml40 上用 tectonic 0.15.0（/lab_data/rxzhou/v2d/tools_bin/tectonic，11/03 装，tar.gz sha256 dfb82876…6407）编译技术报告，PDF 拉回 docs/report/main.pdf。
# Mac 上、仓库根执行。Mac 与三台服务器都没有 pdflatex；tectonic 用的是 XeTeX 引擎，所以只在 ml40 的编译副本里改两处，仓库里的模板原样不动：
#   1. main.tex 的 \pdfoutput=1 注释掉（XeTeX 下它让 hyperref 选错驱动 hpdftex.def，报 Undefined control sequence）；
#   2. nvidiatechreport.cls 的 microtype tracking=smallcaps 去掉（tracking 只在 pdfTeX 下可用，XeTeX 下报错停）。只影响小型大写字母的字距。
# 第一次编译要从 tectonic 的服务器下宏包，经 tools/hf_remote.sh 带上代理（凭据只走 stdin）；之后用 ~/.cache/Tectonic 里的缓存。
# 最后一行 REPORT_BUILD_OK <页数> 或 REPORT_BUILD_FAILED。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
B=/lab_data/rxzhou/v2d/report_build; T=/lab_data/rxzhou/v2d/tools_bin/tectonic
ssh -n ml40 "rm -rf $B && mkdir -p $B" || { echo REPORT_BUILD_FAILED; exit 1; }
scp -q -r docs/report/main.tex docs/report/references.bib docs/report/*.cls docs/report/*.sty docs/report/assets ml40:$B/ < /dev/null || { echo REPORT_BUILD_FAILED; exit 1; }
[ -d docs/report/figs ] && scp -q -r docs/report/figs ml40:$B/ < /dev/null
V2D_ENV=ml40 tools/hf_remote.sh "cd $B && sed -i 's/^\\\\pdfoutput=1/%&/' main.tex && sed -i 's/\\[tracking=smallcaps\\]//' nvidiatechreport.cls && $T -X compile main.tex > tectonic.log 2>&1; echo rc=\$?; grep -E '^(warning|error)' tectonic.log | grep -v -E 'Font shape|Overfull|Underfull' | head -20" 2>&1 | grep -v '^→'
scp -q ml40:$B/main.pdf docs/report/main.pdf < /dev/null || { echo REPORT_BUILD_FAILED; exit 1; }
pages=$(python3 -c "import pypdf; print(len(pypdf.PdfReader('docs/report/main.pdf').pages))")
echo "REPORT_BUILD_OK ${pages}"
