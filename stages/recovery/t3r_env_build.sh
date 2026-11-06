#!/usr/bin/env bash
# t3takeover 第 1b 级：在 /lab_data/rxzhou/v2d/t3recon/env 下建我们自己的两个 Python 环境（训练机 ml40 上执行；只用 CPU、只写我们的目录）。
# 计划 agent-doc/plan-t3takeover-1b-2026-10-21.md 第 1 节第 2、3 条，复核改动第 4、5、6 条，判据 E0。
# 外层：经 stages/recovery/t3r_run.sh 在命名空间里跑内层——她的目录只读，她的 venv、gate/venv、gate/.py 被空的只读 tmpfs 盖住（T3R_MASK），
#       uv 与 pip 就算选错了解释器也碰不到她的；内层跑完，外层用 t3r_env_lists.py --compare-list 拿我们的两个环境与她的清单记录比 RECORD 摘要，
#       再比 CPython 3.11.17 解释器与她的 sha256（只读）。
# 内层：boot venv（系统 python3.10）里从腾讯云镜像装 uv 0.13.0（同她的）；uv 从 npmmirror 的 python-build-standalone 装 CPython 3.11.17（--no-bin）；
#       gate = uv venv（--managed-python，UV_PYTHON_DOWNLOADS=never）+ uv pip install --no-deps（默认索引腾讯云，torch 的 +cpu 经阿里云 pytorch-wheels 平铺页 -f）；
#       host = python3.10 -m venv + pip install --no-deps（腾讯云）；两个环境各做一次 import 冒烟。uv 的缓存与配置都在 env/ 下（UV_NO_CONFIG=1）。
# 用法：stages/recovery/t3r_env_build.sh <她的宿主机清单记录> <她的 gate 清单记录>   （第 1 节第 1 条存下的 results/t3recon/env_list_hers_{host,gate}.txt）
#       stages/recovery/t3r_env_build.sh <两份记录> --recheck            只重做比较（不建）
#       stages/recovery/t3r_env_build.sh <两份记录> --host-pip <pip 参数...>   在命名空间里对 env/host 跑一次 pip（同样盖住她的解释器），再重做比较
# host 先把 pip 换成清单里的 26.2.1 再装其余的：venv 自带的 pip 22.0.2 没有 wheel 包时，只有源码包的发行版（antlr4-python3-runtime）会走老式 setup.py
# 装成 egg-info，与她的 dist-info 对不上（10/21 06:5x 第一次建时遇到）。
# 最后一行 T3R_ENV_BUILD_DONE（两份比较都没有不同）或 T3R_ENV_BUILD_FAILED <哪一步>。env/ 已存在就拒绝。
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
R=/lab_data/rxzhou/v2d/t3recon; E="$R/env"
IDX=https://mirrors.cloud.tencent.com/pypi/simple
TORCH_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/
PYMIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone

if [ "${1:-}" = __inner ]; then   # 命名空间里
  set -e
  say() { echo "=== $(date '+%F %T %Z') $*"; }
  export UV_NO_CONFIG=1 UV_CACHE_DIR="$E/.uvcache" UV_PYTHON_INSTALL_DIR="$E/.py" UV_PYTHON_INSTALL_MIRROR="$PYMIRROR" PIP_DISABLE_PIP_VERSION_CHECK=1
  say "boot venv 与 uv"
  /usr/bin/python3.10 -m venv "$E/boot"
  "$E/boot/bin/pip" install -q --no-cache-dir -i "$IDX" uv==0.13.0
  "$E/boot/bin/uv" --version
  say "CPython 3.11.17"
  "$E/boot/bin/uv" python install --no-bin 3.11.17
  say "gate 环境"
  UV_PYTHON_DOWNLOADS=never "$E/boot/bin/uv" venv --managed-python -p 3.11.17 "$E/gate"
  "$E/boot/bin/uv" pip install --python "$E/gate/bin/python" --no-deps -i "$IDX" -f "$TORCH_LINKS" -r "$REPO/stages/recovery/env/t3r_gate.txt"
  say "host 环境"
  /usr/bin/python3.10 -m venv "$E/host"
  "$E/host/bin/python" -m pip install -q --no-cache-dir -i "$IDX" "$(grep -m1 "^pip==" "$REPO/stages/recovery/env/t3r_host.txt")"
  "$E/host/bin/pip" install -q --no-cache-dir --no-deps -i "$IDX" --build-constraint "$REPO/stages/recovery/env/t3r_host_build.txt" \
    -r "$REPO/stages/recovery/env/t3r_host.txt"   # 源码包在构建隔离环境里照她的 setuptools 84.0.0 构建（pip 26 的 --build-constraint；PIP_CONSTRAINT 管不到构建依赖）
  say "import 冒烟"
  "$E/host/bin/python" -c 'import sys, numpy, pyarrow, scipy, trimesh; print("host", sys.version.split()[0], numpy.__version__, pyarrow.__version__, scipy.__version__, trimesh.__version__)'
  "$E/gate/bin/python" -c 'import sys, numpy, pyarrow, scipy, trimesh, torch, pinocchio, pink, soma, pxr, warp; print("gate", sys.version.split()[0], numpy.__version__, torch.__version__, pinocchio.__version__, warp.config.version)'
  say "T3R_ENV_INNER_DONE"
  exit 0
fi
if [ "${1:-}" = __host_pip ]; then   # 命名空间里
  shift
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  exec "$E/host/bin/pip" "$@"
fi

HOSTLIST=${1:?她的宿主机清单记录}; GATELIST=${2:?她的 gate 清单记录}; MODE=${3:-build}
case $MODE in build) [ -e "$E" ] && { echo "T3R_ENV_BUILD_FAILED $E 已存在"; exit 2; } ;; --recheck|--host-pip) [ -d "$E/host" ] && [ -d "$E/gate" ] || { echo "T3R_ENV_BUILD_FAILED 还没有 $E/host、$E/gate"; exit 2; } ;;
  *) echo "T3R_ENV_BUILD_FAILED 不认识 $MODE"; exit 2 ;; esac
for f in "$HOSTLIST" "$GATELIST" stages/recovery/env/t3r_host.txt stages/recovery/env/t3r_gate.txt; do [ -s "$f" ] || { echo "T3R_ENV_BUILD_FAILED 缺 $f"; exit 2; }; done
H="$(T3R_OTHER_DIR_ONLY=1 stages/recovery/t3r_other.sh)" || { echo "T3R_ENV_BUILD_FAILED 另一位成员的目录没过核对"; exit 2; }
mkdir -p "$E"
export T3R_HER="$(dirname "$H")" T3R_ROOT="$R" T3R_ALLOW="" T3R_MASK="$H/venv:$H/gate/venv:$H/gate/.py"
case $MODE in
  build) bash stages/recovery/t3r_run.sh /bin/bash "$REPO/stages/recovery/t3r_env_build.sh" __inner || { echo "T3R_ENV_BUILD_FAILED 内层"; exit 1; } ;;
  --host-pip) shift 3; bash stages/recovery/t3r_run.sh /bin/bash "$REPO/stages/recovery/t3r_env_build.sh" __host_pip "$@" || { echo "T3R_ENV_BUILD_FAILED host 的 pip"; exit 1; } ;;
esac
bad=0
echo "--- host 对她的清单（去掉 35 个本地包装）"
python3 stages/recovery/t3r_env_lists.py "$E/host" --compare-list "$HOSTLIST" --skip-direct-url || bad=1
echo "--- gate 对她的清单"
python3 stages/recovery/t3r_env_lists.py "$E/gate" --compare-list "$GATELIST" || bad=1
echo "--- CPython 3.11.17 解释器"
ours="$(sha256sum < "$(realpath -e "$E/gate/bin/python")" | cut -c1-64)"; hers="$(sha256sum < "$(realpath -e "$H/gate/venv/bin/python")" | cut -c1-64)"
echo "我们 ${ours:0:16}… 她 ${hers:0:16}… $([ "$ours" = "$hers" ] && echo 相同 || echo 不同)"
[ "$bad" = 0 ] && echo "T3R_ENV_BUILD_DONE" || { echo "T3R_ENV_BUILD_FAILED 环境与她的清单有不同（见上）"; exit 1; }
