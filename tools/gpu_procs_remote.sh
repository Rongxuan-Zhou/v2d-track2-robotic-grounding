# 服务器上的只读快照：GPU 上每个计算进程的命令行、已运行时长、祖先进程链；全部运行中容器的命令与挂载；GPU 进程所属的其他用户在宿主机上
# 的脚本类进程（命令行含 .sh、.py 或 docker）。训练机或 ml40 都可用，经 stdin 送过去执行。
# 用途：同组成员占着 GPU 时，看跑的是什么、已经跑了多久，以便估计还要占多久。
# 不改文件、不进别人的容器、不读环境变量（不碰 /proc/<pid>/environ，docker inspect 只取命令、镜像、时间与挂载）。
# 记录要进 git：不是本用户、也不是 root 的进程，用户名与它的 home 目录名一律换成 <其他用户>；命令行里 token/key/secret/password 一类参数的值换成 <略>。
# 用法（Mac，仓库根）：scripts/dev/record ops/gpu_procs_ml40_<时间> ssh ml40 'bash -s' < tools/gpu_procs_remote.sh
set -uo pipefail
me="$(id -un)"
mapfile -t gpids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -E '^[0-9]+$')
declare -A seen=(); chains=(); users=()
for pid in "${gpids[@]}"; do
  chain=(); p=$pid
  while [[ $p =~ ^[0-9]+$ ]] && [ "$p" -gt 1 ]; do chain+=("$p"); p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"; done
  chains+=("${chain[*]}")
  for q in "${chain[@]}"; do
    u="$(ps -o user= -p "$q" 2>/dev/null | tr -d ' ')"
    [ -n "$u" ] && [ "$u" != "$me" ] && [ "$u" != root ] && [ -z "${seen[$u]:-}" ] && { seen[$u]=1; users+=("$u"); }
  done
done
for u in "${users[@]}"; do h="$(getent passwd "$u" | cut -d: -f6)"; h="${h##*/}"; [ -n "$h" ] && [ "$h" != "$u" ] && users+=("$h"); done
# GPU 空着时上面收不到名字，但容器的挂载路径里仍有别人的目录名：/lab_data 与 /home 下属主不是本用户、也不是 root 的目录，
# 目录名和属主一并换掉（2026-10-09 卡空时的快照漏过一次，没提交）。长的名字先换，免得短名字是长名字的一部分时只换掉一半。
# 这些名字只用来脱敏；列宿主机进程仍只列 GPU 进程的属主（users）。
names=("${users[@]}")
for d in /lab_data/* /home/*; do
  [ -d "$d" ] || continue
  o="$(stat -c %U "$d" 2>/dev/null)"
  [ -n "$o" ] && [ "$o" != "$me" ] && [ "$o" != root ] && names+=("${d##*/}" "$o")
done
mapfile -t names < <(printf '%s\n' "${names[@]}" | awk 'NF' | sort -u | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)
red() {  # 换掉别人的用户名、home 目录名，以及像凭据的参数值
  local s; s="$(cat)"
  for u in "${names[@]}"; do [ -n "$u" ] && s="${s//$u/<其他用户>}"; done
  printf '%s\n' "$s" | sed -E 's/(([Tt][Oo][Kk][Ee][Nn]|[Kk][Ee][Yy]|[Ss][Ee][Cc][Rr][Ee][Tt]|[Pp][Aa][Ss][Ss][Ww])[A-Za-z_-]*[= ]+)[^ ]+/\1<略>/g'
}
who_of() { local u; u="$(ps -o user= -p "$1" 2>/dev/null | tr -d ' ')"; if [ "$u" = "$me" ]; then echo 本用户; elif [ "$u" = root ]; then echo root; else echo 其他用户; fi; }
{
  echo "== $(hostname) $(date '+%F %T %Z')"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  i=0
  for pid in "${gpids[@]}"; do
    echo "## GPU 进程 $pid：祖先链（自下而上：pid、归属、已运行、开始时间、命令行）"
    for q in ${chains[$i]}; do
      echo "  $q $(who_of "$q") $(ps -o etime= -p "$q" | tr -d ' ') [$(ps -o lstart= -p "$q")] $(ps -o args= -p "$q")"
    done
    i=$((i + 1))
  done
  echo "## 运行中的容器（只取镜像、命令、时间、工作目录与挂载）"
  for c in $(docker ps -q 2>/dev/null); do
    docker inspect --format 'name {{.Name}}
image {{.Config.Image}}
created {{.Created}}  started {{.State.StartedAt}}
entrypoint {{json .Config.Entrypoint}}
cmd {{json .Config.Cmd}}
workdir {{.Config.WorkingDir}}
mounts {{range .Mounts}}{{.Source}} -> {{.Destination}}; {{end}}' "$c" 2>&1
  done
  echo "## 运行中的容器（列表）"
  docker ps --format '{{.ID}} {{.Image}} {{.RunningFor}} {{.Status}} {{.Names}}' 2>&1
  echo "## 其他用户在宿主机上的脚本类进程（pid、已运行、开始时间、命令行）"
  for u in "${users[@]}"; do ps -u "$u" -o pid=,etime=,lstart=,args= 2>/dev/null | grep -E '\.sh|\.py|docker' | grep -v grep || true; done
} | red
