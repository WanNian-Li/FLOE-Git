#!/bin/bash
# 容器内存采样（cgroup v2）。用法: _memmon.sh <输出csv> [采样间隔秒]
#
# ⚠️ 为什么不能只看 anon（2026-07-25）：CLAUDE.md 里"判断容器内存是否安全要看 anon"那条
# 是在【训练】路径下总结的——tile 采样的 DataLoader payload 只有几 MB，走不走共享内存
# 无所谓。但【验证】路径一个样本就是一整景（最大 8ch×6767×37468≈8GB），多 worker 时这些
# 张量经 torch.multiprocessing 走 /dev/shm，在 cgroup v2 里记进 **shmem**（与 file 同属
# memory.stat 的文件类），不计入 anon。tmpfs 页不可回收，正是宿主机 OOM killer 看得见的
# 东西——07-23 那两次 oom_kill=0 的宿主机级 SIGKILL 与此自洽。所以调 num_workers_val
# 之前必须同时盯 shmem。
#
# 输出列: epoch_s,anon,file,shmem,current,shm_used_kb
OUT="${1:?用法: _memmon.sh <输出csv> [间隔秒]}"
INT="${2:-2}"
echo "t,anon,file,shmem,current,shm_used_kb" > "$OUT"
while true; do
  A=$(grep '^anon '  /sys/fs/cgroup/memory.stat | awk '{print $2}')
  F=$(grep '^file '  /sys/fs/cgroup/memory.stat | awk '{print $2}')
  S=$(grep '^shmem ' /sys/fs/cgroup/memory.stat | awk '{print $2}')
  C=$(cat /sys/fs/cgroup/memory.current)
  D=$(df /dev/shm | tail -1 | awk '{print $3}')
  echo "$(date +%s),$A,$F,$S,$C,$D" >> "$OUT"
  sleep "$INT"
done
