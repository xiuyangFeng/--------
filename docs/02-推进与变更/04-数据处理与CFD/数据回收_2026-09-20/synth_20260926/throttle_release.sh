#!/bin/bash
# keep <= MAXRUN synthetic Fluent jobs running and >= MINFREE_TB free on /public; release held jobs one at a time (2026-09-26)
export PATH=/public/slurm/bin:$PATH
MAXRUN=${1:-10}; MINFREE_TB=${2:-1.5}
while true; do
  held=$(squeue -u cy -h -o "%i %j %t %r" | awk '$2=="Fluent" && $3=="PD" && $4 ~ /JobHeldUser/ {print $1}' | sort -n | head -200)
  [ -z "$held" ] && { echo "$(date -Is) no held jobs left"; break; }
  run=$(squeue -u cy -h -o "%j %t" | awk '$1=="Fluent" && $2=="R"' | wc -l)
  pend_free=$(squeue -u cy -h -o "%i %j %t %r" | awk '$2=="Fluent" && $3=="PD" && $4 !~ /JobHeldUser|Dependency/' | wc -l)
  free_tb=$(df -BG /public | awk 'NR==2{gsub("G","",$4); printf "%.2f", $4/1024}')
  if [ "$run" -lt "$MAXRUN" ] && [ "$pend_free" -eq 0 ] && [ "$(echo "$free_tb >= $MINFREE_TB" | bc)" -eq 1 ]; then
    j=$(echo "$held" | head -1); scontrol release $j; echo "$(date -Is) released $j (running $run, free ${free_tb} TB)"
    sleep 45
  else
    sleep 240
  fi
done
