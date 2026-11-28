#!/bin/bash
# #2742 measurement matrix. One process per row; rounds interleave configs, order reversed on even rounds.
OUT=~/scratch/t2742-proto/runs
BIN=~/scratch/t2742-proto/target/release/examples
cd ~/scratch/t2742-proto/build
G=$((1<<30)); INF=$((1<<50))
{ echo "host $(hostname) $(date -Is)"; echo "affinity $(taskset -cp $$)"; lscpu | grep -E "Model name|^L3|Socket|Thread|^CPU\(s\)"; lscpu | grep -i "modell\|L3"; free -g | head -2; nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv; rustc --version; } > $OUT/env.txt 2>&1
CFGS=("1G:$G:1024" "4G:$((4*G)):1024" "8G:$((8*G)):1024" "24G:$((24*G)):1024" "cap128:$INF:128" "cap32:$INF:32")
run_n30() { # label limit cap round
  echo "== n30m14 $1 round $4 load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/n30m14.txt
  T2742_LIMIT_BYTES=$2 T2742_CAPACITY=$3 /usr/bin/time -v $BIN/circuit_amplitudes data/n30m14 16 c64 2 2>&1 | grep -E "^T2742|fidelity [0-9.]+, max|Maximum resident" | grep -v artensor | sed "s/^/[$1 r$4] /" >> $OUT/n30m14.txt
}
for r in 1 2 3 4 5; do
  if (( r % 2 )); then ORDER=("${CFGS[@]}"); else ORDER=($(printf '%s\n' "${CFGS[@]}" | tac)); fi
  for c in "${ORDER[@]}"; do IFS=: read l lim cap <<<"$c"; run_n30 $l $lim $cap $r; done
done
RC=("1G:$G" "4G:$((4*G))" "8G:$((8*G))" "64G:$((64*G))")
for r in 1 2 3 4 5; do
  for c in "${RC[@]}"; do IFS=: read l lim <<<"$c"
    for n in 8 122; do
      if [ $n = 8 ] && [ $l = 8G ]; then continue; fi
      echo "== repro26 n=$n $l round $r load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/repro26.txt
      T2742_LIMIT_BYTES=$lim /usr/bin/time -v $BIN/repro26 $n 2>&1 | grep -E "^T2742|Maximum resident" | sed "s/^/[n$n $l r$r] /" >> $OUT/repro26.txt
    done
  done
done
echo done > $OUT/DONE
