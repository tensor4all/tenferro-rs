#!/bin/bash
OUT=~/scratch/t2742-proto/runs; BIN=~/scratch/t2742-proto/target222/release/examples
cd ~/scratch/t2742-proto/build; G=$((1<<30))
for r in 1 2 3; do
 for c in "1G:$G" "24G:$((24*G))"; do IFS=: read l lim <<<"$c"
  echo "== p222 n30m14 $l round $r load $(cut -d' ' -f1-3 /proc/loadavg)" >> $OUT/p222.txt
  T2742_LIMIT_BYTES=$lim /usr/bin/time -v $BIN/circuit_amplitudes data/n30m14 16 c64 2 2>&1 | grep -E "^T2742|fidelity [0-9.]+, max|Maximum resident" | grep -v artensor | sed "s/^/[n30 $l r$r] /" >> $OUT/p222.txt
  for n in 8 122; do
   T2742_LIMIT_BYTES=$lim /usr/bin/time -v $BIN/repro26 $n 2>&1 | grep -E "^T2742|Maximum resident" | sed "s/^/[n$n $l r$r] /" >> $OUT/p222.txt
  done
 done
done
echo done > $OUT/DONE222
