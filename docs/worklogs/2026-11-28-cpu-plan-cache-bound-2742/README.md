# CPU plan-cache bound and release threshold (#2742)

## Decisions

- **Bound: bytes, not a count. Keep `contraction_limit_bytes` at 1 GiB and the
  1024-slot cap as they are.** On n30m14 the byte bound is the only one that
  fires (35 entries held against 1024 slots, 70 byte evictions). A count does
  not bound memory here because plans differ by five orders of magnitude (a few
  KiB for an ordinary plan, 0.4 to 7 GiB for a plan with large free groups):
  128 entries with no byte limit held 15.1 GiB, 32 entries 4.5 GiB. This is the
  opposite of the GPU answer in #2706, where a plan is uniform and the eviction
  count is what costs.
- **No CPU equivalent of the GPU release threshold.** `reset_buffer_pool()` does
  not touch plans, and the memory this issue is about is plan tables, released
  by `plan_cache().clear()`. Whether the CPU buffer pool's own bound costs time
  (32 s of an 88 s n30m14 run is "free") was not measured; it is a separate
  question from this cache.
- **The default does not move, and no larger default is justified by this
  data.** Before tensor4all/tcontract-rs#222 no bound below the whole working
  set (22 GiB) buys back the warm loop. With #222 patched in, 1 GiB and 24 GiB
  are the same run, so the bound stops mattering. The cost is removed by the
  tcontract bump, not by tuning the bound.

## Verification conclusions and constraints

Host `primerose`: AMD EPYC 7713P (64 cores, L3 256 MiB in 8 slices, one NUMA
node), 995 GiB RAM, rustc 1.100.0. Slurm job on `cpu-prime`, 16 CPUs, no GPU
requested (the job saw no GPU). Host load average 15 to 20 from other users
throughout, per-run value in `raw/*.txt`. tenferro-rs `61e0a1083` (`origin/main`),
tcontract 0.13.0, tcontract-kernel 0.3.1, cpueinsum 0.13.1. Workload:
`Fanerst/sampled-einsum@c482b79` n30m14, complex64, 16 threads, 1000
bitstrings, built against the checkout by path dependency (the patch is in
`harness/`); fidelity 0.999999999778 in every run, the same as the issue's.
Runs: one process per row, a cold run then a warm run in that process; rounds
interleave the settings and reverse the order on even rounds. Medians over
**n = 5** processes (n = 3 for the #222 rows). Per-process ranges are in the raw
files. Time is seconds per run; high-water and held are allocator bytes over the
process's baseline; RSS is the process maximum.

### n30m14, `main` (pre-#222)

| setting | cold s | warm s | cold high-water GiB | held after GiB | entries | cache-reported GiB | max RSS GiB |
|---|---|---|---|---|---|---|---|
| 1 GiB (default) | 92.4 | 92.9 | 25.25 | 1.42 | 35 | 0.81 | 24.9 |
| 4 GiB | 92.6 | 93.1 | 26.00 | 3.67 | 32 | 3.06 | 25.8 |
| 8 GiB | 91.8 | 92.5 | 31.25 | 8.17 | 34 | 7.56 | 30.9 |
| 24 GiB | 89.4 | 68.7 | 43.60 | 22.49 | 102 | 21.89 | 46.0 |
| count 128, no byte limit | 90.6 | 81.8 | 39.83 | 15.68 | 72 | 15.07 | 42.2 |
| count 32, no byte limit | 93.1 | 93.0 | 32.27 | 5.12 | 31 | 4.51 | 31.9 |

Against 24 GiB (the unbounded behaviour) the 1 GiB default costs **+3 % cold and
+35 % warm** here (Fanerst's host: +13 % and +35 %). On this host the 4 GiB and
8 GiB bounds do **not** restore the old cold time, unlike on his host: the
whole working set has to fit before the warm run gains anything, and a bound
below it thrashes the same as 1 GiB.

### 26-qubit reproducer from the issue (384 MiB per distinct position), `main`

Seconds per round (two rounds over the position set), n = 5 medians.

| positions | bound | round 1 s | round 2 s | held after GiB | entries | max RSS GiB |
|---|---|---|---|---|---|---|
| 8 | 1 GiB | 5.29 | 5.21 | 0.78 | 2 | 2.3 |
| 8 | 4 GiB | 4.86 | 1.98 | 3.03 | 8 | 4.6 |
| 122 | 1 GiB | 79.2 | 79.1 | 1.30 | 2 | 2.4 |
| 122 | 4 GiB | 77.6 | 76.7 | 4.30 | 10 | 5.4 |
| 122 | 8 GiB | 75.5 | 77.0 | 8.42 | 21 | 9.5 |
| 122 | 64 GiB (all fit) | 70.8 | 38.4 | 41.80 | 111 | 42.8 |

The reproducer's 8 positions hold 3.0 GiB, so 1 GiB rebuilds every call (round 2
equals round 1) and 4 GiB caches them (2.6x). The 122-position set is 41 GiB and
only the full fit helps. The 1 GiB row holds 1.30 GiB after the run, of which
0.52 GiB remains after `clear()`.

### With tcontract-rs `41dcd12f` (#222) patched in, n = 3

| workload | 1 GiB | 24 GiB |
|---|---|---|
| n30m14 cold / warm s | 63.9 / 63.9 | 66.6 / 65.4 |
| n30m14 high-water / held GiB | 24.50 / 0.61 | 24.50 / 0.61 |
| n30m14 cache-reported GiB | 0.002 | 0.002 |
| n30m14 max RSS GiB | 24.1 | 24.1 |
| 26 qubits, 8 positions, round 1 / 2 s | 1.95 / 1.85 | 1.89 / 1.84 |
| 26 qubits, 122 positions, round 1 / 2 s | 30.3 / 28.1 | 30.2 / 28.9 |

The bound no longer fires (0 evictions, retained 0.002 GiB). This is a
patch-in measurement in a scratch build; `main` does not contain #222.

### Limits of this evidence

- One host, shared; load varied. Ranges across processes are 2 to 6 s on a 90 s
  run in the 1 to 8 GiB rows.
- The `evictions` counter did not move for the count-only rows while `misses`
  grew (143 to 282 for count 32), so it appears not to count capacity
  evictions; that was observed, not investigated.
- Counts of 128 held 72 entries, not 122, with 117 first-run misses; I did not
  determine why the entry count and the shape count differ.
- The buffer-pool analogue of the release threshold was not measured.
- Raw output: `raw/`. Harness and patch: `harness/`.
