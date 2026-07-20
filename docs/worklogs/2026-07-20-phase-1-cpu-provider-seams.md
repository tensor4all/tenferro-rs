# Phase 1 CPU provider seams verification

This worklog records the predeclared non-inferiority campaign for Phase 1 of
the execution-engine provider architecture. The first complete campaign failed
the gate and is retained here unchanged as debugging evidence. Promotion is
blocked until a new immutable candidate passes a complete unchanged campaign.

## Immutable inputs and environment

- Baseline source: `85855e272b1495611deb601a9ee06f3546772c3c`, with only the
  frozen benchmark harness from `474ed072` and `e5a16a65` applied.
- Initial candidate: `9ffe1daa13a9d48920ec377bbf3bb65270cbd7fd`.
- Baseline binary SHA-256:
  `dbc2152862042ba0e2d8c27739e06f35467769c682db5e66f5d3b3ebf67b8`.
- Initial candidate binary SHA-256:
  `bf559cefce3694046d59bf5ad35de842b925dd99050fb102c05628e2c668d920`.
- Toolchain: `rustc 1.96.0 (ac68faa20 2026-05-25)`, LLVM 22.1.2;
  `cargo 1.96.0 (30a34c682 2026-05-25)`.
- Features: default `tenferro-ad`/`tenferro-cpu` features (`cpu-faer`).
- Host: AMD EPYC 7713P, 64 allowed CPUs (`0-63`), one socket and one NUMA
  node. CPU 0 was selected once and every benchmark process was launched by
  `taskset -c 0`.
- `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` were all set to one.
- Criterion configuration was the frozen 2-second warm-up, 5-second
  measurement, 100 samples, and 95% confidence interval.

The three complete pairs ran in the required order `A/B`, `B/A`, `A/B` from
2026-07-21 00:08 through 00:33 JST. Endpoint one-minute loads ranged from
3.81 to 7.05, below the invalidity threshold of 16 for 64 allowed CPUs. A
one-second process monitor covering each pair observed no process whose exact
name was `cargo` or `rustc`. Every pair contained all 28 cases. The `B/A`
Criterion intervals were algebraically inverted to retain candidate/baseline
orientation: `[l,u] -> [1/(1+u)-1, 1/(1+l)-1]`.

The raw named-baseline, `new`, and `change` estimates plus process-monitor
observations are preserved under
[`artifacts/2026-07-20-phase-1-cpu-provider-seams/initial-fail`](./artifacts/2026-07-20-phase-1-cpu-provider-seams/initial-fail/).

## Initial campaign result: FAIL

Intervals below are Criterion mean relative-change 95% intervals in percent;
the three following values are median point-estimate ratios retained only as
diagnostics. Classification applies the predeclared rule without adjustment.

| Case | Pair 1 A/B | Pair 2 B/A | Pair 3 A/B | Median ratios | Class |
|---|---:|---:|---:|---:|---|
| lazy add/1 | -2.22..+1.32 | +0.56..+3.69 | +6.07..+9.72 | -0.83/+1.71/+13.43 | INCONCLUSIVE |
| lazy add/64 | -1.99..+1.33 | +9.91..+12.75 | +8.81..+11.44 | -0.52/+13.79/+14.20 | **FAIL** |
| lazy add/8 | -1.64..+17.93 | -8.13..+10.05 | -7.20..+12.67 | +10.57/+2.69/+1.95 | INCONCLUSIVE |
| lazy dot_general/1 | -18.98..+19.90 | -20.52..+24.99 | -24.76..+21.25 | +2.69/+10.71/+1.11 | INCONCLUSIVE |
| lazy dot_general/2 | +5.69..+8.30 | +7.09..+9.36 | -8.74..-5.83 | +10.49/+10.90/-6.69 | **FAIL** |
| lazy neg/1 | -1.28..+3.65 | -0.74..+3.90 | -1.00..+3.52 | +1.24/+0.63/+0.67 | PASS |
| lazy neg/64 | -4.82..-1.85 | +2.15..+5.44 | +9.42..+12.10 | -3.71/+4.85/+14.48 | INCONCLUSIVE |
| lazy neg/8 | +3.30..+6.37 | +2.57..+6.09 | -0.21..+2.96 | +5.22/+4.15/+1.63 | INCONCLUSIVE |
| lazy reduce_sum/1 | -10.95..+11.45 | -10.10..+25.45 | -13.55..+16.61 | -1.49/+15.06/+0.42 | INCONCLUSIVE |
| lazy reduce_sum/64 | -2.20..+0.70 | +8.97..+11.97 | -0.37..+2.69 | -1.18/+14.01/+2.46 | INCONCLUSIVE |
| lazy reduce_sum/8 | +9.01..+11.59 | -1.02..+2.38 | +12.14..+14.64 | +12.58/+0.69/+16.21 | **FAIL** |
| lazy slice/1 | +7.81..+11.03 | -0.01..+2.90 | -1.62..+1.91 | +14.76/+2.30/-0.89 | INCONCLUSIVE |
| lazy slice/64 | +1.00..+4.54 | +5.69..+9.08 | +6.17..+9.12 | +4.07/+11.17/+10.00 | **FAIL** |
| lazy slice/8 | +0.14..+3.01 | +3.29..+6.30 | +13.82..+16.07 | +2.83/+4.38/+17.80 | INCONCLUSIVE |
| materialized add/1 | -1.02..+2.63 | +2.88..+5.73 | +0.14..+3.75 | +0.41/+3.78/+2.98 | INCONCLUSIVE |
| materialized add/64 | +1.87..+5.25 | -1.82..+1.20 | +1.21..+3.97 | +3.73/-0.40/+3.00 | INCONCLUSIVE |
| materialized add/8 | -14.86..+1.01 | +2.29..+20.28 | -7.06..+11.27 | -7.95/+12.99/+3.48 | INCONCLUSIVE |
| materialized dot_general/1 | -23.50..+20.79 | -32.36..+13.39 | -36.29..+22.28 | -1.99/-4.08/-2.17 | INCONCLUSIVE |
| materialized dot_general/2 | -2.35..+0.84 | -2.52..+0.04 | -4.26..-1.49 | -0.58/-2.40/-2.72 | PASS |
| materialized neg/1 | -4.17..-0.69 | +7.29..+11.05 | -3.02..+0.67 | -1.24/+11.63/-1.33 | INCONCLUSIVE |
| materialized neg/64 | +1.35..+4.46 | -3.04..-0.00 | -0.74..+2.10 | +2.12/-1.83/+0.83 | PASS |
| materialized neg/8 | +6.34..+9.52 | +6.05..+9.69 | +4.15..+7.48 | +11.89/+13.18/+5.19 | **FAIL** |
| materialized reduce_sum/1 | -31.56..+17.83 | -1.41..+10.52 | -14.69..+3.59 | -1.02/+0.67/+2.41 | INCONCLUSIVE |
| materialized reduce_sum/64 | +1.46..+4.61 | -2.75..+0.36 | -1.05..+2.23 | +2.05/-1.40/+1.89 | PASS |
| materialized reduce_sum/8 | +0.17..+1.65 | +4.45..+7.42 | +0.89..+3.91 | +0.17/+7.28/+2.12 | INCONCLUSIVE |
| materialized slice/1 | +1.61..+5.45 | +12.04..+14.73 | -4.33..-1.33 | +3.44/+15.89/-4.43 | INCONCLUSIVE |
| materialized slice/64 | +3.94..+7.56 | -1.47..+1.28 | +1.47..+5.36 | +7.71/-0.39/+3.16 | INCONCLUSIVE |
| materialized slice/8 | +2.94..+6.63 | -0.57..+2.77 | -2.44..+0.42 | +4.96/+1.82/+0.31 | INCONCLUSIVE |

Five cases failed, five passed, and eighteen were inconclusive. Because any
case failure makes the campaign fail, this candidate is not promotable.

## Allocation and correctness evidence

The fixed-main direct `CpuBackend` allocation proxy uses 100 counted calls
after 32 warm-up calls. Candidate counts and bytes did not exceed baseline:

| Case | Fixed-main allocations / bytes | Initial candidate allocations / bytes |
|---|---:|---:|
| add | 1,201 / 55,920 | 1,201 / 55,920 |
| reduce_sum | 5,005 / 112,592 | 5,005 / 112,592 |
| slice | 601 / 38,320 | 601 / 38,320 |
| dot_general | 3,802 / 112,440 | 1,002 / 40,640 |

This probe intentionally isolates the CPU backend boundary changed by Phase 1;
the Criterion matrix above measures the complete public AD eager path.

Correctness and source-contract checks before the campaign:

```console
cargo test -p tenferro-cpu --lib                         # 367 passed
cargo test -p tenferro-cpu --test integration backend_capability_contracts
cargo test -p tenferro-cpu --test provider_boundary_allocation_tests
```

## Root-cause investigation

The failed candidate remains frozen above; none of its raw artifacts were
removed or replaced.

The exact eager path is `EagerTensor::nary_op` ->
`exec_single_output_read` -> `EagerRuntime::exec_outputs_read` ->
`exec_standard_op_on_tensor_reads` -> `EagerBackend::with_backend_session` ->
`CpuBackend::run_backend_session_cached`. The only session-entry source delta
for non-contraction operations was replacing a copied `DotGeneralProvider`
enum with a cloned `CpuProviderBundle` `Arc`. A focused empty-session
microbenchmark rejected that as a material explanation: the candidate interval
was `7.0771..7.2245 us` and fixed main was `7.0388..7.2666 us`. The intervals
fully overlap and cannot explain a repeated regression beyond five percent.
No clone-removal or performance code change was made.

### Same-binary A/A control

The five initially failing cases plus the initially passing `lazy neg_f64/1`
control were run candidate-against-itself, one case at a time, for three pairs.
The original 2 s / 5 s / 100-sample settings, CPU 0 pinning, thread environment,
and exact-name Cargo/rustc monitor were retained. Invalid attempts were
discarded. Raw A, B, and change estimates for each accepted pair are under
[`aa-focused`](./artifacts/2026-07-20-phase-1-cpu-provider-seams/aa-focused/).

The single classifier script produced:

| Case | Pair 1 | Pair 2 | Pair 3 | Class |
|---|---:|---:|---:|---|
| control_lazy_neg_1 | -5.41..-0.59 (-3.00) | +1.09..+6.62 (+3.88) | +0.98..+5.58 (+3.27) | INCONCLUSIVE |
| lazy_add_64 | -4.44..-0.17 (-2.41) | -2.58..+1.97 (-0.22) | -1.02..+3.54 (+1.17) | PASS |
| lazy_dot_2 | -0.53..+3.41 (+1.48) | -4.29..-0.42 (-2.45) | -5.61..-1.73 (-3.61) | PASS |
| lazy_reduce_8 | +0.40..+4.86 (+2.66) | -2.08..+2.04 (+0.02) | -1.74..+1.99 (+0.27) | PASS |
| lazy_slice_64 | -0.34..+2.72 (+1.20) | -2.57..-0.01 (-1.27) | +1.27..+4.59 (+2.89) | PASS |
| materialized_neg_8 | -3.15..+1.37 (-0.91) | +2.10..+7.24 (+4.57) | -2.06..+1.97 (+0.01) | INCONCLUSIVE |

Summary: `4 PASS / 0 FAIL / 2 INCONCLUSIVE`; campaign=`INCONCLUSIVE`.
The A/A control did not itself produce a false `FAIL`, but it demonstrated
pair-to-pair shifts large enough to make full-suite temporal separation a
plausible noise amplifier.

### Case-interleaved reproduction and dependency-lock confound

The same six cases were next run fixed-main versus candidate in case-local
`A/B`, `B/A`, `A/B` order. This removed the minutes-long separation between
corresponding cases. It produced `3 PASS / 0 FAIL / 3 INCONCLUSIVE`, so none of
the five initial `FAIL` classifications reproduced under the predeclared
two-of-three rule. Those raw estimates remain under
[`focused-main-candidate`](./artifacts/2026-07-20-phase-1-cpu-provider-seams/focused-main-candidate/).

Inspection then found a stronger protocol confound: the fixed-main and
candidate binaries had been built from different dependency resolutions. The
original fixed-main build used newer `clap`, `enumset`, `serde_json`, and
`syn 3` versions even though the Rust toolchain, profile, and declared features
matched. `85855e27` did not provide a tracked lock that prevented this drift.

The candidate lock was therefore applied byte-for-byte to the fixed-main
worktree. Both benches were rebuilt with `cargo --locked`:

- common `Cargo.lock` SHA-256:
  `09d8dd85de17dbb27425e5c4866ecf0963fe8d1b3c6b517b4bace12c44741cd6`;
- locked fixed-main binary SHA-256:
  `c8f3cd59bab3b29185ed4bdbc4ed57a22d0741144e717fb8b7eaeea7aff68636`;
- locked candidate binary SHA-256:
  `bf559cefce3694046d59bf5ad35de842b925dd99050fb102c05628e2c668d920`.

The locked case-interleaved rerun is preserved under
[`focused-locked-main-candidate`](./artifacts/2026-07-20-phase-1-cpu-provider-seams/focused-locked-main-candidate/).
The classifier output was:

| Case | Pair 1 | Pair 2 | Pair 3 | Class |
|---|---:|---:|---:|---|
| control_lazy_neg_1 | -2.36..+1.96 (-0.24) | -3.00..+1.20 (-0.92) | +0.13..+4.18 (+2.16) | PASS |
| lazy_add_64 | +0.31..+4.59 (+2.50) | -0.68..+3.93 (+1.67) | +0.42..+5.28 (+2.68) | INCONCLUSIVE |
| lazy_dot_2 | -4.83..-0.99 (-2.92) | -3.23..+1.04 (-1.22) | -3.97..+0.24 (-1.92) | PASS |
| lazy_reduce_8 | -1.03..+3.95 (+1.30) | -3.00..+1.79 (-0.58) | -4.58..+0.28 (-2.11) | PASS |
| lazy_slice_64 | -2.77..-0.37 (-1.59) | +4.97..+7.62 (+6.29) | -8.53..-5.49 (-7.02) | INCONCLUSIVE |
| materialized_neg_8 | +3.24..+7.73 (+5.44) | -0.30..+4.09 (+1.93) | +1.05..+5.34 (+3.22) | INCONCLUSIVE |

Summary: `3 PASS / 0 FAIL / 3 INCONCLUSIVE`; campaign=`INCONCLUSIVE`.
Thus the initial `FAIL` is not evidence for a provider-dispatch code regression:
it is confounded by unlocked dependency resolution and full-suite time drift,
and it does not reproduce after controlling both. The locked result still does
not satisfy the phase exit because three cases remain inconclusive.

### Amended formal campaign

Before the final 28-case rerun, the protocol was amended in the Phase 1 design:

- byte-identical lock input, `--locked` builds, and lock/binary SHA recording;
- case-local `A/B`, `B/A`, `A/B` execution and immediate per-pair raw copying;
- each target pair bracketed by candidate/candidate `lazy neg_f64/1` runs;
- an A/A interval wholly outside `-5%..+5%` invalidates the enclosed pair;
- invalid attempts contribute no estimates and are retried unchanged.

Classification is performed only by
[`scripts/classify_criterion_noninferiority.py`](../../scripts/classify_criterion_noninferiority.py).
Its boundary tests fix `PASS` to all three upper endpoints at most +5%, `FAIL`
to at least two lower endpoints strictly above +5%, and every other result to
`INCONCLUSIVE`.

## Formal campaign 1: INCONCLUSIVE

Formal campaign 1 used the manifest-enforcing runner and classifier added in
`ee58e243`. Its immutable inputs were:

- baseline source revision:
  `85855e272b1495611deb601a9ee06f3546772c3c`;
- candidate measurement revision: `72beb164`;
- common `Cargo.lock` SHA-256:
  `09d8dd85de17dbb27425e5c4866ecf0963fe8d1b3c6b517b4bace12c44741cd6`;
- baseline binary SHA-256:
  `c8f3cd59bab3b29185ed4bdbc4ed57a22d0741144e717fb8b7eaeea7aff68636`;
- candidate binary SHA-256:
  `bf559cefce3694046d59bf5ad35de842b925dd99050fb102c05628e2c668d920`;
- CPU 0 of the process-allowed `0-63`, with all five declared thread-count
  environment variables set to one.

The accepted inventory is exactly 28 cases, 84 pairs, and 336 individually
monitored runs. Re-validating `campaign.json` and every pair `validity.json`
found zero monitor violations, zero non-zero exits, zero affinity mismatches,
and a maximum accepted endpoint normalized load of `0.1636962890625`. Twenty
attempts were rejected before manifest adoption: 14 for exact-name
Cargo/rustc overlap and six for an A/A sentinel interval wholly outside the
drift band. No rejected estimate was reused.

The complete evidence is under
[`formal-manifest`](./artifacts/2026-07-20-phase-1-cpu-provider-seams/formal-manifest/)
and occupies 5.0 MiB as reported by `du -sh`. An independent rerun of the
manifest-aware classifier produced the following output without edits:

```console
$ python3 scripts/classify_criterion_noninferiority.py docs/worklogs/artifacts/2026-07-20-phase-1-cpu-provider-seams/formal-manifest
| Case | Pair 1 | Pair 2 | Pair 3 | Class |
|---|---:|---:|---:|---|
| lazy_add_1 | -0.54..+3.65 (+1.51) | -1.83..+2.12 (+0.05) | +0.85..+5.33 (+3.02) | INCONCLUSIVE |
| lazy_add_64 | -0.34..+4.21 (+1.97) | -0.17..+4.22 (+1.98) | -4.09..+0.17 (-1.83) | PASS |
| lazy_add_8 | +0.64..+4.93 (+2.67) | -1.58..+2.52 (+0.45) | -3.28..+0.87 (-1.29) | PASS |
| lazy_dot_1 | -3.31..+0.86 (-1.21) | -6.51..-2.83 (-4.72) | -4.56..-0.92 (-2.72) | PASS |
| lazy_dot_2 | -5.38..-1.54 (-3.52) | -2.56..+0.68 (-0.95) | -3.35..+0.29 (-1.61) | PASS |
| lazy_neg_1 | -2.32..+1.67 (-0.38) | -1.88..+1.72 (-0.12) | -6.34..-2.28 (-4.37) | PASS |
| lazy_neg_64 | +1.37..+6.31 (+3.91) | -0.94..+3.09 (+1.05) | -2.97..+1.37 (-0.88) | INCONCLUSIVE |
| lazy_neg_8 | -1.42..+2.68 (+0.65) | -3.14..+0.76 (-1.20) | -2.02..+2.17 (+0.08) | PASS |
| lazy_reduce_1 | +0.44..+4.55 (+2.57) | -1.43..+2.54 (+0.57) | -0.26..+3.49 (+1.57) | PASS |
| lazy_reduce_64 | -5.37..-1.23 (-3.32) | -3.17..+0.48 (-1.36) | -6.96..-2.78 (-4.90) | PASS |
| lazy_reduce_8 | -1.20..+2.93 (+0.80) | -3.31..+0.53 (-1.48) | -1.81..+1.98 (+0.08) | PASS |
| lazy_slice_1 | -2.96..-0.40 (-1.78) | -1.55..+0.88 (-0.21) | -3.44..-0.47 (-2.06) | PASS |
| lazy_slice_64 | -4.62..-2.58 (-3.64) | -5.55..-2.94 (-4.33) | -0.19..+2.72 (+1.22) | PASS |
| lazy_slice_8 | -0.41..+2.55 (+1.11) | -0.67..+2.96 (+1.20) | -4.45..-3.13 (-3.77) | PASS |
| materialized_add_1 | -4.35..+0.41 (-2.04) | -3.50..+0.44 (-1.45) | -0.28..+3.87 (+1.81) | PASS |
| materialized_add_64 | -2.44..+1.44 (-0.50) | -0.73..+2.68 (+0.92) | -0.83..+3.71 (+1.38) | PASS |
| materialized_add_8 | -1.43..+2.49 (+0.52) | -3.66..+0.70 (-1.50) | -2.36..+2.22 (-0.13) | PASS |
| materialized_dot_1 | -0.75..+3.27 (+1.19) | -4.56..-0.28 (-2.33) | -6.34..-2.60 (-4.49) | PASS |
| materialized_dot_2 | -2.55..+1.10 (-0.76) | -6.94..-3.30 (-5.14) | -1.73..+3.17 (+0.76) | PASS |
| materialized_neg_1 | -1.22..+2.38 (+0.60) | -2.57..+0.92 (-0.78) | +1.99..+6.92 (+4.54) | INCONCLUSIVE |
| materialized_neg_64 | -0.93..+3.45 (+1.25) | -1.87..+2.28 (+0.25) | -1.16..+2.40 (+0.59) | PASS |
| materialized_neg_8 | +0.94..+5.56 (+3.25) | -2.35..+1.86 (-0.25) | -6.27..-2.33 (-4.31) | INCONCLUSIVE |
| materialized_reduce_1 | -2.27..+1.91 (-0.09) | +1.13..+5.43 (+3.25) | +0.27..+4.34 (+2.36) | INCONCLUSIVE |
| materialized_reduce_64 | -3.03..+0.82 (-1.09) | -4.24..-0.42 (-2.28) | -1.46..+2.03 (+0.35) | PASS |
| materialized_reduce_8 | +0.09..+5.26 (+2.57) | +2.18..+6.27 (+4.23) | -4.45..-0.13 (-2.16) | INCONCLUSIVE |
| materialized_slice_1 | -2.02..+0.79 (-0.63) | -2.51..-0.41 (-1.46) | -1.05..+1.25 (+0.01) | PASS |
| materialized_slice_64 | -1.01..+0.88 (-0.10) | -1.31..+0.96 (-0.16) | -1.29..+2.03 (+0.34) | PASS |
| materialized_slice_8 | -0.37..+2.58 (+1.05) | -4.86..-2.17 (-3.47) | -2.11..+1.06 (-0.52) | PASS |

Summary: 22 PASS / 0 FAIL / 6 INCONCLUSIVE; campaign=INCONCLUSIVE
```

The six inconclusive rows above retain all three pair intervals. Because no
case failed, this campaign provides no evidence of a greater-than-five-percent
provider regression, but it does not satisfy the all-PASS phase exit. The next
eligible experiment is a new complete 28-case, three-pair campaign with the
same immutable inputs and protocol; no pair from formal campaign 1 may be
selected or reused.

## Formal campaign 2: INCONCLUSIVE

Formal campaign 2 was a new complete run under the same immutable baseline
and candidate binaries, common lock, CPU 0, single-thread environment, case
matrix, and validity protocol. It used the new empty evidence root
[`formal-manifest-2`](./artifacts/2026-07-20-phase-1-cpu-provider-seams/formal-manifest-2/)
and reused no accepted or rejected pair from formal campaign 1.

The accepted inventory is again exactly 28 cases, 84 pairs, and 336 monitored
runs. Independent manifest re-validation found zero monitor violations, zero
non-zero exits, zero affinity mismatches, and a maximum accepted endpoint
normalized load of `0.13843536376953125`. Thirty-one attempts were rejected
before manifest adoption: 23 for exact-name Cargo/rustc overlap and eight for
an A/A sentinel drift breach. The tracked evidence occupies 3.8 MiB; the local
root occupies 5.6 MiB including gitignored Criterion console logs. The final
`campaign.json` SHA-256 is
`ed99d999bf74e77d9ffef698a25583f00f3672da7a722adfd01f1bc3bc29bb0c`.

The independent manifest-aware classifier rerun produced the following output
without edits:

```console
$ python3 scripts/classify_criterion_noninferiority.py docs/worklogs/artifacts/2026-07-20-phase-1-cpu-provider-seams/formal-manifest-2
| Case | Pair 1 | Pair 2 | Pair 3 | Class |
|---|---:|---:|---:|---|
| lazy_add_1 | +0.72..+4.93 (+2.89) | -2.14..+1.75 (-0.19) | -0.35..+3.67 (+1.62) | PASS |
| lazy_add_64 | -3.36..+0.34 (-1.47) | -1.47..+2.62 (+0.51) | -4.42..-0.21 (-2.36) | PASS |
| lazy_add_8 | -2.52..+2.30 (-0.18) | -3.77..+0.38 (-1.67) | -2.93..+1.50 (-0.75) | PASS |
| lazy_dot_1 | -2.11..+1.85 (-0.15) | -4.00..-0.51 (-2.30) | -6.49..-3.13 (-4.83) | PASS |
| lazy_dot_2 | -2.41..+1.45 (-0.46) | -1.49..+2.57 (+0.37) | -1.26..+2.37 (+0.57) | PASS |
| lazy_neg_1 | -1.09..+2.90 (+0.94) | -0.20..+4.06 (+2.05) | -0.03..+3.96 (+1.89) | PASS |
| lazy_neg_64 | -3.39..+0.25 (-1.58) | -3.65..-0.05 (-1.77) | -1.09..+2.60 (+0.74) | PASS |
| lazy_neg_8 | -3.02..+0.64 (-1.15) | -1.15..+3.80 (+1.31) | +0.46..+5.03 (+2.55) | INCONCLUSIVE |
| lazy_reduce_1 | -1.84..+1.89 (+0.04) | -1.71..+2.39 (+0.37) | -0.33..+4.23 (+1.79) | PASS |
| lazy_reduce_64 | -2.43..+1.84 (-0.38) | -1.51..+2.26 (+0.36) | -1.29..+3.39 (+1.13) | PASS |
| lazy_reduce_8 | -0.68..+3.16 (+1.33) | -1.15..+2.64 (+0.76) | -3.60..+0.68 (-1.43) | PASS |
| lazy_slice_1 | -2.97..-1.19 (-2.07) | +0.85..+3.67 (+2.24) | -1.40..+0.68 (-0.38) | PASS |
| lazy_slice_64 | -4.27..-2.31 (-3.29) | -2.18..-0.45 (-1.29) | +1.97..+4.41 (+3.19) | PASS |
| lazy_slice_8 | -1.70..+0.54 (-0.60) | -1.33..+1.04 (-0.15) | -4.41..-1.89 (-3.10) | PASS |
| materialized_add_1 | -0.35..+4.32 (+1.91) | -6.33..-2.10 (-4.20) | -0.20..+4.19 (+2.03) | PASS |
| materialized_add_64 | -0.49..+3.72 (+1.59) | -4.08..+0.15 (-1.93) | -1.59..+2.03 (+0.17) | PASS |
| materialized_add_8 | -0.15..+3.93 (+1.87) | -2.23..+1.64 (-0.27) | +4.24..+8.61 (+6.51) | INCONCLUSIVE |
| materialized_dot_1 | -6.10..-2.54 (-4.31) | -1.81..+2.11 (+0.21) | -4.76..-0.56 (-2.76) | PASS |
| materialized_dot_2 | -3.82..-0.08 (-2.00) | -4.54..-0.49 (-2.53) | -4.50..-0.92 (-2.67) | PASS |
| materialized_neg_1 | -1.13..+3.24 (+0.95) | -1.73..+2.66 (+0.54) | -1.64..+2.59 (+0.40) | PASS |
| materialized_neg_64 | +0.03..+4.23 (+2.05) | -1.50..+2.22 (+0.43) | -1.98..+2.49 (+0.33) | PASS |
| materialized_neg_8 | +0.78..+5.08 (+2.99) | -2.08..+1.89 (-0.09) | -3.19..+1.07 (-1.06) | INCONCLUSIVE |
| materialized_reduce_1 | -0.42..+3.96 (+1.77) | -2.93..+1.66 (-0.65) | -5.06..-0.46 (-2.79) | PASS |
| materialized_reduce_64 | +3.16..+7.75 (+5.45) | -5.10..-0.54 (-2.67) | +2.34..+6.78 (+4.52) | INCONCLUSIVE |
| materialized_reduce_8 | -0.97..+4.17 (+1.61) | -0.58..+4.32 (+1.82) | -4.08..+0.28 (-1.87) | PASS |
| materialized_slice_1 | -0.98..+2.36 (+0.73) | -4.80..-2.04 (-3.46) | -3.16..-0.16 (-1.67) | PASS |
| materialized_slice_64 | -3.65..-0.69 (-2.22) | -0.41..+2.68 (+1.14) | -3.14..-0.16 (-1.69) | PASS |
| materialized_slice_8 | -0.50..+2.66 (+1.07) | -0.07..+3.05 (+1.56) | -2.45..+0.59 (-0.97) | PASS |

Summary: 24 PASS / 0 FAIL / 4 INCONCLUSIVE; campaign=INCONCLUSIVE
```

The four inconclusive rows retain all three intervals above. Formal campaign 2
again has no `FAIL`, but it does not satisfy the all-PASS phase exit. One final
new complete campaign may be run with unchanged inputs and protocol. If that
campaign is not `PASS`, no additional rerun is authorized: promotion remains
blocked and the complete evidence stays on this branch for review.

## Formal campaign 3: INCONCLUSIVE, final rerun

Formal campaign 3 was the one authorized final rerun. It used the same
immutable baseline and candidate revisions, binary hashes, common lock, CPU 0,
single-thread environment, 28-case matrix, and manifest-enforced protocol. It
used the new empty evidence root
[`formal-manifest-3`](./artifacts/2026-07-20-phase-1-cpu-provider-seams/formal-manifest-3/)
and reused no pair from either earlier campaign.

The accepted inventory is exactly 28 cases, 84 pairs, and 336 monitored runs.
Independent manifest re-validation found zero invalid accepted pairs, zero
monitor violations, zero non-zero or incomplete runs, zero affinity
mismatches, and a maximum accepted endpoint normalized load of
`0.1545257568359375`. Thirty-five attempts were rejected before manifest
adoption: 29 for exact-name Cargo/rustc overlap and six for an A/A sentinel
drift breach. No rejected estimate was reused. Evidence excluding gitignored
console logs occupies 1,045,877 bytes; the complete local root occupies
1,339,857 bytes. The final `campaign.json` SHA-256 is
`81d0d24a86d5e8b2fea0c9a6cdaaf7939fd293d3ea2d3da656039bf2ebfa4138`.

An independent manifest-aware classifier rerun produced the following output
without edits:

```console
$ python3 scripts/classify_criterion_noninferiority.py docs/worklogs/artifacts/2026-07-20-phase-1-cpu-provider-seams/formal-manifest-3
| Case | Pair 1 | Pair 2 | Pair 3 | Class |
|---|---:|---:|---:|---|
| lazy_add_1 | -1.63..+2.68 (+0.51) | -0.48..+4.03 (+1.75) | -4.35..-0.22 (-2.29) | PASS |
| lazy_add_64 | -2.03..+2.01 (-0.08) | +0.05..+4.66 (+2.42) | -3.73..+0.18 (-1.70) | PASS |
| lazy_add_8 | -1.12..+3.38 (+1.01) | +6.65..+11.38 (+9.04) | -2.03..+2.20 (+0.09) | INCONCLUSIVE |
| lazy_dot_1 | -2.63..+1.36 (-0.57) | -7.59..-3.62 (-5.61) | -4.91..-1.03 (-3.01) | PASS |
| lazy_dot_2 | -4.50..-0.47 (-2.38) | -4.13..-0.19 (-2.21) | -5.42..-1.73 (-3.64) | PASS |
| lazy_neg_1 | +0.10..+4.68 (+2.43) | +1.84..+6.15 (+3.88) | +0.30..+4.83 (+2.55) | INCONCLUSIVE |
| lazy_neg_64 | -3.83..+0.48 (-1.65) | -2.51..+2.11 (-0.30) | -0.10..+4.60 (+2.09) | PASS |
| lazy_neg_8 | -3.95..+0.54 (-1.68) | -1.73..+2.43 (+0.36) | -0.52..+3.73 (+1.64) | PASS |
| lazy_reduce_1 | -2.75..+1.31 (-0.76) | -0.25..+3.96 (+1.82) | -1.38..+3.09 (+0.82) | PASS |
| lazy_reduce_64 | -0.46..+3.82 (+1.70) | -0.87..+3.43 (+1.23) | -1.02..+3.42 (+1.19) | PASS |
| lazy_reduce_8 | -3.75..+0.78 (-1.59) | -2.92..+1.25 (-0.80) | -4.54..-0.34 (-2.58) | PASS |
| lazy_slice_1 | -1.22..+1.67 (+0.20) | -2.01..+1.01 (-0.45) | -1.75..+1.15 (-0.34) | PASS |
| lazy_slice_64 | -3.87..-0.75 (-2.31) | +1.44..+4.66 (+3.02) | +3.45..+6.49 (+4.92) | INCONCLUSIVE |
| lazy_slice_8 | -6.06..-3.06 (-4.60) | -2.16..+0.79 (-0.78) | -1.72..+1.00 (-0.41) | PASS |
| materialized_add_1 | -5.60..-1.58 (-3.63) | +1.70..+5.69 (+3.67) | -2.01..+2.36 (+0.26) | INCONCLUSIVE |
| materialized_add_64 | -3.28..+0.95 (-1.24) | -2.69..+1.63 (-0.58) | +2.74..+7.71 (+5.30) | INCONCLUSIVE |
| materialized_add_8 | +2.13..+6.70 (+4.37) | -2.81..+1.23 (-0.85) | +2.57..+6.98 (+4.68) | INCONCLUSIVE |
| materialized_dot_1 | -6.79..-2.78 (-4.73) | -6.87..-2.92 (-4.86) | -3.58..+0.64 (-1.43) | PASS |
| materialized_dot_2 | -6.61..-2.66 (-4.73) | -7.03..-3.18 (-5.10) | +0.12..+4.46 (+2.14) | PASS |
| materialized_neg_1 | +0.69..+5.12 (+3.05) | -2.33..+2.07 (-0.05) | -0.76..+3.66 (+1.46) | INCONCLUSIVE |
| materialized_neg_64 | -2.48..+1.58 (-0.47) | +1.93..+6.68 (+4.35) | -0.58..+3.72 (+1.50) | INCONCLUSIVE |
| materialized_neg_8 | -4.86..-0.66 (-2.75) | -0.01..+4.55 (+2.20) | -3.03..+1.18 (-0.91) | PASS |
| materialized_reduce_1 | -2.42..+1.90 (-0.35) | -1.08..+3.41 (+1.14) | -2.47..+1.98 (-0.32) | PASS |
| materialized_reduce_64 | -2.82..+1.60 (-0.63) | -1.60..+2.75 (+0.64) | -3.53..+0.42 (-1.51) | PASS |
| materialized_reduce_8 | -5.98..-1.91 (-3.87) | -2.61..+1.91 (-0.33) | -4.56..-0.36 (-2.51) | PASS |
| materialized_slice_1 | -0.54..+2.38 (+0.87) | -3.83..-1.03 (-2.48) | -2.64..+0.47 (-1.05) | PASS |
| materialized_slice_64 | -3.15..-0.23 (-1.76) | -1.02..+2.25 (+0.65) | -2.62..+0.30 (-1.23) | PASS |
| materialized_slice_8 | -0.24..+2.98 (+1.34) | -1.60..+1.26 (-0.18) | -0.45..+2.62 (+1.02) | PASS |

Summary: 20 PASS / 0 FAIL / 8 INCONCLUSIVE; campaign=INCONCLUSIVE
```

The eight inconclusive rows retain every pair interval above:
`lazy_add_8`, `lazy_neg_1`, `lazy_slice_64`, `materialized_add_1`,
`materialized_add_64`, `materialized_add_8`, `materialized_neg_1`, and
`materialized_neg_64`. The campaign contains no `FAIL`, so it provides no
evidence of a reproducible greater-than-five-percent regression, but the
all-PASS non-inferiority exit remains unproven. This was the final authorized
rerun. No further campaign or selective pair replacement is allowed under this
decision; Phase 1 promotion remains blocked and the branch is preserved as an
`INCONCLUSIVE` review handoff.
