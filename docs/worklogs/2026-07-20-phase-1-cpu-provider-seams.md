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
