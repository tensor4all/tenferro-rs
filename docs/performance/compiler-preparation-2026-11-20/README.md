# Compiler preparation measurements

The combined layout-alias and conjugation-metadata repair passes the original
local experiment's primary thresholds and declared load/noise checks. Both
unchanged tensor4all-rs AD tests became about seven times faster. These are
unpinned macOS observations: pinned-core idle evidence required by the full
upstream host protocol remains unavailable, so this is draft evidence rather
than a claim that the full host-validity gate is satisfied.

| Case | Baseline median | Candidate median | Median paired candidate/baseline |
|---|---:|---:|---:|
| Layout simplification, 4096 operations | 18.884 ms | 0.227 ms | 0.0120 |
| Unchanged GSE AD test, whole process | 2.409 s | 0.342 s | 0.1418 |
| Unchanged linsolve coefficient AD test, whole process | 1.288 s | 0.180 s | 0.1396 |
| Conjugation metadata diagnostic, 4096 operations | 5.961 ms | 0.166 ms | 0.0276 |

The source, complete case lists, every raw sample, paired ratios, bootstrap
intervals, and before/after load observations are retained in these files:

- [First layout-only experiment](first-inconclusive.json): **INCONCLUSIVE**.
  It failed the noise gate and did not meet the end-to-end thresholds. Its
  isolated layout result did not establish acceptance.
- [Combined candidate](final-local.json): all six original cases retained,
  with the original thresholds. The experiment's local checks pass; the stronger
  upstream pinned-core host check remains unverified.
- [Conjugation diagnostic](conjugation-local.json): all four sizes retained.
  This is secondary evidence and does not replace the original acceptance cases.

## Method and reproduction

The tested downstream baseline is tensor4all-rs
`9476f862f11a6a7f9f4989bcaeec3fecc751e342`, with tenferro
`32a4e98c12f9dc36ffa85b5273b9247b44936959`. The final downstream candidate
uses compiler-only backport `71b50235b21d9aeb79f971339d088b6213fa41f4` and
Hataori pin-only backport `1205da509ac0e8d7677bc20260556aedc2879a12`.
All five compiler/test files match upstream implementation
`f70ff7a7623e78a7e7ba2080f50d37e1014203e6` byte for byte; the backport retains
the downstream dependency versions and public surface.

The host was an Apple M3 Pro, Mac15,6, 11 logical CPUs, macOS 27.0.1.
The final metadata probes used Rust 1.96.0 on both sides, one shared standalone
manifest/lockfile, and the checked-in probe modules. The end-to-end tests used
Homebrew Rust 1.98.1 on both sides, identical test inventories and the same
five-crate feature selection, including `simplett-bridge`. No cross-experiment
comparison of absolute probe times is claimed. The inherited
`compiler_patch_sha256` field identifies the initial layout patch; final sources
are identified by `candidate_source_sha256` and the candidate commits.

The provider was Apple Accelerate. `RAYON_NUM_THREADS`, `BLAS_NUM_THREADS`,
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and
`VECLIB_MAXIMUM_THREADS` were all 1; the actual downstream owning backend also
reported one thread before measurement. Build jobs were capped at two and no
build ran during timing. macOS CPU affinity was unavailable. The declared local
validity checks were load at most 11 before/after each invocation and maximum
arm median divided by minimum arm median at most 1.5 for every case.

Each compiler case has one discarded warmup and 20 retained samples; setup,
cloning and dropping are outside its timer. Three complete paired suites run
in baseline/candidate, candidate/baseline, baseline/candidate order. The primary
gate requires a median paired duration ratio at most 0.5 for the 4096-operation
layout case and **both** unchanged AD tests. All existing assertions must pass.
Intervals are an exploratory paired-cluster percentile bootstrap over the
three pair ratios (27 resamples), not a large-sample confidence claim.

The two ignored probes are
`compiler::tests::layout_simplifier::benchmark_algebraic_layout_simplifier_scaling`
and `compiler::tests::scaling::benchmark_conj_sinking_scaling`. Run each with
`cargo test --release -p tenferro-runtime --no-default-features --features
blas-accelerate --lib <probe> -- --exact --ignored --nocapture`. For the original
baseline, copy only the two probe modules and their module declarations from
the candidate; they use pass APIs already present at the original pin.

For end-to-end reproduction, build the five affected tensor4all-rs crates
(`treetn`, `partitionedtreetn`, `treeaci`, `interpolativeqtt`, and `tutorial-code`)
with `--release --lib --test integration`, then invoke the treetn integration
binary once per exact AD test name in `final-local.json`. Time the complete
process, retain the full paired suite, and do not substitute individual cases.

The conjugation diagnostic avoids a sinking fixpoint to isolate metadata and
redirect traversal. The production outer fixpoint remains; this experiment
does not establish linear complexity for all conjugation graphs or full
compilation. Affinity-controlled Linux validation remains to be performed
before claiming compliance with the full upstream performance protocol.
