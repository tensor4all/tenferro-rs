# Canonical uninitialized copy dispatch candidate

2026-09-17; AMD EPYC7713P, explicit 1T provider-matched diagnostics.

The earlier compact-operand borrowing candidate was rejected: its 0.020–0.030%
LM instruction improvement missed the preregistered 1% gate. Its complete patch,
profiles and rationale remain in benchmark commit `9c4a580`; none of that
production change is retained.

This candidate changes one non-conjugating CPU layout-copy call from generic
`map_into` to `strided_kernel::copy_into_uninit`. Conjugating copies, provider
fallback/retry rules, ownership, allocation and error-path reclamation are
unchanged. The shared implementation is strided revision
`78d519013e8b44bd80f77eb01d31039f0be9aae6`, initially selected with local Cargo
overrides. Integration now pins merged strided commit `5bc5ab75` (PR259).

The shared API reuses existing permutation machinery only for sequential,
non-contiguous f32/f64 copies. It retains map for contiguous, bounded-parallel
and other element types, including complex and padded types. The exact-type
restriction matters because the permutation engine's 4/8-byte paths interpret
storage as native floats. No initialized Rust references are formed over the
unwritten output; successful copying initializes every logical output element.

Kernel-only diagnostic results and nonzero reference checks are committed in
strided-rs-benchmark-suite `47f7c2e`, under
`result/amd-cpu/uninit-copy-kernels/`. The LM-shaped permutation shows 52.265%
fewer instructions; contiguous and tiny controls change by −0.100% and −0.255%
respectively (small regressions, not improvements). These are not wall-clock
claims. Final focused release tests pass 5/5, also under Memcheck with zero
errors; leak checking was disabled.

## Integration validation

Docker Rust1.98.1, shared OpenBLAS0.3.26, no kache; RAYON/OPENBLAS/OMP threads
explicitly 1, test harness threads 1, Cargo jobs 16. Temporary root-manifest
strided patches were restored byte-for-byte after the run. This passes the
patches to trybuild without committing a dependency integration change.

`trybuild` deliberately replaces RUSTFLAGS, so the BLAS-feature positive fixture
initially failed to link cblas symbols. A test-only linker supplies the existing
provider library without changing snapshots or crate behavior:

```sh
#!/bin/sh
exec cc "$@" -L/opt/openblas/lib -lopenblas
```

With `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER` set to that wrapper and
`RUSTDOCFLAGS='-C linker=/tmp/cpu-openblas-linker'`:

```sh
cargo test -j 16 -p tenferro-cpu -p tenferro-ad \
  --features tenferro-cpu/cpu-blas,tenferro-ad/cpu-blas --no-fail-fast \
  -- --test-threads=1
```

Captured results include AD library91, integration355 (all five UI fixtures),
doctests174; CPU library564 and all integration binaries passed. The outer
client timed out at420 seconds during CPU doctests; inspection confirmed its
owned Cargo/rustdoc descendants were still running. Docker's final `die` event
records **exitCode=0, execDuration=455s**, proving completion of the actual gate.
The stdout tail after the client timeout was not captured (CPU collected219
doctests). Do not misreport the outer command as a synchronous successful run.
Logs and the final container event are retained with benchmark evidence.

## Evidence limits and integration decision

The preregistered whole-eager N1/N3 gate requested three matched pairs. One
complete pair measured LM instruction reduction 11.897%, with multiply/GEMM
controls essentially unchanged; the maintainer cancelled the remaining repeats.
This is provisional instruction evidence, not a passed three-pair gate.

Two complete native 1T suites were contaminated by foreign runnable jobs;
the apparent copy effect reversed direction between suites. CPU-domain reservation
failed with EPERM before measurement; independent verification found all 369 live
thread affinities unchanged. No native speedup or regression-free timing claim
is supported. The maintainer stopped timing attempts, deferred 4T combinations,
and explicitly requested PRs and merging with these limits disclosed.

Benchmark evidence is in tenferro-benchmark commit `b127b11` under
`result/amd-cpu/{canonical-copy-dispatch,native-1t,small-gemm-gemv}/`.
The small-GEMM GEMV experiment remains diagnostic-only; no GEMV production change
is included. Final repository gates and the ordinary merged dependency pin are
the integration checks; they do not complete the original performance goal.

## Final ordinary-dependency gate

Docker Rust1.98.1 with all six strided crates resolved from GitHub commit
`5bc5ab75a20277f0c8820cb288b23f6bb6dfbd91`, without path/manifest overrides:

- `scripts/check-pr-fast.sh --base origin/main --coverage-reviewed` with focused
  `cargo test -j 16 -p tenferro-cpu -p tenferro-ad --no-fail-fast -- --test-threads=1`
  passed: formatting, documentation snippets, strict workspace/all-target Clippy,
  Clippy for tropical/sparse/TBLIS extensions, and 1439 Rust tests/doctests.
- BLAS-only six uninitialized-output tests passed with shared OpenBLAS linkage.
- The relocated injected-provider opt-out test passed with `cpu-blas,provider-inject`.
- OpenBLAS/OMP/Rayon and test harness threads were explicitly 1; Cargo jobs 16.

The final preflight corrected one cloned-reference test lint; no lint suppression
or UI snapshot changes were used. Full hosted CI remains the merge gate.

## Consolidated PR CI follow-up

The broader BLAS workspace lane subsequently exposed three existing ellipsis
broadcast tests returning incorrect values. Reproducing locally and temporarily
restoring generic map isolated the problem to shared permutation copying, not
BLAS arithmetic. The HPTT planner selected a zero-stride source inner axis while
its transpose microkernel assumed +1; tightly sized source storage could also
be read beyond its extent.

Strided PR260 fixes that shared planner: non-unit selected inner strides use the
existing general-stride copy loop. Tests cover broadcast, negative and gapped
source/destination strides, holes, and serial/explicit-two-thread execution;
focused Memcheck reports zero errors (leak checking disabled). All upstream CI
passed. The dependency pin now selects its merged commit
`1be41ce4e656376cb3aa533d97a7711243b1b9ce`.

With that ordinary GitHub pin and no overrides, the complete BLAS workspace
nextest run passes all 3059 tests locally, including the three previously failing
tests. Historical instruction measurements above predate this correction and
were not repeated. PR1804's workflow regression test is consolidated into PR1807;
PR1797 is already in main. GPU validation is reserved for the consolidated final
head after CPU gates succeed; PR1800 is intentionally excluded.
