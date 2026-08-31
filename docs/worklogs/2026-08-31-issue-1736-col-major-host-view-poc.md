# Issue #1736 validated column-major host-view PoC

## Scope

This PoC adapts the contributor-supplied `src/col_major.rs` prototype to the
`tenferro-tensor` ownership, layout, placement, and typed-error contracts. It
adds const-generic shared and mutable host views, constructors on static-rank
owned tensors and shared tensor views, focused tests derived from the Poisson
Jacobi example, Criterion cases, and fixed-rank code-generation probes.

The baseline is commit `0e87f88eedf28545767b316a7795ed970069ac83`, and the
implementation candidate is commit `e70cdbfc`.

## Context reviewed

- Issue #1736 and its acceptance criteria.
- `crates/tenferro-tensor/src/types.rs`, `types/accessors.rs`, crate exports,
  existing element-access benchmarks, and storage/view tests.
- `docs/design/tensor.md`, the relevant storage ownership contract, repository
  rules, and current tensor4all shared Rust, performance, validation,
  documentation, and provenance rules.
- The contributor-supplied `src/col_major.rs` and Poisson Jacobi use case in
  the enclosing workspace.

## Design decisions

- Keep the validated views in `tenferro-tensor`, which owns backend-capable
  `TypedTensor` storage and placement checks.
- Reuse `crate::Error::Validation` and `crate::Error::RuntimeState` instead of
  introducing a parallel view error enum.
- Reuse `TypedTensorView::as_slice()` for compact-offset range validation and
  zero-copy logical slicing.
- Store only `[usize; N]` and the exact borrowed slice in the hot-loop wrapper.
- Use slice iteration and `ChunksExact`/`ChunksExactMut` as the guaranteed safe
  fast paths. Keep random checked `get`/`get_mut` access as a convenience path.
- Do not implement `Index` or `IndexMut`: those traits cannot express a typed or
  optional out-of-bounds result, and the repository forbids turning invalid
  public input into a panic.
- Keep the unsafe accessor local to the validated wrapper. Its proof is the
  checked shape product, exact slice length, and caller-provided in-bounds
  coordinates.
- Defer a dynamic-rank counterpart and mutable `TypedTensorViewMut` constructor;
  neither is required to prove the static-rank owned/view concept.

## Correctness evidence

Before the change, `cargo test -p tenferro-tensor` passed 267 unit tests, all
integration and UI contract tests, and 330 doctests.

The focused PoC tests cover rank 3 column-major offsets, checked and unsafe
access, immutable and mutable traversal, rank zero, empty and singleton
dimensions, nonzero compact offsets, zero-copy pointer identity, noncompact
rejection, shape-product overflow, slice-length mismatch, backend storage
rejection, and one Poisson Jacobi update. The initial focused run passed all
seven tests.

`cargo clippy -p tenferro-tensor --all-targets -- -D warnings` and
`python3 scripts/check-public-error-docs.py
crates/tenferro-tensor/src/types/col_major.rs` passed. A focused doctest run
found one temporary-lifetime example error; the example now binds the view
before collecting lane borrows. The rerun passed all 23 focused doctests, and
the focused unit tests passed after the final empty-shape adjustment.

The final full `cargo test -p tenferro-tensor` run passed 274 unit tests, every
integration and UI contract test, and 352 doctests.

After adding the final error-path coverage case, the focused suite passed all
eight tests. The repository fast PR gate also passed with
`--coverage-reviewed` and the focused PoC suite as its test command. Its checks
included formatting, documentation snippets, workspace and standalone-crate
Clippy runs, and the focused tests. The deterministic worktree rules review
reported `pass`.

`cargo llvm-cov` is not installed in the local toolchain, so no generated
coverage report is available; the coverage attestation is based on manual path
review and the added error-path test. `scripts/check-docs-site.py` completed its
snippet check, then stopped because the active Python is older than the script's
Python 3.11 requirement.

Claude Fable reviewed the complete PR diff after it was opened. It reported no
blocking findings and confirmed the unsafe proof chain, mutable iterator
disjointness, compact offset validation, and backend rejection. Follow-up edits
loosen the shared `TypedTensorView` constructor lifetime to the backing-data
lifetime, reuse the existing view element-count helper, remove a duplicate
compactness check, and cover shared-view `Debug`. A subsequent hosted review
correctly identified the panicking `Index`/`IndexMut` surface as incompatible
with the repository public-boundary contract, so those trait implementations
were removed in favor of `get`/`get_mut`.
`IntoIterator`, `Copy`/`Clone`, dynamic rank, and a mutable tensor-view
constructor remain deferred API additions rather than requirements of this PoC.

## Performance experiment

The need gate was already established by the 2026-08-17 element-access record:
rank-2 nested `get2` took 31.315 microseconds versus 3.775 microseconds for the
expert unchecked slice loop.

The experiment was declared before the candidate run:

- baseline commit: `0e87f88eedf28545767b316a7795ed970069ac83`;
- benchmark: `crates/tenferro-tensor/benches/element_access.rs`;
- toolchain and target: Rust 1.98.0, `aarch64-apple-darwin`;
- profile: Criterion release, one-second warm-up, two-second measurement,
  20 samples, scalar single-threaded loops, no CPU affinity control on macOS;
- cases: rank-2/rank-3 raw nested slice, checked validated `get`, unchecked
  validated access, safe axis-0 lanes, mutable lanes, and existing direct
  slice/tensor iteration non-regression cases;
- primary gate: safe axis-0 lanes no more than 5% slower than the corresponding
  raw slice median for rank 2 and rank 3;
- non-regression gate: no existing direct-slice or tensor-iterator median more
  than 5% slower than baseline.

The first comparison command stopped when Criterion encountered a new case
without a saved baseline. Its partial results were discarded, and the complete
candidate suite was rerun under the declared settings.

| Case | Raw slice median | Axis-0 lanes median | Difference | Result |
|---|---:|---:|---:|---|
| rank 2, 4096 elements | 2.3704 us | 2.3936 us | +0.98% | PASS |
| rank 3, 4096 elements | 2.3862 us | 2.3540 us | -1.35% | PASS |

The candidate checked-index medians were 3.4340 microseconds for rank 2 and
5.4746 microseconds for rank 3. The unchecked medians were 2.3495 and 2.4248
microseconds. Rank-2 mutable lanes took 2.7717 microseconds.

Existing `as_slice_iter`, `tensor_iter`, and `tensor_iter_mut` medians changed
from 36.905, 37.633, and 42.040 microseconds to 36.812, 36.958, and 41.309
microseconds. No declared non-regression gate failed.

## Code generation

`scripts/check-storage-static-rank-codegen.py` compiled the updated probes but
reported `INCONCLUSIVE` because its backward-label parser expects ELF-style
labels beginning with `.`, while Apple assembly uses `LBB...` labels. Manual
inspection of the generated AArch64 assembly found vectorized/scalar read loops
at `LBB555_12`/`LBB555_14` and write loops at `LBB556_9`/`LBB556_12`. Those loop
bodies contain loads, floating-point arithmetic, pointer/counter updates, and
loop branches; rank, backend, layout, error, and allocation handling remains
before the loops.

The final PR should obtain an automated passing report on a supported target or
extend the checker to classify Mach-O labels before treating code generation as
a closed gate.

## Remaining verification

- Re-run the deterministic rules review against the final committed candidate.
- Obtain a supported automated codegen report, or extend the checker for Mach-O
  labels, before treating the codegen gate as closed.
- Run the complete documentation-site check in a Python 3.11+ environment.
