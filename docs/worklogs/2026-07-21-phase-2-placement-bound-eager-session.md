# Phase 2 Placement-Bound Eager Session

## Summary

Implemented the phase-2 `EagerRuntime::on_cpu` bridge as a placement-selected
CPU snapshot that retains the original eager runtime identity and enters one
existing backend session for a borrowed closure. The bridge is limited to core
`BackendSession` operations on concrete tensors; operation-family registries
remain outside this phase.

## Context reviewed

- issue #1433 phase architecture and child issue #1436 phase-2 contract;
- `AGENTS.md`, `REPOSITORY_RULES.md`, and the shared tensor4all Rust,
  performance, documentation, and test rules;
- the phase-2 design specification and ordered Task 10 plan;
- `EagerRuntime`, `EagerBackend`, `CpuBackend::for_placement`,
  `CpuBackend::run_backend_session_cached`, `BackendSessionHost`, and
  `CpuOperationEntry`; and
- the CPU re-entry owner guard and graph executor's existing one-session tests.

## Decisions

- `on_cpu` clones only the CPU coordinator/provider snapshot while the eager
  backend guard is held. Placement resolution runs after the guard scope.
- An existing binding is unaffected by later coordinator or provider
  replacement on the runtime.
- `CpuPlacementBoundEager` retains no permit while idle, is mutably entered,
  has hand-written summary `Debug`, and is not `Clone`.
- `with_eager_session` calls `CpuBackend::with_backend_session` exactly once
  and performs no runtime lock, extension lock, executor install, or secondary
  mutex entry. Each core operation remains responsible for one
  `CpuOperationEntry`.
- CPU placement failures reuse the CPU backend's typed conversion so
  `CpuPlacementError` remains in the public source chain. Non-CPU runtimes use
  typed `Unsupported`.

## Re-entry finding

Technical inspection found that ordinary same-runtime `EagerTensor` entry from
inside the borrowed CPU session is not currently representable as a typed
error. `BackendSessionHost` returns an arbitrary callback result and
`inherited_or_new_execution_owner` enforces public CPU backend re-entry with
`BACKEND_REENTRY_PANIC`. The new bridge does not hold the eager backend lock
during its callback, so the attempt fails immediately rather than deadlocking.
The public method documents `# Panics`, and a regression test catches the
unwind, checks the stable rejection message, and proves that a following
session succeeds. Catching and translating this internal panic was rejected
because it would erase unrelated user panics and weaken the existing backend
contract.

## Verification scope

Focused tests cover runtime identity, requested placement, idle permit
behavior, CPU coordinator/provider snapshot semantics, external executor
lifetime, typed placement and executor sources, zero/one/two operation entry
counts, borrowed non-`'static` callbacks, callback error/unwind recovery,
same-runtime re-entry recovery, and core graph parity. Source-contract tests
enforce the lock/entry shape and the no-`Clone`/hand-written-`Debug` contract.
Runnable rustdoc examples exercise construction and one core operation.

The final verification run completed with:

- `cargo test -p tenferro-ad --lib`: 69 passed;
- `cargo test -p tenferro-ad --test integration`: 430 passed;
- `cargo test -p tenferro-runtime graph::executor::tests`: 25 passed;
- `cargo test -p tenferro-ad --doc`: 135 passed;
- `cargo clippy -p tenferro-ad --all-targets -- -D warnings`;
- `cargo check -p tenferro-ad --features cuda,webgpu`;
- `cargo doc --workspace --no-deps` followed by
  `python3 scripts/check-docs-site.py`: 13 workspace library crates and four
  guide dependency snippets verified;
- `python3 scripts/check-public-error-docs.py` and
  `python3 scripts/test-doc-consistency.py`; and
- `cargo fmt --all --check` and `git diff --check`.

`cargo check -p tenferro-ad --all-features` is not a valid Linux feature
matrix: it enables the Apple-only `blas-accelerate` provider and stops in
`accelerate-src` before compiling this crate. The supported non-CPU branches
were therefore compiled explicitly with `cuda,webgpu` above.

## Deferred scope and risks

- Linalg, FFT, einsum, extension-runtime dispatch, and AD-aware placement
  ergonomics are intentionally not supported by this phase-2 adapter.
- Converting recursive public CPU entry to a typed error requires an owning
  change to the CPU/`BackendSessionHost` entry contract; this task leaves the
  established panic boundary explicit instead of adding a compatibility shim.
