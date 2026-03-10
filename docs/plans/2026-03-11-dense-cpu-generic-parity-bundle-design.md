# Dense CPU Generic Parity Bundle Design

## Goal

Define a single coherent PR that closes `#443`, `#444`, and `#445` while
delivering the audit artifacts required by `#446`.

The PR is explicitly **not** the place to implement all missing dense CPU
PyTorch coverage. Its job is to:

- document the dense CPU gap in a form that is actionable
- correct immediate architecture and rustdoc drift introduced by the recent
  protocol split
- remove one concrete abstraction flaw (`LinalgScalar` carrying LAPACK-specific
  eigen helpers)
- remove the most immediate structure violations that would make the bundle
  itself hard to review or extend

## Why One PR Can Only Go This Far

Trying to land full dense CPU parity for scalar, elementwise, reduction,
linalg, VJP, JVP, and oracle-backed HVP in one PR would produce an
unreviewable change set and would almost certainly cement the wrong
architecture.

The largest current gap is still the substrate:

- `TensorScalarPrims`
- `TensorAnalyticPrims`
- dense reductions

The most important architecture debt is still the CPU-centric leakage in:

- `tenferro-linalg`
- `tenferro-dyadtensor`
- legacy blanket adapters over `TensorPrims<A>`

So the maximum coherent one-PR bundle is:

1. ship the audit and mapping for `#446`
2. fix architecture/docs drift for `#443`
3. close immediate rustdoc debt for `#444`
4. remove the most obvious trait-layer leak for `#445`
5. split the worst monolithic modules and inline tests that would otherwise
   block clean CPU/GPU-generic follow-up work

This creates the right base for follow-up implementation PRs without pretending
that broad parity already exists.

## Hard Requirement: CPU/GPU Generic First

This bundle must treat CPU/GPU genericity as a non-negotiable design rule.

That means:

- no new public API or trait that hard-codes `CpuBackend` or `CpuContext`
- no new `with_cpu_runtime(...)` or `ensure_cpu_backend(...)` usage added as
  part of this work
- any code refactor must push responsibilities toward backend-parametric traits
  rather than deeper CPU-only helpers
- audit artifacts must explicitly classify CPU-only runtime assumptions as
  architecture debt
- AD helper and rule APIs touched by this bundle should move away from
  `&mut CpuContext`-specific signatures toward backend-parametric context bounds

This PR does **not** have to remove every existing CPU-only path. It **does**
have to make those paths visible and avoid extending them.

## Review-Driven Must-Fix Items

The current codebase state validates several structural problems that are large
enough to belong in the bundle itself:

- `tenferro-linalg/src/lib.rs` is currently `7,370` lines
- `extension/tenferro-dyadtensor/src/api/mod.rs` is currently `3,687` lines
- `tenferro-linalg/src/lib.rs` contains an inline `eig_scalar_tests` module in
  addition to `mod tests;`
- multiple AD entry points in `tenferro-linalg/src/lib.rs` still hard-code
  `&mut tenferro_prims::CpuContext`
- `ensure_cpu_backend(...)` currently compares backend types with
  `type_name::<...>()`
- `options.expect("checked above")` still exists in library code

These are not merely style nits. They directly affect the CPU/GPU-generic
architecture and the maintainability of the one-PR bundle.

## Review Findings: In Bundle vs Follow-Up

### Must be handled inside this bundle

- split `tenferro-linalg/src/lib.rs` into focused modules
- move `eig_scalar_tests` out of inline test scope
- remove newly touched `CpuContext`-hardcoded AD surface in `tenferro-linalg`
- replace `type_name::<...>()` backend comparison with `TypeId`
- remove `expect(...)` from public/library code paths
- split `extension/tenferro-dyadtensor/src/api/mod.rs` into focused modules

### Must be recorded, but can stay follow-up

- `PrimDescriptor::Permute` still exists in `tenferro-prims`

The `Permute` point is valid architecture debt, but it belongs to the broader
substrate redesign tracked by `#441`, not this bundle. The audit must mention
it explicitly without trying to land the full prims removal here.

## Bundle Scope

### `#446` deliverables

Add a durable audit document to `docs/design/reference/` that contains:

- a family-first dense CPU coverage matrix
- a PyTorch-to-tenferro family mapping appendix
- abstraction/layer findings
- issue-ready backlog categories

This is the main output of the bundle.

### `#443`

Update `AGENTS.md` so its architecture diagrams reflect the actual crate split:

- `tenferro-prims`
- `tenferro-linalg-prims`
- `tenferro-linalg`

### `#444`

Finish rustdoc for the new protocol families:

- `ScalarPrimsDescriptor`
- `TensorScalarPrims`
- `AnalyticPrimsDescriptor`
- `TensorAnalyticPrims`

The docs should clearly distinguish:

- currently supported variants
- forward-declared / reserved variants
- generic execution contract

### `#445`

Extract LAPACK-specific eigen helpers out of `LinalgScalar` into a separate
CPU-oriented trait, tentatively `LapackEigScalar`.

Expected end state:

- `LinalgScalar` describes generic scalar behavior needed across linalg kernels
- the LAPACK real/imag buffer conversion contract is isolated
- CPU eigendecomposition code depends on the LAPACK-specific trait
- future GPU backends are not forced to implement dead-weight LAPACK helpers

### Structural cleanup included in the same PR

The single PR should also perform no-behavior-change file organization cleanup
where it materially improves architecture review:

- split `tenferro-linalg/src/lib.rs` into modules such as:
  - `result_types.rs`
  - `primal.rs`
  - `ad_helpers.rs`
  - `rrules.rs`
  - `frules.rs`
- keep only thin re-export / module wiring in `lib.rs`
- move inline eigen helper tests into crate-local test modules
- split `extension/tenferro-dyadtensor/src/api/mod.rs` by concern, e.g.:
  - `runtime.rs`
  - `primal_builders.rs`
  - `linalg_builders.rs`
  - `ad_builders.rs`
  - `tests/mod.rs`

The exact filenames may change, but the bundle should end with focused modules
instead of multi-thousand-line catch-all files.

## Recommended File Targets

### Docs / audit

- Create: `docs/design/reference/pytorch-dense-cpu-parity.md`
- Modify: `docs/design/index.md`
- Modify: `docs/design/reference/libtorch.md`
- Modify: `docs/design/architecture.md`
- Modify: `docs/design/tensor-prims.md`
- Modify: `docs/design/linalg-prims.md`
- Modify: `docs/design/linalg.md`
- Modify: `docs/design/autodiff.md`

### Agent docs

- Modify: `AGENTS.md`

### Scalar / analytic rustdoc

- Modify: `tenferro-prims/src/scalar_prims.rs`
- Modify: `tenferro-prims/src/analytic_prims.rs`

### Linalg trait cleanup

- Modify: `tenferro-linalg-prims/src/lib.rs`
- Modify: `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/src/tests/mod.rs`
- Create: `tenferro-linalg-prims/src/tests/mod.rs`

### Structural refactor

- Create: `tenferro-linalg/src/result_types.rs`
- Create: `tenferro-linalg/src/primal.rs`
- Create: `tenferro-linalg/src/ad_helpers.rs`
- Create: `tenferro-linalg/src/rrules.rs`
- Create: `tenferro-linalg/src/frules.rs`
- Modify: `tenferro-linalg/src/lib.rs`
- Create: `extension/tenferro-dyadtensor/src/api/runtime.rs`
- Create: `extension/tenferro-dyadtensor/src/api/primal_builders.rs`
- Create: `extension/tenferro-dyadtensor/src/api/linalg_builders.rs`
- Create: `extension/tenferro-dyadtensor/src/api/ad_builders.rs`
- Modify: `extension/tenferro-dyadtensor/src/api/mod.rs`

## Non-Goals

This bundle should **not**:

- add broad new scalar / elementwise / reduction primal kernels
- add broad new VJP/JVP/HVP implementations
- remove all existing CPU-only runtime assumptions in dyadtensor or linalg
- remove `PrimDescriptor::Permute` from the legacy prims surface
- close `#441`

`#441` remains an input and a follow-up implementation stream. This bundle only
turns the dense CPU parity discussion into an actionable audited baseline.

## Success Criteria

The single PR is successful if all of the following are true:

- `#443`, `#444`, and `#445` are closed by merged code/docs
- `#446` has a concrete audit artifact in the repo, not just issue text
- the audit makes CPU-only layer leaks explicit
- no new CPU-only API surface is introduced
- `tenferro-linalg/src/lib.rs` no longer acts as a monolithic implementation
  file
- `extension/tenferro-dyadtensor/src/api/mod.rs` no longer acts as a monolithic
  implementation file
- non-leaf inline tests are moved out of `tenferro-linalg/src/lib.rs`
- required workspace verification passes

## Verification

At minimum, the implementation PR should pass:

- `cargo fmt --all --check`
- `cargo test --workspace --release`
- `cargo llvm-cov --workspace --release --json --output-path coverage.json`
- `python3 scripts/check-coverage.py coverage.json`
- `cargo doc --workspace --no-deps`
- `python3 scripts/check-docs-site.py`

Performance validation is intentionally out of scope for this bundle.
