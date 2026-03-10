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

This PR does **not** have to remove every existing CPU-only path. It **does**
have to make those paths visible and avoid extending them.

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

## Non-Goals

This bundle should **not**:

- add broad new scalar / elementwise / reduction primal kernels
- add broad new VJP/JVP/HVP implementations
- remove all existing CPU-only runtime assumptions in dyadtensor or linalg
- close `#441`

`#441` remains an input and a follow-up implementation stream. This bundle only
turns the dense CPU parity discussion into an actionable audited baseline.

## Success Criteria

The single PR is successful if all of the following are true:

- `#443`, `#444`, and `#445` are closed by merged code/docs
- `#446` has a concrete audit artifact in the repo, not just issue text
- the audit makes CPU-only layer leaks explicit
- no new CPU-only API surface is introduced
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
