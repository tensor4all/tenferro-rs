# Repository quality cleanup

This note tracks the current quality-debt cleanup policy for
tensor4all/tenferro-rs. It exists so implementation issues can stay focused
instead of re-deciding what counts as acceptable cleanup work.

Parent tracker: <https://github.com/tensor4all/tenferro-rs/issues/955>

## Debt categories

### Local lint allowances

The workspace allows a small number of local lints for ABI, backend, layout, or
performance reasons. These should be treated individually:

- `dead_code`: remove when repository search proves the item is unused under
  the relevant feature combinations. Retain with a short reason when the item
  is a feature-gated entry point, benchmark/test helper, or intentionally
  dormant cache/debug surface.
- `clippy::too_many_arguments`: acceptable at FFI, BLAS/LAPACK, CUDA, CubeCL,
  or trait-contract boundaries. Internal helpers should move toward named
  descriptors when that makes call sites clearer.
- `clippy::large_enum_variant`: retain only when boxing would obscure ownership
  or make the hot path worse. Otherwise, evaluate a narrow enum-layout refactor.
- `clippy::should_implement_trait`: prefer a trait implementation or a clearer
  method name unless public naming or existing semantics require keeping it.
- `clippy::uninit_vec`: allowed only for performance-sensitive buffer
  allocation paths that immediately initialize the memory and already document
  the safety invariant.

New or retained allowances in production code should include a reason comment
unless the surrounding ABI or safety comment already explains the reason.

### Large modules

Large files are review triggers, not automatic split targets. Split only along
a real responsibility boundary such as parsing, validation, planning,
execution, cache ownership, backend glue, public API, or tests. Avoid arbitrary
`part1`, `common`, or `utils` modules.

Known high-review files include:

- `crates/tenferro-tensor/src/types.rs`
- `crates/tenferro-gpu/src/cubecl/mod.rs`
- `crates/tenferro-linalg/src/cpu/linalg/faer_linalg.rs`
- `crates/tenferro-runtime/src/compiler/mod.rs`
- `crates/tenferro-cpu/src/gemm/mod.rs`
- `crates/tenferro-cpu/src/backend.rs`
- `crates/tenferro-tensor-core/src/lib.rs`
- `crates/tenferro-internal-ops/src/ad/registry.rs`

### Public wrapper duplication

The tensor, eager, traced, and extension wrapper surfaces contain real
duplication, but they are near public API boundaries. DRY refactors here must
preserve public names, feature gates, rustdoc clarity, error ownership, and
behavior. Prefer one narrow wrapper family per PR.

Do not assume these wrapper surfaces are fully isomorphic. Macro/codegen is
allowed only when the chosen wrapper family is genuinely same-shaped and the
invocation keeps public names, docs, feature gating, and error behavior obvious.
For mixed wrapper families, prefer explicit public wrappers plus small private
helpers or descriptors.

### Performance TODOs

Einsum v2 performance items such as transpose folding, dot decomposition,
stride-aware execution, and buffer pooling are not mechanical cleanup. They
require dedicated performance design and measurement.

### AD semantic TODOs

AD diagonal TODOs such as replacing `ExtractDiag` / `EmbedDiag` with
`Gather` / `Scatter` are semantic implementation work. They require AD
correctness tests and should not be mixed into lint or file-organization PRs.

## Cleanup order

1. Inventory and policy updates.
2. Local lint allowance and stale dead-code cleanup.
3. Focused argument-object or descriptor refactors for internal helpers.
4. Large-file splits only when a responsibility boundary is clear.
5. Narrow public-wrapper DRY refactors after choosing an explicit strategy.

Each PR should cover one debt class or one module family. Numeric behavior,
backend dispatch, tensor layout semantics, and AD semantics must stay unchanged
unless a separate accepted issue explicitly scopes that change.

Cleanup PRs should link a work log under `docs/worklogs/` when they make
nontrivial abstraction, module-split, macro/codegen, or deferral decisions. If
the cleanup establishes durable design intent, update `docs/design/` in the
same PR.
