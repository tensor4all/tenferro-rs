# Tensor AD Oracles Issues 433-438 Completion Design

## Goal

Complete the remaining `tensor-ad-oracles` surface tracked by issues
[#433](https://github.com/tensor4all/tenferro-rs/issues/433) through
[#438](https://github.com/tensor4all/tenferro-rs/issues/438) in one integrated
tenferro PR.

The integrated PR must include, for every family in scope:

- public API in tenferro style
- CPU implementation
- rustdoc examples
- required unit and integration tests
- dyadtensor surface where appropriate
- AD support where appropriate
- oracle replay promotion from unsupported to supported

## Scope

The integrated PR covers these oracle families:

- `cholesky_ex`, `inv_ex`, `solve_ex`
- `lu_factor`, `lu_factor_ex`, `lu_solve`
- `cross`, `householder_product`, `vander`
- `cond`
- `matrix_power`
- `tensorinv`, `tensorsolve`

It also includes replay promotion for already-expressible families that were
previously deferred in practice:

- `multi_dot`
- `pinv_hermitian`
- `vecdot`

## Core Decision

Use tenferro-native Rust APIs rather than PyTorch-shaped wrappers.

That means:

- explicit result structs instead of tuple-heavy compatibility returns
- ordinary `Result<T, Error>` for contract violations
- `*_ex` status/info payloads only for numerical execution state
- naming and result shapes aligned with the current `tenferro-linalg` style

This keeps the new surface consistent with existing APIs such as:

- `SvdResult`
- `QrResult`
- `LuResult`
- `SlogdetResult`

## Error Model

For `*_ex` and `lu_factor_ex` families:

- shape errors, invalid arguments, unsupported backends, and unsupported dtypes
  remain ordinary `Err`
- only successful executions that need numerical status reporting return an
  `Ok(ResultStruct { ..., info })`

The `ex` suffix therefore means "structured numerical status" rather than
"never returns `Err`".

## Architecture

### Layer 1: Public Surface

Add the public API contract in `tenferro-linalg/src/lib.rs` first.

This layer owns:

- new public result structs
- public function signatures
- rustdoc examples
- argument validation shape contracts

The result structs should be minimal and explicit, for example:

- `CholeskyExResult<T>`
- `InvExResult<T>`
- `SolveExResult<T>`
- `LuFactorResult<T>`
- `LuFactorExResult<T>`

### Layer 2: CPU Implementation

Implement or expose CPU functionality behind the new surface.

Primary files:

- `tenferro-linalg/src/backend/tensor_api.rs`
- `tenferro-linalg/src/backend/cpu.rs`
- `tenferro-linalg/src/backend/cpu_tensor_impl.rs`
- `tenferro-linalg/src/lib.rs`

Use existing CPU building blocks whenever possible:

- `lu_factor` backend helper for packed LU work
- existing `solve`, `inv`, `pinv`, `norm`, `matrix_exp`, and `einsum`
  functionality for composed operations

Avoid inventing replay-only helpers. If replay needs an operation, production
surface owns it.

### Layer 3: dyadtensor and AD

Expose the new operations in `extension/tenferro-dyadtensor`.

Not every new public API needs full differentiation:

- `*_ex`, `lu_factor`, and `lu_factor_ex` are primal-only surface by design
- composed differentiable ops should reuse existing linalg/einsum rules where
  possible
- only operations with meaningful differentiable contracts should gain explicit
  frule/rrule support

### Layer 4: Oracle Replay

After production surface and tests exist, update the oracle replay harness:

- `tenferro-linalg/tests/oracle_db/support.rs`
- `tenferro-linalg/tests/oracle_db/replay.rs`
- `tenferro-linalg/tests/oracle_db/hvp.rs`
- observable helpers as needed

Unsupported families move to supported only after the production contract is in
place and covered by tests.

## Delivery Batches

The PR stays integrated, but implementation proceeds in three internal batches.

### Batch A: Existing linalg extensions

- `#433`: `cholesky_ex`, `inv_ex`, `solve_ex`
- `#434`: `lu_factor`, `lu_factor_ex`, `lu_solve`
- `#436`: `cond`
- `#437`: `matrix_power`

These mostly extend existing linalg codepaths and should land first.

### Batch B: New tensor construction and tensorized solves

- `#435`: `cross`, `householder_product`, `vander`
- `#438`: `tensorinv`, `tensorsolve`

These have the highest shape-contract design load and should be isolated from
the existing-linalg extensions.

### Batch C: dyadtensor and replay promotion

After A and B settle:

- add dyadtensor builders / eager wrappers
- add AD coverage where applicable
- promote oracle replay support
- update docs/design and generated support tracking if needed

## AD Policy

### Primal-only APIs

These should exist in dyadtensor as primal operations but should not receive
full differentiation contracts in this PR:

- `cholesky_ex`
- `inv_ex`
- `solve_ex`
- `lu_factor`
- `lu_factor_ex`

### Compose-from-existing-rules

Prefer composition over new derivative formulas for:

- `cond`
- `matrix_power`
- `tensorinv`
- `tensorsolve`
- `pinv_hermitian`
- `multi_dot`
- `vecdot`

### Explicit rule work allowed

Explicit AD work is acceptable where composition is not clean enough:

- `cross`
- `householder_product`
- `vander`
- `lu_solve`

## Testing Strategy

Testing order is fixed:

1. primal linalg tests
2. dyadtensor / AD tests
3. oracle replay promotion

Primary test areas:

- `tenferro-linalg` unit and integration tests for API contracts
- `extension/tenferro-dyadtensor/src/api/ad/tests/`
- `tenferro-linalg/tests/oracle_db/`

The replay harness should never be the first place that validates a production
contract.

## Parallelization Strategy

The implementation is one integrated PR, but the work naturally splits into
independent domains once the public API layer is specified:

- structured `*_ex` and LU factorization surface
- scalar-output and composed linalg utilities
- tensor construction and tensorized solve helpers
- dyadtensor / replay integration

This is the right granularity for parallel worktrees or parallel subagents,
because those groups mostly touch distinct logic even though they converge in
`tenferro-linalg/src/lib.rs`.

## Documentation Requirements

Every new public type and function must include `# Examples`.

Also update:

- `docs/design/linalg.md`
- any README or support report references affected by the newly supported
  oracle families

## Non-Goals

- GPU implementation for any new family
- PyTorch-shaped return compatibility
- replay-only wrappers that bypass production surface
- separate PRs for #433 through #438

## Success Criteria

The integrated PR is complete when:

1. all families for issues #433 through #438 have public APIs and CPU
   implementations
2. dyadtensor and AD support are present where the contract is meaningful
3. oracle replay no longer classifies those families as unsupported
4. docs and examples describe the new surface accurately
5. the repository verification gates still pass
