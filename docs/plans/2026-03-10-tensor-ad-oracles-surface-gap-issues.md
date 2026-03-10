# Tensor AD Oracles Surface Gap Issues

This file tracks the implementation issues that must exist before oracle replay
expansion can cover all families already expressible by the current tenferro
public APIs.

## Backlog

### 1. Add structured `*_ex` linear algebra APIs

- Proposed title: `feat: add structured cholesky_ex, inv_ex, and solve_ex APIs`
- Problem: the oracle database publishes `cholesky_ex`, `inv_ex`, and
  `solve_ex`, but tenferro only exposes the plain success-or-error variants.
  Replay coverage should not invent fake wrappers; tenferro needs an explicit
  public contract for structured status-returning variants.
- Required public API contract: add public APIs that return the primary result
  plus operation status/info in a stable Rust type, with CPU implementations
  and clear error/status semantics.
- Likely implementation crates: `tenferro-linalg`,
  `extension/tenferro-dyadtensor`
- Oracle families affected: `cholesky_ex/identity`, `inv_ex/identity`,
  `solve_ex/identity`
- Acceptance criteria:
  - public APIs exist in `tenferro-linalg` with rustdoc examples
  - CPU implementation exists for the published oracle input families
  - dyadtensor wrappers exist if the op participates in the AD surface
  - first-order AD contract is implemented or explicitly rejected with a typed,
    documented limitation
  - oracle replay can classify these families as replayable instead of missing
    surface
- Issue URL: https://github.com/tensor4all/tenferro-rs/issues/433

### 2. Add LU factorization and solve surface

- Proposed title: `feat: add lu_factor, lu_factor_ex, and lu_solve public APIs`
- Problem: tenferro exposes `lu`, but the oracle database also expects packed
  factorization and solve entrypoints built on LU factors. Those are not the
  same public contract as `lu`.
- Required public API contract: add public APIs for packed LU factorization,
  packed LU factorization with status, and solve from precomputed LU factors
  and pivots.
- Likely implementation crates: `tenferro-linalg`,
  `extension/tenferro-dyadtensor`
- Oracle families affected: `lu_factor/identity`, `lu_factor_ex/identity`,
  `lu_solve/identity`
- Acceptance criteria:
  - public APIs exist with documented packed-factor layout and pivot semantics
  - CPU implementations cover the published oracle shapes
  - dyadtensor wrappers and AD rules exist where the oracle replay requires
    them
  - replay work no longer needs to treat LU factorization families as missing
    public surface
- Issue URL: https://github.com/tensor4all/tenferro-rs/issues/434

### 3. Add tensor construction linalg helpers

- Proposed title: `feat: add cross, householder_product, and vander APIs`
- Problem: the oracle database includes tensor-construction families that are
  not currently exposed as first-class tenferro public APIs.
- Required public API contract: add public APIs for vector cross product,
  Householder reflector product construction, and Vandermonde matrix
  construction with explicit shape and axis semantics.
- Likely implementation crates: `tenferro-linalg`,
  `extension/tenferro-dyadtensor`
- Oracle families affected: `cross/identity`,
  `householder_product/identity`, `vander/identity`
- Acceptance criteria:
  - public APIs exist with examples and shape validation
  - CPU implementations match the oracle family contracts
  - AD coverage exists where the oracle families publish derivative references
  - replay can move these families out of the missing-surface bucket
- Issue URL: https://github.com/tensor4all/tenferro-rs/issues/435

### 4. Add matrix condition number API

- Proposed title: `feat: add cond public API`
- Problem: the oracle database publishes `cond`, but tenferro has no public
  condition-number entrypoint.
- Required public API contract: add a public `cond` API with documented norm
  convention and return shape semantics for supported batched matrices.
- Likely implementation crates: `tenferro-linalg`,
  `extension/tenferro-dyadtensor`
- Oracle families affected: `cond/identity`
- Acceptance criteria:
  - public API exists with examples and norm-selection semantics
  - CPU implementation covers the oracle family inputs
  - first-order AD support is implemented for the published oracle family, or a
    narrower supported subset is documented explicitly
  - replay can classify `cond` as replayable instead of missing surface
- Issue URL: https://github.com/tensor4all/tenferro-rs/issues/436

### 5. Add matrix power API

- Proposed title: `feat: add matrix_power public API`
- Problem: the oracle database publishes `matrix_power`, but tenferro has no
  public matrix-power entrypoint.
- Required public API contract: add a public `matrix_power` API for integer
  exponents on square batched matrices, with documented behavior for zero,
  positive, and negative powers.
- Likely implementation crates: `tenferro-linalg`,
  `extension/tenferro-dyadtensor`
- Oracle families affected: `matrix_power/identity`
- Acceptance criteria:
  - public API exists with examples
  - CPU implementation covers the oracle family inputs
  - AD support is implemented for the supported exponent domain, or documented
    limitations are explicit
  - replay can classify `matrix_power` as replayable instead of missing surface
- Issue URL: https://github.com/tensor4all/tenferro-rs/issues/437

### 6. Add tensor inversion and tensor solve APIs

- Proposed title: `feat: add tensorinv and tensorsolve public APIs`
- Problem: the oracle database publishes tensorized inverse/solve helpers, but
  tenferro only exposes matrix-level solve APIs today.
- Required public API contract: add public `tensorinv` and `tensorsolve` APIs
  with explicit reshaping semantics, axis handling, and shape validation.
- Likely implementation crates: `tenferro-linalg`,
  `extension/tenferro-dyadtensor`
- Oracle families affected: `tensorinv/identity`, `tensorsolve/identity`
- Acceptance criteria:
  - public APIs exist with examples and axis semantics
  - CPU implementations cover the oracle family inputs
  - AD support is implemented for the published oracle families, or the
    supported subset and limitations are documented
  - replay can classify these families as replayable instead of missing surface
- Issue URL: https://github.com/tensor4all/tenferro-rs/issues/438
