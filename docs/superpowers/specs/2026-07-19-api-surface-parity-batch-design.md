# API Surface Parity Batch Design

**Issues:** #1279, #1265, #1264, and #1266

**Goal:** Establish one consistent reduction-axis contract, complete eager and
concrete linalg surfaces, and add the missing eager FFT surface without changing
backend mathematics, AD semantics, or device-transfer behavior.

## Scope And Order

The work is one ordered public-surface batch with four independently testable
slices:

1. #1279 defines reduction-axis semantics used by core eager/traced reductions
   and linalg `norm`.
2. #1265 makes eager linalg match the resulting traced linalg vocabulary.
3. #1264 exposes that vocabulary on concrete erased and typed tensors.
4. #1266 adds eager FFT after PR #1424 has merged and the branch has been
   updated to its backend-capability architecture.

The first three slices may proceed immediately. The FFT slice must not resolve
PR #1424 conflicts piecemeal. If #1424 is not merged when the first three slices
are ready, the batch remains local or draft until the dependency is resolved.

## Alternatives Considered

### Reduction axes

- **Selected:** public eager/traced methods take `Option<&[usize]>`; `None`
  means all axes and `Some(&[])` means identity. Normalize to explicit axes
  before constructing the existing operation.
- Rejected: keep `&[usize]` and add `reduce_*_all`. This preserves a semantic
  trap and leaves reduction-like APIs inconsistent.
- Rejected: add a second all-axes variant to the IR. Rank is known at the public
  eager/traced boundary, so a second internal representation adds no value.

### Linalg parity implementation

- **Selected:** keep the public traits explicit and implement eager composites
  in the linalg crate using existing eager primitives. Share focused validation,
  dtype, option, and output-count helpers where their contracts are genuinely
  identical; enforce algebraic parity with tests.
- Rejected: a broad generic abstraction over eager and traced tensor types. The
  two surfaces differ in symbolic shape handling, runtime ownership, and error
  timing, so a generic parameter bag would obscure those differences.
- Rejected: add new backend kernels for composite operations. The linalg design
  deliberately keeps `det`, `inv`, `pinv`, `norm`, and eigenvalue-only
  conveniences above the kernel basis.

### Concrete and typed linalg

- **Selected:** crate-root extension traits delegate to existing
  `LinalgBackend` owned/read hooks. Typed inputs use the existing dtype-erased
  borrowed-read conversion, avoiding an input clone or a new backend contract.
- Rejected: teach users to call `LinalgBackend` directly. That leaves linalg
  inconsistent with einsum and FFT and exposes `Vec<Tensor>` output arity.
- Rejected: implement typed methods by cloning into an owned `Tensor`. That
  would create a hidden tensor-sized copy at a public convenience boundary.

### Eager FFT

- **Selected:** add an `autodiff`-gated extension trait that registers and
  applies existing FFT extension operations through the eager runtime.
- Rejected: execute a concrete FFT directly and wrap the result. That would
  bypass eager graph recording and existing FFT AD rules.

## 1. Reduction-Axis Contract

The canonical eager and traced signatures become:

```rust
fn reduce_sum(&self, axes: Option<&[usize]>) -> Result<Self>;
fn reduce_prod(&self, axes: Option<&[usize]>) -> Result<Self>;
fn reduce_max(&self, axes: Option<&[usize]>) -> Result<Self>;
fn reduce_min(&self, axes: Option<&[usize]>) -> Result<Self>;
```

Semantics are:

- `None`: expand to `0..rank` and reduce every axis;
- `Some(&[])`: identity, preserving shape and values;
- `Some(axes)`: validate uniqueness and bounds, then reduce those axes.

The public methods normalize to `Vec<usize>` before creating `StdTensorOp`.
`StdTensorOp`, execution IR, and backend reduction traits continue to receive
explicit axes. A rank-zero tensor expands `None` to an empty axis list and is
therefore unchanged. No deprecated overload or alternative spelling remains.

`TracedTensorLinalgExt::norm` and the new eager equivalent retain
`Option<&[usize]>` and the same `None`/empty distinction. The durable contract
is recorded in `docs/spec/operation-categories.md`.

### Errors

`Some(axes)` preserves typed duplicate-axis and out-of-bounds errors. `None`
cannot produce an axis-validation error because its axes are derived from the
input rank. Validation happens before graph mutation or eager execution.

### Tests

Eager and traced tests cover `None`, `Some(&[])`, one explicit axis, all four
reduction operations, rank zero, duplicate axes, and out-of-range axes.
Doctests demonstrate the three valid spellings. Existing callers are migrated
directly to `Some(...)` or `None` according to intent.

## 2. Eager Linalg Parity

`EagerTensorLinalgExt` matches the current `TracedTensorLinalgExt` names and
signatures. Current option-bearing names such as `svd_with_options`,
`qr_with_options`, and `eigh_with_options` are canonical; stale `_with_eps`
wording is not introduced.

The eager surface adds the traced-only composites:

- `slogdet`, `det`, and `inv`;
- `eigvalsh` and `eigvals`;
- `pinv` and `pinv_with_rtol`;
- `norm`.

Each composite uses existing eager linalg extension operations plus core eager
tensor operations. It registers the linalg runtime through the established
eager extension path. Defaults, dtype promotion, tuple order, derivative
options, and error categories match traced behavior. No new primitive,
backend hook, or AD rule is added.

### Errors

Input dtype, rank, shape, option, and axis validation occurs before applying
the first eager operation when the corresponding facts are available. Runtime
and numerical failures retain their existing typed sources. Unexpected
extension output counts remain internal errors with the operation name.

### Tests

Compile-time trait parity tests compare the complete method vocabulary.
Value tests compare eager and traced execution for every added family, including
real and complex cases where output dtype differs. `norm` covers `None`, empty,
vector, matrix, `keepdim`, and invalid axes. Existing AD numerical coverage is
reused because the eager composites emit the same established operations; at
least one gradient path per composite family verifies eager recording.

## 3. Concrete And Typed Linalg Surfaces

The crate root exports:

- `TensorLinalgExt` for owned dynamic-dtype tensors;
- `TensorReadLinalgExt` for owned references and borrowed views;
- `TypedTensorLinalgExt<T>` for statically typed tensors.

Methods use the tensor as receiver and accept `&mut B` where
`B: LinalgBackend`. Unsuffixed owned and typed methods retain normal operation
names. Borrowed methods use the repository `_read` vocabulary. Multi-output
operations return tuples with fixed arity rather than `Vec<Tensor>`.

The operation set matches the completed eager/traced public linalg set. Kernel
operations delegate to the matching `LinalgBackend` owned or read hook.
Composite operations live in `tenferro-linalg` and use backend-explicit core
tensor operations; they do not add backend kernels.

Typed dispatch is restricted by a sealed public `LinalgScalar` contract for
`f32`, `f64`, `Complex32`, and `Complex64`. It associates:

- `Real`: `f32` for `f32`/`Complex32`, `f64` for `f64`/`Complex64`;
- `Complex`: `Complex32` for `f32`/`Complex32`, `Complex64` for
  `f64`/`Complex64`.

This gives typed output contracts such as:

```rust
fn svd<B: LinalgBackend>(
    &self,
    backend: &mut B,
) -> Result<(
    TypedTensor<T>,
    TypedTensor<T::Real>,
    TypedTensor<T>,
)>;

fn eig<B: LinalgBackend>(
    &self,
    backend: &mut B,
) -> Result<(
    TypedTensor<T::Complex>,
    TypedTensor<T::Complex>,
)>;
```

`eigh`/`eigvalsh` and `norm` return the associated real dtype where required;
`eig`/`eigvals` return the associated complex dtype. Same-dtype operations
return `TypedTensor<T>`. Typed inputs become `TensorRead` through the existing
scalar erasure hook, so the convenience API does not clone input storage.
Output downcasts validate the backend contract and report an internal error if
a backend returns an impossible dtype or arity.

### Placement And Layout

The extension traits do not transfer tensors. Owned methods preserve the
backend placement contract. Read methods pass views to existing backend read
hooks; any provider-required same-placement canonicalization remains the
documented backend boundary. Unsupported placement or layout returns the
existing typed error rather than falling back to CPU or an owned operation.

### Tests

Tests cover tuple arity and values for all operation families, all four linalg
scalar dtypes where supported, real/complex associated output dtypes, vector
and matrix right-hand sides, strided reads, placement errors, dtype mismatch,
and backend output-contract violations. Rustdoc examples bind and reuse one
backend. Existing backend-as-receiver examples are rewritten to teach the
extension traits.

## 4. Eager FFT Surface

After PR #1424 is merged, `tenferro-fft` exports `EagerTensorFftExt` behind
`autodiff`. Its `fft`, `ifft`, `rfft`, and `irfft` signatures match the
then-current `TracedTensorFftExt`, including `n`, signed axis handling,
normalization, and output dtype rules.

Each method validates metadata available at the eager boundary, registers the
FFT extension runtime on the tensor's eager runtime, and applies the existing
FFT extension operation. The implementation uses #1424's backend-neutral
capability/cache architecture and does not construct a private host executor or
fall back across backends.

### Tests

Runnable doctests cover all four methods. Integration tests compare eager and
traced values and shapes for every transform kind, normalization modes, explicit
lengths, negative axes, invalid axes, unsupported dtypes, and round trips.
Existing FFT AD rules are exercised through at least one eager gradient test.

## Documentation And Enforcement

The implementation updates:

- `docs/spec/operation-categories.md` for reduction semantics and extension
  surface parity;
- `docs/design/linalg.md` for the concrete/typed extension boundary;
- public rustdoc and user examples in the affected crates;
- the operation-category parity checker if its machine-readable expectations
  need to recognize the new methods.

One curated work log records the ordered slices, decisions, verification, and
the #1424 dependency. The final PR closes only issues whose complete acceptance
criteria and tests are present. The batch is merged with a non-squash merge.

## Non-Goals

- New linalg or FFT mathematics, kernels, providers, or AD rules.
- CUDA/cuFFT implementation or implicit CPU/GPU transfer.
- Compatibility shims for the old reduction signature.
- A generic abstraction that hides differences between eager and traced
  execution.
- Prepared factorization or caller-owned output/workspace APIs.
