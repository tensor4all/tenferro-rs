# Tensor Semantics

**Date:** 2026-05-28
**Parent:** `../index.md`
**Related:** `../architecture/tenferro-crates.md`, `backend-contract.md`,
`primitive-catalog.md`

---

## I. Purpose

This document specifies the current dense tensor data model split between
`tenferro-tensor-core` and `tenferro-tensor`.

The split is intentional:

- `tenferro-tensor-core` is a lightweight rank/layout, dtype and scalar
  metadata layer with no tensor container.
- `tenferro-tensor` owns every tensor type: the host container family
  (`HostTensor`, `DefaultScalars`, `ScalarSet`, `ErasedHostTensor`, moved from
  core by #1938), runtime tensor storage, placement metadata, typed views and
  backend traits.
- `tenferro-cpu` owns CPU backend implementations, CPU kernels, provider
  selection, and CPU execution resources.

`tenferro-tensor-core` must not require computation backends, GPU runtimes,
provider selection, graph execution, or AD. Crates that need only dtype tags,
scalar traits, shape/stride metadata, or metadata-only layouts should depend
on `tenferro-tensor-core`; host tensor data needs `tenferro-tensor`, but no
backend construction, session or AD registration.

---

## II. `tenferro-tensor-core`

`tenferro-tensor-core` owns backend-independent metadata only.

Current public concepts:

- `DType`: runtime dtype tags. The preset set declares `F32`, `F64`, `I32`,
  `I64`, `Bool`, `C32`, and `C64`; a value whose scalar no preset declares
  carries `DType::External(TypeId)`. Tag enums are declared with
  `define_scalar_tag!`; `define_scalar_set!` (in `tenferro-tensor`) builds a
  tag enum plus a host-tensor value enum on top of it.
- `TensorScalar`: sealed scalar trait (`Real` plus `dtype()`) for the preset
  members. The open boundary a downstream scalar implements is `Scalar`
  together with the arithmetic and domain traits.
- Promotion facts: `MemberKind`, `MemberSpec`, `promote_specs`,
  `promote_in_set`.
- `TensorLayout<R>`, `Rank<N>`, `DynRank`, `TensorRank`, `ShapeVec`,
  `StrideVec`, `SliceSpec` and checked layout validation. A zero slice step is
  invalid.

### Metadata-only views

Layouts describe shape, signed strides, and an offset into borrowed storage.
The view operations are metadata-only:

- `reshape_view`
- `transpose_view`
- `slice_view`

Views may be non-contiguous. `as_slice()` succeeds only when the view is
slice-contiguous for the borrowed storage. `TensorLayout` metadata slicing
supports signed strides and negative steps when reachable-range validation
proves every logical element maps inside the backing allocation. Zero step
remains invalid. Host views are `TypedTensorView<'a, T, R, Host>` and use the
same general reachable-range contract as every runtime view; the erased
external value `ErasedHostTensor` exposes only metadata permutation and
materialization.

---

## III. `tenferro-tensor`

`tenferro-tensor` is the runtime dense tensor crate. It reuses the core dtype
and scalar model, then adds runtime storage and backend placement.

The current typed runtime tensor shape is:

```rust
pub struct TypedTensor<T, R = DynRank, D: Representation = Dynamic> {
    shape: R::Shape,
    placement: Placement,
    storage: D::Storage<T, R>, // Host: Vec<T>; Gpu: group root; Dynamic: either
}
```

`Host` gives infallible host access, `Clone`, `Index`/`IndexMut` and owning
host mappings; `Gpu` and `Dynamic` expose checked host access. `into_host`,
`into_gpu` and `into_dynamic` convert between representations without copying;
a rejected narrowing returns the unchanged owner. The owned layout is derived
from the shape (column-major).

`Tensor` is the dynamic runtime enum over the supported scalar types:

- `F32`
- `F64`
- `I32`
- `I64`
- `Bool`
- `C32`
- `C64`
- `External(ErasedHostTensor, Placement)` for a scalar type no preset declares.
  The payload is an erased host tensor that carries its own shape, element
  strides, and offset, so a view over a caller-owned value is metadata-only and a
  projection back to the typed scalar is checked rather than a byte
  reinterpretation.

Runtime placement is explicit metadata:

```rust
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Managed,
    Other(String),
}

pub enum DeviceKind {
    Cpu,
    Gpu(GpuBackendKind),
    Other(String),
}

pub enum GpuBackendKind {
    Cuda,
    Rocm,
    Other(String),
}

pub struct Placement {
    pub memory_kind: MemoryKind,
    pub device: Option<DeviceId>,
}
```

Owned runtime tensors are compact column-major tensors. Arbitrary strides,
offsets, transposes, slices, and reverse layouts live on `TypedTensorView`,
`TypedTensorViewMut`, or `TensorLayout` metadata until an explicit
same-placement canonicalization boundary. Backend buffers are opaque to the
runtime tensor layer; the backend that owns the concrete handle is responsible
for downcasting and execution.

`tenferro-tensor` owns:

- runtime dense tensor types, including `TypedTensor<T, R = DynRank>` and
  dynamic-rank `Tensor`
- backend traits
- host/runtime views used by kernels

### DType conversion

Runtime dtype conversion has two public meanings:

- `convert(dtype)` is checked. It accepts conversions that are valid according
  to tenferro's dtype-promotion lattice and returns a typed error for lossy
  conversions such as float or complex to integer, complex to real, integer to
  boolean, or precision narrowing.
- `cast(dtype)` is explicit. It may perform lossy dtype projection and is the
  API callers use when they intentionally want truncation, precision narrowing,
  complex projection, or boolean truthiness.

The internal primitive and execution IR may continue to use the legacy
`Convert` operation name for dtype projection, including AD cotangent
projection, but public APIs must keep checked `convert` separate from explicit
lossy `cast`.

### Floating-point domain behavior

`F32` and `F64` scalar and elementwise operations preserve IEEE-style special
values wherever the backend can reasonably do so. Inputs at a mathematical
domain edge produce tensor values such as `NaN`, positive or negative infinity,
and signed zero rather than typed domain errors. This includes division and
remainder by zero: a nonzero finite value divided by signed zero produces the
correspondingly signed infinity, zero divided by zero produces `NaN`, and a
floating-point remainder with a zero divisor produces `NaN`. NaN inputs and
signed-zero results follow the operation's IEEE semantics.

Backends do not preflight-scan floating-point tensors for zero, non-finite, or
otherwise exceptional values. Integer operations retain their structured
domain checks because integer dtypes have no IEEE special values: integer
division and remainder by zero return `DivisionByZero`, and integer power with a
negative exponent returns its typed domain error. Structural failures remain
typed errors, including invalid shapes, axes, dtypes, indices, layouts, device
placement, backend capabilities, and operation configurations.

Complex operations follow the same principle where they are defined in terms
of IEEE floating-point components. Operation-specific complex behavior remains
governed by the relevant operation contract. CPU and CUDA must agree on result
classification and on signed-zero behavior where the operation makes the sign
bit contractual; any corner case where Rust, IEEE 754, NumPy, and JAX differ
must be specified explicitly.

Complex division bounds that agreement, and is therefore specified here: it is
not a componentwise IEEE operation, so the sign of a zero component of a quotient
depends on the division algorithm, and the host's own `f32` and `f64` paths do
not agree on it (`2.0 / (0.0 - 2.0i)` is `+0.0` for `C32` and `-0.0` for `C64`).
A complex division therefore has to agree on its classification and on every
non-zero component, but not on the sign of a zero component; add, subtract and
multiply keep that sign. An extreme but finite divisor is checked against its
closed-form value instead of the host, whose result for one element depends on
the machine and on the rest of the tensor (`2.0 / (1e38 + 1e38i)` is `0` on one
machine and a finite `1e-38` on another), so the complex/complex and mixed
real-scalar forms both run the scale-robust algorithm and keep the quotient
finite. A non-finite divisor is checked the same way, against the reference algorithm's value,
because the host is no more stable there: `2.0 / (inf + 1i)` is `NaN` on one machine and
`0` on another, while the reference gives a zero quotient for an infinite divisor and
`NaN` for a zero or `NaN` one.
is what the device reproduces, so the mixed form keeps its componentwise
expansion for a non-finite divisor and the host's `NaN` stays `NaN`.

`norm` with `ord = 0` is a corner case of the same kind, so it is specified here
too: it is the count of entries that are not equal to zero, which is not the same
predicate as `|x| > 0`. In IEEE 754 `NaN > 0` is false while `NaN != 0` is true,
so a `NaN` entry counts as nonzero; `+0.0` and `-0.0` both count as zero, and
`±inf` count as nonzero. The rule is identical on every norm surface
(concrete, borrowed read, typed, eager, and traced) and for real and complex
dtypes, where an entry is zero exactly when both of its components are zero.

CPU backend implementations, CPU kernels, and CPU resource pools belong in
`tenferro-cpu`. GPU backend implementations and GPU transfer helpers belong in
`tenferro-gpu`.

---

## IV. Data Model vs. Execution

The tensor data model does not own graph compilation, AD, or extension
registration.

- `tenferro-runtime` owns concrete tensor helpers, traced tensors, graph
  compilation/execution, extension runtime registration, and extension cache
  storage.
- `tenferro-ad` owns eager AD runtime surfaces and traced AD extension traits.
- `tenferro-einsum`, `tenferro-linalg`, and `tenferro-fft` own their public
  APIs, traced/eager helpers, extension runtimes, and optional AD rules.

Computation should be exposed as free functions, backend dispatch, runtime
execution, or extension runtimes. The tensor types should remain data and
metadata carriers.

There is no implicit CPU<->GPU transfer for user-visible backend operations.
Tensors must already be placed on the correct device for the backend call,
except for explicit upload/download helpers and internal execution conveniences
documented in [`backend-contract.md`](backend-contract.md).

---

## V. Dense Tensor Boundary

`tenferro_tensor::Tensor` is a dense runtime tensor. It does not carry structural
metadata such as diagonal, symmetric, block-diagonal, or sparse layout tags.

This is a deliberate boundary:

- structural variants cause a combinatorial expansion of operation cases
- core graph and execution IR remain easier to reason about when runtime
  tensors are logically dense
- extension crates can add structured algorithms without changing the base
  tensor enum

Structured values can be represented by external crates or higher-level
wrappers that store dense tensor leaves and call tenferro operations.
tenferro's runtime tensor remains the dense leaf type.

---

## VI. Einsum, Diagonal, and Repeated Labels

Trace, diagonal extraction, diagonal embedding, and tensor-network hyper-edge
patterns should be expressed through `tenferro-einsum` rather than by adding
structured tensor variants to the runtime tensor type.

Examples:

```text
einsum("ii->", A)              # trace
einsum("ii->i", A)             # diagonal extraction
einsum("i->ii", v)             # diagonal embedding
einsum("ik,k,kj->ij", U, s, V) # SVD-like reconstruction without dense diag
```

The operation semantics and contraction planning belong to `tenferro-einsum`.
The runtime tensor model only provides dense tensor operands and results.

---

## VII. Linalg Batch Convention

Linalg ops follow trailing-batch convention: core matrix dims are leftmost and
batch dims are rightmost. Shape `[M, N, B1, B2, ...]` means `B1*B2*...`
independent `M x N` matrices. Each batch slice is contiguous in column-major
memory, enabling zero-copy slicing.

This differs from JAX, NumPy, and PyTorch leading-batch convention
`[B, M, N]`. The choice matches tenferro's column-major storage: rightmost
dims have the largest stride, so trailing batch dims make each `[M, N]` slice a
contiguous block.

When `shape.len() == core_rank`, the op is a plain 2D call with zero overhead.

| Op | Input shape | Output shape(s) |
|---|---|---|
| `cholesky` | `[N, N, B...]` | `[N, N, B...]` |
| `svd` | `[M, N, B...]` | U `[M, K, B...]`, S `[K, B...]`, Vt `[K, N, B...]` |
| `qr` | `[M, N, B...]` | Q `[M, K, B...]`, R `[K, N, B...]` |
| `eigh` | `[N, N, B...]` | vals `[N, B...]`, vecs `[N, N, B...]` |
| `solve` | A `[N, N, B...]`, b `[N, M, B...]` | `[N, M, B...]` |

The trailing-batch convention also applies to `DotGeneral` / `BatchedGemm`
(documented in `AGENTS.md` under column-major dimension ordering).

---

## VIII. Composite Operations and `erf`

`erf` is a core primitive; the operations below it are composites of existing
primitives. Each is offered under the same name on the eager surface
(`EagerSession`), the traced surface (`TracedTensor`) and the concrete-session
surface (`TensorSessionOpsExt`; `TypedTensorSessionOpsExt` for all but
`take_along_axis`, which is an indexing operation). The formulation and the
edge-case policy live once in `tenferro-runtime/src/composite.rs`, so every
surface and backend computes the same primitive sequence; AD of the eager and
traced forms follows from the primitives' rules. The traced composites require
concrete input shapes (their scalar constants and reductions are broadcast
against known extents) and report an `InvalidArgument` validation error at
`GraphBuild` for a symbolic shape; the `erf` primitive itself accepts
symbolic shapes.

**`erf`** is defined for real `F32` and `F64` only. Complex, integer and `Bool`
input is a typed `UnsupportedDType` error on CPU and CUDA (and `Unsupported` at
`GraphBuild` on the traced surface). `erf(+-0) = +-0`, `erf(+-inf) = +-1`, and
`NaN` stays `NaN`. The CPU kernel uses `libm` (within 1 ulp of a
high-precision reference); CUDA uses the CUDA math library through CubeCL's
`Arithmetic::Erf` (within 2 ulp). The derivative is `2/sqrt(pi) * exp(-x^2)`.
CPU elementwise fusion declines regions containing `erf` (`strided_fused` has
no instruction for it), and the StableHLO lowering rejects it explicitly
(`erf` is the CHLO op `chlo.erf`).

**Activations** (real `F32`/`F64`; other dtypes are `UnsupportedDType`):

| Operation | Formulation | Edge-case policy |
|---|---|---|
| `sigmoid` | with `e = exp(-abs(x))`: `1/(1+e)` for `x > 0`, `e/(1+e)` otherwise | no intermediate overflows; `sigmoid(+-inf) = 1, 0`; derivatives finite everywhere, `sigmoid'(0) = 1/4`, `sigmoid''(0) = 0` |
| `silu` | `x * sigmoid(x)` | `silu(-inf)` is `-inf * 0 = NaN` (as PyTorch) |
| `softplus` | `max(x, 0) + log1p(exp(-abs(x)))` | never overflows; `softplus'(0) = 1/2`, `softplus''(0) = 1/4`; the derivative tends to `1` / `0` at large positive / negative `x` |
| `gelu` | `x/2 * (1 + erf(x/sqrt(2)))` (PyTorch `approximate="none"`) | the far negative tail loses relative accuracy to the cancellation in `1 + erf` (as PyTorch); `gelu(-inf)` is `NaN` |
| `gelu_tanh` | `x/2 * (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` (`approximate="tanh"`) | `gelu_tanh(-inf)` is `NaN`; the derivative is `NaN` once `x^2` overflows |

`-abs(x)` is written `select(x > 0, -x, x)`, so its derivative at `x = 0` is
`1`; this is what gives the exact derivatives at zero above.

**`reduce_mean(axes)`** divides the sum by the element count and is defined for
float and complex dtypes (integers and `Bool` are `UnsupportedDType`). `None`
reduces every axis and `Some(&[])` is the identity. A mean over zero elements
is `NaN` (0/0), with the reduced shape, as in NumPy and PyTorch.

**`softmax` / `log_softmax` / `masked_softmax` / `masked_log_softmax`** reduce
along one `axis` (real `F32`/`F64`). They subtract the slice maximum before
`exp`. The masked forms take a `Bool` mask that broadcasts to the input; a
`false` entry is treated as `-inf`. The policy:

- a masked-out entry is `0` (softmax) or `-inf` (log-softmax), and its
  gradient is exactly `0`, even when its value is `NaN` or infinite;
- a slice with no participating entry (all masked out, or all `-inf` in the
  unmasked form) is all `0` / all `-inf` instead of the `NaN` of the naive
  `-inf - (-inf)`. Its gradient is finite, and in the masked forms it is `0`.
  The implementation clamps the slice maximum to the dtype's lowest finite
  value and the slice sum to its smallest positive normal value (both with a
  NaN-propagating `maximum`); neither clamp changes a participating slice;
- a participating `NaN` or `+inf` makes the whole slice `NaN`; other slices are
  unaffected;
- a zero-length `axis` returns an empty result of the input shape.

**`layer_norm` / `rms_norm`** normalize along one `axis` (real `F32`/`F64`):
`(x - mean) / sqrt(var + eps)` with the biased variance of the centered values,
and `x / sqrt(mean(x^2) + eps)`, then `* weight + bias`. `weight` and `bias` are
optional rank-1 tensors of length `shape[axis]` and the input's dtype; `eps`
must be finite and non-negative (`InvalidArgument` otherwise). A zero-variance
(layer) or all-zero (RMS) slice normalizes to `0`, then `bias`, with a finite
gradient when `eps > 0`; with `eps = 0` it is `0/0 = NaN`, as in PyTorch. A
zero-length `axis` returns an empty result. These are the composed forms; a
fused CPU kernel is tracked separately (#2006).

**`take_along_axis(indices, axis)`** follows NumPy:
`out[.., i, ..] = x[.., indices[.., i, ..], ..]` along `axis`. `indices`
(`I32` or `I64`) has the input's rank; every other dimension of `indices` is
either the input's extent (a batch dimension, indexed per element) or `1`
(broadcast: the whole extent is taken). The output has the input's shape with
`indices.shape[axis]` along `axis`. It is one `gather` whose index tuples pair
each index with its batch coordinates; the coordinates are built on the
backend from scalar constants, so no host index data is transferred. Indices
must be in `[0, shape[axis])`; out-of-range indices follow `gather` (see the
indexing bounds contract in `ad-contract.md`). The gradient flows to `x` only.
Index views (`TensorRead`) are not accepted yet (#1930).

---

## IX. Source of Truth

Current implementation ownership:

- `crates/tenferro-tensor-core/src/lib.rs` for dtype, scalar and layout
  metadata
- `crates/tenferro-tensor/src/default_scalars.rs`, `scalar_set.rs` and
  `erased_host.rs` for the scalar-set layer and the erased external value,
  all built on `TypedTensor<T, DynRank, Host>`
- `crates/tenferro-tensor/src/types.rs` for runtime dense tensor storage and
  placement metadata
- `crates/tenferro-tensor/src/backend.rs` for backend traits
- `crates/tenferro-runtime/src/*` for graph execution and extension runtime dispatch
- `crates/tenferro-runtime/src/composite.rs` for the composite operations

If this document conflicts with those files, the implementation wins and this
document should be updated.
