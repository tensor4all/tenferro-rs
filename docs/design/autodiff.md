# Automatic Differentiation

For detailed mathematical derivations of AD rules, see [AD Formula Notes](../AD/index.md).

**Complex gradient convention:** All AD formula notes use the Wirtinger
(CR-calculus) convention for complex-valued tensors. VJP formulas produce
$\partial\ell / \partial \bar{A}$ (conjugate variable), which yields
$A^{-\mathsf{H}}$ (not $A^{-\mathsf{T}}$) and conjugated scalar prefactors.
This matches PyTorch's convention and ensures consistency across all
linalg rrule implementations.

## Position in Workspace Architecture

The AD system is split into two crates following Rust convention
(`foo-core` = traits, `foo` = full library):

- **`chainrules-core`** — Pure trait definitions (like Julia's ChainRulesCore.jl).
  Defines `Differentiable`, `ReverseRule<V>`, `ForwardRule<V>`, error types,
  `NodeId`, `SavePolicy`. Depends only on `thiserror`.
- **`chainrules`** — AD engine (like Zygote.jl). Provides `TrackedTensor<V>`,
  `DualTensor<V>`, `pullback()`, `hvp()`, `Gradients<V>`, `PullbackPlan<V>`.
  Depends only on `chainrules-core`. Re-exports all of `chainrules-core`.

Neither crate depends on any tensor or tenferro crate.

```
chainrules-core          ← Core AD traits (Differentiable, no tensor deps)
    ↑
chainrules               ← AD engine (TrackedTensor, pullback, hvp)
    ↑
tenferro-tensor          ← impl Differentiable for Tensor<T> (depends on chainrules-core)
    ↑
tenferro-einsum          ← Einsum + einsum AD rules (depends on chainrules)
tenferro-linalg          ← Linalg + linalg AD rules (depends on chainrules-core)
```

Operation-specific AD rules live with their operations:
- Einsum AD in `tenferro-einsum`
- Linalg AD in `tenferro-linalg`
- Future operations in their own crates

---

## API Layers

### 1. Core AD Traits (chainrules-core)

- `Differentiable` — trait defining tangent space (zero_tangent, accumulate_tangent)
- `ReverseRule<V>`, `ForwardRule<V>` — rule extension traits
  (named after Julia's ChainRules.jl: rrule/frule)
  (`ReverseRule` includes `pullback_with_tangents` for HVP support)
- `AutodiffError`, `AdResult`, `NodeId`, `SavePolicy`

### 2. AD Engine (chainrules)

- `Tape<V>` — explicit tape (TensorFlow GradientTape style)
- `TrackedTensor<V>` — reverse-mode wrapper (with optional tangent for HVP)
- `DualTensor<V>` — forward-mode wrapper
- `Tape::pullback(loss)` — reverse-mode execution
- `Tape::hvp(loss)` — forward-over-reverse HVP execution
- `Gradients<V>`, `PullbackPlan<V>`, `HvpResult<V>` — result and plan types

All types parameterized by `V: Differentiable`, making the framework
independent of any specific tensor type.

### 3. Operation-Specific AD Rules

**Einsum AD (in `tenferro-einsum`):**

All einsum AD functions take `ctx: &mut B::Context` and `B: TensorPrims<A>`,
matching the non-AD einsum functions.

- `tracked_einsum(ctx, subscripts, operands)` — reverse-mode einsum
- `dual_einsum(ctx, subscripts, operands)` — forward-mode einsum
- `einsum_rrule(ctx, subscripts, operands, cotangent)` — local pullback for FFI/manual AD
- `einsum_frule(ctx, subscripts, primals, tangents)` — local pushforward for FFI/manual AD
- `einsum_hvp(ctx, subscripts, primals, tangents, cotangent, cotangent_tangent)` — local HVP

**Linalg AD (in `tenferro-linalg`):**

Linalg uses only stateless `_rrule`/`_frule` functions — no `tracked_*` or
`dual_*` wrappers. The chainrules tape engine composes `permute_backward` +
`reshape_backward` + `svd_rrule` etc. via the standard chain rule automatically.

- `svd_rrule`, `svd_frule` — SVD AD
- `qr_rrule`, `qr_frule` — QR AD
- `lu_rrule`, `lu_frule` — LU AD
- `eigen_rrule`, `eigen_frule` — Eigen AD

---

## Design Decisions

1. **Separate reverse and forward wrappers** (`TrackedTensor` vs `DualTensor`).
2. **Local rrule/frule callable without a tape.**
3. **Backend-neutral public APIs.** Backend-specific execution stays in
   `tenferro-prims` / device layer.
4. **AD framework does not depend on operation crates.** Each operation
   crate owns its AD rules. Avoids circular dependencies.
5. **`Differentiable` does not require `Clone` on the primal type.**
   Only `Tangent: Clone` is required (for gradient accumulation at fan-out
   nodes). The AD engine avoids cloning primals by taking ownership.
6. **Explicit tape (TensorFlow GradientTape style).**
   Users create a tape, register leaf values via `tape.leaf()`, and compute
   gradients via `tape.pullback()`. `TrackedTensor` holds a reference to
   its tape, so the tape cannot be dropped while tracked values exist.
   Multiple independent tapes are supported (nested differentiation).

---

## Minimal Feature Extensions for Linalg AD

Implementing linalg AD rules (Mathieu 2019 et al.) requires operations
beyond the original `tenferro-prims` and `tenferro-tensor` API. Gap analysis
against libtorch's AD formulas led to a minimal design.

**Added to tenferro-prims:**

| Addition | Rationale |
|----------|-----------|
| `UnaryOp` enum (`Negate`, `Reciprocal`, `Abs`, `Sqrt`) | SVD/eigen rrule requires F-matrix: $F_{ij} = 1/(\sigma_j^2 - \sigma_i^2)$, needing `Reciprocal`. `Sqrt` for matrix square root. |
| `PrimDescriptor::ElementwiseUnary { op }` | Maps to `cutensorElementwiseTrinary` (unary case) on GPU. |

`Square` (x²) deliberately omitted — expressible as `ElementwiseMul(x, x)`.

**Added to tenferro-tensor:**

| Addition | Rationale |
|----------|-----------|
| `Tensor::select(dim, index)` | Zero-copy view for batch-dimension manipulation in AD rules. |
| `Tensor::narrow(dim, start, length)` | Zero-copy slicing. Required for matrix block extraction. |
| `Tensor::eye(n, memory_space, order)` | Identity matrix. Required by SVD rrule projector: $(I - U \cdot U^\mathsf{T})$. |
| `Tensor::tril(diagonal)` / `triu(diagonal)` | Triangular extraction. Required by QR/LU rrule. |

**Excluded (with rationale):**

| Candidate | Why excluded |
|-----------|--------------|
| `transpose` / `squeeze` / `unsqueeze` | Trivially derivable from `permute` / `reshape` |
| `full` / `empty` / `arange` / `rand` | Composable from `from_vec` |
| `solve_triangular` | Deferred to P1 |

---

## SVD rrule Algorithm Structure

The `svd_rrule` function body documents the Mathieu 2019 algorithm:

```
Step 1: Forward pass (U, S, Vt) = svd(A)         — cached by caller
Step 2: Build F-matrix (F_ij = 1/(σ_j²−σ_i²))   — ElementwiseMul, ElementwiseUnary(Reciprocal)
Step 3: Compute Uᵀ·dU                             — BatchedGemm
Step 4: Symmetrize: M = Uᵀ·dU − (Uᵀ·dU)ᵀ        — Permute (zero-copy), alpha/beta
Step 5: Hadamard product: F ⊙ M                   — ElementwiseMul
Step 6: Add diagonal dS: F⊙M + diag(dS)          — AntiTrace
Step 7: Assemble: dA = U·(F⊙M + diag(dS))·Vt    — BatchedGemm (×2)
Step 8: Projector term (m > n):
        dA += (I − U·Uᵀ)·dU·diag(1/S)·Vt        — eye, BatchedGemm, ElementwiseUnary(Reciprocal)
```

This demonstrates all linalg AD rules are expressible through `tenferro-prims`
plus the minimal tensor-level additions above.

### tenferro-linalg Dependency

`tenferro-linalg` depends on `tenferro-prims` (in addition to
`tenferro-tensor`, `tenferro-algebra`, `tenferro-device`, `chainrules-core`).
Note: `tenferro-linalg` depends on `chainrules-core` (not full `chainrules`)
because it only uses the `Differentiable` trait and `AdResult` type — it does
not create tapes or tracked tensors.

---

## Algebra and Tropical Support

AD must remain algebra-aware:

- **Standard arithmetic** (`Standard<T>`): direct rrule/frule formulas over
  $+/\times$. The algebra type `A = Standard<T>` determines which primitive
  operations are available (e.g., cuTENSOR-backed `TensorPrims<Standard<T>>`).
- **Tropical algebra** (`tenferro-tropical`): requires algebra-specific state
  during the backward pass. For max-plus/min-plus einsum, the rrule must track
  argmax indices — the positions of the winning elements that achieved the
  tropical sum. These indices are not computable from the output alone and must
  be saved during the forward pass.

  **Argmax tie-break rule**: When multiple elements share the maximum (or minimum)
  value, the element with the **smallest linear index** (row-major traversal order)
  wins. This rule is deterministic and must be consistent across CPU, GPU, and C-API
  backends.

  **GPU tropical AD note:** On GPU, tropical backward requires argmax-capable
  custom kernels that are distinct from cuTENSOR/hipTensor primitives. cuTENSOR
  operates on ring algebras (multiply-add); it has no native support for
  max-plus argmax index tracking. A GPU tropical backward pass therefore
  requires a separately implemented custom kernel (e.g., via CUDA/HIP or a
  vendor-agnostic compute shader). This is a non-trivial infrastructure
  requirement and must be planned separately from standard einsum GPU support.

**Role of `HasAlgebra` and `TensorPrims<A>`:**

- `HasAlgebra` (on `Tensor<T>`) is an **inference convenience trait** — it
  allows the compiler to infer `A` from `T` for the common case, avoiding
  explicit algebra type annotations at call sites.
- The AD math and rule contracts (rrule/frule signatures, cotangent types)
  depend on the algebra type `A` directly. `HasAlgebra` does not affect
  correctness; it only reduces boilerplate for users who do not need a custom
  algebra.

See [algebra.md](./algebra.md) for the algebra system design.

---

## Error Contract for Unsupported AD Modes

When an operation does not support a particular AD mode, it returns
`AutodiffError::ModeNotSupported` rather than a generic error string.
This allows callers — including C-API / FFI layers — to branch on the
error variant without parsing error messages.

```rust
use chainrules_core::AutodiffError;

let err = AutodiffError::ModeNotSupported {
    mode: "frule".into(),
    reason: "tropical einsum supports rrule only (max is not smooth)".into(),
};
```

**Operations that return `ModeNotSupported`:**

| Operation | Unsupported modes | Reason |
|-----------|-------------------|--------|
| Tropical einsum (`MaxPlus`, `MinPlus`, `MaxMul`) | `frule` (JVP) | `max`/`min` is not smooth; JVP is undefined |
| Tropical einsum (`MaxPlus`, `MinPlus`, `MaxMul`) | `hvp` | Requires a smooth frule; not available |

For standard arithmetic algebra, all three modes (`rrule`, `frule`, `hvp`)
are supported where implemented.

---

## Forward-Mode AD Redesign: libtorch-Aligned Tangent-in-Tensor

> **Status:** Planned redesign. Current POC uses separate `DualTensor<V>` and
> per-rule `pullback_with_tangents`. This section describes the target design.

### Motivation

The current design has two problems:

1. **Performance:** `EinsumReverseRule::pullback` uses `manual_einsum` (naive
   multi-index loop) instead of the optimized backend. This is too slow for
   production use.
2. **HVP architecture:** Each `ReverseRule` must explicitly implement
   `pullback_with_tangents` for HVP. This per-rule approach does not scale
   and diverges from PyTorch/JAX where HVP is computed by composing
   `jvp(grad(f))`.

### Target Design (libtorch-aligned)

Align with libtorch's C++ forward-mode AD architecture
(`torch/csrc/autograd/forward_grad.h`, `autograd_meta.cpp`):

**Core principle:** `Tensor<T>` carries optional forward-mode tangent metadata.
Operations detect tangent presence at runtime and propagate automatically.
No separate `DualTensor` type needed. `pullback_with_tangents` not needed.

#### 1. Tangent-in-Tensor

```
Tensor<T>
  ├── buffer, dims, strides, offset, ...  (existing)
  └── fw_grad: Option<Box<Tensor<T>>>     (NEW: forward-mode tangent)
```

libtorch stores `shared_ptr<ForwardGrad>` with a `map<level, Tensor>` to
support nested forward AD. For the initial implementation, single-level
(level 0 only) is sufficient. Multi-level can be added when needed.

Tensor API additions:

- `fw_grad(&self) -> Option<&Tensor<T>>`
- `set_fw_grad(&mut self, grad: Tensor<T>)`
- `has_fw_grad(&self) -> bool`
- `detach_fw_grad(&mut self) -> Option<Tensor<T>>`

#### 2. Automatic Tangent Propagation in Operations

Each operation (einsum, prims ops) checks input tensors for `fw_grad` and
computes the output tangent using the operation's JVP rule:

```
fn einsum(..., primals) -> Tensor<T> {
    let output = backend_einsum(primals);
    if any input has fw_grad {
        let tangents = extract_fw_grads(primals);
        let output_tangent = einsum_frule(subscripts, primals, &tangents);
        output.set_fw_grad(output_tangent);
    }
    output
}
```

This matches libtorch where each op's derivative formula checks
`_fw_grad(level)` on inputs and calls `_set_fw_grad` on outputs.

#### 3. HVP via Composition (not per-rule)

With tangent-in-tensor, HVP becomes:

1. Seed leaf tensors with `fw_grad` (tangent direction v)
2. Run forward pass — tangents propagate automatically through all ops
3. Run `tape.pullback(loss)` — pullback calls einsum internally
4. Those einsum calls see `fw_grad` on their inputs (cotangents carry
   tangent metadata from the forward pass) and propagate automatically
5. Result: both gradient and HVP emerge without `pullback_with_tangents`

This matches PyTorch's `jvp(grad(f))` pattern and JAX's composability.

#### 4. Backend-Optimized Pullback

`EinsumReverseRule` stores `Rc<RefCell<B::Context>>` (or equivalent) to
execute pullback via the optimized backend instead of `manual_einsum`.

The `ReverseRule` trait signature (`pullback(&self, cotangent)`) does not
need `ctx` — the concrete rule struct stores it internally, just as
libtorch's backward functions capture saved tensors and context.

#### 5. Deprecation Path

| Current | Target | Migration |
|---------|--------|-----------|
| `DualTensor<V>` | `Tensor<T>` with `fw_grad` | Thin wrapper or remove |
| `pullback_with_tangents` | Automatic via tangent-in-tensor | Remove from `ReverseRule` trait |
| `Tape::hvp` (explicit) | Composition: seed `fw_grad` + `pullback` | Simplify or keep as convenience |
| `manual_einsum` in pullback | Backend-optimized einsum | Store ctx in rule |
| `einsum_hvp` (standalone) | Automatic via propagation | Keep for FFI/manual use |

### Key Differences from libtorch

| Aspect | libtorch | tenferro-rs (target) |
|--------|----------|---------------------|
| Tangent storage | `map<level, Tensor>` (multi-level) | `Option<Box<Tensor<T>>>` (single-level initially) |
| Dispatch | Runtime dispatcher (CPU/CUDA/autograd/forward_ad) | Static dispatch via generics + runtime `fw_grad` check |
| Thread safety | `mutex` on `ForwardGrad` | Single-threaded initially; `Rc<RefCell>` |
| Type system | Single `at::Tensor` type for everything | `Tensor<T>` generic over scalar type |

### Implementation Phases

**Phase 1:** Add `fw_grad` field to `Tensor<T>`, add accessor API.
Existing code unaffected (field defaults to `None`).

**Phase 2:** Wire tangent propagation into einsum. `einsum` auto-propagates
`fw_grad` using existing `einsum_frule`. Test with `dual_einsum` equivalence.

**Phase 3:** Store backend context in `EinsumReverseRule`, replace
`manual_einsum` with optimized backend calls. Fix pullback performance.

**Phase 4:** Remove `pullback_with_tangents` from `ReverseRule` trait.
Simplify `Tape::hvp`. Deprecate `DualTensor` in favor of tangent-in-tensor.

**Phase 5:** Wire tangent propagation into prims ops and linalg.

---

## Scope

**Current POC:**
- `chainrules-core`: `Differentiable`, `ReverseRule<V>`, `ForwardRule<V>`,
  error types, `NodeId`, `SavePolicy`
- `chainrules`: `Tape<V>`, `TrackedTensor<V>`, `DualTensor<V>`, HVP
- `tenferro-einsum`: `tracked_einsum`, `dual_einsum`, `einsum_rrule`,
  `einsum_frule`, `einsum_hvp`

**Out of scope:**
- Full runtime graph scheduling
- End-to-end optimized GPU pullback kernels
- Full second-order differentiation API beyond HVP
