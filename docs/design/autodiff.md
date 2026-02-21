# Automatic Differentiation

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
tenferro-linalg          ← Linalg + linalg AD rules (depends on chainrules)
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

- `tracked_einsum(subscripts, operands)` — reverse-mode einsum
- `dual_einsum(subscripts, operands)` — forward-mode einsum
- `einsum_rrule(subscripts, operands, cotangent)` — local pullback for FFI/manual AD
- `einsum_frule(subscripts, primals, tangents)` — local pushforward for FFI/manual AD
- `einsum_hvp(subscripts, primals, tangents, cotangent, cotangent_tangent)` — local HVP

**Linalg AD (in `tenferro-linalg`):**

- `tracked_svd`, `dual_svd`, `svd_rrule`, `svd_frule` — SVD AD
- `tracked_qr`, `dual_qr`, `qr_rrule`, `qr_frule` — QR AD
- `tracked_lu`, `dual_lu`, `lu_rrule`, `lu_frule` — LU AD
- `tracked_eigen`, `dual_eigen`, `eigen_rrule`, `eigen_frule` — Eigen AD

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
| `UnaryOp` enum (`Negate`, `Reciprocal`, `Abs`, `Sqrt`) | SVD/eigen rrule requires F-matrix: `F_ij = 1/(σ_j² − σ_i²)`, needing `Reciprocal`. `Sqrt` for matrix square root. |
| `PrimDescriptor::ElementwiseUnary { op }` | Maps to `cutensorElementwiseTrinary` (unary case) on GPU. |

`Square` (x²) deliberately omitted — expressible as `ElementwiseMul(x, x)`.

**Added to tenferro-tensor:**

| Addition | Rationale |
|----------|-----------|
| `Tensor::select(dim, index)` | Zero-copy view for batch-dimension manipulation in AD rules. |
| `Tensor::narrow(dim, start, length)` | Zero-copy slicing. Required for matrix block extraction. |
| `Tensor::eye(n, memory_space, order)` | Identity matrix. Required by SVD rrule projector: `(I − U·Uᵀ)`. |
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
`tenferro-tensor`, `tenferro-algebra`, `tenferro-device`, `chainrules-core`)
to directly compose prims operations in AD rule implementations.

---

## Algebra and Tropical Support

AD must remain algebra-aware:

- Standard arithmetic: direct rrule/frule formulas over `+/*`.
- Tropical algebra (`tenferro-tropical`): may need algebra-specific state
  (e.g., argmax path information for max-plus variants).
- API design relies on `HasAlgebra` and `TensorPrims<A>` for extensibility.

See [algebra.md](./algebra.md) for the algebra system design.

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
