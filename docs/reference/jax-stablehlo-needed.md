# JAX/StableHLO Primitives Needed For tenferro

**Date:** 2026-04-03
**Status:** Draft
**Parent:** `../README.md`
**Related:** `jax-primitives.md`, `stablehlo-primitives.md`, `../spec/primitive-catalog.md`

---

## I. Goal

This note answers a narrower question than `../spec/primitive-catalog.md`:

> If tenferro v2 wants to reuse JAX's `linearize` design with minimal changes,
> which JAX primitives must tenferro expose, and which StableHLO ops are
> needed below that layer?

The goal here is **not** to design a backend-minimal semiring substrate.
`Semiring Core` is intentionally ignored in this document.

The target feature set for this first pass is:

- `einsum`
- `lu`
- `solve`
- `qr`
- `svd`

Both **primal execution** and **forward AD / `linearize` support** are in
scope.

---

## II. Sources

This note is based on:

- current `tenferro-rs` code in `~/tensor4all/tenferro-rs`
- current JAX code in `~/tensor4all/jax`
- the StableHLO spec summarized in `stablehlo-primitives.md` (in this directory)

The most relevant source files were:

- `tenferro-rs/docs/design/supported-ops.md`
- `tenferro-rs/docs/design/tensor-prims.md`
- `tenferro-rs/tenferro-prims/src/families/*.rs`
- `tenferro-rs/tenferro-linalg/src/lib.rs`
- `tenferro-rs/tenferro-linalg/src/prims_bridge.rs`
- `jax/jax/_src/interpreters/ad.py`
- `jax/jax/_src/numpy/einsum.py`
- `jax/jax/_src/lax/control_flow/solves.py`
- `jax/jax/_src/lax/linalg.py`
- `jax/jax/_src/numpy/linalg.py`

---

## III. Phase-1 StableHLO Target Set

The first concrete implementation target is **not** the JAX primitive list.
Those JAX-like primitives will all be new v2 primitives anyway.

What actually has to run at the backend layer is a StableHLO-level vocabulary.
For the target feature set (`einsum`, `lu`, `solve`, `qr`, `svd`), the
recommended phase-1 StableHLO target set is:

### Direct StableHLO tensor ops

- elementwise / scalar:
  `constant`, `convert`, `add`, `subtract`, `multiply`, `divide`, `negate`,
  `compare`, `select`, `abs`, `real`, `imag`, `complex`, `power`, `atan2`,
  `sqrt`, `rsqrt`, `exponential`, `exponential_minus_one`, `log`,
  `log_plus_one`, `sine`, `cosine`, `tan`, `tanh`, `ceil`, `maximum`,
  `minimum`
- shape / indexing / reduction:
  `iota`, `broadcast_in_dim`, `reshape`, `transpose`, `pad`, `reduce`,
  `slice`, `dynamic_slice`, `dynamic_update_slice`, `gather`, `scatter`,
  `sort`
- contraction / control / linalg:
  `dot_general`, `triangular_solve`, `while`, `custom_call`

### What is intentionally not part of the direct StableHLO target set

These currently have **no direct StableHLO op** and should therefore remain
either composites or `custom_call`-backed operations:

- `lu`
- `qr`
- `svd`
- `solve`
- `trace`
- `diag`
- `conj`
- `asin`, `acos`, `atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`
- `hypot`
- `xlogy`
- `topk`
- `var`
- `std`

---

## IV. Current tenferro-rs Inventory

### Tensor/view layer

Current `tenferro-tensor` already has the view/metadata operations that JAX
usually treats as tensor primitives:

- `reshape`
- `permute`
- `broadcast`
- `diagonal`
- `select`
- `narrow`
- `contiguous`
- `tril`
- `triu`
- `eye`

These exist today, but they are **tensor/view APIs**, not first-class v2
tensor primitives with their own `linearize` / `transpose_rule`
implementations.

### Execution families in `tenferro-prims`

Current `tenferro-prims` is organized by backend execution families:

- `TensorSemiringCore`
  - `BatchedGemm`
  - `ReduceSum`
  - `Trace`
  - `AntiTrace`
  - `AntiDiag`
  - `MakeContiguous`
- `TensorSemiringFastPath`
  - `Contract`
  - `ElementwiseBinary::{Add, Mul}`
- `TensorScalarPrims`
  - unary: `Neg`, `Conj`, `Abs`, `Reciprocal`, `Real`, `Imag`, `Square`
  - binary: `Add`, `Sub`, `Mul`, `Div`, `Maximum`, `Minimum`,
    `Greater`, `GreaterEqual`, `ClampMin`, `ClampMax`
  - ternary: `Where`
  - reductions: `Sum`, `Prod`, `Mean`, `Max`, `Min`
- `TensorAnalyticPrims`
  - unary: `Sqrt`, `Rsqrt`, `Exp`, `Expm1`, `Ceil`, `Log`, `Log1p`, `Sin`,
    `Cos`, `Tan`, `Tanh`, `Asin`, `Acos`, `Atan`, `Sinh`, `Cosh`, `Asinh`,
    `Acosh`, `Atanh`
  - binary: `Pow`, `Atan2`, `Hypot`, `Xlogy`
  - reductions: `Var`, `Std`

This is already a substantial operation inventory, but it is **not aligned to
the JAX primitive layer**. It groups by backend family rather than by JAX-style
primitive name such as `dot_general_p`, `qr_p`, or `svd_p`.

### Structured linalg in `tenferro-linalg`

Current `tenferro-linalg` already has backend kernels and public APIs for the
target structured operations:

- backend-facing kernels:
  - `lu`
  - `solve`
  - `solve_triangular`
  - `qr`
  - `svd`
- public primal APIs:
  - `lu`
  - `solve`
  - `solve_triangular`
  - `qr`
  - `svd`
- stateless AD rules already exist for:
  - `lu`
  - `solve`
  - `solve_triangular`
  - `qr`
  - `svd`

So tenferro can already execute and differentiate the target math. The mismatch
is not mathematical coverage. The mismatch is the **primitive boundary** used
to express that coverage.

---

## V. Correspondence: Current tenferro -> StableHLO

This is the table that matters for implementation planning.

### Structural tensor/view operations

| Current tenferro surface | Recommended StableHLO target | Direct? | Note |
|---|---|---:|---|
| `reshape` | `reshape` | yes | Use `dynamic_reshape` only if runtime shapes become part of the contract |
| `permute` | `transpose` | yes | Direct axis permutation |
| `broadcast` | `broadcast_in_dim` | yes | Explicit broadcast, no implicit NumPy-style rules |
| `select` / `narrow` | `slice`, `dynamic_slice`, `gather` | composite | Depends on whether indices are static or dynamic |
| `diagonal` | `iota + compare + select + reduce` or `gather` | no | No direct StableHLO diagonal op |
| `tril` / `triu` | `iota + compare + select` | no | Masked triangular extraction |
| `eye` | `iota + compare + convert` | no | Construct from index equality |
| `contiguous` | none | no | Layout/materialization concern, not a semantic StableHLO op |

### Current semiring/scalar/analytic families

| Current tenferro primitive | Recommended StableHLO target | Direct? | Note |
|---|---|---:|---|
| `TensorSemiringCore::BatchedGemm` | `dot_general` | yes | Main contraction target |
| `TensorSemiringCore::ReduceSum` | `reduce` with add combiner | yes | StableHLO has no dedicated `reduce_sum` op |
| `TensorSemiringCore::Trace` | `iota + compare + select + reduce` | no | No direct trace op |
| `TensorSemiringCore::AntiTrace` | `scatter` + add-style combiner | no | AD helper lowering |
| `TensorSemiringCore::AntiDiag` | `scatter` or masked `select` | no | AD helper lowering |
| `TensorSemiringCore::MakeContiguous` | none | no | Backend/layout concern |
| `TensorSemiringFastPath::Contract` | `dot_general` or `multiply + reduce` | mostly | Depends on the contraction pattern |
| `TensorSemiringFastPath::ElementwiseBinary::{Add,Mul}` | `add`, `multiply` | yes | Direct elementwise lowering |
| `ScalarUnary::{Neg,Abs,Real,Imag}` | `negate`, `abs`, `real`, `imag` | yes | Direct scalar ops |
| `ScalarUnary::Conj` | `real + imag + negate + complex` | no | No direct StableHLO `conj` op |
| `ScalarUnary::Reciprocal` | `divide` | no | Lower as `1 / x` |
| `ScalarUnary::Square` | `multiply` | no | Lower as `x * x` |
| `ScalarBinary::{Add,Sub,Mul,Div}` | `add`, `subtract`, `multiply`, `divide` | yes | Direct elementwise lowering |
| `ScalarBinary::{Maximum,Minimum}` | `maximum`, `minimum` | yes | Direct elementwise lowering |
| `ScalarBinary::{Greater,GreaterEqual}` | `compare` | yes | Comparison direction in attributes |
| `ScalarBinary::{ClampMin,ClampMax}` | `maximum`, `minimum` | yes | Binary clamp lowers to max/min |
| `ScalarTernary::Where` | `select` | yes | Predicate-controlled selection |
| `ScalarReduction::{Sum,Prod,Max,Min}` | `reduce` | yes | Combiner decides semantics |
| `ScalarReduction::Mean` | `reduce + divide` | no | Count/normalization handled outside reduce |
| `AnalyticUnary::{Sqrt,Rsqrt,Exp,Expm1,Ceil,Log,Log1p,Sin,Cos,Tan,Tanh}` | `sqrt`, `rsqrt`, `exponential`, `exponential_minus_one`, `ceil`, `log`, `log_plus_one`, `sine`, `cosine`, `tan`, `tanh` | yes | Direct StableHLO ops exist |
| `AnalyticUnary::{Asin,Acos,Atan,Sinh,Cosh,Asinh,Acosh,Atanh}` | composite or `custom_call` | no | No direct StableHLO ops |
| `AnalyticBinary::Pow` | `power` | yes | Direct elementwise lowering |
| `AnalyticBinary::Atan2` | `atan2` | yes | Direct elementwise lowering |
| `AnalyticBinary::{Hypot,Xlogy}` | composite or `custom_call` | no | No direct StableHLO ops |
| `AnalyticReduction::{Var,Std}` | composite | no | Lower through reduce/add/multiply/subtract/sqrt |

### Current indexing / metadata / sort / complex bridge families

| Current tenferro primitive | Recommended StableHLO target | Direct? | Note |
|---|---|---:|---|
| `IndexSelect` | `gather` | yes | Slice selection by index tensor |
| `Gather` | `gather` | yes | Indexed read |
| `Scatter` | `scatter` | yes | Indexed write / accumulate |
| `IndexPut` | `scatter` | yes | Overwrite or accumulate mode |
| `MetadataGenerate::IotaStartZero` | `iota` | yes | Metadata-side range generation |
| `MetadataGenerate::Constant` | `constant` | yes | Integer/bool constant tensor |
| `MetadataBinary::{Equal,NotEqual}` | `compare` | yes | Bool/int comparison |
| `MetadataBinary::{Add,Sub,Mul,BitAnd}` | `add`, `subtract`, `multiply`, `and` | yes | Integer/bool metadata ops |
| `MetadataTernary::Where` | `select` | yes | Metadata mask select |
| `MetadataReduction::{Sum,All,Any}` | `reduce` | yes | Add / and / or combiner |
| `MetadataCast::PointwiseCast` | `convert` | yes | Metadata-to-scalar cast |
| `MetadataCast::Where` | `select` | yes | Bool metadata as condition |
| `Sort` / `Argsort` | `sort` | yes | Pair/tuple sort when indices are needed |
| `Topk` | `sort + slice` | no | No direct StableHLO `top_k` op |
| `ComplexReal::{Abs,Real,Imag}` | `abs`, `real`, `imag` | yes | Direct lowerings |
| `ComplexScale::PointwiseMul` | `complex + multiply` | no | Real rhs must be promoted to complex first |
| `Rng::{Uniform,Normal,Integer}` | `rng_bit_generator` or legacy `rng` | partial | Not part of the main phase-1 AD target |

### Current linalg kernels

| Current tenferro primitive | Recommended StableHLO target | Direct? | Note |
|---|---|---:|---|
| `solve_triangular` | `triangular_solve` | yes | Best direct match |
| `lu` | `custom_call` | no | No direct StableHLO LU op |
| `qr` | `custom_call` | no | No direct StableHLO QR op |
| `svd` | `custom_call` | no | No direct StableHLO SVD op |
| `solve` | composite over `custom_call(lu)` + `triangular_solve` + shape ops | no | If JAX-like AD boundary is needed, wrap as higher-level `linear_solve_p` |

---

## VI. JAX-like Primitive Layer To Add Above StableHLO

These are the **new v2 primitives** to add so that tenferro is close enough to
JAX for `linearize`-style formulas, but they should be read as a layer
*above* the StableHLO implementation target.

AD helpers:

- ~~`add_jaxvals_p` / `add_any`~~ — not needed in v2; cotangent accumulation
  uses the standard `Add` primitive
- ~~`zeros_like_p`~~ — not needed in v2; zero propagation is handled via
  `Option<LocalValId>` (None = zero tangent) at the graph level
- `stop_gradient_p`

Tensor primitives:

- arithmetic:
  `add_p`, `sub_p`, `mul_p`, `div_p`, `neg_p`
- complex/real helpers:
  `conj_p`, `real_p`
- contraction/reduction/shape:
  `dot_general_p`, `reduce_p`, `broadcast_in_dim_p`, `reshape_p`,
  `transpose_p`, `squeeze_p`, `pad_p`
- predicate/select/index helpers:
  `select_n_p`, `eq_p`, `iota_p`, `sort_p`, `convert_element_type_p`

Structured linalg and control primitives:

- `lu_p`
- `triangular_solve_p`
- `qr_p`
- `svd_p`
- `linear_solve_p` (`custom_linear_solve`)

Important consequences:

- `einsum` stays a composite over tensor primitives
- `solve` stays a composite over `linear_solve_p`, `lu_p`,
  `triangular_solve_p`, and tensor primitives
- `lu`, `qr`, and `svd` remain first-class tenferro primitives even though
  they lower to `custom_call` below

### AD contract around those primitives

For JAX, the reference implementation uses:

- `primitive_jvps`
- `primitive_transposes`

For the v2 Rust stack, the aligned form is:

- each concrete tenferro primitive implements
  `PrimitiveOp::linearize`
- each concrete tenferro primitive implements
  `PrimitiveOp::transpose_rule`
- `tidu-rs` drives AD through those trait methods
- partial-evaluation / zero-propagation support still has to exist at the
  transform level

### Helper kernels that may exist below the public primitive layer

These are useful lowering helpers, but they do **not** need to be part of the
public v2 primitive catalog:

- `geqrf_p`
- `geqp3_p`
- `householder_product_p`
- `lu_pivots_to_permutation_p`

These only become part of the required primitive surface if tenferro decides
to port JAX's pure fallback algorithms instead of lowering `lu_p`, `qr_p`, and
`svd_p` directly to backend kernels / `custom_call`.

---

## VII. Design Consequence

If the target is really "copy JAX `linearize` into tenferro with minimal
semantic change", then tenferro v2 should be described in two distinct layers:

1. a **new JAX-like primitive layer**
   - this is what `PrimitiveOp::linearize` and `PrimitiveOp::transpose_rule`
     talk about
   - these primitives are new v2 definitions, not the current tenferro family
     enums
2. a **StableHLO lowering layer**
   - this is the concrete implementation target for backend execution
   - the correspondence tables above are the real implementation checklist

So the correct planning order is:

1. decide the new v2 primitive layer
2. decide which StableHLO ops are implemented directly
3. decide which structured ops stay above StableHLO and lower to `custom_call`
4. map current tenferro families and kernels into that new layering
