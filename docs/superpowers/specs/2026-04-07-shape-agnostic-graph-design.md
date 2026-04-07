# Shape-Agnostic Graph with DimExpr

**Issue:** #651
**Date:** 2026-04-07

## Motivation

TracedTensor graphs currently bake all tensor shapes (`Vec<usize>`) into op
definitions. Tensor network algorithms (DMRG, TT decomposition, ALS) change
bond dimensions dynamically via truncated SVD, making shape-fixed graphs
non-reusable across iterations.

**Usage model:** A graph is built once, executed many times with varying axis
sizes (bond dimensions change between iterations), then dropped. No
cross-graph caching or graph comparison is needed.

## Design Principles

1. **Rank is static, sizes are dynamic.** `TracedTensor` stores `rank: usize`
   instead of `shape: Vec<usize>`. All ops must support rank inference from
   their parameters and input ranks.
2. **Shape expressions reference graph values.** Ops that need output size
   information (Reshape, BroadcastInDim) store `Vec<DimExpr>` instead of
   `Vec<usize>`. DimExpr is evaluated at execution time from actual tensor
   shapes.
3. **AD rules are graph transformations.** They construct DimExpr values
   referencing primal inputs/outputs, keeping the backward graph
   shape-agnostic.
4. **Explicit API.** Users build symbolic size expressions via
   `TracedTensor::sym_size()` and operator overloading. No heuristic inference.

## DimExpr

An arithmetic expression over dimension sizes, evaluated at execution time.

```rust
#[derive(Clone, Debug)]
pub enum DimExpr {
    /// A concrete constant.
    Const(usize),
    /// Axis size of the op's i-th input tensor.
    /// Resolved from actual tensor shapes at execution time.
    InputDim { input_idx: usize, axis: usize },

    // Arithmetic
    Add(Box<DimExpr>, Box<DimExpr>),
    Sub(Box<DimExpr>, Box<DimExpr>),
    Mul(Box<DimExpr>, Box<DimExpr>),
    FloorDiv(Box<DimExpr>, Box<DimExpr>),

    // Comparison-based
    Min(Box<DimExpr>, Box<DimExpr>),
    Max(Box<DimExpr>, Box<DimExpr>),
}
```

Additional operations (Pow, Sqrt, Prod, Mod) can be added as needed. Since
DimExpr is never compared or cached (only evaluated), extending the enum has
no architectural cost.

### Evaluation

```rust
impl DimExpr {
    pub fn eval(&self, input_shapes: &[&[usize]]) -> usize {
        match self {
            Const(v) => *v,
            InputDim { input_idx, axis } => input_shapes[*input_idx][*axis],
            Add(a, b) => a.eval(input_shapes) + b.eval(input_shapes),
            Sub(a, b) => a.eval(input_shapes) - b.eval(input_shapes),
            Mul(a, b) => a.eval(input_shapes) * b.eval(input_shapes),
            FloorDiv(a, b) => a.eval(input_shapes) / b.eval(input_shapes),
            Min(a, b) => a.eval(input_shapes).min(b.eval(input_shapes)),
            Max(a, b) => a.eval(input_shapes).max(b.eval(input_shapes)),
        }
    }
}
```

## Lowering: TracedTensor -> DimExpr

Two-stage mapping converts user-level symbolic sizes into op-level DimExpr.

### Stage 1: User API -- SymDim

`TracedTensor::sym_size(axis)` returns a `SymDim` that references the tensor
by identity (an opaque ID assigned at TracedTensor creation).

```rust
impl TracedTensor {
    pub fn sym_size(&self, axis: usize) -> SymDim {
        SymDim(RawSymDim::TensorAxis { tensor_id: self.id, axis })
    }
}
```

`SymDim` supports operator overloading to build expression trees:

```rust
// User writes:
let merged = x.sym_size(0) * x.sym_size(1);
let y = x.reshape(&[merged, 4.into()])?;
```

`SymDim` is a transient API-level type. It is never stored in the graph.

### Stage 2: Op construction -- SymDim to DimExpr

When an op method is called (e.g., `reshape`), the implementation:

1. Knows its input tensors and their order (e.g., `x` is input 0)
2. Builds a map: `{ tensor_id -> input_idx }`
3. Walks the SymDim expression tree and replaces each `TensorAxis { tensor_id, axis }`
   with `DimExpr::InputDim { input_idx, axis }`
4. If a tensor_id is not found in the input map, returns an error

```
User code                       Op stored in graph
─────────────                   ──────────────────
x.sym_size(0) * x.sym_size(1)  Mul(InputDim(0,0), InputDim(0,1))
4.into()                        Const(4)
```

### Backward compatibility and constant arithmetic

`usize` values are accepted via `Into<SymDim>` / `Into<DimExpr>`:

```rust
impl From<usize> for SymDim {
    fn from(v: usize) -> Self { SymDim(RawSymDim::Const(v)) }
}

// Mixed arithmetic with usize constants:
impl Mul<SymDim> for usize { ... }  // 2 * x.sym_size(0)
impl Mul<usize> for SymDim { ... }  // x.sym_size(0) * 2
impl Add<usize> for SymDim { ... }  // x.sym_size(0) + 3
// etc. for Sub, FloorDiv
```

Usage examples:

```rust
let a = 2 * x.sym_size(0);                   // Mul(Const(2), InputDim(0,0))
let b = x.sym_size(0) * x.sym_size(1) + 3;   // Add(Mul(...), Const(3))
let y = x.reshape(&[a, b])?;                  // SymDim and usize mix freely
let z = x.reshape(&[3, 4])?;                  // pure usize still works
```

## Changes to StdTensorOp

### Shape fields: remove or convert to DimExpr

| Op | Field change |
|---|---|
| `Reshape` | remove `from_shape`; change `to_shape: Vec<usize>` to `Vec<DimExpr>` |
| `BroadcastInDim` | change `shape: Vec<usize>` to `Vec<DimExpr>`; `dims` unchanged |
| `ReduceSum` | remove `input_shape` |
| `ReduceProd` | remove `input_shape` |
| `ReduceMax` | remove `input_shape` |
| `ReduceMin` | remove `input_shape` |
| `Svd` | remove `input_shape` |
| `Qr` | remove `input_shape` |
| `Cholesky` | remove `input_shape` |
| `Lu` | remove `input_shape` |
| `Eigh` | remove `input_shape` |
| `Eig` | remove `input_shape` |
| `TriangularSolve` | remove `lhs_shape`, `rhs_shape` |

### Hash / Eq

`DimExpr` must implement `Hash` and `Eq` (structurally). This is needed
because `StdTensorOp` derives Hash/Eq for `GlobalOpKey` identity in
computegraph. Two DimExpr values are equal iff they have identical tree
structure -- no algebraic simplification.

### Dynamic n_inputs

When AD rules add shape-source inputs to Reshape or BroadcastInDim, the
op's `n_inputs()` must reflect the actual number of inputs. This is
computed from the DimExpr fields:

```rust
fn n_inputs(&self) -> usize {
    match self {
        Self::Reshape { to_shape } | Self::BroadcastInDim { shape: to_shape, .. } => {
            let max_idx = to_shape.iter()
                .flat_map(|d| d.max_input_idx())
                .max()
                .map_or(0, |m| m + 1);
            max_idx.max(1) // at least 1 (data input)
        }
        // ...
    }
}
```

For user-facing ops, DimExpr only references InputDim(0, ..) (the data
input itself), so n_inputs remains 1. AD-generated ops may reference
InputDim(1, ..) (the primal tensor), making n_inputs = 2.

### New op: NaryEinsum

```rust
NaryEinsum {
    subscripts: String,   // e.g. "ij,jk,kl->il"
    n_inputs: usize,
}
```

Records N-ary einsum as a single graph node. Contraction path optimization
is deferred to execution time, where `EinsumCache` is consulted with
actual input shapes.

## TracedTensor Changes

```rust
// Before
pub struct TracedTensor {
    pub shape: Vec<usize>,
    ...
}

// After
pub struct TracedTensor {
    pub id: TracedTensorId,   // unique ID for SymDim references
    pub rank: usize,          // static rank (ndim)
    ...
}
```

### Rank inference

| Op | Output rank |
|---|---|
| Elementwise (Add, Neg, Exp, ...) | same as input |
| `ReduceSum { axes }` | input_rank - axes.len() |
| `Reshape { to_shape }` | to_shape.len() |
| `BroadcastInDim { shape, dims }` | shape.len() |
| `Transpose { perm }` | perm.len() |
| `DotGeneral(config)` | computed from config |
| `Svd` | 3 outputs with ranks derivable from input rank |
| `Qr` | 2 outputs with ranks equal to input rank |
| `NaryEinsum { subscripts }` | output subscript length |

## AD Rules

AD rules construct `DimExpr` values referencing primal inputs/outputs.
The `FragmentBuilder` and AD rule signatures provide access to primal
values via `ValRef` / `GlobalValKey`. The AD rule knows:
- The forward op (including its DimExpr fields)
- The primal input/output `GlobalValKey`s
- Input ranks (from the forward op's rank inference)

### Example: ReduceSum VJP (transpose rule)

```rust
// Before:
let StdTensorOp::ReduceSum { axes, input_shape } = op;
let kept_dims = (0..input_shape.len())
    .filter(|d| !axes.contains(d)).collect();
builder.add_op(
    BroadcastInDim { shape: input_shape.clone(), dims: kept_dims },
    vec![cotangent],
    ...
);

// After:
let StdTensorOp::ReduceSum { axes } = op;
let input_rank = /* known from primal input's rank */;
let kept_dims: Vec<usize> = (0..input_rank)
    .filter(|d| !axes.contains(d)).collect();
// primal_input is inputs[0] of the forward op, exposed to the
// backward fragment as some input at index K
let shape: Vec<DimExpr> = (0..input_rank)
    .map(|a| DimExpr::InputDim { input_idx: K, axis: a })
    .collect();
builder.add_op(
    BroadcastInDim { shape, dims: kept_dims },
    vec![cotangent, primal_input_ref],  // primal_input added as shape source
    ...
);
```

The BroadcastInDim op gains an extra input (the primal tensor) solely to
make `InputDim` references resolvable. At execution, only its shape is read.

### Example: Reshape VJP

```rust
// Before:
Reshape { from_shape: to_shape.clone(), to_shape: from_shape.clone() }
inputs: [cotangent]

// After:
// primal_input is added as input 1 to the backward reshape
Reshape { to_shape: (0..primal_input_rank)
    .map(|a| DimExpr::InputDim { input_idx: 1, axis: a })
    .collect()
}
inputs: [cotangent, primal_input_ref]
```

### Linalg AD

Linalg AD rules (SVD, QR, Cholesky, Eigh, TriangularSolve) currently extract
`(m, n, batch_shape)` from `input_shape`. After the change:
- `m` = `DimExpr::InputDim { input_idx: K, axis: 0 }`
- `n` = `DimExpr::InputDim { input_idx: K, axis: 1 }`
- `min(m, n)` = `DimExpr::Min(m, n)`
- Batch dims = `DimExpr::InputDim { input_idx: K, axis: 2.. }`

Where K is the input index of the primal tensor in the backward fragment.

## Execution Pipeline

```
Trace time (shape-agnostic):
  User builds graph via TracedTensor API
  TracedTensor stores rank only
  Ops store DimExpr (not concrete sizes)
  AD transformations produce shape-agnostic backward graph

Execution time (shapes resolved):
  1. Input tensors provide actual shapes
  2. Lower StdTensorOp -> StableHloOp:
     - Evaluate DimExpr fields using actual input shapes
     - Reshape { to_shape: Vec<DimExpr> } -> Reshape { to_shape: Vec<usize> }
     - BroadcastInDim { shape: Vec<DimExpr> } -> BroadcastInDim { shape: Vec<usize> }
     - NaryEinsum -> expand via EinsumCache to binary DotGeneral ops
     - ReduceSum { axes } -> unchanged
     - Linalg ops -> unchanged (shapes from actual tensors)
  3. StableHloOp -> ExecOp (unchanged)
  4. Backend execution (unchanged)
```

StableHloOp, ExecOp, and backend layers require no changes. All DimExpr
resolution happens during the StdTensorOp-to-StableHloOp lowering step.

## NaryEinsum Execution

At execution time when `NaryEinsum` is encountered:

1. Read actual input tensor shapes
2. Look up `EinsumCache` with key `(subscripts, input_shapes)`
3. Cache miss: run contraction tree optimization, store result
4. Cache hit: reuse stored `ContractionTree`
5. Expand tree to binary ops with concrete shapes and execute

When bond dimensions change between iterations, a cache miss triggers
re-optimization. When the same size pattern recurs (e.g., left-to-right
then right-to-left DMRG sweep), the cache hits.

## Graph Reusability

A graph built with DimExpr is reusable with any input sizes as long as
ranks match. The same graph handles:
- Bond dimension chi=10 in iteration 1
- Bond dimension chi=8 in iteration 2 (after truncated SVD)
- No re-tracing needed

Ops with all-`Const` DimExpr (from `x.reshape(&[3, 4])`) are valid but
their concrete sizes must match at execution time.

## Constraints

- **Rank must be statically known.** Ops that change rank (Reshape,
  BroadcastInDim) determine output rank from the length of the DimExpr
  vector, which is fixed at trace time.
- **DimExpr references must be op inputs.** A DimExpr `InputDim { input_idx, axis }`
  must refer to one of the op's inputs. This is validated at op construction
  and ensured structurally by AD rules.
- **No DimExpr comparison or normalization.** DimExpr values are only
  evaluated, never simplified or compared for semantic equality. Structural
  Hash/Eq is sufficient for computegraph's GlobalOpKey.

## Comparison with PyTorch SymInt

| Aspect | PyTorch SymInt | tenferro DimExpr |
|---|---|---|
| Scope | Global symbol space (s0, s1, ...) | Op-local (InputDim references) |
| Expression engine | Full sympy algebra | Simple eval-only enum |
| Guard system | Runtime guards + recompilation | None needed (single graph, reused) |
| Graph caching | Guard-based cache lookup | N/A (graph held directly, not cached) |
| Complexity | Very high | Low |
| Use case | General-purpose JIT | Iterative tensor network algorithms |

## Scope

### Phase 1 (this issue)

1. Define `DimExpr` type with eval + Hash/Eq
2. Define `SymDim` API type with operator overloading + `Into<DimExpr>` conversion
3. Change `TracedTensor.shape` to `TracedTensor.rank` + add `TracedTensor.id`
4. Change `Reshape.to_shape` and `BroadcastInDim.shape` to `Vec<DimExpr>`
5. Remove `from_shape` from `Reshape`
6. Remove `input_shape` from `ReduceSum`, `ReduceProd`, `ReduceMax`, `ReduceMin`
7. Update corresponding AD rules
8. Update StableHLO lowering to evaluate DimExpr

### Phase 2

9. Remove `input_shape` from linalg ops (Svd, Qr, Cholesky, Lu, Eigh, Eig)
10. Remove `lhs_shape`, `rhs_shape` from TriangularSolve
11. Update linalg AD rules
12. Add `NaryEinsum` variant

### Future extensions

- Add Pow, Sqrt, Prod, Mod to DimExpr as needed
- XLA/StableHLO backend: NaryEinsum lowering to custom_call or binary expansion

## Files to Modify

- `tenferro-ops/src/std_tensor_op.rs` -- op definitions, DimExpr type
- `tenferro-ops/src/ad/structural.rs` -- Reshape, BroadcastInDim AD rules
- `tenferro-ops/src/ad/contraction.rs` -- ReduceSum AD rules, DotGeneral
- `tenferro-ops/src/ad/linalg.rs` -- linalg AD rules (phase 2)
- `tenferro-ops/src/semiring_ops.rs` -- SemiringOps trait (reshape, broadcast_in_dim signatures)
- `tenferro/src/traced.rs` -- TracedTensor API, sym_size(), rank inference
- `tenferro/src/compiler.rs` -- StableHLO lowering with DimExpr evaluation
- `tenferro/src/einsum.rs` -- NaryEinsum emission (phase 2)
- `tenferro-einsum/src/builder.rs` -- adapt to DimExpr shapes
