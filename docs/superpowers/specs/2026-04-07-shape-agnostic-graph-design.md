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
2. **All shape fields become `Vec<DimExpr>`.** Every `Vec<usize>` shape
   field in StdTensorOp is uniformly converted to `Vec<DimExpr>`. This
   includes output shape parameters (Reshape.to_shape, BroadcastInDim.shape)
   AND input shape metadata (ReduceSum.input_shape, Svd.input_shape, etc.).
   DimExpr is evaluated at execution time from actual tensor shapes.
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

### Shape fields: uniformly convert to `Vec<DimExpr>`

All `Vec<usize>` shape fields become `Vec<DimExpr>`. No fields are removed.
Rank is always available as `field.len()`.

| Op | Field change |
|---|---|
| `Reshape` | `from_shape: Vec<usize>` -> `Vec<DimExpr>`; `to_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `BroadcastInDim` | `shape: Vec<usize>` -> `Vec<DimExpr>`; `dims` unchanged |
| `ReduceSum` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `ReduceProd` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `ReduceMax` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `ReduceMin` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `Svd` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `Qr` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `Cholesky` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `Lu` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `Eigh` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `Eig` | `input_shape: Vec<usize>` -> `Vec<DimExpr>` |
| `TriangularSolve` | `lhs_shape: Vec<usize>` -> `Vec<DimExpr>`; `rhs_shape` -> `Vec<DimExpr>` |

At trace time, shape fields are populated with `DimExpr::InputDim`
references to the op's own inputs:

```rust
// Example: ReduceSum at trace time
ReduceSum {
    axes: vec![1],
    input_shape: vec![
        DimExpr::InputDim { input_idx: 0, axis: 0 },
        DimExpr::InputDim { input_idx: 0, axis: 1 },
    ],
}
```

### Hash / Eq

`DimExpr` must implement `Hash` and `Eq` (structurally). This is needed
because `StdTensorOp` derives Hash/Eq for `GlobalOpKey` identity in
computegraph. Two DimExpr values are equal iff they have identical tree
structure -- no algebraic simplification.

### Dynamic n_inputs

Ops with DimExpr fields may reference additional shape-source inputs
beyond their data inputs. `n_inputs()` is computed from all DimExpr
fields in the op:

```rust
fn n_inputs(&self) -> usize {
    match self {
        Self::Reshape { from_shape, to_shape } => {
            let all_exprs = from_shape.iter().chain(to_shape.iter());
            let max_idx = all_exprs
                .filter_map(|d| d.max_input_idx())
                .max()
                .map_or(0, |m| m + 1);
            max_idx.max(1) // at least 1 (data input)
        }
        // Same pattern for BroadcastInDim, ReduceSum, linalg ops...
    }
}
```

For user-facing ops, DimExpr only references InputDim(0, ..) (the data
input itself), so n_inputs remains 1. AD-generated ops may reference
InputDim(1, ..) (the primal tensor), making n_inputs = 2.

### AD helper: remap_input_idx

AD rules read DimExpr from the forward op and pass them to backward ops.
Since backward ops have different input ordering, `InputDim` references
must be remapped:

```rust
/// Remap InputDim references: input_idx `from` -> `to`.
fn remap_input_idx(exprs: &[DimExpr], from: usize, to: usize) -> Vec<DimExpr> {
    exprs.iter().map(|e| e.remap(from, to)).collect()
}
```

Example: forward ReduceSum has `input_shape` with `InputDim(0, ..)`.
The backward BroadcastInDim has inputs `[cotangent, primal_input]`, so
the primal is at index 1. Remap: `InputDim(0, axis) -> InputDim(1, axis)`.

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
| `ReduceSum { axes, input_shape }` | input_shape.len() - axes.len() |
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
let StdTensorOp::ReduceSum { axes, input_shape } = op;  // input_shape: Vec<usize>
let kept_dims = (0..input_shape.len())
    .filter(|d| !axes.contains(d)).collect();
builder.add_op(
    BroadcastInDim { shape: input_shape.clone(), dims: kept_dims },
    vec![cotangent],
    ...
);

// After:
let StdTensorOp::ReduceSum { axes, input_shape } = op;  // input_shape: Vec<DimExpr>
let input_rank = input_shape.len();
let kept_dims: Vec<usize> = (0..input_rank)
    .filter(|d| !axes.contains(d)).collect();
// Remap: forward's InputDim(0, ..) -> backward's InputDim(1, ..)
// because backward inputs are [cotangent, primal_input]
let shape = remap_input_idx(input_shape, 0, 1);
builder.add_op(
    BroadcastInDim { shape, dims: kept_dims },
    vec![cotangent, primal_input_ref],
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
// Remap from_shape's InputDim(0, ..) -> InputDim(1, ..)
// primal_input is added as input 1
Reshape {
    from_shape: remap_input_idx(to_shape, 0, 1),
    to_shape: remap_input_idx(from_shape, 0, 1),
}
inputs: [cotangent, primal_input_ref]
```

### Linalg AD

Linalg AD rules (SVD, QR, Cholesky, Eigh, TriangularSolve) currently extract
`(m, n, batch_shape)` from `input_shape: Vec<usize>`. After the change,
`input_shape` is `Vec<DimExpr>`:
- `m` = `input_shape[0]` (already a `DimExpr::InputDim`)
- `n` = `input_shape[1]`
- `min(m, n)` = `DimExpr::Min(m.clone(), n.clone())`
- Batch dims = `input_shape[2..]`

The AD rules remap these DimExpr values using `remap_input_idx` when
constructing backward ops with different input ordering.

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
     - ReduceSum { input_shape: Vec<DimExpr> } -> ReduceSum { input_shape: Vec<usize> }
     - Linalg ops { input_shape: Vec<DimExpr> } -> evaluate to concrete
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

All shape fields are converted uniformly. No phasing needed for the
DimExpr migration itself — all ops change together.

### DimExpr migration (this issue)

1. Define `DimExpr` type with eval + Hash/Eq + `remap_input_idx`
2. Define `SymDim` API type with operator overloading
3. Change `TracedTensor.shape` to `TracedTensor.rank` + add `TracedTensor.id`
4. Convert ALL `Vec<usize>` shape fields in StdTensorOp to `Vec<DimExpr>`
5. Update SemiringOps trait signatures
6. Update ALL AD rules (structural, contraction, linalg)
7. Update StableHLO lowering to evaluate DimExpr
8. Add `TracedTensor::sym_size()` + `reshape_sym()` API

### Separate issue

9. Add `NaryEinsum` variant (independent of DimExpr migration)

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
