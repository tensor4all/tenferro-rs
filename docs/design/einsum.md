# Einsum

High-level einsum API on `Tensor<T>`. Supports string notation with
parenthesized contraction order, integer label notation, and pre-optimized
contraction trees.

See [contraction-pipeline.md](./contraction-pipeline.md) for the binary
contraction pipeline details and [tensor-prims.md](./tensor-prims.md)
for the `TensorPrims<A>` protocol.

---

## Public API

### Subscripts

```rust
/// Einsum subscripts using integer labels (omeinsum-rs compatible).
#[derive(Debug, Clone)]
pub struct Subscripts {
    pub inputs: Vec<Vec<u32>>,
    pub output: Vec<u32>,
}

impl Subscripts {
    /// Create from integer label arrays.
    pub fn new(inputs: &[&[u32]], output: &[u32]) -> Self;

    /// Parse from string notation: "ij,jk->ik"
    /// Supports parenthesized order: "ij,(jk,kl)->il"
    pub fn parse(notation: &str) -> Result<Self>;
}
```

### ContractionTree

```rust
pub struct ContractionTree { /* internal */ }

impl ContractionTree {
    /// Automatically optimize contraction order (cost-based heuristic).
    pub fn optimize(subscripts: &Subscripts, shapes: &[&[usize]]) -> Result<Self>;

    /// Manually specify pairwise contraction sequence.
    pub fn from_pairs(
        subscripts: &Subscripts,
        shapes: &[&[usize]],
        pairs: &[(usize, usize)],
    ) -> Result<Self>;
}
```

### Three API Levels

```rust
/// Level 1: String notation — parse + optimize + execute.
pub fn einsum<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

/// Level 2: Pre-built subscripts — optimize + execute.
pub fn einsum_with_subscripts<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

/// Level 3: Pre-optimized tree — execute only.
pub fn einsum_with_plan<T, A, B>(
    ctx: &mut B::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;
```

| Level | Parsing | Optimization | Execution | Use case |
|-------|---------|-------------|-----------|----------|
| `einsum` | Yes | Yes | Yes | One-off, convenience |
| `einsum_with_subscripts` | Cached | Yes | Yes | Same pattern, varying shapes |
| `einsum_with_plan` | Cached | Cached | Yes | Hot loops, same shapes |

All functions take `ctx: &mut B::Context` for explicit backend context
passing (thread pool, plan cache). `size_dict` provides dimension sizes
for output labels not present in any input (generative einsum).

### Accumulating Variants

Each allocating function has an `_into` counterpart with BLAS-style scaling:

```rust
/// output = alpha * einsum(operands) + beta * output
pub fn einsum_into<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

pub fn einsum_with_subscripts_into<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

pub fn einsum_with_plan_into<T, A, B>(
    ctx: &mut B::Context,
    tree: &ContractionTree,
    operands: &[&Tensor<T>],
    alpha: T,
    beta: T,
    output: &mut Tensor<T>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<()>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;
```

The `_into` variants eliminate output allocation per call and enable
accumulation semantics that map directly to `TensorPrims::execute` alpha/beta.

### Consuming Variants

```rust
/// Level 1: Consuming — input buffers may be reused.
pub fn einsum_owned<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: Vec<Tensor<T>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

/// Level 2: Consuming — optimize + execute.
pub fn einsum_with_subscripts_owned<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &Subscripts,
    operands: Vec<Tensor<T>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

/// Level 3: Consuming — execute only.
pub fn einsum_with_plan_owned<T, A, B>(
    ctx: &mut B::Context,
    tree: &ContractionTree,
    operands: Vec<Tensor<T>>,
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;
```

Input tensors are moved. The implementation may reuse their buffers for
intermediate results, avoiding allocation in contraction trees. Buffer
reuse is deterministic — Rust ownership guarantees no other references.

### Summary: Nine API Functions

| | Allocating | Accumulating (`_into`) | Consuming (`_owned`) |
|---|---|---|---|
| String notation | `einsum` | `einsum_into` | `einsum_owned` |
| Pre-built subscripts | `einsum_with_subscripts` | `einsum_with_subscripts_into` | `einsum_with_subscripts_owned` |
| Pre-optimized tree | `einsum_with_plan` | `einsum_with_plan_into` | `einsum_with_plan_owned` |

All 9 functions share the same generic signature pattern `<T, A, B>` with
`ctx: &mut B::Context` and `size_dict: Option<&HashMap<u32, usize>>`.

---

## User Examples

```rust
use tenferro_einsum::{einsum, einsum_owned, Subscripts, ContractionTree};
use tenferro_tensor::{Tensor, MemoryOrder};
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::{CpuBackend, CpuContext};

let col = MemoryOrder::ColumnMajor;
let mut ctx = CpuContext::new(4);

let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();

// Matrix multiplication
let c = einsum::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();

// Trace
let tr = einsum::<_, _, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();

// Batch matrix multiplication
let ba = Tensor::<f64>::zeros(&[10, 3, 4], LogicalMemorySpace::MainMemory, col);
let bb = Tensor::<f64>::zeros(&[10, 4, 5], LogicalMemorySpace::MainMemory, col);
let bc = einsum::<_, _, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&ba, &bb], None).unwrap();

// Explicit contraction order via parentheses
let d = einsum::<_, _, CpuBackend>(
    &mut ctx, "ij,(jk,kl)->il", &[&a, &b, &c], None,
).unwrap();

// Integer label notation (for programmatic use)
let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
let c = einsum_with_subscripts::<_, _, CpuBackend>(&mut ctx, &subs, &[&a, &b], None).unwrap();

// Pre-optimized tree (hot loops)
let tree = ContractionTree::optimize(&subs, &[&[2, 2], &[2, 2]]).unwrap();
for _ in 0..n_steps {
    let c = einsum_with_plan::<_, _, CpuBackend>(&mut ctx, &tree, &[&a, &b], None).unwrap();
}

// Consuming variant: operands moved, buffers reused
let x = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
let y = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
let z = einsum_owned::<_, _, CpuBackend>(&mut ctx, "ij,jk->ik", vec![x, y], None).unwrap();
```

---

## N-ary Contraction (Internal)

For N > 2 inputs, the einsum engine uses contraction tree optimization
to find the optimal pairwise contraction order.

### Dispatch Strategy

```rust
fn einsum_impl<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &Subscripts,
    operands: &[&Tensor<T>],
    size_dict: Option<&HashMap<u32, usize>>,
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
{
    match operands.len() {
        0 => Err(Error::InvalidArgument("no inputs".into())),
        1 => single_tensor_op::<T, A, B>(ctx, operands[0], &subscripts),
        2 => binary_contraction::<T, A, B>(ctx, operands[0], operands[1], &subscripts),
        _ => {
            let tree = ContractionTree::optimize(&subscripts, ...)?;
            execute_tree::<T, A, B>(ctx, &tree, operands)
        }
    }
}
```

### Binary Contraction Decomposition

For each binary contraction, the engine chooses between:

**Path A — Extended Contract available** (`has_extension_for::<T>(Contract)`):
```
let desc = PrimDescriptor::Contract { modes_a, modes_b, modes_c };
let plan = B::plan::<T>(&mut ctx, &desc, &shapes)?;
B::execute(&mut ctx, &plan, alpha, &[&a, &b], beta, &mut c)?;
→ backend handles diag, trace, permute, GEMM internally
```

**Path B — Core ops only** (decompose into primitives):
```
1. diag(a, paired_axes)        // zero-copy stride trick on Tensor<T>
2. diag(b, paired_axes)        // zero-copy stride trick on Tensor<T>
3. trace/reduce(a, trace_axes) // TensorPrims::trace or TensorPrims::reduce
4. trace/reduce(b, trace_axes)
5. permute_view(a, canonical)  // zero-copy metadata on Tensor<T>
6. permute_view(b, canonical)
7. make_contiguous(a)          // TensorPrims::make_contiguous (conditional copy)
8. make_contiguous(b)          // TensorPrims::make_contiguous (conditional copy)
9. batched_gemm(a, b, c)       // plan + execute BatchedGemm
```

See [contraction-pipeline.md](./contraction-pipeline.md) for details on
the `permute_view + MakeContiguous + BatchedGemm` pipeline.

**Example: `"iij,jkk->ik"`**:
```
1. a' = a.diagonal([(0,1)])    → a'[i,j]    (zero-copy)
2. b' = b.diagonal([(1,2)])    → b'[j,k]    (zero-copy)
3. (no trace/reduce needed)
4. a'' = a'.permute_view([i,j])  → canonical  (zero-copy)
5. b'' = b'.permute_view([j,k])  → canonical  (zero-copy)
6. make_contiguous(a'')          → conditional copy
7. make_contiguous(b'')          → conditional copy
8. batched_gemm(a'', b'', c)   → c[i,k]     (computation)
```

### Single-Tensor Decomposition (Unary Operations)

| Einsum | Decomposition |
|--------|--------------|
| `ii→` (full trace) | `trace(A, [(0,1)])` |
| `ii→i` (diagonal) | `diag(A, [(0,1)])` (zero-copy on Tensor) |
| `iij→j` (partial trace) | `diag(A, [(0,1)])` → `reduce(A', axis=0)` |
| `ij→ji` (transpose) | `permute(A, [1,0])` |
| `i→ij` (broadcast) | `repeat(A, j_dim)` (zero-copy on Tensor) |
| `i→ii` (embed diagonal) | `anti_diag(A, [(0,1)])` |
| `ij→i` (sum axis) | `reduce(A, axis=1, ReduceOp::Sum)` |

---

## Algebra Dispatch

Backend selection is determined by `T: HasAlgebra → infers algebra A`:

```rust
// impl TensorPrims<Standard> for CpuBackend  → faer/cblas GEMM
// impl TensorPrims<MaxPlus> for CpuBackend   → tropical-gemm (tenferro-tropical)
// impl TensorPrims<Standard> for CudaBackend → cuTENSOR [future]
// impl TensorPrims<Standard> for RocmBackend → hipTensor [future]
// impl TensorPrims<MyAlgebra> for CpuBackend → user-provided kernels
```

See [algebra.md](./algebra.md) for `HasAlgebra` and `Semiring` details.

---

## Automatic Differentiation

### Five AD Functions

The einsum AD API provides five functions for different AD modes:

```rust
/// Tracked einsum (reverse-mode AD via tape).
pub fn tracked_einsum<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&TrackedTensor<Tensor<T>>],
) -> AdResult<TrackedTensor<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
    Tensor<T>: Differentiable;

/// Dual einsum (forward-mode JVP propagation).
pub fn dual_einsum<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&DualTensor<Tensor<T>>],
) -> AdResult<DualTensor<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>,
    Tensor<T>: Differentiable;

/// Reverse-mode rule (rrule) for einsum without building a global tape.
/// Returns one gradient tensor per input operand.
pub fn einsum_rrule<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    operands: &[&Tensor<T>],
    cotangent: &Tensor<T>,
) -> Result<Vec<Tensor<T>>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

/// Forward-mode rule (frule) for einsum without building a global tape.
/// Inputs without tangent should use `None`.
pub fn einsum_frule<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T>>],
) -> Result<Tensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;

/// Local HVP rule for einsum (forward-over-reverse).
/// Returns (gradient, hvp) pairs for each input operand.
pub fn einsum_hvp<T, A, B>(
    ctx: &mut B::Context,
    subscripts: &str,
    primals: &[&Tensor<T>],
    tangents: &[Option<&Tensor<T>>],
    cotangent: &Tensor<T>,
    cotangent_tangent: &Tensor<T>,
) -> Result<Vec<(Tensor<T>, Tensor<T>)>>
where
    T: Scalar + HasAlgebra<Algebra = A>,
    B: TensorPrims<A>;
```

All AD functions take `ctx: &mut B::Context` and `B: TensorPrims<A>`,
matching the non-AD einsum functions.

### Adjoint Rules

Each forward operation has a clean adjoint:

```
Forward:  C[i,k] = einsum("ij,jk->ik", A, B)
         = batched_gemm(A[i,j], B[j,k])

Backward: ∂A[i,j] = batched_gemm(∂C[i,k], B^T[k,j])
          ∂B[j,k] = batched_gemm(A^T[j,i], ∂C[i,k])

Forward:  y[j] = einsum("iij->j", A)
         = reduce(diag(A, [(0,1)]), axis=0)

Backward: ∂A = anti_diag(repeat(∂y, i_dim), [(0,1)])
```

Both VJP and JVP go through `TensorPrims` primitives, working on CPU and
GPU uniformly.

### Optimizations from strided-opteinsum (Future)

- **Borrowed-view passthrough**: Leaf nodes return borrows, not clones.
- **Permutation-only detection**: Metadata-only transformation for
  nodes that only permute axes (no contraction).
- **Direct root write**: Final contraction writes into user's output
  buffer (no extra allocation).
