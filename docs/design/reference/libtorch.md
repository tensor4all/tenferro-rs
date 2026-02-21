# libtorch / PyTorch Feature Reference for tenferro Design

This document provides a comprehensive survey of PyTorch's (libtorch) C++ tensor
infrastructure, covering features relevant to tenferro's scope. It is intended as
a reference when designing tenferro crate APIs to follow the `torch` approach where
appropriate — adapted for column-major layout and Rust idioms.

> **Note on batch convention**: PyTorch uses row-major (C-contiguous) with
> `(*, m, n)` (last 2 dims = matrix). tenferro uses col-major (Fortran-contiguous)
> with `(m, n, *)` (first 2 dims = matrix). All shape descriptions below use
> PyTorch's convention unless otherwise noted.

---

## Table of Contents

1. [Core Tensor API](#1-core-tensor-api)
2. [Einsum, BLAS, and Contraction](#2-einsum-blas-and-contraction)
3. [Linear Algebra (`torch.linalg`)](#3-linear-algebra-torchlinalg)
4. [Automatic Differentiation (Autograd)](#4-automatic-differentiation-autograd)
5. [Device Management and C API](#5-device-management-and-c-api)
6. [Mapping to tenferro Crates](#6-mapping-to-tenferro-crates)

---

## 1. Core Tensor API

### 1.1 Tensor Creation: Factory Functions

All factory functions follow the schema:
```cpp
torch::<function>(sizes, TensorOptions)
```
where `TensorOptions` bundles dtype, device, layout, requires_grad, memory_format.

| Function | Description |
|---|---|
| `zeros(sizes, opts)` | All elements zero |
| `ones(sizes, opts)` | All elements one |
| `empty(sizes, opts)` | Uninitialized storage |
| `full(sizes, fill, opts)` | All elements set to `fill` |
| `eye(n, m, opts)` | Identity matrix |
| `arange(start, end, step, opts)` | Sequence; size inferred |
| `linspace(start, end, steps, opts)` | Linearly spaced values |
| `logspace(start, end, steps, base, opts)` | Logarithmically spaced |
| `rand(sizes, opts)` | Uniform on [0,1) |
| `randn(sizes, opts)` | Standard normal |
| `randint(low, high, sizes, opts)` | Random integers in [low, high) |
| `randperm(n, opts)` | Random permutation of 0..n-1 |

**`from_blob` — wrap external memory (zero-copy, non-owning)**:
```cpp
Tensor torch::from_blob(void* data, IntArrayRef sizes, TensorOptions opts);
Tensor torch::from_blob(void* data, IntArrayRef sizes, IntArrayRef strides,
                        const Deleter& deleter, TensorOptions opts);
```
Does not take ownership of `data`. Optional `Deleter` called when refcount reaches zero.

### 1.2 Internal Structure

`at::Tensor` is a value-type wrapper (`c10::intrusive_ptr<c10::TensorImpl>`).

| Field | Type | Purpose |
|---|---|---|
| `storage_` | `c10::Storage` | Owns or shares the raw data buffer |
| `storage_offset_` | `int64_t` | Offset into storage (in elements) |
| `sizes_and_strides_` | Small-buffer-optimized arrays | Shape + strides |
| `numel_` | `int64_t` | Cached product of sizes |
| `data_type_` | `caffe2::TypeMeta` | Element dtype |
| `device_opt_` | `optional<Device>` | Device placement |
| `key_set_` | `DispatchKeySet` | Dispatch keys |

Multiple tensors can share the same `Storage` (e.g., after `view()` or `slice()`).

### 1.3 Memory Layout and MemoryFormat

```cpp
IntArrayRef tensor.sizes();
IntArrayRef tensor.strides();
bool        tensor.is_contiguous(MemoryFormat fmt = Contiguous);
Tensor      tensor.contiguous(MemoryFormat fmt = Contiguous);
```

| MemoryFormat | Description |
|---|---|
| `Contiguous` | Row-major (C-contiguous) NCHW |
| `ChannelsLast` | NHWC for 4D tensors |
| `ChannelsLast3d` | NTHWC for 5D tensors |
| `Preserve` | Match input format |

`contiguous()` returns `*this` if already contiguous (zero-copy), otherwise allocates.

### 1.4 View Operations (Zero-Copy)

All return a **view** sharing the same Storage. Only metadata changes.

| Operation | Description |
|---|---|
| `view(shape)` | Reshape; requires contiguous input |
| `reshape(shape)` | Like `view` but may copy if not contiguous |
| `permute(dims)` | Reorder dimensions |
| `transpose(dim0, dim1)` | Swap two dimensions |
| `t()` | 2D transpose (alias for `transpose(0,1)`) |
| `expand(sizes)` | Broadcast (stride=0 on broadcast dims) |
| `narrow(dim, start, length)` | Slice along one dimension |
| `select(dim, index)` | Select single index (reduces ndim by 1) |
| `slice(dim, start, end, step)` | General slice with step |
| `unfold(dim, size, step)` | Extract sliding windows |
| `diagonal(offset, dim1, dim2)` | Extract diagonal as a view |
| `squeeze()` / `squeeze(dim)` | Remove dimensions of size 1 |
| `unsqueeze(dim)` | Insert dimension of size 1 |
| `as_strided(size, stride, offset)` | Universal view: arbitrary size/stride/offset |
| `movedim(src, dst)` | Move dimensions to new positions |
| `unflatten(dim, sizes)` | Split a dimension into multiple |
| `view_as_real()` | Complex → real (doubles last dim) |
| `view_as_complex()` | Real → complex (halves last dim) |

**Attributes that return views**: `.T`, `.H`, `.mT` (last-2-dims transposed), `.mH` (last-2-dims conjugate-transposed), `.real`, `.imag`.

**Split family**: `split`, `split_with_sizes`, `chunk`, `tensor_split`, `unbind`.

### 1.5 Indexing

**Basic indexing** (returns views): integer, slice, None, Ellipsis.

**Advanced indexing** (returns copies):

| Function | Description |
|---|---|
| `index_select(dim, index)` | Select entries along dim using 1D LongTensor |
| `gather(dim, index)` | Gather values at positions |
| `scatter_(dim, index, src)` | Inverse of gather (in-place) |
| `masked_select(mask)` | Select where mask is true → 1D |
| `masked_fill_(mask, value)` | Fill where mask is true |
| `index_put_(indices, values, accumulate)` | Advanced indexing assignment |

### 1.6 Type System

#### ScalarType (dtype)

| ScalarType | C++ type | Constant |
|---|---|---|
| `Float` | `float` | `kFloat32` |
| `Double` | `double` | `kFloat64` |
| `Half` | `at::Half` | `kFloat16` |
| `BFloat16` | `at::BFloat16` | `kBFloat16` |
| `ComplexFloat` | `c10::complex<float>` | `kComplexFloat32` |
| `ComplexDouble` | `c10::complex<double>` | `kComplexFloat64` |
| `Byte` | `uint8_t` | `kUInt8` |
| `Char` | `int8_t` | `kInt8` |
| `Short` | `int16_t` | `kInt16` |
| `Int` | `int32_t` | `kInt32` |
| `Long` | `int64_t` | `kInt64` |
| `Bool` | `bool` | `kBool` |
| `Float8_e5m2`, `Float8_e4m3fn` | 8-bit floats | FP8 types |

#### DeviceType

| DeviceType | Value | Description |
|---|---|---|
| `CPU` | 0 | Host memory |
| `CUDA` | 1 | NVIDIA GPU |
| `HIP` | 6 | AMD ROCm |
| `XLA` | 9 | TPU via XLA |
| `MPS` | 13 | Apple Metal |
| `XPU` | 14 | Intel GPU |
| `Meta` | 15 | Shape-only (no data) |
| `PrivateUse1` | 20 | Custom backend extension point |

#### Layout

| Layout | Description |
|---|---|
| `Strided` | Dense with per-dimension strides (default) |
| `SparseCoo` | Coordinate-format sparse |
| `SparseCsr`/`Csc`/`Bsr`/`Bsc` | Compressed sparse formats |

#### TensorOptions (builder pattern)

```cpp
TensorOptions().dtype(kFloat32).layout(kStrided).device(kCUDA, 0).requires_grad(true);
```

### 1.7 DLPack Interop

```cpp
DLManagedTensor* at::toDLPack(const Tensor& src);              // zero-copy export
Tensor at::fromDLPack(DLManagedTensor* src);                    // zero-copy import
DLManagedTensorVersioned* at::toDLPackVersioned(const Tensor&); // DLPack >= 1.0
```

### 1.8 Copy/Conversion Operations

| Operation | Semantics | Allocates? |
|---|---|---|
| `clone(fmt)` | Deep copy; **differentiable** | Yes |
| `contiguous(fmt)` | Returns self if already contiguous, else copies | Maybe |
| `detach()` | View disconnected from autograd graph | No |
| `to(device)` | Move to device; returns self if already there | Maybe |
| `to(dtype)` | Cast dtype; returns self if same | Maybe |
| `copy_(src, non_blocking)` | In-place copy from src | No |

---

## 2. Einsum, BLAS, and Contraction

### 2.1 `torch.einsum`

```python
torch.einsum(equation: str, *operands) -> Tensor
torch.einsum(op1, sublist1, op2, sublist2, ..., [sublist_out]) -> Tensor  # sublist format
```

**Notation modes**:
- **Explicit** (with `->` arrow): output subscripts listed after arrow
- **Implicit** (without `->` arrow): subscripts appearing once are kept (alphabetical order)
- **Ellipsis** (`...`): broadcasts covered dimensions

**Internal decomposition**: Decomposes into pairwise contractions via `torch.bmm` (batch
matrix multiply) + reshape/permute. Left-to-right order by default.

**opt_einsum integration**: When installed, automatically optimizes contraction order for
3+ operands. Configurable via `torch.backends.opt_einsum.enabled` and `.strategy`:

| Strategy | Description | Used for |
|---|---|---|
| `'optimal'` | Exhaustive search | ≤4 inputs |
| `'dp'` | Dynamic programming | Moderate sizes |
| `'branch-*'` | Restricted search | 5-8 inputs |
| `'greedy'` | One-step heuristic | >14 inputs |
| `'auto'` (default) | Auto-selects, targets ~1ms | All sizes |

### 2.2 Matrix Multiplication Variants

| Function | Inputs | Description |
|---|---|---|
| `mm(A, B)` | 2D × 2D | Matrix multiply → BLAS GEMM |
| `bmm(A, B)` | 3D × 3D | Batch matrix multiply. **No broadcasting.** |
| `matmul(A, B)` | Any × Any | General: dot, mv, mm, bmm with broadcasting |
| `addmm(C, A, B, beta, alpha)` | 2D | `beta*C + alpha*A@B` → full BLAS GEMM |
| `addbmm(C, A, B, beta, alpha)` | 3D→2D | Batched GEMM + reduce over batch |
| `baddbmm(C, A, B, beta, alpha)` | 3D | Batched GEMM without batch reduction |

### 2.3 BLAS Dispatch

**CPU**: MKL (default) or OpenBLAS → `cblas_{s,d}gemm`.
**GPU**: cuBLAS / cuBLASLt (selectable via `torch.backends.cuda.preferred_blas_library()`).

| PyTorch function | BLAS operation |
|---|---|
| `mm(A, B)` | `C = A*B` (GEMM, α=1, β=0) |
| `addmm(C, A, B, α, β)` | `C = β*C + α*A*B` (full GEMM) |
| `bmm(A, B)` | Strided batched GEMM |

### 2.4 Pointwise Operations

`add`, `sub`, `mul`, `div` (and in-place `_` variants). All support tensor-tensor
with broadcasting and tensor-scalar.

Implemented via **TensorIterator**: device-agnostic iteration that handles
shape inference, output allocation, stride reordering, dimension coalescing,
and parallel dispatch. The programmer defines only a functor.

### 2.5 Reduction Operations

| Function | Description |
|---|---|
| `sum(dim, keepdim)` | Sum over dimensions |
| `prod(dim, keepdim)` | Product over dimensions |
| `mean(dim, keepdim)` | Mean over dimensions |
| `max(dim)` / `min(dim)` | Returns `(values, indices)` namedtuple |
| `trace()` | Sum of diagonal (2D only) |
| `linalg.vector_norm(ord, dim)` | Vector norms |
| `linalg.matrix_norm(ord, dim)` | Matrix norms |

`keepdim=True` retains reduced dims with size 1 for broadcasting compatibility.

### 2.6 Broadcasting Rules (NumPy-compatible)

Iterating from trailing dimension: sizes must be equal, one must be 1, or one
does not exist (padded with 1 on the left). Broadcasting uses `expand()` (stride=0,
no allocation).

### 2.7 `torch.tensordot`

```python
torch.tensordot(a, b, dims) -> Tensor
```

Subset of einsum for pairwise contractions. Potentially more efficient because
it reshapes into a single BLAS GEMM call:
1. Permute contracted dims to be adjacent
2. Reshape to 2D
3. Single `mm` call
4. Reshape back

### 2.8 Operator Dispatch (ATen Dispatcher)

Layered dispatch keys with priority ordering:

| Priority | Keys | Role |
|---|---|---|
| Highest | `Functionalize`, `Batched` | Transform wrappers |
| High | `AutogradCPU`, `AutogradCUDA` | Autograd wrappers |
| Medium | `BackendSelect` | Backend selection |
| Low | `CPU`, `CUDA`, `XLA`, `Sparse*`, `Quantized*` | Actual kernels |

Dispatch key computed by union of input tensor keys + global + thread-local include − thread-local exclude. Highest-priority key selects kernel.

---

## 3. Linear Algebra (`torch.linalg`)

### 3.1 Complete Function Catalog

All functions use the `(*, m, n)` batch convention: last 2 dims are matrix, preceding
are independent batch dims. Batch dims broadcast following standard rules.

#### 3.1.1 Decompositions

| Function | Input | Return | VJP | JVP | Notes |
|---|---|---|---|---|---|
| `svd(A, full_matrices=True, driver=None)` | `(*, m, n)` | `(U, S, Vh)` | Yes | Yes | Unstable for repeated σ |
| `svdvals(A)` | `(*, m, n)` | `Tensor` (real) | Yes | Yes | Always stable |
| `eig(A)` | `(*, n, n)` | `(eigenvalues, eigenvectors)` | Yes | Yes | Unstable for repeated λ |
| `eigvals(A)` | `(*, n, n)` | `Tensor` (complex) | Yes | Yes | Always stable |
| `eigh(A, UPLO='L')` | `(*, n, n)` Hermitian | `(eigenvalues, eigenvectors)` | Yes | Yes | `1/(λ_i - λ_j)` instability |
| `eigvalsh(A, UPLO='L')` | `(*, n, n)` Hermitian | `Tensor` (real) | Yes | Yes | Always stable |
| `qr(A, mode='reduced')` | `(*, m, n)` | `(Q, R)` | Yes | Yes | Not for `mode='r'` |
| `lu(A, pivot=True)` | `(*, m, n)` | `(P, L, U)` | Yes | No | P non-differentiable |
| `lu_factor(A, pivot=True)` | `(*, m, n)` | `(LU, pivots)` | Yes | Yes | Compact form |
| `lu_factor_ex(A)` | `(*, m, n)` | `(LU, pivots, info)` | Yes | Yes | With error info |
| `cholesky(A, upper=False)` | `(*, n, n)` PD | `Tensor` | Yes | Yes | |
| `cholesky_ex(A)` | `(*, n, n)` PD | `(L, info)` | Yes | Yes | |
| `ldl_factor(A)` | `(*, n, n)` sym | `(LD, pivots)` | Yes | No | |
| `householder_product(A, tau)` | `(*, m, n)` | `Tensor` | Yes | Yes | |

#### 3.1.2 Solvers

| Function | Input | Return | VJP | JVP |
|---|---|---|---|---|
| `solve(A, B, left=True)` | A: `(*, n, n)`, B: `(*, n, k)` | `Tensor` | Yes | Yes |
| `solve_triangular(A, B, upper, left=True)` | A: `(*, n, n)`, B: `(*, n, k)` | `Tensor` | Yes | Yes |
| `lu_solve(LU, pivots, B)` | | `Tensor` | Yes | Yes |
| `lstsq(A, B, rcond, driver)` | A: `(*, m, n)`, B: `(*, m, k)` | `(solution, residuals, rank, singular_values)` | Yes | Yes |
| `tensorsolve(A, B)` | Higher-order | `Tensor` | Yes | No |

#### 3.1.3 Inverses

| Function | Input | Return | VJP | JVP |
|---|---|---|---|---|
| `inv(A)` | `(*, n, n)` | `Tensor` | Yes | Yes |
| `inv_ex(A)` | `(*, n, n)` | `(inverse, info)` | Yes | Yes |
| `pinv(A, atol, rtol)` | `(*, m, n)` | `Tensor` | Yes | Yes |
| `tensorinv(A, ind=2)` | Higher-order | `Tensor` | Yes | No |

#### 3.1.4 Norms and Condition Numbers

| Function | Input | Return | VJP |
|---|---|---|---|
| `norm(A, ord, dim)` | Arbitrary | `Tensor` (real) | Yes |
| `vector_norm(x, ord=2, dim)` | Arbitrary | `Tensor` (real) | Yes |
| `matrix_norm(A, ord='fro', dim)` | `(*, m, n)` | `Tensor` (real) | Yes |
| `cond(A, p)` | `(*, m, n)` | `Tensor` (real) | Yes |
| `matrix_rank(A, atol, rtol)` | `(*, m, n)` | `Tensor` (integer) | No |

#### 3.1.5 Determinants

| Function | Input | Return | VJP | JVP |
|---|---|---|---|---|
| `det(A)` | `(*, n, n)` | `Tensor` | Yes | Yes |
| `slogdet(A)` | `(*, n, n)` | `(sign, logabsdet)` | Yes | Yes |

#### 3.1.6 Matrix Functions

| Function | Input | Return | VJP |
|---|---|---|---|
| `matrix_exp(A)` | `(*, n, n)` | `Tensor` | Yes |
| `matrix_power(A, n)` | `(*, m, m)` | `Tensor` | Yes |

#### 3.1.7 Other

| Function | Description | VJP |
|---|---|---|
| `cross(a, b, dim)` | Cross product along dim | Yes |
| `multi_dot(tensors)` | Optimal chain matrix multiply | Yes |
| `vecdot(x, y, dim)` | Batched vector dot product | Yes |
| `diagonal(A, offset, dim1, dim2)` | Diagonal view | Yes |

### 3.2 SVD Options Detail

| Parameter | Values | Description |
|---|---|---|
| `full_matrices` | `True`/`False` | Full vs reduced (economy) |
| `driver` | `None`, `"gesvd"`, `"gesvdj"`, `"gesvda"` | CUDA backend selection |

### 3.3 QR Options Detail

| Parameter | Values | Description |
|---|---|---|
| `mode` | `"reduced"`, `"complete"`, `"r"` | Reduced, full, or R-only |

### 3.4 LAPACK / cuSOLVER Dispatch

**CPU**: Calls LAPACK routines via linked BLAS/LAPACK library.

| Operation | LAPACK Routine |
|---|---|
| SVD | `?gesdd` (divide-and-conquer) |
| QR | `?geqrf` + `?orgqr` |
| Cholesky | `?potrf` |
| LU | `?getrf` |
| Eigen (symmetric) | `?syevd` / `?heevd` |
| Eigen (general) | `?geev` |
| Solve | `?gesv` |
| Solve triangular | `?trsm` (BLAS Level 3) |
| lstsq | `?gels`, `?gelsy`, `?gelsd`, `?gelss` |

**CUDA**: cuSOLVER / MAGMA / cuBLAS with heuristic-based dispatch.
User control: `torch.backends.cuda.preferred_linalg_library()` (`"default"`, `"cusolver"`, `"magma"`).

### 3.5 AD Rules for Linalg

All registered in `tools/autograd/derivatives.yaml`. Backward functions in
`torch/csrc/autograd/FunctionsManual.cpp`.

**Degenerate cases**:
- **SVD**: `1/(σ_i² - σ_j²)` diverges for repeated singular values → NaN gradients.
  Use `svdvals` when only singular values are needed.
- **eigh**: `1/(λ_i - λ_j)` diverges for degenerate eigenvalues.
  Use `eigvalsh` for stable value-only gradients.
- **QR**: Requires first `min(m,n)` columns of input to be linearly independent.

### 3.6 Legacy vs New API

| Legacy | New | Key Changes |
|---|---|---|
| `torch.svd(some=True)` | `torch.linalg.svd(full_matrices=True)` | Returns `Vh` not `V`; default behavior differs |
| `torch.eig` | `torch.linalg.eig` | Complex return; batching; autograd |
| `torch.symeig` | `torch.linalg.eigh` | Default triangle `'L'` not `'U'` |
| `torch.qr(some=True)` | `torch.linalg.qr(mode='reduced')` | |
| `torch.solve(B, A)` | `torch.linalg.solve(A, B)` | Arg order reversed; adds `left` param |
| `torch.lu` | `torch.linalg.lu_factor` / `lu` | Cleaner separation |

### 3.7 Return Types: Named Tuples

| Function | Fields | C++ Return |
|---|---|---|
| `svd` | `(U, S, Vh)` | `std::tuple<Tensor, Tensor, Tensor>` |
| `eig`/`eigh` | `(eigenvalues, eigenvectors)` | `std::tuple<Tensor, Tensor>` |
| `qr` | `(Q, R)` | `std::tuple<Tensor, Tensor>` |
| `lu` | `(P, L, U)` | `std::tuple<Tensor, Tensor, Tensor>` |
| `lu_factor` | `(LU, pivots)` | `std::tuple<Tensor, Tensor>` |
| `slogdet` | `(sign, logabsdet)` | `std::tuple<Tensor, Tensor>` |

The `_ex` variants add an `info` tensor (LAPACK error code per batch element) and
`check_errors` flag to avoid CPU-CUDA sync on error path.

---

## 4. Automatic Differentiation (Autograd)

### 4.1 Core Concepts

| Concept | Description |
|---|---|
| `requires_grad` | Boolean attribute on Tensor. Enables gradient tracking. |
| `grad_fn` | Points to the `Node` that produced this tensor. `None` for leaves. |
| `backward()` | Traverses graph, accumulates gradients in `.grad`. Destroys graph by default. |
| `torch.autograd.grad()` | Returns gradients directly as tensors. Preferred with `create_graph=True`. |
| `retain_graph` | Preserves graph for multiple backward passes. |
| `create_graph` | Records backward pass itself → enables higher-order derivatives. |

### 4.2 Graph Structure (Dynamic DAG)

**Define-by-run**: Graph rebuilt every forward pass. No persistent tape.

**Node** (base class in `torch/csrc/autograd/function.h`):
- `apply(variable_list&& grads) -> variable_list`: computes VJP.
- `next_edges_`: vector of `Edge` objects pointing to predecessor nodes.

**Edge** = `(shared_ptr<Node>, uint32_t input_nr)`: destination node + input index.

**AccumulateGrad**: Terminal node for leaf tensors. Accumulates into `.grad`.

**Graph storage**: Implicit via `grad_fn` pointer chain. Reference-counted.

### 4.3 Reverse Mode (VJP)

Rules registered in `tools/autograd/derivatives.yaml`:

```yaml
- name: mul.Tensor(Tensor self, Tensor other) -> Tensor
  self: mul_tensor_backward(grad, other, self.scalar_type())
  other: mul_tensor_backward(grad, self, other.scalar_type())
  result: other_t * self_p + self_t * other_p    # JVP rule
```

Build-time code generation produces C++ `Node` subclasses (`MulBackward0`, etc.)
in `torch/csrc/autograd/generated/Functions.cpp`.

**Engine execution**: `GraphTask` + `NodeTask` + per-thread `ReadyQueue`. Reverse
topological order. Cotangents flow via chain rule. Concurrent backward since v1.6.

### 4.4 Forward Mode (JVP)

Introduced in **PyTorch 1.11** (beta). Uses **dual numbers**:

```python
import torch.autograd.forward_ad as fwAD

with fwAD.dual_level():
    dual_input = fwAD.make_dual(primal, tangent)
    dual_output = torch.sin(dual_input)
    primal_out, tangent_out = fwAD.unpack_dual(dual_output)
    # tangent_out = cos(primal) * tangent
```

JVP rules specified in the `result:` line of `derivatives.yaml`.

### 4.5 Custom Autograd Functions

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0).float()

    @staticmethod
    def jvp(ctx, input_tangent):
        input, = ctx.saved_tensors
        return input_tangent * (input > 0).float()

# Usage: MyOp.apply(x)
```

C++ equivalent uses `torch::autograd::Function<T>` with `AutogradContext`.

### 4.6 Detach and No-Grad

| Mechanism | Scope | Graph Recording | Usable in Autograd Later? |
|---|---|---|---|
| `detach()` | Single tensor | No | Detached: no; original: yes |
| `torch.no_grad()` | Code block | No | Yes |
| `torch.inference_mode()` | Code block | No | No (strictest) |

### 4.7 Higher-Order Derivatives

`create_graph=True` → backward is recorded → enables differentiation through gradients.

**`torch.func` (composable transforms, JAX-inspired)**:

```python
from torch.func import grad, vjp, jvp, vmap, jacrev, jacfwd

# HVP (forward-over-reverse, memory efficient):
def hvp(f, primals, tangents):
    return jvp(grad(f), primals, tangents)[1]

# Full Jacobian:
J = jacrev(func)(input)   # via vmap + vjp
J = jacfwd(func)(input)   # via vmap + jvp
```

### 4.8 Checkpointing

`torch.utils.checkpoint` trades compute for memory — recomputes forward during backward
instead of storing activations.

```python
from torch.utils.checkpoint import checkpoint
output = checkpoint(function, *args, use_reentrant=False)
```

### 4.9 View Operations and Autograd

View ops have backward nodes registered in `derivatives.yaml`:

```yaml
- name: permute(Tensor(a) self, int[] dims) -> Tensor(a)
  self: permute_backwards(grad, dims)
  result: auto_linear

- name: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
  self: grad.reshape_symint(self.sym_sizes())
  result: auto_linear
```

The `auto_linear` annotation means JVP is trivially the same op on the tangent.

Backward nodes: `PermuteBackward0`, `TransposeBackward0`, `ViewBackward0`,
`AsStridedBackward0`, `SliceBackward0`, etc. Each applies the inverse transformation
to the gradient. Chain naturally: minimal overhead since no data copy.

### 4.10 Linalg Backward Integration

Registered through the **same** `derivatives.yaml` mechanism. Not special-cased.

```yaml
- name: _linalg_svd(Tensor A, ...) -> (Tensor U, Tensor S, Tensor Vh)
  A: "svd_backward(grad_U, grad_S, grad_Vh, U, S, Vh)"
  U, S, Vh: linalg_svd_jvp(A_t, U, S, Vh, full_matrices)
```

Generated nodes (`LinalgSvdBackward0`, `LinalgQrBackward0`, etc.) participate
in the graph identically to simple operations. Complex math lives in
`torch/csrc/autograd/FunctionsManual.cpp`.

---

## 5. Device Management and C API

### 5.1 Device Abstraction

`c10::Device` = `(DeviceType, DeviceIndex)` — 2-byte value type.

```cpp
Device(kCUDA, 0)             // CUDA device 0
Device("cuda:1")             // from string
device.type(), device.index() // accessors
device.is_cuda(), device.is_cpu() // predicates
```

`PrivateUse1` slot allows custom backends without modifying the enum.

### 5.2 Memory Management

**Allocator interface** (`c10::Allocator`):
```cpp
virtual DataPtr allocate(size_t n) = 0;
virtual void copy_data(void* dest, const void* src, size_t count) const = 0;
```

`DataPtr` = data pointer + device + deleter + context.

**CUDA caching allocator** key features:
- Two pools: small (<1 MB) and large (≥1 MB)
- Best-fit allocation with size rounding (512 bytes)
- Block splitting and merging
- Stream-aware: `record_stream()` prevents premature reuse
- OOM recovery: frees all cached blocks and retries
- Configurable via `PYTORCH_CUDA_ALLOC_CONF`

### 5.3 Device Transfer

```cpp
auto t_gpu = t.to(torch::kCUDA);
auto t_cpu = t_gpu.to(torch::kCPU);
auto t_gpu = t.to(device, /*non_blocking=*/true);  // async with pinned memory
```

`to()` returns self if no conversion needed (zero-copy).

### 5.4 CUDA Streams and Events

**Stream**: ordered sequence of GPU operations. Operations on different streams may
execute concurrently.

```cpp
CUDAStreamGuard guard(my_stream);  // RAII; restores previous on destruction
stream.synchronize();               // blocks CPU
stream.query();                     // non-blocking check
```

**Event**: marks points in a stream for synchronization and timing.

```cpp
CUDAEvent event;
event.record(stream_a);
stream_b.wait_event(event);  // GPU stream_b waits for event
float ms = event.elapsed_time(end_event);
```

### 5.5 C++ API (libtorch) Layers

| Layer | Namespace | Purpose |
|---|---|---|
| C10 | `c10::` | Core: Device, Allocator, Error, ScalarType |
| ATen | `at::` | Tensor ops, no autograd |
| Torch | `torch::` | Full API with autograd |
| C++ Frontend | `torch::nn::` | Module system (mirrors Python nn) |
| JIT | `torch::jit::` | TorchScript (deprecated) |

`at::Tensor` = ATen tensor (no autograd). `torch::Tensor` = wraps with autograd.

### 5.6 Stable C/C++ ABI (PyTorch 2.10+)

Three layers:

| Layer | Description |
|---|---|
| Header-only (`torch/headeronly/`) | No libtorch.so link needed |
| Stable C++ (`torch/csrc/stable/`) | ABI-stable tensor wrapper, op interface |
| C shim (`c_shim.h`) | Pure C API, **2-year compatibility guarantee** |

### 5.7 Extension Mechanism

```cpp
TORCH_LIBRARY(my_ops, m) {
  m.def("myop(Tensor a, Tensor b) -> Tensor");
}
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
  m.impl("myop", &myop_cpu);
}
TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
  m.impl("myop", &myop_cuda);
}
```

Compilation: `torch.utils.cpp_extension` (ahead-of-time or JIT).

### 5.8 Error Handling

Exception hierarchy rooted at `c10::Error` (inherits `std::exception`):

| Subtype | Usage |
|---|---|
| `c10::IndexError` | Index out of bounds |
| `c10::ValueError` | Invalid value |
| `c10::TypeError` | Type mismatch |
| `c10::LinAlgError` | Linalg failures |
| `c10::OutOfMemoryError` | Allocation failure |
| `c10::NotImplementedError` | Unimplemented |

Macros: `TORCH_CHECK(cond, msg)`, `TORCH_CHECK_INDEX(...)`, `TORCH_INTERNAL_ASSERT(...)`.

### 5.9 Serialization

`torch.save` produces ZIP64 archive with pickled metadata + raw tensor storage.
Storage sharing preserved. `weights_only=True` default since PyTorch 2.6.

---

## 6. Mapping to tenferro Crates

This section maps PyTorch features to the tenferro crate where they are (or will be)
implemented.

| PyTorch Feature | tenferro Crate | Notes |
|---|---|---|
| **Tensor type, storage, views** | `tenferro-tensor` | `Tensor<T>`, `DataBuffer<T>`, view ops |
| **Device enum, errors** | `tenferro-device` | `Device`, `Error`/`Result` |
| **Algebra dispatch** | `tenferro-algebra` | `HasAlgebra`, `Semiring` (compile-time, not runtime) |
| **TensorPrims (GEMM, reduce, etc.)** | `tenferro-prims` | `TensorPrims<A>` trait (plan-based execution) |
| **Einsum with contraction tree** | `tenferro-einsum` | `Subscripts`, `ContractionTree`, opt_einsum-style optimization |
| **Linalg decompositions + AD** | `tenferro-linalg` | SVD/QR/LU/eigen with `(m, n, *)` col-major convention |
| **AD core traits** | `chainrules-core` | `Differentiable`, `ReverseRule`, `ForwardRule` |
| **AD tape engine** | `chainrules` | `Tape`, `TrackedTensor`, `DualTensor`, `pullback` |
| **C FFI** | `tenferro-capi` | Opaque handles, DLPack interop |
| **Tropical algebras** | `tenferro-tropical` | MaxPlus, MinPlus, MaxMul |

### Key Design Differences from PyTorch

| Aspect | PyTorch | tenferro |
|---|---|---|
| **Memory layout** | Row-major (C-contiguous) default | Col-major (Fortran-contiguous) default |
| **Batch convention** | `(*, m, n)` — last 2 dims | `(m, n, *)` — first 2 dims |
| **Type dispatch** | Runtime (dynamic dtype/device) | Compile-time generics `Tensor<T>` |
| **Algebra dispatch** | N/A (always standard arithmetic) | `TensorPrims<A>` parameterized by algebra |
| **AD system** | Integrated (autograd built into Tensor) | Separated (chainrules-core + chainrules) |
| **Backend dispatch** | Runtime dispatcher with dispatch keys | Trait-based static dispatch |

---

## Sources

### Core Tensor API
- [PyTorch C++ API Documentation](https://docs.pytorch.org/cppdocs/)
- [at::Tensor Class Reference](https://docs.pytorch.org/cppdocs/api/classat_1_1_tensor.html)
- [Tensor Creation API (C++)](https://docs.pytorch.org/cppdocs/notes/tensor_creation.html)
- [Tensor Basics (C++)](https://docs.pytorch.org/cppdocs/notes/tensor_basics.html)
- [Tensor Indexing API (C++)](https://docs.pytorch.org/cppdocs/notes/tensor_indexing.html)
- [Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)
- [Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
- [c10::TensorImpl.h](https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.h)
- [c10::ScalarType.h](https://github.com/pytorch/pytorch/blob/main/c10/core/ScalarType.h)
- [ATen DLConvertor.h](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/DLConvertor.h)

### Einsum, BLAS, Contraction
- [torch.einsum documentation](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html)
- [Optimize torch.einsum — Issue #60295](https://github.com/pytorch/pytorch/issues/60295)
- [opt_einsum documentation](https://optimized-einsum.readthedocs.io/en/stable/)
- [torch.matmul documentation](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
- [torch.tensordot documentation](https://docs.pytorch.org/docs/stable/generated/torch.tensordot.html)
- [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
- [PyTorch dispatcher walkthrough](https://github.com/pytorch/pytorch/wiki/PyTorch-dispatcher-walkthrough)
- [Let's talk about the PyTorch dispatcher (Edward Z. Yang)](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)
- [TensorIterator Internals](https://labs.quansight.org/blog/2020/04/pytorch-tensoriterator-internals)

### Linear Algebra
- [torch.linalg documentation](https://docs.pytorch.org/docs/stable/linalg.html)
- [torch.linalg autograd blog post](https://pytorch.org/blog/torch-linalg-autograd/)
- [derivatives.yaml](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)
- [BatchLinearAlgebra.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/BatchLinearAlgebra.cpp)
- [Forward mode AD for linalg — Issue #64545](https://github.com/pytorch/pytorch/issues/64545)
- [SVD backward instability — Issue #49886](https://github.com/pytorch/pytorch/issues/49886)

### Autograd
- [How Computational Graphs are Constructed (blog)](https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/)
- [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [Forward-mode AD tutorial](https://docs.pytorch.org/tutorials/intermediate/forward_ad_usage.html)
- [Custom autograd functions](https://docs.pytorch.org/docs/stable/notes/extending.func.html)
- [torch.autograd documentation](https://docs.pytorch.org/docs/stable/autograd.html)
- [Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html)
- [Jacobians, Hessians tutorial](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html)

### Device and C API
- [Device struct (C++ docs)](https://docs.pytorch.org/cppdocs/api/structc10_1_1_device.html)
- [Allocator.h](https://github.com/pytorch/pytorch/blob/main/c10/core/Allocator.h)
- [CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)
- [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [LibTorch Stable ABI](https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html)
- [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [Exception.h](https://github.com/pytorch/pytorch/blob/main/c10/util/Exception.h)
- [Serialization semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
