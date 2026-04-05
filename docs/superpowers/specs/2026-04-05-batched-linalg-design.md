# Batched Linalg Design

## Goals

1. Add trailing-batch convention to tensor4all-meta as a cross-repo standard
2. Extend linalg ops (cholesky, svd, qr, eigh, solve) to support batched inputs
3. Add Complex64 support alongside existing f64

## Batch Convention

Trailing (rightmost) batch dims, col-major storage. Each batch slice is contiguous.

```
Shape: [M, N, B1, B2, ...]
       ^^^^^^ core (leftmost)
              ^^^^^^^^^^^^^ batch (rightmost, contiguous in col-major)
```

This differs from JAX/NumPy/PyTorch which use leading-batch `[B, M, N]`.
The choice matches tenferro's col-major storage for zero-copy batch slicing.

### Per-op shapes

| Op | Core rank | Input | Outputs |
|---|---|---|---|
| cholesky | 2 | `[N, N, B...]` | `[N, N, B...]` |
| svd | 2 | `[M, N, B...]` | U `[M, K, B...]`, S `[K, B...]`, Vt `[K, N, B...]` |
| qr | 2 | `[M, N, B...]` | Q `[M, K, B...]`, R `[K, N, B...]` |
| eigh | 2 | `[N, N, B...]` | vals `[N, B...]`, vecs `[N, N, B...]` |
| solve | 2+2 | A `[N, N, B...]`, b `[N, M, B...]` | `[N, M, B...]` |

When `shape.len() == core_rank`, it's a plain 2D call (no batch). Zero overhead.

## Implementation

### Batch loop helper

```rust
fn batched_linalg_single<T, F>(
    input: &TypedTensor<T>,
    core_rank: usize,
    kernel: F,
) -> TypedTensor<T>
where
    T: Clone,
    F: Fn(&TypedTensor<T>) -> TypedTensor<T>,
{
    if input.shape.len() <= core_rank {
        return kernel(input);
    }
    let core_shape = &input.shape[..core_rank];
    let batch_shape = &input.shape[core_rank..];
    let slice_size: usize = core_shape.iter().product();
    let batch_total: usize = batch_shape.iter().product();
    let data = input.host_data();

    let mut result_data = Vec::with_capacity(slice_size * batch_total);
    for b in 0..batch_total {
        let slice = &data[b * slice_size..(b + 1) * slice_size];
        let slice_tensor = TypedTensor::from_vec(core_shape.to_vec(), slice.to_vec());
        let out = kernel(&slice_tensor);
        result_data.extend_from_slice(out.host_data());
    }
    let mut out_shape = /* kernel output core shape */ ;
    out_shape.extend_from_slice(batch_shape);
    TypedTensor::from_vec(out_shape, result_data)
}
```

Similar `batched_linalg_multi` for multi-output ops (svd, qr, eigh).
Similar `batched_linalg_binary` for solve (two inputs with matching batch dims).

### Zero-copy slicing

Col-major + trailing batch means batch slice `b` occupies
`data[b * slice_size .. (b+1) * slice_size]` contiguously.
No gather or permutation needed.

## dtype Support

### f64 (existing)

Already implemented. Extend with batch loop.

### Complex64 (new)

faer uses `faer::complex_native::c64` internally. Conversion needed:

```rust
use num_complex::Complex64;

// Complex64 and c64 have identical memory layout (two f64s).
// Safe transmute for zero-copy: &[Complex64] → &[c64]
```

faer's `MatRef::from_column_major_slice` works with `c64`. The linalg functions
(llt, thin_svd, qr, self_adjoint_eigen, partial_piv_lu) all support `c64`.

Implementation: duplicate each kernel function for Complex64, or use a trait
to abstract over f64/c64. Recommended: trait-based.

```rust
trait FaerLinalg: Sized {
    fn cholesky_impl(input: &TypedTensor<Self>) -> TypedTensor<Self>;
    fn svd_impl(input: &TypedTensor<Self>) -> Vec<TypedTensor<Self>>;
    // ...
}

impl FaerLinalg for f64 { ... }
impl FaerLinalg for Complex64 { ... }
```

### CpuBackend dispatch

```rust
fn cholesky(&mut self, input: &Tensor) -> Tensor {
    match input {
        Tensor::F64(t) => Tensor::F64(linalg::cholesky(t)),
        Tensor::C64(t) => Tensor::C64(linalg::cholesky(t)),
        _ => todo!("cholesky: unsupported dtype"),
    }
}
```

## TensorBackend trait

No change to trait signatures. Batch is determined by input shape at runtime.

## tensor4all-meta update

Add to `tensor4all-meta/docs/design-v2/spec/tensor-semantics.md`:

```markdown
## Linalg Batch Convention

Linalg ops follow trailing-batch convention: core matrix dims are leftmost,
batch dims are rightmost. Shape `[M, N, B1, B2, ...]` means `B1*B2*...`
independent M x N matrices. Each batch slice is contiguous in col-major memory.

This differs from JAX/NumPy/PyTorch which use leading-batch `[B, M, N]`.
The choice matches tenferro's col-major storage for zero-copy batch slicing.
```

## Testing

1. Batched cholesky: `[3, 3, 2]` SPD matrices, verify L @ L^T = A per batch
2. Batched SVD: `[4, 3, 2]`, verify U @ diag(S) @ Vt = A per batch
3. Batched QR: `[4, 3, 2]`, verify Q @ R = A per batch
4. Batched solve: A `[3, 3, 2]`, b `[3, 1, 2]`, verify A @ x = b per batch
5. Complex64 cholesky: Hermitian PD matrix
6. Complex64 SVD: complex matrix reconstruction
7. Existing 2D f64 tests continue to pass

## Files changed

| File | Action |
|---|---|
| `tenferro-tensor/src/cpu/linalg/faer_linalg.rs` | Add batch loop, Complex64 support, FaerLinalg trait |
| `tenferro-tensor/src/cpu/backend.rs` | Add C64 dispatch for linalg methods |
| `tenferro-tensor/src/tests/cpu_tests.rs` | Add batched + complex linalg tests |
| `tensor4all-meta/docs/design-v2/spec/tensor-semantics.md` | Add batch convention |
