# Linalg Backend Abstraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the hard-wired `FaerOps` trait with a `LinalgBackend<T>` trait behind feature flags, so tenferro-linalg can support multiple CPU backends (faer, LAPACK) and eventually GPU backends.

**Architecture:** Introduce a `LinalgBackend<T>` trait with pre-allocated output buffers and `&mut self` context. Move faer code into a feature-gated module. All public API functions take an explicit `&mut B: LinalgBackend<T>` parameter. AD rules also take the backend parameter since they call decompositions internally.

**Tech Stack:** Rust, faer, tenferro-device (Error/Result), Cargo feature flags

**Design doc:** `docs/plans/2026-02-22-linalg-backend-design.md`

---

### Task 1: Create `LinalgBackend<T>` trait definition

**Files:**
- Create: `tenferro-linalg/src/backend/mod.rs`
- Modify: `tenferro-linalg/src/lib.rs` (change `mod backend;` to `mod backend;` pointing to directory)

**Step 1: Create `backend/mod.rs` with the trait**

Create `tenferro-linalg/src/backend/mod.rs`:

```rust
//! Backend abstraction for linear algebra operations.
//!
//! Each backend implements [`LinalgBackend<T>`] for the scalar types it supports.
//! Backend selection is compile-time via Cargo feature flags.

#[cfg(feature = "faer")]
pub mod faer_backend;
#[cfg(feature = "faer")]
pub use faer_backend::FaerBackend;

use tenferro_device::Result;

/// Backend trait for linear algebra decompositions.
///
/// Methods take pre-allocated output slices in column-major layout.
/// `&mut self` allows backends to hold reusable workspace buffers.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::{LinalgBackend, FaerBackend};
///
/// let mut backend = FaerBackend::new();
/// let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
/// let mut u = vec![0.0; 4];
/// let mut s = vec![0.0; 2];
/// let mut vt = vec![0.0; 4];
/// backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
/// ```
pub trait LinalgBackend<T: Copy + 'static> {
    /// Real scalar type (f64 for both f64 and Complex64).
    type Real: Copy + 'static;

    /// Thin SVD: A = U diag(S) V^H.
    /// - a: m*n col-major input (read-only)
    /// - u: m*k col-major output, k = min(m,n)
    /// - s: k singular values output (descending)
    /// - vt: k*n col-major output
    fn thin_svd(
        &mut self,
        a: &[T],
        m: usize,
        n: usize,
        u: &mut [T],
        s: &mut [Self::Real],
        vt: &mut [T],
    ) -> Result<()>;

    /// QR: A = Q R.
    /// - q: m*k col-major output, k = min(m,n)
    /// - r: k*n col-major output
    fn qr(
        &mut self,
        a: &[T],
        m: usize,
        n: usize,
        q: &mut [T],
        r: &mut [T],
    ) -> Result<()>;

    /// LU with partial pivoting: P A = L U.
    /// - perm: m-element permutation output
    /// - l: m*k col-major output, k = min(m,n)
    /// - u_out: k*n col-major output
    fn lu(
        &mut self,
        a: &[T],
        m: usize,
        n: usize,
        perm: &mut [usize],
        l: &mut [T],
        u_out: &mut [T],
    ) -> Result<()>;

    /// Cholesky: A = L L^H.
    /// - l: n*n col-major output
    fn cholesky(&mut self, a: &[T], n: usize, l: &mut [T]) -> Result<()>;

    /// Symmetric eigendecomposition: A = V diag(lambda) V^H.
    /// - values: n eigenvalues output (ascending)
    /// - vectors: n*n col-major output
    fn eigen_sym(
        &mut self,
        a: &[T],
        n: usize,
        values: &mut [Self::Real],
        vectors: &mut [T],
    ) -> Result<()>;

    /// Matrix multiply: C = A * B.
    /// A is m*k col-major, B is k*n col-major, C is m*n col-major output.
    fn mat_mul(
        &mut self,
        a: &[T],
        m: usize,
        k: usize,
        b: &[T],
        n: usize,
        c: &mut [T],
    ) -> Result<()>;

    /// Solve A x = b.
    /// A is n*n col-major, b is n*nrhs col-major, x is n*nrhs col-major output.
    fn solve(
        &mut self,
        a: &[T],
        b: &[T],
        n: usize,
        nrhs: usize,
        x: &mut [T],
    ) -> Result<()>;

    /// Solve triangular system.
    /// A is n*n col-major (upper or lower), b is n*nrhs col-major,
    /// x is n*nrhs col-major output.
    fn solve_triangular(
        &mut self,
        a: &[T],
        b: &[T],
        n: usize,
        nrhs: usize,
        upper: bool,
        x: &mut [T],
    ) -> Result<()>;
}

/// Compute column-major strides for given dimensions.
pub(crate) fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; dims.len()];
    if dims.is_empty() {
        return strides;
    }
    strides[0] = 1;
    for i in 1..dims.len() {
        strides[i] = strides[i - 1] * dims[i - 1] as isize;
    }
    strides
}
```

**Step 2: Verify compilation**

Run: `cargo build -p tenferro-linalg`
Expected: Compilation error because `mod backend` now expects a directory — this is expected, we'll fix it in Step 3.

**Step 3: Move old backend.rs out of the way**

- Rename `tenferro-linalg/src/backend.rs` to `tenferro-linalg/src/backend/faer_backend.rs`
- The new `backend/mod.rs` already references `pub mod faer_backend;`
- Update `lib.rs`: change `use backend::FaerOps;` to `use backend::faer_backend::FaerOps;` (temporary, will be removed later)
- Also add `pub mod backend;` to `lib.rs` (the trait needs to be publicly accessible)

Run: `cargo build -p tenferro-linalg`
Expected: Success (FaerOps still used internally, new trait exists but unused)

**Step 4: Commit**

```bash
git add tenferro-linalg/src/backend/
git commit -m "refactor(linalg): create LinalgBackend trait and backend module directory"
```

---

### Task 2: Update Cargo.toml with feature flags

**Files:**
- Modify: `tenferro-linalg/Cargo.toml`

**Step 1: Add feature flags**

Change `tenferro-linalg/Cargo.toml`:

```toml
[package]
name = "tenferro-linalg"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Batched matrix linear algebra decompositions (SVD, QR, LU, eigen) with AD rules for the tenferro workspace."
publish = false

[features]
default = ["faer"]
faer = ["dep:faer"]

[dependencies]
tenferro-device = { path = "../tenferro-device" }
tenferro-algebra = { path = "../tenferro-algebra" }
tenferro-prims = { path = "../tenferro-prims" }
tenferro-tensor = { path = "../tenferro-tensor" }
chainrules-core = { path = "../extern/chainrules-core" }
faer = { workspace = true, optional = true }
num-traits = { workspace = true }
```

**Step 2: Guard faer imports in faer_backend.rs**

Wrap `faer_backend.rs` contents with `#[cfg(feature = "faer")]` at the module level (already done in mod.rs via `#[cfg(feature = "faer")] pub mod faer_backend;`).

**Step 3: Verify compilation**

Run: `cargo build -p tenferro-linalg`
Expected: Success (faer feature is default-on)

Run: `cargo build -p tenferro-linalg --no-default-features`
Expected: Compilation errors (FaerOps not available). This is expected — we'll fix it in later tasks when we remove the FaerOps dependency from the public API.

**Step 4: Commit**

```bash
git add tenferro-linalg/Cargo.toml
git commit -m "build(linalg): add feature flag for faer backend"
```

---

### Task 3: Implement `LinalgBackend<f64>` and `LinalgBackend<f32>` for `FaerBackend`

**Files:**
- Modify: `tenferro-linalg/src/backend/faer_backend.rs`

**Step 1: Write a unit test for FaerBackend::thin_svd**

Add to `tenferro-linalg/tests/linalg_tests.rs`:

```rust
#[cfg(feature = "faer")]
mod backend_tests {
    use tenferro_linalg::backend::{FaerBackend, LinalgBackend};

    #[test]
    fn faer_backend_thin_svd_identity() {
        let mut backend = FaerBackend::new();
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity col-major
        let mut u = vec![0.0; 4];
        let mut s = vec![0.0; 2];
        let mut vt = vec![0.0; 4];
        backend.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
        // Singular values should be 1.0
        for v in &s {
            assert!((v - 1.0).abs() < 1e-10, "expected 1.0, got {v}");
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p tenferro-linalg faer_backend_thin_svd_identity`
Expected: FAIL — `FaerBackend` doesn't impl `LinalgBackend` yet.

**Step 3: Add `FaerBackend` struct and impl `LinalgBackend<f64>`**

In `faer_backend.rs`, add after the existing `FaerOps` code:

```rust
use super::LinalgBackend;
use tenferro_device::Result;

/// CPU backend using the faer linear algebra library.
///
/// # Examples
///
/// ```ignore
/// use tenferro_linalg::backend::{FaerBackend, LinalgBackend};
///
/// let mut b = FaerBackend::new();
/// let a = vec![1.0, 0.0, 0.0, 1.0];
/// let mut u = vec![0.0; 4];
/// let mut s = vec![0.0; 2];
/// let mut vt = vec![0.0; 4];
/// b.thin_svd(&a, 2, 2, &mut u, &mut s, &mut vt).unwrap();
/// ```
pub struct FaerBackend;

impl FaerBackend {
    pub fn new() -> Self {
        FaerBackend
    }
}

impl Default for FaerBackend {
    fn default() -> Self {
        Self::new()
    }
}
```

Then implement `LinalgBackend<f64>` for `FaerBackend` using a macro for both f64 and f32. Each method calls the existing faer code (same logic as `FaerOps` but writes to output slices instead of returning Vecs):

```rust
macro_rules! impl_linalg_backend {
    ($ty:ty) => {
        impl LinalgBackend<$ty> for FaerBackend {
            type Real = $ty;

            fn thin_svd(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                u: &mut [$ty],
                s: &mut [Self::Real],
                vt: &mut [$ty],
            ) -> Result<()> {
                let mat = faer::mat::from_column_major_slice(a, m, n);
                let svd = mat.thin_svd();
                let k = m.min(n);
                let u_ref = svd.u();
                let s_col = svd.s_diagonal();
                let v_ref = svd.v();
                for j in 0..k {
                    for i in 0..m {
                        u[i + j * m] = u_ref[(i, j)];
                    }
                }
                for i in 0..k {
                    s[i] = s_col[i];
                }
                // vt[i + j*k] = v[j + i*n] (transpose of V)
                for j in 0..n {
                    for i in 0..k {
                        vt[i + j * k] = v_ref[(j, i)];
                    }
                }
                Ok(())
            }

            fn qr(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                q: &mut [$ty],
                r: &mut [$ty],
            ) -> Result<()> {
                let mat = faer::mat::from_column_major_slice(a, m, n);
                let qr = mat.qr();
                let k = m.min(n);
                let q_mat = qr.compute_thin_q();
                let r_mat = qr.compute_thin_r();
                for j in 0..k {
                    for i in 0..m {
                        q[i + j * m] = q_mat[(i, j)];
                    }
                }
                for j in 0..n {
                    for i in 0..k {
                        r[i + j * k] = r_mat[(i, j)];
                    }
                }
                Ok(())
            }

            fn lu(
                &mut self,
                a: &[$ty],
                m: usize,
                n: usize,
                perm: &mut [usize],
                l: &mut [$ty],
                u_out: &mut [$ty],
            ) -> Result<()> {
                let mat = faer::mat::from_column_major_slice(a, m, n);
                let lu = mat.partial_piv_lu();
                let k = m.min(n);
                let l_mat = lu.compute_l();
                let u_mat = lu.compute_u();
                for j in 0..k {
                    for i in 0..m {
                        l[i + j * m] = l_mat[(i, j)];
                    }
                }
                for j in 0..n {
                    for i in 0..k {
                        u_out[i + j * k] = u_mat[(i, j)];
                    }
                }
                let perm_ref = lu.row_permutation();
                let (fwd, _inv) = perm_ref.arrays();
                perm.copy_from_slice(&fwd[..m]);
                Ok(())
            }

            fn cholesky(&mut self, a: &[$ty], n: usize, l: &mut [$ty]) -> Result<()> {
                let mat = faer::mat::from_column_major_slice(a, n, n);
                match mat.cholesky(faer::Side::Lower) {
                    Ok(chol) => {
                        let l_mat = chol.compute_l();
                        for j in 0..n {
                            for i in 0..n {
                                l[i + j * n] = l_mat[(i, j)];
                            }
                        }
                        Ok(())
                    }
                    Err(_) => Err(tenferro_device::Error::InvalidArgument(
                        "matrix is not positive definite".into(),
                    )),
                }
            }

            fn eigen_sym(
                &mut self,
                a: &[$ty],
                n: usize,
                values: &mut [Self::Real],
                vectors: &mut [$ty],
            ) -> Result<()> {
                let mat = faer::mat::from_column_major_slice(a, n, n);
                let eig = mat.selfadjoint_eigendecomposition(faer::Side::Lower);
                let u_ref = eig.u();
                let s_diag = eig.s();
                for j in 0..n {
                    for i in 0..n {
                        vectors[i + j * n] = u_ref[(i, j)];
                    }
                }
                let s_col = s_diag.column_vector();
                for i in 0..n {
                    values[i] = s_col[i];
                }
                Ok(())
            }

            fn mat_mul(
                &mut self,
                a: &[$ty],
                m: usize,
                k: usize,
                b: &[$ty],
                n: usize,
                c: &mut [$ty],
            ) -> Result<()> {
                let a_mat = faer::mat::from_column_major_slice(a, m, k);
                let b_mat = faer::mat::from_column_major_slice(b, k, n);
                let result = &a_mat * &b_mat;
                for j in 0..n {
                    for i in 0..m {
                        c[i + j * m] = result[(i, j)];
                    }
                }
                Ok(())
            }

            fn solve(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                x: &mut [$ty],
            ) -> Result<()> {
                let a_mat = faer::mat::from_column_major_slice(a, n, n);
                let b_mat = faer::mat::from_column_major_slice(b, n, nrhs);
                let lu = a_mat.partial_piv_lu();
                let result = lu.solve(&b_mat);
                for j in 0..nrhs {
                    for i in 0..n {
                        x[i + j * n] = result[(i, j)];
                    }
                }
                Ok(())
            }

            fn solve_triangular(
                &mut self,
                a: &[$ty],
                b: &[$ty],
                n: usize,
                nrhs: usize,
                upper: bool,
                x: &mut [$ty],
            ) -> Result<()> {
                let a_mat = faer::mat::from_column_major_slice(a, n, n);
                for col in 0..nrhs {
                    let b_col = &b[col * n..(col + 1) * n];
                    let x_col = &mut x[col * n..(col + 1) * n];
                    if upper {
                        for i in (0..n).rev() {
                            let mut sum = b_col[i];
                            for j in (i + 1)..n {
                                sum = sum - a_mat[(i, j)] * x_col[j];
                            }
                            x_col[i] = sum / a_mat[(i, i)];
                        }
                    } else {
                        for i in 0..n {
                            let mut sum = b_col[i];
                            for j in 0..i {
                                sum = sum - a_mat[(i, j)] * x_col[j];
                            }
                            x_col[i] = sum / a_mat[(i, i)];
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

impl_linalg_backend!(f64);
impl_linalg_backend!(f32);
```

**Step 4: Run the new test**

Run: `cargo test -p tenferro-linalg faer_backend_thin_svd_identity`
Expected: PASS

**Step 5: Run all existing tests (should still pass)**

Run: `cargo test -p tenferro-linalg`
Expected: All PASS (FaerOps still in use by public API)

**Step 6: Commit**

```bash
git add tenferro-linalg/
git commit -m "feat(linalg): implement LinalgBackend for FaerBackend (f64, f32)"
```

---

### Task 4: Migrate public decomposition functions to use `LinalgBackend`

This is the largest task. We migrate all 15 public functions to take `backend: &mut B` instead of using `T::method()` (FaerOps dispatch).

**Files:**
- Modify: `tenferro-linalg/src/lib.rs`
- Modify: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Update `LinalgScalar` — remove `FaerOps` bound**

In `lib.rs`, change:

```rust
// OLD (line 149-156):
#[allow(private_bounds)]
pub trait LinalgScalar:
    Scalar + Float + std::ops::Sub<Output = Self> + std::fmt::Debug + 'static + FaerOps
{
}

// NEW:
pub trait LinalgScalar:
    Scalar + Float + std::ops::Sub<Output = Self> + std::fmt::Debug + 'static
{
}
```

Remove `use backend::faer_backend::FaerOps;` import line.

**Step 2: Update `svd()` signature and body**

Change from:
```rust
pub fn svd<T: LinalgScalar>(
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T>>
```

To:
```rust
pub fn svd<T: LinalgScalar, B: LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    options: Option<&SvdOptions>,
) -> Result<SvdResult<T>>
```

Inside the batch loop, replace:
```rust
let (u_b, s_b, v_b) = T::thin_svd(batch_data, m, n);
```
with:
```rust
let mut u_full = vec![T::zero(); m * k];
let mut s_full = vec![T::zero(); k];
let mut vt_full = vec![T::zero(); k * n];
backend.thin_svd(batch_data, m, n, &mut u_full, &mut s_full, &mut vt_full)?;
```

Then update the copy-out code to read from `u_full`, `s_full`, `vt_full` instead of `u_b`, `s_b`, `v_b`. Note: `v_b` was V (n*k), but the new backend returns `vt_full` which is Vt (k*n). The copy-out for Vt changes from `v_b[j + i * n]` to `vt_full[i + j * k]` (direct copy).

**Step 3: Repeat for all other public functions**

Apply the same pattern to each function (add `backend: &mut B` parameter, replace `T::method()` calls with `backend.method()` calls):

| Function | Line | FaerOps call to replace |
|----------|------|------------------------|
| `svd` | 652 | `T::thin_svd` |
| `qr` | 751 | `T::qr_decomp` |
| `lu` | 815 | `T::lu_decomp` |
| `eigen` | 882 | `T::eigen_sym` |
| `lstsq` | 940 | Uses `qr` + `T::mat_solve_triangular` internally |
| `cholesky` | 1025 | `T::cholesky_decomp` |
| `solve` | 1065 | `T::mat_solve` |
| `inv` | 1106 | `T::mat_solve` |
| `det` | 1148 | `T::lu_decomp` |
| `slogdet` | 1224 | `T::lu_decomp` |
| `eig` | 1312 | returns error (no change needed except signature) |
| `pinv` | 1339 | Uses `svd` internally |
| `matrix_exp` | 1416 | returns error (no change needed except signature) |
| `solve_triangular` | 1442 | `T::mat_solve_triangular` |
| `norm` | 1495 | Uses `svd` internally for Nuclear/Spectral |

For functions that call other public functions internally (e.g., `lstsq` calls `qr`, `pinv` calls `svd`, `norm` calls `svd`), pass the `backend` through.

**Step 4: Update all tests**

In `linalg_tests.rs`, every test that calls `svd(&a, None)` becomes `svd(&mut FaerBackend::new(), &a, None)`.

Add import at top:
```rust
#[cfg(feature = "faer")]
use tenferro_linalg::backend::FaerBackend;
```

Pattern for each test:
```rust
// OLD:
let result = svd(&a, None).unwrap();

// NEW:
let mut backend = FaerBackend::new();
let result = svd(&mut backend, &a, None).unwrap();
```

**Step 5: Run all tests**

Run: `cargo test -p tenferro-linalg`
Expected: All PASS

**Step 6: Commit**

```bash
git add tenferro-linalg/
git commit -m "refactor(linalg): migrate all decompositions to LinalgBackend"
```

---

### Task 5: Migrate AD rules to use `LinalgBackend`

**Files:**
- Modify: `tenferro-linalg/src/lib.rs` (lines 1850-4052)
- Modify: `tenferro-linalg/tests/linalg_tests.rs`

**Step 1: Update AD rrule/frule signatures**

All 28 AD functions get `backend: &mut B` as first parameter.

Example for `svd_rrule`:
```rust
// OLD:
pub fn svd_rrule<T: LinalgScalar>(
    tensor: &Tensor<T>,
    cotangent: &SvdCotangent<T>,
    options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>>

// NEW:
pub fn svd_rrule<T: LinalgScalar, B: LinalgBackend<T, Real = T>>(
    backend: &mut B,
    tensor: &Tensor<T>,
    cotangent: &SvdCotangent<T>,
    options: Option<&SvdOptions>,
) -> AdResult<Tensor<T>>
```

**Step 2: Replace `T::mat_mul(...)` calls in AD code**

The AD rules use `T::mat_mul(a, m, k, b, n)` (returns `Vec<T>`). These must change to:
```rust
let mut result = vec![T::zero(); m * n];
backend.mat_mul(a, m, k, b, n, &mut result)?;
```

Similarly replace `T::mat_solve(...)` and `T::mat_solve_triangular(...)`.

Note: The AD functions call `svd()`, `qr()`, etc. internally. These now require `backend`, which is already passed in.

There are many `T::mat_mul` calls in the AD code (roughly 50+). Each one needs the same transformation: allocate output buffer, call `backend.mat_mul()`.

Consider creating a helper:
```rust
fn backend_mat_mul<T: LinalgScalar, B: LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &[T], m: usize, k: usize,
    b: &[T], n: usize,
) -> std::result::Result<Vec<T>, chainrules_core::AutodiffError> {
    let mut c = vec![T::zero(); m * n];
    backend.mat_mul(a, m, k, b, n, &mut c)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok(c)
}
```

This keeps the diff smaller — replace `T::mat_mul(a, m, k, b, n)` with `backend_mat_mul(backend, a, m, k, b, n)?`.

Similarly for solve:
```rust
fn backend_solve<T: LinalgScalar, B: LinalgBackend<T, Real = T>>(
    backend: &mut B,
    a: &[T], b: &[T], n: usize, nrhs: usize,
) -> std::result::Result<Vec<T>, chainrules_core::AutodiffError> {
    let mut x = vec![T::zero(); n * nrhs];
    backend.solve(a, b, n, nrhs, &mut x)
        .map_err(|e| chainrules_core::AutodiffError::InvalidArgument(e.to_string()))?;
    Ok(x)
}
```

**Step 3: Update AD tests**

Same pattern: add `&mut FaerBackend::new()` as first argument to all AD rule calls.

**Step 4: Run tests**

Run: `cargo test -p tenferro-linalg`
Expected: All PASS

**Step 5: Commit**

```bash
git add tenferro-linalg/
git commit -m "refactor(linalg): migrate AD rules to LinalgBackend"
```

---

### Task 6: Remove old FaerOps trait

**Files:**
- Modify: `tenferro-linalg/src/backend/faer_backend.rs`

**Step 1: Delete the FaerOps trait and its implementations**

Remove the `FaerOps` trait definition (lines 9-61 of old backend.rs), the `impl_faer_ops!` macro (lines 63-254), and the `impl_faer_ops!(f64)` / `impl_faer_ops!(f32)` invocations (lines 257-258).

Keep the `col_major_strides` function (move to `backend/mod.rs` if needed — it's already there in the new mod.rs).

**Step 2: Remove any remaining FaerOps references in lib.rs**

Search for `FaerOps` in lib.rs and remove any leftover imports or uses.

**Step 3: Run tests**

Run: `cargo test -p tenferro-linalg`
Expected: All PASS

**Step 4: Commit**

```bash
git add tenferro-linalg/
git commit -m "refactor(linalg): remove old FaerOps trait"
```

---

### Task 7: Update tenferro-capi

**Files:**
- Modify: `tenferro-capi/src/lib.rs`
- Modify: `tenferro-capi/tests/capi_tests.rs`
- Modify: `tenferro-capi/Cargo.toml` (may need to add `faer` feature forwarding)

**Step 1: Update capi imports**

Change:
```rust
use tenferro_linalg::{svd, svd_frule, svd_rrule, SvdCotangent, SvdOptions};
```
Add:
```rust
use tenferro_linalg::backend::FaerBackend;
```

**Step 2: Update svd call in capi (line 1310)**

```rust
// OLD:
let result = svd(&matrix, opts.as_ref()).map_err(|e| map_device_error(&e))?;

// NEW:
let mut backend = FaerBackend::new();
let result = svd(&mut backend, &matrix, opts.as_ref()).map_err(|e| map_device_error(&e))?;
```

**Step 3: Update svd_rrule and svd_frule calls**

Similarly add `&mut FaerBackend::new()` to the rrule (line 1453) and frule (line 1552) calls.

**Step 4: Update capi tests**

Same pattern for test lines 930 and 1008.

**Step 5: Run capi tests**

Run: `cargo test -p tenferro-capi`
Expected: All PASS

**Step 6: Commit**

```bash
git add tenferro-capi/
git commit -m "refactor(capi): update to use LinalgBackend"
```

---

### Task 8: Update docstrings and examples

**Files:**
- Modify: `tenferro-linalg/src/lib.rs` (crate-level docs and per-function docs)
- Modify: `tenferro-linalg/src/backend/mod.rs`

**Step 1: Update crate-level doc examples**

All `/// # Examples` blocks in lib.rs that show `svd(&a, None)` must change to `svd(&mut FaerBackend::new(), &a, None)`.

**Step 2: Update per-function doc examples**

Same pattern for each function's docstring.

**Step 3: Update AD rule doc examples**

Same pattern for rrule/frule examples.

**Step 4: Run doc tests**

Run: `cargo test -p tenferro-linalg --doc`
Expected: All PASS (or all `ignore`-tagged)

**Step 5: Commit**

```bash
git add tenferro-linalg/
git commit -m "docs(linalg): update examples for LinalgBackend API"
```

---

### Task 9: Update design doc and run full CI checks

**Files:**
- Modify: `docs/design/linalg.md` (if it references old API)

**Step 1: Format check**

Run: `cargo fmt --all --check`
Fix any issues with `cargo fmt --all`.

**Step 2: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All PASS

**Step 3: Run coverage check**

Run: `cargo llvm-cov --workspace --json --output-path coverage.json && python3 scripts/check-coverage.py coverage.json`
Expected: PASS

**Step 4: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore(linalg): format and coverage fixes"
```

---

### Summary of Tasks

| Task | Description | Est. Complexity |
|------|-------------|-----------------|
| 1 | Create LinalgBackend trait + backend module directory | Small |
| 2 | Add feature flags to Cargo.toml | Small |
| 3 | Implement LinalgBackend for FaerBackend | Medium |
| 4 | Migrate 15 public functions to LinalgBackend | Large |
| 5 | Migrate 28 AD rules to LinalgBackend | Large |
| 6 | Remove old FaerOps trait | Small |
| 7 | Update tenferro-capi | Small |
| 8 | Update docstrings and examples | Medium |
| 9 | Full CI checks | Small |
