//! Typed tensor computation facade for tenferro-rs.
//!
//! This crate re-exports the most commonly used items from the lower-level
//! tenferro crates so that downstream users need only a single dependency
//! for typed `Tensor<T>` computation with einsum and linear algebra.
//!
//! # When to use this crate
//!
//! | Crate | Use when |
//! |-------|----------|
//! | **`tenferro-tensor-compute`** | You want typed `Tensor<T>` with einsum and linalg |
//! | `tenferro` | You need automatic differentiation (VJP/JVP) |
//! | `tenferro-tensor` | You only need the data type (library/crate authors) |
//!
//! # Examples
//!
//! Matrix multiplication via einsum, then SVD:
//!
//! ```
//! use tenferro_tensor_compute::{
//!     einsum, CpuBackend, CpuContext, MemoryOrder, Standard, Tensor,
//! };
//!
//! let col = MemoryOrder::ColumnMajor;
//! // 4 threads for parallel operations
//! let mut ctx = CpuContext::new(4);
//!
//! // Column-major: data is stored column-first.
//! // Use Tensor::from_row_major_slice if your data is row-major.
//! let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], col).unwrap();
//! let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], col).unwrap();
//!
//! // Standard<f64> = standard arithmetic (multiply and add) over f64
//! let c = einsum::<Standard<f64>, CpuBackend>(
//!     &mut ctx, "ij,jk->ik", &[&a, &b], None,
//! ).unwrap();
//!
//! // Read values back
//! assert_eq!(c.dims(), &[2, 2]);
//! assert_eq!(c.get(&[0, 0]), Some(&23.0));
//! let values = c.to_vec(); // column-major order
//! assert_eq!(values, vec![23.0, 34.0, 31.0, 46.0]);
//! ```
//!
//! SVD decomposition (requires `linalg` feature, enabled by default):
//!
//! ```
//! # #[cfg(feature = "linalg")]
//! # {
//! use tenferro_tensor_compute::{svd, CpuContext, MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(1);
//! let a = Tensor::<f64>::from_slice(
//!     &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], MemoryOrder::ColumnMajor,
//! ).unwrap();
//! let result = svd(&mut ctx, &a, None).unwrap();
//! assert_eq!(result.u.dims(), &[2, 2]);
//! assert_eq!(result.s.dims(), &[2]);
//! assert_eq!(result.vt.dims(), &[2, 3]);
//! # }
//! ```

// --- Tensor type and layout ---
pub use tenferro_tensor::{MemoryOrder, StructuredTensor, Tensor};

// --- Device / memory ---
pub use tenferro_device::LogicalMemorySpace;

// --- Algebra ---
pub use tenferro_algebra::Standard;

// --- Backend ---
pub use tenferro_einsum::BackendContext;
pub use tenferro_prims::{CpuBackend, CpuContext};

// --- Einsum ---
pub use tenferro_einsum::{einsum, einsum_into, einsum_with_subscripts, Subscripts};

// --- Linalg (behind feature flag) ---
#[cfg(feature = "linalg")]
pub use tenferro_linalg::{
    cholesky, cross, det, eig, eigen, inv, lstsq, lu, lu_factor, lu_solve, matrix_exp,
    matrix_power, norm, pinv, qr, slogdet, solve, solve_triangular, svd, svdvals, tensorinv,
    tensorsolve, vander, CholeskyExResult, EigResult, EigenResult, LstsqResult, LuFactorExResult,
    LuFactorResult, LuResult, NormKind, QrResult, SlogdetResult, SolveExResult, SvdOptions,
    SvdResult,
};
