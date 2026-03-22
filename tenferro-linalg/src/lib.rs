#![allow(clippy::multiple_bound_locations)]

//! Batched matrix linear algebra decompositions with AD rules.
//!
//! CPU decompositions and solvers are fully implemented via the
//! [`faer`](https://crates.io/crates/faer) backend. CUDA/HIP linalg contracts
//! are already part of the public surface, but backend coverage there remains
//! partial and capability-gated.
//!
//! This crate provides SVD, QR, LU, eigendecomposition, Cholesky, least squares,
//! linear solve, matrix inverse, determinant, pseudoinverse, matrix exponential,
//! triangular solve, and norms for tensors
//! with shape `(m, n, *)`, adapted from PyTorch's `torch.linalg` for
//! column-major layout:
//!
//! - **First 2 dimensions** are the matrix (`m × n`).
//! - **All following dimensions** (`*`) are independent batch dimensions.
//! - Inputs are **internally normalized** to column-major contiguous layout.
//!   If an input is not already contiguous, an internal copy is performed.
//!   Calling `.contiguous(ColumnMajor)` explicitly is optional but useful
//!   when you want to control exactly where copies happen.
//!
//! This convention mirrors PyTorch's `(*, m, n)` but is flipped for
//! col-major: in col-major the first dimensions are contiguous, so
//! placing the matrix there ensures LAPACK can operate directly without
//! transposition.
//!
//! This module is **context-agnostic**: it does not know about tensor
//! networks, MPS, or any specific application. If you need to decompose
//! a tensor along arbitrary legs, `permute` + `reshape` before calling
//! these functions.
//!
//! # AD rules
//!
//! Each decomposition has stateless `_rrule` (reverse-mode / VJP) and
//! `_frule` (forward-mode / JVP) functions. These implement matrix-level
//! AD formulas (Mathieu 2019 et al.) using batched operations that
//! naturally broadcast over batch dimensions `*`.
//!
//! There are no `tracked_*` / `dual_*` functions — the chainrules tape
//! engine composes `permute_backward` + `reshape_backward` + `svd_rrule`
//! via the standard chain rule automatically.
//!
//! # Examples
//!
//! ## SVD of a matrix
//!
//! ```
//! use tenferro_linalg::{svd, SvdOptions};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! // 2D matrix: shape [3, 4]
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let result = svd(&mut ctx, &a, None).unwrap();
//! // result.u:  shape [3, 3]  (m × k, k = min(m,n) = 3)
//! // result.s:  shape [3]     (singular values)
//! // result.vt: shape [3, 4]  (k × n)
//! ```
//!
//! ## Batched SVD
//!
//! ```
//! use tenferro_linalg::svd;
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! // Batched: shape [m, n, batch] = [3, 4, 10]
//! let a = Tensor::<f64>::zeros(&[3, 4, 10], mem, col);
//! let result = svd(&mut ctx, &a, None).unwrap();
//! // result.u:  shape [3, 3, 10]
//! // result.s:  shape [3, 10]
//! // result.vt: shape [3, 4, 10]
//! ```
//!
//! ## Decomposing a 4D tensor along specific legs
//!
//! ```
//! use tenferro_linalg::svd;
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! // 4D tensor [2, 3, 4, 5] — want SVD with left=[0,1], right=[2,3]
//! let t = Tensor::<f64>::zeros(&[2, 3, 4, 5], mem, col);
//!
//! // permute + reshape (contiguous is handled internally, but can be called explicitly)
//! let mat = t.permute(&[0, 1, 2, 3]).unwrap()  // already in order
//!            .reshape(&[6, 20]).unwrap();        // m = 2*3 = 6, n = 4*5 = 20
//! let result = svd(&mut ctx, &mat, None).unwrap();
//! // Then reshape result.u, result.vt back to desired tensor shape
//! ```
//!
//! ## Reverse-mode AD (stateless rrule)
//!
//! ```
//! use tenferro_linalg::{svd, svd_rrule, SvdCotangent};
//! use tenferro_prims::CpuContext;
//! use tenferro_tensor::{Tensor, MemoryOrder};
//! use tenferro_device::LogicalMemorySpace;
//!
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let mut ctx = CpuContext::new(1);
//!
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let result = svd(&mut ctx, &a, None).unwrap();
//!
//! // Full cotangent: gradient through U, S, and Vt
//! let cotangent = SvdCotangent {
//!     u: Some(Tensor::ones(&[3, 3], mem, col)),
//!     s: Some(Tensor::ones(&[3], mem, col)),
//!     vt: Some(Tensor::ones(&[3, 4], mem, col)),
//! };
//! let grad_a = svd_rrule(&mut ctx, &a, &cotangent, None).unwrap();
//! // grad_a has same shape as a: [3, 4]
//!
//! // Partial cotangent: gradient only through singular values (always stable)
//! let cotangent_s_only = SvdCotangent {
//!     u: None,
//!     s: Some(Tensor::ones(&[3], mem, col)),
//!     vt: None,
//! };
//! let grad_a2 = svd_rrule(&mut ctx, &a, &cotangent_s_only, None).unwrap();
//! ```

#[cfg(all(feature = "provider-src", not(feature = "linalg-lapack")))]
compile_error!("provider-src requires linalg-lapack");
#[cfg(all(feature = "provider-inject", not(feature = "linalg-lapack")))]
compile_error!("provider-inject requires linalg-lapack");
#[cfg(all(
    any(
        feature = "src-openblas",
        feature = "src-netlib",
        feature = "src-accelerate",
        feature = "src-r",
        feature = "src-intel-mkl-dynamic-sequential",
        feature = "src-intel-mkl-dynamic-parallel",
        feature = "src-intel-mkl-static-sequential",
        feature = "src-intel-mkl-static-parallel"
    ),
    not(feature = "linalg-lapack")
))]
compile_error!("src-* features require linalg-lapack and provider-src");

#[cfg(feature = "linalg-lapack")]
const _: () = {
    let provider_count =
        (cfg!(feature = "provider-src") as usize) + (cfg!(feature = "provider-inject") as usize);
    assert!(
        provider_count == 1,
        "linalg-lapack requires exactly one provider: provider-src or provider-inject"
    );

    let src_count = (cfg!(feature = "src-openblas") as usize)
        + (cfg!(feature = "src-netlib") as usize)
        + (cfg!(feature = "src-accelerate") as usize)
        + (cfg!(feature = "src-r") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-dynamic-parallel") as usize)
        + (cfg!(feature = "src-intel-mkl-static-sequential") as usize)
        + (cfg!(feature = "src-intel-mkl-static-parallel") as usize);

    if cfg!(feature = "provider-src") {
        assert!(
            src_count == 1,
            "provider-src requires exactly one src-* feature"
        );
    }
    if cfg!(feature = "provider-inject") {
        assert!(src_count == 0, "provider-inject forbids src-* features");
    }
};

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;
#[cfg(feature = "provider-src")]
extern crate lapack_src as _;

#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-inject")]
extern crate lapack_inject as _;

pub mod backend;
#[cfg(all(feature = "linalg-lapack", feature = "provider-inject"))]
pub mod inject;
mod prims_bridge;

use chainrules_core::AdResult;
use num_traits::Zero;
use tenferro_device::{Error, Result};
use tenferro_tensor::{MemoryOrder, Tensor};

mod ad_helpers;
mod frules;
mod primal;
mod result_types;
mod rrules;

pub(crate) use ad_helpers::*;

#[doc(hidden)]
pub use ad_helpers::MatrixExpAbsTensor;
pub use frules::*;
pub(crate) use primal::require_linalg_support;
pub use primal::*;
#[doc(hidden)]
pub use prims_bridge::ScaleTensorByRealSameShape;
pub use result_types::*;
pub use rrules::*;
#[doc(inline)]
pub use tenferro_linalg_prims::{KernelLinalgScalar, LinalgCapabilityOp, LinalgScalar};

#[cfg(test)]
mod tests;
