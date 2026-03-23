//! Tensor primitive execution families for the tenferro workspace.
//!
//! The crate is organized around focused backend contracts instead of a single
//! monolithic descriptor surface:
//!
//! - [`TensorSemiringCore`] for the minimal semiring substrate used by
//!   `tenferro-einsum`
//! - [`TensorSemiringFastPath`] for optional semiring performance paths such as
//!   contraction fast paths
//! - [`TensorScalarPrims`] for standard scalar pointwise and reduction families
//! - [`TensorAnalyticPrims`] for analytic pointwise and reduction families
//! - [`TensorComplexRealPrims`] for cross-dtype complex-to-real unary families
//! - [`TensorComplexScalePrims`] for complex payload scaled by real-valued tensors
//! - [`TensorMetadataPrims`] for integer/bool metadata tensor families
//!
//! Every family follows the same plan/execute pattern:
//!
//! 1. Create a family descriptor
//! 2. Build a backend plan for concrete tensor shapes
//! 3. Execute the plan with BLAS-style `alpha`/`beta` scaling
//!
//! # CPU GEMM backend selection
//!
//! `BatchedGemm` on [`CpuBackend`] requires exactly one CPU GEMM backend feature:
//! - `gemm-faer` (default): pure-Rust faer matmul backend
//! - `gemm-blas`: CBLAS backend (`cblas-sys`) with selectable symbol provider
//!
//! If `gemm-blas` is selected, choose exactly one provider:
//! - `provider-src`: link BLAS source crates (`blas-src` + `cblas-src`)
//! - `provider-inject`: link runtime-injected symbols (`cblas-inject`)
//!
//! With `provider-src`, choose exactly one `src-*` implementation:
//! `src-openblas`, `src-netlib`, `src-accelerate`, `src-r`,
//! `src-intel-mkl-dynamic-sequential`, `src-intel-mkl-dynamic-parallel`,
//! `src-intel-mkl-static-sequential`, `src-intel-mkl-static-parallel`.
//!
//! Example (OpenBLAS source provider):
//! `cargo test -p tenferro-prims --no-default-features --features "gemm-blas,provider-src,src-openblas"`
//!
//! Example (runtime-injected provider):
//! `cargo test -p tenferro-prims --no-default-features --features "gemm-blas,provider-inject"`
//!
//! On [`CpuBackend`], semiring-core `BatchedGemm` supports `f32`, `f64`,
//! `Complex32`, and `Complex64`.
//!
//! # Examples
//!
//! ## Semiring core planning
//!
//! ```ignore
//! use tenferro_algebra::Standard;
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{CpuBackend, CpuContext, SemiringCoreDescriptor, TensorSemiringCore};
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(4);
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let b = Tensor::<f64>::zeros(&[4, 5], mem, col);
//! let mut c = Tensor::<f64>::zeros(&[3, 5], mem, col);
//!
//! let desc = SemiringCoreDescriptor::BatchedGemm {
//!     batch_dims: vec![],
//!     m: 3,
//!     n: 5,
//!     k: 4,
//! };
//! let plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(
//!     &mut ctx,
//!     &desc,
//!     &[&[3, 4], &[4, 5], &[3, 5]],
//! )
//! .unwrap();
//! <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
//!     &mut ctx,
//!     &plan,
//!     1.0,
//!     &[&a, &b],
//!     0.0,
//!     &mut c,
//! )
//! .unwrap();
//! ```
//!
//! ## Scalar family planning
//!
//! ```ignore
//! use tenferro_algebra::Standard;
//! use tenferro_device::LogicalMemorySpace;
//! use tenferro_prims::{
//!     CpuBackend, CpuContext, ScalarPrimsDescriptor, ScalarReductionOp, TensorScalarPrims,
//! };
//! use tenferro_tensor::{MemoryOrder, Tensor};
//!
//! let mut ctx = CpuContext::new(4);
//! let col = MemoryOrder::ColumnMajor;
//! let mem = LogicalMemorySpace::MainMemory;
//! let a = Tensor::<f64>::zeros(&[3, 4], mem, col);
//! let mut c = Tensor::<f64>::zeros(&[3], mem, col);
//!
//! let desc = ScalarPrimsDescriptor::Reduction {
//!     modes_a: vec![0, 1],
//!     modes_c: vec![0],
//!     op: ScalarReductionOp::Sum,
//! };
//! let plan = <CpuBackend as TensorScalarPrims<Standard<f64>>>::plan(
//!     &mut ctx,
//!     &desc,
//!     &[&[3, 4], &[3]],
//! )
//! .unwrap();
//! ```

#[cfg(all(feature = "gemm-faer", feature = "gemm-blas"))]
compile_error!("enable exactly one GEMM backend: gemm-faer or gemm-blas");

#[cfg(all(not(feature = "gemm-faer"), not(feature = "gemm-blas")))]
compile_error!("enable exactly one GEMM backend: gemm-faer or gemm-blas");

#[cfg(all(feature = "provider-src", not(feature = "gemm-blas")))]
compile_error!("provider-src requires gemm-blas");
#[cfg(all(feature = "provider-inject", not(feature = "gemm-blas")))]
compile_error!("provider-inject requires gemm-blas");
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
    not(feature = "gemm-blas")
))]
compile_error!("src-* features require gemm-blas and provider-src");

#[cfg(feature = "gemm-blas")]
const _: () = {
    let provider_count =
        (cfg!(feature = "provider-src") as usize) + (cfg!(feature = "provider-inject") as usize);
    assert!(
        provider_count == 1,
        "gemm-blas requires exactly one provider: provider-src or provider-inject"
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
#[cfg(feature = "provider-inject")]
extern crate cblas_inject as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;

mod cpu;
mod families;
mod infra;
#[cfg(all(feature = "gemm-blas", feature = "provider-inject"))]
pub mod inject;

// CUDA backend: real implementation when `cuda` feature is enabled,
// otherwise stub types that return errors.
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
mod cuda_ffi;

mod gpu_stubs;

#[doc(hidden)]
pub use cpu::CpuAnalyticPlan;
#[doc(hidden)]
pub use cpu::CpuComplexRealPlan;
#[doc(hidden)]
pub use cpu::CpuComplexScalePlan;
#[doc(hidden)]
pub use cpu::CpuScalarPlan;
pub use cpu::*;
pub use families::*;
pub use infra::*;

#[cfg(feature = "cuda")]
pub use cuda::*;
#[cfg(feature = "cuda")]
pub use cuda_ffi::*;

#[cfg(not(feature = "cuda"))]
pub use gpu_stubs::CudaBackend;
#[cfg(not(feature = "cuda"))]
pub use gpu_stubs::CudaContext;
#[cfg(not(feature = "cuda"))]
pub use gpu_stubs::CudaPlan;

// ROCm stubs are always from gpu_stubs (no real ROCm backend yet)
pub use gpu_stubs::RocmBackend;
pub use gpu_stubs::RocmContext;
pub use gpu_stubs::RocmPlan;

use tenferro_algebra::Scalar;
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

// ===========================================================================
// Helpers for multi-index iteration
// ===========================================================================

/// Iterate over all index combinations for the given dimensions (column-major order).
pub(crate) fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    let ndim = dims.len();
    if ndim == 0 {
        f(&[]);
        return;
    }
    let total: usize = dims.iter().product();
    if total == 0 {
        return;
    }
    let mut index = vec![0usize; ndim];
    for _ in 0..total {
        f(&index);
        // Increment column-major
        for d in 0..ndim {
            index[d] += 1;
            if index[d] < dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
}

/// Find the position of a mode label in a mode list, returning an error if not found.
pub(crate) fn mode_position(modes: &[u32], label: u32) -> Result<usize> {
    modes
        .iter()
        .position(|&m| m == label)
        .ok_or_else(|| Error::InvalidArgument(format!("mode label {label} not found")))
}

/// Validate that the number of shapes matches expectations for an operation.
pub(crate) fn validate_shape_count(
    shapes: &[&[usize]],
    expected: usize,
    op_name: &str,
) -> Result<()> {
    if shapes.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects {expected} shapes (got {})",
            shapes.len()
        )));
    }
    Ok(())
}

/// Validate that a shape has the expected rank.
pub(crate) fn validate_rank(shape: &[usize], expected: usize, _operand_name: &str) -> Result<()> {
    if shape.len() != expected {
        return Err(Error::RankMismatch {
            expected,
            got: shape.len(),
        });
    }
    Ok(())
}

/// Validate that a shape exactly matches the expected shape.
pub(crate) fn validate_shape_eq(
    got: &[usize],
    expected: &[usize],
    _operand_name: &str,
) -> Result<()> {
    if got != expected {
        return Err(Error::ShapeMismatch {
            expected: expected.to_vec(),
            got: got.to_vec(),
        });
    }
    Ok(())
}

/// Validate the number of input operands for execute.
pub(crate) fn validate_execute_inputs<T: Scalar>(
    inputs: &[&Tensor<T>],
    expected: usize,
    op_name: &str,
) -> Result<()> {
    if inputs.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{op_name} expects {expected} input(s) (got {})",
            inputs.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
