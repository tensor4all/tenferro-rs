//! Experimental StableHLO lowering and runtime PJRT plugin loading for tenferro.
//!
//! This crate is an optional peer executor over `tenferro-runtime`
//! [`GraphProgram`](tenferro_runtime::GraphProgram) values. It does not
//! implement `TensorBackend` and it does not change native CPU, CUDA, or
//! WebGPU execution.
//!
//! # Examples
//!
//! ```
//! use tenferro_runtime::{GraphCompiler, TracedTensor};
//! use tenferro_xla::lower_to_stablehlo;
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
//! let y = &x + &x;
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let module = lower_to_stablehlo(&program).unwrap();
//! assert!(module.as_str().contains("stablehlo.add"));
//! ```

mod error;
mod executor;
mod layout;
mod lowering;
mod stablehlo;

#[cfg(feature = "pjrt")]
mod pjrt;

pub use error::{Error, Result};
pub use executor::{XlaExecutor, XlaExecutorOptions};
#[cfg(feature = "pjrt")]
pub use pjrt::PjrtPlugin;
pub use stablehlo::{StableHloModule, StableHloModuleFingerprint};

/// Environment variable used for the default PJRT plugin path.
///
/// # Examples
///
/// ```
/// use tenferro_xla::TENFERRO_PJRT_PLUGIN_ENV;
///
/// assert_eq!(TENFERRO_PJRT_PLUGIN_ENV, "TENFERRO_PJRT_PLUGIN");
/// ```
pub const TENFERRO_PJRT_PLUGIN_ENV: &str = "TENFERRO_PJRT_PLUGIN";

/// Environment variable used for a GPU-specific PJRT plugin path.
///
/// # Examples
///
/// ```
/// use tenferro_xla::TENFERRO_PJRT_GPU_PLUGIN_ENV;
///
/// assert_eq!(TENFERRO_PJRT_GPU_PLUGIN_ENV, "TENFERRO_PJRT_GPU_PLUGIN");
/// ```
pub const TENFERRO_PJRT_GPU_PLUGIN_ENV: &str = "TENFERRO_PJRT_GPU_PLUGIN";

/// Lower a static-shaped graph program to StableHLO MLIR text.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{GraphCompiler, TracedTensor};
/// use tenferro_xla::lower_to_stablehlo;
///
/// let x = TracedTensor::from_vec_col_major(vec![1], vec![3.0_f64]);
/// let mut compiler = GraphCompiler::new();
/// let program = compiler.compile(&x.neg()).unwrap();
/// let module = lower_to_stablehlo(&program).unwrap();
/// assert!(module.as_str().contains("stablehlo.negate"));
/// ```
pub fn lower_to_stablehlo(program: &tenferro_runtime::GraphProgram) -> Result<StableHloModule> {
    lowering::lower_to_stablehlo(program)
}
