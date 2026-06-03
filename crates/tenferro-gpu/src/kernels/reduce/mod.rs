//! Reduction kernels.
//!
//! The public launch functions reduce one axis and expect keepdims output
//! shape. Higher-level tensor crates can call them repeatedly for multi-axis
//! reductions and then reshape metadata to their public output convention.
//!
//! # Examples
//!
//! ```
//! use tenferro_gpu::kernels::reduce::{keepdims_output_shape, ReduceOp};
//!
//! assert_eq!(keepdims_output_shape(&[2, 3, 4], 1).unwrap(), vec![2, 1, 4]);
//! let _op = ReduceOp::Prod;
//! ```

mod definition;
mod kernels;
mod launch;
mod routines;

#[cfg(feature = "cpu-reference")]
pub mod cpu_reference;

#[cfg(test)]
mod tests;

pub use definition::{
    axis_reduce_len, keepdims_output_shape, reduced_output_len, supports_dtype, validate_axis,
    validate_keepdims_output_shape, ReduceDType, ReduceOp,
};
pub use launch::{
    launch_max_float, launch_min_float, launch_prod_complex, launch_prod_float, launch_prod_int,
    launch_sum_complex, launch_sum_float, launch_sum_int, ReduceStrategy,
};
