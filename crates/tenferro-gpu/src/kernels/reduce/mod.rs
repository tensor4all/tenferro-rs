//! Internal reduction kernels.
//!
//! Launch functions reduce one axis and expect keepdims output shape. The
//! backend dispatch layer owns all public tensor semantics.

mod definition;
mod kernels;
mod launch;
mod routines;

#[cfg(feature = "cpu-reference")]
pub(crate) mod cpu_reference;

#[cfg(test)]
mod tests;

pub(crate) use launch::{
    launch_max_float, launch_min_float, launch_prod_complex, launch_prod_float, launch_prod_int,
    launch_sum_complex, launch_sum_float, launch_sum_int, ReduceStrategy,
};
