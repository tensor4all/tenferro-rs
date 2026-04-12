//! Singular-matrix validation helpers shared across backends and exec layers.
//!
//! # Examples
//!
//! ```ignore
//! use tenferro_tensor::validate::validate_nonsingular_u;
//! use tenferro_tensor::{Tensor, TypedTensor};
//!
//! let t = Tensor::F64(TypedTensor::from_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]));
//! assert!(validate_nonsingular_u(&t).is_ok());
//! ```

use num_complex::{Complex32, Complex64};

use crate::{Error, Result, Tensor, TypedTensor};

pub trait DiagSingularity {
    fn is_singular_or_nonfinite(&self) -> bool;
}

macro_rules! impl_diag_singularity_float {
    ($($t:ty),* $(,)?) => {
        $(
            impl DiagSingularity for $t {
                fn is_singular_or_nonfinite(&self) -> bool {
                    !self.is_finite() || *self == 0.0
                }
            }
        )*
    };
}

impl_diag_singularity_float!(f64, f32);

macro_rules! impl_diag_singularity_complex {
    ($($t:ty),* $(,)?) => {
        $(
            impl DiagSingularity for $t {
                fn is_singular_or_nonfinite(&self) -> bool {
                    !self.re.is_finite() || !self.im.is_finite() || self.norm_sqr() == 0.0
                }
            }
        )*
    };
}

impl_diag_singularity_complex!(Complex64, Complex32);

pub fn check_singular_diagonal<T: DiagSingularity + Copy>(t: &TypedTensor<T>) -> Result<()> {
    let n = t.shape[0].min(t.shape[1]);
    let batch_total: usize = t.shape[2..].iter().product();
    let batch_total = batch_total.max(1);
    let slice_size = t.shape[0] * t.shape[1];
    for batch_idx in 0..batch_total {
        let batch = &t.host_data()[batch_idx * slice_size..(batch_idx + 1) * slice_size];
        for i in 0..n {
            let diag = batch[i + i * t.shape[0]];
            if diag.is_singular_or_nonfinite() {
                return Err(Error::BackendFailure {
                    op: "solve",
                    message: "singular matrix".into(),
                });
            }
        }
    }
    Ok(())
}

pub fn validate_nonsingular_u(u: &Tensor) -> Result<()> {
    match u {
        Tensor::F64(t) => check_singular_diagonal(t),
        Tensor::F32(t) => check_singular_diagonal(t),
        Tensor::C64(t) => check_singular_diagonal(t),
        Tensor::C32(t) => check_singular_diagonal(t),
    }
}

#[cfg(test)]
mod tests;
