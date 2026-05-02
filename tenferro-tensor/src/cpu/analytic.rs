use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use strided_kernel::{map_into, zip_map2_into};

use super::{tensor_from_array, typed_array_uninit, typed_view};
use crate::types::{Tensor, TypedTensor};

trait UnaryAnalyticElem: Copy + Clone + One + Zero {
    fn exp_elem(self) -> Self;
    fn log_elem(self) -> Self;
    fn sin_elem(self) -> Self;
    fn cos_elem(self) -> Self;
    fn tanh_elem(self) -> Self;
    fn sqrt_elem(self) -> Self;
    fn rsqrt_elem(self) -> Self;
    fn expm1_elem(self) -> Self;
    fn log1p_elem(self) -> Self;
}

trait PowElem: Copy + Clone + Zero {
    fn pow_elem(self, exponent: Self) -> Self;
}

macro_rules! impl_real_analytic_elem {
    ($ty:ty) => {
        impl UnaryAnalyticElem for $ty {
            fn exp_elem(self) -> Self {
                self.exp()
            }

            fn log_elem(self) -> Self {
                self.ln()
            }

            fn sin_elem(self) -> Self {
                self.sin()
            }

            fn cos_elem(self) -> Self {
                self.cos()
            }

            fn tanh_elem(self) -> Self {
                self.tanh()
            }

            fn sqrt_elem(self) -> Self {
                self.sqrt()
            }

            fn rsqrt_elem(self) -> Self {
                Self::one() / self.sqrt()
            }

            fn expm1_elem(self) -> Self {
                self.exp_m1()
            }

            fn log1p_elem(self) -> Self {
                self.ln_1p()
            }
        }

        impl PowElem for $ty {
            fn pow_elem(self, exponent: Self) -> Self {
                self.powf(exponent)
            }
        }
    };
}

macro_rules! impl_complex_analytic_elem {
    ($ty:ty) => {
        impl UnaryAnalyticElem for $ty {
            fn exp_elem(self) -> Self {
                self.exp()
            }

            fn log_elem(self) -> Self {
                self.ln()
            }

            fn sin_elem(self) -> Self {
                self.sin()
            }

            fn cos_elem(self) -> Self {
                self.cos()
            }

            fn tanh_elem(self) -> Self {
                self.tanh()
            }

            fn sqrt_elem(self) -> Self {
                self.sqrt()
            }

            fn rsqrt_elem(self) -> Self {
                Self::one() / self.sqrt()
            }

            fn expm1_elem(self) -> Self {
                self.exp() - Self::one()
            }

            fn log1p_elem(self) -> Self {
                (self + Self::one()).ln()
            }
        }

        impl PowElem for $ty {
            fn pow_elem(self, exponent: Self) -> Self {
                self.powc(exponent)
            }
        }
    };
}

impl_real_analytic_elem!(f32);
impl_real_analytic_elem!(f64);
impl_complex_analytic_elem!(Complex32);
impl_complex_analytic_elem!(Complex64);

fn backend_failure(op: &'static str, err: impl ToString) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: err.to_string(),
    }
}

macro_rules! define_unary_analytic_op {
    ($dispatch_fn:ident, $typed_fn:ident, $elem_fn:ident) => {
        pub fn $dispatch_fn(input: &Tensor) -> crate::Result<Tensor> {
            match input {
                Tensor::F32(t) => Ok(Tensor::F32($typed_fn(t)?)),
                Tensor::F64(t) => Ok(Tensor::F64($typed_fn(t)?)),
                Tensor::I64(_) => Err(crate::Error::BackendFailure {
                    op: stringify!($dispatch_fn),
                    message: "unsupported dtype I64".into(),
                }),
                Tensor::C32(t) => Ok(Tensor::C32($typed_fn(t)?)),
                Tensor::C64(t) => Ok(Tensor::C64($typed_fn(t)?)),
            }
        }

        fn $typed_fn<T>(input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
        where
            T: UnaryAnalyticElem,
        {
            let mut out = unsafe { typed_array_uninit(&input.shape) };
            map_into(&mut out.view_mut(), &typed_view(input), |x| x.$elem_fn())
                .map_err(|err| backend_failure(stringify!($typed_fn), err))?;
            Ok(tensor_from_array(out))
        }
    };
}

define_unary_analytic_op!(exp, typed_exp, exp_elem);
define_unary_analytic_op!(log, typed_log, log_elem);
define_unary_analytic_op!(sin, typed_sin, sin_elem);
define_unary_analytic_op!(cos, typed_cos, cos_elem);
define_unary_analytic_op!(tanh, typed_tanh, tanh_elem);
define_unary_analytic_op!(sqrt, typed_sqrt, sqrt_elem);
define_unary_analytic_op!(rsqrt, typed_rsqrt, rsqrt_elem);
define_unary_analytic_op!(expm1, typed_expm1, expm1_elem);
define_unary_analytic_op!(log1p, typed_log1p, log1p_elem);

pub fn pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_pow(a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_pow(a, b)?)),
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_pow(a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_pow(a, b)?)),
        _ => Err(crate::Error::DTypeMismatch {
            op: "pow",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

fn typed_pow<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: PowElem,
{
    if lhs.shape != rhs.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "pow",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        });
    }
    let mut out = unsafe { typed_array_uninit(&lhs.shape) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x.pow_elem(y),
    )
    .map_err(|err| backend_failure("pow", err))?;
    Ok(tensor_from_array(out))
}
