use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use strided_kernel::{map_into, zip_map2_into};

use super::{tensor_from_array, typed_array_uninit_from_pool, typed_view, typed_view_from_view};
use crate::buffer_pool::{BufferPool, PoolScalar};
use tenferro_tensor::{Tensor, TensorRank, TensorRead, TensorView, TypedTensor, TypedTensorView};

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

#[cfg(test)]
fn with_local_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

enum AnalyticReadView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32,
    I64,
    Bool,
    C32(TypedTensorView<'a, Complex32>),
    C64(TypedTensorView<'a, Complex64>),
}

fn read_as_analytic_view(input: TensorRead<'_>) -> AnalyticReadView<'_> {
    match input {
        TensorRead::Tensor(Tensor::F32(tensor)) => AnalyticReadView::F32(tensor.as_view()),
        TensorRead::Tensor(Tensor::F64(tensor)) => AnalyticReadView::F64(tensor.as_view()),
        TensorRead::Tensor(Tensor::I32(_)) => AnalyticReadView::I32,
        TensorRead::Tensor(Tensor::I64(_)) => AnalyticReadView::I64,
        TensorRead::Tensor(Tensor::Bool(_)) => AnalyticReadView::Bool,
        TensorRead::Tensor(Tensor::C32(tensor)) => AnalyticReadView::C32(tensor.as_view()),
        TensorRead::Tensor(Tensor::C64(tensor)) => AnalyticReadView::C64(tensor.as_view()),
        TensorRead::View(TensorView::F32(view)) => AnalyticReadView::F32(view),
        TensorRead::View(TensorView::F64(view)) => AnalyticReadView::F64(view),
        TensorRead::View(TensorView::I32(_)) => AnalyticReadView::I32,
        TensorRead::View(TensorView::I64(_)) => AnalyticReadView::I64,
        TensorRead::View(TensorView::Bool(_)) => AnalyticReadView::Bool,
        TensorRead::View(TensorView::C32(view)) => AnalyticReadView::C32(view),
        TensorRead::View(TensorView::C64(view)) => AnalyticReadView::C64(view),
    }
}

fn typed_unary_view_with_pool<T, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, T, R>,
    f: impl Fn(T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
    R: TensorRank,
{
    // SAFETY: the following kernel overwrites every output element before any read.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(&mut out.view_mut(), &typed_view_from_view(op, input)?, f)
        .map_err(|err| crate::Error::backend_failure(op, err))?;
    Ok(tensor_from_array(out))
}

fn typed_pow_view_with_pool<T, L, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: PowElem + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::ShapeMismatch {
            op,
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }
    // SAFETY: the following kernel overwrites every output element before any read.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view_from_view(op, lhs)?,
        &typed_view_from_view(op, rhs)?,
        |x, y| x.pow_elem(y),
    )
    .map_err(|err| crate::Error::backend_failure(op, err))?;
    Ok(tensor_from_array(out))
}

macro_rules! define_unary_analytic_op {
    ($dispatch_fn:ident, $dispatch_with_pool_fn:ident, $dispatch_read_with_pool_fn:ident, $typed_fn:ident, $typed_with_pool_fn:ident, $elem_fn:ident) => {
        #[cfg(test)]
        pub(crate) fn $dispatch_fn(input: &Tensor) -> crate::Result<Tensor> {
            with_local_pool(|buffers| $dispatch_with_pool_fn(buffers, input))
        }

        pub(crate) fn $dispatch_with_pool_fn(
            buffers: &mut BufferPool,
            input: &Tensor,
        ) -> crate::Result<Tensor> {
            match input {
                Tensor::F32(t) => Ok(Tensor::F32($typed_with_pool_fn(buffers, t)?)),
                Tensor::F64(t) => Ok(Tensor::F64($typed_with_pool_fn(buffers, t)?)),
                Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => {
                    Err(crate::Error::backend_failure(
                        stringify!($dispatch_fn),
                        format!("unsupported dtype {:?}", input.dtype()),
                    ))
                }
                Tensor::C32(t) => Ok(Tensor::C32($typed_with_pool_fn(buffers, t)?)),
                Tensor::C64(t) => Ok(Tensor::C64($typed_with_pool_fn(buffers, t)?)),
            }
        }

        pub(crate) fn $dispatch_read_with_pool_fn(
            buffers: &mut BufferPool,
            input: TensorRead<'_>,
        ) -> crate::Result<Tensor> {
            let dtype = input.dtype();
            match read_as_analytic_view(input) {
                AnalyticReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
                    stringify!($dispatch_fn),
                    buffers,
                    &t,
                    |x| x.$elem_fn(),
                )?)),
                AnalyticReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
                    stringify!($dispatch_fn),
                    buffers,
                    &t,
                    |x| x.$elem_fn(),
                )?)),
                AnalyticReadView::C32(t) => Ok(Tensor::C32(typed_unary_view_with_pool(
                    stringify!($dispatch_fn),
                    buffers,
                    &t,
                    |x| x.$elem_fn(),
                )?)),
                AnalyticReadView::C64(t) => Ok(Tensor::C64(typed_unary_view_with_pool(
                    stringify!($dispatch_fn),
                    buffers,
                    &t,
                    |x| x.$elem_fn(),
                )?)),
                _ => Err(crate::Error::backend_failure(
                    stringify!($dispatch_fn),
                    format!("unsupported dtype {dtype:?}"),
                )),
            }
        }

        fn $typed_with_pool_fn<T>(
            buffers: &mut BufferPool,
            input: &TypedTensor<T>,
        ) -> crate::Result<TypedTensor<T>>
        where
            T: UnaryAnalyticElem + PoolScalar,
        {
            // SAFETY: the following kernel overwrites every output element before any read.
            let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
            map_into(
                &mut out.view_mut(),
                &typed_view(stringify!($typed_fn), input)?,
                |x| x.$elem_fn(),
            )
            .map_err(|err| crate::Error::backend_failure(stringify!($typed_fn), err))?;
            Ok(tensor_from_array(out))
        }
    };
}

define_unary_analytic_op!(
    exp,
    exp_with_pool,
    exp_read_with_pool,
    typed_exp,
    typed_exp_with_pool,
    exp_elem
);
define_unary_analytic_op!(
    log,
    log_with_pool,
    log_read_with_pool,
    typed_log,
    typed_log_with_pool,
    log_elem
);
define_unary_analytic_op!(
    sin,
    sin_with_pool,
    sin_read_with_pool,
    typed_sin,
    typed_sin_with_pool,
    sin_elem
);
define_unary_analytic_op!(
    cos,
    cos_with_pool,
    cos_read_with_pool,
    typed_cos,
    typed_cos_with_pool,
    cos_elem
);
define_unary_analytic_op!(
    tanh,
    tanh_with_pool,
    tanh_read_with_pool,
    typed_tanh,
    typed_tanh_with_pool,
    tanh_elem
);
define_unary_analytic_op!(
    sqrt,
    sqrt_with_pool,
    sqrt_read_with_pool,
    typed_sqrt,
    typed_sqrt_with_pool,
    sqrt_elem
);
define_unary_analytic_op!(
    rsqrt,
    rsqrt_with_pool,
    rsqrt_read_with_pool,
    typed_rsqrt,
    typed_rsqrt_with_pool,
    rsqrt_elem
);
define_unary_analytic_op!(
    expm1,
    expm1_with_pool,
    expm1_read_with_pool,
    typed_expm1,
    typed_expm1_with_pool,
    expm1_elem
);
define_unary_analytic_op!(
    log1p,
    log1p_with_pool,
    log1p_read_with_pool,
    typed_log1p,
    typed_log1p_with_pool,
    log1p_elem
);

#[cfg(test)]
pub(crate) fn pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| pow_with_pool(buffers, lhs, rhs))
}

pub(crate) fn pow_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_pow_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_pow_with_pool(buffers, a, b)?)),
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_pow_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_pow_with_pool(buffers, a, b)?)),
        _ => Err(crate::Error::DTypeMismatch {
            op: "pow",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

pub(crate) fn pow_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_analytic_view(lhs), read_as_analytic_view(rhs)) {
        (AnalyticReadView::F32(a), AnalyticReadView::F32(b)) => Ok(Tensor::F32(
            typed_pow_view_with_pool("pow", buffers, &a, &b)?,
        )),
        (AnalyticReadView::F64(a), AnalyticReadView::F64(b)) => Ok(Tensor::F64(
            typed_pow_view_with_pool("pow", buffers, &a, &b)?,
        )),
        (AnalyticReadView::C32(a), AnalyticReadView::C32(b)) => Ok(Tensor::C32(
            typed_pow_view_with_pool("pow", buffers, &a, &b)?,
        )),
        (AnalyticReadView::C64(a), AnalyticReadView::C64(b)) => Ok(Tensor::C64(
            typed_pow_view_with_pool("pow", buffers, &a, &b)?,
        )),
        _ => Err(crate::Error::DTypeMismatch {
            op: "pow",
            lhs: lhs_dtype,
            rhs: rhs_dtype,
        }),
    }
}

fn typed_pow_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: PowElem + PoolScalar,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "pow",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }
    // SAFETY: the following kernel overwrites every output element before any read.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view("pow", lhs)?,
        &typed_view("pow", rhs)?,
        |x, y| x.pow_elem(y),
    )
    .map_err(|err| crate::Error::backend_failure("pow", err))?;
    Ok(tensor_from_array(out))
}
