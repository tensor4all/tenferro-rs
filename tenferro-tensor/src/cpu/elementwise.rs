use std::ops::{Add, Div, Mul, Neg};

use num_complex::Complex;
use num_traits::{One, Zero};
use strided_kernel::{map_into, zip_map2_into, zip_map3_into};

use crate::{
    buffer_pool::{BufferPool, PoolScalar},
    config::CompareDir,
    types::{ConjElem, Tensor, TypedTensor},
};

use super::{tensor_from_array, typed_array_uninit_from_pool, typed_view};

macro_rules! dispatch_ternary_result_with_pool {
    ($op:literal, $a:expr, $b:expr, $c:expr, |$x:ident, $y:ident, $z:ident| $body:expr) => {
        match ($a, $b, $c) {
            (Tensor::F32($x), Tensor::F32($y), Tensor::F32($z)) => Ok(Tensor::F32($body?)),
            (Tensor::F64($x), Tensor::F64($y), Tensor::F64($z)) => Ok(Tensor::F64($body?)),
            (Tensor::C32($x), Tensor::C32($y), Tensor::C32($z)) => Ok(Tensor::C32($body?)),
            (Tensor::C64($x), Tensor::C64($y), Tensor::C64($z)) => Ok(Tensor::C64($body?)),
            _ => Err(crate::Error::BackendFailure {
                op: $op,
                message: "dtype mismatch".into(),
            }),
        }
    };
}

pub(crate) trait Tier2Elem: Copy + Clone + One + Zero {
    fn abs_elem(self) -> Self;
    fn sign_elem(self) -> Self;
    fn max_elem(self, other: Self) -> Self;
    fn min_elem(self, other: Self) -> Self;
    fn compare_elem(self, other: Self, dir: &CompareDir) -> Self;
    fn is_nonzero(self) -> bool;
}

macro_rules! impl_tier2_elem_real {
    ($ty:ty) => {
        impl Tier2Elem for $ty {
            fn abs_elem(self) -> Self {
                self.abs()
            }

            fn sign_elem(self) -> Self {
                if self == Self::zero() {
                    Self::zero()
                } else {
                    self.signum()
                }
            }

            fn max_elem(self, other: Self) -> Self {
                if self >= other {
                    self
                } else {
                    other
                }
            }

            fn min_elem(self, other: Self) -> Self {
                if self <= other {
                    self
                } else {
                    other
                }
            }

            fn compare_elem(self, other: Self, dir: &CompareDir) -> Self {
                let pred = match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self < other,
                    CompareDir::Le => self <= other,
                    CompareDir::Gt => self > other,
                    CompareDir::Ge => self >= other,
                };
                if pred {
                    Self::one()
                } else {
                    Self::zero()
                }
            }

            fn is_nonzero(self) -> bool {
                self != Self::zero()
            }
        }
    };
}

macro_rules! impl_tier2_elem_complex {
    ($real:ty) => {
        impl Tier2Elem for Complex<$real> {
            fn abs_elem(self) -> Self {
                Self::new(self.norm(), <$real>::zero())
            }

            fn sign_elem(self) -> Self {
                if self.is_zero() {
                    Self::zero()
                } else {
                    self / self.abs_elem()
                }
            }

            fn max_elem(self, other: Self) -> Self {
                if self.norm_sqr() >= other.norm_sqr() {
                    self
                } else {
                    other
                }
            }

            fn min_elem(self, other: Self) -> Self {
                if self.norm_sqr() <= other.norm_sqr() {
                    self
                } else {
                    other
                }
            }

            fn compare_elem(self, other: Self, dir: &CompareDir) -> Self {
                let pred = match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self.norm_sqr() < other.norm_sqr(),
                    CompareDir::Le => self.norm_sqr() <= other.norm_sqr(),
                    CompareDir::Gt => self.norm_sqr() > other.norm_sqr(),
                    CompareDir::Ge => self.norm_sqr() >= other.norm_sqr(),
                };
                if pred {
                    Self::one()
                } else {
                    Self::zero()
                }
            }

            fn is_nonzero(self) -> bool {
                !self.is_zero()
            }
        }
    };
}

impl_tier2_elem_real!(f32);
impl_tier2_elem_real!(f64);
impl_tier2_elem_complex!(f32);
impl_tier2_elem_complex!(f64);

fn complex_scalar_tensor<T>(scalar: T) -> TypedTensor<Complex<T>>
where
    T: Copy + Clone + Zero,
{
    TypedTensor::from_vec(vec![], vec![Complex::new(scalar, T::zero())])
}

fn backend_failure(op: &'static str, err: impl ToString) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: err.to_string(),
    }
}

fn with_local_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| add_with_pool(buffers, lhs, rhs))
}

pub(crate) fn add_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::I64(a), Tensor::I64(b)) => Ok(Tensor::I64(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(a.host_data()[0]);
            Ok(Tensor::C32(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(b.host_data()[0]);
            Ok(Tensor::C32(typed_add_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(a.host_data()[0]);
            Ok(Tensor::C64(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(b.host_data()[0]);
            Ok(Tensor::C64(typed_add_with_pool(buffers, a, &scalar)?))
        }
        _ => Err(crate::Error::DTypeMismatch {
            op: "add",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| mul_with_pool(buffers, lhs, rhs))
}

pub(crate) fn mul_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::I64(a), Tensor::I64(b)) => Ok(Tensor::I64(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(a.host_data()[0]);
            Ok(Tensor::C32(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(b.host_data()[0]);
            Ok(Tensor::C32(typed_mul_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(a.host_data()[0]);
            Ok(Tensor::C64(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(b.host_data()[0]);
            Ok(Tensor::C64(typed_mul_with_pool(buffers, a, &scalar)?))
        }
        _ => Err(crate::Error::DTypeMismatch {
            op: "mul",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

pub fn div(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| div_with_pool(buffers, lhs, rhs))
}

pub(crate) fn div_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => Ok(Tensor::F32(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::F64(a), Tensor::F64(b)) => Ok(Tensor::F64(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_div_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(a.host_data()[0]);
            Ok(Tensor::C32(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(b.host_data()[0]);
            Ok(Tensor::C32(typed_div_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(a.host_data()[0]);
            Ok(Tensor::C64(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(b.host_data()[0]);
            Ok(Tensor::C64(typed_div_with_pool(buffers, a, &scalar)?))
        }
        _ => Err(crate::Error::DTypeMismatch {
            op: "div",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

pub fn neg(input: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| neg_with_pool(buffers, input))
}

pub(crate) fn neg_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_neg_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_neg_with_pool(buffers, t)?)),
        Tensor::I64(_) => Err(crate::Error::BackendFailure {
            op: "neg",
            message: "unsupported dtype I64".into(),
        }),
        Tensor::C32(t) => Ok(Tensor::C32(typed_neg_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_neg_with_pool(buffers, t)?)),
    }
}

pub fn conj(input: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| conj_with_pool(buffers, input))
}

pub(crate) fn conj_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_conj_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_conj_with_pool(buffers, t)?)),
        Tensor::I64(_) => Err(crate::Error::BackendFailure {
            op: "conj",
            message: "unsupported dtype I64".into(),
        }),
        Tensor::C32(t) => Ok(Tensor::C32(typed_conj_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_conj_with_pool(buffers, t)?)),
    }
}

pub fn abs(input: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| abs_with_pool(buffers, input))
}

pub(crate) fn abs_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_abs_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_abs_with_pool(buffers, t)?)),
        Tensor::I64(_) => Err(crate::Error::BackendFailure {
            op: "abs",
            message: "unsupported dtype I64".into(),
        }),
        Tensor::C32(t) => Ok(Tensor::C32(typed_abs_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_abs_with_pool(buffers, t)?)),
    }
}

pub fn sign(input: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| sign_with_pool(buffers, input))
}

pub(crate) fn sign_with_pool(buffers: &mut BufferPool, input: &Tensor) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(t) => Ok(Tensor::F32(typed_sign_with_pool(buffers, t)?)),
        Tensor::F64(t) => Ok(Tensor::F64(typed_sign_with_pool(buffers, t)?)),
        Tensor::I64(_) => Err(crate::Error::BackendFailure {
            op: "sign",
            message: "unsupported dtype I64".into(),
        }),
        Tensor::C32(t) => Ok(Tensor::C32(typed_sign_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::C64(typed_sign_with_pool(buffers, t)?)),
    }
}

pub fn maximum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| maximum_with_pool(buffers, lhs, rhs))
}

pub(crate) fn maximum_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => {
            Ok(Tensor::F32(typed_maximum_with_pool(buffers, a, b)?))
        }
        (Tensor::F64(a), Tensor::F64(b)) => {
            Ok(Tensor::F64(typed_maximum_with_pool(buffers, a, b)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => {
            Ok(Tensor::C32(typed_maximum_with_pool(buffers, a, b)?))
        }
        (Tensor::C64(a), Tensor::C64(b)) => {
            Ok(Tensor::C64(typed_maximum_with_pool(buffers, a, b)?))
        }
        _ => Err(crate::Error::DTypeMismatch {
            op: "maximum",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

pub fn minimum(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| minimum_with_pool(buffers, lhs, rhs))
}

pub(crate) fn minimum_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => {
            Ok(Tensor::F32(typed_minimum_with_pool(buffers, a, b)?))
        }
        (Tensor::F64(a), Tensor::F64(b)) => {
            Ok(Tensor::F64(typed_minimum_with_pool(buffers, a, b)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => {
            Ok(Tensor::C32(typed_minimum_with_pool(buffers, a, b)?))
        }
        (Tensor::C64(a), Tensor::C64(b)) => {
            Ok(Tensor::C64(typed_minimum_with_pool(buffers, a, b)?))
        }
        _ => Err(crate::Error::DTypeMismatch {
            op: "minimum",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

pub fn compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
    with_local_pool(|buffers| compare_with_pool(buffers, lhs, rhs, dir))
}

pub(crate) fn compare_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
    dir: &CompareDir,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(a), Tensor::F32(b)) => {
            Ok(Tensor::F32(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::F64(a), Tensor::F64(b)) => {
            Ok(Tensor::F64(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => {
            Ok(Tensor::C32(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::C64(a), Tensor::C64(b)) => {
            Ok(Tensor::C64(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        _ => Err(crate::Error::DTypeMismatch {
            op: "compare",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        }),
    }
}

pub fn select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| select_with_pool(buffers, pred, on_true, on_false))
}

pub(crate) fn select_with_pool(
    buffers: &mut BufferPool,
    pred: &Tensor,
    on_true: &Tensor,
    on_false: &Tensor,
) -> crate::Result<Tensor> {
    dispatch_ternary_result_with_pool!("select", pred, on_true, on_false, |p, t, f| {
        typed_select_with_pool(buffers, p, t, f)
    })
}

pub fn clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| clamp_with_pool(buffers, input, lower, upper))
}

pub(crate) fn clamp_with_pool(
    buffers: &mut BufferPool,
    input: &Tensor,
    lower: &Tensor,
    upper: &Tensor,
) -> crate::Result<Tensor> {
    dispatch_ternary_result_with_pool!("clamp", input, lower, upper, |x, lo, hi| {
        typed_clamp_with_pool(buffers, x, lo, hi)
    })
}

pub fn typed_add<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T> + PoolScalar,
{
    with_local_pool(|buffers| typed_add_with_pool(buffers, lhs, rhs))
}

pub(crate) fn typed_add_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T> + PoolScalar,
{
    if lhs.shape == rhs.shape {
        // SAFETY: zip_map2_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view(lhs),
            &typed_view(rhs),
            |x, y| x + y,
        )
        .map_err(|err| crate::Error::BackendFailure {
            op: "add",
            message: err.to_string(),
        })?;
        Ok(tensor_from_array(out))
    } else if lhs.shape.is_empty() {
        let scalar = lhs.host_data()[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &rhs.shape) };
        map_into(&mut out.view_mut(), &typed_view(rhs), |x| scalar + x).map_err(|err| {
            crate::Error::BackendFailure {
                op: "add",
                message: err.to_string(),
            }
        })?;
        Ok(tensor_from_array(out))
    } else if rhs.shape.is_empty() {
        let scalar = rhs.host_data()[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        map_into(&mut out.view_mut(), &typed_view(lhs), |x| x + scalar).map_err(|err| {
            crate::Error::BackendFailure {
                op: "add",
                message: err.to_string(),
            }
        })?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "add",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        })
    }
}

pub fn typed_mul<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar,
{
    with_local_pool(|buffers| typed_mul_with_pool(buffers, lhs, rhs))
}

pub(crate) fn typed_mul_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar,
{
    if lhs.shape == rhs.shape {
        // SAFETY: zip_map2_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view(lhs),
            &typed_view(rhs),
            |x, y| x * y,
        )
        .map_err(|err| backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape.is_empty() {
        let scalar = lhs.host_data()[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &rhs.shape) };
        map_into(&mut out.view_mut(), &typed_view(rhs), |x| scalar * x)
            .map_err(|err| backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape.is_empty() {
        let scalar = rhs.host_data()[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        map_into(&mut out.view_mut(), &typed_view(lhs), |x| x * scalar)
            .map_err(|err| backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "mul",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        })
    }
}

pub fn typed_div<T>(lhs: &TypedTensor<T>, rhs: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Div<Output = T> + PoolScalar,
{
    with_local_pool(|buffers| typed_div_with_pool(buffers, lhs, rhs))
}

pub(crate) fn typed_div_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Div<Output = T> + PoolScalar,
{
    if lhs.shape == rhs.shape {
        // SAFETY: zip_map2_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view(lhs),
            &typed_view(rhs),
            |x, y| x / y,
        )
        .map_err(|err| backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape.is_empty() {
        let scalar = lhs.host_data()[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &rhs.shape) };
        map_into(&mut out.view_mut(), &typed_view(rhs), |x| scalar / x)
            .map_err(|err| backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape.is_empty() {
        let scalar = rhs.host_data()[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        map_into(&mut out.view_mut(), &typed_view(lhs), |x| x / scalar)
            .map_err(|err| backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "div",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        })
    }
}

pub fn typed_neg<T>(input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Neg<Output = T> + PoolScalar,
{
    with_local_pool(|buffers| typed_neg_with_pool(buffers, input))
}

pub(crate) fn typed_neg_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Neg<Output = T> + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &input.shape) };
    map_into(&mut out.view_mut(), &typed_view(input), |x| -x)
        .map_err(|err| backend_failure("neg", err))?;
    Ok(tensor_from_array(out))
}

pub fn typed_conj<T>(input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + ConjElem + PoolScalar,
{
    with_local_pool(|buffers| typed_conj_with_pool(buffers, input))
}

pub(crate) fn typed_conj_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + ConjElem + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &input.shape) };
    map_into(&mut out.view_mut(), &typed_view(input), |x| x.conj_elem())
        .map_err(|err| backend_failure("conj", err))?;
    Ok(tensor_from_array(out))
}

#[allow(dead_code)]
pub(crate) fn typed_abs<T>(input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_abs_with_pool(buffers, input))
}

pub(crate) fn typed_abs_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &input.shape) };
    map_into(&mut out.view_mut(), &typed_view(input), |x| x.abs_elem())
        .map_err(|err| backend_failure("abs", err))?;
    Ok(tensor_from_array(out))
}

#[allow(dead_code)]
pub(crate) fn typed_sign<T>(input: &TypedTensor<T>) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_sign_with_pool(buffers, input))
}

pub(crate) fn typed_sign_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &input.shape) };
    map_into(&mut out.view_mut(), &typed_view(input), |x| x.sign_elem())
        .map_err(|err| backend_failure("sign", err))?;
    Ok(tensor_from_array(out))
}

#[allow(dead_code)]
pub(crate) fn typed_maximum<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_maximum_with_pool(buffers, lhs, rhs))
}

pub(crate) fn typed_maximum_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    if lhs.shape != rhs.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "maximum",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        });
    }
    // SAFETY: zip_map2_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x.max_elem(y),
    )
    .map_err(|err| backend_failure("maximum", err))?;
    Ok(tensor_from_array(out))
}

#[allow(dead_code)]
pub(crate) fn typed_minimum<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_minimum_with_pool(buffers, lhs, rhs))
}

pub(crate) fn typed_minimum_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    if lhs.shape != rhs.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "minimum",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        });
    }
    // SAFETY: zip_map2_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x.min_elem(y),
    )
    .map_err(|err| backend_failure("minimum", err))?;
    Ok(tensor_from_array(out))
}

#[allow(dead_code)]
pub(crate) fn typed_compare<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    with_local_pool(|buffers| typed_compare_with_pool(buffers, lhs, rhs, dir))
}

pub(crate) fn typed_compare_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    if lhs.shape != rhs.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "compare",
            lhs: lhs.shape.clone(),
            rhs: rhs.shape.clone(),
        });
    }
    // SAFETY: zip_map2_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view(lhs),
        &typed_view(rhs),
        |x, y| x.compare_elem(y, dir),
    )
    .map_err(|err| backend_failure("compare", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_select_with_pool<T>(
    buffers: &mut BufferPool,
    pred: &TypedTensor<T>,
    on_true: &TypedTensor<T>,
    on_false: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    if pred.shape != on_true.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "select",
            lhs: pred.shape.clone(),
            rhs: on_true.shape.clone(),
        });
    }
    if pred.shape != on_false.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "select",
            lhs: pred.shape.clone(),
            rhs: on_false.shape.clone(),
        });
    }
    // SAFETY: zip_map3_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &pred.shape) };
    zip_map3_into(
        &mut out.view_mut(),
        &typed_view(pred),
        &typed_view(on_true),
        &typed_view(on_false),
        |p, t, f| if p.is_nonzero() { t } else { f },
    )
    .map_err(|err| backend_failure("select", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_clamp_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    lower: &TypedTensor<T>,
    upper: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    if input.shape != lower.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "clamp",
            lhs: input.shape.clone(),
            rhs: lower.shape.clone(),
        });
    }
    if input.shape != upper.shape {
        return Err(crate::Error::ShapeMismatch {
            op: "clamp",
            lhs: input.shape.clone(),
            rhs: upper.shape.clone(),
        });
    }
    // SAFETY: zip_map3_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, &input.shape) };
    zip_map3_into(
        &mut out.view_mut(),
        &typed_view(input),
        &typed_view(lower),
        &typed_view(upper),
        |x, lo, hi| lo.max_elem(hi.min_elem(x)),
    )
    .map_err(|err| backend_failure("clamp", err))?;
    Ok(tensor_from_array(out))
}
