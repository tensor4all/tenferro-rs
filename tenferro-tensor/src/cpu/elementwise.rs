use std::ops::{Add, Div, Mul, Neg};

use num_complex::Complex;
use num_traits::{One, Zero};
use strided_kernel::{map_into, zip_map2_into, zip_map3_into};

use crate::{
    buffer_pool::{BufferPool, PoolScalar},
    config::CompareDir,
    types::{ConjElem, Tensor, TypedTensor},
};

use super::{tensor_from_array, typed_array_uninit_from_pool, typed_host_data, typed_view};

macro_rules! dispatch_ternary_result_with_pool {
    ($op:literal, $a:expr, $b:expr, $c:expr, |$x:ident, $y:ident, $z:ident| $body:expr) => {
        match ($a, $b, $c) {
            (Tensor::F32($x), Tensor::F32($y), Tensor::F32($z)) => Ok(Tensor::F32($body?)),
            (Tensor::F64($x), Tensor::F64($y), Tensor::F64($z)) => Ok(Tensor::F64($body?)),
            (Tensor::C32($x), Tensor::C32($y), Tensor::C32($z)) => Ok(Tensor::C32($body?)),
            (Tensor::C64($x), Tensor::C64($y), Tensor::C64($z)) => Ok(Tensor::C64($body?)),
            _ => Err(crate::Error::backend_failure($op, "dtype mismatch")),
        }
    };
}

pub(crate) trait Tier2Elem: Copy + Clone + One + Zero {
    fn abs_elem(self) -> Self;
    fn sign_elem(self) -> Self;
    fn max_elem(self, other: Self) -> Self;
    fn min_elem(self, other: Self) -> Self;
}

pub(crate) trait CompareElem: Copy {
    fn compare_elem(self, other: Self, dir: &CompareDir) -> bool;
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
        }

        impl CompareElem for $ty {
            fn compare_elem(self, other: Self, dir: &CompareDir) -> bool {
                match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self < other,
                    CompareDir::Le => self <= other,
                    CompareDir::Gt => self > other,
                    CompareDir::Ge => self >= other,
                }
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
        }

        impl CompareElem for Complex<$real> {
            fn compare_elem(self, other: Self, dir: &CompareDir) -> bool {
                match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self.norm_sqr() < other.norm_sqr(),
                    CompareDir::Le => self.norm_sqr() <= other.norm_sqr(),
                    CompareDir::Gt => self.norm_sqr() > other.norm_sqr(),
                    CompareDir::Ge => self.norm_sqr() >= other.norm_sqr(),
                }
            }
        }
    };
}

impl_tier2_elem_real!(f32);
impl_tier2_elem_real!(f64);
impl_tier2_elem_complex!(f32);
impl_tier2_elem_complex!(f64);

macro_rules! impl_compare_elem_ord {
    ($ty:ty) => {
        impl CompareElem for $ty {
            fn compare_elem(self, other: Self, dir: &CompareDir) -> bool {
                match dir {
                    CompareDir::Eq => self == other,
                    CompareDir::Lt => self < other,
                    CompareDir::Le => self <= other,
                    CompareDir::Gt => self > other,
                    CompareDir::Ge => self >= other,
                }
            }
        }
    };
}

impl_compare_elem_ord!(i32);
impl_compare_elem_ord!(i64);
impl_compare_elem_ord!(bool);

fn complex_scalar_tensor<T>(scalar: T) -> TypedTensor<Complex<T>>
where
    T: Copy + Clone + Zero,
{
    TypedTensor::from_vec_col_major(vec![], vec![Complex::new(scalar, T::zero())])
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
        (Tensor::I32(a), Tensor::I32(b)) => Ok(Tensor::I32(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::I64(a), Tensor::I64(b)) => Ok(Tensor::I64(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_add_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", a)?[0]);
            Ok(Tensor::C32(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", b)?[0]);
            Ok(Tensor::C32(typed_add_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", a)?[0]);
            Ok(Tensor::C64(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", b)?[0]);
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
        (Tensor::I32(a), Tensor::I32(b)) => Ok(Tensor::I32(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::I64(a), Tensor::I64(b)) => Ok(Tensor::I64(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::C32(a), Tensor::C32(b)) => Ok(Tensor::C32(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::C64(a), Tensor::C64(b)) => Ok(Tensor::C64(typed_mul_with_pool(buffers, a, b)?)),
        (Tensor::F32(a), Tensor::C32(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", a)?[0]);
            Ok(Tensor::C32(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", b)?[0]);
            Ok(Tensor::C32(typed_mul_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", a)?[0]);
            Ok(Tensor::C64(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", b)?[0]);
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
            let scalar = complex_scalar_tensor(typed_host_data("div", a)?[0]);
            Ok(Tensor::C32(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", b)?[0]);
            Ok(Tensor::C32(typed_div_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", a)?[0]);
            Ok(Tensor::C64(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape.is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", b)?[0]);
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
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(crate::Error::backend_failure(
            "neg",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
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
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(crate::Error::backend_failure(
            "conj",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
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
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(crate::Error::backend_failure(
            "abs",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
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
        Tensor::I32(_) | Tensor::I64(_) | Tensor::Bool(_) => Err(crate::Error::backend_failure(
            "sign",
            format!("unsupported dtype {:?}", input.dtype()),
        )),
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
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::F64(a), Tensor::F64(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::I32(a), Tensor::I32(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::I64(a), Tensor::I64(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::Bool(a), Tensor::Bool(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::C32(a), Tensor::C32(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
        }
        (Tensor::C64(a), Tensor::C64(b)) => {
            Ok(Tensor::Bool(typed_compare_with_pool(buffers, a, b, dir)?))
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
    match (pred, on_true, on_false) {
        (Tensor::Bool(p), Tensor::F32(t), Tensor::F32(f)) => {
            Ok(Tensor::F32(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::F64(t), Tensor::F64(f)) => {
            Ok(Tensor::F64(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::I32(t), Tensor::I32(f)) => {
            Ok(Tensor::I32(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::I64(t), Tensor::I64(f)) => {
            Ok(Tensor::I64(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::Bool(t), Tensor::Bool(f)) => {
            Ok(Tensor::Bool(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::C32(t), Tensor::C32(f)) => {
            Ok(Tensor::C32(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(p), Tensor::C64(t), Tensor::C64(f)) => {
            Ok(Tensor::C64(typed_select_with_pool(buffers, p, t, f)?))
        }
        (Tensor::Bool(_), _, _) => Err(crate::Error::DTypeMismatch {
            op: "select",
            lhs: on_true.dtype(),
            rhs: on_false.dtype(),
        }),
        _ => Err(crate::Error::DTypeMismatch {
            op: "select",
            lhs: pred.dtype(),
            rhs: crate::DType::Bool,
        }),
    }
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
            &typed_view("add", lhs)?,
            &typed_view("add", rhs)?,
            |x, y| x + y,
        )
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape.is_empty() {
        let scalar = typed_host_data("add", lhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &rhs.shape) };
        map_into(&mut out.view_mut(), &typed_view("add", rhs)?, |x| {
            scalar + x
        })
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape.is_empty() {
        let scalar = typed_host_data("add", rhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        map_into(&mut out.view_mut(), &typed_view("add", lhs)?, |x| {
            x + scalar
        })
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
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
            &typed_view("mul", lhs)?,
            &typed_view("mul", rhs)?,
            |x, y| x * y,
        )
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape.is_empty() {
        let scalar = typed_host_data("mul", lhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &rhs.shape) };
        map_into(&mut out.view_mut(), &typed_view("mul", rhs)?, |x| {
            scalar * x
        })
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape.is_empty() {
        let scalar = typed_host_data("mul", rhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        map_into(&mut out.view_mut(), &typed_view("mul", lhs)?, |x| {
            x * scalar
        })
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
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
            &typed_view("div", lhs)?,
            &typed_view("div", rhs)?,
            |x, y| x / y,
        )
        .map_err(|err| crate::Error::backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape.is_empty() {
        let scalar = typed_host_data("div", lhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &rhs.shape) };
        map_into(&mut out.view_mut(), &typed_view("div", rhs)?, |x| {
            scalar / x
        })
        .map_err(|err| crate::Error::backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape.is_empty() {
        let scalar = typed_host_data("div", rhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, &lhs.shape) };
        map_into(&mut out.view_mut(), &typed_view("div", lhs)?, |x| {
            x / scalar
        })
        .map_err(|err| crate::Error::backend_failure("div", err))?;
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
    map_into(&mut out.view_mut(), &typed_view("neg", input)?, |x| -x)
        .map_err(|err| crate::Error::backend_failure("neg", err))?;
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
    map_into(&mut out.view_mut(), &typed_view("conj", input)?, |x| {
        x.conj_elem()
    })
    .map_err(|err| crate::Error::backend_failure("conj", err))?;
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
    map_into(&mut out.view_mut(), &typed_view("abs", input)?, |x| {
        x.abs_elem()
    })
    .map_err(|err| crate::Error::backend_failure("abs", err))?;
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
    map_into(&mut out.view_mut(), &typed_view("sign", input)?, |x| {
        x.sign_elem()
    })
    .map_err(|err| crate::Error::backend_failure("sign", err))?;
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
        &typed_view("maximum", lhs)?,
        &typed_view("maximum", rhs)?,
        |x, y| x.max_elem(y),
    )
    .map_err(|err| crate::Error::backend_failure("maximum", err))?;
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
        &typed_view("minimum", lhs)?,
        &typed_view("minimum", rhs)?,
        |x, y| x.min_elem(y),
    )
    .map_err(|err| crate::Error::backend_failure("minimum", err))?;
    Ok(tensor_from_array(out))
}

#[allow(dead_code)]
pub(crate) fn typed_compare<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<bool>>
where
    T: CompareElem,
{
    with_local_pool(|buffers| typed_compare_with_pool(buffers, lhs, rhs, dir))
}

pub(crate) fn typed_compare_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    dir: &CompareDir,
) -> crate::Result<TypedTensor<bool>>
where
    T: CompareElem,
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
        &typed_view("compare", lhs)?,
        &typed_view("compare", rhs)?,
        |x, y| x.compare_elem(y, dir),
    )
    .map_err(|err| crate::Error::backend_failure("compare", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_select_with_pool<T>(
    buffers: &mut BufferPool,
    pred: &TypedTensor<bool>,
    on_true: &TypedTensor<T>,
    on_false: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar,
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
        &typed_view("select", pred)?,
        &typed_view("select", on_true)?,
        &typed_view("select", on_false)?,
        |p, t, f| if p { t } else { f },
    )
    .map_err(|err| crate::Error::backend_failure("select", err))?;
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
        &typed_view("clamp", input)?,
        &typed_view("clamp", lower)?,
        &typed_view("clamp", upper)?,
        |x, lo, hi| lo.max_elem(hi.min_elem(x)),
    )
    .map_err(|err| crate::Error::backend_failure("clamp", err))?;
    Ok(tensor_from_array(out))
}
