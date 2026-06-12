use std::ops::{Add, Div, Mul, Neg};
use std::sync::Arc;

use num_complex::Complex;
use num_traits::{One, Zero};
use strided_kernel::{
    batched_outer_product_into, broadcast_mul_into, map_into, mul_into, zip_map2_into,
    zip_map3_into,
};

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::ConjElem;
use tenferro_tensor::{
    col_major_strides, CompareDir, Tensor, TensorOwnedView, TensorRank, TensorRead, TensorValue,
    TensorView, TypedTensor, TypedTensorView,
};

use super::{
    tensor_from_array, typed_array_uninit_from_pool, typed_host_data, typed_view,
    typed_view_from_view,
};

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

pub(crate) trait Tier2Elem: Copy + Clone + One + Zero + Send + Sync {
    fn abs_elem(self) -> Self;
    fn sign_elem(self) -> Self;
    fn max_elem(self, other: Self) -> Self;
    fn min_elem(self, other: Self) -> Self;
}

pub(crate) trait CompareElem: Copy + Send + Sync {
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

fn complex_scalar_tensor_from_tensor<T>(
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<Complex<T>>>
where
    T: Copy + Clone + Zero,
{
    Ok(complex_scalar_tensor(typed_host_data("add", input)?[0]))
}

fn complex_scalar_tensor_from_view<T, R>(
    input: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<Complex<T>>>
where
    T: Copy + Clone + Zero + 'static,
    R: TensorRank,
{
    Ok(complex_scalar_tensor(
        typed_view_from_view("add", input)?.get(&[]),
    ))
}

fn with_local_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

/// Add two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::add;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let out = add(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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
        (Tensor::F32(a), Tensor::C32(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", a)?[0]);
            Ok(Tensor::C32(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", b)?[0]);
            Ok(Tensor::C32(typed_add_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("add", a)?[0]);
            Ok(Tensor::C64(typed_add_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape().is_empty() => {
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

pub(crate) fn add_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return add_with_pool(buffers, lhs, rhs);
    }

    macro_rules! dispatch {
        ($variant:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    let a = a.as_view();
                    return Ok(Tensor::$variant(typed_add_view_with_pool(buffers, &a, b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                ) => {
                    let b = b.as_view();
                    return Ok(Tensor::$variant(typed_add_view_with_pool(buffers, a, &b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    return Ok(Tensor::$variant(typed_add_view_with_pool(buffers, a, b)?));
                }
                _ => {}
            }
        };
    }

    macro_rules! dispatch_real_complex_scalar {
        ($real_variant:ident, $complex_variant:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    let complex = complex.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &scalar, &complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let complex = complex.as_view();
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, &complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_add_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                _ => {}
            }
        };
    }

    dispatch_real_complex_scalar!(F32, C32);
    dispatch_real_complex_scalar!(F64, C64);

    dispatch!(F32);
    dispatch!(F64);
    dispatch!(I32);
    dispatch!(I64);
    dispatch!(C32);
    dispatch!(C64);

    Err(crate::Error::DTypeMismatch {
        op: "add",
        lhs: lhs.dtype(),
        rhs: rhs.dtype(),
    })
}

/// Multiply two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::mul;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]);
/// let b = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 5.0]);
/// let out = mul(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[8.0, 15.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_local_pool(|buffers| mul_with_pool(buffers, lhs, rhs))
}

fn binary_read_with_pool(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    f: impl FnOnce(&mut BufferPool, &Tensor, &Tensor) -> crate::Result<Tensor>,
) -> crate::Result<Tensor> {
    if let (Some(lhs), Some(rhs)) = (lhs.as_tensor(), rhs.as_tensor()) {
        return f(buffers, lhs, rhs);
    }

    Err(crate::Error::DTypeMismatch {
        op,
        lhs: lhs.dtype(),
        rhs: rhs.dtype(),
    })
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
        (Tensor::F32(a), Tensor::C32(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", a)?[0]);
            Ok(Tensor::C32(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", b)?[0]);
            Ok(Tensor::C32(typed_mul_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("mul", a)?[0]);
            Ok(Tensor::C64(typed_mul_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape().is_empty() => {
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

pub(crate) fn mul_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    if let (TensorRead::Tensor(lhs), TensorRead::Tensor(rhs)) = (&lhs, &rhs) {
        return mul_with_pool(buffers, lhs, rhs);
    }

    macro_rules! dispatch {
        ($variant:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    let a = a.as_view();
                    return Ok(Tensor::$variant(typed_mul_view_with_pool(buffers, &a, b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::Tensor(Tensor::$variant(b)),
                ) => {
                    let b = b.as_view();
                    return Ok(Tensor::$variant(typed_mul_view_with_pool(buffers, a, &b)?));
                }
                (
                    TensorRead::View(TensorView::$variant(a)),
                    TensorRead::View(TensorView::$variant(b)),
                ) => {
                    return Ok(Tensor::$variant(typed_mul_view_with_pool(buffers, a, b)?));
                }
                _ => {}
            }
        };
    }

    macro_rules! dispatch_real_complex_scalar {
        ($real_variant:ident, $complex_variant:ident) => {
            match (&lhs, &rhs) {
                (
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    let complex = complex.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &scalar, &complex,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$real_variant(real)),
                    TensorRead::View(TensorView::$complex_variant(complex)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &scalar, complex,
                    )?));
                }
                (
                    TensorRead::Tensor(Tensor::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let complex = complex.as_view();
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, &complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::Tensor(Tensor::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_tensor(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                (
                    TensorRead::View(TensorView::$complex_variant(complex)),
                    TensorRead::View(TensorView::$real_variant(real)),
                ) if real.shape().is_empty() => {
                    let scalar = complex_scalar_tensor_from_view(real)?;
                    let scalar = scalar.as_view();
                    return Ok(Tensor::$complex_variant(typed_mul_view_with_pool(
                        buffers, complex, &scalar,
                    )?));
                }
                _ => {}
            }
        };
    }

    dispatch_real_complex_scalar!(F32, C32);
    dispatch_real_complex_scalar!(F64, C64);

    dispatch!(F32);
    dispatch!(F64);
    dispatch!(I32);
    dispatch!(I64);
    dispatch!(C32);
    dispatch!(C64);

    binary_read_with_pool("mul", buffers, lhs, rhs, mul_with_pool)
}

enum CpuReadView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool(TypedTensorView<'a, bool>),
    C32(TypedTensorView<'a, Complex<f32>>),
    C64(TypedTensorView<'a, Complex<f64>>),
}

fn read_as_cpu_view(input: TensorRead<'_>) -> CpuReadView<'_> {
    match input {
        TensorRead::Tensor(Tensor::F32(tensor)) => CpuReadView::F32(tensor.as_view()),
        TensorRead::Tensor(Tensor::F64(tensor)) => CpuReadView::F64(tensor.as_view()),
        TensorRead::Tensor(Tensor::I32(tensor)) => CpuReadView::I32(tensor.as_view()),
        TensorRead::Tensor(Tensor::I64(tensor)) => CpuReadView::I64(tensor.as_view()),
        TensorRead::Tensor(Tensor::Bool(tensor)) => CpuReadView::Bool(tensor.as_view()),
        TensorRead::Tensor(Tensor::C32(tensor)) => CpuReadView::C32(tensor.as_view()),
        TensorRead::Tensor(Tensor::C64(tensor)) => CpuReadView::C64(tensor.as_view()),
        TensorRead::View(TensorView::F32(view)) => CpuReadView::F32(view),
        TensorRead::View(TensorView::F64(view)) => CpuReadView::F64(view),
        TensorRead::View(TensorView::I32(view)) => CpuReadView::I32(view),
        TensorRead::View(TensorView::I64(view)) => CpuReadView::I64(view),
        TensorRead::View(TensorView::Bool(view)) => CpuReadView::Bool(view),
        TensorRead::View(TensorView::C32(view)) => CpuReadView::C32(view),
        TensorRead::View(TensorView::C64(view)) => CpuReadView::C64(view),
    }
}

fn typed_binary_view_with_pool<T, L, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
    f: impl Fn(T, T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() == rhs.shape() {
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view_from_view(op, lhs)?,
            &typed_view_from_view(op, rhs)?,
            f,
        )
        .map_err(|err| crate::Error::backend_failure(op, err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape().is_empty() {
        let scalar = typed_view_from_view(op, lhs)?.get(&[]);
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, rhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view_from_view(op, rhs)?, |x| {
            f(scalar, x)
        })
        .map_err(|err| crate::Error::backend_failure(op, err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape().is_empty() {
        let scalar = typed_view_from_view(op, rhs)?.get(&[]);
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view_from_view(op, lhs)?, |x| {
            f(x, scalar)
        })
        .map_err(|err| crate::Error::backend_failure(op, err))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op,
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
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
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(&mut out.view_mut(), &typed_view_from_view(op, input)?, f)
        .map_err(|err| crate::Error::backend_failure(op, err))?;
    Ok(tensor_from_array(out))
}

fn typed_same_shape_binary_view_with_pool<T, O, L, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
    f: impl Fn(T, T) -> O + Copy + Sync,
) -> crate::Result<TypedTensor<O>>
where
    T: Copy + Send + Sync + 'static,
    O: Copy + PoolScalar,
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
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view_from_view(op, lhs)?,
        &typed_view_from_view(op, rhs)?,
        f,
    )
    .map_err(|err| crate::Error::backend_failure(op, err))?;
    Ok(tensor_from_array(out))
}

fn typed_select_view_with_pool<T, P, A, B>(
    buffers: &mut BufferPool,
    pred: &TypedTensorView<'_, bool, P>,
    on_true: &TypedTensorView<'_, T, A>,
    on_false: &TypedTensorView<'_, T, B>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
    P: TensorRank,
    A: TensorRank,
    B: TensorRank,
{
    if pred.shape() != on_true.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "select",
            lhs: pred.shape().to_vec(),
            rhs: on_true.shape().to_vec(),
        });
    }
    if pred.shape() != on_false.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "select",
            lhs: pred.shape().to_vec(),
            rhs: on_false.shape().to_vec(),
        });
    }
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, pred.shape()) };
    zip_map3_into(
        &mut out.view_mut(),
        &typed_view_from_view("select", pred)?,
        &typed_view_from_view("select", on_true)?,
        &typed_view_from_view("select", on_false)?,
        |p, t, f| if p { t } else { f },
    )
    .map_err(|err| crate::Error::backend_failure("select", err))?;
    Ok(tensor_from_array(out))
}

fn typed_clamp_view_with_pool<T, I, L, U>(
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, T, I>,
    lower: &TypedTensorView<'_, T, L>,
    upper: &TypedTensorView<'_, T, U>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar + 'static,
    I: TensorRank,
    L: TensorRank,
    U: TensorRank,
{
    if input.shape() != lower.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "clamp",
            lhs: input.shape().to_vec(),
            rhs: lower.shape().to_vec(),
        });
    }
    if input.shape() != upper.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "clamp",
            lhs: input.shape().to_vec(),
            rhs: upper.shape().to_vec(),
        });
    }
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    zip_map3_into(
        &mut out.view_mut(),
        &typed_view_from_view("clamp", input)?,
        &typed_view_from_view("clamp", lower)?,
        &typed_view_from_view("clamp", upper)?,
        |x, lo, hi| lo.max_elem(hi.min_elem(x)),
    )
    .map_err(|err| crate::Error::backend_failure("clamp", err))?;
    Ok(tensor_from_array(out))
}

#[derive(Clone, Copy)]
enum SplitOuterProductLayout {
    LhsPrefix,
    RhsPrefix,
}

struct SplitOuterProductPlan {
    #[allow(dead_code)]
    rows: usize,
    #[allow(dead_code)]
    cols: usize,
    #[allow(dead_code)]
    batches: usize,
    layout: SplitOuterProductLayout,
    lhs_free_axes: Vec<usize>,
    rhs_free_axes: Vec<usize>,
    lhs_batch_axes: Vec<usize>,
    rhs_batch_axes: Vec<usize>,
}

struct OuterProductAxisPartition {
    lhs_free_output_axes: Vec<usize>,
    rhs_free_output_axes: Vec<usize>,
    batch_output_axes: Vec<usize>,
    lhs_free_axes: Vec<usize>,
    rhs_free_axes: Vec<usize>,
    lhs_batch_axes: Vec<usize>,
    rhs_batch_axes: Vec<usize>,
}

fn shape_matches_dims(source_shape: &[usize], output_shape: &[usize], dims: &[usize]) -> bool {
    source_shape.len() == dims.len()
        && source_shape
            .iter()
            .zip(dims.iter())
            .all(|(&dim, &axis)| output_shape.get(axis).copied() == Some(dim))
}

fn axes_by_output(dims: &[usize], output_rank: usize) -> Option<Vec<Option<usize>>> {
    let mut axes = vec![None; output_rank];
    for (src_axis, &dst_axis) in dims.iter().enumerate() {
        let slot = axes.get_mut(dst_axis)?;
        if slot.replace(src_axis).is_some() {
            return None;
        }
    }
    Some(axes)
}

fn axes_shape_product<T>(
    op: &'static str,
    view: &TypedTensorView<'_, T>,
    axes: &[usize],
) -> crate::Result<usize>
where
    T: 'static,
{
    axes.iter().try_fold(1usize, |acc, &axis| {
        acc.checked_mul(view.shape()[axis])
            .ok_or_else(|| crate::Error::backend_failure(op, "shape size overflows usize"))
    })
}

fn classify_outer_product_axes(
    lhs_dims: &[usize],
    rhs_dims: &[usize],
    output_rank: usize,
) -> Option<OuterProductAxisPartition> {
    let lhs_axes_by_output = axes_by_output(lhs_dims, output_rank)?;
    let rhs_axes_by_output = axes_by_output(rhs_dims, output_rank)?;

    let mut lhs_free_output_axes = Vec::new();
    let mut rhs_free_output_axes = Vec::new();
    let mut batch_output_axes = Vec::new();
    let mut lhs_free_axes = Vec::new();
    let mut rhs_free_axes = Vec::new();
    let mut lhs_batch_axes = Vec::new();
    let mut rhs_batch_axes = Vec::new();

    for output_axis in 0..output_rank {
        match (
            lhs_axes_by_output[output_axis],
            rhs_axes_by_output[output_axis],
        ) {
            (Some(lhs_axis), Some(rhs_axis)) => {
                batch_output_axes.push(output_axis);
                lhs_batch_axes.push(lhs_axis);
                rhs_batch_axes.push(rhs_axis);
            }
            (Some(lhs_axis), None) => {
                lhs_free_output_axes.push(output_axis);
                lhs_free_axes.push(lhs_axis);
            }
            (None, Some(rhs_axis)) => {
                rhs_free_output_axes.push(output_axis);
                rhs_free_axes.push(rhs_axis);
            }
            (None, None) => return None,
        }
    }

    Some(OuterProductAxisPartition {
        lhs_free_output_axes,
        rhs_free_output_axes,
        batch_output_axes,
        lhs_free_axes,
        rhs_free_axes,
        lhs_batch_axes,
        rhs_batch_axes,
    })
}

fn output_axes_match_partition(output_rank: usize, groups: &[&[usize]]) -> bool {
    groups
        .iter()
        .flat_map(|group| group.iter().copied())
        .eq(0..output_rank)
}

fn split_outer_product_plan<T>(
    lhs: &TypedTensorView<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<SplitOuterProductPlan>>
where
    T: 'static,
{
    let output_rank = lhs_shape.len();
    if lhs_shape != rhs_shape
        || !shape_matches_dims(lhs.shape(), lhs_shape, lhs_dims)
        || !shape_matches_dims(rhs.shape(), rhs_shape, rhs_dims)
        || lhs.backend_buffer().is_some()
        || rhs.backend_buffer().is_some()
        || lhs.offset() < 0
        || rhs.offset() < 0
        || lhs.strides().iter().any(|&stride| stride < 0)
        || rhs.strides().iter().any(|&stride| stride < 0)
    {
        return Ok(None);
    }

    let Some(partition) = classify_outer_product_axes(lhs_dims, rhs_dims, output_rank) else {
        return Ok(None);
    };

    let lhs_free_size = axes_shape_product("broadcast_multiply", lhs, &partition.lhs_free_axes)?;
    let rhs_free_size = axes_shape_product("broadcast_multiply", rhs, &partition.rhs_free_axes)?;
    if lhs_free_size <= 1 || rhs_free_size <= 1 {
        return Ok(None);
    }
    let batches = axes_shape_product("broadcast_multiply", lhs, &partition.lhs_batch_axes)?;

    let lhs_prefix = output_axes_match_partition(
        output_rank,
        &[
            &partition.lhs_free_output_axes,
            &partition.rhs_free_output_axes,
            &partition.batch_output_axes,
        ],
    );
    if lhs_prefix {
        return Ok(Some(SplitOuterProductPlan {
            rows: lhs_free_size,
            cols: rhs_free_size,
            batches,
            layout: SplitOuterProductLayout::LhsPrefix,
            lhs_free_axes: partition.lhs_free_axes,
            rhs_free_axes: partition.rhs_free_axes,
            lhs_batch_axes: partition.lhs_batch_axes,
            rhs_batch_axes: partition.rhs_batch_axes,
        }));
    }

    let rhs_prefix = output_axes_match_partition(
        output_rank,
        &[
            &partition.rhs_free_output_axes,
            &partition.lhs_free_output_axes,
            &partition.batch_output_axes,
        ],
    );
    if rhs_prefix {
        return Ok(Some(SplitOuterProductPlan {
            rows: rhs_free_size,
            cols: lhs_free_size,
            batches,
            layout: SplitOuterProductLayout::RhsPrefix,
            lhs_free_axes: partition.lhs_free_axes,
            rhs_free_axes: partition.rhs_free_axes,
            lhs_batch_axes: partition.lhs_batch_axes,
            rhs_batch_axes: partition.rhs_batch_axes,
        }));
    }

    Ok(None)
}

fn try_outer_product_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<TypedTensor<T>>>
where
    T: Copy + Clone + Mul<Output = T> + PoolScalar + 'static,
{
    let Some(plan) = split_outer_product_plan(lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims)?
    else {
        return Ok(None);
    };

    // SAFETY: every element in the column-major output is assigned below.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs_shape) };
    let lhs_view = typed_view_from_view("broadcast_multiply", lhs)?;
    let rhs_view = typed_view_from_view("broadcast_multiply", rhs)?;
    match plan.layout {
        SplitOuterProductLayout::LhsPrefix => {
            let lhs_perm: Vec<_> = plan
                .lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = plan
                .rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            batched_outer_product_into(
                &mut out.view_mut(),
                &lhs_outer,
                &rhs_outer,
                plan.lhs_free_axes.len(),
                plan.rhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
        }
        SplitOuterProductLayout::RhsPrefix => {
            let lhs_perm: Vec<_> = plan
                .lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = plan
                .rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            batched_outer_product_into(
                &mut out.view_mut(),
                &rhs_outer,
                &lhs_outer,
                plan.rhs_free_axes.len(),
                plan.lhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
        }
    }
    Ok(Some(tensor_from_array(out)))
}

struct LazyOuterProduct<T> {
    base: TypedTensor<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

fn axes_by_physical_stride<T>(view: &TypedTensorView<'_, T>, axes: &[usize]) -> Vec<usize>
where
    T: 'static,
{
    let mut sorted = axes.to_vec();
    sorted.sort_by(|&lhs_axis, &rhs_axis| {
        view.strides()[lhs_axis]
            .cmp(&view.strides()[rhs_axis])
            .then_with(|| lhs_axis.cmp(&rhs_axis))
    });
    sorted
}

fn append_axis_shapes<T>(shape: &mut Vec<usize>, view: &TypedTensorView<'_, T>, axes: &[usize])
where
    T: 'static,
{
    shape.extend(axes.iter().map(|&axis| view.shape()[axis]));
}

fn set_lazy_stride(
    logical_strides: &mut [Option<isize>],
    output_axis: usize,
    stride: isize,
) -> crate::Result<()> {
    let rank = logical_strides.len();
    let slot = logical_strides
        .get_mut(output_axis)
        .ok_or(crate::Error::AxisOutOfBounds {
            op: "broadcast_multiply",
            axis: output_axis,
            rank,
        })?;
    if slot.replace(stride).is_some() {
        return Err(crate::Error::DuplicateAxis {
            op: "broadcast_multiply",
            axis: output_axis,
            role: "lazy output layout",
        });
    }
    Ok(())
}

struct LazyOuterProductStrideSpec<'a> {
    output_shape: &'a [usize],
    base_shape: &'a [usize],
    leading_axes: &'a [usize],
    leading_dims: &'a [usize],
    trailing_axes: &'a [usize],
    trailing_dims: &'a [usize],
    lhs_batch_axes: &'a [usize],
    rhs_batch_axes: &'a [usize],
    lhs_dims: &'a [usize],
    rhs_dims: &'a [usize],
}

fn lazy_outer_product_strides(spec: LazyOuterProductStrideSpec<'_>) -> crate::Result<Vec<isize>> {
    let base_strides = col_major_strides(spec.base_shape);
    let mut logical_strides = vec![None; spec.output_shape.len()];
    let mut base_axis = 0usize;

    for &axis in spec.leading_axes {
        set_lazy_stride(
            &mut logical_strides,
            spec.leading_dims[axis],
            base_strides[base_axis],
        )?;
        base_axis += 1;
    }
    for &axis in spec.trailing_axes {
        set_lazy_stride(
            &mut logical_strides,
            spec.trailing_dims[axis],
            base_strides[base_axis],
        )?;
        base_axis += 1;
    }
    for (&lhs_axis, &rhs_axis) in spec.lhs_batch_axes.iter().zip(spec.rhs_batch_axes.iter()) {
        let output_axis = spec.lhs_dims[lhs_axis];
        if spec.rhs_dims[rhs_axis] != output_axis {
            return Err(crate::Error::backend_failure(
                "broadcast_multiply",
                "batch axes disagree while building lazy outer-product layout",
            ));
        }
        set_lazy_stride(&mut logical_strides, output_axis, base_strides[base_axis])?;
        base_axis += 1;
    }

    logical_strides
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            crate::Error::backend_failure(
                "broadcast_multiply",
                "lazy outer-product layout did not cover every output axis",
            )
        })
}

fn lazy_outer_product_value(
    tensor: Tensor,
    shape: Vec<usize>,
    strides: Vec<isize>,
) -> crate::Result<TensorValue> {
    Ok(TensorValue::View(TensorOwnedView::from_parts(
        Arc::new(tensor),
        shape,
        strides,
        0,
    )?))
}

fn try_lazy_outer_product_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<LazyOuterProduct<T>>>
where
    T: Copy + Clone + Mul<Output = T> + PoolScalar + 'static,
{
    let Some(plan) = split_outer_product_plan(lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims)?
    else {
        return Ok(None);
    };

    let lhs_free_axes = axes_by_physical_stride(lhs, &plan.lhs_free_axes);
    let rhs_free_axes = axes_by_physical_stride(rhs, &plan.rhs_free_axes);
    if lhs_free_axes == plan.lhs_free_axes && rhs_free_axes == plan.rhs_free_axes {
        return Ok(None);
    }

    let lhs_view = typed_view_from_view("broadcast_multiply", lhs)?;
    let rhs_view = typed_view_from_view("broadcast_multiply", rhs)?;

    match plan.layout {
        SplitOuterProductLayout::LhsPrefix => {
            let lhs_perm: Vec<_> = lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;

            let mut base_shape = Vec::with_capacity(lhs_shape.len());
            append_axis_shapes(&mut base_shape, lhs, &lhs_free_axes);
            append_axis_shapes(&mut base_shape, rhs, &rhs_free_axes);
            append_axis_shapes(&mut base_shape, lhs, &plan.lhs_batch_axes);
            let strides = lazy_outer_product_strides(LazyOuterProductStrideSpec {
                output_shape: lhs_shape,
                base_shape: &base_shape,
                leading_axes: &lhs_free_axes,
                leading_dims: lhs_dims,
                trailing_axes: &rhs_free_axes,
                trailing_dims: rhs_dims,
                lhs_batch_axes: &plan.lhs_batch_axes,
                rhs_batch_axes: &plan.rhs_batch_axes,
                lhs_dims,
                rhs_dims,
            })?;

            // SAFETY: every element in the physical base output is assigned below.
            let mut base = unsafe { typed_array_uninit_from_pool(buffers, &base_shape) };
            batched_outer_product_into(
                &mut base.view_mut(),
                &lhs_outer,
                &rhs_outer,
                lhs_free_axes.len(),
                rhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            Ok(Some(LazyOuterProduct {
                base: tensor_from_array(base),
                shape: lhs_shape.to_vec(),
                strides,
            }))
        }
        SplitOuterProductLayout::RhsPrefix => {
            let lhs_perm: Vec<_> = lhs_free_axes
                .iter()
                .chain(plan.lhs_batch_axes.iter())
                .copied()
                .collect();
            let rhs_perm: Vec<_> = rhs_free_axes
                .iter()
                .chain(plan.rhs_batch_axes.iter())
                .copied()
                .collect();
            let lhs_outer = lhs_view
                .permute(&lhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            let rhs_outer = rhs_view
                .permute(&rhs_perm)
                .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;

            let mut base_shape = Vec::with_capacity(lhs_shape.len());
            append_axis_shapes(&mut base_shape, rhs, &rhs_free_axes);
            append_axis_shapes(&mut base_shape, lhs, &lhs_free_axes);
            append_axis_shapes(&mut base_shape, lhs, &plan.lhs_batch_axes);
            let strides = lazy_outer_product_strides(LazyOuterProductStrideSpec {
                output_shape: lhs_shape,
                base_shape: &base_shape,
                leading_axes: &rhs_free_axes,
                leading_dims: rhs_dims,
                trailing_axes: &lhs_free_axes,
                trailing_dims: lhs_dims,
                lhs_batch_axes: &plan.lhs_batch_axes,
                rhs_batch_axes: &plan.rhs_batch_axes,
                lhs_dims,
                rhs_dims,
            })?;

            // SAFETY: every element in the physical base output is assigned below.
            let mut base = unsafe { typed_array_uninit_from_pool(buffers, &base_shape) };
            batched_outer_product_into(
                &mut base.view_mut(),
                &rhs_outer,
                &lhs_outer,
                rhs_free_axes.len(),
                lhs_free_axes.len(),
            )
            .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
            Ok(Some(LazyOuterProduct {
                base: tensor_from_array(base),
                shape: lhs_shape.to_vec(),
                strides,
            }))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn typed_broadcast_mul_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &TypedTensorView<'_, T, R>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs_shape != rhs_shape {
        return Err(crate::Error::ShapeMismatch {
            op: "broadcast_multiply",
            lhs: lhs_shape.to_vec(),
            rhs: rhs_shape.to_vec(),
        });
    }

    // SAFETY: broadcast_mul_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs_shape) };
    let lhs_view = typed_view_from_view("broadcast_multiply", lhs)?;
    let rhs_view = typed_view_from_view("broadcast_multiply", rhs)?;
    broadcast_mul_into(
        &mut out.view_mut(),
        &lhs_view,
        lhs_dims,
        &rhs_view,
        rhs_dims,
    )
    .map_err(|err| crate::Error::backend_failure("broadcast_multiply", err))?;
    Ok(tensor_from_array(out))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn broadcast_multiply_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<Tensor>> {
    let lhs = read_as_cpu_view(lhs);
    let rhs = read_as_cpu_view(rhs);

    macro_rules! dispatch {
        ($variant:ident, $lhs:expr, $rhs:expr) => {{
            if let Some(out) = try_outer_product_with_pool(
                buffers, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims,
            )? {
                return Ok(Some(Tensor::$variant(out)));
            }
            Ok(Some(Tensor::$variant(typed_broadcast_mul_view_with_pool(
                buffers, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims,
            )?)))
        }};
    }

    match (lhs, rhs) {
        (CpuReadView::F32(lhs), CpuReadView::F32(rhs)) => dispatch!(F32, lhs, rhs),
        (CpuReadView::F64(lhs), CpuReadView::F64(rhs)) => dispatch!(F64, lhs, rhs),
        (CpuReadView::I32(lhs), CpuReadView::I32(rhs)) => dispatch!(I32, lhs, rhs),
        (CpuReadView::I64(lhs), CpuReadView::I64(rhs)) => dispatch!(I64, lhs, rhs),
        (CpuReadView::C32(lhs), CpuReadView::C32(rhs)) => dispatch!(C32, lhs, rhs),
        (CpuReadView::C64(lhs), CpuReadView::C64(rhs)) => dispatch!(C64, lhs, rhs),
        _ => Ok(None),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn broadcast_multiply_value_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: TensorRead<'_>,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> crate::Result<Option<TensorValue>> {
    let lhs_view = read_as_cpu_view(lhs.clone());
    let rhs_view = read_as_cpu_view(rhs.clone());

    macro_rules! dispatch_lazy {
        ($variant:ident, $lhs:expr, $rhs:expr) => {{
            if let Some(out) = try_lazy_outer_product_with_pool(
                buffers, &$lhs, lhs_shape, lhs_dims, &$rhs, rhs_shape, rhs_dims,
            )? {
                return Ok(Some(lazy_outer_product_value(
                    Tensor::$variant(out.base),
                    out.shape,
                    out.strides,
                )?));
            }
        }};
    }

    match (lhs_view, rhs_view) {
        (CpuReadView::F32(lhs_view), CpuReadView::F32(rhs_view)) => {
            dispatch_lazy!(F32, lhs_view, rhs_view);
        }
        (CpuReadView::F64(lhs_view), CpuReadView::F64(rhs_view)) => {
            dispatch_lazy!(F64, lhs_view, rhs_view);
        }
        (CpuReadView::I32(lhs_view), CpuReadView::I32(rhs_view)) => {
            dispatch_lazy!(I32, lhs_view, rhs_view);
        }
        (CpuReadView::I64(lhs_view), CpuReadView::I64(rhs_view)) => {
            dispatch_lazy!(I64, lhs_view, rhs_view);
        }
        (CpuReadView::C32(lhs_view), CpuReadView::C32(rhs_view)) => {
            dispatch_lazy!(C32, lhs_view, rhs_view);
        }
        (CpuReadView::C64(lhs_view), CpuReadView::C64(rhs_view)) => {
            dispatch_lazy!(C64, lhs_view, rhs_view);
        }
        _ => {}
    }

    broadcast_multiply_read_with_pool(buffers, lhs, lhs_shape, lhs_dims, rhs, rhs_shape, rhs_dims)
        .map(|tensor| tensor.map(TensorValue::from_tensor))
}

/// Divide two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::div;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![8.0_f64, 15.0]);
/// let b = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 5.0]);
/// let out = div(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[4.0, 3.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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
        (Tensor::F32(a), Tensor::C32(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", a)?[0]);
            Ok(Tensor::C32(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C32(a), Tensor::F32(b)) if b.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", b)?[0]);
            Ok(Tensor::C32(typed_div_with_pool(buffers, a, &scalar)?))
        }
        (Tensor::F64(a), Tensor::C64(b)) if a.shape().is_empty() => {
            let scalar = complex_scalar_tensor(typed_host_data("div", a)?[0]);
            Ok(Tensor::C64(typed_div_with_pool(buffers, &scalar, b)?))
        }
        (Tensor::C64(a), Tensor::F64(b)) if b.shape().is_empty() => {
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

pub(crate) fn div_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::F32(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::F64(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::C32(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::C64(typed_binary_view_with_pool(
            "div",
            buffers,
            &a,
            &b,
            |x, y| x / y,
        )?)),
        (CpuReadView::F32(real), CpuReadView::C32(complex)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C32(typed_binary_view_with_pool(
                "div",
                buffers,
                &scalar,
                &complex,
                |x, y| x / y,
            )?))
        }
        (CpuReadView::C32(complex), CpuReadView::F32(real)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C32(typed_binary_view_with_pool(
                "div",
                buffers,
                &complex,
                &scalar,
                |x, y| x / y,
            )?))
        }
        (CpuReadView::F64(real), CpuReadView::C64(complex)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C64(typed_binary_view_with_pool(
                "div",
                buffers,
                &scalar,
                &complex,
                |x, y| x / y,
            )?))
        }
        (CpuReadView::C64(complex), CpuReadView::F64(real)) if real.shape().is_empty() => {
            let scalar = complex_scalar_tensor_from_view(&real)?;
            let scalar = scalar.as_view();
            Ok(Tensor::C64(typed_binary_view_with_pool(
                "div",
                buffers,
                &complex,
                &scalar,
                |x, y| x / y,
            )?))
        }
        _ => Err(crate::Error::DTypeMismatch {
            op: "div",
            lhs: lhs_dtype,
            rhs: rhs_dtype,
        }),
    }
}

/// Negate a CPU tensor elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::neg;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]);
/// let out = neg(&input)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn neg_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::C32(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::C64(typed_unary_view_with_pool(
            "neg",
            buffers,
            &t,
            |x| -x,
        )?)),
        _ => Err(crate::Error::backend_failure(
            "neg",
            format!("unsupported dtype {dtype:?}"),
        )),
    }
}

/// Conjugate a real or complex CPU tensor elementwise.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_cpu::conj;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)]);
/// let out = conj(&input)?;
/// assert_eq!(out.as_slice::<Complex64>().unwrap(), &[Complex64::new(1.0, -2.0)]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn conj_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::C32(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::C64(typed_unary_view_with_pool(
            "conj",
            buffers,
            &t,
            |x| x.conj_elem(),
        )?)),
        _ => Err(crate::Error::backend_failure(
            "conj",
            format!("unsupported dtype {dtype:?}"),
        )),
    }
}

/// Compute elementwise absolute values.
///
/// Complex inputs return real magnitudes (`C32 -> F32`, `C64 -> F64`).
///
/// # Examples
///
/// ```
/// use tenferro_cpu::abs;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![2], vec![-3.0_f64, 4.0]);
/// let out = abs(&input)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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
        Tensor::C32(t) => Ok(Tensor::F32(typed_complex_abs_with_pool(buffers, t)?)),
        Tensor::C64(t) => Ok(Tensor::F64(typed_complex_abs_with_pool(buffers, t)?)),
    }
}

pub(crate) fn abs_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "abs",
            buffers,
            &t,
            |x| x.abs_elem(),
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "abs",
            buffers,
            &t,
            |x| x.abs_elem(),
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::F32(typed_complex_abs_view_with_pool(buffers, &t)?)),
        CpuReadView::C64(t) => Ok(Tensor::F64(typed_complex_abs_view_with_pool(buffers, &t)?)),
        _ => Err(crate::Error::backend_failure(
            "abs",
            format!("unsupported dtype {dtype:?}"),
        )),
    }
}

/// Compute elementwise signs.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::sign;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![3], vec![-2.0_f64, 0.0, 3.0]);
/// let out = sign(&input)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0, 0.0, 1.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn sign_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let dtype = input.dtype();
    match read_as_cpu_view(input) {
        CpuReadView::F32(t) => Ok(Tensor::F32(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        CpuReadView::F64(t) => Ok(Tensor::F64(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        CpuReadView::C32(t) => Ok(Tensor::C32(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        CpuReadView::C64(t) => Ok(Tensor::C64(typed_unary_view_with_pool(
            "sign",
            buffers,
            &t,
            |x| x.sign_elem(),
        )?)),
        _ => Err(crate::Error::backend_failure(
            "sign",
            format!("unsupported dtype {dtype:?}"),
        )),
    }
}

/// Compute elementwise maximum values.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::maximum;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0]);
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let out = maximum(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0, 5.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn maximum_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::F32(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::F64(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::C32(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::C64(
            typed_same_shape_binary_view_with_pool("maximum", buffers, &a, &b, |x, y| {
                x.max_elem(y)
            })?,
        )),
        _ => Err(crate::Error::DTypeMismatch {
            op: "maximum",
            lhs: lhs_dtype,
            rhs: rhs_dtype,
        }),
    }
}

/// Compute elementwise minimum values.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::minimum;
/// use tenferro_tensor::Tensor;
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0]);
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let out = minimum(&a, &b)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn minimum_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::F32(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::F64(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::C32(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::C64(
            typed_same_shape_binary_view_with_pool("minimum", buffers, &a, &b, |x, y| {
                x.min_elem(y)
            })?,
        )),
        _ => Err(crate::Error::DTypeMismatch {
            op: "minimum",
            lhs: lhs_dtype,
            rhs: rhs_dtype,
        }),
    }
}

/// Compare two CPU tensors elementwise.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::compare;
/// use tenferro_tensor::{CompareDir, Tensor};
///
/// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 5.0]);
/// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let out = compare(&a, &b, &CompareDir::Gt)?;
/// assert_eq!(out.as_slice::<bool>().unwrap(), &[false, true]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn compare_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    dir: &CompareDir,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    match (read_as_cpu_view(lhs), read_as_cpu_view(rhs)) {
        (CpuReadView::F32(a), CpuReadView::F32(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        (CpuReadView::F64(a), CpuReadView::F64(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        (CpuReadView::I32(a), CpuReadView::I32(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        (CpuReadView::I64(a), CpuReadView::I64(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        (CpuReadView::Bool(a), CpuReadView::Bool(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        (CpuReadView::C32(a), CpuReadView::C32(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        (CpuReadView::C64(a), CpuReadView::C64(b)) => Ok(Tensor::Bool(
            typed_same_shape_binary_view_with_pool("compare", buffers, &a, &b, |x, y| {
                x.compare_elem(y, dir)
            })?,
        )),
        _ => Err(crate::Error::DTypeMismatch {
            op: "compare",
            lhs: lhs_dtype,
            rhs: rhs_dtype,
        }),
    }
}

/// Select values from two tensors using a boolean predicate tensor.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::select;
/// use tenferro_tensor::Tensor;
///
/// let pred = Tensor::from_vec_col_major(vec![2], vec![true, false]);
/// let on_true = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
/// let on_false = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]);
/// let out = select(&pred, &on_true, &on_false)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn select_read_with_pool(
    buffers: &mut BufferPool,
    pred: TensorRead<'_>,
    on_true: TensorRead<'_>,
    on_false: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let pred_dtype = pred.dtype();
    let true_dtype = on_true.dtype();
    let false_dtype = on_false.dtype();
    match (
        read_as_cpu_view(pred),
        read_as_cpu_view(on_true),
        read_as_cpu_view(on_false),
    ) {
        (CpuReadView::Bool(p), CpuReadView::F32(t), CpuReadView::F32(f)) => Ok(Tensor::F32(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::F64(t), CpuReadView::F64(f)) => Ok(Tensor::F64(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::I32(t), CpuReadView::I32(f)) => Ok(Tensor::I32(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::I64(t), CpuReadView::I64(f)) => Ok(Tensor::I64(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::Bool(t), CpuReadView::Bool(f)) => Ok(Tensor::Bool(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::C32(t), CpuReadView::C32(f)) => Ok(Tensor::C32(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(p), CpuReadView::C64(t), CpuReadView::C64(f)) => Ok(Tensor::C64(
            typed_select_view_with_pool(buffers, &p, &t, &f)?,
        )),
        (CpuReadView::Bool(_), _, _) => Err(crate::Error::DTypeMismatch {
            op: "select",
            lhs: true_dtype,
            rhs: false_dtype,
        }),
        _ => Err(crate::Error::DTypeMismatch {
            op: "select",
            lhs: pred_dtype,
            rhs: crate::DType::Bool,
        }),
    }
}

/// Clamp CPU tensor values elementwise between lower and upper bounds.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::clamp;
/// use tenferro_tensor::Tensor;
///
/// let input = Tensor::from_vec_col_major(vec![3], vec![-1.0_f64, 2.0, 8.0]);
/// let lower = Tensor::from_vec_col_major(vec![3], vec![0.0_f64, 0.0, 0.0]);
/// let upper = Tensor::from_vec_col_major(vec![3], vec![5.0_f64, 5.0, 5.0]);
/// let out = clamp(&input, &lower, &upper)?;
/// assert_eq!(out.as_slice::<f64>().unwrap(), &[0.0, 2.0, 5.0]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
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

pub(crate) fn clamp_read_with_pool(
    buffers: &mut BufferPool,
    input: TensorRead<'_>,
    lower: TensorRead<'_>,
    upper: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let input_dtype = input.dtype();
    let lower_dtype = lower.dtype();
    match (
        read_as_cpu_view(input),
        read_as_cpu_view(lower),
        read_as_cpu_view(upper),
    ) {
        (CpuReadView::F32(input), CpuReadView::F32(lower), CpuReadView::F32(upper)) => Ok(
            Tensor::F32(typed_clamp_view_with_pool(buffers, &input, &lower, &upper)?),
        ),
        (CpuReadView::F64(input), CpuReadView::F64(lower), CpuReadView::F64(upper)) => Ok(
            Tensor::F64(typed_clamp_view_with_pool(buffers, &input, &lower, &upper)?),
        ),
        (CpuReadView::C32(input), CpuReadView::C32(lower), CpuReadView::C32(upper)) => Ok(
            Tensor::C32(typed_clamp_view_with_pool(buffers, &input, &lower, &upper)?),
        ),
        (CpuReadView::C64(input), CpuReadView::C64(lower), CpuReadView::C64(upper)) => Ok(
            Tensor::C64(typed_clamp_view_with_pool(buffers, &input, &lower, &upper)?),
        ),
        _ => Err(crate::Error::DTypeMismatch {
            op: "clamp",
            lhs: input_dtype,
            rhs: lower_dtype,
        }),
    }
}

pub(crate) fn typed_add_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T> + PoolScalar,
{
    if lhs.shape() == rhs.shape() {
        // SAFETY: zip_map2_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view("add", lhs)?,
            &typed_view("add", rhs)?,
            |x, y| x + y,
        )
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape().is_empty() {
        let scalar = typed_host_data("add", lhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, rhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view("add", rhs)?, |x| {
            scalar + x
        })
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape().is_empty() {
        let scalar = typed_host_data("add", rhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view("add", lhs)?, |x| {
            x + scalar
        })
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "add",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }
}

pub(crate) fn typed_add_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Add<Output = T> + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() == rhs.shape() {
        // SAFETY: zip_map2_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view_from_view("add", lhs)?,
            &typed_view_from_view("add", rhs)?,
            |x, y| x + y,
        )
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape().is_empty() {
        let scalar = typed_view_from_view("add", lhs)?.get(&[]);
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, rhs.shape()) };
        map_into(
            &mut out.view_mut(),
            &typed_view_from_view("add", rhs)?,
            |x| scalar + x,
        )
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape().is_empty() {
        let scalar = typed_view_from_view("add", rhs)?.get(&[]);
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        map_into(
            &mut out.view_mut(),
            &typed_view_from_view("add", lhs)?,
            |x| x + scalar,
        )
        .map_err(|err| crate::Error::backend_failure("add", err.to_string()))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "add",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }
}

pub(crate) fn typed_mul_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar + 'static,
{
    if lhs.shape() == rhs.shape() {
        // SAFETY: mul_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        mul_into(
            &mut out.view_mut(),
            &typed_view("mul", lhs)?,
            &typed_view("mul", rhs)?,
        )
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape().is_empty() {
        let scalar = typed_host_data("mul", lhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, rhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view("mul", rhs)?, |x| {
            scalar * x
        })
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape().is_empty() {
        let scalar = typed_host_data("mul", rhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view("mul", lhs)?, |x| {
            x * scalar
        })
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "mul",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }
}

pub(crate) fn typed_mul_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Mul<Output = T> + PoolScalar + 'static,
    L: TensorRank,
    R: TensorRank,
{
    if lhs.shape() == rhs.shape() {
        // SAFETY: mul_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        mul_into(
            &mut out.view_mut(),
            &typed_view_from_view("mul", lhs)?,
            &typed_view_from_view("mul", rhs)?,
        )
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape().is_empty() {
        let scalar = typed_view_from_view("mul", lhs)?.get(&[]);
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, rhs.shape()) };
        map_into(
            &mut out.view_mut(),
            &typed_view_from_view("mul", rhs)?,
            |x| scalar * x,
        )
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape().is_empty() {
        let scalar = typed_view_from_view("mul", rhs)?.get(&[]);
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        map_into(
            &mut out.view_mut(),
            &typed_view_from_view("mul", lhs)?,
            |x| x * scalar,
        )
        .map_err(|err| crate::Error::backend_failure("mul", err))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "mul",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }
}

pub(crate) fn typed_div_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Div<Output = T> + PoolScalar,
{
    if lhs.shape() == rhs.shape() {
        // SAFETY: zip_map2_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        zip_map2_into(
            &mut out.view_mut(),
            &typed_view("div", lhs)?,
            &typed_view("div", rhs)?,
            |x, y| x / y,
        )
        .map_err(|err| crate::Error::backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else if lhs.shape().is_empty() {
        let scalar = typed_host_data("div", lhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, rhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view("div", rhs)?, |x| {
            scalar / x
        })
        .map_err(|err| crate::Error::backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else if rhs.shape().is_empty() {
        let scalar = typed_host_data("div", rhs)?[0];
        // SAFETY: map_into overwrites every output element.
        let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
        map_into(&mut out.view_mut(), &typed_view("div", lhs)?, |x| {
            x / scalar
        })
        .map_err(|err| crate::Error::backend_failure("div", err))?;
        Ok(tensor_from_array(out))
    } else {
        Err(crate::Error::ShapeMismatch {
            op: "div",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        })
    }
}

pub(crate) fn typed_neg_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + Neg<Output = T> + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(&mut out.view_mut(), &typed_view("neg", input)?, |x| -x)
        .map_err(|err| crate::Error::backend_failure("neg", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_conj_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + Clone + Zero + ConjElem + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(&mut out.view_mut(), &typed_view("conj", input)?, |x| {
        x.conj_elem()
    })
    .map_err(|err| crate::Error::backend_failure("conj", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_abs_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(&mut out.view_mut(), &typed_view("abs", input)?, |x| {
        x.abs_elem()
    })
    .map_err(|err| crate::Error::backend_failure("abs", err))?;
    Ok(tensor_from_array(out))
}

fn typed_complex_abs_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<Complex<T>>,
) -> crate::Result<TypedTensor<T>>
where
    T: num_traits::Float + PoolScalar,
{
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(&mut out.view_mut(), &typed_view("abs", input)?, |x| {
        x.norm()
    })
    .map_err(|err| crate::Error::backend_failure("abs", err))?;
    Ok(tensor_from_array(out))
}

fn typed_complex_abs_view_with_pool<T, R>(
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, Complex<T>, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: num_traits::Float + PoolScalar + 'static,
    R: TensorRank,
{
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(
        &mut out.view_mut(),
        &typed_view_from_view("abs", input)?,
        |x| x.norm(),
    )
    .map_err(|err| crate::Error::backend_failure("abs", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_sign_with_pool<T>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    // SAFETY: map_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
    map_into(&mut out.view_mut(), &typed_view("sign", input)?, |x| {
        x.sign_elem()
    })
    .map_err(|err| crate::Error::backend_failure("sign", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_maximum_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "maximum",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }
    // SAFETY: zip_map2_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view("maximum", lhs)?,
        &typed_view("maximum", rhs)?,
        |x, y| x.max_elem(y),
    )
    .map_err(|err| crate::Error::backend_failure("maximum", err))?;
    Ok(tensor_from_array(out))
}

pub(crate) fn typed_minimum_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: Tier2Elem + PoolScalar,
{
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "minimum",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }
    // SAFETY: zip_map2_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
    zip_map2_into(
        &mut out.view_mut(),
        &typed_view("minimum", lhs)?,
        &typed_view("minimum", rhs)?,
        |x, y| x.min_elem(y),
    )
    .map_err(|err| crate::Error::backend_failure("minimum", err))?;
    Ok(tensor_from_array(out))
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
    if lhs.shape() != rhs.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "compare",
            lhs: lhs.shape().to_vec(),
            rhs: rhs.shape().to_vec(),
        });
    }
    // SAFETY: zip_map2_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, lhs.shape()) };
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
    if pred.shape() != on_true.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "select",
            lhs: pred.shape().to_vec(),
            rhs: on_true.shape().to_vec(),
        });
    }
    if pred.shape() != on_false.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "select",
            lhs: pred.shape().to_vec(),
            rhs: on_false.shape().to_vec(),
        });
    }
    // SAFETY: zip_map3_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, pred.shape()) };
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
    if input.shape() != lower.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "clamp",
            lhs: input.shape().to_vec(),
            rhs: lower.shape().to_vec(),
        });
    }
    if input.shape() != upper.shape() {
        return Err(crate::Error::ShapeMismatch {
            op: "clamp",
            lhs: input.shape().to_vec(),
            rhs: upper.shape().to_vec(),
        });
    }
    // SAFETY: zip_map3_into overwrites every output element.
    let mut out = unsafe { typed_array_uninit_from_pool(buffers, input.shape()) };
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

#[cfg(test)]
mod tests;
