use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use std::mem::MaybeUninit;
use strided_kernel::{map_into, reduce, zip_map2_into, StridedView};
use tenferro_core_ops::PrimitiveOpKind;

use super::{typed_view, typed_view_from_view, PooledUninitOutput};
use crate::buffer_pool::{BufferPool, PoolScalar};
use tenferro_tensor::{
    BackendId, CapabilityAxis, DType, Tensor, TensorRank, TensorRead, TensorScalar, TensorView,
    TypedTensor, TypedTensorView,
};

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

trait IntegerPowElem: PowElem + TensorScalar + PoolScalar + Send + Sync + 'static {
    fn is_negative_exponent(self) -> bool;
    fn wrapping_pow_nonnegative(self, exponent: Self) -> Self;
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

macro_rules! impl_integer_pow_elem {
    ($ty:ty) => {
        impl PowElem for $ty {
            fn pow_elem(self, exponent: Self) -> Self {
                self.wrapping_pow_nonnegative(exponent)
            }
        }

        impl IntegerPowElem for $ty {
            fn is_negative_exponent(self) -> bool {
                self < 0
            }

            fn wrapping_pow_nonnegative(self, exponent: Self) -> Self {
                let mut base = self;
                let mut exp = exponent as u64;
                let mut acc: Self = 1;
                while exp != 0 {
                    if exp & 1 == 1 {
                        acc = acc.wrapping_mul(base);
                    }
                    exp >>= 1;
                    if exp != 0 {
                        base = base.wrapping_mul(base);
                    }
                }
                acc
            }
        }
    };
}

impl_integer_pow_elem!(i32);
impl_integer_pow_elem!(i64);

#[cfg(test)]
fn with_test_pool<T>(f: impl FnOnce(&mut BufferPool) -> T) -> T {
    let mut buffers = BufferPool::new();
    f(&mut buffers)
}

enum AnalyticReadView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool,
    C32(TypedTensorView<'a, Complex32>),
    C64(TypedTensorView<'a, Complex64>),
}

/// The typed tensor behind a read adapter's tensor, or the refusal this view reports for one.
fn analytic_read_operand<T: tenferro_tensor::TensorScalar>(
    tensor: &Tensor,
) -> crate::Result<&TypedTensor<T>> {
    tensor.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported_dtype(
            "read_as_analytic_view",
            tensor.dtype(),
            "an externally defined payload has no analytic read view",
        )
    })
}

fn read_as_analytic_view(input: TensorRead<'_>) -> crate::Result<AnalyticReadView<'_>> {
    Ok(match input {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            DType::F32 => AnalyticReadView::F32(analytic_read_operand::<f32>(tensor)?.as_view()),
            DType::F64 => AnalyticReadView::F64(analytic_read_operand::<f64>(tensor)?.as_view()),
            DType::I32 => AnalyticReadView::I32(analytic_read_operand::<i32>(tensor)?.as_view()),
            DType::I64 => AnalyticReadView::I64(analytic_read_operand::<i64>(tensor)?.as_view()),
            DType::C32 => {
                AnalyticReadView::C32(analytic_read_operand::<Complex32>(tensor)?.as_view())
            }
            DType::C64 => {
                AnalyticReadView::C64(analytic_read_operand::<Complex64>(tensor)?.as_view())
            }
            DType::Bool => AnalyticReadView::Bool,
            // A caller-owned payload has no analytic read view.
            DType::External(type_id) => {
                return Err(crate::Error::unsupported_dtype(
                    "read_as_analytic_view",
                    DType::External(type_id),
                    "an externally defined payload has no analytic read view",
                ));
            }
        },
        TensorRead::View(TensorView::F32(view)) => AnalyticReadView::F32(view),
        TensorRead::View(TensorView::F64(view)) => AnalyticReadView::F64(view),
        TensorRead::View(TensorView::I32(view)) => AnalyticReadView::I32(view),
        TensorRead::View(TensorView::I64(view)) => AnalyticReadView::I64(view),
        TensorRead::View(TensorView::Bool(_)) => AnalyticReadView::Bool,
        TensorRead::View(TensorView::C32(view)) => AnalyticReadView::C32(view),
        TensorRead::View(TensorView::C64(view)) => AnalyticReadView::C64(view),
    })
}

// Share the traversal for owned/read entrypoints while keeping F statically known.
fn map_unary<T: Copy + PoolScalar>(
    op: &'static str,
    output: &mut strided_kernel::StridedViewMut<'_, MaybeUninit<T>>,
    input: &StridedView<'_, T>,
    f: impl Fn(T) -> T + Copy + Sync,
) -> crate::Result<()> {
    map_into(output, input, |x| MaybeUninit::new(f(x)))
        .map_err(|err| crate::Error::backend_source(op, err))
}

fn typed_unary_with_pool<T>(
    op: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    f: impl Fn(T) -> T + Copy + Sync,
) -> crate::Result<TypedTensor<T>>
where
    T: Copy + PoolScalar + 'static,
{
    let mut out = PooledUninitOutput::<T>::new(buffers, input.shape().to_vec())?;
    map_unary(
        op,
        &mut out.as_uninit_view_mut()?,
        &typed_view(op, input)?,
        f,
    )?;
    // SAFETY: the successful map replay writes every logical destination element.
    unsafe { out.assume_init() }
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
    let mut out = PooledUninitOutput::<T>::new(buffers, input.shape().to_vec())?;
    map_unary(
        op,
        &mut out.as_uninit_view_mut()?,
        &typed_view_from_view(op, input)?,
        f,
    )?;
    // SAFETY: the successful map replay writes every logical destination element.
    unsafe { out.assume_init() }
}

fn typed_unary_tensor_with_pool<T>(
    op: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    f: impl Fn(T) -> T + Copy + Sync,
) -> crate::Result<Tensor>
where
    T: Copy + PoolScalar + TensorScalar + 'static,
{
    let out = typed_unary_with_pool(op, buffers, input, f)?;
    Ok(T::typed_tensor_into_tensor(out))
}

fn typed_unary_view_tensor_with_pool<T, R>(
    op: &'static str,
    buffers: &mut BufferPool,
    input: &TypedTensorView<'_, T, R>,
    f: impl Fn(T) -> T + Copy + Sync,
) -> crate::Result<Tensor>
where
    T: Copy + PoolScalar + TensorScalar + 'static,
    R: TensorRank,
{
    let out = typed_unary_view_with_pool(op, buffers, input, f)?;
    Ok(T::typed_tensor_into_tensor(out))
}

fn unsupported_analytic_dtype_message(dtype: DType) -> String {
    let remedy = matches!(dtype, DType::I32 | DType::I64)
        .then_some("; convert to F64 before this operation");
    format!(
        "CPU backend does not support this operation for {dtype:?}; supported dtypes: F32/F64/C32/C64{}",
        remedy.unwrap_or("")
    )
}

fn require_cpu_capability(
    op_kind: PrimitiveOpKind,
    op: &'static str,
    dtype: DType,
    axis: CapabilityAxis,
) -> crate::Result<()> {
    let supported = crate::cpu_capabilities()
        .iter()
        .copied()
        .find(|entry| entry.op == op_kind && entry.dtype == dtype)
        .is_some_and(|entry| entry.axis(axis).is_supported());
    if supported {
        Ok(())
    } else {
        Err(crate::Error::unsupported_dtype(
            op,
            dtype,
            unsupported_analytic_dtype_message(dtype),
        ))
    }
}

fn strided_view_contains<T>(
    op: &'static str,
    view: &StridedView<'_, T>,
    pred: impl Fn(T) -> bool + Copy + Sync,
) -> crate::Result<bool>
where
    T: Copy + Send + Sync,
{
    reduce(view, pred, |lhs, rhs| lhs || rhs, false)
        .map_err(|err| crate::Error::backend_source(op, err))
}

fn ensure_nonnegative_integer_exponents<T>(
    op: &'static str,
    rhs: &StridedView<'_, T>,
) -> crate::Result<()>
where
    T: IntegerPowElem,
{
    if strided_view_contains(op, rhs, |value| value.is_negative_exponent())? {
        return Err(crate::cpu_negative_integer_exponent(op, T::dtype()));
    }
    Ok(())
}

fn replay_pow_pair<T: PowElem + PoolScalar>(
    op: &'static str,
    out: &mut strided_kernel::StridedViewMut<'_, MaybeUninit<T>>,
    lhs: &StridedView<'_, T>,
    rhs: &StridedView<'_, T>,
) -> crate::Result<()> {
    zip_map2_into(out, lhs, rhs, |x, y| MaybeUninit::new(x.pow_elem(y)))
        .map_err(|err| crate::Error::backend_source(op, err))
}

fn replay_pow_left<T: PowElem + PoolScalar>(
    op: &'static str,
    out: &mut strided_kernel::StridedViewMut<'_, MaybeUninit<T>>,
    input: &StridedView<'_, T>,
    scalar: T,
) -> crate::Result<()> {
    map_into(out, input, |x| MaybeUninit::new(scalar.pow_elem(x)))
        .map_err(|err| crate::Error::backend_source(op, err))
}

fn replay_pow_right<T: PowElem + PoolScalar>(
    op: &'static str,
    out: &mut strided_kernel::StridedViewMut<'_, MaybeUninit<T>>,
    input: &StridedView<'_, T>,
    scalar: T,
) -> crate::Result<()> {
    map_into(out, input, |x| MaybeUninit::new(x.pow_elem(scalar)))
        .map_err(|err| crate::Error::backend_source(op, err))
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
    let output_shape = if lhs.shape() == rhs.shape() {
        lhs.shape()
    } else if lhs.shape().is_empty() {
        rhs.shape()
    } else if rhs.shape().is_empty() {
        lhs.shape()
    } else {
        return Err(crate::Error::shape_mismatch(
            op,
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    };
    let mut out = PooledUninitOutput::<T>::new(buffers, output_shape.to_vec())?;
    if lhs.shape() == rhs.shape() {
        replay_pow_pair(
            op,
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view(op, lhs)?,
            &typed_view_from_view(op, rhs)?,
        )?;
    } else if lhs.shape().is_empty() {
        let scalar = typed_view_from_view(op, lhs)?.get(&[]);
        replay_pow_left(
            op,
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view(op, rhs)?,
            scalar,
        )?;
    } else {
        let scalar = typed_view_from_view(op, rhs)?.get(&[]);
        replay_pow_right(
            op,
            &mut out.as_uninit_view_mut()?,
            &typed_view_from_view(op, lhs)?,
            scalar,
        )?;
    }
    // SAFETY: the selected map kernel writes every logical destination element.
    unsafe { out.assume_init() }
}

/// Generate only the read half.
///
/// Used for an operation whose owner one-shot was removed by issue #1926: the
/// owned `*_with_pool` helper no longer has a library caller, so generating it
/// would leave dead code. Each arm below adds its own test-only owned
/// convenience on top of whichever helper it keeps.
macro_rules! define_unary_analytic_read_with_pool {
    ($dispatch_fn:ident, $dispatch_read_with_pool_fn:ident, $op_kind:ident, $elem_fn:ident) => {
        pub(crate) fn $dispatch_read_with_pool_fn(
            buffers: &mut BufferPool,
            input: TensorRead<'_>,
        ) -> crate::Result<Tensor> {
            let dtype = input.dtype();
            require_cpu_capability(
                PrimitiveOpKind::$op_kind,
                stringify!($dispatch_fn),
                dtype,
                CapabilityAxis::ReadInputs,
            )?;
            tenferro_tensor::with_scalar_read!(
                input,
                float_complex,
                backend = BackendId::Cpu,
                op = stringify!($dispatch_fn),
                |view| -> crate::Result<Tensor> {
                    typed_unary_view_tensor_with_pool(
                        stringify!($dispatch_fn),
                        buffers,
                        &view,
                        UnaryAnalyticElem::$elem_fn,
                    )
                }
            )
        }
    };
}

macro_rules! define_unary_analytic_dispatch {
    // The owner one-shot was removed, so the read half is the only entry point
    // and the test-only convenience is built on it.
    (read_only: $dispatch_fn:ident, $dispatch_read_with_pool_fn:ident, $op_kind:ident, $elem_fn:ident) => {
        #[cfg(test)]
        pub(crate) fn $dispatch_fn(input: &Tensor) -> crate::Result<Tensor> {
            with_test_pool(|buffers| {
                $dispatch_read_with_pool_fn(buffers, TensorRead::from_tensor(input))
            })
        }

        define_unary_analytic_read_with_pool!(
            $dispatch_fn,
            $dispatch_read_with_pool_fn,
            $op_kind,
            $elem_fn
        );
    };

    ($dispatch_fn:ident, $dispatch_with_pool_fn:ident, $dispatch_read_with_pool_fn:ident, $op_kind:ident, $elem_fn:ident) => {
        #[cfg(test)]
        pub(crate) fn $dispatch_fn(input: &Tensor) -> crate::Result<Tensor> {
            with_test_pool(|buffers| $dispatch_with_pool_fn(buffers, input))
        }

        pub(crate) fn $dispatch_with_pool_fn(
            buffers: &mut BufferPool,
            input: &Tensor,
        ) -> crate::Result<Tensor> {
            require_cpu_capability(
                PrimitiveOpKind::$op_kind,
                stringify!($dispatch_fn),
                input.dtype(),
                CapabilityAxis::OwnedResult,
            )?;
            tenferro_tensor::with_scalar!(
                input,
                float_complex,
                backend = BackendId::Cpu,
                op = stringify!($dispatch_fn),
                |tensor| -> crate::Result<Tensor> {
                    typed_unary_tensor_with_pool(
                        stringify!($dispatch_fn),
                        buffers,
                        tensor,
                        UnaryAnalyticElem::$elem_fn,
                    )
                }
            )
        }

        define_unary_analytic_read_with_pool!(
            $dispatch_fn,
            $dispatch_read_with_pool_fn,
            $op_kind,
            $elem_fn
        );
    };
}

define_unary_analytic_dispatch!(exp, exp_with_pool, exp_read_with_pool, Exp, exp_elem);
define_unary_analytic_dispatch!(read_only: log, log_read_with_pool, Log, log_elem);
define_unary_analytic_dispatch!(read_only: sin, sin_read_with_pool, Sin, sin_elem);
define_unary_analytic_dispatch!(read_only: cos, cos_read_with_pool, Cos, cos_elem);
define_unary_analytic_dispatch!(read_only: tanh, tanh_read_with_pool, Tanh, tanh_elem);
define_unary_analytic_dispatch!(read_only: sqrt, sqrt_read_with_pool, Sqrt, sqrt_elem);
define_unary_analytic_dispatch!(read_only: rsqrt, rsqrt_read_with_pool, Rsqrt, rsqrt_elem);
define_unary_analytic_dispatch!(read_only: expm1, expm1_read_with_pool, Expm1, expm1_elem);
define_unary_analytic_dispatch!(read_only: log1p, log1p_read_with_pool, Log1p, log1p_elem);

#[cfg(test)]
pub(crate) fn pow(lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
    with_test_pool(|buffers| pow_with_pool(buffers, lhs, rhs))
}

pub(crate) fn pow_with_pool(
    buffers: &mut BufferPool,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    match (lhs.dtype(), rhs.dtype()) {
        (DType::F32, DType::F32) => {
            let (a, b) = analytic_pair_operands::<f32>(lhs, rhs)?;
            Ok(Tensor::from_typed::<f32>(typed_pow_with_pool(
                buffers, a, b,
            )?))
        }
        (DType::F64, DType::F64) => {
            let (a, b) = analytic_pair_operands::<f64>(lhs, rhs)?;
            Ok(Tensor::from_typed::<f64>(typed_pow_with_pool(
                buffers, a, b,
            )?))
        }
        (DType::I32, DType::I32) => {
            let (a, b) = analytic_pair_operands::<i32>(lhs, rhs)?;
            Ok(Tensor::from_typed::<i32>(typed_integer_pow_with_pool(
                buffers, a, b,
            )?))
        }
        (DType::I64, DType::I64) => {
            let (a, b) = analytic_pair_operands::<i64>(lhs, rhs)?;
            Ok(Tensor::from_typed::<i64>(typed_integer_pow_with_pool(
                buffers, a, b,
            )?))
        }
        (DType::C32, DType::C32) => {
            let (a, b) = analytic_pair_operands::<Complex32>(lhs, rhs)?;
            Ok(Tensor::from_typed::<Complex32>(typed_pow_with_pool(
                buffers, a, b,
            )?))
        }
        (DType::C64, DType::C64) => {
            let (a, b) = analytic_pair_operands::<Complex64>(lhs, rhs)?;
            Ok(Tensor::from_typed::<Complex64>(typed_pow_with_pool(
                buffers, a, b,
            )?))
        }
        _ => Err(crate::Error::dtype_mismatch(
            "pow",
            lhs.dtype(),
            rhs.dtype(),
        )),
    }
}
/// The typed operands behind a same-dtype pair, or the refusal a mismatched pair reports.
fn analytic_pair_operands<'a, T: tenferro_tensor::TensorScalar>(
    lhs: &'a Tensor,
    rhs: &'a Tensor,
) -> crate::Result<(&'a TypedTensor<T>, &'a TypedTensor<T>)> {
    let a = lhs
        .as_typed::<T>()
        .ok_or_else(|| crate::Error::dtype_mismatch("pow", lhs.dtype(), rhs.dtype()))?;
    let b = rhs
        .as_typed::<T>()
        .ok_or_else(|| crate::Error::dtype_mismatch("pow", lhs.dtype(), rhs.dtype()))?;
    Ok((a, b))
}

pub(crate) fn pow_read_with_pool(
    buffers: &mut BufferPool,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
) -> crate::Result<Tensor> {
    let lhs_dtype = lhs.dtype();
    let rhs_dtype = rhs.dtype();
    let (lhs_view, rhs_view) = (read_as_analytic_view(lhs)?, read_as_analytic_view(rhs)?);
    match (lhs_view, rhs_view) {
        (AnalyticReadView::F32(a), AnalyticReadView::F32(b)) => Ok(Tensor::from_typed::<f32>(
            typed_pow_view_with_pool("pow", buffers, &a, &b)?,
        )),
        (AnalyticReadView::F64(a), AnalyticReadView::F64(b)) => Ok(Tensor::from_typed::<f64>(
            typed_pow_view_with_pool("pow", buffers, &a, &b)?,
        )),
        (AnalyticReadView::I32(a), AnalyticReadView::I32(b)) => Ok(Tensor::from_typed::<i32>(
            typed_integer_pow_view_with_pool(buffers, &a, &b)?,
        )),
        (AnalyticReadView::I64(a), AnalyticReadView::I64(b)) => Ok(Tensor::from_typed::<i64>(
            typed_integer_pow_view_with_pool(buffers, &a, &b)?,
        )),
        (AnalyticReadView::C32(a), AnalyticReadView::C32(b)) => Ok(
            Tensor::from_typed::<Complex32>(typed_pow_view_with_pool("pow", buffers, &a, &b)?),
        ),
        (AnalyticReadView::C64(a), AnalyticReadView::C64(b)) => Ok(
            Tensor::from_typed::<Complex64>(typed_pow_view_with_pool("pow", buffers, &a, &b)?),
        ),
        _ => Err(crate::Error::dtype_mismatch("pow", lhs_dtype, rhs_dtype)),
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
    let output_shape = if lhs.shape() == rhs.shape() {
        lhs.shape()
    } else if lhs.shape().is_empty() {
        rhs.shape()
    } else if rhs.shape().is_empty() {
        lhs.shape()
    } else {
        return Err(crate::Error::shape_mismatch(
            "pow",
            lhs.shape().to_vec(),
            rhs.shape().to_vec(),
        ));
    };
    let mut out = PooledUninitOutput::<T>::new(buffers, output_shape.to_vec())?;
    if lhs.shape() == rhs.shape() {
        replay_pow_pair(
            "pow",
            &mut out.as_uninit_view_mut()?,
            &typed_view("pow", lhs)?,
            &typed_view("pow", rhs)?,
        )?;
    } else if lhs.shape().is_empty() {
        let scalar = typed_view("pow", lhs)?.get(&[]);
        replay_pow_left(
            "pow",
            &mut out.as_uninit_view_mut()?,
            &typed_view("pow", rhs)?,
            scalar,
        )?;
    } else {
        let scalar = typed_view("pow", rhs)?.get(&[]);
        replay_pow_right(
            "pow",
            &mut out.as_uninit_view_mut()?,
            &typed_view("pow", lhs)?,
            scalar,
        )?;
    }
    // SAFETY: the selected map kernel writes every logical destination element.
    unsafe { out.assume_init() }
}

fn typed_integer_pow_with_pool<T>(
    buffers: &mut BufferPool,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
) -> crate::Result<TypedTensor<T>>
where
    T: IntegerPowElem,
{
    let rhs_view = typed_view("pow", rhs)?;
    ensure_nonnegative_integer_exponents("pow", &rhs_view)?;
    typed_pow_with_pool(buffers, lhs, rhs)
}

fn typed_integer_pow_view_with_pool<T, L, R>(
    buffers: &mut BufferPool,
    lhs: &TypedTensorView<'_, T, L>,
    rhs: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: IntegerPowElem,
    L: TensorRank,
    R: TensorRank,
{
    let rhs_view = typed_view_from_view("pow", rhs)?;
    ensure_nonnegative_integer_exponents("pow", &rhs_view)?;
    typed_pow_view_with_pool("pow", buffers, lhs, rhs)
}
