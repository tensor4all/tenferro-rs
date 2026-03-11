use std::any::TypeId;

use num_complex::{Complex32, Complex64, ComplexFloat};
use num_traits::Float;
use tenferro_algebra::{Scalar, Standard};
use tenferro_device::{Error, Result};
use tenferro_tensor::Tensor;

use crate::cpu::{tensor_to_view, tensor_to_view_mut};
use crate::family_cpu_common::{
    execute_binary_map, execute_unary_map, is_supported_ordered_real_type,
    is_supported_scalar_type, validate_pointwise_shapes, ComplexCpuScalarValue, CpuScalarValue,
};
use crate::{
    validate_execute_inputs, AnalyticBinaryOp, AnalyticPrimsDescriptor, AnalyticUnaryOp,
    CpuBackend, CpuContext, TensorAnalyticPrims,
};

/// CPU execution plan for the analytic protocol family.
///
/// # Examples
///
/// ```ignore
/// use tenferro_prims::CpuAnalyticPlan;
/// let _ = std::mem::size_of::<CpuAnalyticPlan>();
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuAnalyticPlan {
    PointwiseUnary { op: AnalyticUnaryOp },
    PointwiseBinary { op: AnalyticBinaryOp },
}

fn supports_analytic_unary<S: Scalar + 'static>(op: AnalyticUnaryOp) -> bool {
    is_supported_scalar_type::<S>()
        && matches!(
            op,
            AnalyticUnaryOp::Sqrt
                | AnalyticUnaryOp::Rsqrt
                | AnalyticUnaryOp::Exp
                | AnalyticUnaryOp::Expm1
                | AnalyticUnaryOp::Log
                | AnalyticUnaryOp::Log1p
                | AnalyticUnaryOp::Sin
                | AnalyticUnaryOp::Cos
                | AnalyticUnaryOp::Tan
                | AnalyticUnaryOp::Tanh
        )
}

fn supports_analytic_binary<S: Scalar + 'static>(op: AnalyticBinaryOp) -> bool {
    match op {
        AnalyticBinaryOp::Pow | AnalyticBinaryOp::Xlogy => is_supported_scalar_type::<S>(),
        AnalyticBinaryOp::Atan2 | AnalyticBinaryOp::Hypot => is_supported_ordered_real_type::<S>(),
    }
}

fn execute_analytic_unary_typed<S: CpuScalarValue>(
    alpha: S,
    input: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: AnalyticUnaryOp,
) -> Result<()> {
    match op {
        AnalyticUnaryOp::Sqrt => execute_unary_map(alpha, input, beta, output, |x| x.sqrt()),
        AnalyticUnaryOp::Rsqrt => {
            execute_unary_map(alpha, input, beta, output, |x| S::one() / x.sqrt())
        }
        AnalyticUnaryOp::Exp => execute_unary_map(alpha, input, beta, output, |x| x.exp()),
        AnalyticUnaryOp::Expm1 => {
            execute_unary_map(alpha, input, beta, output, |x| x.exp() - S::one())
        }
        AnalyticUnaryOp::Log => execute_unary_map(alpha, input, beta, output, |x| x.ln()),
        AnalyticUnaryOp::Log1p => {
            execute_unary_map(alpha, input, beta, output, |x| (x + S::one()).ln())
        }
        AnalyticUnaryOp::Sin => execute_unary_map(alpha, input, beta, output, |x| x.sin()),
        AnalyticUnaryOp::Cos => execute_unary_map(alpha, input, beta, output, |x| x.cos()),
        AnalyticUnaryOp::Tan => execute_unary_map(alpha, input, beta, output, |x| x.tan()),
        AnalyticUnaryOp::Tanh => execute_unary_map(alpha, input, beta, output, |x| x.tanh()),
    }
}

fn execute_analytic_binary_real<S: Float + CpuScalarValue>(
    alpha: S,
    lhs: &strided_view::StridedView<S>,
    rhs: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: AnalyticBinaryOp,
) -> Result<()> {
    match op {
        AnalyticBinaryOp::Pow => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| Float::powf(x, y))
        }
        AnalyticBinaryOp::Atan2 => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x.atan2(y))
        }
        AnalyticBinaryOp::Hypot => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x.hypot(y))
        }
        AnalyticBinaryOp::Xlogy => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| {
            if x == S::zero() {
                S::zero()
            } else {
                x * Float::ln(y)
            }
        }),
    }
}

fn execute_analytic_binary_complex<S: ComplexCpuScalarValue>(
    alpha: S,
    lhs: &strided_view::StridedView<S>,
    rhs: &strided_view::StridedView<S>,
    beta: S,
    output: &mut strided_view::StridedViewMut<S>,
    op: AnalyticBinaryOp,
) -> Result<()> {
    match op {
        AnalyticBinaryOp::Pow => {
            execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| x.pow_complex(y))
        }
        AnalyticBinaryOp::Xlogy => execute_binary_map(alpha, lhs, rhs, beta, output, |x, y| {
            if x == S::zero() {
                S::zero()
            } else {
                x * ComplexFloat::ln(y)
            }
        }),
        _ => Err(Error::InvalidArgument(format!(
            "analytic binary operation {op:?} requires ordered real scalars"
        ))),
    }
}

fn execute_analytic_unary<T: Scalar + 'static>(
    alpha: T,
    input: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    op: AnalyticUnaryOp,
) -> Result<()> {
    macro_rules! dispatch {
        ($ty:ty) => {{
            let input = unsafe {
                &*(input as *const strided_view::StridedView<T>
                    as *const strided_view::StridedView<$ty>)
            };
            let output = unsafe {
                &mut *(output as *mut strided_view::StridedViewMut<T>
                    as *mut strided_view::StridedViewMut<$ty>)
            };
            let alpha = unsafe { *(&alpha as *const T as *const $ty) };
            let beta = unsafe { *(&beta as *const T as *const $ty) };
            return execute_analytic_unary_typed(alpha, input, beta, output, op);
        }};
    }

    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<f64>() {
        dispatch!(f64);
    }
    if tid == TypeId::of::<f32>() {
        dispatch!(f32);
    }
    if tid == TypeId::of::<Complex64>() {
        dispatch!(Complex64);
    }
    if tid == TypeId::of::<Complex32>() {
        dispatch!(Complex32);
    }

    Err(Error::InvalidArgument(format!(
        "analytic unary operation {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

fn execute_analytic_binary<T: Scalar + 'static>(
    alpha: T,
    lhs: &strided_view::StridedView<T>,
    rhs: &strided_view::StridedView<T>,
    beta: T,
    output: &mut strided_view::StridedViewMut<T>,
    op: AnalyticBinaryOp,
) -> Result<()> {
    macro_rules! dispatch_real {
        ($ty:ty) => {{
            let lhs = unsafe {
                &*(lhs as *const strided_view::StridedView<T>
                    as *const strided_view::StridedView<$ty>)
            };
            let rhs = unsafe {
                &*(rhs as *const strided_view::StridedView<T>
                    as *const strided_view::StridedView<$ty>)
            };
            let output = unsafe {
                &mut *(output as *mut strided_view::StridedViewMut<T>
                    as *mut strided_view::StridedViewMut<$ty>)
            };
            let alpha = unsafe { *(&alpha as *const T as *const $ty) };
            let beta = unsafe { *(&beta as *const T as *const $ty) };
            return execute_analytic_binary_real(alpha, lhs, rhs, beta, output, op);
        }};
    }
    macro_rules! dispatch_complex {
        ($ty:ty) => {{
            let lhs = unsafe {
                &*(lhs as *const strided_view::StridedView<T>
                    as *const strided_view::StridedView<$ty>)
            };
            let rhs = unsafe {
                &*(rhs as *const strided_view::StridedView<T>
                    as *const strided_view::StridedView<$ty>)
            };
            let output = unsafe {
                &mut *(output as *mut strided_view::StridedViewMut<T>
                    as *mut strided_view::StridedViewMut<$ty>)
            };
            let alpha = unsafe { *(&alpha as *const T as *const $ty) };
            let beta = unsafe { *(&beta as *const T as *const $ty) };
            return execute_analytic_binary_complex(alpha, lhs, rhs, beta, output, op);
        }};
    }

    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<f64>() {
        dispatch_real!(f64);
    }
    if tid == TypeId::of::<f32>() {
        dispatch_real!(f32);
    }
    if tid == TypeId::of::<Complex64>() {
        dispatch_complex!(Complex64);
    }
    if tid == TypeId::of::<Complex32>() {
        dispatch_complex!(Complex32);
    }

    Err(Error::InvalidArgument(format!(
        "analytic binary operation {op:?} is not supported for {}",
        std::any::type_name::<T>()
    )))
}

impl<S: Scalar + 'static> TensorAnalyticPrims<Standard<S>> for CpuBackend {
    type Plan = CpuAnalyticPlan;
    type Context = CpuContext;

    fn plan(
        _ctx: &mut Self::Context,
        desc: &AnalyticPrimsDescriptor,
        shapes: &[&[usize]],
    ) -> Result<Self::Plan> {
        match desc {
            AnalyticPrimsDescriptor::PointwiseUnary { op } => {
                validate_pointwise_shapes(shapes, 1, "AnalyticPointwiseUnary")?;
                if !supports_analytic_unary::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "analytic unary operation {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                Ok(CpuAnalyticPlan::PointwiseUnary { op: *op })
            }
            AnalyticPrimsDescriptor::PointwiseBinary { op } => {
                validate_pointwise_shapes(shapes, 2, "AnalyticPointwiseBinary")?;
                if !supports_analytic_binary::<S>(*op) {
                    return Err(Error::InvalidArgument(format!(
                        "analytic binary operation {op:?} is not supported on CpuBackend for {}",
                        std::any::type_name::<S>()
                    )));
                }
                Ok(CpuAnalyticPlan::PointwiseBinary { op: *op })
            }
            AnalyticPrimsDescriptor::Reduction { op, .. } => Err(Error::InvalidArgument(format!(
                "analytic reduction {op:?} is not implemented in phase 1"
            ))),
        }
    }

    fn execute(
        _ctx: &mut Self::Context,
        plan: &Self::Plan,
        alpha: S,
        inputs: &[&Tensor<S>],
        beta: S,
        output: &mut Tensor<S>,
    ) -> Result<()> {
        let views: Vec<_> = inputs
            .iter()
            .map(|tensor| tensor_to_view(tensor))
            .collect::<Result<_>>()?;
        let view_refs: Vec<_> = views.iter().collect();
        let mut out_view = tensor_to_view_mut(output)?;

        match plan {
            CpuAnalyticPlan::PointwiseUnary { op } => {
                validate_execute_inputs(inputs, 1, "AnalyticPointwiseUnary")?;
                execute_analytic_unary(alpha, view_refs[0], beta, &mut out_view, *op)
            }
            CpuAnalyticPlan::PointwiseBinary { op } => {
                validate_execute_inputs(inputs, 2, "AnalyticPointwiseBinary")?;
                execute_analytic_binary(alpha, view_refs[0], view_refs[1], beta, &mut out_view, *op)
            }
        }
    }

    fn has_analytic_support(desc: AnalyticPrimsDescriptor) -> bool {
        match desc {
            AnalyticPrimsDescriptor::PointwiseUnary { op } => supports_analytic_unary::<S>(op),
            AnalyticPrimsDescriptor::PointwiseBinary { op } => supports_analytic_binary::<S>(op),
            AnalyticPrimsDescriptor::Reduction { .. } => false,
        }
    }
}
