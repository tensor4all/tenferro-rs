use chainrules_scalarops::ScalarAd;
use num_traits::NumCast;
use tenferro_algebra::{HasAlgebra, Scalar, Standard};
use tenferro_prims::{
    AnalyticUnaryOp, CpuBackend, CudaBackend, RocmBackend, ScalarBinaryOp, ScalarReductionOp,
    ScalarUnaryOp, TensorAnalyticPrims, TensorPrims, TensorScalarPrims,
};
use tenferro_tensor::Tensor;

use crate::{reverse_tape, AdTensor, AdValue, Error, Result};

use super::runtime::{
    broadcast_scalar_like, collect_reverse_input_specs, compress_pullback_like, has_any_tangent,
    has_forward, scalar_from_rank0_tensor, wrap_dense_ad_output,
};
use super::scalar_runtime::{
    analytic_unary_primal, dense_input_snapshot_in_runtime, scalar_binary_primal,
    scalar_full_reduction_primal, scalar_unary_primal,
};

fn scalar_from_usize<T: ScalarAd>(value: usize) -> Result<T> {
    let Some(real) = <T::Real as NumCast>::from(value) else {
        return Err(Error::InvalidAdTensor {
            message: format!(
                "cannot represent reduction size {value} for scalar type {}",
                std::any::type_name::<T>()
            ),
        });
    };
    Ok(T::from_real(real))
}

fn run_scalar_unary_ad<T, FPrimal, FTangent, FPullback>(
    op_name: &'static str,
    _pullback_op_name: &'static str,
    input: &AdTensor<T>,
    primal_fn: FPrimal,
    tangent_fn: FTangent,
    pullback_fn: FPullback,
) -> Result<AdTensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    FPrimal: Fn(&Tensor<T>) -> Result<Tensor<T>>,
    FTangent: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FPullback: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>> + 'static,
{
    let operands = [input];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
    let (input_primal, input_tangent) =
        dense_input_snapshot_in_runtime(op_name, input, needs_tangent)?;
    let primal = primal_fn(&input_primal)?;
    let tangent = if needs_tangent {
        let tangent = input_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized tangent"),
        })?;
        Some(tangent_fn(&input_primal, &primal, &tangent)?)
    } else {
        None
    };

    let out = wrap_dense_ad_output(op_name, &operands, primal.clone(), tangent, 0)?;

    if let AdValue::Reverse { node, tape, .. } = out.as_value() {
        let input_spec = collect_reverse_input_specs(&operands)
            .into_iter()
            .next()
            .flatten();
        let output_node = *node;
        let tape_id = *tape;

        reverse_tape::register_rule::<T>(
            tape_id,
            output_node,
            Box::new(move |cotangent| {
                let grad = pullback_fn(&input_primal, &primal, cotangent)?;
                let Some(spec) = &input_spec else {
                    return Ok(Vec::new());
                };
                let grad = compress_pullback_like(op_name, grad, &spec.layout)?;
                Ok(vec![(spec.node, grad)])
            }),
        )?;
    }

    Ok(out)
}

fn run_scalar_binary_ad<T, FPrimal, FTangent, FPullback>(
    op_name: &'static str,
    _pullback_op_name: &'static str,
    lhs: &AdTensor<T>,
    rhs: &AdTensor<T>,
    primal_fn: FPrimal,
    tangent_fn: FTangent,
    pullback_fn: FPullback,
) -> Result<AdTensor<T>>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    FPrimal: Fn(&Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FTangent: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<Tensor<T>>,
    FPullback: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Result<(Tensor<T>, Tensor<T>)>
        + 'static,
{
    let operands = [lhs, rhs];
    let needs_tangent = has_forward(&operands) || has_any_tangent(&operands);
    let (lhs_snapshot, rhs_snapshot) = (
        dense_input_snapshot_in_runtime(op_name, lhs, needs_tangent)?,
        dense_input_snapshot_in_runtime(op_name, rhs, needs_tangent)?,
    );
    let (lhs_primal, lhs_tangent) = lhs_snapshot;
    let (rhs_primal, rhs_tangent) = rhs_snapshot;
    let primal = primal_fn(&lhs_primal, &rhs_primal)?;
    let tangent = if needs_tangent {
        let lhs_tangent = lhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized lhs tangent"),
        })?;
        let rhs_tangent = rhs_tangent.ok_or_else(|| Error::InvalidAdTensor {
            message: format!("{op_name} missing materialized rhs tangent"),
        })?;
        Some(tangent_fn(
            &lhs_primal,
            &rhs_primal,
            &primal,
            &lhs_tangent,
            &rhs_tangent,
        )?)
    } else {
        None
    };

    let out = wrap_dense_ad_output(op_name, &operands, primal.clone(), tangent, 0)?;

    if let AdValue::Reverse { node, tape, .. } = out.as_value() {
        let reverse_specs = collect_reverse_input_specs(&operands);
        let output_node = *node;
        let tape_id = *tape;

        reverse_tape::register_rule::<T>(
            tape_id,
            output_node,
            Box::new(move |cotangent| {
                let (grad_lhs, grad_rhs) =
                    pullback_fn(&lhs_primal, &rhs_primal, &primal, cotangent)?;
                let mut input_grads = Vec::new();
                if let Some(spec) = &reverse_specs[0] {
                    let grad = compress_pullback_like(op_name, grad_lhs, &spec.layout)?;
                    input_grads.push((spec.node, grad));
                }
                if let Some(spec) = &reverse_specs[1] {
                    let grad = compress_pullback_like(op_name, grad_rhs, &spec.layout)?;
                    input_grads.push((spec.node, grad));
                }
                Ok(input_grads)
            }),
        )?;
    }

    Ok(out)
}

/// Builder for analytic `exp` on AD tensors.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::exp_ad(&x).run()?;
/// ```
pub struct ExpAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> ExpAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    /// Executes AD `exp`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = builder.run()?;
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_scalar_unary_ad(
            "exp_ad",
            "exp_ad_pullback",
            self.tensor,
            |input| analytic_unary_primal("exp_ad_primal", AnalyticUnaryOp::Exp, input),
            |_input, primal, tangent| {
                let conj_primal =
                    scalar_unary_primal("exp_ad_tangent_conj", ScalarUnaryOp::Conj, primal)?;
                scalar_binary_primal(
                    "exp_ad_tangent_mul",
                    ScalarBinaryOp::Mul,
                    tangent,
                    &conj_primal,
                )
            },
            |_input, primal, cotangent| {
                let conj_primal =
                    scalar_unary_primal("exp_ad_pullback_conj", ScalarUnaryOp::Conj, primal)?;
                scalar_binary_primal(
                    "exp_ad_pullback_mul",
                    ScalarBinaryOp::Mul,
                    cotangent,
                    &conj_primal,
                )
            },
        )
    }
}

/// Creates a builder for AD `exp`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::exp_ad(&x).run()?;
/// ```
pub fn exp_ad<'a, T>(tensor: &'a AdTensor<T>) -> ExpAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    ExpAdBuilder { tensor }
}

/// Builder for scalar `add` on AD tensors.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::add_ad(&a, &b).run()?;
/// ```
pub struct AddAdBuilder<'a, T: Scalar> {
    lhs: &'a AdTensor<T>,
    rhs: &'a AdTensor<T>,
}

impl<'a, T> AddAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    /// Executes AD `add`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = builder.run()?;
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_scalar_binary_ad(
            "add_ad",
            "add_ad_pullback",
            self.lhs,
            self.rhs,
            |lhs, rhs| scalar_binary_primal("add_ad_primal", ScalarBinaryOp::Add, lhs, rhs),
            |_lhs, _rhs, _primal, lhs_tangent, rhs_tangent| {
                scalar_binary_primal(
                    "add_ad_tangent",
                    ScalarBinaryOp::Add,
                    lhs_tangent,
                    rhs_tangent,
                )
            },
            |_lhs, _rhs, _primal, cotangent| Ok((cotangent.clone(), cotangent.clone())),
        )
    }
}

/// Creates a builder for AD `add`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::add_ad(&a, &b).run()?;
/// ```
pub fn add_ad<'a, T>(lhs: &'a AdTensor<T>, rhs: &'a AdTensor<T>) -> AddAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    AddAdBuilder { lhs, rhs }
}

/// Builder for full `mean` reduction on AD tensors.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::mean_ad(&x).run()?;
/// ```
pub struct MeanAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> MeanAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    /// Executes AD `mean`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let out = builder.run()?;
    /// ```
    pub fn run(self) -> Result<AdTensor<T>> {
        run_scalar_unary_ad(
            "mean_ad",
            "mean_ad_pullback",
            self.tensor,
            |input| scalar_full_reduction_primal("mean_ad_primal", ScalarReductionOp::Mean, input),
            |_input, _primal, tangent| {
                scalar_full_reduction_primal("mean_ad_tangent", ScalarReductionOp::Mean, tangent)
            },
            |input, _primal, cotangent| {
                let scalar = scalar_from_rank0_tensor(cotangent, "mean_ad")?;
                let denom = scalar_from_usize::<T>(input.len())?;
                let payload = broadcast_scalar_like(scalar / denom, input)?;
                Ok(payload)
            },
        )
    }
}

/// Creates a builder for AD `mean`.
///
/// # Examples
///
/// ```ignore
/// let out = tenferro_dyadtensor::mean_ad(&x).run()?;
/// ```
pub fn mean_ad<'a, T>(tensor: &'a AdTensor<T>) -> MeanAdBuilder<'a, T>
where
    T: Scalar + HasAlgebra<Algebra = Standard<T>> + ScalarAd + Copy + 'static,
    CpuBackend: TensorPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CpuBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CpuContext>,
    CudaBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    CudaBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::CudaContext>,
    RocmBackend: TensorScalarPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
    RocmBackend: TensorAnalyticPrims<Standard<T>, Context = tenferro_prims::RocmContext>,
{
    MeanAdBuilder { tensor }
}
