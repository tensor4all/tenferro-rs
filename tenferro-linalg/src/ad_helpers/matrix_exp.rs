use super::*;
use num_complex::{Complex32, Complex64};
use tenferro_algebra::Standard;
use tenferro_prims::{
    AnalyticUnaryOp, ComplexRealUnaryOp, ScalarBinaryOp, ScalarReductionOp, ScalarUnaryOp,
    TensorAnalyticPrims, TensorComplexRealContextFor, TensorScalarContextFor,
    TensorSemiringContextFor,
};

/// Padé[13/13] coefficients b[0]..b[13] (integer values as f64).
pub(crate) const PADE13_COEFFS: [f64; 14] = [
    64764752532480000.0,
    32382376266240000.0,
    7771770303897600.0,
    1187353796428800.0,
    129060195264000.0,
    10559470521600.0,
    670442572800.0,
    33522128640.0,
    1323241920.0,
    40840800.0,
    960960.0,
    16380.0,
    182.0,
    1.0,
];

/// Theta threshold for order-13 Padé (f64).
pub(crate) const THETA_13: f64 = 5.371920351148152;

#[doc(hidden)]
/// Hidden dispatch trait for `matrix_exp` absolute-value preparation.
///
/// # Examples
///
/// ```ignore
/// fn require_matrix_exp_abs<T, C>()
/// where
///     T: tenferro_linalg::MatrixExpAbsTensor<C>,
/// {
/// }
/// ```
pub trait MatrixExpAbsTensor<C>: KernelLinalgScalar {
    fn matrix_exp_abs_tensor(ctx: &mut C, input: &Tensor<Self>) -> Result<Tensor<Self::Real>>;
}

impl<C> MatrixExpAbsTensor<C> for f32
where
    C: TensorScalarContextFor<Standard<f32>>,
{
    fn matrix_exp_abs_tensor(ctx: &mut C, input: &Tensor<Self>) -> Result<Tensor<Self::Real>> {
        prims_bridge::scalar_unary_same_shape(ctx, input, ScalarUnaryOp::Abs)
    }
}

impl<C> MatrixExpAbsTensor<C> for f64
where
    C: TensorScalarContextFor<Standard<f64>>,
{
    fn matrix_exp_abs_tensor(ctx: &mut C, input: &Tensor<Self>) -> Result<Tensor<Self::Real>> {
        prims_bridge::scalar_unary_same_shape(ctx, input, ScalarUnaryOp::Abs)
    }
}

impl<C> MatrixExpAbsTensor<C> for Complex32
where
    C: TensorComplexRealContextFor<Complex32>,
    C::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<Complex32, Context = C, Real = f32>,
{
    fn matrix_exp_abs_tensor(ctx: &mut C, input: &Tensor<Self>) -> Result<Tensor<Self::Real>> {
        prims_bridge::complex_real_unary_same_shape(ctx, input, ComplexRealUnaryOp::Abs)
    }
}

impl<C> MatrixExpAbsTensor<C> for Complex64
where
    C: TensorComplexRealContextFor<Complex64>,
    C::ComplexRealBackend:
        tenferro_prims::TensorComplexRealPrims<Complex64, Context = C, Real = f64>,
{
    fn matrix_exp_abs_tensor(ctx: &mut C, input: &Tensor<Self>) -> Result<Tensor<Self::Real>> {
        prims_bridge::complex_real_unary_same_shape(ctx, input, ComplexRealUnaryOp::Abs)
    }
}

/// Compute the matrix 1-norm for each batch using tensor-native reductions.
pub(crate) fn matrix_exp_batch_1_norms_tensor<T, C>(
    ctx: &mut C,
    tensor: &Tensor<T>,
) -> Result<Tensor<T::Real>>
where
    T: MatrixExpAbsTensor<C>,
    C: TensorScalarContextFor<Standard<T::Real>>,
{
    let abs_tensor = T::matrix_exp_abs_tensor(ctx, tensor)?;
    let kept_axes: Vec<usize> = (1..abs_tensor.ndim()).collect();
    let col_sums = prims_bridge::scalar_reduce_keep_axes(
        ctx,
        &abs_tensor,
        &kept_axes,
        ScalarReductionOp::Sum,
    )?;
    let batch_axes: Vec<usize> = (1..col_sums.ndim()).collect();
    prims_bridge::scalar_reduce_keep_axes(ctx, &col_sums, &batch_axes, ScalarReductionOp::Max)
}

/// Compute per-batch Padé scaling counts `s = ceil(log2(max(norm/theta, 1)))`.
pub(crate) fn matrix_exp_batch_squaring_counts_tensor<R, C>(
    ctx: &mut C,
    batch_norms: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: KernelLinalgScalar<Real = R> + num_traits::Float,
    C: TensorScalarContextFor<Standard<R>>,
    <C as TensorScalarContextFor<Standard<R>>>::ScalarBackend:
        TensorAnalyticPrims<Standard<R>, Context = C>,
{
    let theta = prims_bridge::full_like_constant(
        scalar_from::<R>(THETA_13)?,
        batch_norms.dims(),
        batch_norms.logical_memory_space(),
    )?;
    let one = prims_bridge::full_like_constant(
        scalar_from::<R>(1.0)?,
        batch_norms.dims(),
        batch_norms.logical_memory_space(),
    )?;
    let log2_e = prims_bridge::full_like_constant(
        scalar_from::<R>(std::f64::consts::LOG2_E)?,
        batch_norms.dims(),
        batch_norms.logical_memory_space(),
    )?;

    let ratio =
        prims_bridge::scalar_binary_same_shape(ctx, batch_norms, &theta, ScalarBinaryOp::Div)?;
    let ratio = prims_bridge::scalar_binary_same_shape(ctx, &ratio, &one, ScalarBinaryOp::Maximum)?;
    let log_ratio = prims_bridge::analytic_unary_same_shape(ctx, &ratio, AnalyticUnaryOp::Log)?;
    let log2_ratio =
        prims_bridge::scalar_binary_same_shape(ctx, &log_ratio, &log2_e, ScalarBinaryOp::Mul)?;
    prims_bridge::analytic_unary_same_shape(ctx, &log2_ratio, AnalyticUnaryOp::Ceil)
}

/// Blend two same-shape tensors with a real-valued numeric mask.
///
/// `mask == 1` selects `on_true`, and `mask == 0` selects `on_false`.
pub(crate) fn blend_tensor_by_real_mask_same_shape<T, C>(
    ctx: &mut C,
    on_true: &Tensor<T>,
    on_false: &Tensor<T>,
    mask: &Tensor<T::Real>,
) -> Result<Tensor<T>>
where
    T: LinalgScalar + prims_bridge::ScaleTensorByRealSameShape<C>,
    C: TensorScalarContextFor<Standard<T>> + TensorScalarContextFor<Standard<T::Real>>,
{
    let one = prims_bridge::full_like_constant(
        scalar_from::<T::Real>(1.0)?,
        mask.dims(),
        mask.logical_memory_space(),
    )?;
    let inv_mask = prims_bridge::scalar_binary_same_shape(ctx, &one, mask, ScalarBinaryOp::Sub)?;
    let true_part =
        <T as prims_bridge::ScaleTensorByRealSameShape<C>>::scale_tensor_by_real_same_shape(
            ctx, on_true, mask,
        )?;
    let false_part =
        <T as prims_bridge::ScaleTensorByRealSameShape<C>>::scale_tensor_by_real_same_shape(
            ctx, on_false, &inv_mask,
        )?;
    prims_bridge::scalar_binary_same_shape(ctx, &true_part, &false_part, ScalarBinaryOp::Add)
}

fn batched_identity<T: KernelLinalgScalar>(
    n: usize,
    batch_dims: &[usize],
    logical_memory_space: tenferro_device::LogicalMemorySpace,
) -> Result<Tensor<T>> {
    let mut reshape_dims = vec![n, n];
    reshape_dims.extend(std::iter::repeat(1).take(batch_dims.len()));
    let eye = crate::prims_bridge::identity_matrix(n, logical_memory_space)?;
    let eye = eye.reshape(&reshape_dims)?;
    eye.broadcast(&output_dims(&[n, n], batch_dims))
}

fn tensor_scale_by_constant<T, C>(ctx: &mut C, tensor: &Tensor<T>, coeff: T) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    let coeff =
        prims_bridge::full_like_constant(coeff, tensor.dims(), tensor.logical_memory_space())?;
    prims_bridge::scalar_binary_same_shape(ctx, tensor, &coeff, ScalarBinaryOp::Mul)
}

fn tensor_add_same_shape<T, C>(ctx: &mut C, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    prims_bridge::scalar_binary_same_shape(ctx, lhs, rhs, ScalarBinaryOp::Add)
}

fn tensor_sub_same_shape<T, C>(ctx: &mut C, lhs: &Tensor<T>, rhs: &Tensor<T>) -> Result<Tensor<T>>
where
    T: LinalgScalar,
    C: TensorScalarContextFor<Standard<T>>,
{
    prims_bridge::scalar_binary_same_shape(ctx, lhs, rhs, ScalarBinaryOp::Sub)
}

/// Evaluate Padé[13/13] on a scaled matrix using tensor-native operators.
pub(crate) fn matrix_exp_tensor_native<T, C>(
    ctx: &mut C,
    scaled_input: &Tensor<T>,
    n: usize,
    batch_dims: &[usize],
    s: usize,
) -> Result<Tensor<T>>
where
    T: KernelLinalgScalar + crate::prims_bridge::ScaleTensorByRealSameShape<C> + LinalgScalar,
    C: backend::TensorLinalgContextFor<T>
        + TensorScalarContextFor<Standard<T>>
        + TensorSemiringContextFor<Standard<T>>,
    C::Backend: 'static,
{
    if n == 0 {
        return Ok(Tensor::zeros(
            &output_dims(&[n, n], batch_dims),
            scaled_input.logical_memory_space(),
            MemoryOrder::ColumnMajor,
        )?);
    }

    let coeffs = PADE13_COEFFS
        .iter()
        .map(|&c| scalar_from::<T>(c))
        .collect::<Result<Vec<T>>>()?;

    let eye = batched_identity::<T>(n, batch_dims, scaled_input.logical_memory_space())?;
    let a2 =
        prims_bridge::batched_gemm_with_semiring_tensors(ctx, scaled_input, scaled_input, n, n, n)?;
    let a4 = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a2, &a2, n, n, n)?;
    let a6 = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a4, &a2, n, n, n)?;

    let u_a6 = tensor_scale_by_constant(ctx, &a6, coeffs[13])?;
    let u_a4 = tensor_scale_by_constant(ctx, &a4, coeffs[11])?;
    let u_a2 = tensor_scale_by_constant(ctx, &a2, coeffs[9])?;
    let mut inner_u = tensor_add_same_shape(ctx, &u_a6, &u_a4)?;
    inner_u = tensor_add_same_shape(ctx, &inner_u, &u_a2)?;
    let a6_inner_u = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a6, &inner_u, n, n, n)?;
    let u_a6 = tensor_scale_by_constant(ctx, &a6, coeffs[7])?;
    let u_a4 = tensor_scale_by_constant(ctx, &a4, coeffs[5])?;
    let u_a2 = tensor_scale_by_constant(ctx, &a2, coeffs[3])?;
    let u_eye = tensor_scale_by_constant(ctx, &eye, coeffs[1])?;
    let mut u_inner = tensor_add_same_shape(ctx, &a6_inner_u, &u_a6)?;
    u_inner = tensor_add_same_shape(ctx, &u_inner, &u_a4)?;
    u_inner = tensor_add_same_shape(ctx, &u_inner, &u_a2)?;
    u_inner = tensor_add_same_shape(ctx, &u_inner, &u_eye)?;
    let u = prims_bridge::batched_gemm_with_semiring_tensors(ctx, scaled_input, &u_inner, n, n, n)?;

    let v_a6 = tensor_scale_by_constant(ctx, &a6, coeffs[12])?;
    let v_a4 = tensor_scale_by_constant(ctx, &a4, coeffs[10])?;
    let v_a2 = tensor_scale_by_constant(ctx, &a2, coeffs[8])?;
    let mut inner_v = tensor_add_same_shape(ctx, &v_a6, &v_a4)?;
    inner_v = tensor_add_same_shape(ctx, &inner_v, &v_a2)?;
    let a6_inner_v = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &a6, &inner_v, n, n, n)?;
    let v_a6 = tensor_scale_by_constant(ctx, &a6, coeffs[6])?;
    let v_a4 = tensor_scale_by_constant(ctx, &a4, coeffs[4])?;
    let v_a2 = tensor_scale_by_constant(ctx, &a2, coeffs[2])?;
    let v_eye = tensor_scale_by_constant(ctx, &eye, coeffs[0])?;
    let mut v = tensor_add_same_shape(ctx, &a6_inner_v, &v_a6)?;
    v = tensor_add_same_shape(ctx, &v, &v_a4)?;
    v = tensor_add_same_shape(ctx, &v, &v_a2)?;
    v = tensor_add_same_shape(ctx, &v, &v_eye)?;

    let lhs = tensor_sub_same_shape(ctx, &v, &u)?;
    let rhs = tensor_add_same_shape(ctx, &v, &u)?;
    let mut result = crate::solve(ctx, &lhs, &rhs)?;
    for _ in 0..s {
        result = prims_bridge::batched_gemm_with_semiring_tensors(ctx, &result, &result, n, n, n)?;
    }

    Ok(result)
}
