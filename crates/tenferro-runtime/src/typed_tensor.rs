//! Typed tensor operation extension traits.
//!
//! Operation families that are no longer part of core, including einsum, live
//! in their extension crates.

use num_complex::Complex64;
use tenferro_ops::broadcast::{
    broadcast_input_plan, broadcast_shape, broadcast_shapes, BroadcastError,
};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{
    ActivationOp, BackendSession, CompareDir, DotGeneralConfig, Error, Result, Tensor, TensorRead,
    TensorScalar, ValidationError,
};

use crate::composite;
use crate::composite::session::{run_session_composite, typed_borrowed};
use crate::{TypedTensorMaskSessionOpsExt, TypedTensorSessionOpsExt};
use tenferro_tensor::TypedTensor;

impl<T: TensorScalar> TypedTensorSessionOpsExt<T> for TypedTensor<T> {
    fn add(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.add_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("add", out)
    }

    fn mul(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.mul_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("mul", out)
    }

    fn exp(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.exp_read(T::tensor_read(self))?;
        into_typed_result("exp", out)
    }

    fn reduce_sum(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let axes = crate::tensor::all_axes_if_none(self.shape().len(), axes);
        let out = session.reduce_sum_read(T::tensor_read(self), &axes)?;
        into_typed_result("reduce_sum", out)
    }

    fn sub(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.sub_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("sub", out)
    }

    fn div(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.div_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("div", out)
    }

    fn rem(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.rem_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("rem", out)
    }

    fn pow(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.pow_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("pow", out)
    }

    fn maximum(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.maximum_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("maximum", out)
    }

    fn minimum(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.minimum_read(lhs.tensor_read(), rhs.tensor_read())?;
        into_typed_result("minimum", out)
    }

    fn neg(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.neg_read(T::tensor_read(self))?;
        into_typed_result("neg", out)
    }

    fn abs(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T::Real>> {
        let out = session.abs_read(T::tensor_read(self))?;
        into_typed_result::<T::Real>("abs", out)
    }

    fn sign(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.sign_read(T::tensor_read(self))?;
        into_typed_result("sign", out)
    }

    fn conj(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.conj_read(T::tensor_read(self))?;
        into_typed_result("conj", out)
    }

    fn log(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.log_read(T::tensor_read(self))?;
        into_typed_result("log", out)
    }

    fn expm1(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.expm1_read(T::tensor_read(self))?;
        into_typed_result("expm1", out)
    }

    fn log1p(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.log1p_read(T::tensor_read(self))?;
        into_typed_result("log1p", out)
    }

    fn erf(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.erf_read(T::tensor_read(self))?;
        into_typed_result("erf", out)
    }

    fn sin(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.sin_read(T::tensor_read(self))?;
        into_typed_result("sin", out)
    }

    fn cos(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.cos_read(T::tensor_read(self))?;
        into_typed_result("cos", out)
    }

    fn tanh(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.tanh_read(T::tensor_read(self))?;
        into_typed_result("tanh", out)
    }

    fn sqrt(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.sqrt_read(T::tensor_read(self))?;
        into_typed_result("sqrt", out)
    }

    fn rsqrt(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.rsqrt_read(T::tensor_read(self))?;
        into_typed_result("rsqrt", out)
    }

    fn compare(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<bool>> {
        let (lhs, rhs) = broadcast_binary_in_read(self, rhs, session)?;
        let out = session.compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)?;
        into_typed_result("compare", out)
    }

    fn clamp(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let (input, lower, upper) = broadcast_ternary_in_read(self, lower, upper, session)?;
        let out = session.clamp_read(
            input.tensor_read(),
            lower.tensor_read(),
            upper.tensor_read(),
        )?;
        into_typed_result("clamp", out)
    }

    fn matmul(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let config = matmul_config_for_shapes("matmul", self.shape(), rhs.shape())?;
        let out = session.dot_general_read(T::tensor_read(self), T::tensor_read(rhs), &config)?;
        into_typed_result("matmul", out)
    }

    fn reshape(&self, shape: &[usize], session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.reshape_read(T::tensor_read(self), shape)?;
        into_typed_result("reshape", out)
    }

    fn transpose(
        &self,
        perm: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.transpose_read(T::tensor_read(self), perm)?;
        into_typed_result("transpose", out)
    }

    fn broadcast_in_dim(
        &self,
        shape: &[usize],
        dims: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.broadcast_in_dim_read(T::tensor_read(self), shape, dims)?;
        into_typed_result("broadcast_in_dim", out)
    }

    fn reduce_max(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let axes = crate::tensor::all_axes_if_none(self.shape().len(), axes);
        let out = session.reduce_max_read(T::tensor_read(self), &axes)?;
        into_typed_result("reduce_max", out)
    }

    fn reduce_min(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let axes = crate::tensor::all_axes_if_none(self.shape().len(), axes);
        let out = session.reduce_min_read(T::tensor_read(self), &axes)?;
        into_typed_result("reduce_min", out)
    }

    fn reduce_prod(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let axes = crate::tensor::all_axes_if_none(self.shape().len(), axes);
        let out = session.reduce_prod_read(T::tensor_read(self), &axes)?;
        into_typed_result("reduce_prod", out)
    }

    fn reduce_sum_squares(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let axes = crate::tensor::all_axes_if_none(self.shape().len(), axes);
        let out = session.reduce_sum_squares_read(T::tensor_read(self), &axes)?;
        into_typed_result("reduce_sum_squares", out)
    }

    fn dot_general(
        &self,
        rhs: &TypedTensor<T>,
        config: DotGeneralConfig,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.dot_general_read(T::tensor_read(self), T::tensor_read(rhs), &config)?;
        into_typed_result("dot_general", out)
    }

    fn dot_general_with_conj(
        &self,
        rhs: &TypedTensor<T>,
        config: DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.dot_general_with_conj_read(
            T::tensor_read(self),
            T::tensor_read(rhs),
            &config,
            lhs_conj,
            rhs_conj,
        )?;
        into_typed_result("dot_general_with_conj", out)
    }

    fn scale_real(&self, factor: f64, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let scalar = crate::scale::real_scale_scalar(T::dtype(), factor)?;
        let scalar = session.upload_host_tensor(TensorRead::from_tensor(&scalar))?;
        let scalar = into_typed_result::<T>("scale_real", scalar)?;
        TypedTensorSessionOpsExt::mul(self, &scalar, session)
    }

    fn scale_complex(
        &self,
        factor: Complex64,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let scalar = crate::scale::complex_scale_scalar(T::dtype(), factor)?;
        let scalar = session.upload_host_tensor(TensorRead::from_tensor(&scalar))?;
        let scalar = into_typed_result::<T>("scale_complex", scalar)?;
        TypedTensorSessionOpsExt::mul(self, &scalar, session)
    }

    fn sigmoid(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Sigmoid, T::tensor_read(self))?
        {
            return into_typed_result("sigmoid", out);
        }
        let out = run_session_composite(session, |ops| {
            composite::sigmoid(ops, &typed_borrowed(self))
        })?;
        into_typed_result("sigmoid", out)
    }

    fn silu(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Silu, T::tensor_read(self))?
        {
            return into_typed_result("silu", out);
        }
        let out =
            run_session_composite(session, |ops| composite::silu(ops, &typed_borrowed(self)))?;
        into_typed_result("silu", out)
    }

    fn softplus(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Softplus, T::tensor_read(self))?
        {
            return into_typed_result("softplus", out);
        }
        let out = run_session_composite(session, |ops| {
            composite::softplus(ops, &typed_borrowed(self))
        })?;
        into_typed_result("softplus", out)
    }

    fn gelu(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Gelu, T::tensor_read(self))?
        {
            return into_typed_result("gelu", out);
        }
        let out =
            run_session_composite(session, |ops| composite::gelu(ops, &typed_borrowed(self)))?;
        into_typed_result("gelu", out)
    }

    fn gelu_tanh(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::GeluTanh, T::tensor_read(self))?
        {
            return into_typed_result("gelu_tanh", out);
        }
        let out = run_session_composite(session, |ops| {
            composite::gelu_tanh(ops, &typed_borrowed(self))
        })?;
        into_typed_result("gelu_tanh", out)
    }

    fn reduce_mean(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = run_session_composite(session, |ops| {
            composite::reduce_mean(ops, &typed_borrowed(self), axes)
        })?;
        into_typed_result("reduce_mean", out)
    }

    fn softmax(&self, axis: usize, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = run_session_composite(session, |ops| {
            composite::softmax(ops, &typed_borrowed(self), axis)
        })?;
        into_typed_result("softmax", out)
    }

    fn log_softmax(&self, axis: usize, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = run_session_composite(session, |ops| {
            composite::log_softmax(ops, &typed_borrowed(self), axis)
        })?;
        into_typed_result("log_softmax", out)
    }

    fn masked_softmax(
        &self,
        mask: &TypedTensor<bool>,
        axis: usize,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let mask = typed_borrowed(mask);
        let out = run_session_composite(session, |ops| {
            composite::masked_softmax(ops, &typed_borrowed(self), &mask, axis)
        })?;
        into_typed_result("masked_softmax", out)
    }

    fn masked_log_softmax(
        &self,
        mask: &TypedTensor<bool>,
        axis: usize,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let mask = typed_borrowed(mask);
        let out = run_session_composite(session, |ops| {
            composite::masked_log_softmax(ops, &typed_borrowed(self), &mask, axis)
        })?;
        into_typed_result("masked_log_softmax", out)
    }

    fn layer_norm(
        &self,
        axis: usize,
        weight: Option<&TypedTensor<T>>,
        bias: Option<&TypedTensor<T>>,
        eps: f64,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let weight = weight.map(typed_borrowed);
        let bias = bias.map(typed_borrowed);
        let out = run_session_composite(session, |ops| {
            composite::layer_norm(
                ops,
                &typed_borrowed(self),
                axis,
                weight.as_ref(),
                bias.as_ref(),
                eps,
            )
        })?;
        into_typed_result("layer_norm", out)
    }

    fn rms_norm(
        &self,
        axis: usize,
        weight: Option<&TypedTensor<T>>,
        bias: Option<&TypedTensor<T>>,
        eps: f64,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let weight = weight.map(typed_borrowed);
        let bias = bias.map(typed_borrowed);
        let out = run_session_composite(session, |ops| {
            composite::rms_norm(
                ops,
                &typed_borrowed(self),
                axis,
                weight.as_ref(),
                bias.as_ref(),
                eps,
            )
        })?;
        into_typed_result("rms_norm", out)
    }
}

impl TypedTensorMaskSessionOpsExt for TypedTensor<bool> {
    fn where_select<U: TensorScalar>(
        &self,
        on_true: &TypedTensor<U>,
        on_false: &TypedTensor<U>,
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<U>> {
        let (condition, on_true, on_false) =
            broadcast_ternary_in_read(self, on_true, on_false, session)?;
        let out = session.select_read(
            condition.tensor_read(),
            on_true.tensor_read(),
            on_false.tensor_read(),
        )?;
        into_typed_result("where_select", out)
    }
}

// INVARIANT: this private adapter keeps borrowed reads borrowed and owns only
// the explicit fallback tensor; it is never exposed or cloned.
#[allow(clippy::large_enum_variant)]
pub(crate) enum ReadInput<'a> {
    Borrowed(TensorRead<'a>),
    Owned(Tensor),
}

impl ReadInput<'_> {
    pub(crate) fn tensor_read(&self) -> TensorRead<'_> {
        match self {
            Self::Borrowed(read) => read.clone(),
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
        }
    }
}

pub(crate) fn broadcast_to_in_read<'a>(
    input: TensorRead<'a>,
    target_shape: &[usize],
    session: &mut dyn BackendSession,
) -> Result<ReadInput<'a>> {
    if input.shape() == target_shape {
        return Ok(ReadInput::Borrowed(input));
    }

    let plan = broadcast_input_plan(input.shape(), target_shape).map_err(broadcast_error)?;
    let source = if plan.source_shape == input.shape() {
        ReadInput::Borrowed(input)
    } else {
        let reshaped = session.reshape_read(input, &plan.source_shape)?;
        ReadInput::Owned(reshaped)
    };
    let out = session.broadcast_in_dim_read(source.tensor_read(), target_shape, &plan.dims)?;
    Ok(ReadInput::Owned(out))
}

fn broadcast_binary_in_read<'a, T: TensorScalar>(
    lhs: &'a TypedTensor<T>,
    rhs: &'a TypedTensor<T>,
    session: &mut dyn BackendSession,
) -> Result<(ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(T::tensor_read(lhs), &shape, session)?,
        broadcast_to_in_read(T::tensor_read(rhs), &shape, session)?,
    ))
}

fn broadcast_ternary_in_read<'a, C: TensorScalar, T: TensorScalar>(
    first: &'a TypedTensor<C>,
    second: &'a TypedTensor<T>,
    third: &'a TypedTensor<T>,
    session: &mut dyn BackendSession,
) -> Result<(ReadInput<'a>, ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(C::tensor_read(first), &shape, session)?,
        broadcast_to_in_read(T::tensor_read(second), &shape, session)?,
        broadcast_to_in_read(T::tensor_read(third), &shape, session)?,
    ))
}

pub(crate) fn broadcast_error(err: BroadcastError) -> Error {
    match err {
        BroadcastError::IncompatibleBinary { lhs, rhs } => {
            Error::shape_mismatch("broadcast", lhs, rhs)
        }
        BroadcastError::IncompatibleInput { input, output } => {
            Error::shape_mismatch("broadcast", input, output)
        }
        BroadcastError::RankTooLarge { input, output } => {
            Error::rank_mismatch("broadcast", output.len(), input.len())
        }
    }
}

fn into_typed_result<T: TensorScalar>(op: &'static str, tensor: Tensor) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::into_typed(tensor).map_err(|_| {
        Error::validation(
            op,
            ValidationError::DTypeMismatch {
                expected: T::dtype(),
                actual,
            },
        )
    })
}
