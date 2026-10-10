//! Concrete tensor operation extension trait.
//!
//! `tenferro-tensor` owns storage and backend traits. This runtime crate
//! provides backend-parametric session-explicit operation methods through
//! [`TensorSessionOpsExt`].

use crate::composite;
use crate::composite::session::{borrowed, run_session_composite};
use std::borrow::Cow;

use num_complex::Complex64;
use tenferro_ops::broadcast::{broadcast_error_to_validation, broadcast_shape, broadcast_shapes};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{
    ActivationOp, BackendSession, CompareDir, DType, DotGeneralConfig, Error, GatherConfig,
    PadConfig, Result, ScatterConfig, SliceConfig, TensorRead,
};

use crate::typed_tensor::{broadcast_to_in_read, ReadInput};

use crate::TensorSessionOpsExt;
use tenferro_tensor::Tensor;

impl TensorSessionOpsExt for Tensor {
    fn add(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.add_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn mul(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.mul_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn exp(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.exp_read(TensorRead::from_tensor(self))
    }

    fn reduce_sum(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let axes = all_axes_if_none(self.shape().len(), axes);
        session.reduce_sum_read(TensorRead::from_tensor(self), &axes)
    }

    fn convert(&self, to: DType, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.convert(self, to)
    }

    fn cast(&self, to: DType, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.cast(self, to)
    }

    fn sub(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.sub_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn div(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.div_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn rem(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.rem_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn pow(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.pow_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn maximum(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.maximum_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn minimum(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.minimum_read(lhs.tensor_read(), rhs.tensor_read())
    }

    fn neg(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.neg_read(TensorRead::from_tensor(self))
    }

    fn abs(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.abs_read(TensorRead::from_tensor(self))
    }

    fn sign(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sign_read(TensorRead::from_tensor(self))
    }

    fn conj(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.conj_read(TensorRead::from_tensor(self))
    }

    fn log(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.log_read(TensorRead::from_tensor(self))
    }

    fn expm1(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.expm1_read(TensorRead::from_tensor(self))
    }

    fn log1p(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.log1p_read(TensorRead::from_tensor(self))
    }

    fn erf(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.erf_read(TensorRead::from_tensor(self))
    }

    fn sin(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sin_read(TensorRead::from_tensor(self))
    }

    fn cos(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.cos_read(TensorRead::from_tensor(self))
    }

    fn tanh(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.tanh_read(TensorRead::from_tensor(self))
    }

    fn sqrt(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sqrt_read(TensorRead::from_tensor(self))
    }

    fn rsqrt(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.rsqrt_read(TensorRead::from_tensor(self))
    }

    fn compare(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.compare_read(lhs.tensor_read(), rhs.tensor_read(), &dir)
    }

    fn where_select(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (condition, on_true, on_false) =
            broadcast_ternary_in(self, on_true, on_false, session)?;
        session.select_read(
            condition.tensor_read(),
            on_true.tensor_read(),
            on_false.tensor_read(),
        )
    }

    fn clamp(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (input, lower, upper) = broadcast_ternary_in(self, lower, upper, session)?;
        session.clamp_read(
            input.tensor_read(),
            lower.tensor_read(),
            upper.tensor_read(),
        )
    }

    fn matmul(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let config = matmul_config_for_shapes("matmul", self.shape(), rhs.shape())?;
        session.dot_general_read(
            TensorRead::from_tensor(self),
            TensorRead::from_tensor(rhs),
            &config,
        )
    }

    fn reshape(&self, shape: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.reshape_read(TensorRead::from_tensor(self), shape)
    }

    fn transpose(&self, perm: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.transpose_read(TensorRead::from_tensor(self), perm)
    }

    fn gather(
        &self,
        indices: &Tensor,
        config: GatherConfig,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.gather(self, indices, &config)
    }

    fn scatter(
        &self,
        indices: &Tensor,
        updates: &Tensor,
        config: ScatterConfig,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.scatter(self, indices, updates, &config)
    }

    fn slice(&self, config: SliceConfig, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.slice(self, &config)
    }

    fn dynamic_slice(
        &self,
        starts: &Tensor,
        sizes: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.dynamic_slice(self, starts, sizes)
    }

    fn pad(&self, config: PadConfig, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.pad(self, &config)
    }

    fn concatenate(
        inputs: &[&Tensor],
        axis: usize,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.concatenate(inputs, axis)
    }

    fn reverse(&self, axes: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.reverse(self, axes)
    }

    fn reduce_max(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let axes = all_axes_if_none(self.shape().len(), axes);
        session.reduce_max_read(TensorRead::from_tensor(self), &axes)
    }

    fn reduce_min(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let axes = all_axes_if_none(self.shape().len(), axes);
        session.reduce_min_read(TensorRead::from_tensor(self), &axes)
    }

    fn reduce_prod(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let axes = all_axes_if_none(self.shape().len(), axes);
        session.reduce_prod_read(TensorRead::from_tensor(self), &axes)
    }

    fn reduce_sum_squares(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let axes = all_axes_if_none(self.shape().len(), axes);
        session.reduce_sum_squares_read(TensorRead::from_tensor(self), &axes)
    }

    fn broadcast_in_dim(
        &self,
        shape: &[usize],
        dims: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.broadcast_in_dim_read(TensorRead::from_tensor(self), shape, dims)
    }

    fn tril(&self, k: i64, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.tril(self, k)
    }

    fn triu(&self, k: i64, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.triu(self, k)
    }

    fn extract_diag(
        &self,
        axis_a: usize,
        axis_b: usize,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.extract_diagonal(self, axis_a, axis_b)
    }

    fn embed_diag(
        &self,
        axis_a: usize,
        axis_b: usize,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.embed_diagonal(self, axis_a, axis_b)
    }

    fn dot_general(
        &self,
        rhs: &Tensor,
        config: DotGeneralConfig,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.dot_general_read(
            TensorRead::from_tensor(self),
            TensorRead::from_tensor(rhs),
            &config,
        )
    }

    fn dot_general_with_conj(
        &self,
        rhs: &Tensor,
        config: DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        session.dot_general_with_conj(self, rhs, &config, lhs_conj, rhs_conj)
    }

    fn scale_real(&self, factor: f64, session: &mut dyn BackendSession) -> Result<Tensor> {
        let scalar = crate::scale::real_scale_scalar(self.dtype(), factor)?;
        let scalar = session.upload_host_tensor(TensorRead::from_tensor(&scalar))?;
        TensorSessionOpsExt::mul(self, &scalar, session)
    }

    fn scale_complex(&self, factor: Complex64, session: &mut dyn BackendSession) -> Result<Tensor> {
        let scalar = crate::scale::complex_scale_scalar(self.dtype(), factor)?;
        let scalar = session.upload_host_tensor(TensorRead::from_tensor(&scalar))?;
        TensorSessionOpsExt::mul(self, &scalar, session)
    }

    fn sigmoid(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Sigmoid, TensorRead::from_tensor(self))?
        {
            return Ok(out);
        }
        run_session_composite(session, |ops| composite::sigmoid(ops, &borrowed(self)))
    }

    fn silu(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Silu, TensorRead::from_tensor(self))?
        {
            return Ok(out);
        }
        run_session_composite(session, |ops| composite::silu(ops, &borrowed(self)))
    }

    fn softplus(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Softplus, TensorRead::from_tensor(self))?
        {
            return Ok(out);
        }
        run_session_composite(session, |ops| composite::softplus(ops, &borrowed(self)))
    }

    fn gelu(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::Gelu, TensorRead::from_tensor(self))?
        {
            return Ok(out);
        }
        run_session_composite(session, |ops| composite::gelu(ops, &borrowed(self)))
    }

    fn gelu_tanh(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        if let Some(out) =
            session.fused_activation_read(ActivationOp::GeluTanh, TensorRead::from_tensor(self))?
        {
            return Ok(out);
        }
        run_session_composite(session, |ops| composite::gelu_tanh(ops, &borrowed(self)))
    }

    fn reduce_mean(
        &self,
        axes: Option<&[usize]>,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        run_session_composite(session, |ops| {
            composite::reduce_mean(ops, &borrowed(self), axes)
        })
    }

    fn softmax(&self, axis: usize, session: &mut dyn BackendSession) -> Result<Tensor> {
        run_session_composite(session, |ops| {
            composite::softmax(ops, &borrowed(self), axis)
        })
    }

    fn log_softmax(&self, axis: usize, session: &mut dyn BackendSession) -> Result<Tensor> {
        run_session_composite(session, |ops| {
            composite::log_softmax(ops, &borrowed(self), axis)
        })
    }

    fn masked_softmax(
        &self,
        mask: &Tensor,
        axis: usize,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let mask = borrowed(mask);
        run_session_composite(session, |ops| {
            composite::masked_softmax(ops, &borrowed(self), &mask, axis)
        })
    }

    fn masked_log_softmax(
        &self,
        mask: &Tensor,
        axis: usize,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let mask = borrowed(mask);
        run_session_composite(session, |ops| {
            composite::masked_log_softmax(ops, &borrowed(self), &mask, axis)
        })
    }

    fn layer_norm(
        &self,
        axis: usize,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let weight = weight.map(borrowed);
        let bias = bias.map(borrowed);
        run_session_composite(session, |ops| {
            composite::layer_norm(
                ops,
                &borrowed(self),
                axis,
                weight.as_ref(),
                bias.as_ref(),
                eps,
            )
        })
    }

    fn rms_norm(
        &self,
        axis: usize,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f64,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let weight = weight.map(borrowed);
        let bias = bias.map(borrowed);
        run_session_composite(session, |ops| {
            composite::rms_norm(
                ops,
                &borrowed(self),
                axis,
                weight.as_ref(),
                bias.as_ref(),
                eps,
            )
        })
    }

    fn take_along_axis(
        &self,
        indices: &Tensor,
        axis: usize,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let indices = borrowed(indices);
        run_session_composite(session, |ops| {
            composite::take_along_axis(ops, &borrowed(self), &indices, axis)
        })
    }
}

fn broadcast_binary_in<'a>(
    lhs: &'a Tensor,
    rhs: &'a Tensor,
    session: &mut dyn BackendSession,
) -> Result<(ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(TensorRead::from_tensor(lhs), &shape, session)?,
        broadcast_to_in_read(TensorRead::from_tensor(rhs), &shape, session)?,
    ))
}

fn broadcast_ternary_in<'a>(
    first: &'a Tensor,
    second: &'a Tensor,
    third: &'a Tensor,
    session: &mut dyn BackendSession,
) -> Result<(ReadInput<'a>, ReadInput<'a>, ReadInput<'a>)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to_in_read(TensorRead::from_tensor(first), &shape, session)?,
        broadcast_to_in_read(TensorRead::from_tensor(second), &shape, session)?,
        broadcast_to_in_read(TensorRead::from_tensor(third), &shape, session)?,
    ))
}

fn broadcast_error(err: tenferro_ops::broadcast::BroadcastError) -> Error {
    Error::validation("broadcast", broadcast_error_to_validation(err))
}

/// Resolve a reduction-family axis argument: `None` selects every axis.
///
/// An explicit axis list stays borrowed; only `None` builds a list.
pub(crate) fn all_axes_if_none(rank: usize, axes: Option<&[usize]>) -> Cow<'_, [usize]> {
    match axes {
        Some(axes) => Cow::Borrowed(axes),
        None => Cow::Owned((0..rank).collect()),
    }
}
