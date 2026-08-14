//! Concrete tensor operation extension trait.
//!
//! `tenferro-tensor` owns storage and backend traits. This runtime crate
//! provides backend-parametric session-explicit operation methods through
//! [`TensorSessionOpsExt`].

use tenferro_ops::broadcast::{
    broadcast_error_to_validation, broadcast_input_plan, broadcast_shape, broadcast_shapes,
};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{BackendSession, CompareDir, DType, Error, Result};

use crate::TensorSessionOpsExt;
use tenferro_tensor::Tensor;

impl TensorSessionOpsExt for Tensor {
    fn add(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.add(&lhs, &rhs)
    }

    fn mul(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.mul(&lhs, &rhs)
    }

    fn exp(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.exp(self)
    }

    fn reduce_sum(&self, axes: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.reduce_sum(self, axes)
    }

    fn convert(&self, to: DType, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.convert(self, to)
    }

    fn cast(&self, to: DType, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.cast(self, to)
    }

    fn sub(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.sub(&lhs, &rhs)
    }

    fn div(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.div(&lhs, &rhs)
    }

    fn rem(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.rem(&lhs, &rhs)
    }

    fn pow(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.pow(&lhs, &rhs)
    }

    fn maximum(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.maximum(&lhs, &rhs)
    }

    fn minimum(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.minimum(&lhs, &rhs)
    }

    fn neg(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.neg(self)
    }

    fn abs(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.abs(self)
    }

    fn sign(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sign(self)
    }

    fn conj(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.conj(self)
    }

    fn log(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.log(self)
    }

    fn expm1(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.expm1(self)
    }

    fn log1p(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.log1p(self)
    }

    fn sin(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sin(self)
    }

    fn cos(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.cos(self)
    }

    fn tanh(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.tanh(self)
    }

    fn sqrt(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.sqrt(self)
    }

    fn rsqrt(&self, session: &mut dyn BackendSession) -> Result<Tensor> {
        session.rsqrt(self)
    }

    fn compare(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (lhs, rhs) = broadcast_binary_in(self, rhs, session)?;
        session.compare(&lhs, &rhs, &dir)
    }

    fn where_select(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (condition, on_true, on_false) =
            broadcast_ternary_in(self, on_true, on_false, session)?;
        session.select(&condition, &on_true, &on_false)
    }

    fn clamp(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        session: &mut dyn BackendSession,
    ) -> Result<Tensor> {
        let (input, lower, upper) = broadcast_ternary_in(self, lower, upper, session)?;
        session.clamp(&input, &lower, &upper)
    }

    fn matmul(&self, rhs: &Tensor, session: &mut dyn BackendSession) -> Result<Tensor> {
        let config = matmul_config_for_shapes("matmul", self.shape(), rhs.shape())?;
        session.dot_general(self, rhs, &config)
    }

    fn reshape(&self, shape: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.reshape(self, shape)
    }

    fn transpose(&self, perm: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.transpose(self, perm)
    }
}

fn broadcast_to_in(
    input: &Tensor,
    target_shape: &[usize],
    session: &mut dyn BackendSession,
) -> Result<Tensor> {
    let input_shape = input.shape();
    if input_shape == target_shape {
        return input.duplicate();
    }

    let plan = broadcast_input_plan(input_shape, target_shape).map_err(broadcast_error)?;
    let source = if plan.source_shape == input_shape {
        input.duplicate()?
    } else {
        session.reshape(input, &plan.source_shape)?
    };
    session.broadcast_in_dim(&source, target_shape, &plan.dims)
}

fn broadcast_binary_in(
    lhs: &Tensor,
    rhs: &Tensor,
    session: &mut dyn BackendSession,
) -> Result<(Tensor, Tensor)> {
    let shape = broadcast_shape(lhs.shape(), rhs.shape()).map_err(broadcast_error)?;
    Ok((
        broadcast_to_in(lhs, &shape, session)?,
        broadcast_to_in(rhs, &shape, session)?,
    ))
}

fn broadcast_ternary_in(
    first: &Tensor,
    second: &Tensor,
    third: &Tensor,
    session: &mut dyn BackendSession,
) -> Result<(Tensor, Tensor, Tensor)> {
    let shape = broadcast_shapes([first.shape(), second.shape(), third.shape()])
        .map_err(broadcast_error)?;
    Ok((
        broadcast_to_in(first, &shape, session)?,
        broadcast_to_in(second, &shape, session)?,
        broadcast_to_in(third, &shape, session)?,
    ))
}

fn broadcast_error(err: tenferro_ops::broadcast::BroadcastError) -> Error {
    Error::validation("broadcast", broadcast_error_to_validation(err))
}
