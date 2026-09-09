//! Concrete tensor operation extension trait.
//!
//! `tenferro-tensor` owns storage and backend traits. This runtime crate
//! provides backend-parametric session-explicit operation methods through
//! [`TensorSessionOpsExt`].

use tenferro_ops::broadcast::{broadcast_error_to_validation, broadcast_shape, broadcast_shapes};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{BackendSession, CompareDir, DType, Error, Result, TensorRead};

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
        session.dot_general(self, rhs, &config)
    }

    fn reshape(&self, shape: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.reshape(self, shape)
    }

    fn transpose(&self, perm: &[usize], session: &mut dyn BackendSession) -> Result<Tensor> {
        session.transpose(self, perm)
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
