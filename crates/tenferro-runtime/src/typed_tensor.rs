//! Typed tensor operation extension traits.
//!
//! Operation families that are no longer part of core, including einsum, live
//! in their extension crates.

use tenferro_ops::broadcast::{
    broadcast_input_plan, broadcast_shape, broadcast_shapes, BroadcastError,
};
use tenferro_tensor::validate::matmul_config_for_shapes;
use tenferro_tensor::{
    BackendSession, CompareDir, DType, Error, Result, Tensor, TensorRead, TensorScalar,
    ValidationError,
};

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
        axes: &[usize],
        session: &mut dyn BackendSession,
    ) -> Result<TypedTensor<T>> {
        let out = session.reduce_sum_read(T::tensor_read(self), axes)?;
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

    fn abs(&self, session: &mut dyn BackendSession) -> Result<TypedTensor<T>> {
        let out = session.abs_read(T::tensor_read(self))?;
        into_typed_result("abs", out)
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
enum ReadInput<'a> {
    Borrowed(TensorRead<'a>),
    Owned(Tensor),
}

impl ReadInput<'_> {
    fn tensor_read(&self) -> TensorRead<'_> {
        match self {
            Self::Borrowed(read) => read.clone(),
            Self::Owned(tensor) => TensorRead::from_tensor(tensor),
        }
    }
}

fn broadcast_to_in_read<'a, T: TensorScalar>(
    input: &'a TypedTensor<T>,
    target_shape: &[usize],
    session: &mut dyn BackendSession,
) -> Result<ReadInput<'a>> {
    if input.shape() == target_shape {
        return Ok(ReadInput::Borrowed(T::tensor_read(input)));
    }

    let plan = broadcast_input_plan(input.shape(), target_shape).map_err(broadcast_error)?;
    let source = if plan.source_shape == input.shape() {
        ReadInput::Borrowed(T::tensor_read(input))
    } else {
        let reshaped = session.reshape_read(T::tensor_read(input), &plan.source_shape)?;
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
        broadcast_to_in_read(lhs, &shape, session)?,
        broadcast_to_in_read(rhs, &shape, session)?,
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
        broadcast_to_in_read(first, &shape, session)?,
        broadcast_to_in_read(second, &shape, session)?,
        broadcast_to_in_read(third, &shape, session)?,
    ))
}

fn broadcast_error(err: BroadcastError) -> Error {
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
                expected: core_dtype(T::dtype()),
                actual: core_dtype(actual),
            },
        )
    })
}

fn core_dtype(dtype: DType) -> tenferro_tensor::core::DType {
    match dtype {
        DType::F32 => tenferro_tensor::core::DType::F32,
        DType::F64 => tenferro_tensor::core::DType::F64,
        DType::I32 => tenferro_tensor::core::DType::I32,
        DType::I64 => tenferro_tensor::core::DType::I64,
        DType::Bool => tenferro_tensor::core::DType::Bool,
        DType::C32 => tenferro_tensor::core::DType::C32,
        DType::C64 => tenferro_tensor::core::DType::C64,
    }
}
