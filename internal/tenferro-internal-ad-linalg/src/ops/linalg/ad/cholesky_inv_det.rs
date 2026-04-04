use super::super::super::*;
use super::common::run_unary_tensor_ad;

pub struct CholeskyAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> CholeskyAdBuilder<'a, T>
where
    T: RealLinalgRuntimeValue + DynAdTensorTyped,
{
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Cholesky,
            op = "cholesky_ad",
            pullback = "cholesky_ad_pullback",
            input = self.tensor,
            primal = |ctx, tensor| {
                tenferro_linalg::cholesky::<T, _>(ctx, tensor).map_err(Error::from)
            },
            frule = |ctx, tensor, tangent| {
                tenferro_linalg::cholesky_frule::<T, _>(ctx, tensor, tangent).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::cholesky_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

pub fn cholesky_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> CholeskyAdBuilder<'a, T> {
    CholeskyAdBuilder { tensor }
}

pub struct InvAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> InvAdBuilder<'a, T>
where
    T: ScaledRealLinalgDispatchValue + DynAdTensorTyped,
{
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Inv,
            op = "inv_ad",
            pullback = "inv_ad_pullback",
            input = self.tensor,
            primal =
                |ctx, tensor| { tenferro_linalg::inv::<T, _>(ctx, tensor).map_err(Error::from) },
            frule = |ctx, tensor, tangent| {
                tenferro_linalg::inv_frule::<T, _>(ctx, tensor, tangent).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::inv_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

pub fn inv_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> InvAdBuilder<'a, T> {
    InvAdBuilder { tensor }
}

pub struct DetAdBuilder<'a, T: Scalar> {
    tensor: &'a AdTensor<T>,
}

impl<'a, T> DetAdBuilder<'a, T>
where
    T: ScaledRealLinalgDispatchValue + DynAdTensorTyped,
{
    pub fn run(self) -> Result<AdTensor<T>> {
        run_unary_tensor_ad!(
            ty = T,
            capability = tenferro_linalg::backend::LinalgCapabilityOp::Det,
            op = "det_ad",
            pullback = "det_ad_pullback",
            input = self.tensor,
            primal =
                |ctx, tensor| { tenferro_linalg::det::<T, _>(ctx, tensor).map_err(Error::from) },
            frule = |ctx, tensor, tangent| {
                tenferro_linalg::det_frule::<T, _>(ctx, tensor, tangent).map_err(Error::from)
            },
            rrule = |ctx, tensor, cotangent| {
                tenferro_linalg::det_rrule::<T, _>(ctx, tensor, cotangent).map_err(Error::from)
            },
        )
    }
}

pub fn det_ad<'a, T: Scalar>(tensor: &'a AdTensor<T>) -> DetAdBuilder<'a, T> {
    DetAdBuilder { tensor }
}
