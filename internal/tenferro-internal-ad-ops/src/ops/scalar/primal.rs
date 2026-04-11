use crate::Result;
use tenferro_algebra::Scalar;
use tenferro_tensor::Tensor;

pub fn scalar_binary_primal<T: Scalar>(
    _name: &str,
    _op: tenferro_prims::ScalarBinaryOp,
    _lhs: &Tensor<T>,
    _rhs: &Tensor<T>,
) -> Result<Tensor<T>> {
    unimplemented!("scalar_binary_primal not yet implemented")
}

pub fn scalar_unary_primal<T: Scalar>(
    _name: &str,
    _op: tenferro_prims::ScalarUnaryOp,
    _input: &Tensor<T>,
) -> Result<Tensor<T>> {
    unimplemented!("scalar_unary_primal not yet implemented")
}

pub fn scalar_full_reduction_primal<T: Scalar>(
    _name: &str,
    _op: tenferro_prims::ScalarReductionOp,
    _input: &Tensor<T>,
) -> Result<Tensor<T>> {
    unimplemented!("scalar_full_reduction_primal not yet implemented")
}

pub fn analytic_binary_primal<T: Scalar>(
    _name: &str,
    _op: tenferro_prims::AnalyticBinaryOp,
    _lhs: &Tensor<T>,
    _rhs: &Tensor<T>,
) -> Result<Tensor<T>> {
    unimplemented!("analytic_binary_primal not yet implemented")
}

pub fn analytic_unary_primal<T: Scalar>(
    _name: &str,
    _op: tenferro_prims::AnalyticUnaryOp,
    _input: &Tensor<T>,
) -> Result<Tensor<T>> {
    unimplemented!("analytic_unary_primal not yet implemented")
}

pub fn analytic_full_reduction_primal<T: Scalar>(
    _name: &str,
    _op: tenferro_prims::AnalyticReductionOp,
    _input: &Tensor<T>,
) -> Result<Tensor<T>> {
    unimplemented!("analytic_full_reduction_primal not yet implemented")
}
