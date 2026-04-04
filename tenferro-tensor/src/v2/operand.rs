use super::tensor_data::TensorData;
use super::types::Tensor;
use computegraph::Operand;

impl Operand for Tensor {
    fn zero(_shape: &[usize]) -> Self {
        todo!()
    }
    fn one(_shape: &[usize]) -> Self {
        todo!()
    }
    fn reshape(&self, _shape: &[usize]) -> Self {
        todo!()
    }
    fn broadcast_in_dim(&self, _shape: &[usize], _dims: &[usize]) -> Self {
        todo!()
    }
    fn add(&self, _other: &Self) -> Self {
        todo!()
    }
    fn multiply(&self, _other: &Self) -> Self {
        todo!()
    }
    fn reduce_sum(&self, _axes: &[usize]) -> Self {
        todo!()
    }
    fn dot_general(
        &self,
        _other: &Self,
        _lhs_contracting: &[usize],
        _rhs_contracting: &[usize],
        _lhs_batch: &[usize],
        _rhs_batch: &[usize],
    ) -> Self {
        todo!()
    }
    fn conj(&self) -> Self {
        todo!()
    }
}

pub fn generic_transpose<T: TensorData>(_t: &T, _perm: &[usize]) -> T {
    todo!()
}
pub fn generic_reshape<T: TensorData>(_t: &T, _shape: &[usize]) -> T {
    todo!()
}
pub fn generic_broadcast_in_dim<T: TensorData>(_t: &T, _shape: &[usize], _dims: &[usize]) -> T {
    todo!()
}
