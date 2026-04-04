use computegraph::Operand;
use tenferro_tensor::Tensor;

pub fn permute(input: &Tensor, perm: &[usize]) -> Tensor {
    input.transpose_perm(perm)
}

pub fn reshape(input: &Tensor, shape: &[usize]) -> Tensor {
    Operand::reshape(input, shape)
}

pub fn broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> Tensor {
    Operand::broadcast_in_dim(input, shape, dims)
}

pub fn extract_diag(input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor {
    input.extract_diagonal(axis_a, axis_b)
}
