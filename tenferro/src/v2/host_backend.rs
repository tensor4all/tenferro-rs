use computegraph::Operand;
use tenferro_ops::config::DotGeneralConfig;
use tenferro_tensor::v2::Tensor;

use super::backend::SemiringCore;

pub struct HostBackend;

impl HostBackend {
    pub fn new() -> Self {
        Self
    }
}

impl SemiringCore for HostBackend {
    type Operand = Tensor;

    fn batched_gemm(&mut self, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor {
        lhs.dot_general(
            rhs,
            &config.lhs_contracting_dims,
            &config.rhs_contracting_dims,
            &config.lhs_batch_dims,
            &config.rhs_batch_dims,
        )
    }

    fn reduce_sum(&mut self, operand: &Tensor, axes: &[usize]) -> Tensor {
        operand.reduce_sum(axes)
    }

    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Operand::add(lhs, rhs)
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Operand::multiply(lhs, rhs)
    }
}
