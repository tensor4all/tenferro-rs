use computegraph::Operand;
use tenferro_ops::config::DotGeneralConfig;
use tenferro_tensor::v2::Tensor;

use super::backend::SemiringCore;

/// CPU backend for the v2 engine.
///
/// When the `cpu-faer` feature is enabled (default), batched GEMM uses
/// faer's strided matmul for zero-copy, zero-allocation execution.
/// Otherwise a naive fallback is used.
///
/// # Examples
///
/// ```ignore
/// use tenferro::v2::cpu_backend::CpuBackend;
/// use tenferro::v2::engine::Engine;
///
/// let engine = Engine::new(CpuBackend::new());
/// ```
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend instance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro::v2::cpu_backend::CpuBackend;
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiringCore for CpuBackend {
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
