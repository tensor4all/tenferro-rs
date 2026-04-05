use tenferro_algebra::Semiring;

use crate::backend::{SemiringBackend, TensorBackend};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::types::{dispatch_binary, flat_to_multi};
use crate::{Tensor, TypedTensor};

use super::{analytic, elementwise, gemm, indexing, reduction, structural};

/// CPU execution backend.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::cpu::CpuBackend;
///
/// let backend = CpuBackend::new();
/// ```
#[derive(Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Create a CPU backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        Self
    }
}

impl TensorBackend for CpuBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        elementwise::add(lhs, rhs)
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        elementwise::mul(lhs, rhs)
    }

    fn neg(&mut self, input: &Tensor) -> Tensor {
        elementwise::neg(input)
    }

    fn conj(&mut self, input: &Tensor) -> Tensor {
        elementwise::conj(input)
    }

    fn div(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
        todo!("div")
    }

    fn abs(&mut self, _input: &Tensor) -> Tensor {
        todo!("abs")
    }

    fn sign(&mut self, _input: &Tensor) -> Tensor {
        todo!("sign")
    }

    fn maximum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
        todo!("maximum")
    }

    fn minimum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> Tensor {
        todo!("minimum")
    }

    fn compare(&mut self, _lhs: &Tensor, _rhs: &Tensor, _dir: &CompareDir) -> Tensor {
        todo!("compare")
    }

    fn select(&mut self, _pred: &Tensor, _on_true: &Tensor, _on_false: &Tensor) -> Tensor {
        todo!("select")
    }

    fn clamp(&mut self, _input: &Tensor, _lower: &Tensor, _upper: &Tensor) -> Tensor {
        todo!("clamp")
    }

    fn exp(&mut self, input: &Tensor) -> Tensor {
        analytic::exp(input)
    }

    fn log(&mut self, input: &Tensor) -> Tensor {
        analytic::log(input)
    }

    fn sin(&mut self, input: &Tensor) -> Tensor {
        analytic::sin(input)
    }

    fn cos(&mut self, input: &Tensor) -> Tensor {
        analytic::cos(input)
    }

    fn tanh(&mut self, input: &Tensor) -> Tensor {
        analytic::tanh(input)
    }

    fn sqrt(&mut self, input: &Tensor) -> Tensor {
        analytic::sqrt(input)
    }

    fn rsqrt(&mut self, input: &Tensor) -> Tensor {
        analytic::rsqrt(input)
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        analytic::pow(lhs, rhs)
    }

    fn expm1(&mut self, input: &Tensor) -> Tensor {
        analytic::expm1(input)
    }

    fn log1p(&mut self, input: &Tensor) -> Tensor {
        analytic::log1p(input)
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> Tensor {
        structural::transpose(input, perm)
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> Tensor {
        structural::reshape(input, shape)
    }

    fn broadcast_in_dim(&mut self, input: &Tensor, shape: &[usize], dims: &[usize]) -> Tensor {
        structural::broadcast_in_dim(input, shape, dims)
    }

    fn extract_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor {
        structural::extract_diagonal(input, axis_a, axis_b)
    }

    fn embed_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor {
        structural::embed_diagonal(input, axis_a, axis_b)
    }

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        reduction::reduce_sum(input, axes)
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        reduction::reduce_prod(input, axes)
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        reduction::reduce_max(input, axes)
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        reduction::reduce_min(input, axes)
    }

    fn dot_general(&mut self, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor {
        dispatch_binary!(lhs, rhs, |a, b| gemm::dot_general(a, b, config))
    }

    fn gather(&mut self, input: &Tensor, config: &GatherConfig) -> Tensor {
        indexing::gather(input, config)
    }

    fn scatter(&mut self, input: &Tensor, updates: &Tensor, config: &ScatterConfig) -> Tensor {
        indexing::scatter(input, updates, config)
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> Tensor {
        indexing::slice(input, config)
    }

    fn dynamic_slice(&mut self, input: &Tensor, starts: &Tensor) -> Tensor {
        indexing::dynamic_slice(input, starts)
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> Tensor {
        indexing::pad(input, config)
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> Tensor {
        indexing::concatenate(inputs, axis)
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        indexing::reverse(input, axes)
    }

    fn cholesky(&mut self, _input: &Tensor) -> Tensor {
        todo!("cholesky")
    }

    fn svd(&mut self, _input: &Tensor) -> Vec<Tensor> {
        todo!("svd")
    }

    fn qr(&mut self, _input: &Tensor) -> Vec<Tensor> {
        todo!("qr")
    }

    fn eigh(&mut self, _input: &Tensor) -> Vec<Tensor> {
        todo!("eigh")
    }

    fn solve(&mut self, _a: &Tensor, _b: &Tensor) -> Tensor {
        todo!("solve")
    }
}

impl<Alg: Semiring> SemiringBackend<Alg> for CpuBackend {
    fn batched_gemm(
        &mut self,
        lhs: &TypedTensor<Alg::Scalar>,
        rhs: &TypedTensor<Alg::Scalar>,
        config: &DotGeneralConfig,
    ) -> TypedTensor<Alg::Scalar> {
        assert_eq!(
            config.lhs_contracting_dims.len(),
            config.rhs_contracting_dims.len()
        );
        assert_eq!(config.lhs_batch_dims.len(), config.rhs_batch_dims.len());

        let lhs_rank = lhs.shape.len();
        let rhs_rank = rhs.shape.len();
        let lhs_free: Vec<usize> = (0..lhs_rank)
            .filter(|d| {
                !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d)
            })
            .collect();
        let rhs_free: Vec<usize> = (0..rhs_rank)
            .filter(|d| {
                !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d)
            })
            .collect();

        let batch_shape: Vec<usize> = config
            .lhs_batch_dims
            .iter()
            .map(|&d| lhs.shape[d])
            .collect();
        let lhs_free_shape: Vec<usize> = lhs_free.iter().map(|&d| lhs.shape[d]).collect();
        let rhs_free_shape: Vec<usize> = rhs_free.iter().map(|&d| rhs.shape[d]).collect();
        let contract_shape: Vec<usize> = config
            .lhs_contracting_dims
            .iter()
            .map(|&d| lhs.shape[d])
            .collect();

        let mut out_shape = Vec::new();
        out_shape.extend_from_slice(&batch_shape);
        out_shape.extend_from_slice(&lhs_free_shape);
        out_shape.extend_from_slice(&rhs_free_shape);
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_n: usize = out_shape.iter().product();
        let contract_n: usize = contract_shape.iter().product();

        let mut result = TypedTensor::zeros(out_shape.clone());
        let n_batch = batch_shape.len();
        let n_lhs_free = lhs_free_shape.len();

        let mut out_idx = vec![0usize; out_shape.len()];
        let mut lhs_idx = vec![0usize; lhs_rank];
        let mut rhs_idx = vec![0usize; rhs_rank];
        let mut contract_idx = vec![0usize; contract_shape.len()];

        for flat_out in 0..out_n {
            flat_to_multi(flat_out, &out_shape, &mut out_idx);

            let batch_vals = &out_idx[..n_batch];
            let lhs_free_vals = &out_idx[n_batch..n_batch + n_lhs_free];
            let rhs_free_vals = &out_idx[n_batch + n_lhs_free..];

            for (bi, &ld) in config.lhs_batch_dims.iter().enumerate() {
                lhs_idx[ld] = batch_vals[bi];
            }
            for (bi, &rd) in config.rhs_batch_dims.iter().enumerate() {
                rhs_idx[rd] = batch_vals[bi];
            }
            for (fi, &ld) in lhs_free.iter().enumerate() {
                lhs_idx[ld] = lhs_free_vals[fi];
            }
            for (fi, &rd) in rhs_free.iter().enumerate() {
                rhs_idx[rd] = rhs_free_vals[fi];
            }

            let mut acc = Alg::zero();
            for flat_k in 0..contract_n {
                flat_to_multi(flat_k, &contract_shape, &mut contract_idx);
                for (ci, &ld) in config.lhs_contracting_dims.iter().enumerate() {
                    lhs_idx[ld] = contract_idx[ci];
                }
                for (ci, &rd) in config.rhs_contracting_dims.iter().enumerate() {
                    rhs_idx[rd] = contract_idx[ci];
                }
                acc = Alg::add(acc, Alg::mul(*lhs.get(&lhs_idx), *rhs.get(&rhs_idx)));
            }

            *result.get_mut(&out_idx) = acc;
        }

        result
    }
}
