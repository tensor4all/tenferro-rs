use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tenferro_algebra::Semiring;

use crate::backend::{SemiringBackend, TensorBackend};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::types::{dispatch_binary, flat_to_multi};
use crate::{Tensor, TypedTensor};

use super::{analytic, elementwise, gemm, indexing, linalg, reduction, structural};

/// CPU execution backend.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::cpu::CpuBackend;
///
/// let backend = CpuBackend::new();
/// ```
pub struct CpuBackend {
    pub(crate) pool: Arc<rayon::ThreadPool>,
}

fn shared_pools() -> &'static Mutex<HashMap<usize, Arc<rayon::ThreadPool>>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    POOLS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_or_create_pool(num_threads: usize) -> Arc<rayon::ThreadPool> {
    let mut pools = shared_pools()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = pools.get(&num_threads) {
        return Arc::clone(pool);
    }

    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|e| panic!("failed to create rayon thread pool: {e}")),
    );
    pools.insert(num_threads, Arc::clone(&pool));
    pool
}

impl CpuBackend {
    /// Create a CPU backend using the default thread count.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self {
            pool: get_or_create_pool(num_threads),
        }
    }

    /// Create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn with_threads(num_threads: usize) -> Self {
        assert!(num_threads >= 1, "thread count must be >= 1");
        Self {
            pool: get_or_create_pool(num_threads),
        }
    }

    /// Return the number of threads in the shared rayon thread pool.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    /// Run a closure inside this backend's shared rayon thread pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(1);
    /// let value = backend.install(|| 1 + 1);
    /// assert_eq!(value, 2);
    /// ```
    pub fn install<R>(&self, op: impl FnOnce() -> R + Send) -> R
    where
        R: Send,
    {
        self.pool.install(op)
    }
}

impl TensorBackend for CpuBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        self.install(|| elementwise::add(lhs, rhs))
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        self.install(|| elementwise::mul(lhs, rhs))
    }

    fn neg(&mut self, input: &Tensor) -> Tensor {
        self.install(|| elementwise::neg(input))
    }

    fn conj(&mut self, input: &Tensor) -> Tensor {
        self.install(|| elementwise::conj(input))
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        self.install(|| elementwise::div(lhs, rhs))
    }

    fn abs(&mut self, input: &Tensor) -> Tensor {
        self.install(|| elementwise::abs(input))
    }

    fn sign(&mut self, input: &Tensor) -> Tensor {
        self.install(|| elementwise::sign(input))
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        self.install(|| elementwise::maximum(lhs, rhs))
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        self.install(|| elementwise::minimum(lhs, rhs))
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> Tensor {
        self.install(|| elementwise::compare(lhs, rhs, dir))
    }

    fn select(&mut self, pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> Tensor {
        self.install(|| elementwise::select(pred, on_true, on_false))
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> Tensor {
        self.install(|| elementwise::clamp(input, lower, upper))
    }

    fn scale(&mut self, input: &Tensor, factor: f64) -> Tensor {
        self.install(|| elementwise::scale(input, factor))
    }

    fn scale_complex(&mut self, input: &Tensor, re: f64, im: f64) -> Tensor {
        self.install(|| elementwise::scale_complex(input, re, im))
    }

    fn exp(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::exp(input))
    }

    fn log(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::log(input))
    }

    fn sin(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::sin(input))
    }

    fn cos(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::cos(input))
    }

    fn tanh(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::tanh(input))
    }

    fn sqrt(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::sqrt(input))
    }

    fn rsqrt(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::rsqrt(input))
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        self.install(|| analytic::pow(lhs, rhs))
    }

    fn expm1(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::expm1(input))
    }

    fn log1p(&mut self, input: &Tensor) -> Tensor {
        self.install(|| analytic::log1p(input))
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> Tensor {
        self.install(|| structural::transpose(input, perm))
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> Tensor {
        self.install(|| structural::reshape(input, shape))
    }

    fn broadcast_in_dim(&mut self, input: &Tensor, shape: &[usize], dims: &[usize]) -> Tensor {
        self.install(|| structural::broadcast_in_dim(input, shape, dims))
    }

    fn extract_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor {
        self.install(|| structural::extract_diagonal(input, axis_a, axis_b))
    }

    fn embed_diagonal(&mut self, input: &Tensor, axis_a: usize, axis_b: usize) -> Tensor {
        self.install(|| structural::embed_diagonal(input, axis_a, axis_b))
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> Tensor {
        self.install(|| structural::tril(input, k))
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> Tensor {
        self.install(|| structural::triu(input, k))
    }

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        self.install(|| reduction::reduce_sum(input, axes))
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        self.install(|| reduction::reduce_prod(input, axes))
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        self.install(|| reduction::reduce_max(input, axes))
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        self.install(|| reduction::reduce_min(input, axes))
    }

    fn dot_general(&mut self, lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor {
        self.install(|| dispatch_binary!(lhs, rhs, |a, b| gemm::dot_general(a, b, config)))
    }

    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> Tensor {
        self.install(|| indexing::gather(operand, start_indices, config))
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> Tensor {
        self.install(|| indexing::scatter(operand, scatter_indices, updates, config))
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> Tensor {
        self.install(|| indexing::slice(input, config))
    }

    fn dynamic_slice(&mut self, input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> Tensor {
        self.install(|| indexing::dynamic_slice(input, starts, slice_sizes))
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> Tensor {
        self.install(|| indexing::pad(input, config))
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> Tensor {
        self.install(|| indexing::concatenate(inputs, axis))
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> Tensor {
        self.install(|| indexing::reverse(input, axes))
    }

    fn cholesky(&mut self, input: &Tensor) -> Tensor {
        self.install(|| match input {
            Tensor::F64(t) => Tensor::F64(linalg::cholesky(t)),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => Tensor::C64(linalg::cholesky(t)),
            _ => todo!("cholesky: unsupported dtype"),
        })
    }

    fn triangular_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Tensor {
        self.install(|| match (a, b) {
            (Tensor::F64(a), Tensor::F64(b)) => Tensor::F64(linalg::triangular_solve(
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => Tensor::C64(linalg::triangular_solve(
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )),
            _ => todo!("triangular_solve: unsupported dtype"),
        })
    }

    fn svd(&mut self, input: &Tensor) -> Vec<Tensor> {
        self.install(|| match input {
            Tensor::F64(t) => linalg::svd(t).into_iter().map(Tensor::F64).collect(),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::svd(t).into_iter().map(Tensor::C64).collect(),
            _ => todo!("svd: unsupported dtype"),
        })
    }

    fn qr(&mut self, input: &Tensor) -> Vec<Tensor> {
        self.install(|| match input {
            Tensor::F64(t) => linalg::qr(t).into_iter().map(Tensor::F64).collect(),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::qr(t).into_iter().map(Tensor::C64).collect(),
            _ => todo!("qr: unsupported dtype"),
        })
    }

    fn eigh(&mut self, input: &Tensor) -> Vec<Tensor> {
        self.install(|| match input {
            Tensor::F64(t) => linalg::eigh(t).into_iter().map(Tensor::F64).collect(),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::eigh(t).into_iter().map(Tensor::C64).collect(),
            _ => todo!("eigh: unsupported dtype"),
        })
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> Tensor {
        self.install(|| match (a, b) {
            (Tensor::F64(a), Tensor::F64(b)) => Tensor::F64(linalg::solve(a, b)),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => Tensor::C64(linalg::solve(a, b)),
            _ => todo!("solve: unsupported dtype"),
        })
    }
}

impl<Alg: Semiring> SemiringBackend<Alg> for CpuBackend {
    fn batched_gemm(
        &mut self,
        lhs: &TypedTensor<Alg::Scalar>,
        rhs: &TypedTensor<Alg::Scalar>,
        config: &DotGeneralConfig,
    ) -> TypedTensor<Alg::Scalar> {
        self.install(|| {
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
        })
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
