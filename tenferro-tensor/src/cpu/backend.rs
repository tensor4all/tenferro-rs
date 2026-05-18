use std::sync::Arc;

use crate::backend::{TensorBackend, TensorExec};
use crate::buffer_pool::{BufferPool, BufferPoolStats, PoolScalar};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::validate::validate_nonsingular_u;
use crate::{Buffer, Tensor, TypedTensor};

use super::exec_session::CpuExecSession;
use super::{analytic, elementwise, gemm, indexing, linalg, reduction, structural, CpuContext};

/// CPU execution backend.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::cpu::CpuBackend;
///
/// let backend = CpuBackend::new();
/// ```
pub struct CpuBackend {
    pub(crate) ctx: Arc<CpuContext>,
    pub(crate) buffers: BufferPool,
}

impl CpuBackend {
    /// Create a CPU backend using the environment-driven CPU context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    pub fn new() -> Self {
        Self::from_context(Arc::new(CpuContext::from_env()))
    }

    /// Try to create a CPU backend using `RAYON_NUM_THREADS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::try_new()
    ///     .unwrap_or_else(|_| CpuBackend::with_threads(1));
    /// let _ = backend.num_threads();
    /// ```
    pub fn try_new() -> crate::Result<Self> {
        CpuContext::try_from_env().map(|ctx| Self::from_context(Arc::new(ctx)))
    }

    /// Create a CPU backend from an existing context.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    /// use tenferro_tensor::cpu::{CpuBackend, CpuContext};
    ///
    /// let ctx = Arc::new(CpuContext::with_threads(2));
    /// let backend = CpuBackend::from_context(ctx);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn from_context(ctx: Arc<CpuContext>) -> Self {
        Self {
            ctx,
            buffers: BufferPool::new(),
        }
    }

    /// Create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn with_threads(num_threads: usize) -> Self {
        match Self::try_with_threads(num_threads) {
            Ok(backend) => backend,
            Err(err) => panic!("{err}"),
        }
    }

    /// Try to create a CPU backend with a custom thread count.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::try_with_threads(1).unwrap();
    /// assert_eq!(backend.num_threads(), 1);
    /// ```
    pub fn try_with_threads(num_threads: usize) -> crate::Result<Self> {
        CpuContext::try_with_threads(num_threads)
            .map(|ctx| Self::from_context(Arc::new(ctx)))
            .map_err(|err| match err {
                crate::Error::InvalidConfig { message, .. } => crate::Error::InvalidConfig {
                    op: "CpuBackend::try_with_threads",
                    message,
                },
                crate::Error::BackendFailure { message, .. } => crate::Error::BackendFailure {
                    op: "CpuBackend::try_with_threads",
                    message,
                },
                err => err,
            })
    }

    /// Return the number of threads in this backend's CPU context.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn num_threads(&self) -> usize {
        self.ctx.num_threads()
    }

    /// Number of retained typed host buffers currently held by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn buffer_pool_len(&self) -> usize {
        self.buffers.len()
    }

    /// Snapshot reusable typed host buffers currently retained by this backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// let stats = backend.buffer_pool_stats();
    /// assert_eq!(stats.buffers, 0);
    /// assert_eq!(stats.capacity_bytes, 0);
    /// ```
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        self.buffers.stats()
    }

    /// Reset reusable typed host buffers currently retained by this backend.
    ///
    /// This releases pool-owned vectors to the process allocator. Operating
    /// system RSS may not fall immediately because allocators can retain freed
    /// pages for future allocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let mut backend = CpuBackend::new();
    /// backend.reset_buffer_pool();
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn reset_buffer_pool(&mut self) {
        self.buffers.clear();
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
        self.ctx.install(op)
    }

    fn install_with_pool<R>(&mut self, op: impl FnOnce(&mut BufferPool) -> R + Send) -> R
    where
        R: Send,
    {
        let mut buffers = std::mem::take(&mut self.buffers);
        let (result, buffers) = self.ctx.install(|| {
            let result = op(&mut buffers);
            (result, buffers)
        });
        self.buffers = buffers;
        result
    }
}

impl TensorBackend for CpuBackend {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::add(lhs, rhs))
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::mul(lhs, rhs))
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::neg(input))
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::conj(input))
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::div(lhs, rhs))
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::abs(input))
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::sign(input))
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::maximum(lhs, rhs))
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::minimum(lhs, rhs))
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor> {
        self.install(|| elementwise::compare(lhs, rhs, dir))
    }

    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor> {
        self.install(|| elementwise::select(pred, on_true, on_false))
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor> {
        self.install(|| elementwise::clamp(input, lower, upper))
    }

    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::exp(input))
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::log(input))
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::sin(input))
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::cos(input))
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::tanh(input))
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::sqrt(input))
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::rsqrt(input))
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::pow(lhs, rhs))
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::expm1(input))
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        self.install(|| analytic::log1p(input))
    }

    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor> {
        self.install(|| structural::transpose(input, perm))
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor> {
        self.install(|| structural::reshape(input, shape))
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.install(|| structural::broadcast_in_dim(input, shape, dims))
    }

    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor> {
        self.install(|| structural::convert(input, to))
    }

    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.install(|| structural::extract_diagonal(input, axis_a, axis_b))
    }

    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor> {
        self.install(|| structural::embed_diagonal(input, axis_a, axis_b))
    }

    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.install(|| structural::tril(input, k))
    }

    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor> {
        self.install(|| structural::triu(input, k))
    }

    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_sum(input, axes))
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_prod(input, axes))
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_max(input, axes))
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| reduction::reduce_min(input, axes))
    }

    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match (lhs, rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                gemm::dot_general(buffers, ctx.as_ref(), a, b, config).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                gemm::dot_general(buffers, ctx.as_ref(), a, b, config).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                gemm::dot_general(buffers, ctx.as_ref(), a, b, config).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                gemm::dot_general(buffers, ctx.as_ref(), a, b, config).map(Tensor::C64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                gemm::dot_general(buffers, a, b, config).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                gemm::dot_general(buffers, a, b, config).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                gemm::dot_general(buffers, a, b, config).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                gemm::dot_general(buffers, a, b, config).map(Tensor::C64)
            }
            _ => Err(crate::Error::DTypeMismatch {
                op: "dot_general",
                lhs: lhs.dtype(),
                rhs: rhs.dtype(),
            }),
        })
    }

    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| {
            indexing::gather_with_pool(buffers, operand, start_indices, config)
        })
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        self.install(|| indexing::scatter(operand, scatter_indices, updates, config))
    }

    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor> {
        self.install(|| indexing::try_slice(input, config))
    }

    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor> {
        self.install(|| indexing::dynamic_slice(input, starts, slice_sizes))
    }

    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor> {
        self.install(|| indexing::dynamic_update_slice(operand, update, starts))
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        self.install(|| indexing::try_pad(input, config))
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.install_with_pool(|buffers| indexing::try_concatenate_with_pool(buffers, inputs, axis))
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| indexing::reverse(input, axes))
    }

    fn cholesky(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::cholesky(buffers, t).map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::cholesky(buffers, t).map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::cholesky(buffers, t).map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::cholesky(buffers, t).map(Tensor::C64),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::cholesky(ctx.as_ref(), buffers, t).map(Tensor::C64),
            _ => Err(unsupported_dtype("cholesky", input.dtype())),
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
    ) -> crate::Result<Tensor> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match (a, b) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::F64),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C32),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => linalg::triangular_solve(
                ctx.as_ref(),
                buffers,
                a,
                b,
                left_side,
                lower,
                transpose_a,
                unit_diagonal,
            )
            .map(Tensor::C64),
            _ => {
                if a.dtype() != b.dtype() {
                    Err(crate::Error::DTypeMismatch {
                        op: "triangular_solve",
                        lhs: a.dtype(),
                        rhs: b.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("triangular_solve", a.dtype()))
                }
            }
        })
    }

    fn lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F64).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => {
                linalg::lu(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C64).collect())
            }
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("lu", input.dtype())),
        })
    }

    fn full_piv_lu(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::full_piv_lu(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::full_piv_lu(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("full_piv_lu", input.dtype())),
        })
    }

    fn full_piv_lu_solve(
        &mut self,
        a: &Tensor,
        b: &Tensor,
        transpose_a: bool,
    ) -> crate::Result<Tensor> {
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return Ok(zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        let result = self.install_with_pool(|buffers| match (a, &rhs) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F32(a), Tensor::F32(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::F32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-blas")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(buffers, a, b, transpose_a).map(Tensor::C64)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C32(a), Tensor::C32(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::C32)
            }
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => {
                linalg::full_piv_lu_solve(ctx.as_ref(), buffers, a, b, transpose_a).map(Tensor::C64)
            }
            _ => {
                if a.dtype() != rhs.dtype() {
                    Err(crate::Error::DTypeMismatch {
                        op: "full_piv_lu_solve",
                        lhs: a.dtype(),
                        rhs: rhs.dtype(),
                    })
                } else {
                    Err(unsupported_dtype("full_piv_lu_solve", a.dtype()))
                }
            }
        })?;

        if let Some(shape) = restore_shape {
            self.reshape(&result, &shape)
        } else {
            Ok(result)
        }
    }

    fn svd(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::svd(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::svd(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("svd", input.dtype())),
        })
    }

    fn qr(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::F64).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C32).collect())
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => {
                linalg::qr(buffers, t).map(|outputs| outputs.into_iter().map(Tensor::C64).collect())
            }
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::qr(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("qr", input.dtype())),
        })
    }

    fn eigh(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F32(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F32(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::F64).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C32(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-blas")]
            Tensor::C64(t) => linalg::eigh(buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C32(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C32).collect()),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => linalg::eigh(ctx.as_ref(), buffers, t)
                .map(|outputs| outputs.into_iter().map(Tensor::C64).collect()),
            _ => Err(unsupported_dtype("eigh", input.dtype())),
        })
    }

    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        if !matches!(
            input,
            Tensor::F32(_) | Tensor::F64(_) | Tensor::C32(_) | Tensor::C64(_)
        ) {
            return Err(unsupported_dtype("eig", input.dtype()));
        }
        #[cfg(feature = "cpu-faer")]
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| {
            #[cfg(feature = "cpu-faer")]
            {
                linalg::eig(ctx.as_ref(), buffers, input)
            }
            #[cfg(feature = "cpu-blas")]
            {
                linalg::eig(buffers, input)
            }
        })
    }

    fn solve(&mut self, a: &Tensor, b: &Tensor) -> crate::Result<Tensor> {
        if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
            return Ok(zeros_like_tensor(b));
        }

        let (rhs, restore_shape) = if let Some(matrix_rhs_shape) = batched_vector_rhs_shape(a, b) {
            (
                self.reshape(b, &matrix_rhs_shape)?,
                Some(b.shape().to_vec()),
            )
        } else {
            (b.clone(), None)
        };

        let outputs = self.lu(a)?;
        let p = &outputs[0];
        let l = &outputs[1];
        let u = &outputs[2];
        validate_nonsingular_u(u)?;

        let pb = matmul_preserve_trailing_batch(self, p, &rhs)?;
        let z = self.triangular_solve(l, &pb, true, true, false, true)?;
        let x = self.triangular_solve(u, &z, true, false, false, false)?;
        if let Some(shape) = restore_shape {
            self.reshape(&x, &shape)
        } else {
            Ok(x)
        }
    }

    fn with_exec_session<R: Send>(&mut self, f: impl FnOnce(&mut dyn TensorExec) -> R + Send) -> R {
        let mut buffers = std::mem::take(&mut self.buffers);
        let ctx = Arc::clone(&self.ctx);
        let result = ctx.install(|| {
            let mut session = CpuExecSession {
                ctx: ctx.as_ref(),
                buffers: &mut buffers,
            };
            f(&mut session)
        });
        self.buffers = buffers;
        result
    }

    fn reclaim_buffer(&mut self, tensor: Tensor) {
        match tensor {
            Tensor::F32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::F64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::I64(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C32(t) => reclaim_typed(&mut self.buffers, t),
            Tensor::C64(t) => reclaim_typed(&mut self.buffers, t),
        }
    }
}

fn has_zero_dim(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn batched_vector_rhs_shape(a: &Tensor, b: &Tensor) -> Option<Vec<usize>> {
    if b.shape().len() == 1 {
        return Some(vec![b.shape()[0], 1]);
    }

    let is_batched_vector_rhs = a.shape().len() == b.shape().len() + 1
        && !b.shape().is_empty()
        && b.shape()[0] == a.shape()[0]
        && b.shape()[1..] == a.shape()[2..];
    if !is_batched_vector_rhs {
        return None;
    }

    let mut rhs_shape = vec![b.shape()[0], 1];
    rhs_shape.extend_from_slice(&b.shape()[1..]);
    Some(rhs_shape)
}

fn matmul_preserve_trailing_batch(
    backend: &mut CpuBackend,
    lhs: &Tensor,
    rhs: &Tensor,
) -> crate::Result<Tensor> {
    let rank = lhs.shape().len();
    let batch_dims: Vec<usize> = (2..rank).collect();
    backend.dot_general(
        lhs,
        rhs,
        &DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: batch_dims.clone(),
            rhs_batch_dims: batch_dims,
        },
    )
}

pub(crate) fn reclaim_typed<T: PoolScalar>(pool: &mut BufferPool, typed: TypedTensor<T>) {
    match typed.buffer {
        Buffer::Host(data) => T::pool_release(pool, data),
        Buffer::Backend(_) => {}
        #[cfg(feature = "cubecl")]
        Buffer::Cubecl(_) => panic!("GPU tensor (Buffer::Cubecl) passed to CPU backend. Use cubecl::download_tensor() to transfer to CPU first."),
    }
}

fn zeros_like_tensor(input: &Tensor) -> Tensor {
    match input {
        Tensor::F32(t) => Tensor::F32(TypedTensor::zeros(t.shape.clone())),
        Tensor::F64(t) => Tensor::F64(TypedTensor::zeros(t.shape.clone())),
        Tensor::I64(t) => Tensor::I64(TypedTensor::zeros(t.shape.clone())),
        Tensor::C32(t) => Tensor::C32(TypedTensor::zeros(t.shape.clone())),
        Tensor::C64(t) => Tensor::C64(TypedTensor::zeros(t.shape.clone())),
    }
}

pub(crate) fn unsupported_dtype(op: &'static str, dtype: crate::DType) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: format!("unsupported dtype {dtype:?}"),
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
