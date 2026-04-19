use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use tenferro_algebra::Semiring;

use crate::backend::{SemiringBackend, TensorBackend, TensorExec};
use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::types::flat_to_multi;
use crate::validate::validate_nonsingular_u;
use crate::{Buffer, Tensor, TypedTensor};

use super::exec_session::CpuExecSession;
use super::{analytic, elementwise, gemm, indexing, linalg, reduction, structural, CpuContext};

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
    pub(crate) ctx: Arc<CpuContext>,
    pub(crate) buffers: BufferPool,
}

impl CpuBackend {
    /// Create a CPU backend using the environment-driven CPU context.
    ///
    /// # Examples
    ///
    /// ```ignore
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
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::try_new().unwrap();
    /// let _ = backend.num_threads();
    /// ```
    pub fn try_new() -> crate::Result<Self> {
        CpuContext::try_from_env().map(|ctx| Self::from_context(Arc::new(ctx)))
    }

    /// Create a CPU backend from an existing context.
    ///
    /// # Examples
    ///
    /// ```ignore
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
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::with_threads(2);
    /// assert_eq!(backend.num_threads(), 2);
    /// ```
    pub fn with_threads(num_threads: usize) -> Self {
        assert!(num_threads >= 1, "thread count must be >= 1");
        Self::from_context(Arc::new(CpuContext::with_threads(num_threads)))
    }

    /// Return the number of threads in this backend's CPU context.
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
        self.ctx.num_threads()
    }

    /// Number of retained typed host buffers currently held by this backend.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// assert_eq!(backend.buffer_pool_len(), 0);
    /// ```
    pub fn buffer_pool_len(&self) -> usize {
        self.buffers.len()
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
        Ok(self.install(|| structural::convert(input, to)))
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
        self.install(|| {
            catch_backend_panic("gather", || {
                indexing::gather(operand, start_indices, config)
            })
        })
    }

    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor> {
        self.install(|| {
            catch_backend_panic("scatter", || {
                indexing::scatter(operand, scatter_indices, updates, config)
            })
        })
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
        self.install(|| {
            catch_backend_panic("dynamic_slice", || {
                indexing::dynamic_slice(input, starts, slice_sizes)
            })
        })
    }

    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor> {
        self.install(|| indexing::try_pad(input, config))
    }

    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor> {
        self.install(|| indexing::try_concatenate(inputs, axis))
    }

    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor> {
        self.install(|| catch_backend_panic("reverse", || indexing::reverse(input, axes)))
    }

    fn cholesky(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => {
                catch_backend_panic("cholesky", || linalg::cholesky(ctx.as_ref(), buffers, t))
                    .and_then(|result| result)
                    .map(Tensor::F64)
            }
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => {
                catch_backend_panic("cholesky", || linalg::cholesky(buffers, t)).map(Tensor::F64)
            }
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => {
                catch_backend_panic("cholesky", || linalg::cholesky(ctx.as_ref(), buffers, t))
                    .and_then(|result| result)
                    .map(Tensor::C64)
            }
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
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match (a, b) {
            #[cfg(feature = "cpu-faer")]
            (Tensor::F64(a), Tensor::F64(b)) => catch_backend_panic("triangular_solve", || {
                Tensor::F64(linalg::triangular_solve(
                    ctx.as_ref(),
                    buffers,
                    a,
                    b,
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                ))
            }),
            #[cfg(feature = "cpu-blas")]
            (Tensor::F64(a), Tensor::F64(b)) => catch_backend_panic("triangular_solve", || {
                Tensor::F64(linalg::triangular_solve(
                    buffers,
                    a,
                    b,
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                ))
            }),
            #[cfg(feature = "cpu-faer")]
            (Tensor::C64(a), Tensor::C64(b)) => catch_backend_panic("triangular_solve", || {
                Tensor::C64(linalg::triangular_solve(
                    ctx.as_ref(),
                    buffers,
                    a,
                    b,
                    left_side,
                    lower,
                    transpose_a,
                    unit_diagonal,
                ))
            }),
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
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => catch_backend_panic("lu", || {
                linalg::lu(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => catch_backend_panic("lu", || {
                linalg::lu(buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => catch_backend_panic("lu", || {
                linalg::lu(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::C64)
                    .collect()
            }),
            _ => Err(unsupported_dtype("lu", input.dtype())),
        })
    }

    fn svd(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => catch_backend_panic("svd", || {
                linalg::svd(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => catch_backend_panic("svd", || {
                linalg::svd(buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => catch_backend_panic("svd", || {
                linalg::svd(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::C64)
                    .collect()
            }),
            _ => Err(unsupported_dtype("svd", input.dtype())),
        })
    }

    fn qr(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => catch_backend_panic("qr", || {
                linalg::qr(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => catch_backend_panic("qr", || {
                linalg::qr(buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => catch_backend_panic("qr", || {
                linalg::qr(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::C64)
                    .collect()
            }),
            _ => Err(unsupported_dtype("qr", input.dtype())),
        })
    }

    fn eigh(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| match input {
            #[cfg(feature = "cpu-faer")]
            Tensor::F64(t) => catch_backend_panic("eigh", || {
                linalg::eigh(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-blas")]
            Tensor::F64(t) => catch_backend_panic("eigh", || {
                linalg::eigh(buffers, t)
                    .into_iter()
                    .map(Tensor::F64)
                    .collect()
            }),
            #[cfg(feature = "cpu-faer")]
            Tensor::C64(t) => catch_backend_panic("eigh", || {
                linalg::eigh(ctx.as_ref(), buffers, t)
                    .into_iter()
                    .map(Tensor::C64)
                    .collect()
            }),
            _ => Err(unsupported_dtype("eigh", input.dtype())),
        })
    }

    fn eig(&mut self, input: &Tensor) -> crate::Result<Vec<Tensor>> {
        let ctx = Arc::clone(&self.ctx);
        self.install_with_pool(|buffers| {
            catch_backend_panic("eig", || {
                #[cfg(feature = "cpu-faer")]
                {
                    linalg::eig(ctx.as_ref(), buffers, input)
                }
                #[cfg(feature = "cpu-blas")]
                {
                    linalg::eig(buffers, input)
                }
            })
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
        Tensor::C32(t) => Tensor::C32(TypedTensor::zeros(t.shape.clone())),
        Tensor::C64(t) => Tensor::C64(TypedTensor::zeros(t.shape.clone())),
    }
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "backend panic".into()
    }
}

pub(crate) fn catch_backend_panic<R>(op: &'static str, f: impl FnOnce() -> R) -> crate::Result<R> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| crate::Error::BackendFailure {
        op,
        message: panic_payload_message(payload),
    })
}

pub(crate) fn unsupported_dtype(op: &'static str, dtype: crate::DType) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: format!("unsupported dtype {dtype:?}"),
    }
}

fn validate_axis_role_conflicts(
    op: &'static str,
    first_role: &'static str,
    first_axes: &[usize],
    second_role: &'static str,
    second_axes: &[usize],
) -> crate::Result<()> {
    for &axis in first_axes {
        if second_axes.contains(&axis) {
            return Err(crate::Error::AxisRoleConflict {
                op,
                axis,
                first_role,
                second_role,
            });
        }
    }
    Ok(())
}

fn validate_axis_list(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
        }
        if seen[axis] {
            return Err(crate::Error::DuplicateAxis { op, axis, role });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_semiring_batched_gemm_config<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<()> {
    const OP: &str = "batched_gemm";

    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: OP,
            message: "contracting dim count mismatch".into(),
        });
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(crate::Error::InvalidConfig {
            op: OP,
            message: "batch dim count mismatch".into(),
        });
    }

    let lhs_rank = lhs.shape.len();
    let rhs_rank = rhs.shape.len();
    validate_axis_list(
        OP,
        "lhs_contracting_dims",
        &config.lhs_contracting_dims,
        lhs_rank,
    )?;
    validate_axis_list(
        OP,
        "rhs_contracting_dims",
        &config.rhs_contracting_dims,
        rhs_rank,
    )?;
    validate_axis_list(OP, "lhs_batch_dims", &config.lhs_batch_dims, lhs_rank)?;
    validate_axis_list(OP, "rhs_batch_dims", &config.rhs_batch_dims, rhs_rank)?;
    validate_axis_role_conflicts(
        OP,
        "lhs_contracting_dims",
        &config.lhs_contracting_dims,
        "lhs_batch_dims",
        &config.lhs_batch_dims,
    )?;
    validate_axis_role_conflicts(
        OP,
        "rhs_contracting_dims",
        &config.rhs_contracting_dims,
        "rhs_batch_dims",
        &config.rhs_batch_dims,
    )?;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        if lhs.shape[lhs_axis] != rhs.shape[rhs_axis] {
            return Err(crate::Error::ShapeMismatch {
                op: OP,
                lhs: vec![lhs.shape[lhs_axis]],
                rhs: vec![rhs.shape[rhs_axis]],
            });
        }
    }

    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs.shape[lhs_axis] != rhs.shape[rhs_axis] {
            return Err(crate::Error::ShapeMismatch {
                op: OP,
                lhs: vec![lhs.shape[lhs_axis]],
                rhs: vec![rhs.shape[rhs_axis]],
            });
        }
    }

    Ok(())
}

impl<Alg: Semiring> SemiringBackend<Alg> for CpuBackend {
    fn batched_gemm(
        &mut self,
        lhs: &TypedTensor<Alg::Scalar>,
        rhs: &TypedTensor<Alg::Scalar>,
        config: &DotGeneralConfig,
    ) -> crate::Result<TypedTensor<Alg::Scalar>> {
        validate_semiring_batched_gemm_config(lhs, rhs, config)?;
        Ok(self.install(|| {
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
            out_shape.extend_from_slice(&lhs_free_shape);
            out_shape.extend_from_slice(&rhs_free_shape);
            out_shape.extend_from_slice(&batch_shape);

            let out_n: usize = out_shape.iter().product();
            let contract_n: usize = contract_shape.iter().product();

            let mut result = TypedTensor::zeros(out_shape.clone());
            let n_lhs_free = lhs_free_shape.len();
            let n_rhs_free = rhs_free_shape.len();

            let mut out_idx = vec![0usize; out_shape.len()];
            let mut lhs_idx = vec![0usize; lhs_rank];
            let mut rhs_idx = vec![0usize; rhs_rank];
            let mut contract_idx = vec![0usize; contract_shape.len()];

            for flat_out in 0..out_n {
                flat_to_multi(flat_out, &out_shape, &mut out_idx);

                let lhs_free_vals = &out_idx[..n_lhs_free];
                let rhs_free_vals = &out_idx[n_lhs_free..n_lhs_free + n_rhs_free];
                let batch_vals = &out_idx[n_lhs_free + n_rhs_free..];

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
        }))
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}
