use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::{RuntimeCacheControl, Tensor, TensorRead};

/// Canonical elementwise fusion plan shared between segmented execution and backends.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementwiseFusionPlan {
    pub dtype: crate::DType,
    pub n_inputs: usize,
    pub outputs: Vec<usize>,
    pub ops: Vec<ElementwiseFusionInst>,
}

/// One node in a canonical elementwise fusion plan.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementwiseFusionInst {
    pub op: ElementwiseFusionOp,
    pub inputs: Vec<usize>,
}

/// Elementwise op kinds supported by backend fusion implementations.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum ElementwiseFusionOp {
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Maximum,
    Minimum,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
}

/// Elementwise tensor operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorElementwise};
///
/// fn accepts_elementwise<B: TensorElementwise>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_elementwise(&mut backend);
/// ```
pub trait TensorElementwise {
    fn add(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor>;
    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
}

/// Analytic unary and binary tensor operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorAnalytic};
///
/// fn accepts_analytic<B: TensorAnalytic>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_analytic(&mut backend);
/// ```
pub trait TensorAnalytic {
    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor>;
}

/// Shape, layout, and dtype transformation operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorStructural};
///
/// fn accepts_structural<B: TensorStructural>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_structural(&mut backend);
/// ```
pub trait TensorStructural {
    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor>;
    fn convert(&mut self, input: &Tensor, to: crate::DType) -> crate::Result<Tensor>;
    fn extract_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn embed_diagonal(
        &mut self,
        input: &Tensor,
        axis_a: usize,
        axis_b: usize,
    ) -> crate::Result<Tensor>;
    fn tril(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
    fn triu(&mut self, input: &Tensor, k: i64) -> crate::Result<Tensor>;
}

/// Reduction operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorReduction};
///
/// fn accepts_reduction<B: TensorReduction>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_reduction(&mut backend);
/// ```
pub trait TensorReduction {
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
}

/// Dot-general operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorDot};
///
/// fn accepts_dot<B: TensorDot>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_dot(&mut backend);
/// ```
pub trait TensorDot: TensorElementwise {
    fn dot_general(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor>;

    #[doc(hidden)]
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.dot_general(lhs, rhs, config),
            _ => {
                let lhs = lhs.to_tensor();
                let rhs = rhs.to_tensor();
                self.dot_general(&lhs, &rhs, config)
            }
        }
    }

    #[doc(hidden)]
    fn dot_general_with_conj(
        &mut self,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        if !lhs_conj && !rhs_conj {
            return self.dot_general(lhs, rhs, config);
        }

        let lhs_tmp;
        let lhs_ref = if lhs_conj {
            lhs_tmp = self.conj(lhs)?;
            &lhs_tmp
        } else {
            lhs
        };
        let rhs_tmp;
        let rhs_ref = if rhs_conj {
            rhs_tmp = self.conj(rhs)?;
            &rhs_tmp
        } else {
            rhs
        };
        self.dot_general(lhs_ref, rhs_ref, config)
    }
}

/// Session-scoped cached dot-general operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, BackendSession, BackendSessionHost};
///
/// let mut backend = CpuBackend::new();
/// backend.with_backend_session(|session| {
///     fn accepts_session_dot<S: BackendSession + ?Sized>(_session: &mut S) {}
///     accepts_session_dot(session);
/// });
/// ```
pub trait SessionCachedDot: TensorDot {
    #[doc(hidden)]
    fn dot_general_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general(lhs, rhs, config)
    }

    #[doc(hidden)]
    fn dot_general_with_conj_cached(
        &mut self,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_with_conj(lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

/// Indexing, slicing, and padding operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorIndexing};
///
/// fn accepts_indexing<B: TensorIndexing>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_indexing(&mut backend);
/// ```
pub trait TensorIndexing {
    fn gather(
        &mut self,
        operand: &Tensor,
        start_indices: &Tensor,
        config: &GatherConfig,
    ) -> crate::Result<Tensor>;
    fn scatter(
        &mut self,
        operand: &Tensor,
        scatter_indices: &Tensor,
        updates: &Tensor,
        config: &ScatterConfig,
    ) -> crate::Result<Tensor>;
    fn slice(&mut self, input: &Tensor, config: &SliceConfig) -> crate::Result<Tensor>;
    fn dynamic_slice(
        &mut self,
        input: &Tensor,
        starts: &Tensor,
        slice_sizes: &[usize],
    ) -> crate::Result<Tensor>;
    fn dynamic_update_slice(
        &mut self,
        operand: &Tensor,
        update: &Tensor,
        starts: &Tensor,
    ) -> crate::Result<Tensor>;
    fn pad(&mut self, input: &Tensor, config: &PadConfig) -> crate::Result<Tensor>;
    fn concatenate(&mut self, inputs: &[&Tensor], axis: usize) -> crate::Result<Tensor>;
    fn reverse(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;
}

/// Optional elementwise fusion execution.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorFusion};
///
/// fn accepts_fusion<B: TensorFusion>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_fusion(&mut backend);
/// ```
pub trait TensorFusion {
    #[doc(hidden)]
    fn execute_elementwise_fusion(
        &mut self,
        _inputs: &[&Tensor],
        _plan: &ElementwiseFusionPlan,
    ) -> crate::Result<Option<Vec<Tensor>>> {
        Ok(None)
    }
}

/// Backend buffer lifecycle operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorBuffer};
///
/// fn accepts_buffer<B: TensorBuffer>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_buffer(&mut backend);
/// ```
pub trait TensorBuffer {
    fn reclaim_buffer(&mut self, _tensor: Tensor) {}
}

/// Device transfer operations on backend boundaries.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorDeviceTransfer};
///
/// fn accepts_transfer<B: TensorDeviceTransfer>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_transfer(&mut backend);
/// ```
pub trait TensorDeviceTransfer {
    fn download_to_host(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        Ok(tensor.clone())
    }

    fn upload_host_tensor(&mut self, tensor: &Tensor) -> crate::Result<Tensor> {
        Ok(tensor.clone())
    }
}

/// Runtime cache associated with a backend.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, BackendRuntimeCache};
///
/// fn accepts_runtime_cache<B: BackendRuntimeCache>(_backend: &B) {}
/// let backend = CpuBackend::new();
/// accepts_runtime_cache(&backend);
/// ```
pub trait BackendRuntimeCache {
    #[doc(hidden)]
    type RuntimeCache: RuntimeCacheControl;
}

/// Backend-owned cached dot-general operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, BackendCachedDot};
///
/// fn accepts_backend_cached_dot<B: BackendCachedDot>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_backend_cached_dot(&mut backend);
/// ```
pub trait BackendCachedDot: BackendRuntimeCache + TensorDot {
    #[doc(hidden)]
    fn dot_general_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> crate::Result<Tensor> {
        self.dot_general(lhs, rhs, config)
    }

    #[doc(hidden)]
    fn dot_general_with_conj_cached(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        _cache_slot: Option<usize>,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
        lhs_conj: bool,
        rhs_conj: bool,
    ) -> crate::Result<Tensor> {
        self.dot_general_with_conj(lhs, rhs, config, lhs_conj, rhs_conj)
    }
}

/// Backend execution-session entry points.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, BackendSessionHost};
///
/// fn accepts_session_host<B: BackendSessionHost>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_session_host(&mut backend);
/// ```
pub trait BackendSessionHost: BackendRuntimeCache {
    fn with_backend_session<R>(&mut self, f: impl FnOnce(&mut dyn BackendSession) -> R) -> R
    where
        Self: TensorBackend + Sized,
    {
        default_backend_session(self, f)
    }

    #[doc(hidden)]
    fn with_backend_session_cached<R>(
        &mut self,
        _cache: &mut Self::RuntimeCache,
        f: impl FnOnce(&mut dyn BackendSession) -> R,
    ) -> R
    where
        Self: TensorBackend + Sized,
    {
        self.with_backend_session(f)
    }
}

/// Operation capabilities shared by backends and backend sessions.
#[doc(hidden)]
pub trait TensorBackendOps:
    TensorElementwise
    + TensorAnalytic
    + TensorStructural
    + TensorReduction
    + TensorIndexing
    + TensorDot
    + TensorFusion
    + TensorBuffer
{
}

impl<T> TensorBackendOps for T where
    T: TensorElementwise
        + TensorAnalytic
        + TensorStructural
        + TensorReduction
        + TensorIndexing
        + TensorDot
        + TensorFusion
        + TensorBuffer
        + ?Sized
{
}

/// Execution session surface for dense tensor backends.
///
/// All operations run within a backend-owned execution scope such as a CPU
/// thread policy or a GPU stream. Individual ops must not try to re-enter that
/// scope.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, BackendSessionHost, Tensor, TypedTensor};
///
/// let mut backend = CpuBackend::new();
/// let a = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
/// let b = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![3.0, 4.0]));
/// let sum = backend
///     .with_backend_session(|exec| exec.add(&a, &b))
///     .unwrap();
/// assert_eq!(sum.shape(), &[2]);
/// ```
pub trait BackendSession: TensorBackendOps + SessionCachedDot {}

impl<T> BackendSession for T where T: TensorBackendOps + SessionCachedDot + ?Sized {}

/// Standard runtime backend over dynamic [`Tensor`] values.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorBackend};
///
/// fn accepts_backend<B: TensorBackend>(_backend: &mut B) {}
/// let mut backend = CpuBackend::new();
/// accepts_backend(&mut backend);
/// ```
pub trait TensorBackend:
    BackendRuntimeCache
    + TensorBackendOps
    + BackendCachedDot
    + TensorDeviceTransfer
    + BackendSessionHost
{
}

impl<T> SessionCachedDot for T where T: TensorBackend + ?Sized {}

/// Run a closure using the backend itself as a default execution session.
///
/// This is suitable for backends whose individual ops already manage their own
/// execution context.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, default_backend_session};
///
/// let mut backend = CpuBackend::new();
/// let _ = default_backend_session(&mut backend, |_exec| 1usize);
/// ```
pub fn default_backend_session<B: TensorBackend, R>(
    backend: &mut B,
    f: impl FnOnce(&mut dyn BackendSession) -> R,
) -> R {
    f(backend)
}
