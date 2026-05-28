use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::types::{TensorRank, TypedTensor, TypedTensorView, TypedTensorViewMut};
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

tenferro_core_ops::define_elementwise_fusion_op!();

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

    /// Elementwise addition accepting either owned tensors or borrowed views.
    ///
    /// Backends that implement this method must not silently move data across
    /// devices. A backend that cannot consume views should return an explicit
    /// backend error rather than materializing or transferring implicitly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     cpu::CpuBackend, TensorElementwise, TensorRead, TensorView, TypedTensor,
    /// };
    ///
    /// let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]);
    /// let view = TensorView::F64(tensor.as_view());
    /// let mut backend = CpuBackend::new();
    /// let out = backend.add_read(
    ///     TensorRead::from_view(view.clone()),
    ///     TensorRead::from_view(view),
    /// )?;
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(lhs), Some(rhs)) => self.add(lhs, rhs),
            _ => Err(crate::Error::backend_failure(
                "add",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

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

    /// Sum elements across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     cpu::CpuBackend, TensorRead, TensorReduction, TensorView, TypedTensor,
    /// };
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// let mut backend = CpuBackend::new();
    /// let out = backend.reduce_sum_read(
    ///     TensorRead::from_view(TensorView::F64(input.as_view())),
    ///     &[0],
    /// )?;
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[3.0, 7.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn reduce_sum_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_sum(input, axes),
            None => Err(crate::Error::backend_failure(
                "reduce_sum",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

    fn reduce_prod(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Multiply elements across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     cpu::CpuBackend, TensorRead, TensorReduction, TensorView, TypedTensor,
    /// };
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// let mut backend = CpuBackend::new();
    /// let out = backend.reduce_prod_read(
    ///     TensorRead::from_view(TensorView::F64(input.as_view())),
    ///     &[0],
    /// )?;
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 12.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_prod(input, axes),
            None => Err(crate::Error::backend_failure(
                "reduce_prod",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

    fn reduce_max(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Take maximum values across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     cpu::CpuBackend, TensorRead, TensorReduction, TensorView, TypedTensor,
    /// };
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// let mut backend = CpuBackend::new();
    /// let out = backend.reduce_max_read(
    ///     TensorRead::from_view(TensorView::F64(input.as_view())),
    ///     &[0],
    /// )?;
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_max(input, axes),
            None => Err(crate::Error::backend_failure(
                "reduce_max",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }

    fn reduce_min(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Take minimum values across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{
    ///     cpu::CpuBackend, TensorRead, TensorReduction, TensorView, TypedTensor,
    /// };
    ///
    /// let input = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// let mut backend = CpuBackend::new();
    /// let out = backend.reduce_min_read(
    ///     TensorRead::from_view(TensorView::F64(input.as_view())),
    ///     &[0],
    /// )?;
    /// assert_eq!(out.as_slice::<f64>().unwrap(), &[1.0, 3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> crate::Result<Tensor> {
        match input.as_tensor() {
            Some(input) => self.reduce_min(input, axes),
            None => Err(crate::Error::backend_failure(
                "reduce_min",
                "backend does not accept borrowed tensor views at this execution boundary",
            )),
        }
    }
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

/// Backend-owned canonicalization for typed tensor views.
///
/// Implementations must preserve the input placement family. CPU backends
/// canonicalize host views through explicit host copies and reject backend
/// buffers with a diagnostic that asks the caller to download first. GPU
/// backends canonicalize GPU-resident views on the same device and reject host
/// buffers with an upload hint.
///
/// This trait is intentionally separate from [`BackendSession`] so generic
/// typed methods do not change the object-safety contract of `dyn BackendSession`.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{cpu::CpuBackend, TensorViewCanonicalization, TypedTensor};
///
/// let mut backend = CpuBackend::new();
/// let tensor = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]);
/// let compact = backend.to_contiguous(&tensor.as_view())?;
/// assert_eq!(compact.as_slice(), &[1, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorViewCanonicalization<T: Clone + 'static, R: TensorRank> {
    fn to_contiguous(
        &mut self,
        view: &TypedTensorView<'_, T, R>,
    ) -> crate::Result<TypedTensor<T, R>>;

    fn copy_from_contiguous(
        &mut self,
        src: &TypedTensor<T, R>,
        dst: &mut TypedTensorViewMut<'_, T, R>,
    ) -> crate::Result<()>;
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
