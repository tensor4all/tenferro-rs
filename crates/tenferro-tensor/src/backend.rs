use crate::config::{
    CompareDir, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
};
use crate::types::{TensorRank, TypedTensor, TypedTensorView, TypedTensorViewMut};
use crate::{RuntimeCacheControl, Tensor, TensorRead};

fn read_boundary_error(op: &'static str) -> crate::Error {
    crate::Error::backend_failure(
        op,
        "backend does not accept borrowed tensor views at this execution boundary",
    )
}

fn read_tensor<'a>(op: &'static str, input: TensorRead<'a>) -> crate::Result<&'a Tensor> {
    input.as_tensor().ok_or_else(|| read_boundary_error(op))
}

/// Canonical elementwise fusion plan shared between segmented execution and backends.
#[doc(hidden)]
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ElementwiseFusionPlan {
    pub dtype: crate::DType,
    pub input_count: usize,
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
/// use tenferro_tensor::TensorElementwise;
///
/// fn accepts_elementwise<B: TensorElementwise>(_backend: &mut B) {}
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
    /// use tenferro_tensor::{Tensor, TensorElementwise, TensorRead};
    ///
    /// fn add_owned<B: TensorElementwise>(
    ///     backend: &mut B,
    ///     lhs: &Tensor,
    ///     rhs: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.add_read(TensorRead::from_tensor(lhs), TensorRead::from_tensor(rhs))
    /// }
    /// ```
    fn add_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.add(read_tensor("add", lhs)?, read_tensor("add", rhs)?)
    }

    fn mul(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn mul_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.mul(read_tensor("mul", lhs)?, read_tensor("mul", rhs)?)
    }

    fn neg(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn neg_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.neg(read_tensor("neg", input)?)
    }

    fn conj(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn conj_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.conj(read_tensor("conj", input)?)
    }

    fn div(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn div_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.div(read_tensor("div", lhs)?, read_tensor("div", rhs)?)
    }

    fn abs(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn abs_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.abs(read_tensor("abs", input)?)
    }

    fn sign(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sign_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sign(read_tensor("sign", input)?)
    }

    fn maximum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn maximum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.maximum(read_tensor("maximum", lhs)?, read_tensor("maximum", rhs)?)
    }

    fn minimum(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.minimum(read_tensor("minimum", lhs)?, read_tensor("minimum", rhs)?)
    }

    fn compare(&mut self, lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> crate::Result<Tensor>;
    fn compare_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        dir: &CompareDir,
    ) -> crate::Result<Tensor> {
        self.compare(
            read_tensor("compare", lhs)?,
            read_tensor("compare", rhs)?,
            dir,
        )
    }

    fn select(
        &mut self,
        pred: &Tensor,
        on_true: &Tensor,
        on_false: &Tensor,
    ) -> crate::Result<Tensor>;
    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.select(
            read_tensor("select", pred)?,
            read_tensor("select", on_true)?,
            read_tensor("select", on_false)?,
        )
    }

    fn clamp(&mut self, input: &Tensor, lower: &Tensor, upper: &Tensor) -> crate::Result<Tensor>;
    fn clamp_read(
        &mut self,
        input: TensorRead<'_>,
        lower: TensorRead<'_>,
        upper: TensorRead<'_>,
    ) -> crate::Result<Tensor> {
        self.clamp(
            read_tensor("clamp", input)?,
            read_tensor("clamp", lower)?,
            read_tensor("clamp", upper)?,
        )
    }
}

/// Analytic unary and binary tensor operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorAnalytic;
///
/// fn accepts_analytic<B: TensorAnalytic>(_backend: &mut B) {}
/// ```
pub trait TensorAnalytic {
    fn exp(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn exp_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.exp(read_tensor("exp", input)?)
    }

    fn log(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.log(read_tensor("log", input)?)
    }

    fn sin(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sin_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sin(read_tensor("sin", input)?)
    }

    fn cos(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn cos_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.cos(read_tensor("cos", input)?)
    }

    fn tanh(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn tanh_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.tanh(read_tensor("tanh", input)?)
    }

    fn sqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn sqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.sqrt(read_tensor("sqrt", input)?)
    }

    fn rsqrt(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn rsqrt_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.rsqrt(read_tensor("rsqrt", input)?)
    }

    fn pow(&mut self, lhs: &Tensor, rhs: &Tensor) -> crate::Result<Tensor>;
    fn pow_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> crate::Result<Tensor> {
        self.pow(read_tensor("pow", lhs)?, read_tensor("pow", rhs)?)
    }

    fn expm1(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn expm1_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.expm1(read_tensor("expm1", input)?)
    }

    fn log1p(&mut self, input: &Tensor) -> crate::Result<Tensor>;
    fn log1p_read(&mut self, input: TensorRead<'_>) -> crate::Result<Tensor> {
        self.log1p(read_tensor("log1p", input)?)
    }
}

/// Shape, layout, and dtype transformation operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorStructural;
///
/// fn accepts_structural<B: TensorStructural>(_backend: &mut B) {}
/// ```
pub trait TensorStructural {
    fn transpose(&mut self, input: &Tensor, perm: &[usize]) -> crate::Result<Tensor>;
    fn transpose_read(&mut self, input: TensorRead<'_>, perm: &[usize]) -> crate::Result<Tensor> {
        self.transpose(read_tensor("transpose", input)?, perm)
    }

    fn reshape(&mut self, input: &Tensor, shape: &[usize]) -> crate::Result<Tensor>;
    fn reshape_read(&mut self, input: TensorRead<'_>, shape: &[usize]) -> crate::Result<Tensor> {
        self.reshape(read_tensor("reshape", input)?, shape)
    }

    fn broadcast_in_dim(
        &mut self,
        input: &Tensor,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor>;
    fn broadcast_in_dim_read(
        &mut self,
        input: TensorRead<'_>,
        shape: &[usize],
        dims: &[usize],
    ) -> crate::Result<Tensor> {
        self.broadcast_in_dim(read_tensor("broadcast_in_dim", input)?, shape, dims)
    }

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
/// use tenferro_tensor::TensorReduction;
///
/// fn accepts_reduction<B: TensorReduction>(_backend: &mut B) {}
/// ```
pub trait TensorReduction {
    fn reduce_sum(&mut self, input: &Tensor, axes: &[usize]) -> crate::Result<Tensor>;

    /// Sum elements across axes from an owned tensor or borrowed view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn sum_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_sum_read(TensorRead::from_tensor(input), &[0])
    /// }
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
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn prod_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_prod_read(TensorRead::from_tensor(input), &[0])
    /// }
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
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn max_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_max_read(TensorRead::from_tensor(input), &[0])
    /// }
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
    /// use tenferro_tensor::{Tensor, TensorRead, TensorReduction};
    ///
    /// fn min_owned<B: TensorReduction>(
    ///     backend: &mut B,
    ///     input: &Tensor,
    /// ) -> tenferro_tensor::Result<Tensor> {
    ///     backend.reduce_min_read(TensorRead::from_tensor(input), &[0])
    /// }
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
/// use tenferro_tensor::TensorDot;
///
/// fn accepts_dot<B: TensorDot>(_backend: &mut B) {}
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
/// use tenferro_tensor::BackendSession;
///
/// fn accepts_session_dot<S: BackendSession + ?Sized>(_session: &mut S) {}
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

    // Mirrors the dot-general signature plus runtime-cache metadata.
    #[allow(clippy::too_many_arguments)]
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
/// use tenferro_tensor::TensorIndexing;
///
/// fn accepts_indexing<B: TensorIndexing>(_backend: &mut B) {}
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
/// use tenferro_tensor::{DynRank, TensorViewCanonicalization, TypedTensor};
///
/// fn compact_i32<B: TensorViewCanonicalization<i32, DynRank>>(
///     backend: &mut B,
///     tensor: &TypedTensor<i32>,
/// ) -> tenferro_tensor::Result<TypedTensor<i32>> {
///     backend.to_contiguous(&tensor.as_view())
/// }
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
/// use tenferro_tensor::TensorFusion;
///
/// fn accepts_fusion<B: TensorFusion>(_backend: &mut B) {}
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
/// use tenferro_tensor::TensorBuffer;
///
/// fn accepts_buffer<B: TensorBuffer>(_backend: &mut B) {}
/// ```
pub trait TensorBuffer {
    fn reclaim_buffer(&mut self, _tensor: Tensor) {}
}

/// Device transfer operations on backend boundaries.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorDeviceTransfer;
///
/// fn accepts_transfer<B: TensorDeviceTransfer>(_backend: &mut B) {}
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
/// use tenferro_tensor::BackendRuntimeCache;
///
/// fn accepts_runtime_cache<B: BackendRuntimeCache>(_backend: &B) {}
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
/// use tenferro_tensor::BackendCachedDot;
///
/// fn accepts_backend_cached_dot<B: BackendCachedDot>(_backend: &mut B) {}
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

    // Mirrors the dot-general signature plus runtime-cache metadata.
    #[allow(clippy::too_many_arguments)]
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
/// use tenferro_tensor::BackendSessionHost;
///
/// fn accepts_session_host<B: BackendSessionHost>(_backend: &mut B) {}
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
/// use tenferro_tensor::{BackendSessionHost, Tensor, TypedTensor};
///
/// fn add_in_session<B: BackendSessionHost>(
///     backend: &mut B,
///     a: &Tensor,
///     b: &Tensor,
/// ) -> tenferro_tensor::Result<Tensor>
/// where
///     B: tenferro_tensor::TensorBackend,
/// {
///     backend.with_backend_session(|exec| exec.add(a, b))
/// }
/// ```
pub trait BackendSession: TensorBackendOps + SessionCachedDot {}

impl<T> BackendSession for T where T: TensorBackendOps + SessionCachedDot + ?Sized {}

/// Standard runtime backend over dynamic [`Tensor`] values.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorBackend;
///
/// fn accepts_backend<B: TensorBackend>(_backend: &mut B) {}
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
/// use tenferro_tensor::{default_backend_session, TensorBackend};
///
/// fn run_with_default_session<B: TensorBackend>(backend: &mut B) -> usize {
///     default_backend_session(backend, |_exec| 1usize)
/// }
/// ```
pub fn default_backend_session<B: TensorBackend, R>(
    backend: &mut B,
    f: impl FnOnce(&mut dyn BackendSession) -> R,
) -> R {
    f(backend)
}
