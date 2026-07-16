use std::fmt;
#[cfg(feature = "cpu-faer")]
use std::mem::size_of;
#[cfg(feature = "cpu-faer")]
use std::sync::Arc;

#[cfg(feature = "cpu-faer")]
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
#[cfg(feature = "cpu-faer")]
use faer::{diag::DiagMut, MatMut, MatRef};
#[cfg(feature = "cpu-faer")]
use num_complex::{Complex32, Complex64};
#[cfg(feature = "cpu-faer")]
use tenferro_cpu::CpuLinalgBinding;
use tenferro_cpu::{CpuBackend, CpuBackendKind};
#[cfg(feature = "cpu-faer")]
use tenferro_tensor::{
    validate::checked_shape_product, MemoryKind, Tensor, TensorView, TensorViewMut, TypedTensor,
    TypedTensorView, TypedTensorViewMut,
};
use tenferro_tensor::{BackendId, DType, Error, Placement, TensorRead, TensorWrite};

use crate::prepared_factorization::{
    private::PreparedFactorizationDispatch, PreparedFactorizationBackendExt,
    PreparedFactorizationSession, PreparedFactorizationSessionInner,
};
use crate::{backend::LinalgBackend, SvdOptions};
#[cfg(feature = "cpu-faer")]
use crate::{
    extension::{
        canonicalize_svd_gauge_c32, canonicalize_svd_gauge_c64, canonicalize_svd_gauge_f32,
        canonicalize_svd_gauge_f64, validate_derivative_eps,
    },
    SvdGauge,
};

const PREPARE_OP: &str = "prepare_svd";
const EXECUTE_OP: &str = "PreparedSvd::execute_into";
const PREPARED_CAPABILITY: &str = "prepared compact SVD";
#[cfg(feature = "cpu-faer")]
const BINDING_CAPABILITY: &str = "prepared SVD backend/context binding";
#[cfg(feature = "cpu-faer")]
const DESTINATION_CAPABILITY: &str = "compact column-major prepared SVD destination";

/// Exact metadata for one prepared SVD output.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
/// assert_eq!(plan.output_specs().u().shape(), &[3, 2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvdOutputSpec {
    shape: Vec<usize>,
    dtype: DType,
    placement: Placement,
}

impl SvdOutputSpec {
    /// Return the exact output shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::DType;
    /// # let mut backend = CpuBackend::new();
    /// # let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().u().shape(), &[3, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the exact output dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::DType;
    /// # let mut backend = CpuBackend::new();
    /// # let plan = backend.prepare_svd([2, 2], DType::C64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().s().dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the default host placement expected for a newly allocated output.
    ///
    /// Pinned host destinations are also accepted by execution.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::{DType, MemoryKind};
    /// # let mut backend = CpuBackend::new();
    /// # let plan = backend.prepare_svd([2, 2], DType::F64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().u().placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }
}

/// Output metadata for prepared compact SVD.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([2, 4], DType::C64, SvdOptions::default())?;
/// let specs = plan.output_specs();
/// assert_eq!(specs.s().dtype(), DType::F64);
/// assert_eq!(specs.vt().shape(), &[2, 4]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SvdOutputSpecs {
    u: SvdOutputSpec,
    s: SvdOutputSpec,
    vt: SvdOutputSpec,
}

impl SvdOutputSpecs {
    /// Return the left-singular-vector output specification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::DType;
    /// # let mut backend = CpuBackend::new();
    /// # let plan = backend.prepare_svd([4, 2], DType::F64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().u().shape(), &[4, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn u(&self) -> &SvdOutputSpec {
        &self.u
    }

    /// Return the singular-value output specification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::DType;
    /// # let mut backend = CpuBackend::new();
    /// # let plan = backend.prepare_svd([4, 2], DType::F64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().s().shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn s(&self) -> &SvdOutputSpec {
        &self.s
    }

    /// Return the conjugate-transposed right-singular-vector output specification.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::DType;
    /// # let mut backend = CpuBackend::new();
    /// # let plan = backend.prepare_svd([2, 4], DType::F64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().vt().shape(), &[2, 4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn vt(&self) -> &SvdOutputSpec {
        &self.vt
    }
}

/// Caller-provided writable outputs for compact SVD.
///
/// # Examples
///
/// ```rust
/// use tenferro_linalg::SvdOutputWrites;
/// use tenferro_tensor::{Tensor, TensorWrite};
///
/// let mut u = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
/// let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
/// let writes = SvdOutputWrites::new(
///     TensorWrite::from_tensor(&mut u),
///     TensorWrite::from_tensor(&mut s),
///     TensorWrite::from_tensor(&mut vt),
/// );
/// assert_eq!(writes.u().shape(), &[1, 1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct SvdOutputWrites<'a> {
    pub(crate) u: TensorWrite<'a>,
    pub(crate) s: TensorWrite<'a>,
    pub(crate) vt: TensorWrite<'a>,
}

impl<'a> SvdOutputWrites<'a> {
    /// Bundle caller-owned `U`, `S`, and `Vt` destinations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_linalg::SvdOutputWrites;
    /// use tenferro_tensor::{Tensor, TensorWrite};
    /// let mut u = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// let writes = SvdOutputWrites::new(
    ///     TensorWrite::from_tensor(&mut u),
    ///     TensorWrite::from_tensor(&mut s),
    ///     TensorWrite::from_tensor(&mut vt),
    /// );
    /// assert_eq!(writes.s().shape(), &[1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn new(u: TensorWrite<'a>, s: TensorWrite<'a>, vt: TensorWrite<'a>) -> Self {
        Self { u, s, vt }
    }

    /// Inspect the `U` destination without mutating it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_linalg::SvdOutputWrites;
    /// # use tenferro_tensor::{Tensor, TensorWrite};
    /// # let mut u = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    /// # let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// # let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// # let writes = SvdOutputWrites::new(TensorWrite::from_tensor(&mut u), TensorWrite::from_tensor(&mut s), TensorWrite::from_tensor(&mut vt));
    /// assert_eq!(writes.u().shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn u(&self) -> &TensorWrite<'a> {
        &self.u
    }

    /// Inspect the `S` destination without mutating it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_linalg::SvdOutputWrites;
    /// # use tenferro_tensor::{Tensor, TensorWrite};
    /// # let mut u = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    /// # let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// # let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// # let writes = SvdOutputWrites::new(TensorWrite::from_tensor(&mut u), TensorWrite::from_tensor(&mut s), TensorWrite::from_tensor(&mut vt));
    /// assert_eq!(writes.s().shape(), &[1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn s(&self) -> &TensorWrite<'a> {
        &self.s
    }

    /// Inspect the `Vt` destination without mutating it.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_linalg::SvdOutputWrites;
    /// # use tenferro_tensor::{Tensor, TensorWrite};
    /// # let mut u = Tensor::from_vec_col_major(vec![2, 1], vec![0.0_f64; 2])?;
    /// # let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// # let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// # let writes = SvdOutputWrites::new(TensorWrite::from_tensor(&mut u), TensorWrite::from_tensor(&mut s), TensorWrite::from_tensor(&mut vt));
    /// assert_eq!(writes.vt().shape(), &[1, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn vt(&self) -> &TensorWrite<'a> {
        &self.vt
    }
}

/// Immutable prepared compact-SVD plan.
///
/// A plan binds shape, dtype, provider, execution context, and options. Allocate
/// one workspace per concurrent execution lane.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions, SvdOutputWrites};
/// use tenferro_tensor::{DType, Tensor, TensorRead, TensorWrite};
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([2, 2], DType::F64, SvdOptions::default())?;
/// let mut workspace = plan.allocate_workspace(&mut backend)?;
/// let input = Tensor::from_vec_col_major(
///     vec![2, 2],
///     vec![3.0_f64, 0.0, 0.0, 2.0],
/// )?;
/// let mut u = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
/// let mut s = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2])?;
/// let mut vt = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4])?;
/// for _ in 0..2 {
///     plan.execute_into(
///         &mut backend,
///         &mut workspace,
///         TensorRead::from_tensor(&input),
///         SvdOutputWrites::new(
///             TensorWrite::from_tensor(&mut u),
///             TensorWrite::from_tensor(&mut s),
///             TensorWrite::from_tensor(&mut vt),
///         ),
///     )?;
/// }
/// assert_eq!(s.as_slice::<f64>()?, &[3.0, 2.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct PreparedSvd {
    shape: [usize; 2],
    dtype: DType,
    options: SvdOptions,
    specs: SvdOutputSpecs,
    #[cfg(feature = "cpu-faer")]
    binding: CpuFaerBinding,
    #[cfg(feature = "cpu-faer")]
    plan_token: Arc<()>,
    #[cfg(feature = "cpu-faer")]
    provider: FaerPlan,
}

impl fmt::Debug for PreparedSvd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("PreparedSvd");
        debug
            .field("operation", &"compact_svd")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("options", &self.options)
            .field("provider", &"faer")
            .field("device", &"cpu");
        #[cfg(feature = "cpu-faer")]
        debug
            .field("context", &self.binding)
            .field("plan_retained_bytes", &self.retained_bytes())
            .field(
                "workspace_required_bytes",
                &self.provider.workspace_required_bytes(self.shape),
            );
        debug.finish_non_exhaustive()
    }
}

impl PreparedSvd {
    /// Return the exact provider-private heap bytes currently retained by this plan.
    ///
    /// This read-only snapshot does not allocate or mutate the plan. It excludes
    /// inline Rust object size, backend binding and identity-token metadata,
    /// shared backend context, workspaces, and outputs. Values are provider
    /// representations, so no ordering is promised across shapes, dtypes,
    /// options, or providers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// use tenferro_tensor::DType;
    ///
    /// let mut backend = CpuBackend::new();
    /// let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
    /// let plan_bytes = plan.retained_bytes();
    /// assert_eq!(plan_bytes, plan.retained_bytes());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn retained_bytes(&self) -> usize {
        #[cfg(feature = "cpu-faer")]
        {
            self.provider.retained_bytes()
        }
        #[cfg(not(feature = "cpu-faer"))]
        {
            0
        }
    }

    /// Return exact `U`, `S`, and `Vt` output metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::DType;
    /// # let mut backend = CpuBackend::new();
    /// let plan = backend.prepare_svd([5, 3], DType::F64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().s().shape(), &[3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn output_specs(&self) -> &SvdOutputSpecs {
        &self.specs
    }

    /// Allocate one opaque workspace bound to this plan and backend context.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tenferro_cpu::CpuBackend;
    /// # use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// # use tenferro_tensor::DType;
    /// let mut backend = CpuBackend::new();
    /// let plan = backend.prepare_svd([2, 2], DType::F64, SvdOptions::default())?;
    /// let workspace = plan.allocate_workspace(&mut backend)?;
    /// assert!(format!("{workspace:?}").contains("SvdWorkspace"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn allocate_workspace<B: PreparedSvdBackendExt>(
        &self,
        backend: &mut B,
    ) -> tenferro_tensor::Result<SvdWorkspace> {
        private::PreparedSvdDispatch::allocate_svd_workspace_impl(backend, self)
    }

    /// Execute into caller-owned outputs, reusing the supplied workspace.
    ///
    /// All metadata and overlap checks complete before the first output write.
    /// Once the provider call starts, a numerical provider failure may leave
    /// destinations partially overwritten.
    ///
    /// `TensorRead` and `TensorWrite` descriptor construction is caller-owned
    /// setup evaluated before this method is entered. In particular, creating
    /// dynamic-rank shape and stride metadata may allocate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions, SvdOutputWrites};
    /// use tenferro_tensor::{DType, Tensor, TensorRead, TensorWrite};
    /// let mut backend = CpuBackend::new();
    /// let plan = backend.prepare_svd([1, 1], DType::F64, SvdOptions::default())?;
    /// let mut workspace = plan.allocate_workspace(&mut backend)?;
    /// let input = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64])?;
    /// let mut u = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// plan.execute_into(
    ///     &mut backend,
    ///     &mut workspace,
    ///     TensorRead::from_tensor(&input),
    ///     SvdOutputWrites::new(
    ///         TensorWrite::from_tensor(&mut u),
    ///         TensorWrite::from_tensor(&mut s),
    ///         TensorWrite::from_tensor(&mut vt),
    ///     ),
    /// )?;
    /// assert_eq!(s.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn execute_into<B: PreparedSvdBackendExt>(
        &self,
        backend: &mut B,
        workspace: &mut SvdWorkspace,
        input: TensorRead<'_>,
        outputs: SvdOutputWrites<'_>,
    ) -> tenferro_tensor::Result<()> {
        backend.with_prepared_factorization_session(move |session| {
            self.execute_into_session(session, workspace, input, outputs)
        })
    }

    /// Execute inside an already-entered prepared factorization session.
    ///
    /// This leaf operation does not reacquire backend resources or re-enter a
    /// backend worker pool. All metadata and overlap checks complete before
    /// the first output write.
    ///
    /// The prepared-leaf allocation boundary begins with already-constructed
    /// `TensorRead` and `TensorWrite` descriptors. Creating dynamic-rank view
    /// descriptors is caller setup and may allocate before function entry.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{
    ///     PreparedFactorizationBackendExt, PreparedSvdBackendExt, SvdOptions,
    ///     SvdOutputWrites,
    /// };
    /// use tenferro_tensor::{DType, Tensor, TensorRead, TensorWrite};
    /// let mut backend = CpuBackend::new();
    /// let plan = backend.prepare_svd([1, 1], DType::F64, SvdOptions::default())?;
    /// let mut workspace = plan.allocate_workspace(&mut backend)?;
    /// let input = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64])?;
    /// let mut u = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// let mut s = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
    /// let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![0.0_f64])?;
    /// backend.with_prepared_factorization_session(|session| {
    ///     plan.execute_into_session(
    ///         session,
    ///         &mut workspace,
    ///         TensorRead::from_tensor(&input),
    ///         SvdOutputWrites::new(
    ///             TensorWrite::from_tensor(&mut u),
    ///             TensorWrite::from_tensor(&mut s),
    ///             TensorWrite::from_tensor(&mut vt),
    ///         ),
    ///     )
    /// })?;
    /// assert_eq!(s.as_slice::<f64>()?, &[2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn execute_into_session(
        &self,
        session: &mut PreparedFactorizationSession<'_>,
        workspace: &mut SvdWorkspace,
        input: TensorRead<'_>,
        outputs: SvdOutputWrites<'_>,
    ) -> tenferro_tensor::Result<()> {
        match &mut session.inner {
            PreparedFactorizationSessionInner::Cpu(cpu_session) => {
                cpu::execute_prepared_svd_into_session(cpu_session, self, workspace, input, outputs)
            }
        }
    }
}

/// Opaque caller-owned mutable workspace for a [`PreparedSvd`].
///
/// The workspace is intentionally not `Clone`; pass it as `&mut` to enforce
/// exclusive use and allocate another workspace for another concurrency lane.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([2, 2], DType::F64, SvdOptions::default())?;
/// let workspace = plan.allocate_workspace(&mut backend)?;
/// assert!(format!("{workspace:?}").contains("workspace_retained_bytes"));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct SvdWorkspace {
    #[cfg(feature = "cpu-faer")]
    binding: CpuFaerBinding,
    shape: [usize; 2],
    dtype: DType,
    #[cfg(feature = "cpu-faer")]
    plan_token: Arc<()>,
    #[cfg(feature = "cpu-faer")]
    inner: FaerWorkspace,
}

impl fmt::Debug for SvdWorkspace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("SvdWorkspace");
        debug
            .field("operation", &"compact_svd")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("provider", &"faer")
            .field("device", &"cpu");
        #[cfg(feature = "cpu-faer")]
        debug
            .field("context", &self.binding)
            .field("workspace_required_bytes", &self.inner.required_bytes())
            .field("workspace_retained_bytes", &self.retained_bytes());
        debug.finish_non_exhaustive()
    }
}

impl SvdWorkspace {
    /// Return the exact provider-private heap bytes currently retained by this workspace.
    ///
    /// This read-only snapshot does not allocate or mutate the workspace. It
    /// includes provider scratch and staging capacities, but excludes inline
    /// Rust object size, backend binding and identity-token metadata, shared
    /// backend context, plans, and outputs. Values are provider representations,
    /// so no ordering is promised across shapes, dtypes, options, or providers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// use tenferro_tensor::DType;
    ///
    /// let mut backend = CpuBackend::new();
    /// let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
    /// let workspace = plan.allocate_workspace(&mut backend)?;
    /// let workspace_bytes = workspace.retained_bytes();
    /// assert_eq!(workspace_bytes, workspace.retained_bytes());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn retained_bytes(&self) -> usize {
        #[cfg(feature = "cpu-faer")]
        {
            self.inner.retained_bytes()
        }
        #[cfg(not(feature = "cpu-faer"))]
        {
            0
        }
    }
}

/// Backend capability for explicit prepared compact SVD execution.
///
/// Provider dispatch is sealed and required for every backend implementation.
/// Unsupported implementations return an explicit capability error rather than
/// calling an owned SVD or allocating replacement outputs as a fallback.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBackend, CpuBackendKind};
/// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
/// use tenferro_tensor::DType;
///
/// let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer)?;
/// let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
/// assert_eq!(plan.output_specs().s().shape(), &[2]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait PreparedSvdBackendExt:
    private::PreparedSvdDispatch + PreparedFactorizationBackendExt
{
    /// Prepare compact SVD for one fixed rank-2 shape and dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_linalg::{PreparedSvdBackendExt, SvdOptions};
    /// use tenferro_tensor::DType;
    /// let mut backend = CpuBackend::new();
    /// let plan = backend.prepare_svd([3, 2], DType::F64, SvdOptions::default())?;
    /// assert_eq!(plan.output_specs().u().shape(), &[3, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn prepare_svd(
        &mut self,
        shape: [usize; 2],
        dtype: DType,
        options: SvdOptions,
    ) -> tenferro_tensor::Result<PreparedSvd> {
        private::PreparedSvdDispatch::prepare_svd_impl(self, shape, dtype, options)
    }
}

impl<T> PreparedSvdBackendExt for T where
    T: private::PreparedSvdDispatch + PreparedFactorizationBackendExt
{
}

mod private {
    use super::*;

    pub trait PreparedSvdDispatch: LinalgBackend + PreparedFactorizationDispatch {
        fn prepare_svd_impl(
            &mut self,
            shape: [usize; 2],
            dtype: DType,
            options: SvdOptions,
        ) -> tenferro_tensor::Result<PreparedSvd>;

        fn allocate_svd_workspace_impl(
            &mut self,
            plan: &PreparedSvd,
        ) -> tenferro_tensor::Result<SvdWorkspace>;
    }
}

mod cpu;
#[cfg(feature = "cpu-faer")]
use cpu::CpuFaerBinding;

#[cfg(feature = "cpu-faer")]
mod faer_impl;
#[cfg(feature = "cpu-faer")]
use faer_impl::{execute_faer, FaerPlan, FaerWorkspace};
