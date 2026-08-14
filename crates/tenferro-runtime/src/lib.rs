//! Traced graph runtime and extension dispatch infrastructure for tenferro.
//!
//! This crate owns graph construction, lowering to execution IR, graph
//! execution, and backend-parametric extension runtime dispatch. Standard
//! operations are lowered through the runtime's internal operation vocabulary;
//! tensor storage and backend kernels live in `tenferro-tensor`.
//!
//! Use this crate directly when you want concrete tensor helpers or reusable
//! traced graph execution without depending on `tenferro-ad`. Start with
//! [`TypedTensor`] when the scalar type is fixed in Rust, [`Tensor`] when dtype
//! is selected at runtime, and [`TracedTensor`] plus [`GraphCompiler`] and
//! [`Runtime`] when the same expression should be compiled once and run
//! repeatedly. Operation-family crates such as `tenferro-einsum`,
//! `tenferro-linalg`, and `tenferro-fft` register extension runtimes through
//! runtime engine registrations when compiled execution reaches those
//! operations.
//!
//! User-facing guides live at
//! <https://tensor4all.org/tenferro-rs/guides/choosing-an-api.html> and
//! <https://tensor4all.org/tenferro-rs/guides/execution-models.html>.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_runtime::{GraphCompiler, Runtime, TracedTensor};
//! use tenferro_cpu::CpuBackend;
//!
//! let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
//! let y = (&x + &x).unwrap();
//! let mut compiler = GraphCompiler::new();
//! let program = compiler.compile(&y).unwrap();
//! let backend = CpuBackend::default();
//! let mut builder = Runtime::builder();
//! builder.register_engine(tenferro_cpu::runtime_engine_registration(&backend).unwrap()).unwrap();
//! let runtime = builder.build().unwrap();
//! let out = runtime.run_compiled(&program, &[]).unwrap().pop().unwrap();
//! assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
//! ```

#[doc(hidden)]
pub mod ad_support;
mod checkpoint;
mod compiler;
pub mod error;
mod exec;
pub mod extension;
pub mod extension_cache;
mod extension_execution_context;
pub mod graph;
mod metadata;
pub mod prelude;
pub mod program;
pub mod runtime;
#[doc(hidden)]
pub mod scalar_semantics;
mod segment;
mod shape_constraint;
mod shape_infer;
mod shape_packing;
pub mod sym_dim;
mod tensor;
mod trace;
pub mod traced;
mod typed_tensor;

pub use compiler::{CompilerOptions, OptimizerConfig};
pub use error::{ContextId, Error, ErrorPhase, Result, ShapeConstraintEvalError};
pub use extension_cache::{
    ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
};
pub use extension_execution_context::ExtensionExecutionContext;
pub use graph::{CompiledGraph, GraphCompiler};
pub use runtime::{
    assemble_executable_engine_registration, assemble_preparation_only_engine_registration,
    CacheInFlightBehavior, CacheOwnerError, CacheOwnerFailure, CacheOwnerId, CoreCapabilityBundle,
    CoreCapabilityBundleBuilder, CoreCapabilityKind, CorePrepareContext, Determinism,
    DotGeneralPreparation, DotGeneralPrepareRequest, ElementwisePrepareRequest, ElementwiseRuntime,
    EngineExecutionContractError, EngineId, EngineRegistration, EngineRegistrationMetadata,
    EngineSnapshotView, ErasedExecutionContext, EventDomainDriver, EventDomainError, EventDomainId,
    EventDomainOperation, EventDomainRun, EventToken, ExecutableEngineRegistrationConfig,
    ExecutionBundle, ExecutionContextIdentity, ExecutionContextMismatch, ExecutionHandle,
    ExecutionInputs, ExecutionOutcome, ExecutionPolicy, ExecutionPolicyError, ExtensionEngine,
    ExtensionModule, ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar,
    ExtensionPlanningConfig, ExtensionPrepareRequest, HardwareClassId, IdentityError, IdentityKind,
    ImmediateEventDomainDriver, IndexingPrepareRequest, IndexingRuntime, InputIngressContract,
    InputIngressContractError, InputPlacementContract, InputSignature, InputSignatureContract,
    InputSignatureEntry, InputSignatureError, InputSpecializationProjection,
    InputSpecializationRequirements, InputSpecializationRequirementsBuilder,
    InputSpecializationRequirementsError, LayoutClass, LayoutPrepareRequest, LayoutProjection,
    LayoutRuntime, LayoutSpecialization, OutputAccessError, OutputExtractError, OutputMetadata,
    OutputRef, PlacementConstraintError, PlacementProjection, PlacementSpecialization,
    PreparationKeySummary, PreparationOnlyEngineRegistrationConfig, PrepareCapability,
    PrepareError, PrepareOptions, PrepareOptionsKey, PreparedCompiledGraph, PreparedOperation,
    PreparedOperationBinding, PreparedOperationExecutor, PreparedOperationExecutorHandle,
    PreparedOperationHandle, PreparedOperationPlan, PreparedPlanCacheLimits,
    PreparedPlanCacheStats, ProgramPlacementConstraint, ProviderContractError,
    ProviderDeviceIdentity, ProviderId, RankRequirement, ReductionPrepareRequest, ReductionRuntime,
    RegistrationIdentity, RegistrationKey, ResidentOutputContract, ResolvedPlanningConfig,
    ResolvedPlanningKey, ResolvedProgramPlacement, Runtime, RuntimeCacheError, RuntimeCacheOwner,
    RuntimeCacheStats, RuntimeConfigBuilder, RuntimeConfigError, RuntimeConfigSnapshot,
    RuntimeEpoch, RuntimeId, RuntimeInputContract, RuntimeReconfiguration, RuntimeReconfigureError,
    RuntimeStateError, ScopedExecutionBundle, ScopedExecutionOutcome, ScopedOutput,
    ScopedOutputExtractError, ScopedReadBinding, ScopedReadInputs, ScopedSubmitRejected,
    SpecializationError, SpecializationProjection, SpecializationRequirements, StorageClass,
    SubmissionError, SubmitError, TransferEndpoint, TransferError, TransferProvider,
    TransferProviderContractError, TransferRequest, UnsupportedReason,
};
#[doc(hidden)]
pub use shape_constraint::ShapeGuard;
pub use shape_packing::TracedSliceBuilder;
pub use sym_dim::SymDim;
pub use tenferro_ops::ShapeRelation;
pub use tenferro_tensor::{
    BackendSession, BackendSessionHost, CacheStats, CompareDir, DType, DotGeneralConfig,
    GatherConfig, MemoryKind, PadConfig, ScatterConfig, SliceConfig, Tensor, TensorBackend,
    TensorRead, TensorScalar, TensorValue, TensorView, TypedTensor, TypedTensorView,
};
pub use trace::{TraceContext, TraceValue, TracedGraph};

/// Backend-explicit concrete tensor operations.
///
/// `Tensor` is owned by `tenferro-tensor`, so `tenferro-runtime` exposes these
/// operations as a crate-root extension trait rather than as inherent methods.
///
/// # Public API rationale
///
/// This trait is intentionally public: it is the supported non-AD concrete
/// tensor operation surface for downstream users who want to run operations on
/// an explicit backend. The old public module/free-function surface was
/// removed; the private `tensor` module now contains implementation helpers
/// only and must not be treated as a compatibility API.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorOpsExt};
///
/// let mut backend = CpuBackend::new();
/// let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
/// let b = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64; 4]).unwrap();
/// let c = a.matmul(&b, &mut backend).unwrap();
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
pub trait TensorOpsExt {
    /// Convert to a different dtype using the checked conversion lattice.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// conversion is outside the checked lattice,
    /// [`tenferro_tensor::Error::Validation`] with `DTypeMismatch` or
    /// `InvalidArgument` for invalid tensor metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] when the backend reports a
    /// typed failure.
    fn convert<B: TensorBackend>(
        &self,
        to: DType,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Cast to a different dtype using explicit lossy projection.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// requested cast is unsupported, [`tenferro_tensor::Error::Validation`]
    /// with `DTypeMismatch` or `InvalidArgument` for invalid tensor metadata,
    /// or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cast<B: TensorBackend>(&self, to: DType, backend: &mut B)
        -> tenferro_tensor::Result<Tensor>;
    /// Elementwise addition with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with a
    /// [`ShapeMismatch`](tenferro_tensor::ValidationError::ShapeMismatch) or
    /// `DTypeMismatch` payload when operands are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn add<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise subtraction with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn sub<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise multiplication with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn mul<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise division with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility,
    /// [`tenferro_tensor::Error::Extension`] with a numerical classification
    /// for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn div<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise remainder with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn rem<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise power with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible metadata, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected negative integer
    /// exponent, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn pow<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise maximum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn maximum<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise minimum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch` or `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn minimum<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise negation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the dtype is not
    /// supported by the operation, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn neg<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise absolute value.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sign.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise complex conjugate.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise exponential.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn exp<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise natural logarithm.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sin<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise cosine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cos<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise hyperbolic tangent.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise reciprocal square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `exp(x) - 1`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `log(1 + x)`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise comparison with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible shape/dtype metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn compare<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Select values from `on_true` or `on_false` using this tensor as condition.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when the condition and branches are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn where_select<B: TensorBackend>(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Clamp values elementwise between lower and upper bounds.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when bounds are incompatible with the input, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn clamp<B: TensorBackend>(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Rank-2 matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch`,
    /// `ShapeMismatch`, or `DTypeMismatch` for incompatible matrices, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul<B: TensorBackend>(
        &self,
        rhs: &Tensor,
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Reshape without changing element order.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch`, `RankMismatch`, or `InvalidArgument` when element
    /// counts or ranks are invalid, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape<B: TensorBackend>(
        &self,
        shape: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Permute axes.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength`, `AxisOutOfBounds`, or `DuplicateAxis` for
    /// an invalid permutation, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn transpose<B: TensorBackend>(
        &self,
        perm: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Sum over one or more axes.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// or `DuplicateAxis` for invalid reductions, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reduce_sum<B: TensorBackend>(
        &self,
        axes: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<Tensor>;
}

/// Backend-explicit session operations for concrete [`Tensor`] values.
///
/// These operations run inside a caller-provided [`BackendSession`], so a
/// chain of `_in` operations executes in a single backend session entry and
/// never enters a session of its own. Each one-shot [`TensorOpsExt`]
/// counterpart performs its broadcast and execution inside a single session
/// entry of its own.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
/// use tenferro_tensor::BackendSessionHost;
///
/// let mut backend = CpuBackend::new();
/// let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
/// let b = Tensor::from_vec_col_major(vec![8], vec![2.0_f64; 8]).unwrap();
/// let total = backend.with_backend_session(|session| {
///     let sum = a.add_in(&b, session).unwrap();
///     let grown = sum.exp_in(session).unwrap();
///     grown.mul_in(&a, session).unwrap().reduce_sum_in(&[0], session).unwrap()
/// });
/// assert!(total.as_slice::<f64>().unwrap()[0].is_finite());
/// ```
pub trait TensorSessionOpsExt {
    /// Elementwise addition with NumPy-style broadcasting inside a session.
    ///
    /// The broadcast (reshape + `broadcast_in_dim`, or a copy when shapes
    /// already match) and the add itself all run in the caller's `session`;
    /// this op never enters a session of its own.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    /// let sum = backend.with_backend_session(|session| a.add_in(&b, session)).unwrap();
    /// assert_eq!(sum.as_slice::<f64>().unwrap(), &[4.0, 6.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with a
    /// [`ShapeMismatch`](tenferro_tensor::ValidationError::ShapeMismatch) or
    /// `DTypeMismatch` payload when operands are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn add_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise multiplication with NumPy-style broadcasting inside a session.
    ///
    /// Like [`Self::add_in`], broadcast and multiply run in the one `session`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![4], vec![3.0_f64; 4]).unwrap();
    /// let product = backend.with_backend_session(|session| a.mul_in(&b, session)).unwrap();
    /// assert_eq!(product.as_slice::<f64>().unwrap(), &[6.0; 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn mul_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise exponential inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.exp_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!((y[0] - 1.0).abs() < 1.0e-12);
    /// assert!((y[1] - std::f64::consts::E).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn exp_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Sum over one or more axes inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let sums = backend.with_backend_session(|session| x.reduce_sum_in(&[1], session)).unwrap();
    /// assert_eq!(sums.as_slice::<f64>().unwrap(), &[3.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// or `DuplicateAxis` for invalid reductions, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reduce_sum_in(
        &self,
        axes: &[usize],
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Convert to a different dtype using the checked conversion lattice inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.convert_in(DType::C64, session)).unwrap();
    /// assert_eq!(y.dtype(), DType::C64);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// conversion is outside the checked lattice,
    /// [`tenferro_tensor::Error::Validation`] with `DTypeMismatch` or
    /// `InvalidArgument` for invalid tensor metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] when the backend reports a
    /// typed failure.
    fn convert_in(
        &self,
        to: DType,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Cast to a different dtype using explicit lossy projection inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{DType, Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.2_f64, -2.8]).unwrap();
    /// let y = backend.with_backend_session(|session| x.cast_in(DType::I32, session)).unwrap();
    /// assert_eq!(y.as_slice::<i32>().unwrap(), &[1, -2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::UnsupportedDTypeConversion`] when the
    /// requested cast is unsupported, [`tenferro_tensor::Error::Validation`]
    /// with `DTypeMismatch` or `InvalidArgument` for invalid tensor metadata,
    /// or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cast_in(
        &self,
        to: DType,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise subtraction with NumPy-style broadcasting inside a session.
    ///
    /// Like [`Self::add_in`], the broadcast and the subtraction run in the one
    /// `session`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.sub_in(&b, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, -4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn sub_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise division with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 8.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.div_in(&b, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility,
    /// [`tenferro_tensor::Error::Extension`] with a numerical classification
    /// for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn div_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise remainder with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![5.0_f64, 7.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.rem_in(&b, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for shape/dtype incompatibility, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected zero divisor, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn rem_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise power with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.pow_in(&b, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[8.0, 9.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible metadata, a numerical
    /// [`tenferro_tensor::Error::Extension`] for a detected negative integer
    /// exponent, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn pow_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise maximum with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.maximum_in(&b, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 8.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn maximum_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise minimum with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.minimum_in(&b, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn minimum_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise negation inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.neg_in(session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[-1.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] when the dtype is not
    /// supported by the operation, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn neg_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise absolute value inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![-1.0_f64, 2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.abs_in(session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sign inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sign_in(session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, -1.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise complex conjugate inside a session.
    ///
    /// For real dtypes the conjugate is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.conj_in(session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, -2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise natural logarithm inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, std::f64::consts::E]).unwrap();
    /// let y = backend.with_backend_session(|session| x.log_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `exp(x) - 1` inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.expm1_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - (std::f64::consts::E - 1.0)).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn expm1_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise `log(1 + x)` inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::E - 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.log1p_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log1p_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise sine inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::FRAC_PI_2]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sin_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sin_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise cosine inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::PI]).unwrap();
    /// let y = backend.with_backend_session(|session| x.cos_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!((y[0] - 1.0).abs() < 1.0e-12);
    /// assert!((y[1] + 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cos_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise hyperbolic tangent inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.tanh_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 0.7615941559557649).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn tanh_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise square root inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 9.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sqrt_in(session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise reciprocal square root inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![4.0_f64, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.rsqrt_in(session)).unwrap();
    /// let y = y.as_slice::<f64>().unwrap();
    /// assert!((y[0] - 0.5).abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn rsqrt_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor>;
    /// Elementwise comparison with NumPy-style broadcasting inside a session.
    ///
    /// The result is a bool tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{CompareDir, Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 4.0]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.compare_in(&b, CompareDir::Gt, session)).unwrap();
    /// assert_eq!(y.as_slice::<bool>().unwrap(), &[true, false]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` for incompatible shape/dtype metadata, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn compare_in(
        &self,
        rhs: &Tensor,
        dir: CompareDir,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Select values from `on_true` or `on_false` using this tensor as condition inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let condition = Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    /// let on_true = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    /// let on_false = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| condition.where_select_in(&on_true, &on_false, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[1.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when the condition and branches are incompatible, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn where_select_in(
        &self,
        on_true: &Tensor,
        on_false: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Clamp values elementwise between lower and upper bounds inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2], vec![-2.0_f64, 4.0]).unwrap();
    /// let lower = Tensor::from_vec_col_major(vec![], vec![0.0_f64]).unwrap();
    /// let upper = Tensor::from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    /// let y = backend.with_backend_session(|session| x.clamp_in(&lower, &upper, session)).unwrap();
    /// assert_eq!(y.as_slice::<f64>().unwrap(), &[0.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` or
    /// `DTypeMismatch` when bounds are incompatible with the input, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn clamp_in(
        &self,
        lower: &Tensor,
        upper: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Rank-2 matrix multiplication inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    /// let c = backend.with_backend_session(|session| a.matmul_in(&b, session)).unwrap();
    /// assert_eq!(c.shape(), &[2, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch`,
    /// `ShapeMismatch`, or `DTypeMismatch` for incompatible matrices, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul_in(
        &self,
        rhs: &Tensor,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Reshape without changing element order inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.reshape_in(&[4], session)).unwrap();
    /// assert_eq!(y.shape(), &[4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch`, `RankMismatch`, or `InvalidArgument` when element
    /// counts or ranks are invalid, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape_in(
        &self,
        shape: &[usize],
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
    /// Permute axes inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{Tensor, TensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    /// let y = backend.with_backend_session(|session| x.transpose_in(&[1, 0], session)).unwrap();
    /// assert_eq!(y.shape(), &[3, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength`, `AxisOutOfBounds`, or `DuplicateAxis` for
    /// an invalid permutation, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn transpose_in(
        &self,
        perm: &[usize],
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<Tensor>;
}

/// Backend-explicit operations for dynamic-rank typed tensors.
///
/// `TypedTensor` is owned by `tenferro-tensor`, so `tenferro-runtime` exposes
/// these operations as a crate-root extension trait rather than as inherent
/// methods.
///
/// # Public API rationale
///
/// This trait is intentionally public for the same reason as [`TensorOpsExt`]:
/// downstream users need a supported backend-explicit typed tensor surface, and
/// `tenferro-runtime` cannot add inherent methods to a type owned by
/// `tenferro-tensor`. The private `typed_tensor` module is implementation
/// detail, not a retained module/free-function API.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
///
/// let mut backend = CpuBackend::new();
/// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
/// let y = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
/// let sum = x.add(&y, &mut backend).unwrap();
/// assert_eq!(sum.host_data().unwrap(), &[4.0, 6.0]);
/// ```
pub trait TypedTensorOpsExt<T: TensorScalar> {
    /// Elementwise addition with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn add<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise subtraction with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn sub<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise multiplication with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn mul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise division with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn div<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise remainder with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn rem<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise power with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected negative integer exponent, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn pow<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise maximum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn maximum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise minimum with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn minimum<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise negation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn neg<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise absolute value.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sign.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise complex conjugate.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise exponential.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn exp<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise natural logarithm.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sin<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise cosine.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cos<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise hyperbolic tangent.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn tanh<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise reciprocal square root.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn rsqrt<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `exp(x) - 1`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn expm1<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `log(1 + x)`.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log1p<B: TensorBackend>(&self, backend: &mut B) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise comparison with NumPy-style broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when broadcasting the operands is
    /// impossible, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn compare<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<bool>>;
    /// Clamp values elementwise between lower and upper bounds.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when a bound cannot broadcast to
    /// the input, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn clamp<B: TensorBackend>(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Rank-2 matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// either operand is not rank two or `ShapeMismatch::ContractedDimensions`
    /// when the inner dimensions differ, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul<B: TensorBackend>(
        &self,
        rhs: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum over one or more axes.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// for an axis outside the input rank or `DuplicateAxis` when `axes`
    /// repeats an axis, or [`tenferro_tensor::Error::BackendSource`] for a
    /// typed backend failure.
    fn reduce_sum<B: TensorBackend>(
        &self,
        axes: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Reshape through the backend structural operation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::ReshapeElementCount` when the element counts differ,
    /// `IntegerOverflow` when shape arithmetic overflows, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape<B: TensorBackend>(
        &self,
        shape: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Permute axes through the backend structural operation.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength` when `perm` has the wrong length,
    /// `AxisOutOfBounds` for an invalid axis, or `DuplicateAxis` for a
    /// repeated axis, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn transpose<B: TensorBackend>(
        &self,
        perm: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Broadcast into a larger shape.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// `dims` does not match the input rank, `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid mapping, or
    /// `ShapeMismatch::IncompatibleShapes` when known dimensions cannot
    /// broadcast. [`tenferro_tensor::Error::BackendSource`] reports a typed
    /// backend failure.
    fn broadcast_in_dim<B: TensorBackend>(
        &self,
        shape: &[usize],
        dims: &[usize],
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

/// Backend-explicit session operations for dynamic-rank typed tensors.
///
/// Like [`TensorSessionOpsExt`] but for [`TypedTensor`] with a statically
/// known scalar type `T`. The typed surface keeps inputs borrowed (read-based
/// session ops) and validates the output dtype through
/// `into_typed_result`-style conversion before returning.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBackend;
/// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
/// use tenferro_tensor::BackendSessionHost;
///
/// let mut backend = CpuBackend::new();
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
/// let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![2.0; 4]).unwrap();
/// let total = backend.with_backend_session(|session| {
///     let sum = a.add_in(&b, session).unwrap();
///     sum.exp_in(session).unwrap().reduce_sum_in(&[0], session).unwrap()
/// });
/// assert!(total.host_data().unwrap()[0].is_finite());
/// ```
pub trait TypedTensorSessionOpsExt<T: TensorScalar> {
    /// Elementwise addition with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 4.0]).unwrap();
    /// let sum = backend.with_backend_session(|session| a.add_in(&b, session)).unwrap();
    /// assert_eq!(sum.host_data().unwrap(), &[4.0, 6.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn add_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise multiplication with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![3.0; 4]).unwrap();
    /// let product = backend.with_backend_session(|session| a.mul_in(&b, session)).unwrap();
    /// assert_eq!(product.host_data().unwrap(), &[6.0; 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn mul_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise exponential inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.exp_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!((y[0] - 1.0).abs() < 1.0e-12);
    /// assert!((y[1] - std::f64::consts::E).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn exp_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Sum over one or more axes inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let sums = backend.with_backend_session(|session| x.reduce_sum_in(&[1], session)).unwrap();
    /// assert_eq!(sums.host_data().unwrap(), &[3.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `AxisOutOfBounds`
    /// for an axis outside the input rank or `DuplicateAxis` when `axes`
    /// repeats an axis, or [`tenferro_tensor::Error::BackendSource`] for a
    /// typed backend failure.
    fn reduce_sum_in(
        &self,
        axes: &[usize],
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise subtraction with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.sub_in(&b, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn sub_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise division with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 8.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.div_in(&b, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[2.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn div_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise remainder with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![5.0, 7.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.rem_in(&b, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[1.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected zero divisor, or [`tenferro_tensor::Error::BackendSource`]
    /// for a typed backend failure.
    fn rem_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise power with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 3.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![3.0, 2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.pow_in(&b, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[8.0, 9.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible shapes, a numerical [`tenferro_tensor::Error::Extension`]
    /// for a detected negative integer exponent, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn pow_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise maximum with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.maximum_in(&b, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[2.0, 8.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn maximum_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise minimum with NumPy-style broadcasting inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.minimum_in(&b, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[1.0, 4.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `ShapeMismatch` for
    /// incompatible operands, or [`tenferro_tensor::Error::BackendSource`] for
    /// a typed backend failure.
    fn minimum_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise negation inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.neg_in(session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[-1.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn neg_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise absolute value inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-1.0, 2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.abs_in(session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[1.0, 2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn abs_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sign inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sign_in(session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -1.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sign_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise complex conjugate inside a session.
    ///
    /// For real dtypes the conjugate is the identity.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, -2.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.conj_in(session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[1.0, -2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn conj_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise natural logarithm inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, std::f64::consts::E]).unwrap();
    /// let y = backend.with_backend_session(|session| x.log_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `exp(x) - 1` inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.expm1_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - (std::f64::consts::E - 1.0)).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn expm1_in(&self, session: &mut dyn BackendSession)
        -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise `log(1 + x)` inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::E - 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.log1p_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn log1p_in(&self, session: &mut dyn BackendSession)
        -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise sine inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::FRAC_PI_2]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sin_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sin_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise cosine inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, std::f64::consts::PI]).unwrap();
    /// let y = backend.with_backend_session(|session| x.cos_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!((y[0] - 1.0).abs() < 1.0e-12);
    /// assert!((y[1] + 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn cos_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise hyperbolic tangent inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.tanh_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!(y[0].abs() < 1.0e-12);
    /// assert!((y[1] - 0.7615941559557649).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn tanh_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise square root inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 9.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.sqrt_in(session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[2.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn sqrt_in(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise reciprocal square root inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 1.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.rsqrt_in(session)).unwrap();
    /// let y = y.host_data().unwrap();
    /// assert!((y[0] - 0.5).abs() < 1.0e-12);
    /// assert!((y[1] - 1.0).abs() < 1.0e-12);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Unsupported`] for an unsupported
    /// dtype or [`tenferro_tensor::Error::BackendSource`] for a typed backend
    /// failure.
    fn rsqrt_in(&self, session: &mut dyn BackendSession)
        -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Elementwise comparison with NumPy-style broadcasting inside a session.
    ///
    /// The result is a bool typed tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{CompareDir, TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 4.0]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 8.0]).unwrap();
    /// let y = backend.with_backend_session(|session| a.compare_in(&b, CompareDir::Gt, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[true, false]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when broadcasting the operands is
    /// impossible, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn compare_in(
        &self,
        rhs: &TypedTensor<T>,
        dir: CompareDir,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<bool>>;
    /// Clamp values elementwise between lower and upper bounds inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![-2.0, 4.0]).unwrap();
    /// let lower = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]).unwrap();
    /// let upper = TypedTensor::<f64>::from_vec_col_major(vec![], vec![3.0]).unwrap();
    /// let y = backend.with_backend_session(|session| x.clamp_in(&lower, &upper, session)).unwrap();
    /// assert_eq!(y.host_data().unwrap(), &[0.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when a bound cannot broadcast to
    /// the input, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn clamp_in(
        &self,
        lower: &TypedTensor<T>,
        upper: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Rank-2 matrix multiplication inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], vec![1.0; 6]).unwrap();
    /// let c = backend.with_backend_session(|session| a.matmul_in(&b, session)).unwrap();
    /// assert_eq!(c.shape(), &[2, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// either operand is not rank two or `ShapeMismatch::ContractedDimensions`
    /// when the inner dimensions differ, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn matmul_in(
        &self,
        rhs: &TypedTensor<T>,
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Reshape through the backend structural operation inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let y = backend.with_backend_session(|session| x.reshape_in(&[3, 2], session)).unwrap();
    /// assert_eq!(y.shape(), &[3, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::ReshapeElementCount` when the element counts differ,
    /// `IntegerOverflow` when shape arithmetic overflows, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn reshape_in(
        &self,
        shape: &[usize],
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Permute axes through the backend structural operation inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let y = backend.with_backend_session(|session| x.transpose_in(&[1, 0], session)).unwrap();
    /// assert_eq!(y.shape(), &[3, 2]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `InvalidPermutationLength` when `perm` has the wrong length,
    /// `AxisOutOfBounds` for an invalid axis, or `DuplicateAxis` for a
    /// repeated axis, or [`tenferro_tensor::Error::BackendSource`] for a typed
    /// backend failure.
    fn transpose_in(
        &self,
        perm: &[usize],
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
    /// Broadcast into a larger shape inside a session.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBackend;
    /// use tenferro_runtime::{TypedTensor, TypedTensorSessionOpsExt};
    /// use tenferro_tensor::BackendSessionHost;
    ///
    /// let mut backend = CpuBackend::new();
    /// let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    /// let matrix = backend.with_backend_session(|session| row.broadcast_in_dim_in(&[2, 3], &[1], session)).unwrap();
    /// assert_eq!(matrix.shape(), &[2, 3]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with `RankMismatch` when
    /// `dims` does not match the input rank, `AxisOutOfBounds` or
    /// `DuplicateAxis` for an invalid mapping, or
    /// `ShapeMismatch::IncompatibleShapes` when known dimensions cannot
    /// broadcast. [`tenferro_tensor::Error::BackendSource`] reports a typed
    /// backend failure.
    fn broadcast_in_dim_in(
        &self,
        shape: &[usize],
        dims: &[usize],
        session: &mut dyn BackendSession,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

/// Backend-explicit bool-mask operations for typed tensors.
///
/// # Public API rationale
///
/// This trait keeps `where_select` available as a method on bool
/// `TypedTensor`s while preserving the crate-root extension-trait surface. It
/// is public because downstream users call it directly; the implementation
/// helper in the private `typed_tensor` module is not a compatibility API.
pub trait TypedTensorMaskOpsExt {
    /// Select typed values using this bool tensor as condition.
    ///
    /// # Errors
    ///
    /// Returns [`tenferro_tensor::Error::Validation`] with
    /// `ShapeMismatch::IncompatibleShapes` when the condition or either branch
    /// cannot broadcast to the other operands, or
    /// [`tenferro_tensor::Error::BackendSource`] for a typed backend failure.
    fn where_select<T: TensorScalar, B: TensorBackend>(
        &self,
        on_true: &TypedTensor<T>,
        on_false: &TypedTensor<T>,
        backend: &mut B,
    ) -> tenferro_tensor::Result<TypedTensor<T>>;
}

pub use traced::TracedTensor;
