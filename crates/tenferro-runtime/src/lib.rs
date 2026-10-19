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
#[doc(hidden)]
pub mod composite;
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
#[doc(hidden)]
pub mod scale;
mod segment;
mod session_ext;
mod shape_constraint;
mod shape_infer;
mod shape_packing;
pub mod sym_dim;
mod tensor;
mod trace;
pub mod traced;
mod typed_session_ext;
mod typed_tensor;

pub use compiler::{CompilerOptions, OptimizerConfig};
pub use error::{
    ContextId, Error, ErrorPhase, Result, RuntimeFailureReasonRef, ShapeConstraintEvalError,
};
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
    EngineSnapshotView, EngineTarget, ErasedExecutionContext, EventDomainDriver, EventDomainError,
    EventDomainId, EventDomainOperation, EventDomainRun, EventToken,
    ExecutableEngineRegistrationConfig, ExecutionContextIdentity, ExecutionContextMismatch,
    ExecutionEngine, ExecutionPolicy, ExecutionPolicyError, ExtensionEngine, ExtensionModule,
    ExtensionModuleError, ExtensionModuleId, ExtensionModuleRegistrar, ExtensionPlanningConfig,
    ExtensionPrepareRequest, HardwareClassId, IdentityError, IdentityKind,
    ImmediateEventDomainDriver, IndexingPrepareRequest, IndexingRuntime, InputIngressContract,
    InputIngressContractError, InputPlacementContract, InputSignature, InputSignatureContract,
    InputSignatureEntry, InputSignatureError, InputSpecializationProjection,
    InputSpecializationRequirements, InputSpecializationRequirementsBuilder,
    InputSpecializationRequirementsError, LayoutClass, LayoutPrepareRequest, LayoutProjection,
    LayoutRuntime, LayoutSpecialization, Output, PlacementConstraintError, PlacementProjection,
    PlacementSpecialization, PreparationKeySummary, PreparationOnlyEngineRegistrationConfig,
    PrepareCapability, PrepareError, PrepareOptions, PrepareOptionsKey, Prepared,
    PreparedOperation, PreparedOperationBinding, PreparedOperationExecutor,
    PreparedOperationExecutorHandle, PreparedOperationHandle, PreparedOperationPlan,
    PreparedPlanCacheLimits, PreparedPlanCacheStats, ProgramPlacementConstraint,
    ProviderContractError, ProviderDeviceIdentity, ProviderId, RankRequirement,
    ReductionPrepareRequest, ReductionRuntime, RegistrationIdentity, RegistrationKey,
    ResidentOutputContract, ResolvedPlanningConfig, ResolvedPlanningKey, ResolvedProgramPlacement,
    Runtime, RuntimeCacheError, RuntimeCacheOwner, RuntimeCacheStats, RuntimeConfigBuilder,
    RuntimeConfigError, RuntimeConfigSnapshot, RuntimeEpoch, RuntimeId, RuntimeInputContract,
    RuntimeReconfiguration, RuntimeReconfigureError, RuntimeStateError, SpecializationError,
    SpecializationProjection, SpecializationRequirements, StorageClass, TransferEndpoint,
    TransferError, TransferProvider, TransferProviderContractError, TransferRequest,
    UnsupportedReason,
};
pub use session_ext::TensorSessionOpsExt;
#[doc(hidden)]
pub use shape_constraint::ShapeGuard;
pub use shape_packing::TracedSliceBuilder;
pub use sym_dim::SymDim;
pub use tenferro_ops::ShapeRelation;
pub use tenferro_tensor::{
    CacheStats, CompareDir, DType, DotGeneralConfig, Engine, GatherConfig, MemoryKind, PadConfig,
    ScatterConfig, SliceConfig, Tensor, TensorRead, TensorScalar, TensorValue, TensorView,
    TypedTensor, TypedTensorView,
};
pub use trace::{TraceContext, TraceValue, TracedGraph};
pub use typed_session_ext::{TypedTensorMaskSessionOpsExt, TypedTensorSessionOpsExt};

pub use traced::TracedTensor;
