use std::any::Any;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use crate::program::SemanticOperationView;

use super::{
    ExecutionContextIdentity, ExecutionContextMismatch, HardwareClassId, InputSignature,
    PrepareError, PrepareOptionsKey, RegistrationIdentity, ResolvedPlanningConfig,
    ResolvedProgramPlacement, RuntimeEpoch, RuntimeId, SpecializationProjection,
    SpecializationRequirements,
};

/// Core operation family handled by a direct runtime capability slot.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::CoreCapabilityKind;
///
/// assert_eq!(CoreCapabilityKind::Elementwise, CoreCapabilityKind::Elementwise);
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CoreCapabilityKind {
    /// Elementwise core operations.
    Elementwise,
    /// Reduction core operations.
    Reduction,
    /// Indexing and slicing core operations.
    Indexing,
    /// Dot-general preparation.
    DotGeneral,
    /// Layout-only operations.
    Layout,
}

/// Deterministic reason why a provider cannot prepare an operation.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{CoreCapabilityKind, UnsupportedReason};
///
/// let reason = UnsupportedReason::MissingCapability {
///     capability: CoreCapabilityKind::Elementwise,
/// };
/// assert!(reason.to_string().contains("Elementwise"));
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum UnsupportedReason {
    /// The runtime snapshot has no provider for this core capability.
    MissingCapability {
        /// Missing core capability kind.
        capability: CoreCapabilityKind,
    },
    /// The named operation is unsupported by the selected provider.
    Operation {
        /// Stable operation diagnostic.
        operation: &'static str,
    },
    /// The selected provider does not support the required storage class.
    StorageClass {
        /// Unsupported storage class.
        storage_class: super::StorageClass,
    },
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCapability { capability } => {
                write!(formatter, "missing {capability:?} capability")
            }
            Self::Operation { operation } => {
                write!(formatter, "unsupported operation {operation:?}")
            }
            Self::StorageClass { storage_class } => {
                write!(
                    formatter,
                    "unsupported storage class {:?}",
                    storage_class.as_str()
                )
            }
        }
    }
}

/// Bounded diagnostic summary of a preparation cache key.
///
/// A1 exposes only the summary shape. C1 owns the concrete key construction.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::PreparationKeySummary;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<PreparationKeySummary>();
/// ```
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct PreparationKeySummary {
    compact_digest: u128,
}

impl PreparationKeySummary {
    #[allow(dead_code)]
    pub(crate) const fn new(compact_digest: u128) -> Self {
        Self { compact_digest }
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) const fn for_test(value: u64) -> Self {
        Self {
            compact_digest: value as u128,
        }
    }
}

/// Runtime-created immutable binding for one prepared operation.
///
/// A1 has no public constructor. B0/C1 provide the runtime creation path.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::PreparedOperationBinding;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<PreparedOperationBinding>();
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PreparedOperationBinding {
    runtime_id: RuntimeId,
    epoch: RuntimeEpoch,
    engine_id: super::EngineId,
    registration_identity: RegistrationIdentity,
    context_identity: ExecutionContextIdentity,
    hardware_class: HardwareClassId,
}

impl PreparedOperationBinding {
    #[allow(dead_code)]
    pub(crate) fn new(
        runtime_id: RuntimeId,
        epoch: RuntimeEpoch,
        engine_id: super::EngineId,
        registration_identity: RegistrationIdentity,
        context_identity: ExecutionContextIdentity,
        hardware_class: HardwareClassId,
    ) -> Self {
        Self {
            runtime_id,
            epoch,
            engine_id,
            registration_identity,
            context_identity,
            hardware_class,
        }
    }

    /// Return the runtime identity that created this binding.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PreparedOperationBinding;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PreparedOperationBinding>();
    /// ```
    pub fn runtime_id(&self) -> RuntimeId {
        self.runtime_id
    }

    /// Return the immutable runtime epoch that created this binding.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PreparedOperationBinding;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PreparedOperationBinding>();
    /// ```
    pub fn epoch(&self) -> RuntimeEpoch {
        self.epoch
    }

    /// Return the selected engine identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PreparedOperationBinding;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PreparedOperationBinding>();
    /// ```
    pub fn engine_id(&self) -> &super::EngineId {
        &self.engine_id
    }

    /// Return the runtime-local engine registration identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PreparedOperationBinding;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PreparedOperationBinding>();
    /// ```
    pub fn registration_identity(&self) -> RegistrationIdentity {
        self.registration_identity
    }

    /// Return the execution-context type identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PreparedOperationBinding;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PreparedOperationBinding>();
    /// ```
    pub fn context_identity(&self) -> ExecutionContextIdentity {
        self.context_identity
    }

    /// Return the selected hardware-class identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt::Debug;
    /// use tenferro_runtime::PreparedOperationBinding;
    ///
    /// fn requires_debug<T: Debug>() {}
    /// requires_debug::<PreparedOperationBinding>();
    /// ```
    pub fn hardware_class(&self) -> &HardwareClassId {
        &self.hardware_class
    }
}

/// Immutable metadata for a prepared provider operation.
///
/// The trait intentionally exposes no execution method. Phase 5 owns common
/// execution scheduling and invocation.
pub trait PreparedOperation: fmt::Debug + Send + Sync + 'static {
    /// Return the runtime-created binding carried by this operation.
    fn binding(&self) -> &PreparedOperationBinding;
    /// Return the specialization projection this operation depends on.
    fn specialization(&self) -> &SpecializationProjection;
    /// Return logical heap bytes owned exclusively by this operation.
    fn retained_bytes(&self) -> usize;
}

/// Shared prepared-operation handle.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::PreparedOperationHandle;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<PreparedOperationHandle>();
/// ```
pub type PreparedOperationHandle = Arc<dyn PreparedOperation>;

/// Result of asking a provider to prepare an operation.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{CoreCapabilityKind, PrepareCapability, UnsupportedReason};
///
/// let capability = PrepareCapability::Unsupported(UnsupportedReason::MissingCapability {
///     capability: CoreCapabilityKind::Elementwise,
/// });
/// assert!(matches!(capability, PrepareCapability::Unsupported { .. }));
/// ```
#[derive(Clone, Debug)]
pub enum PrepareCapability {
    /// Provider returned an immutable prepared operation.
    Prepared(PreparedOperationHandle),
    /// Provider requires a wider specialization before it can prepare.
    NeedsSpecialization(SpecializationRequirements),
    /// Provider does not support this request.
    Unsupported(UnsupportedReason),
}

/// Type-erased execution context with a checked identity.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::{ErasedExecutionContext, ExecutionContextIdentity};
///
/// let mut value = 1_u64;
/// let erased = ErasedExecutionContext::new(&mut value);
/// assert_eq!(erased.identity(), ExecutionContextIdentity::of::<u64>());
/// ```
pub struct ErasedExecutionContext<'a> {
    identity: ExecutionContextIdentity,
    value: &'a mut (dyn Any + Send + Sync),
}

impl<'a> ErasedExecutionContext<'a> {
    /// Erase a concrete execution context while retaining its type identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::ErasedExecutionContext;
    ///
    /// let mut value = 1_u64;
    /// let erased = ErasedExecutionContext::new(&mut value);
    /// assert_eq!(erased.identity().type_name(), std::any::type_name::<u64>());
    /// ```
    pub fn new<T: Send + Sync + 'static>(value: &'a mut T) -> Self {
        Self {
            identity: ExecutionContextIdentity::of::<T>(),
            value,
        }
    }

    /// Return the stored execution-context type identity.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::{ErasedExecutionContext, ExecutionContextIdentity};
    ///
    /// let mut value = 1_u64;
    /// let erased = ErasedExecutionContext::new(&mut value);
    /// assert_eq!(erased.identity(), ExecutionContextIdentity::of::<u64>());
    /// ```
    pub fn identity(&self) -> ExecutionContextIdentity {
        self.identity
    }

    /// Downcast to the requested execution-context type after checking identity.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{ErasedExecutionContext, ExecutionContextIdentity};
    ///
    /// let mut value = 1_u64;
    /// let mut erased = ErasedExecutionContext::new(&mut value);
    /// *erased.downcast_mut::<u64>(ExecutionContextIdentity::of::<u64>())? = 2;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionContextMismatch`] when the stored identity, supplied
    /// expected identity, and requested type do not all agree.
    pub fn downcast_mut<T: Send + Sync + 'static>(
        &mut self,
        expected: ExecutionContextIdentity,
    ) -> Result<&mut T, ExecutionContextMismatch> {
        let requested = ExecutionContextIdentity::of::<T>();
        if requested != expected {
            return Err(ExecutionContextMismatch {
                expected: requested,
                actual: self.identity,
            });
        }
        if self.identity != expected {
            return Err(ExecutionContextMismatch {
                expected,
                actual: self.identity,
            });
        }
        self.value
            .downcast_mut::<T>()
            .ok_or(ExecutionContextMismatch {
                expected,
                actual: self.identity,
            })
    }
}

impl fmt::Debug for ErasedExecutionContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ErasedExecutionContext")
            .field("identity", &self.identity)
            .finish_non_exhaustive()
    }
}

/// Runtime-created borrowed context passed to direct core providers.
///
/// # Examples
///
/// ```
/// use std::fmt::Debug;
/// use tenferro_runtime::CorePrepareContext;
///
/// fn requires_debug<T: Debug>() {}
/// requires_debug::<CorePrepareContext<'_>>();
/// ```
pub struct CorePrepareContext<'a> {
    binding: &'a PreparedOperationBinding,
    inputs: &'a InputSignature,
    resolved_placement: &'a ResolvedProgramPlacement,
    planning: &'a ResolvedPlanningConfig,
    prepare_options_key: &'a PrepareOptionsKey,
    specialization: &'a SpecializationProjection,
}

impl<'a> CorePrepareContext<'a> {
    #[allow(dead_code)]
    pub(crate) fn new(
        binding: &'a PreparedOperationBinding,
        inputs: &'a InputSignature,
        resolved_placement: &'a ResolvedProgramPlacement,
        planning: &'a ResolvedPlanningConfig,
        prepare_options_key: &'a PrepareOptionsKey,
        specialization: &'a SpecializationProjection,
    ) -> Self {
        Self {
            binding,
            inputs,
            resolved_placement,
            planning,
            prepare_options_key,
            specialization,
        }
    }

    /// Return the runtime-created binding.
    pub fn binding(&self) -> &'a PreparedOperationBinding {
        self.binding
    }

    /// Return value-free input metadata.
    pub fn inputs(&self) -> &'a InputSignature {
        self.inputs
    }

    /// Return the selected program placement.
    pub fn resolved_placement(&self) -> &'a ResolvedProgramPlacement {
        self.resolved_placement
    }

    /// Return resolved planning policy.
    pub fn planning(&self) -> &'a ResolvedPlanningConfig {
        self.planning
    }

    /// Return the normalized prepare-options key.
    pub fn prepare_options_key(&self) -> &'a PrepareOptionsKey {
        self.prepare_options_key
    }

    /// Return the concrete specialization projection.
    pub fn specialization(&self) -> &'a SpecializationProjection {
        self.specialization
    }
}

impl fmt::Debug for CorePrepareContext<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CorePrepareContext")
            .field("binding", self.binding)
            .field("inputs", &self.inputs.entries().len())
            .field("resolved_placement", self.resolved_placement)
            .field("planning", self.planning)
            .field("prepare_options_key", self.prepare_options_key)
            .field("specialization", self.specialization)
            .finish()
    }
}

macro_rules! request_type {
    ($name:ident) => {
        /// Runtime-created direct core preparation request.
        pub struct $name<'a> {
            operation: SemanticOperationView<'a>,
            context: &'a CorePrepareContext<'a>,
        }

        impl<'a> $name<'a> {
            #[allow(dead_code)]
            pub(crate) fn new(
                operation: SemanticOperationView<'a>,
                context: &'a CorePrepareContext<'a>,
            ) -> Self {
                Self { operation, context }
            }

            /// Return the immutable semantic operation view.
            pub fn operation(&self) -> SemanticOperationView<'a> {
                self.operation
            }

            /// Return the shared runtime preparation context.
            pub fn context(&self) -> &'a CorePrepareContext<'a> {
                self.context
            }
        }

        impl fmt::Debug for $name<'_> {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_struct(stringify!($name))
                    .field("operation", &self.operation)
                    .field("context", &self.context)
                    .finish()
            }
        }
    };
}

request_type!(ElementwisePrepareRequest);
request_type!(ReductionPrepareRequest);
request_type!(IndexingPrepareRequest);
request_type!(DotGeneralPrepareRequest);
request_type!(LayoutPrepareRequest);

/// Direct runtime provider for elementwise core operations.
pub trait ElementwiseRuntime: fmt::Debug + Send + Sync + 'static {
    /// Prepare one elementwise operation.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError`] when provider preparation fails.
    fn prepare(
        &self,
        request: ElementwisePrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

/// Direct runtime provider for reduction core operations.
pub trait ReductionRuntime: fmt::Debug + Send + Sync + 'static {
    /// Prepare one reduction operation.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError`] when provider preparation fails.
    fn prepare(
        &self,
        request: ReductionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

/// Direct runtime provider for indexing core operations.
pub trait IndexingRuntime: fmt::Debug + Send + Sync + 'static {
    /// Prepare one indexing operation.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError`] when provider preparation fails.
    fn prepare(
        &self,
        request: IndexingPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

/// Preparation-only direct runtime provider for dot-general operations.
pub trait DotGeneralPreparation: fmt::Debug + Send + Sync + 'static {
    /// Prepare one dot-general operation.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError`] when provider preparation fails.
    fn prepare(
        &self,
        request: DotGeneralPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}

/// Direct runtime provider for layout core operations.
pub trait LayoutRuntime: fmt::Debug + Send + Sync + 'static {
    /// Prepare one layout operation.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError`] when provider preparation fails.
    fn prepare(&self, request: LayoutPrepareRequest<'_>)
        -> Result<PrepareCapability, PrepareError>;
}

#[derive(Clone, Copy, Debug)]
enum ReservedSubgraphSlot {}

/// Direct core capability slots captured by a runtime snapshot.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::CoreCapabilityBundle;
///
/// assert!(CoreCapabilityBundle::builder().build().elementwise().is_none());
/// ```
#[derive(Clone, Default)]
pub struct CoreCapabilityBundle {
    elementwise: Option<Arc<dyn ElementwiseRuntime>>,
    reduction: Option<Arc<dyn ReductionRuntime>>,
    indexing: Option<Arc<dyn IndexingRuntime>>,
    dot_general: Option<Arc<dyn DotGeneralPreparation>>,
    layout: Option<Arc<dyn LayoutRuntime>>,
    reserved_subgraph: Option<ReservedSubgraphSlot>,
}

impl CoreCapabilityBundle {
    /// Return an empty direct-capability builder.
    pub fn builder() -> CoreCapabilityBundleBuilder {
        CoreCapabilityBundleBuilder::new()
    }

    /// Return the elementwise provider slot.
    pub fn elementwise(&self) -> Option<&Arc<dyn ElementwiseRuntime>> {
        self.elementwise.as_ref()
    }

    /// Return the reduction provider slot.
    pub fn reduction(&self) -> Option<&Arc<dyn ReductionRuntime>> {
        self.reduction.as_ref()
    }

    /// Return the indexing provider slot.
    pub fn indexing(&self) -> Option<&Arc<dyn IndexingRuntime>> {
        self.indexing.as_ref()
    }

    /// Return the dot-general preparation slot.
    pub fn dot_general(&self) -> Option<&Arc<dyn DotGeneralPreparation>> {
        self.dot_general.as_ref()
    }

    /// Return the layout provider slot.
    pub fn layout(&self) -> Option<&Arc<dyn LayoutRuntime>> {
        self.layout.as_ref()
    }
}

impl fmt::Debug for CoreCapabilityBundle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CoreCapabilityBundle")
            .field("elementwise", &self.elementwise.is_some())
            .field("reduction", &self.reduction.is_some())
            .field("indexing", &self.indexing.is_some())
            .field("dot_general", &self.dot_general.is_some())
            .field("layout", &self.layout.is_some())
            .field("reserved_subgraph", &self.reserved_subgraph.is_some())
            .finish()
    }
}

/// Builder for direct core capability slots.
///
/// Setters replace any previous value for the same slot and therefore are
/// infallible.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::CoreCapabilityBundleBuilder;
///
/// assert!(CoreCapabilityBundleBuilder::new().build().layout().is_none());
/// ```
#[derive(Clone, Default)]
pub struct CoreCapabilityBundleBuilder {
    elementwise: Option<Arc<dyn ElementwiseRuntime>>,
    reduction: Option<Arc<dyn ReductionRuntime>>,
    indexing: Option<Arc<dyn IndexingRuntime>>,
    dot_general: Option<Arc<dyn DotGeneralPreparation>>,
    layout: Option<Arc<dyn LayoutRuntime>>,
}

impl CoreCapabilityBundleBuilder {
    /// Return an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set or replace the elementwise slot.
    pub fn elementwise(&mut self, capability: Arc<dyn ElementwiseRuntime>) -> &mut Self {
        self.elementwise = Some(capability);
        self
    }

    /// Set or replace the reduction slot.
    pub fn reduction(&mut self, capability: Arc<dyn ReductionRuntime>) -> &mut Self {
        self.reduction = Some(capability);
        self
    }

    /// Set or replace the indexing slot.
    pub fn indexing(&mut self, capability: Arc<dyn IndexingRuntime>) -> &mut Self {
        self.indexing = Some(capability);
        self
    }

    /// Set or replace the dot-general preparation slot.
    pub fn dot_general(&mut self, capability: Arc<dyn DotGeneralPreparation>) -> &mut Self {
        self.dot_general = Some(capability);
        self
    }

    /// Set or replace the layout slot.
    pub fn layout(&mut self, capability: Arc<dyn LayoutRuntime>) -> &mut Self {
        self.layout = Some(capability);
        self
    }

    /// Build an immutable capability bundle.
    pub fn build(self) -> CoreCapabilityBundle {
        CoreCapabilityBundle {
            elementwise: self.elementwise,
            reduction: self.reduction,
            indexing: self.indexing,
            dot_general: self.dot_general,
            layout: self.layout,
            reserved_subgraph: None,
        }
    }
}

impl fmt::Debug for CoreCapabilityBundleBuilder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CoreCapabilityBundleBuilder")
            .field("elementwise", &self.elementwise.is_some())
            .field("reduction", &self.reduction.is_some())
            .field("indexing", &self.indexing.is_some())
            .field("dot_general", &self.dot_general.is_some())
            .field("layout", &self.layout.is_some())
            .finish()
    }
}
