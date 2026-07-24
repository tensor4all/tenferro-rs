use std::any::Any;
use std::fmt;
use std::hash::Hasher;

use tenferro_ops::ext_op::ExtensionOp;

use super::{
    ExecutionContextIdentity, HardwareClassId, InputSignature, PrepareCapability, PrepareError,
    PrepareOptionsKey, PreparedOperationBinding, ResolvedPlanningConfig, ResolvedProgramPlacement,
    SpecializationProjection,
};

/// Snapshot-retained extension planning configuration.
///
/// The family id follows the existing extension `&'static str` contract used by
/// [`ExtensionOp::family_id`].
pub trait ExtensionPlanningConfig: Any + fmt::Debug + Send + Sync + 'static {
    /// Return the extension family id this configuration belongs to.
    fn family_id(&self) -> &'static str;
    /// Return this config as [`Any`] for typed equality checks.
    fn as_any(&self) -> &dyn Any;
    /// Hash only the family-specific payload.
    fn payload_hash(&self, state: &mut dyn Hasher);
    /// Compare only the family-specific payload.
    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool;
    /// Return logical retained bytes owned by this config.
    fn retained_bytes(&self) -> usize;
}

/// Runtime-created borrowed extension preparation request.
pub struct ExtensionPrepareRequest<'a> {
    operation: &'a dyn ExtensionOp,
    binding: &'a PreparedOperationBinding,
    resolved_placement: &'a ResolvedProgramPlacement,
    hardware_class: &'a HardwareClassId,
    planning: &'a ResolvedPlanningConfig,
    extension_config: &'a dyn ExtensionPlanningConfig,
    inputs: &'a InputSignature,
    prepare_options_key: &'a PrepareOptionsKey,
    specialization: &'a SpecializationProjection,
}

impl<'a> ExtensionPrepareRequest<'a> {
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        operation: &'a dyn ExtensionOp,
        binding: &'a PreparedOperationBinding,
        resolved_placement: &'a ResolvedProgramPlacement,
        hardware_class: &'a HardwareClassId,
        planning: &'a ResolvedPlanningConfig,
        extension_config: &'a dyn ExtensionPlanningConfig,
        inputs: &'a InputSignature,
        prepare_options_key: &'a PrepareOptionsKey,
        specialization: &'a SpecializationProjection,
    ) -> Self {
        Self {
            operation,
            binding,
            resolved_placement,
            hardware_class,
            planning,
            extension_config,
            inputs,
            prepare_options_key,
            specialization,
        }
    }

    /// Return the extension operation payload.
    pub fn operation(&self) -> &'a dyn ExtensionOp {
        self.operation
    }

    /// Return the runtime-created binding.
    pub fn binding(&self) -> &'a PreparedOperationBinding {
        self.binding
    }

    /// Return the selected program placement.
    pub fn resolved_placement(&self) -> &'a ResolvedProgramPlacement {
        self.resolved_placement
    }

    /// Return the selected hardware class.
    pub fn hardware_class(&self) -> &'a HardwareClassId {
        self.hardware_class
    }

    /// Return resolved planning policy.
    pub fn planning(&self) -> &'a ResolvedPlanningConfig {
        self.planning
    }

    /// Return the snapshot-retained extension planning config.
    pub fn extension_config(&self) -> &'a dyn ExtensionPlanningConfig {
        self.extension_config
    }

    /// Return value-free input metadata.
    pub fn inputs(&self) -> &'a InputSignature {
        self.inputs
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

impl fmt::Debug for ExtensionPrepareRequest<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExtensionPrepareRequest")
            .field("operation", &self.operation.family_id())
            .field("binding", self.binding)
            .field("resolved_placement", self.resolved_placement)
            .field("hardware_class", self.hardware_class)
            .field("planning", self.planning)
            .field("extension_config", &self.extension_config.family_id())
            .field("inputs", &self.inputs.entries().len())
            .field("prepare_options_key", self.prepare_options_key)
            .field("specialization", self.specialization)
            .finish()
    }
}

/// Preparation provider for one extension family.
pub trait ExtensionEngine: fmt::Debug + Send + Sync + 'static {
    /// Return the extension family id handled by this engine.
    fn family_id(&self) -> &'static str;
    /// Return the runtime engine id used by this provider.
    fn engine_id(&self) -> &super::EngineId;
    /// Return the execution-context type identity needed by this provider.
    fn context_identity(&self) -> ExecutionContextIdentity;
    /// Prepare an extension operation.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError`] when provider preparation fails.
    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError>;
}
