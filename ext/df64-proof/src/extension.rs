//! A minimal extension-owned operation for the external scalar.
//!
//! The operation is an `ExtensionOp` family whose payload, numerical body, and
//! output construction all live in this crate. The runtime reaches it through a
//! registered `ExtensionModule` and prepared execution, and the input travels as
//! a runtime `Tensor` that carries a caller-owned payload.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ad::extension::{apply_eager_with_extension_session, ExtensionOp};
use tenferro_ad::EagerTensor;
use tenferro_cpu::{scalar_fold, CpuBackend};
use tenferro_ops::{ExtensionShapeContext, SymDim};
use tenferro_runtime::{
    EngineId, ErasedExecutionContext, ExecutionContextIdentity, ExtensionCacheStore,
    ExtensionEngine, ExtensionModule, ExtensionModuleError, ExtensionModuleId,
    ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest, PrepareCapability,
    PrepareError, PreparedOperation, PreparedOperationBinding, PreparedOperationExecutor,
    PreparedOperationExecutorHandle, PreparedOperationHandle, PreparedOperationPlan,
    SpecializationProjection,
};
use tenferro_tensor::{DType, Tensor, TensorRead};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor, Scalar};

use crate::{Df64, Df64Add};

/// Family identifier of the extension-owned total sum.
pub const DF64_TOTAL_FAMILY: &str = "tenferro-df64-proof.df64_total.v1";

/// Total sum of an externally defined scalar tensor.
///
/// The payload carries no parameters, so every instance is equal to every other.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::Df64Total;
/// use tenferro_ad::extension::ExtensionOp;
///
/// assert_eq!(<Df64Total as ExtensionOp>::input_count(&Df64Total), 1);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Df64Total;

impl ExtensionOp for Df64Total {
    fn family_id(&self) -> &'static str {
        DF64_TOTAL_FAMILY
    }

    fn payload_hash(&self, _hasher: &mut dyn Hasher) {}

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    /// The body reads its input and writes only its own output.
    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    /// The total is a new value rather than a view of the input.
    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::External(_)) {
            // The body is only defined for the external scalar, so anything else
            // fails explicitly instead of being coerced.
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "df64_total",
                dtype,
                "df64_total takes an externally defined scalar",
            ));
        }
        // A total sum has rank zero.
        Ok(vec![(dtype, Vec::new())])
    }
}

fn external_payload<T: Scalar>(tensor: &Tensor) -> tenferro_tensor::Result<&HostTensor<T>> {
    match tensor {
        Tensor::External(payload, _) => payload.downcast_ref::<T>().ok_or_else(|| {
            tenferro_tensor::Error::unsupported_dtype(
                "df64_total",
                tensor.dtype(),
                "the external payload does not hold the expected scalar",
            )
        }),
        other => Err(tenferro_tensor::Error::unsupported_dtype(
            "df64_total",
            other.dtype(),
            "df64_total takes an externally defined payload",
        )),
    }
}

fn total_of(inputs: &[TensorRead<'_>]) -> tenferro_runtime::Result<Vec<Tensor>> {
    let tensor = match inputs.first() {
        Some(TensorRead::Tensor(tensor)) => *tensor,
        Some(TensorRead::View(_)) => {
            return Err(tenferro_runtime::Error::from(
                tenferro_tensor::Error::invalid_argument(
                    "df64_total",
                    "input",
                    "df64_total takes an owned externally defined payload",
                ),
            ));
        }
        None => {
            return Err(tenferro_runtime::Error::from(
                tenferro_tensor::Error::invalid_argument(
                    "df64_total",
                    "input",
                    "df64_total takes one input",
                ),
            ));
        }
    };
    let payload = external_payload::<Df64>(tensor).map_err(tenferro_runtime::Error::from)?;
    let total = scalar_fold::<Df64, Df64Add>("df64_total", payload, Df64::zero())
        .map_err(tenferro_runtime::Error::from)?;
    let output = HostTensor::from_vec_col_major(vec![], vec![total])
        .map_err(|source| tenferro_tensor::Error::validation("df64_total", source))
        .map_err(tenferro_runtime::Error::from)?;
    Ok(vec![Tensor::external(ErasedHostTensor::new(output))])
}

#[derive(Debug)]
struct Df64TotalPrepared {
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
}

impl PreparedOperation for Df64TotalPrepared {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

impl PreparedOperationExecutor for Df64TotalPrepared {
    fn execute(
        &self,
        _context: &mut ErasedExecutionContext<'_>,
        _caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        total_of(inputs)
    }

    fn supports_session(&self) -> bool {
        true
    }

    fn execute_in_session(
        &self,
        _session: &mut dyn tenferro_tensor::BackendSession,
        _caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> tenferro_runtime::Result<Vec<Tensor>> {
        total_of(inputs)
    }
}

#[derive(Debug)]
struct Df64TotalEngine {
    engine_id: EngineId,
}

impl ExtensionEngine for Df64TotalEngine {
    fn family_id(&self) -> &'static str {
        DF64_TOTAL_FAMILY
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<CpuBackend>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> Result<PrepareCapability, PrepareError> {
        let prepared = Arc::new(Df64TotalPrepared {
            binding: request.binding().clone(),
            specialization: request.specialization().clone(),
        });
        let operation: PreparedOperationHandle = Arc::clone(&prepared) as PreparedOperationHandle;
        let executor: PreparedOperationExecutorHandle = prepared as PreparedOperationExecutorHandle;
        Ok(PrepareCapability::Prepared(
            PreparedOperationPlan::executable(operation, executor),
        ))
    }
}

#[derive(Debug)]
struct Df64TotalConfig {
    family_id: &'static str,
}

impl ExtensionPlanningConfig for Df64TotalConfig {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn payload_hash(&self, state: &mut dyn Hasher) {
        state.write(self.family_id.as_bytes());
    }

    fn payload_eq(&self, other: &dyn ExtensionPlanningConfig) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self.family_id == other.family_id)
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

#[derive(Debug)]
struct Df64TotalModule {
    module_id: ExtensionModuleId,
    engine_id: EngineId,
}

impl ExtensionModule for Df64TotalModule {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> Result<(), ExtensionModuleError> {
        registrar.register_engine(Arc::new(Df64TotalEngine {
            engine_id: self.engine_id.clone(),
        }))?;
        registrar.register_planning_config(
            self.engine_id.clone(),
            Arc::new(Df64TotalConfig {
                family_id: DF64_TOTAL_FAMILY,
            }),
        )
    }
}

/// The module a downstream application installs to enable [`Df64Total`].
///
/// # Errors
///
/// Returns an error when the CPU runtime engine identifier is unavailable in this
/// process.
///
/// # Examples
///
/// ```rust
/// use tenferro_df64_proof::extension::module;
///
/// assert!(module().is_ok());
/// ```
pub fn module() -> Result<Arc<dyn ExtensionModule>, tenferro_runtime::RuntimeConfigError> {
    Ok(Arc::new(Df64TotalModule {
        module_id: ExtensionModuleId::new("tenferro-df64-proof.module")?,
        engine_id: tenferro_cpu::runtime_engine_id()?,
    }))
}

/// Run [`Df64Total`] on an eagerly held external tensor.
///
/// # Errors
///
/// Returns the runtime's typed error when the extension cannot be prepared or
/// executed for this input.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::ExtensionOp;
/// use tenferro_df64_proof::extension::{module, Df64Total};
///
/// assert_eq!(<Df64Total as ExtensionOp>::input_count(&Df64Total), 1);
/// assert!(module().is_ok());
/// ```
///
/// Executing the operation needs the caller-owned payload ownership contract from
/// #1789, which `ext/df64-proof/tests/extension_execution.rs` records.
pub fn apply_total(input: &EagerTensor) -> tenferro_runtime::Result<Vec<EagerTensor>> {
    let module = module().map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "df64_total",
            tenferro_runtime::ErrorPhase::Execution,
            source,
        )
    })?;
    apply_eager_with_extension_session(Arc::new(Df64Total), &[input], module)
}
