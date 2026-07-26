use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use tenferro_ad::semantic_extension::{
    AdValue, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
    SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
use tenferro_einsum::Subscripts;
use tenferro_ops::ext_op::{ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp};
use tenferro_ops::SymDim;
#[cfg(feature = "autodiff")]
use tenferro_runtime::program::{ProgramValue, SemanticProgramBuilder};
use tenferro_runtime::{
    CoreCapabilityKind, EngineId, ErasedExecutionContext, Error as RuntimeError, ErrorPhase,
    ExecutionContextIdentity, ExtensionCacheStore, ExtensionEngine, ExtensionModule,
    ExtensionModuleId, ExtensionModuleRegistrar, ExtensionPlanningConfig, ExtensionPrepareRequest,
    PrepareCapability, PrepareError, PreparedOperation, PreparedOperationBinding,
    ProviderContractError, Result as RuntimeResult, RuntimeConfigError, SpecializationProjection,
    UnsupportedReason,
};
#[cfg(feature = "autodiff")]
use tenferro_tensor::TensorScalar;
use tenferro_tensor::{DType, Tensor, TensorBackend, TensorRead};

use crate::einsum::tropical_einsum_subscripts_with_argmax;
#[cfg(feature = "autodiff")]
use crate::einsum::TropicalArgmaxStep;
use crate::error::unsupported_dtype;
use crate::TropicalKind;

pub(crate) const TROPICAL_EINSUM_FAMILY_ID: &str = "tenferro-ext-tropical.einsum.v1";
#[cfg(feature = "autodiff")]
const TROPICAL_EINSUM_JVP_FAMILY_ID: &str = "tenferro-ext-tropical.einsum_jvp.v1";
#[cfg(feature = "autodiff")]
const TROPICAL_EINSUM_VJP_FAMILY_ID: &str = "tenferro-ext-tropical.einsum_vjp.v1";

fn invalid_config(op: &'static str, message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op, "configuration", message)
}

struct TropicalReferenceEngine<B: TensorBackend + 'static> {
    family_id: &'static str,
    engine_id: EngineId,
    _backend: PhantomData<fn() -> B>,
}

struct TropicalReferenceModule<B: TensorBackend + 'static> {
    family_id: &'static str,
    module_id: ExtensionModuleId,
    engine_id: EngineId,
    _backend: PhantomData<fn() -> B>,
}

#[derive(Debug)]
struct TropicalReferencePlanningConfig {
    family_id: &'static str,
}

struct TropicalReferencePreparedOperation<B: TensorBackend + 'static> {
    family_id: &'static str,
    binding: PreparedOperationBinding,
    specialization: SpecializationProjection,
    op: Arc<dyn ExtensionOp>,
    _backend: PhantomData<fn() -> B>,
}

fn tropical_reference_module_supports(family_id: &'static str, op: &dyn ExtensionOp) -> bool {
    match family_id {
        TROPICAL_EINSUM_FAMILY_ID => op.as_any().is::<TropicalEinsumOp>(),
        #[cfg(feature = "autodiff")]
        TROPICAL_EINSUM_JVP_FAMILY_ID => op.as_any().is::<TropicalEinsumJvpOp>(),
        #[cfg(feature = "autodiff")]
        TROPICAL_EINSUM_VJP_FAMILY_ID => op.as_any().is::<TropicalEinsumVjpOp>(),
        _ => false,
    }
}

fn unsupported_tropical_reference_payload(family_id: &'static str) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported(
        "tropical_extension",
        format!("family_id {family_id:?} has no tropical host-reference module payload"),
    )
}

fn execute_tropical_reference_payload(
    family_id: &'static str,
    op: &dyn ExtensionOp,
    inputs: &[&Tensor],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    match family_id {
        TROPICAL_EINSUM_FAMILY_ID => {
            let op = op
                .as_any()
                .downcast_ref::<TropicalEinsumOp>()
                .ok_or_else(|| unsupported_tropical_reference_payload(family_id))?;
            let result = tropical_einsum_subscripts_with_argmax(op.kind, inputs, &op.subscripts)?;
            Ok(vec![result.output])
        }
        #[cfg(feature = "autodiff")]
        TROPICAL_EINSUM_JVP_FAMILY_ID => {
            let op = op
                .as_any()
                .downcast_ref::<TropicalEinsumJvpOp>()
                .ok_or_else(|| unsupported_tropical_reference_payload(family_id))?;
            validate_tropical_jvp_inputs(inputs, &op.subscripts, &op.active_inputs)?;
            let primal = tropical_einsum_subscripts_with_argmax(
                op.kind,
                &[inputs[0], inputs[1]],
                &op.subscripts,
            )?;
            let step = single_argmax_step(&primal.argmax)?;
            match primal.output.dtype() {
                DType::F32 => {
                    execute_jvp_typed::<f32>(inputs, &op.subscripts, step, &op.active_inputs)
                        .map(|tensor| vec![tensor])
                }
                DType::F64 => {
                    execute_jvp_typed::<f64>(inputs, &op.subscripts, step, &op.active_inputs)
                        .map(|tensor| vec![tensor])
                }
                dtype => Err(unsupported_dtype("tropical_einsum_jvp", dtype)),
            }
        }
        #[cfg(feature = "autodiff")]
        TROPICAL_EINSUM_VJP_FAMILY_ID => {
            let op = op
                .as_any()
                .downcast_ref::<TropicalEinsumVjpOp>()
                .ok_or_else(|| unsupported_tropical_reference_payload(family_id))?;
            validate_tropical_vjp_inputs(inputs, &op.subscripts, op.active_input)?;
            let primal = tropical_einsum_subscripts_with_argmax(
                op.kind,
                &[inputs[0], inputs[1]],
                &op.subscripts,
            )?;
            let step = single_argmax_step(&primal.argmax)?;
            match inputs[op.active_input].dtype() {
                DType::F32 => {
                    execute_vjp_typed::<f32>(inputs, &op.subscripts, step, op.active_input)
                        .map(|tensor| vec![tensor])
                }
                DType::F64 => {
                    execute_vjp_typed::<f64>(inputs, &op.subscripts, step, op.active_input)
                        .map(|tensor| vec![tensor])
                }
                dtype => Err(unsupported_dtype("tropical_einsum_vjp", dtype)),
            }
        }
        _ => Err(unsupported_tropical_reference_payload(family_id)),
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for TropicalReferenceEngine<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TropicalReferenceEngine")
            .field("family_id", &self.family_id)
            .field("engine_id", &self.engine_id)
            .field("backend_type", &std::any::type_name::<B>())
            .finish()
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for TropicalReferenceModule<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TropicalReferenceModule")
            .field("family_id", &self.family_id)
            .field("module_id", &self.module_id)
            .field("engine_id", &self.engine_id)
            .field("backend_type", &std::any::type_name::<B>())
            .finish()
    }
}

impl<B: TensorBackend + 'static> fmt::Debug for TropicalReferencePreparedOperation<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TropicalReferencePreparedOperation")
            .field("family_id", &self.family_id)
            .field("binding", &self.binding)
            .field("specialization", &self.specialization)
            .field("backend_type", &std::any::type_name::<B>())
            .finish_non_exhaustive()
    }
}

impl<B: TensorBackend + 'static> ExtensionEngine for TropicalReferenceEngine<B> {
    fn family_id(&self) -> &'static str {
        self.family_id
    }

    fn engine_id(&self) -> &EngineId {
        &self.engine_id
    }

    fn context_identity(&self) -> ExecutionContextIdentity {
        ExecutionContextIdentity::of::<B>()
    }

    fn prepare(
        &self,
        request: ExtensionPrepareRequest<'_>,
    ) -> std::result::Result<PrepareCapability, PrepareError> {
        if request.operation().family_id() != self.family_id {
            return Err(PrepareError::ProviderContract {
                source: ProviderContractError::WrongOperationFamily {
                    expected: CoreCapabilityKind::Elementwise,
                    operation: self.family_id,
                },
            });
        }
        if !tropical_reference_module_supports(self.family_id, request.operation()) {
            return Ok(PrepareCapability::Unsupported(
                UnsupportedReason::Operation {
                    operation: self.family_id,
                },
            ));
        }
        Ok(PrepareCapability::Prepared(Arc::new(
            TropicalReferencePreparedOperation::<B> {
                family_id: self.family_id,
                binding: request.binding().clone(),
                specialization: request.specialization().clone(),
                op: request.operation().clone_arc(),
                _backend: PhantomData,
            },
        )))
    }
}

impl ExtensionPlanningConfig for TropicalReferencePlanningConfig {
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
            .is_some_and(|other| other.family_id == self.family_id)
    }

    fn retained_bytes(&self) -> usize {
        0
    }
}

impl<B: TensorBackend + 'static> PreparedOperation for TropicalReferencePreparedOperation<B> {
    fn binding(&self) -> &PreparedOperationBinding {
        &self.binding
    }

    fn specialization(&self) -> &SpecializationProjection {
        &self.specialization
    }

    fn retained_bytes(&self) -> usize {
        0
    }

    fn execute(
        &self,
        context: &mut ErasedExecutionContext<'_>,
        extension_caches: &mut ExtensionCacheStore,
        inputs: &[TensorRead<'_>],
    ) -> RuntimeResult<Vec<Tensor>> {
        let backend = context
            .downcast_mut::<B>(self.binding.context_identity())
            .map_err(|source| {
                RuntimeError::runtime_state_source("extension", ErrorPhase::Execution, source)
            })?;
        let mut ctx = tenferro_runtime::ExtensionExecutionContext::new(backend, extension_caches);
        let materialized_inputs = ctx.backend_mut().with_backend_session(|exec| {
            inputs
                .iter()
                .cloned()
                .map(|input| exec.to_contiguous_read(input))
                .collect::<tenferro_tensor::Result<Vec<_>>>()
        })?;
        let input_refs: Vec<&Tensor> = materialized_inputs.iter().collect();
        Ok(execute_tropical_reference_payload(
            self.family_id,
            self.op.as_ref(),
            &input_refs,
        )?)
    }
}

impl<B: TensorBackend + 'static> ExtensionModule for TropicalReferenceModule<B> {
    fn module_id(&self) -> &ExtensionModuleId {
        &self.module_id
    }

    fn configure(
        &self,
        registrar: &mut ExtensionModuleRegistrar<'_>,
    ) -> std::result::Result<(), tenferro_runtime::ExtensionModuleError> {
        registrar.register_engine(Arc::new(TropicalReferenceEngine::<B> {
            family_id: self.family_id,
            engine_id: self.engine_id.clone(),
            _backend: PhantomData,
        }))?;
        registrar.register_planning_config(
            self.engine_id.clone(),
            Arc::new(TropicalReferencePlanningConfig {
                family_id: self.family_id,
            }),
        )?;
        Ok(())
    }
}

fn reference_module<B: TensorBackend + 'static>(
    family_id: &'static str,
    engine_id: EngineId,
) -> std::result::Result<Arc<dyn ExtensionModule>, RuntimeConfigError> {
    Ok(Arc::new(TropicalReferenceModule::<B> {
        family_id,
        module_id: ExtensionModuleId::new(format!("{family_id}.module"))?,
        engine_id,
        _backend: PhantomData,
    }))
}

/// Build tropical extension modules for one runtime engine.
///
/// AD rules are registered separately through `tropical_ad_rules` when the
/// `autodiff` feature is enabled.
///
/// # Errors
///
/// Returns [`RuntimeConfigError`] when a generated module id is invalid.
pub fn extension_modules<B: TensorBackend + 'static>(
    engine_id: EngineId,
) -> std::result::Result<Vec<Arc<dyn ExtensionModule>>, RuntimeConfigError> {
    #[cfg(feature = "autodiff")]
    {
        Ok(vec![
            reference_module::<B>(TROPICAL_EINSUM_FAMILY_ID, engine_id.clone())?,
            reference_module::<B>(TROPICAL_EINSUM_JVP_FAMILY_ID, engine_id.clone())?,
            reference_module::<B>(TROPICAL_EINSUM_VJP_FAMILY_ID, engine_id)?,
        ])
    }
    #[cfg(not(feature = "autodiff"))]
    {
        Ok(vec![reference_module::<B>(
            TROPICAL_EINSUM_FAMILY_ID,
            engine_id,
        )?])
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TropicalEinsumOp {
    kind: TropicalKind,
    subscripts: Subscripts,
}

impl TropicalEinsumOp {
    pub(crate) fn new(kind: TropicalKind, subscripts: Subscripts) -> Self {
        Self { kind, subscripts }
    }
}

impl ExtensionOp for TropicalEinsumOp {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_kind(self.kind, hasher);
        hash_subscripts(&self.subscripts, hasher);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = [ctx.input_dtype(0)?, ctx.input_dtype(1)?];
        let input_shapes = [ctx.input_shape(0)?.to_vec(), ctx.input_shape(1)?.to_vec()];
        let meta = infer_tropical_output_meta(
            ctx,
            &self.subscripts,
            &input_dtypes,
            &input_shapes,
            "tropical_einsum",
        )?;
        Ok(vec![meta])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TropicalEinsumJvpOp {
    kind: TropicalKind,
    subscripts: Subscripts,
    active_inputs: Vec<usize>,
}

#[cfg(feature = "autodiff")]
impl TropicalEinsumJvpOp {
    fn new(kind: TropicalKind, subscripts: Subscripts, active_inputs: Vec<usize>) -> Self {
        Self {
            kind,
            subscripts,
            active_inputs,
        }
    }
}

#[cfg(feature = "autodiff")]
impl ExtensionOp for TropicalEinsumJvpOp {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_JVP_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_kind(self.kind, hasher);
        hash_subscripts(&self.subscripts, hasher);
        hasher.write_usize(self.active_inputs.len());
        for &active in &self.active_inputs {
            hasher.write_usize(active);
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        2 + self.active_inputs.len()
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = (0..self.input_count())
            .map(|input| ctx.input_dtype(input))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let input_shapes = (0..self.input_count())
            .map(|input| ctx.input_shape(input).map(<[_]>::to_vec))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let primal = infer_tropical_output_meta(
            ctx,
            &self.subscripts,
            &input_dtypes[..2],
            &input_shapes[..2],
            "tropical_einsum_jvp",
        )?;
        for (active_pos, &active) in self.active_inputs.iter().enumerate() {
            let tangent_idx = 2 + active_pos;
            if active >= 2 || input_dtypes[tangent_idx] != input_dtypes[active] {
                return Err(invalid_config(
                    "tropical_einsum_jvp",
                    "active input tangent dtype does not match primal dtype",
                ));
            }
            ctx.require_same_shape(tangent_idx, active)?;
        }
        Ok(vec![primal])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct TropicalEinsumVjpOp {
    kind: TropicalKind,
    subscripts: Subscripts,
    active_input: usize,
}

#[cfg(feature = "autodiff")]
impl TropicalEinsumVjpOp {
    fn new(kind: TropicalKind, subscripts: Subscripts, active_input: usize) -> Self {
        Self {
            kind,
            subscripts,
            active_input,
        }
    }
}

#[cfg(feature = "autodiff")]
impl ExtensionOp for TropicalEinsumVjpOp {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_VJP_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hash_kind(self.kind, hasher);
        hash_subscripts(&self.subscripts, hasher);
        hasher.write_usize(self.active_input);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        3
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if self.active_input >= 2 {
            return Err(invalid_config(
                "tropical_einsum_vjp",
                format!(
                    "expected active input < 2, got active_input={}",
                    self.active_input
                ),
            ));
        }
        let input_dtypes = [
            ctx.input_dtype(0)?,
            ctx.input_dtype(1)?,
            ctx.input_dtype(2)?,
        ];
        let input_shapes = [
            ctx.input_shape(0)?.to_vec(),
            ctx.input_shape(1)?.to_vec(),
            ctx.input_shape(2)?.to_vec(),
        ];
        let (_, primal_output_shape) = infer_tropical_output_meta(
            ctx,
            &self.subscripts,
            &input_dtypes[..2],
            &input_shapes[..2],
            "tropical_einsum_vjp",
        )?;
        if input_dtypes[2] != input_dtypes[self.active_input] {
            return Err(tenferro_tensor::Error::dtype_mismatch(
                "tropical_einsum_vjp",
                input_dtypes[self.active_input],
                input_dtypes[2],
            ));
        }
        if input_shapes[2].len() != primal_output_shape.len() {
            return Err(tenferro_tensor::Error::rank_mismatch(
                "tropical_einsum_vjp",
                primal_output_shape.len(),
                input_shapes[2].len(),
            ));
        }
        for (cotangent_dim, output_dim) in input_shapes[2].iter().cloned().zip(primal_output_shape)
        {
            ctx.require_equal(cotangent_dim, output_dim)?;
        }
        Ok(vec![(
            input_dtypes[self.active_input],
            input_shapes[self.active_input].clone(),
        )])
    }
}

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct TropicalEinsumAdRule;

#[cfg(feature = "autodiff")]
#[derive(Debug)]
struct TropicalEinsumJvpTransposeRule;

#[cfg(feature = "autodiff")]
impl SemanticLinearizeRule for TropicalEinsumAdRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_FAMILY_ID
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<SemanticLinearizeResult, SemanticAdError> {
        let op = semantic_primal_op(request.op(), SemanticAdRuleRole::Linearize)?;
        validate_semantic_ad_supported(&op.subscripts, SemanticAdRuleRole::Linearize)?;
        if !request.active_outputs()[0] {
            return Ok(SemanticLinearizeResult::new([AdValue::Absent], []));
        }
        let active_inputs: Vec<usize> = request
            .tangent_inputs()
            .iter()
            .enumerate()
            .filter_map(|(index, tangent)| matches!(tangent, AdValue::Value(_)).then_some(index))
            .collect();
        if active_inputs.is_empty() {
            return Ok(SemanticLinearizeResult::new([AdValue::Absent], []));
        }
        let mut inputs = request.primal_inputs().to_vec();
        inputs.extend(active_inputs.iter().filter_map(|&index| {
            request
                .tangent_inputs()
                .get(index)
                .copied()
                .and_then(AdValue::value)
        }));
        let tangent = builder.add_extension(
            Arc::new(TropicalEinsumJvpOp::new(
                op.kind,
                op.subscripts.clone(),
                active_inputs,
            )),
            &inputs,
        )?[0];
        Ok(SemanticLinearizeResult::new([AdValue::Value(tangent)], []))
    }
}

#[cfg(feature = "autodiff")]
impl SemanticLinearTransposeRule for TropicalEinsumAdRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_FAMILY_ID
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_tropical_vjp(
            request.op(),
            request.primal_inputs(),
            request.cotangent_outputs()[0],
            request.active_inputs(),
            builder,
            SemanticAdRuleRole::LinearTranspose,
        )
    }
}

#[cfg(feature = "autodiff")]
impl SemanticLinearTransposeRule for TropicalEinsumJvpTransposeRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_JVP_FAMILY_ID
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_tropical_jvp_vjp(
            request.op(),
            request.primal_inputs(),
            request.cotangent_outputs()[0],
            request.active_inputs(),
            builder,
            SemanticAdRuleRole::LinearTranspose,
        )
    }
}

#[cfg(feature = "autodiff")]
impl SemanticPrimalVjpRule for TropicalEinsumJvpTransposeRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_JVP_FAMILY_ID
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_tropical_jvp_vjp(
            request.op(),
            request.primal_inputs(),
            request.cotangent_outputs()[0],
            request.active_inputs(),
            builder,
            SemanticAdRuleRole::PrimalVjp,
        )
    }
}

#[cfg(feature = "autodiff")]
impl SemanticPrimalVjpRule for TropicalEinsumAdRule {
    fn family_id(&self) -> &'static str {
        TROPICAL_EINSUM_FAMILY_ID
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_tropical_vjp(
            request.op(),
            request.primal_inputs(),
            request.cotangent_outputs()[0],
            request.active_inputs(),
            builder,
            SemanticAdRuleRole::PrimalVjp,
        )
    }
}

/// Build semantic-program AD rules for tropical traced einsum extensions.
///
/// # Errors
///
/// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] if the
/// tropical family identifier is invalid, or
/// [`SemanticExtensionRegistryError::DuplicateRule`] if a semantic rule role
/// is already registered.
///
/// # Examples
///
/// ```
/// let rules = tenferro_ext_tropical::tropical_semantic_ad_rules().unwrap();
/// assert!(rules
///     .lookup_linearize("tenferro-ext-tropical.einsum.v1")
///     .is_some());
/// assert!(rules
///     .lookup_linear_transpose("tenferro-ext-tropical.einsum.v1")
///     .is_some());
/// assert!(rules
///     .lookup_linear_transpose("tenferro-ext-tropical.einsum_jvp.v1")
///     .is_some());
/// assert!(rules
///     .lookup_primal_vjp("tenferro-ext-tropical.einsum_jvp.v1")
///     .is_some());
/// assert!(rules
///     .lookup_primal_vjp("tenferro-ext-tropical.einsum.v1")
///     .is_some());
/// ```
#[cfg(feature = "autodiff")]
pub fn tropical_semantic_ad_rules(
) -> Result<SemanticExtensionRuleSet, SemanticExtensionRegistryError> {
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(TropicalEinsumAdRule))?
        .with_linear_transpose(Arc::new(TropicalEinsumAdRule))?
        .with_linear_transpose(Arc::new(TropicalEinsumJvpTransposeRule))?
        .with_primal_vjp(Arc::new(TropicalEinsumJvpTransposeRule))?
        .with_primal_vjp(Arc::new(TropicalEinsumAdRule))
}

#[cfg(feature = "autodiff")]
fn semantic_tropical_vjp(
    op: &dyn ExtensionOp,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
    let op = semantic_primal_op(op, role)?;
    validate_semantic_ad_supported(&op.subscripts, role)?;
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(vec![AdValue::Absent; op.input_count()].into_boxed_slice());
    };
    active_inputs
        .iter()
        .copied()
        .enumerate()
        .map(|(active_input, active)| {
            if !active {
                return Ok(AdValue::Absent);
            }
            let mut inputs = primal_inputs.to_vec();
            inputs.push(cotangent);
            let value = builder.add_extension(
                Arc::new(TropicalEinsumVjpOp::new(
                    op.kind,
                    op.subscripts.clone(),
                    active_input,
                )),
                &inputs,
            )?[0];
            Ok(AdValue::Value(value))
        })
        .collect()
}

#[cfg(feature = "autodiff")]
fn semantic_primal_op(
    op: &dyn ExtensionOp,
    role: SemanticAdRuleRole,
) -> std::result::Result<&TropicalEinsumOp, SemanticAdError> {
    op.as_any()
        .downcast_ref::<TropicalEinsumOp>()
        .ok_or_else(|| SemanticAdError::Unsupported {
            family_id: TROPICAL_EINSUM_FAMILY_ID,
            role,
            message: "tropical einsum semantic AD received an incompatible payload".into(),
        })
}

#[cfg(feature = "autodiff")]
fn semantic_tropical_jvp_vjp(
    op: &dyn ExtensionOp,
    primal_inputs: &[ProgramValue],
    cotangent: AdValue,
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
    role: SemanticAdRuleRole,
) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
    let op = semantic_jvp_op(op, role)?;
    validate_semantic_ad_supported(&op.subscripts, role)?;
    if active_inputs.iter().take(2).any(|active| *active) {
        return Err(SemanticAdError::Unsupported {
            family_id: TROPICAL_EINSUM_JVP_FAMILY_ID,
            role,
            message: "tropical einsum JVP semantic AD supports transpose only for tangent inputs"
                .into(),
        });
    }
    let mut cotangent_inputs = vec![AdValue::Absent; op.input_count()];
    let AdValue::Value(cotangent) = cotangent else {
        return Ok(cotangent_inputs.into_boxed_slice());
    };
    for (active_pos, &active_input) in op.active_inputs.iter().enumerate() {
        let tangent_input = 2 + active_pos;
        if !active_inputs[tangent_input] {
            continue;
        }
        let mut inputs = primal_inputs[..2].to_vec();
        inputs.push(cotangent);
        let value = builder.add_extension(
            Arc::new(TropicalEinsumVjpOp::new(
                op.kind,
                op.subscripts.clone(),
                active_input,
            )),
            &inputs,
        )?[0];
        cotangent_inputs[tangent_input] = AdValue::Value(value);
    }
    Ok(cotangent_inputs.into_boxed_slice())
}

#[cfg(feature = "autodiff")]
fn semantic_jvp_op(
    op: &dyn ExtensionOp,
    role: SemanticAdRuleRole,
) -> std::result::Result<&TropicalEinsumJvpOp, SemanticAdError> {
    op.as_any()
        .downcast_ref::<TropicalEinsumJvpOp>()
        .ok_or_else(|| SemanticAdError::Unsupported {
            family_id: TROPICAL_EINSUM_JVP_FAMILY_ID,
            role,
            message: "tropical einsum JVP semantic AD received an incompatible payload".into(),
        })
}

#[cfg(feature = "autodiff")]
fn validate_semantic_ad_supported(
    subscripts: &Subscripts,
    role: SemanticAdRuleRole,
) -> std::result::Result<(), SemanticAdError> {
    let message = if subscripts.inputs.len() != 2 {
        Some("tropical einsum AD supports only binary inputs")
    } else if has_repeated_labels(&subscripts.output)
        || subscripts
            .inputs
            .iter()
            .any(|labels| has_repeated_labels(labels))
    {
        Some("tropical einsum AD does not support repeated labels")
    } else if !subscripts.inputs[0]
        .iter()
        .any(|label| subscripts.inputs[1].contains(label) && !subscripts.output.contains(label))
    {
        Some("tropical einsum AD requires contracted modes")
    } else {
        None
    };
    if let Some(message) = message {
        return Err(SemanticAdError::Unsupported {
            family_id: TROPICAL_EINSUM_FAMILY_ID,
            role,
            message: message.into(),
        });
    }
    Ok(())
}

fn infer_tropical_output_meta(
    ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    subscripts: &Subscripts,
    input_dtypes: &[DType],
    input_shapes: &[Vec<SymDim>],
    op: &'static str,
) -> tenferro_tensor::Result<(DType, Vec<SymDim>)> {
    if input_shapes.len() != 2 || input_dtypes.len() != input_shapes.len() {
        return Err(invalid_config(
            op,
            format!(
                "tropical einsum expected two input metadata records, got dtypes={} shapes={}",
                input_dtypes.len(),
                input_shapes.len()
            ),
        ));
    }
    if subscripts.inputs.len() != 2 {
        return Err(invalid_config(op, "tropical einsum requires two inputs"));
    }
    if input_dtypes[0] != input_dtypes[1] {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            op,
            input_dtypes[0],
            input_dtypes[1],
        ));
    }
    if !matches!(input_dtypes[0], DType::F32 | DType::F64) {
        return Err(unsupported_dtype(op, input_dtypes[0]));
    }

    let mut label_dims: HashMap<u32, SymDim> = HashMap::new();
    for (labels, shape) in subscripts.inputs.iter().zip(input_shapes.iter()) {
        if labels.len() != shape.len() {
            return Err(tenferro_tensor::Error::rank_mismatch(
                op,
                labels.len(),
                shape.len(),
            ));
        }
        for (&label, dim) in labels.iter().zip(shape.iter()) {
            if let Some(existing) = label_dims.get(&label) {
                ctx.require_equal(existing.clone(), dim.clone())?;
            } else {
                label_dims.insert(label, dim.clone());
            }
        }
    }
    let output_shape = subscripts
        .output
        .iter()
        .map(|label| label_dims.get(label).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| invalid_config(op, "output labels must be present in input metadata"))?;
    Ok((input_dtypes[0], output_shape))
}

#[cfg(feature = "autodiff")]
fn validate_tropical_primal_host_meta(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    op: &'static str,
) -> tenferro_tensor::Result<Vec<usize>> {
    if inputs.len() != 2 || subscripts.inputs.len() != 2 {
        return Err(invalid_config(
            op,
            format!(
                "expected two primal inputs, got tensors={} subscripts={}",
                inputs.len(),
                subscripts.inputs.len()
            ),
        ));
    }
    let dtype = inputs[0].dtype();
    if inputs[1].dtype() != dtype {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            op,
            dtype,
            inputs[1].dtype(),
        ));
    }
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(unsupported_dtype(op, dtype));
    }

    let mut label_dims = HashMap::new();
    for (labels, input) in subscripts.inputs.iter().zip(inputs) {
        if labels.len() != input.shape().len() {
            return Err(tenferro_tensor::Error::rank_mismatch(
                op,
                labels.len(),
                input.shape().len(),
            ));
        }
        for (&label, &dim) in labels.iter().zip(input.shape()) {
            if let Some(existing) = label_dims.insert(label, dim) {
                if existing != dim {
                    return Err(tenferro_tensor::Error::shape_mismatch(
                        op,
                        vec![existing],
                        vec![dim],
                    ));
                }
            }
        }
    }
    subscripts
        .output
        .iter()
        .map(|label| {
            label_dims
                .get(label)
                .copied()
                .ok_or_else(|| invalid_config(op, format!("output label {label} is absent")))
        })
        .collect()
}

#[cfg(feature = "autodiff")]
fn validate_tropical_jvp_inputs(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    active_inputs: &[usize],
) -> tenferro_tensor::Result<()> {
    let expected = 2 + active_inputs.len();
    if inputs.len() != expected {
        return Err(invalid_config(
            "tropical_einsum_jvp",
            format!("expected {expected} inputs, got {}", inputs.len()),
        ));
    }
    validate_tropical_primal_host_meta(&inputs[..2], subscripts, "tropical_einsum_jvp")?;
    for (active_pos, &active) in active_inputs.iter().enumerate() {
        if active >= 2 {
            return Err(invalid_config(
                "tropical_einsum_jvp",
                format!("invalid active input {active}"),
            ));
        }
        let tangent = inputs[2 + active_pos];
        if tangent.dtype() != inputs[active].dtype() {
            return Err(tenferro_tensor::Error::dtype_mismatch(
                "tropical_einsum_jvp",
                inputs[active].dtype(),
                tangent.dtype(),
            ));
        }
        if tangent.shape() != inputs[active].shape() {
            return Err(tenferro_tensor::Error::shape_mismatch(
                "tropical_einsum_jvp",
                inputs[active].shape().to_vec(),
                tangent.shape().to_vec(),
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "autodiff")]
fn validate_tropical_vjp_inputs(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    active_input: usize,
) -> tenferro_tensor::Result<()> {
    if inputs.len() != 3 {
        return Err(invalid_config(
            "tropical_einsum_vjp",
            format!("expected 3 inputs, got {}", inputs.len()),
        ));
    }
    if active_input >= 2 {
        return Err(invalid_config(
            "tropical_einsum_vjp",
            format!("invalid active input {active_input}"),
        ));
    }
    let output_shape =
        validate_tropical_primal_host_meta(&inputs[..2], subscripts, "tropical_einsum_vjp")?;
    let cotangent = inputs[2];
    if cotangent.dtype() != inputs[active_input].dtype() {
        return Err(tenferro_tensor::Error::dtype_mismatch(
            "tropical_einsum_vjp",
            inputs[active_input].dtype(),
            cotangent.dtype(),
        ));
    }
    if cotangent.shape() != output_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            "tropical_einsum_vjp",
            output_shape,
            cotangent.shape().to_vec(),
        ));
    }
    Ok(())
}

#[cfg(feature = "autodiff")]
fn has_repeated_labels(labels: &[u32]) -> bool {
    labels
        .iter()
        .enumerate()
        .any(|(idx, label)| labels[..idx].contains(label))
}

#[cfg(feature = "autodiff")]
fn execute_jvp_typed<T>(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    step: &TropicalArgmaxStep,
    active_inputs: &[usize],
) -> tenferro_tensor::Result<Tensor>
where
    T: TensorScalar + Copy + Default + std::ops::AddAssign,
{
    let output_shape = step.output_shape().to_vec();
    let output_len = element_count(&output_shape)?;
    let mut out = vec![T::default(); output_len];
    for (active_pos, &active_input) in active_inputs.iter().enumerate() {
        let tangent = typed_slice::<T>(inputs[2 + active_pos])?;
        let labels = &subscripts.inputs[active_input];
        for (output_index, out_value) in out.iter_mut().enumerate() {
            let offset = routed_input_offset(
                step,
                labels,
                inputs[active_input].shape(),
                output_index,
                "tropical_einsum_jvp",
            )?;
            let tangent_value = tangent.get(offset).ok_or_else(|| {
                invalid_config("tropical_einsum_jvp", "tangent offset is out of bounds")
            })?;
            *out_value += *tangent_value;
        }
    }
    Tensor::from_vec_col_major(output_shape, out)
}

#[cfg(feature = "autodiff")]
fn execute_vjp_typed<T>(
    inputs: &[&Tensor],
    subscripts: &Subscripts,
    step: &TropicalArgmaxStep,
    active_input: usize,
) -> tenferro_tensor::Result<Tensor>
where
    T: TensorScalar + Copy + Default + std::ops::AddAssign,
{
    let cotangent = typed_slice::<T>(inputs[2])?;
    let output_len = element_count(step.output_shape())?;
    if cotangent.len() != output_len {
        return Err(invalid_config(
            "tropical_einsum_vjp",
            format!(
                "cotangent length {} does not match tropical output length {output_len}",
                cotangent.len()
            ),
        ));
    }
    let active_shape = inputs[active_input].shape().to_vec();
    let mut out = vec![T::default(); element_count(&active_shape)?];
    let labels = &subscripts.inputs[active_input];
    for (output_index, &ct) in cotangent.iter().enumerate() {
        let offset = routed_input_offset(
            step,
            labels,
            &active_shape,
            output_index,
            "tropical_einsum_vjp",
        )?;
        let slot = out.get_mut(offset).ok_or_else(|| {
            invalid_config("tropical_einsum_vjp", "scatter offset is out of bounds")
        })?;
        *slot += ct;
    }
    Tensor::from_vec_col_major(active_shape, out)
}

#[cfg(feature = "autodiff")]
fn routed_input_offset(
    step: &TropicalArgmaxStep,
    input_labels: &[u32],
    input_shape: &[usize],
    output_index: usize,
    op: &'static str,
) -> tenferro_tensor::Result<usize> {
    if input_labels.len() != input_shape.len() {
        return Err(invalid_config(
            op,
            "input labels do not match active input rank",
        ));
    }
    let output_coords = decode_col_major_index(output_index, step.output_shape())
        .ok_or_else(|| invalid_config(op, "output index is outside argmax output shape"))?;
    let winner_coords = step
        .winner_coordinates(output_index)
        .ok_or_else(|| invalid_config(op, "argmax winner is outside contracted shape"))?;
    let strides = col_major_strides(input_shape)?;
    input_labels
        .iter()
        .zip(strides.iter())
        .try_fold(0usize, |offset, (&label, &stride)| {
            let coordinate = if let Some(axis) = step
                .output_subscripts()
                .iter()
                .position(|candidate| *candidate == label)
            {
                output_coords[axis]
            } else if let Some(axis) = step
                .contracted_subscripts()
                .iter()
                .position(|candidate| *candidate == label)
            {
                winner_coords[axis]
            } else {
                return Err(invalid_config(
                    op,
                    format!("input label {label} requires unsupported pre-reduction"),
                ));
            };
            offset
                .checked_add(coordinate.checked_mul(stride).ok_or_else(|| {
                    invalid_config(op, "routed offset multiplication overflows usize")
                })?)
                .ok_or_else(|| invalid_config(op, "routed offset addition overflows usize"))
        })
}

#[cfg(feature = "autodiff")]
fn typed_slice<T>(tensor: &Tensor) -> tenferro_tensor::Result<&[T]>
where
    T: TensorScalar,
{
    tensor.as_slice::<T>()
}

#[cfg(feature = "autodiff")]
fn single_argmax_step(
    steps: &[TropicalArgmaxStep],
) -> tenferro_tensor::Result<&TropicalArgmaxStep> {
    if steps.len() != 1 {
        return Err(invalid_config(
            "tropical_einsum_ad",
            format!("expected one argmax step, got {}", steps.len()),
        ));
    }
    Ok(&steps[0])
}

fn hash_kind(kind: TropicalKind, hasher: &mut dyn Hasher) {
    match kind {
        TropicalKind::MaxPlus => hasher.write_u8(0),
        TropicalKind::MinPlus => hasher.write_u8(1),
    }
}

fn hash_subscripts(subscripts: &Subscripts, hasher: &mut dyn Hasher) {
    hasher.write_usize(subscripts.inputs.len());
    for input in &subscripts.inputs {
        hasher.write_usize(input.len());
        for label in input {
            hasher.write_u32(*label);
        }
    }
    hasher.write_usize(subscripts.output.len());
    for label in &subscripts.output {
        hasher.write_u32(*label);
    }
}

#[cfg(feature = "autodiff")]
fn col_major_strides(shape: &[usize]) -> tenferro_tensor::Result<Vec<usize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| invalid_config("tropical_einsum_ad", "shape overflows usize"))?;
    }
    Ok(strides)
}

#[cfg(feature = "autodiff")]
fn element_count(shape: &[usize]) -> tenferro_tensor::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &extent| {
        acc.checked_mul(extent)
            .ok_or_else(|| invalid_config("tropical_einsum_ad", "shape overflows usize"))
    })
}

#[cfg(feature = "autodiff")]
fn decode_col_major_index(mut flat: usize, shape: &[usize]) -> Option<Vec<usize>> {
    let total = shape
        .iter()
        .try_fold(1usize, |acc, &extent| acc.checked_mul(extent))?;
    if flat >= total {
        return None;
    }
    let mut coordinates = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            return None;
        }
        coordinates.push(flat % extent);
        flat /= extent;
    }
    Some(coordinates)
}

#[cfg(all(test, feature = "autodiff"))]
mod tests;
