use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
#[cfg(feature = "autodiff")]
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::graph::GraphBuilder;
use computegraph::types::ValueRef;
#[cfg(feature = "autodiff")]
use tenferro_ad::semantic_extension::{
    AdValue, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRegistryError,
    SemanticExtensionRuleSet, SemanticLinearTransposeRequest, SemanticLinearTransposeRule,
    SemanticLinearizeRequest, SemanticLinearizeResult, SemanticLinearizeRule,
    SemanticPrimalVjpRequest, SemanticPrimalVjpRule,
};
use tenferro_extension_macros::define_extension_runtime;
#[cfg(feature = "autodiff")]
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::{
    ExtensionLoweringError, ExtensionLoweringResult, ExtensionOp, ExtensionStandardLowering,
};
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_runtime::extension::{ExtensionCacheKey, ExtensionExecutionContext};
#[cfg(feature = "autodiff")]
use tenferro_runtime::program::{CoreSemanticOp, ProgramValue, SemanticProgramBuilder};
use tenferro_tensor::{
    BackendSession, DType, Error as TensorError, Tensor, TensorBackend, TensorRead,
};

use crate::builder::build_einsum_graph;
use crate::cache::{
    einsum_subscripts_retained_bytes, saturating_sum, vec_retained_bytes,
    EINSUM_EXTENSION_FAMILY_ID, EINSUM_RUNTIME_PLANS_CACHE,
};
#[cfg(test)]
use crate::optimize::default_auto_options;
#[cfg(feature = "autodiff")]
use crate::optimize::jax_path_to_v1_pairs;
use crate::optimize::{hash_einsum_plan_spec, plan_specs_equal, resolve_plan_spec, EinsumPlanSpec};
#[cfg(feature = "autodiff")]
use crate::util::map_label_occurrences;
use crate::{
    ContractionTree, EinsumSubscripts, Error as EinsumError, Result as EinsumResult, Subscripts,
};

/// Standard einsum extension payload.
///
/// This mirrors the current `tenferro.einsum.v1` payload shape. Runtime-owned
/// execution goes through [`EinsumRuntime`].
#[derive(Clone)]
pub(crate) struct EinsumExtensionOp {
    subscripts: EinsumSubscripts,
    plan_spec: EinsumPlanSpec,
    output_shape_hint: Option<Vec<SymDim>>,
}

impl std::fmt::Debug for EinsumExtensionOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EinsumExtensionOp")
            .field("subscripts", &self.subscripts)
            .field("plan_spec", &self.plan_spec)
            .field("output_shape_hint", &self.output_shape_hint)
            .finish()
    }
}

impl EinsumExtensionOp {
    /// Create an einsum extension payload without a precomputed plan.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn new(subscripts: EinsumSubscripts) -> Self {
        Self::with_plan_spec(subscripts, EinsumPlanSpec::Auto(default_auto_options()))
    }

    #[must_use]
    pub(crate) fn with_plan_spec(subscripts: EinsumSubscripts, plan_spec: EinsumPlanSpec) -> Self {
        Self {
            subscripts,
            plan_spec,
            output_shape_hint: None,
        }
    }

    /// Create an einsum extension payload with an explicit output shape hint.
    #[must_use]
    #[cfg(any(feature = "autodiff", test))]
    pub(crate) fn with_output_shape_hint(
        subscripts: EinsumSubscripts,
        output_shape_hint: Vec<SymDim>,
        plan_spec: EinsumPlanSpec,
    ) -> Self {
        let mut op = Self::with_plan_spec(subscripts, plan_spec);
        op.output_shape_hint = Some(output_shape_hint);
        op
    }

    /// Return the canonical subscripts.
    #[must_use]
    pub(crate) fn subscripts(&self) -> &EinsumSubscripts {
        &self.subscripts
    }

    /// Return the shape-independent planning policy.
    #[must_use]
    pub(crate) fn plan_spec(&self) -> &EinsumPlanSpec {
        &self.plan_spec
    }
}

impl ExtensionOp for EinsumExtensionOp {
    fn family_id(&self) -> &'static str {
        EINSUM_EXTENSION_FAMILY_ID
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_usize(self.subscripts.inputs.len());
        for input in &self.subscripts.inputs {
            hasher.write_usize(input.len());
            for label in input {
                hasher.write_u32(*label);
            }
        }
        hasher.write_usize(self.subscripts.output.len());
        for label in &self.subscripts.output {
            hasher.write_u32(*label);
        }
        hash_einsum_plan_spec(self.plan_spec(), hasher);
        if let Some(shape) = &self.output_shape_hint {
            hasher.write_usize(shape.len());
            for dim in shape {
                match dim.constant_value() {
                    Some(value) => {
                        hasher.write_u8(1);
                        hasher.write_usize(value);
                    }
                    None => hasher.write_u8(0),
                }
            }
        } else {
            hasher.write_usize(usize::MAX);
        }
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>().is_some_and(|that| {
            self.subscripts == that.subscripts
                && plan_specs_equal(self.plan_spec(), that.plan_spec())
                && self.output_shape_hint == that.output_shape_hint
        })
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        self.subscripts.inputs.len()
    }

    fn output_count(&self) -> usize {
        1
    }

    fn semantic_effects(&self) -> tenferro_ops::ext_op::ExtensionEffectDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionEffectDeclaration::Declared(&[])
    }

    fn semantic_aliases(&self) -> tenferro_ops::ext_op::ExtensionAliasDeclaration<'_> {
        tenferro_ops::ext_op::ExtensionAliasDeclaration::AllFresh
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        let input_dtypes = (0..self.input_count())
            .map(|input| ctx.input_dtype(input))
            .collect::<Result<Vec<_>, _>>()?;
        let input_shapes = (0..self.input_count())
            .map(|input| ctx.input_shape(input).map(<[_]>::to_vec))
            .collect::<Result<Vec<_>, _>>()?;

        let mut label_dims: HashMap<u32, SymDim> = HashMap::new();
        for (labels, shape) in self.subscripts.inputs.iter().zip(input_shapes.iter()) {
            if labels.len() != shape.len() {
                return Err(TensorError::rank_mismatch(
                    "einsum",
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

        let output_shape = match &self.output_shape_hint {
            Some(shape) if shape.iter().all(|dim| dim.constant_value().is_some()) => shape.clone(),
            _ => self
                .subscripts
                .output
                .iter()
                .map(|label| label_dims.get(label).cloned())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    TensorError::invalid_argument(
                        "einsum",
                        "output labels",
                        "must be present in input metadata",
                    )
                })?,
        };
        if output_shape.len() != self.subscripts.output.len() {
            return Err(TensorError::rank_mismatch(
                "einsum",
                self.subscripts.output.len(),
                output_shape.len(),
            ));
        }
        Ok(vec![(
            promote_dtypes(input_dtypes.iter().copied()),
            output_shape,
        )])
    }

    fn lower_to_standard_ops(
        &self,
        builder: &mut GraphBuilder<StdTensorOp>,
        inputs: &[ValueRef<StdTensorOp>],
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> ExtensionLoweringResult {
        if inputs.len() != self.input_count()
            || input_dtypes.len() != self.input_count()
            || input_shapes.len() != self.input_count()
        {
            return Err(ExtensionLoweringError::new(format!(
                "einsum extension expects {} inputs, got values={}, dtypes={}, shapes={}",
                self.input_count(),
                inputs.len(),
                input_dtypes.len(),
                input_shapes.len()
            )));
        }

        let Some(shapes) = concrete_sym_shape_slices(input_shapes) else {
            return Ok(ExtensionStandardLowering::Unsupported);
        };
        let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let subs = Subscripts::from(&self.subscripts);
        let tree = resolve_plan_spec(self.plan_spec(), &subs, &shape_refs).map_err(|source| {
            ExtensionLoweringError::from_source_with_kind(source.kind(), source)
        })?;
        let output = build_einsum_graph(builder, &tree, inputs, &shapes).map_err(|source| {
            ExtensionLoweringError::from_source_with_kind(source.kind(), source)
        })?;
        Ok(ExtensionStandardLowering::Lowered(vec![output]))
    }
}

fn concrete_sym_shape_slices(input_shapes: &[&[SymDim]]) -> Option<Vec<Vec<usize>>> {
    input_shapes
        .iter()
        .map(|shape| {
            shape
                .iter()
                .map(SymDim::constant_value)
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

/// Return the semantic-program einsum extension AD rules.
#[cfg(feature = "autodiff")]
///
/// # Errors
///
/// Returns [`SemanticExtensionRegistryError::MalformedFamilyId`] for an
/// invalid family identifier, or
/// [`SemanticExtensionRegistryError::DuplicateRule`] for a duplicate role.
pub fn semantic_ad_rules(
) -> std::result::Result<SemanticExtensionRuleSet, SemanticExtensionRegistryError> {
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(EinsumAdRule))?
        .with_linear_transpose(Arc::new(EinsumAdRule))?
        .with_primal_vjp(Arc::new(EinsumAdRule))
}

#[derive(Debug)]
#[cfg(feature = "autodiff")]
struct EinsumAdRule;

#[cfg(feature = "autodiff")]
impl SemanticLinearizeRule for EinsumAdRule {
    fn family_id(&self) -> &'static str {
        EINSUM_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<SemanticLinearizeResult, SemanticAdError> {
        let op = semantic_einsum_payload(request.op(), SemanticAdRuleRole::Linearize)?;
        if !request.active_outputs()[0] {
            return Ok(SemanticLinearizeResult::new([AdValue::Absent], []));
        }
        let mut terms = Vec::new();
        for (active_idx, tangent) in request.tangent_inputs().iter().copied().enumerate() {
            let AdValue::Value(tangent) = tangent else {
                continue;
            };
            let inputs: Vec<_> = request
                .primal_inputs()
                .iter()
                .copied()
                .enumerate()
                .map(|(input_idx, primal)| {
                    if input_idx == active_idx {
                        tangent
                    } else {
                        primal
                    }
                })
                .collect();
            terms.push(builder.add_extension(Arc::new(op.clone()), &inputs)?[0]);
        }
        let tangent = semantic_sum_terms(builder, terms)?;
        Ok(SemanticLinearizeResult::new([tangent], []))
    }
}

#[cfg(feature = "autodiff")]
impl SemanticLinearTransposeRule for EinsumAdRule {
    fn family_id(&self) -> &'static str {
        EINSUM_EXTENSION_FAMILY_ID
    }

    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_einsum_vjp(
            request.op(),
            request.primal_inputs(),
            request.primal_outputs(),
            request.cotangent_outputs(),
            request.active_inputs(),
            builder,
        )
    }
}

#[cfg(feature = "autodiff")]
impl SemanticPrimalVjpRule for EinsumAdRule {
    fn family_id(&self) -> &'static str {
        EINSUM_EXTENSION_FAMILY_ID
    }

    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
        semantic_einsum_vjp(
            request.op(),
            request.primal_inputs(),
            request.primal_outputs(),
            request.cotangent_outputs(),
            request.active_inputs(),
            builder,
        )
    }
}

#[cfg(feature = "autodiff")]
fn semantic_einsum_vjp(
    payload: &dyn ExtensionOp,
    primal_inputs: &[ProgramValue],
    primal_outputs: &[ProgramValue],
    cotangent_outputs: &[AdValue],
    active_inputs: &[bool],
    builder: &mut SemanticProgramBuilder,
) -> std::result::Result<Box<[AdValue]>, SemanticAdError> {
    let op = semantic_einsum_payload(payload, SemanticAdRuleRole::LinearTranspose)?;
    let input_count = op.subscripts.inputs.len();
    let AdValue::Value(cotangent) = cotangent_outputs[0] else {
        return Ok(vec![AdValue::Absent; input_count].into_boxed_slice());
    };
    let primal_input_shapes = primal_inputs
        .iter()
        .copied()
        .map(|value| semantic_value_shape(builder, value))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let cotangent_shape = semantic_value_shape(builder, primal_outputs[0])?;

    let input_labels = &op.subscripts.inputs;
    let output_labels = &op.subscripts.output;
    let mut result = Vec::with_capacity(input_count);
    for active_idx in 0..input_count {
        if !active_inputs[active_idx] {
            result.push(AdValue::Absent);
            continue;
        }
        let mut available_labels: HashSet<u32> = output_labels.iter().copied().collect();
        for (input_idx, labels) in input_labels.iter().enumerate() {
            if input_idx != active_idx {
                available_labels.extend(labels.iter().copied());
            }
        }
        let vjp_output_labels: Vec<u32> = input_labels[active_idx]
            .iter()
            .copied()
            .filter(|label| available_labels.contains(label))
            .collect();
        let mut vjp_input_labels = vec![output_labels.clone()];
        let mut vjp_inputs = vec![cotangent];
        let mut vjp_input_shapes = vec![cotangent_shape.clone()];
        for input_idx in 0..input_count {
            if input_idx == active_idx {
                continue;
            }
            vjp_input_labels.push(input_labels[input_idx].clone());
            vjp_input_shapes.push(primal_input_shapes[input_idx].clone());
            vjp_inputs.push(semantic_conjugate_if_complex(
                builder,
                primal_inputs[input_idx],
            )?);
        }
        let vjp_op = semantic_vjp_einsum_op(
            op,
            active_idx,
            EinsumSubscripts {
                inputs: vjp_input_labels,
                output: vjp_output_labels.clone(),
            },
            &vjp_input_shapes,
        )?;
        let mut input_cotangent = builder.add_extension(Arc::new(vjp_op), &vjp_inputs)?[0];
        if vjp_output_labels != input_labels[active_idx] {
            input_cotangent = semantic_broadcast_einsum_vjp(
                builder,
                input_cotangent,
                &vjp_output_labels,
                &input_labels[active_idx],
                primal_input_shapes[active_idx].clone(),
            )?;
        }
        result.push(AdValue::Value(input_cotangent));
    }
    Ok(result.into_boxed_slice())
}

#[cfg(feature = "autodiff")]
fn semantic_vjp_einsum_op(
    primal_op: &EinsumExtensionOp,
    active_idx: usize,
    subscripts: EinsumSubscripts,
    input_shapes: &[Vec<DimExpr>],
) -> std::result::Result<EinsumExtensionOp, SemanticAdError> {
    let plan_spec =
        vjp_plan_spec_for_active(primal_op.plan_spec(), primal_op.input_count(), active_idx)?;
    let sym_shapes: Vec<Vec<SymDim>> = input_shapes
        .iter()
        .enumerate()
        .map(|(input_idx, shape)| {
            let tensor_id = u64::MAX - input_idx as u64;
            shape
                .iter()
                .enumerate()
                .map(|(axis, dim)| match dim {
                    DimExpr::Const(value) => SymDim::from(*value),
                    _ => SymDim::tensor_axis(tensor_id, axis),
                })
                .collect()
        })
        .collect();
    if let Some(concrete_shapes) = concrete_sym_shapes(&sym_shapes) {
        let shape_refs: Vec<&[usize]> = concrete_shapes.iter().map(Vec::as_slice).collect();
        let raw_subscripts = Subscripts::from(&subscripts);
        let _tree = resolve_plan_spec(&plan_spec, &raw_subscripts, &shape_refs)
            .map_err(|source| semantic_einsum_unsupported(source.to_string()))?;
    }
    Ok(EinsumExtensionOp::with_plan_spec(subscripts, plan_spec))
}

#[cfg(feature = "autodiff")]
fn semantic_value_shape(
    builder: &SemanticProgramBuilder,
    value: ProgramValue,
) -> std::result::Result<Vec<DimExpr>, SemanticAdError> {
    builder
        .value_metadata(value)?
        .shape()
        .iter()
        .map(|extent| {
            extent.bound_expr().cloned().ok_or_else(|| {
                semantic_einsum_unsupported(
                    "einsum semantic AD requires a symbolic expression for every extent",
                )
            })
        })
        .collect()
}

#[cfg(feature = "autodiff")]
fn semantic_conjugate_if_complex(
    builder: &mut SemanticProgramBuilder,
    value: ProgramValue,
) -> std::result::Result<ProgramValue, SemanticAdError> {
    if matches!(
        builder.value_metadata(value)?.dtype(),
        DType::C32 | DType::C64
    ) {
        Ok(builder.add_op(CoreSemanticOp::Conj, &[value])?[0])
    } else {
        Ok(value)
    }
}

#[cfg(feature = "autodiff")]
fn semantic_broadcast_einsum_vjp(
    builder: &mut SemanticProgramBuilder,
    cotangent: ProgramValue,
    cotangent_labels: &[u32],
    input_labels: &[u32],
    shape: Vec<DimExpr>,
) -> std::result::Result<ProgramValue, SemanticAdError> {
    let dims = map_label_occurrences(cotangent_labels, input_labels).ok_or_else(|| {
        semantic_einsum_unsupported(format!(
            "einsum VJP cannot remap labels {cotangent_labels:?} into {input_labels:?}"
        ))
    })?;
    let broadcast =
        builder.add_op(CoreSemanticOp::BroadcastInDim { shape, dims }, &[cotangent])?[0];
    semantic_project_repeated_labels(builder, broadcast, input_labels)
}

#[cfg(feature = "autodiff")]
fn semantic_project_repeated_labels(
    builder: &mut SemanticProgramBuilder,
    cotangent: ProgramValue,
    labels: &[u32],
) -> std::result::Result<ProgramValue, SemanticAdError> {
    let mut result = cotangent;
    let mut first_axis_by_label = HashMap::new();
    for (axis_b, label) in labels.iter().copied().enumerate() {
        let Some(&axis_a) = first_axis_by_label.get(&label) else {
            first_axis_by_label.insert(label, axis_b);
            continue;
        };
        let extracted =
            builder.add_op(CoreSemanticOp::ExtractDiag { axis_a, axis_b }, &[result])?[0];
        result = builder.add_op(CoreSemanticOp::EmbedDiag { axis_a, axis_b }, &[extracted])?[0];
    }
    Ok(result)
}

#[cfg(feature = "autodiff")]
fn semantic_sum_terms(
    builder: &mut SemanticProgramBuilder,
    terms: Vec<ProgramValue>,
) -> std::result::Result<AdValue, SemanticAdError> {
    let mut terms = terms.into_iter();
    let Some(mut sum) = terms.next() else {
        return Ok(AdValue::Absent);
    };
    for term in terms {
        sum = builder.add_op(CoreSemanticOp::Add, &[sum, term])?[0];
    }
    Ok(AdValue::Value(sum))
}

#[cfg(feature = "autodiff")]
fn semantic_einsum_payload(
    op: &dyn ExtensionOp,
    role: SemanticAdRuleRole,
) -> std::result::Result<&EinsumExtensionOp, SemanticAdError> {
    op.as_any()
        .downcast_ref::<EinsumExtensionOp>()
        .ok_or_else(|| SemanticAdError::Unsupported {
            family_id: EINSUM_EXTENSION_FAMILY_ID,
            role,
            message: "einsum semantic AD received an incompatible payload".into(),
        })
}

#[cfg(feature = "autodiff")]
fn semantic_einsum_unsupported(message: impl Into<String>) -> SemanticAdError {
    SemanticAdError::Unsupported {
        family_id: EINSUM_EXTENSION_FAMILY_ID,
        role: SemanticAdRuleRole::LinearTranspose,
        message: message.into(),
    }
}

#[cfg(feature = "autodiff")]
fn vjp_plan_spec_for_active(
    primal_plan: &EinsumPlanSpec,
    input_count: usize,
    active_idx: usize,
) -> std::result::Result<EinsumPlanSpec, SemanticAdError> {
    if active_idx >= input_count {
        return Err(semantic_einsum_unsupported(format!(
            "einsum VJP active input {active_idx} is outside {input_count} inputs"
        )));
    }

    match primal_plan {
        EinsumPlanSpec::Auto(options) => Ok(EinsumPlanSpec::Auto(options.clone())),
        EinsumPlanSpec::LeftToRight => Ok(EinsumPlanSpec::LeftToRight),
        EinsumPlanSpec::Path(path) => {
            let pairs = jax_path_to_v1_pairs(path, input_count).map_err(|err| {
                semantic_einsum_unsupported(format!(
                    "failed to inherit einsum Path plan for VJP active input {active_idx}: {err}"
                ))
            })?;
            derive_vjp_fixed_pairs(&pairs, input_count, active_idx).map(EinsumPlanSpec::FixedPairs)
        }
        EinsumPlanSpec::FixedPairs(pairs) => {
            derive_vjp_fixed_pairs(pairs, input_count, active_idx).map(EinsumPlanSpec::FixedPairs)
        }
    }
}

#[cfg(feature = "autodiff")]
fn derive_vjp_fixed_pairs(
    primal_pairs: &[(usize, usize)],
    input_count: usize,
    active_idx: usize,
) -> std::result::Result<Vec<(usize, usize)>, SemanticAdError> {
    if input_count == 0 {
        return Err(semantic_einsum_unsupported(
            "einsum VJP cannot derive a plan for zero primal inputs",
        ));
    }
    if active_idx >= input_count {
        return Err(semantic_einsum_unsupported(format!(
            "einsum VJP active input {active_idx} is outside {input_count} inputs"
        )));
    }
    let required_steps = input_count.saturating_sub(1);
    if primal_pairs.len() != required_steps {
        return Err(semantic_einsum_unsupported(format!(
            "einsum VJP cannot inherit explicit plan for active input {active_idx}: \
             expected {required_steps} primal steps for {input_count} inputs, got {}",
            primal_pairs.len()
        )));
    }
    if input_count == 1 {
        return Ok(Vec::new());
    }

    let children = fixed_pair_children(primal_pairs, input_count, active_idx)?;
    let mut primal_to_vjp = vec![None; input_count];
    let mut next_vjp_input = 1;
    for (input_idx, slot) in primal_to_vjp.iter_mut().enumerate() {
        if input_idx != active_idx {
            *slot = Some(next_vjp_input);
            next_vjp_input += 1;
        }
    }

    let root = input_count + primal_pairs.len() - 1;
    let mut pairs = Vec::with_capacity(required_steps);
    let final_id = emit_vjp_adjoint(
        root,
        0,
        &children,
        input_count,
        active_idx,
        &primal_to_vjp,
        &mut pairs,
    )?;
    let expected_final = input_count + pairs.len() - 1;
    if final_id != expected_final || pairs.len() != required_steps {
        return Err(semantic_einsum_unsupported(format!(
            "einsum VJP plan derivation for active input {active_idx} produced an invalid \
             tree: final id {final_id}, expected {expected_final}, steps {}",
            pairs.len()
        )));
    }
    Ok(pairs)
}

#[cfg(feature = "autodiff")]
fn fixed_pair_children(
    pairs: &[(usize, usize)],
    input_count: usize,
    active_idx: usize,
) -> std::result::Result<Vec<Option<(usize, usize)>>, SemanticAdError> {
    let mut live = vec![false; input_count + pairs.len()];
    for slot in live.iter_mut().take(input_count) {
        *slot = true;
    }
    let mut children = vec![None; input_count + pairs.len()];

    for (step_idx, &(left, right)) in pairs.iter().enumerate() {
        let next_idx = input_count + step_idx;
        if left == right {
            return Err(invalid_vjp_plan_error(
                active_idx,
                format!("pair ({left}, {right}) references the same operand"),
            ));
        }
        if left >= next_idx || right >= next_idx {
            return Err(invalid_vjp_plan_error(
                active_idx,
                format!("pair ({left}, {right}) references a non-existent operand"),
            ));
        }
        if !live[left] || !live[right] {
            return Err(invalid_vjp_plan_error(
                active_idx,
                format!("pair ({left}, {right}) references an operand that is no longer live"),
            ));
        }

        live[left] = false;
        live[right] = false;
        live[next_idx] = true;
        children[next_idx] = Some((left, right));
    }

    let live_count = live.iter().filter(|&&is_live| is_live).count();
    if live_count != 1 {
        return Err(invalid_vjp_plan_error(
            active_idx,
            format!("explicit plan leaves {live_count} live operands"),
        ));
    }

    Ok(children)
}

#[cfg(feature = "autodiff")]
fn emit_vjp_adjoint(
    node: usize,
    cotangent_id: usize,
    children: &[Option<(usize, usize)>],
    input_count: usize,
    active_idx: usize,
    primal_to_vjp: &[Option<usize>],
    pairs: &mut Vec<(usize, usize)>,
) -> std::result::Result<usize, SemanticAdError> {
    if node < input_count {
        return if node == active_idx {
            Ok(cotangent_id)
        } else {
            Err(invalid_vjp_plan_error(
                active_idx,
                format!("adjoint walk reached inactive leaf {node}"),
            ))
        };
    }

    let (left, right) = children.get(node).and_then(|child| *child).ok_or_else(|| {
        invalid_vjp_plan_error(active_idx, format!("missing children for node {node}"))
    })?;
    let left_has_active = subtree_contains_active(left, children, input_count, active_idx)?;
    let right_has_active = subtree_contains_active(right, children, input_count, active_idx)?;
    match (left_has_active, right_has_active) {
        (true, false) => {
            let sibling_id = emit_vjp_subtree(
                right,
                children,
                input_count,
                active_idx,
                primal_to_vjp,
                pairs,
            )?;
            let next = push_vjp_pair(cotangent_id, sibling_id, input_count, pairs);
            emit_vjp_adjoint(
                left,
                next,
                children,
                input_count,
                active_idx,
                primal_to_vjp,
                pairs,
            )
        }
        (false, true) => {
            let sibling_id = emit_vjp_subtree(
                left,
                children,
                input_count,
                active_idx,
                primal_to_vjp,
                pairs,
            )?;
            let next = push_vjp_pair(cotangent_id, sibling_id, input_count, pairs);
            emit_vjp_adjoint(
                right,
                next,
                children,
                input_count,
                active_idx,
                primal_to_vjp,
                pairs,
            )
        }
        (true, true) => Err(invalid_vjp_plan_error(
            active_idx,
            format!("both children of node {node} contain the active input"),
        )),
        (false, false) => Err(invalid_vjp_plan_error(
            active_idx,
            format!("neither child of node {node} contains the active input"),
        )),
    }
}

#[cfg(feature = "autodiff")]
fn emit_vjp_subtree(
    node: usize,
    children: &[Option<(usize, usize)>],
    input_count: usize,
    active_idx: usize,
    primal_to_vjp: &[Option<usize>],
    pairs: &mut Vec<(usize, usize)>,
) -> std::result::Result<usize, SemanticAdError> {
    if node < input_count {
        return primal_to_vjp[node].ok_or_else(|| {
            invalid_vjp_plan_error(
                active_idx,
                format!("sibling subtree unexpectedly reached active leaf {node}"),
            )
        });
    }

    let (left, right) = children.get(node).and_then(|child| *child).ok_or_else(|| {
        invalid_vjp_plan_error(active_idx, format!("missing children for node {node}"))
    })?;
    let left_id = emit_vjp_subtree(
        left,
        children,
        input_count,
        active_idx,
        primal_to_vjp,
        pairs,
    )?;
    let right_id = emit_vjp_subtree(
        right,
        children,
        input_count,
        active_idx,
        primal_to_vjp,
        pairs,
    )?;
    Ok(push_vjp_pair(left_id, right_id, input_count, pairs))
}

#[cfg(feature = "autodiff")]
fn push_vjp_pair(
    left: usize,
    right: usize,
    n_vjp_inputs: usize,
    pairs: &mut Vec<(usize, usize)>,
) -> usize {
    pairs.push((left, right));
    n_vjp_inputs + pairs.len() - 1
}

#[cfg(feature = "autodiff")]
fn subtree_contains_active(
    node: usize,
    children: &[Option<(usize, usize)>],
    input_count: usize,
    active_idx: usize,
) -> std::result::Result<bool, SemanticAdError> {
    if node < input_count {
        return Ok(node == active_idx);
    }
    let (left, right) = children.get(node).and_then(|child| *child).ok_or_else(|| {
        invalid_vjp_plan_error(active_idx, format!("missing children for node {node}"))
    })?;
    Ok(
        subtree_contains_active(left, children, input_count, active_idx)?
            || subtree_contains_active(right, children, input_count, active_idx)?,
    )
}

#[cfg(feature = "autodiff")]
fn invalid_vjp_plan_error(active_idx: usize, reason: String) -> SemanticAdError {
    semantic_einsum_unsupported(format!(
        "einsum VJP cannot inherit explicit plan for active input {active_idx}: {reason}"
    ))
}

#[cfg(feature = "autodiff")]
fn concrete_sym_shapes(shapes: &[Vec<SymDim>]) -> Option<Vec<Vec<usize>>> {
    shapes
        .iter()
        .map(|shape| shape.iter().map(SymDim::constant_value).collect())
        .collect()
}

define_extension_runtime! {
    runtime = EinsumRuntime,
    family_id = EINSUM_EXTENSION_FAMILY_ID,
    op_type = EinsumExtensionOp,
    execute = execute_einsum_extension,
    execute_reads = execute_einsum_extension_reads,
}

fn execute_einsum_extension<B: TensorBackend + 'static>(
    op: &EinsumExtensionOp,
    inputs: &[&Tensor],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(tenferro_tensor::Error::invalid_argument(
            "einsum_extension",
            "inputs",
            "einsum requires at least one input tensor",
        ));
    }

    let shapes: Vec<Vec<usize>> = inputs
        .iter()
        .map(|tensor| tensor.shape().to_vec())
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::from(op.subscripts());
    let tree = cached_runtime_tree(ctx, op.subscripts(), op.plan_spec(), &shapes, || {
        resolve_plan_spec(op.plan_spec(), &subs, &shape_refs)
    })?;

    let output = ctx
        .backend_mut()
        .with_backend_session(|exec| crate::eager::eager_einsum_exec(exec, inputs, &tree))?;
    Ok(vec![output])
}

pub(crate) fn execute_einsum_extension_reads<B: TensorBackend + 'static>(
    op: &EinsumExtensionOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs
        .iter()
        .all(|input| matches!(input, TensorRead::Tensor(_)))
    {
        let input_refs: Vec<&Tensor> = inputs
            .iter()
            .map(|input| match input {
                TensorRead::Tensor(tensor) => *tensor,
                TensorRead::View(_) => unreachable!("view input filtered above"),
            })
            .collect();
        return execute_einsum_extension(op, &input_refs, ctx);
    }

    if inputs.is_empty() {
        return Err(tenferro_tensor::Error::invalid_argument(
            "einsum_extension",
            "inputs",
            "einsum requires at least one input tensor",
        ));
    }

    let shapes: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape().to_vec()).collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::from(op.subscripts());
    let tree = cached_runtime_tree(ctx, op.subscripts(), op.plan_spec(), &shapes, || {
        resolve_plan_spec(op.plan_spec(), &subs, &shape_refs)
    })?;
    let output = ctx
        .backend_mut()
        .with_backend_session(|exec| crate::eager::eager_einsum_exec_read(exec, inputs, &tree))?;
    Ok(vec![output])
}

pub(crate) fn execute_einsum_extension_session_reads(
    op: &EinsumExtensionOp,
    inputs: &[TensorRead<'_>],
    ctx: &mut ExtensionExecutionContext<'_, dyn BackendSession + '_>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(tenferro_tensor::Error::invalid_argument(
            "einsum_extension",
            "inputs",
            "einsum requires at least one input tensor",
        ));
    }

    let shapes: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape().to_vec()).collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::from(op.subscripts());
    let tree = cached_runtime_tree(ctx, op.subscripts(), op.plan_spec(), &shapes, || {
        resolve_plan_spec(op.plan_spec(), &subs, &shape_refs)
    })?;
    let output = crate::eager::eager_einsum_exec_read(ctx.backend_mut(), inputs, &tree)?;
    Ok(vec![output])
}

#[derive(Clone)]
struct RuntimeTreeCacheKeyData {
    subscripts: EinsumSubscripts,
    shapes: Vec<Vec<usize>>,
    plan_spec: EinsumPlanSpec,
}

impl RuntimeTreeCacheKeyData {
    fn new(
        subscripts: &EinsumSubscripts,
        shapes: &[Vec<usize>],
        plan_spec: &EinsumPlanSpec,
    ) -> Self {
        Self {
            subscripts: subscripts.clone(),
            shapes: shapes.to_vec(),
            plan_spec: plan_spec.clone(),
        }
    }

    fn matches_runtime_tree(
        &self,
        subscripts: &EinsumSubscripts,
        shapes: &[Vec<usize>],
        plan_spec: &EinsumPlanSpec,
    ) -> bool {
        self.subscripts == *subscripts
            && self.shapes.as_slice() == shapes
            && plan_specs_equal(&self.plan_spec, plan_spec)
    }

    fn retained_bytes(&self) -> usize {
        saturating_sum([
            einsum_subscripts_retained_bytes(&self.subscripts),
            saturating_sum(self.shapes.iter().map(vec_retained_bytes)),
            plan_spec_retained_bytes(&self.plan_spec),
        ])
    }
}

struct CachedRuntimeTree {
    key_data: RuntimeTreeCacheKeyData,
    tree: Arc<ContractionTree>,
}

fn cached_runtime_tree<B: BackendSession + ?Sized>(
    ctx: &mut ExtensionExecutionContext<'_, B>,
    subscripts: &EinsumSubscripts,
    plan_spec: &EinsumPlanSpec,
    shapes: &[Vec<usize>],
    build: impl FnOnce() -> EinsumResult<ContractionTree>,
) -> tenferro_tensor::Result<Arc<ContractionTree>> {
    let plan_hash = plan_spec_hash(plan_spec);
    let key = ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_RUNTIME_PLANS_CACHE,
        runtime_tree_cache_discriminator(subscripts, shapes, plan_hash),
    );
    if let Some(cached) = ctx.caches_mut().get::<CachedRuntimeTree>(&key) {
        let key_data = &cached.key_data;
        if key_data.matches_runtime_tree(subscripts, shapes, plan_spec) {
            return Ok(Arc::clone(&cached.tree));
        }
    }

    let tree = Arc::new(build().map_err(einsum_runtime_error)?);
    let key_data = RuntimeTreeCacheKeyData::new(subscripts, shapes, plan_spec);
    let retained_bytes = saturating_sum([
        key_data.retained_bytes(),
        tree.retained_bytes_for_cache_stats(),
    ]);
    ctx.caches_mut().put(
        key,
        CachedRuntimeTree {
            key_data,
            tree: Arc::clone(&tree),
        },
        retained_bytes,
    );
    Ok(tree)
}

fn einsum_runtime_error(error: EinsumError) -> tenferro_tensor::Error {
    error.into_tensor_error("einsum_extension")
}

fn runtime_tree_cache_discriminator(
    subscripts: &EinsumSubscripts,
    shapes: &[Vec<usize>],
    plan_hash: u64,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    subscripts.hash(&mut hasher);
    shapes.hash(&mut hasher);
    plan_hash.hash(&mut hasher);
    hasher.finish()
}

fn plan_spec_hash(plan_spec: &EinsumPlanSpec) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_einsum_plan_spec(plan_spec, &mut hasher);
    hasher.finish()
}

fn plan_spec_retained_bytes(plan_spec: &EinsumPlanSpec) -> usize {
    match plan_spec {
        EinsumPlanSpec::Auto(options) => saturating_sum([
            std::mem::size_of::<EinsumPlanSpec>(),
            vec_retained_bytes(&options.betas),
        ]),
        EinsumPlanSpec::LeftToRight => std::mem::size_of::<EinsumPlanSpec>(),
        EinsumPlanSpec::Path(path) | EinsumPlanSpec::FixedPairs(path) => saturating_sum([
            std::mem::size_of::<EinsumPlanSpec>(),
            vec_retained_bytes(path),
        ]),
    }
}

fn promote_dtypes(dtypes: impl IntoIterator<Item = DType>) -> DType {
    dtypes
        .into_iter()
        .reduce(tenferro_tensor::validate::promote_dtype)
        .unwrap_or(DType::F64)
}

#[cfg(test)]
mod tests;
