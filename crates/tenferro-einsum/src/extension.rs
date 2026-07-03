use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
#[cfg(feature = "autodiff")]
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use computegraph::compile::compile;
use computegraph::graph::GraphBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValueId, OperationRole};
use computegraph::types::{ValueKey, ValueRef};
use smallvec::SmallVec;
use tenferro_extension_macros::define_extension_runtime;
#[cfg(feature = "autodiff")]
use tenferro_ops::ad::context::ShapeGuardContext;
#[cfg(feature = "autodiff")]
use tenferro_ops::ad::PrimitiveRuleBuilder;
#[cfg(feature = "autodiff")]
use tenferro_ops::dim_expr::DimExpr;
#[cfg(feature = "autodiff")]
use tenferro_ops::ext_op::ExtensionAdRule;
use tenferro_ops::ext_op::{ExtensionLoweringError, ExtensionLoweringResult, ExtensionOp};
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
#[cfg(feature = "autodiff")]
use tenferro_ops::{ExtensionRegistryError, ExtensionRuleSet};
use tenferro_runtime::extension::{
    ExecInstruction, ExecOp, ExecProgram, ExtensionCacheKey, ExtensionExecutionContext,
};
use tenferro_tensor::{
    DType, Error as TensorError, RuntimeCacheControl, Tensor, TensorBackend, TensorRead,
};
#[cfg(feature = "autodiff")]
use tidu::{ADRuleError, ADRuleKind, ADRuleResult};

use crate::builder::build_einsum_graph;
use crate::cache::{
    einsum_subscripts_retained_bytes, saturating_sum, vec_of_vec_retained_bytes,
    vec_retained_bytes, EINSUM_EXTENSION_FAMILY_ID, EINSUM_RUNTIME_EXEC_PROGRAMS_CACHE,
    EINSUM_RUNTIME_PLANS_CACHE,
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

type InputIndexVec = SmallVec<[usize; 8]>;

/// Standard einsum extension payload.
///
/// This mirrors the current `tenferro.einsum.v1` payload shape. Runtime-owned
/// execution goes through [`EinsumRuntime`]; [`ExtensionOp::eager_execute`]
/// remains only as a host reference implementation for direct context-free
/// extension calls.
#[derive(Clone)]
pub(crate) struct EinsumExtensionOp {
    subscripts: EinsumSubscripts,
    plan_spec: EinsumPlanSpec,
    /// Optional execution hint. This is intentionally excluded from
    /// `ExtensionOp` identity: the shape-independent `plan_spec` carries
    /// user planning policy, while this tree is a resolved cacheable hint.
    static_tree: Option<Arc<ContractionTree>>,
    output_shape_hint: Option<Vec<SymDim>>,
}

impl std::fmt::Debug for EinsumExtensionOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EinsumExtensionOp")
            .field("subscripts", &self.subscripts)
            .field("plan_spec", &self.plan_spec)
            .field("has_static_tree", &self.static_tree.is_some())
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
            static_tree: None,
            output_shape_hint: None,
        }
    }

    /// Create an einsum extension payload with a precomputed plan.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn with_static_tree(
        subscripts: EinsumSubscripts,
        tree: Arc<ContractionTree>,
    ) -> Self {
        Self::new(subscripts).with_static_tree_hint(tree)
    }

    /// Create an einsum extension payload with an explicit output shape hint.
    #[must_use]
    pub(crate) fn with_output_shape_hint(
        subscripts: EinsumSubscripts,
        output_shape_hint: Vec<SymDim>,
        plan_spec: EinsumPlanSpec,
    ) -> Self {
        let mut op = Self::with_plan_spec(subscripts, plan_spec);
        op.output_shape_hint = Some(output_shape_hint);
        op
    }

    /// Attach a precomputed contraction tree as an execution hint.
    #[must_use]
    #[cfg(any(test, feature = "autodiff"))]
    pub(crate) fn with_static_tree_hint(mut self, tree: Arc<ContractionTree>) -> Self {
        self.static_tree = Some(tree);
        self
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

    /// Return the precomputed contraction tree, if present.
    #[must_use]
    pub(crate) fn static_tree(&self) -> Option<&Arc<ContractionTree>> {
        self.static_tree.as_ref()
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

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        if input_shapes.len() != self.subscripts.inputs.len()
            || input_dtypes.len() != input_shapes.len()
        {
            return Err(TensorError::InvalidConfig {
                op: "einsum",
                message: format!(
                    "expected {} input metadata entries, got dtypes={} shapes={}",
                    self.subscripts.inputs.len(),
                    input_dtypes.len(),
                    input_shapes.len()
                ),
            });
        }

        let mut label_dims: HashMap<u32, SymDim> = HashMap::new();
        for (labels, shape) in self.subscripts.inputs.iter().zip(input_shapes.iter()) {
            if labels.len() != shape.len() {
                return Err(TensorError::InvalidConfig {
                    op: "einsum",
                    message: format!(
                        "subscript rank {} does not match input rank {}",
                        labels.len(),
                        shape.len()
                    ),
                });
            }
            for (&label, dim) in labels.iter().zip(shape.iter()) {
                if let Some(existing) = label_dims.get(&label) {
                    if let (Some(lhs), Some(rhs)) =
                        (existing.constant_value(), dim.constant_value())
                    {
                        if lhs != rhs {
                            return Err(TensorError::ShapeMismatch {
                                op: "einsum",
                                lhs: vec![lhs],
                                rhs: vec![rhs],
                            });
                        }
                    }
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
                .ok_or_else(|| TensorError::InvalidConfig {
                    op: "einsum",
                    message: "output labels must be present in input metadata".into(),
                })?,
        };
        if output_shape.len() != self.subscripts.output.len() {
            return Err(TensorError::InvalidConfig {
                op: "einsum",
                message: format!(
                    "output rank {} does not match subscript rank {}",
                    output_shape.len(),
                    self.subscripts.output.len()
                ),
            });
        }
        Ok(vec![(
            promote_dtypes(input_dtypes.iter().copied()),
            output_shape,
        )])
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        let mut backend = tenferro_cpu::CpuBackend::new();
        let subscripts = Subscripts::from(&self.subscripts);
        crate::eager::eager_einsum_subscripts(&mut backend, inputs, &subscripts)
            .map(|output| vec![output])
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
            return Ok(None);
        };
        let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
        let subs = Subscripts::from(&self.subscripts);
        let tree = resolve_plan_spec(self.plan_spec(), &subs, &shape_refs)
            .map_err(|err| ExtensionLoweringError::new(err.to_string()))?;
        let output = build_einsum_graph(builder, &tree, inputs, &shapes)
            .map_err(|err| ExtensionLoweringError::new(err.to_string()))?;
        Ok(Some(vec![output]))
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

/// Return the explicit einsum extension AD rule set.
#[cfg(feature = "autodiff")]
pub fn ad_rules() -> Result<ExtensionRuleSet, ExtensionRegistryError> {
    ExtensionRuleSet::new().with_rule(Arc::new(EinsumAdRule))
}

#[derive(Debug)]
#[cfg(feature = "autodiff")]
struct EinsumAdRule;

#[cfg(feature = "autodiff")]
impl ExtensionAdRule for EinsumAdRule {
    fn family_id(&self) -> &'static str {
        EINSUM_EXTENSION_FAMILY_ID
    }

    fn linearize(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        primal_in: &[ValueKey<StdTensorOp>],
        _primal_out: &[ValueKey<StdTensorOp>],
        tangent_in: &[Option<LocalValueId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Jvp)?;
        let mut terms = Vec::new();

        for (active_idx, tangent) in tangent_in.iter().enumerate() {
            let Some(dt) = tangent else {
                continue;
            };

            let mut inputs = Vec::with_capacity(primal_in.len());
            for (input_idx, key) in primal_in.iter().enumerate() {
                if input_idx == active_idx {
                    inputs.push(ValueRef::Local(*dt));
                } else {
                    inputs.push(ValueRef::External(key.clone()));
                }
            }

            let out = builder.add_operation(
                StdTensorOp::Extension(Arc::new(op.clone())),
                inputs,
                OperationRole::Linearized {
                    active_mask: (0..primal_in.len()).map(|idx| idx == active_idx).collect(),
                },
            );
            terms.push(out[0]);
        }

        Ok(vec![sum_terms(builder, terms)])
    }

    fn transpose_rule(
        &self,
        op: &dyn ExtensionOp,
        builder: &mut dyn PrimitiveRuleBuilder,
        cotangent_out: &[Option<LocalValueId>],
        inputs: &[ValueRef<StdTensorOp>],
        mode: &OperationRole,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValueId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Transpose)?;
        let input_labels = &op.subscripts.inputs;
        let output_labels = &op.subscripts.output;
        let input_count = input_labels.len();

        let Some(ct) = cotangent_out.first().copied().flatten() else {
            return Ok(vec![None; input_count]);
        };
        let active_mask = match mode {
            OperationRole::Linearized { active_mask } => active_mask,
            OperationRole::Primary => return Ok(vec![None; input_count]),
        };
        let primal_input_shapes: Vec<Vec<SymDim>> = inputs
            .iter()
            .map(|input| ctx.shape_of(input).map(|shape| shape.to_vec()))
            .collect::<Result<_, _>>()?;
        let cotangent_shape = op.output_shape_hint.clone().ok_or_else(|| {
            ADRuleError::unsupported(
                "einsum VJP requires an output shape hint for cotangent planning",
                ADRuleKind::Transpose,
            )
        })?;

        let mut result = Vec::with_capacity(input_count);
        for active_idx in 0..input_count {
            if !active_mask.get(active_idx).copied().unwrap_or(false) {
                result.push(None);
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
            let mut vjp_input_labels = Vec::with_capacity(input_count);
            let mut vjp_inputs = Vec::with_capacity(input_count);
            let mut vjp_input_shapes = Vec::with_capacity(input_count);
            vjp_input_labels.push(output_labels.clone());
            vjp_inputs.push(ValueRef::Local(ct));
            vjp_input_shapes.push(cotangent_shape.clone());

            for input_idx in 0..input_count {
                if input_idx == active_idx {
                    continue;
                }
                vjp_input_labels.push(input_labels[input_idx].clone());
                vjp_input_shapes.push(primal_input_shapes[input_idx].clone());
                vjp_inputs.push(conjugate_primal_if_complex(
                    builder,
                    inputs[input_idx].clone(),
                    ctx,
                )?);
            }

            let output_shape_hint = primal_input_shapes[active_idx].clone();
            let vjp_op = vjp_einsum_op_with_inherited_plan(
                op,
                active_idx,
                EinsumSubscripts {
                    inputs: vjp_input_labels,
                    output: vjp_output_labels.clone(),
                },
                output_shape_hint.clone(),
                &vjp_input_shapes,
            )?;
            let out = builder.add_operation(
                StdTensorOp::Extension(Arc::new(vjp_op)),
                vjp_inputs,
                OperationRole::Linearized {
                    active_mask: std::iter::once(true)
                        .chain(std::iter::repeat_n(false, input_count.saturating_sub(1)))
                        .collect(),
                },
            );
            let mut cotangent = out[0];
            if vjp_output_labels != input_labels[active_idx] {
                let remapped = broadcast_einsum_vjp_to_input_shape(
                    builder,
                    cotangent,
                    &vjp_output_labels,
                    &input_labels[active_idx],
                    inputs[active_idx].clone(),
                    &output_shape_hint,
                )?;
                cotangent = remapped;
            }
            result.push(Some(cotangent));
        }

        Ok(result)
    }
}

#[cfg(feature = "autodiff")]
fn vjp_einsum_op_with_inherited_plan(
    primal_op: &EinsumExtensionOp,
    active_idx: usize,
    subscripts: EinsumSubscripts,
    output_shape_hint: Vec<SymDim>,
    input_shapes: &[Vec<SymDim>],
) -> ADRuleResult<EinsumExtensionOp> {
    let plan_spec =
        vjp_plan_spec_for_active(primal_op.plan_spec(), primal_op.input_count(), active_idx)?;
    let mut op = EinsumExtensionOp::with_output_shape_hint(
        subscripts.clone(),
        output_shape_hint,
        plan_spec.clone(),
    );
    if let Some(concrete_shapes) = concrete_sym_shapes(input_shapes) {
        let shape_refs: Vec<&[usize]> = concrete_shapes.iter().map(Vec::as_slice).collect();
        let raw_subscripts = Subscripts::from(&subscripts);
        let tree =
            resolve_plan_spec(&plan_spec, &raw_subscripts, &shape_refs).map_err(|err| {
                ADRuleError::unsupported(
                    format!(
                        "failed to resolve inherited einsum VJP plan for active input {active_idx}: {err}"
                    ),
                    ADRuleKind::Transpose,
                )
            })?;
        op = op.with_static_tree_hint(Arc::new(tree));
    }
    Ok(op)
}

#[cfg(feature = "autodiff")]
fn vjp_plan_spec_for_active(
    primal_plan: &EinsumPlanSpec,
    input_count: usize,
    active_idx: usize,
) -> ADRuleResult<EinsumPlanSpec> {
    if active_idx >= input_count {
        return Err(ADRuleError::unsupported(
            format!("einsum VJP active input {active_idx} is outside {input_count} inputs"),
            ADRuleKind::Transpose,
        ));
    }

    match primal_plan {
        EinsumPlanSpec::Auto(options) => Ok(EinsumPlanSpec::Auto(options.clone())),
        EinsumPlanSpec::LeftToRight => Ok(EinsumPlanSpec::LeftToRight),
        EinsumPlanSpec::Path(path) => {
            let pairs = jax_path_to_v1_pairs(path, input_count).map_err(|err| {
                ADRuleError::unsupported(
                    format!(
                        "failed to inherit einsum Path plan for VJP active input {active_idx}: {err}"
                    ),
                    ADRuleKind::Transpose,
                )
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
) -> ADRuleResult<Vec<(usize, usize)>> {
    if input_count == 0 {
        return Err(ADRuleError::unsupported(
            "einsum VJP cannot derive a plan for zero primal inputs",
            ADRuleKind::Transpose,
        ));
    }
    if active_idx >= input_count {
        return Err(ADRuleError::unsupported(
            format!("einsum VJP active input {active_idx} is outside {input_count} inputs"),
            ADRuleKind::Transpose,
        ));
    }
    let required_steps = input_count.saturating_sub(1);
    if primal_pairs.len() != required_steps {
        return Err(ADRuleError::unsupported(
            format!(
                "einsum VJP cannot inherit explicit plan for active input {active_idx}: \
                 expected {required_steps} primal steps for {input_count} inputs, got {}",
                primal_pairs.len()
            ),
            ADRuleKind::Transpose,
        ));
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
        return Err(ADRuleError::unsupported(
            format!(
                "einsum VJP plan derivation for active input {active_idx} produced an invalid \
                 tree: final id {final_id}, expected {expected_final}, steps {}",
                pairs.len()
            ),
            ADRuleKind::Transpose,
        ));
    }
    Ok(pairs)
}

#[cfg(feature = "autodiff")]
fn fixed_pair_children(
    pairs: &[(usize, usize)],
    input_count: usize,
    active_idx: usize,
) -> ADRuleResult<Vec<Option<(usize, usize)>>> {
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
) -> ADRuleResult<usize> {
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
) -> ADRuleResult<usize> {
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
) -> ADRuleResult<bool> {
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
fn invalid_vjp_plan_error(active_idx: usize, reason: String) -> ADRuleError {
    ADRuleError::unsupported(
        format!("einsum VJP cannot inherit explicit plan for active input {active_idx}: {reason}"),
        ADRuleKind::Transpose,
    )
}

#[cfg(feature = "autodiff")]
fn concrete_sym_shapes(shapes: &[Vec<SymDim>]) -> Option<Vec<Vec<usize>>> {
    shapes
        .iter()
        .map(|shape| shape.iter().map(SymDim::constant_value).collect())
        .collect()
}

#[cfg(feature = "autodiff")]
fn broadcast_einsum_vjp_to_input_shape(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent: LocalValueId,
    cotangent_labels: &[u32],
    input_labels: &[u32],
    shape_source: ValueRef<StdTensorOp>,
    input_shape: &[SymDim],
) -> ADRuleResult<LocalValueId> {
    let shape: Vec<DimExpr> = input_shape
        .iter()
        .enumerate()
        .map(|(axis, _)| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    let dims = map_label_occurrences(cotangent_labels, input_labels).ok_or_else(|| {
        ADRuleError::unsupported(
            format!(
                "einsum VJP broadcast remap failed for cotangent labels {cotangent_labels:?} \
                 into active input labels {input_labels:?}"
            ),
            ADRuleKind::Transpose,
        )
    })?;
    let mut inputs = vec![ValueRef::Local(cotangent)];
    let mut active_mask = vec![true];
    if !shape.is_empty() {
        inputs.push(shape_source);
        active_mask.push(false);
    }
    let broadcast = builder.add_operation(
        StdTensorOp::BroadcastInDim { shape, dims },
        inputs,
        OperationRole::Linearized { active_mask },
    )[0];
    Ok(project_repeated_labels_to_diagonal(
        builder,
        broadcast,
        input_labels,
    ))
}

#[cfg(feature = "autodiff")]
fn project_repeated_labels_to_diagonal(
    builder: &mut dyn PrimitiveRuleBuilder,
    cotangent: LocalValueId,
    labels: &[u32],
) -> LocalValueId {
    let mut result = cotangent;
    let mut first_axis_by_label = HashMap::new();
    for (axis_b, label) in labels.iter().copied().enumerate() {
        let Some(&axis_a) = first_axis_by_label.get(&label) else {
            first_axis_by_label.insert(label, axis_b);
            continue;
        };
        let extracted = builder.add_operation(
            StdTensorOp::ExtractDiag { axis_a, axis_b },
            vec![ValueRef::Local(result)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0];
        result = builder.add_operation(
            StdTensorOp::EmbedDiag { axis_a, axis_b },
            vec![ValueRef::Local(extracted)],
            OperationRole::Linearized {
                active_mask: vec![true],
            },
        )[0];
    }
    result
}

define_extension_runtime! {
    runtime = EinsumRuntime,
    family_id = EINSUM_EXTENSION_FAMILY_ID,
    op_type = EinsumExtensionOp,
    execute = execute_einsum_extension,
    execute_reads = execute_einsum_extension_reads,
    register_fn = register_runtime,
}

fn execute_einsum_extension<B: TensorBackend + 'static>(
    op: &EinsumExtensionOp,
    inputs: &[&Tensor],
    ctx: &mut ExtensionExecutionContext<'_, B>,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    if inputs.is_empty() {
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "einsum_extension",
            message: "einsum requires at least one input tensor".into(),
        });
    }

    let shapes: Vec<Vec<usize>> = inputs
        .iter()
        .map(|tensor| tensor.shape().to_vec())
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::from(op.subscripts());
    let tree = if let Some(tree) = op.static_tree() {
        Arc::clone(tree)
    } else {
        cached_runtime_tree(ctx, op.subscripts(), op.plan_spec(), &shapes, || {
            resolve_plan_spec(op.plan_spec(), &subs, &shape_refs)
        })?
    };

    if is_binary_non_contracting(&subs) {
        let output = ctx
            .backend_mut()
            .with_backend_session(|exec| crate::eager::eager_einsum_exec(exec, inputs, &tree))?;
        return Ok(vec![output]);
    }

    let (backend, caches) = ctx.parts_mut();
    let compiler_options = tenferro_runtime::extension::CompilerOptions::default();
    let optimizer_fingerprint = compiler_options.optimizer.fingerprint();
    let plan_hash = plan_spec_hash(op.plan_spec());
    let key = runtime_exec_program_cache_key(op, inputs, &shapes, plan_hash, optimizer_fingerprint);
    let cache_matches = caches
        .get::<CachedRuntimeExecProgram<B::RuntimeCache>>(&key)
        .map_or(false, |cached| {
            let key_data = &cached.key_data;
            key_data.matches_runtime_exec_program(op, inputs, &shapes, optimizer_fingerprint)
        });
    if !cache_matches {
        let key_data =
            RuntimeExecProgramCacheKeyData::new(op, inputs, &shapes, optimizer_fingerprint);
        let cached = build_runtime_exec_program::<B>(
            tree.as_ref(),
            inputs,
            &shapes,
            compiler_options,
            key_data,
        )?;
        caches.put_with_retained_bytes(key, cached, |cached| {
            cached_runtime_exec_program_retained_bytes(cached)
        });
    }
    let cached = caches
        .get_mut::<CachedRuntimeExecProgram<B::RuntimeCache>>(&key)
        .ok_or_else(|| {
            tenferro_tensor::Error::backend_failure(
                "einsum_extension",
                "runtime exec program cache entry missing after insertion",
            )
        })?;
    let key_data = &cached.key_data;
    if !key_data.matches_runtime_exec_program(op, inputs, &shapes, optimizer_fingerprint) {
        return Err(tenferro_tensor::Error::backend_failure(
            "einsum_extension",
            "runtime exec program cache hash collision was not replaced",
        ));
    }
    let program_inputs = runtime_program_inputs(inputs, cached.input_indices.as_slice())?;
    let mut outputs = tenferro_runtime::extension::execute_lowered_program_with_backend_cache(
        backend,
        &cached.program,
        program_inputs,
        &mut cached.backend_cache,
    )
    .map_err(|err| tenferro_tensor::Error::backend_failure("einsum_extension", err.to_string()))?;
    if outputs.len() != 1 {
        return Err(tenferro_tensor::Error::backend_failure(
            "einsum_extension",
            format!("expected 1 output, got {}", outputs.len()),
        ));
    }
    Ok(vec![outputs.remove(0)])
}

fn execute_einsum_extension_reads<B: TensorBackend + 'static>(
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
        return Err(tenferro_tensor::Error::InvalidConfig {
            op: "einsum_extension",
            message: "einsum requires at least one input tensor".into(),
        });
    }

    let shapes: Vec<Vec<usize>> = inputs.iter().map(|input| input.shape().to_vec()).collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::from(op.subscripts());
    let tree = if let Some(tree) = op.static_tree() {
        Arc::clone(tree)
    } else {
        cached_runtime_tree(ctx, op.subscripts(), op.plan_spec(), &shapes, || {
            resolve_plan_spec(op.plan_spec(), &subs, &shape_refs)
        })?
    };
    let output = ctx
        .backend_mut()
        .with_backend_session(|exec| crate::eager::eager_einsum_exec_read(exec, inputs, &tree))?;
    Ok(vec![output])
}

fn is_binary_non_contracting(subs: &Subscripts) -> bool {
    if subs.inputs.len() != 2 {
        return false;
    }

    let lhs = &subs.inputs[0];
    let rhs = &subs.inputs[1];
    let output = &subs.output;
    !lhs.iter()
        .any(|label| rhs.contains(label) && !output.contains(label))
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

#[derive(Clone)]
struct RuntimeExecProgramCacheKeyData {
    subscripts: EinsumSubscripts,
    shapes: Vec<Vec<usize>>,
    input_dtypes: Vec<DType>,
    plan_spec: EinsumPlanSpec,
    optimizer_fingerprint: u64,
}

impl RuntimeExecProgramCacheKeyData {
    fn new(
        op: &EinsumExtensionOp,
        inputs: &[&Tensor],
        shapes: &[Vec<usize>],
        optimizer_fingerprint: u64,
    ) -> Self {
        Self {
            subscripts: op.subscripts().clone(),
            shapes: shapes.to_vec(),
            input_dtypes: inputs.iter().map(|tensor| tensor.dtype()).collect(),
            plan_spec: op.plan_spec().clone(),
            optimizer_fingerprint,
        }
    }

    fn matches_runtime_exec_program(
        &self,
        op: &EinsumExtensionOp,
        inputs: &[&Tensor],
        shapes: &[Vec<usize>],
        optimizer_fingerprint: u64,
    ) -> bool {
        self.subscripts == *op.subscripts()
            && self.shapes.as_slice() == shapes
            && self.optimizer_fingerprint == optimizer_fingerprint
            && plan_specs_equal(&self.plan_spec, op.plan_spec())
            && self.input_dtypes.len() == inputs.len()
            && self
                .input_dtypes
                .iter()
                .zip(inputs.iter())
                .all(|(&dtype, tensor)| dtype == tensor.dtype())
    }

    fn retained_bytes(&self) -> usize {
        saturating_sum([
            einsum_subscripts_retained_bytes(&self.subscripts),
            saturating_sum(self.shapes.iter().map(vec_retained_bytes)),
            vec_retained_bytes(&self.input_dtypes),
            plan_spec_retained_bytes(&self.plan_spec),
            std::mem::size_of_val(&self.optimizer_fingerprint),
        ])
    }
}

struct CachedRuntimeExecProgram<C> {
    key_data: RuntimeExecProgramCacheKeyData,
    program: ExecProgram,
    input_indices: InputIndexVec,
    backend_cache: C,
}

fn runtime_exec_program_cache_key(
    op: &EinsumExtensionOp,
    inputs: &[&Tensor],
    shapes: &[Vec<usize>],
    plan_hash: u64,
    optimizer_fingerprint: u64,
) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    op.subscripts().hash(&mut hasher);
    shapes.hash(&mut hasher);
    for input in inputs {
        input.dtype().hash(&mut hasher);
    }
    plan_hash.hash(&mut hasher);
    optimizer_fingerprint.hash(&mut hasher);
    ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_RUNTIME_EXEC_PROGRAMS_CACHE,
        hasher.finish(),
    )
}

fn build_runtime_exec_program<B: TensorBackend>(
    tree: &ContractionTree,
    inputs: &[&Tensor],
    shapes: &[Vec<usize>],
    compiler_options: tenferro_runtime::extension::CompilerOptions,
    key_data: RuntimeExecProgramCacheKeyData,
) -> tenferro_tensor::Result<CachedRuntimeExecProgram<B::RuntimeCache>> {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut input_vals = Vec::with_capacity(inputs.len());
    for input_idx in 0..inputs.len() {
        let local = builder.add_input(TensorInputKey::User {
            id: input_idx as u64,
        });
        input_vals.push(ValueRef::Local(local));
    }

    let result_ref = build_einsum_graph(&mut builder, tree, &input_vals, shapes)
        .map_err(einsum_runtime_error)?;
    let result_local = match result_ref {
        ValueRef::Local(local) => local,
        ValueRef::External(_) => {
            return Err(tenferro_tensor::Error::backend_failure(
                "einsum_extension",
                "einsum builder returned an external value at runtime",
            ))
        }
    };
    builder.set_outputs(vec![result_local]);
    let graph = Arc::new(builder.build());
    let output_key = graph.values()[result_local].key.clone();

    let view = resolve(vec![graph]);
    let graph = materialize_merge(&view, &[output_key]);
    let compiled = compile(&graph);

    let mut input_indices = InputIndexVec::new();
    let mut input_dtypes = Vec::with_capacity(graph.inputs.len());
    let mut input_shapes = Vec::with_capacity(graph.inputs.len());
    for key in &graph.inputs {
        match key {
            ValueKey::Input(TensorInputKey::User { id }) => {
                let input_idx = *id as usize;
                let tensor = inputs.get(input_idx).ok_or_else(|| {
                    tenferro_tensor::Error::backend_failure(
                        "einsum_extension",
                        format!("runtime input {input_idx} missing"),
                    )
                })?;
                input_indices.push(input_idx);
                input_dtypes.push(tensor.dtype());
                input_shapes.push(tenferro_ops::dim_expr::DimExpr::from_concrete(
                    tensor.shape(),
                ));
            }
            other => {
                return Err(tenferro_tensor::Error::backend_failure(
                    "einsum_extension",
                    format!("unexpected runtime input key: {other:?}"),
                ))
            }
        }
    }

    let program = tenferro_runtime::extension::compile_std_to_exec_with_options(
        &compiled,
        &input_dtypes,
        &input_shapes,
        compiler_options,
    )
    .map_err(|err| tenferro_tensor::Error::backend_failure("einsum_extension", err.to_string()))?;
    Ok(CachedRuntimeExecProgram {
        key_data,
        program,
        input_indices,
        backend_cache: B::RuntimeCache::default(),
    })
}

fn runtime_program_inputs(
    inputs: &[&Tensor],
    input_indices: &[usize],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let mut program_inputs = Vec::with_capacity(input_indices.len());
    for &input_idx in input_indices {
        let tensor = inputs.get(input_idx).ok_or_else(|| {
            tenferro_tensor::Error::backend_failure(
                "einsum_extension",
                format!("runtime input {input_idx} missing"),
            )
        })?;
        program_inputs.push((*tensor).clone());
    }
    Ok(program_inputs)
}

fn cached_runtime_exec_program_retained_bytes<C: RuntimeCacheControl>(
    cached: &CachedRuntimeExecProgram<C>,
) -> usize {
    saturating_sum([
        std::mem::size_of::<CachedRuntimeExecProgram<C>>(),
        cached.key_data.retained_bytes(),
        exec_program_retained_bytes(&cached.program),
        smallvec_retained_bytes(&cached.input_indices),
        cached.backend_cache.stats().retained_bytes,
    ])
}

fn smallvec_retained_bytes<A: smallvec::Array>(values: &SmallVec<A>) -> usize {
    if values.spilled() {
        values
            .capacity()
            .saturating_mul(std::mem::size_of::<A::Item>())
    } else {
        0
    }
}

fn exec_program_retained_bytes(program: &ExecProgram) -> usize {
    saturating_sum([
        std::mem::size_of::<ExecProgram>(),
        vec_retained_bytes(&program.instructions),
        saturating_sum(
            program
                .instructions
                .iter()
                .map(exec_instruction_retained_bytes),
        ),
        vec_retained_bytes(&program.input_slots),
        vec_retained_bytes(&program.output_slots),
    ])
}

fn exec_instruction_retained_bytes(inst: &ExecInstruction) -> usize {
    saturating_sum([
        std::mem::size_of::<ExecInstruction>(),
        exec_op_retained_bytes(&inst.op),
        vec_retained_bytes(&inst.input_slots),
        vec_retained_bytes(&inst.output_slots),
        vec_of_vec_retained_bytes(&inst.output_shapes),
        vec_of_vec_retained_bytes(&inst.output_extents),
        vec_retained_bytes(&inst.last_use),
    ])
}

fn exec_op_retained_bytes(op: &ExecOp) -> usize {
    match op {
        ExecOp::Constant { bytes, .. } => vec_retained_bytes(bytes),
        ExecOp::Extension(extension) => std::mem::size_of_val(extension),
        _ => 0,
    }
}

fn cached_runtime_tree<B: TensorBackend>(
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
    error.to_tensor_error("einsum_extension")
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

#[cfg(feature = "autodiff")]
fn downcast_ad_op(op: &dyn ExtensionOp, kind: ADRuleKind) -> ADRuleResult<&EinsumExtensionOp> {
    op.as_any()
        .downcast_ref::<EinsumExtensionOp>()
        .ok_or_else(|| ADRuleError::unsupported("tenferro.einsum.v1 payload type mismatch", kind))
}

#[cfg(feature = "autodiff")]
fn sum_terms(
    builder: &mut dyn PrimitiveRuleBuilder,
    terms: Vec<LocalValueId>,
) -> Option<LocalValueId> {
    match terms.as_slice() {
        [] => None,
        [only] => Some(*only),
        [head, tail @ ..] => {
            let mut result = *head;
            for &term in tail {
                let sum = builder.add_operation(
                    StdTensorOp::Add,
                    vec![ValueRef::Local(result), ValueRef::Local(term)],
                    OperationRole::Linearized {
                        active_mask: vec![true, true],
                    },
                );
                result = sum[0];
            }
            Some(result)
        }
    }
}

#[cfg(feature = "autodiff")]
fn conjugate_primal_if_complex(
    builder: &mut dyn PrimitiveRuleBuilder,
    input: ValueRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ADRuleResult<ValueRef<StdTensorOp>> {
    Ok(match ctx.dtype_of(&input)? {
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => input,
        DType::C32 | DType::C64 => ValueRef::Local(
            builder.add_operation(StdTensorOp::Conj, vec![input], OperationRole::Primary)[0],
        ),
    })
}

fn promote_dtypes(dtypes: impl IntoIterator<Item = DType>) -> DType {
    dtypes
        .into_iter()
        .reduce(tenferro_tensor::validate::promote_dtype)
        .unwrap_or(DType::F64)
}

#[cfg(test)]
mod tests;
