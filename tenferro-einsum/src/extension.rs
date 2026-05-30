use std::any::Any;
use std::collections::HashMap;
#[cfg(feature = "autodiff")]
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[cfg(feature = "autodiff")]
use chainrules_core::{ADRuleError, ADRuleKind, ADRuleResult};
use computegraph::compile::compile;
use computegraph::fragment::FragmentBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, ValRef};
#[cfg(feature = "autodiff")]
use computegraph::types::{LocalValId, OpMode};
#[cfg(feature = "autodiff")]
use computegraph::OpEmitter;
use tenferro_extension_macros::define_extension_runtime;
#[cfg(feature = "autodiff")]
use tenferro_extension_macros::define_idempotent_rule_registration;
#[cfg(feature = "autodiff")]
use tenferro_ops::ad::context::ShapeGuardContext;
#[cfg(feature = "autodiff")]
use tenferro_ops::dim_expr::DimExpr;
#[cfg(feature = "autodiff")]
use tenferro_ops::ext_op::ExtensionAdRule;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::sym_dim::SymDim;
use tenferro_runtime::extension::{ExtensionCacheKey, ExtensionExecutionContext};
use tenferro_tensor::{DType, Tensor, TensorBackend};

use crate::builder::build_einsum_fragment;
use crate::cache::{
    einsum_subscripts_retained_bytes, EINSUM_EXTENSION_FAMILY_ID, EINSUM_RUNTIME_PLANS_CACHE,
};
use crate::{
    ContractionTree, EinsumSubscripts, Error as EinsumError, Result as EinsumResult, Subscripts,
};

/// Standard einsum extension payload.
///
/// This mirrors the current `tenferro.einsum.v1` payload shape. Runtime-owned
/// execution goes through [`EinsumRuntime`]; [`ExtensionOp::eager_execute`] is
/// kept only as a context-free compatibility fallback.
#[derive(Clone)]
pub(crate) struct EinsumExtensionOp {
    subscripts: EinsumSubscripts,
    /// Optional execution hint. This is intentionally excluded from
    /// `ExtensionOp` identity: contraction order affects performance, not the
    /// mathematical value of a fixed einsum.
    static_tree: Option<Arc<ContractionTree>>,
    output_shape_hint: Option<Vec<SymDim>>,
}

impl std::fmt::Debug for EinsumExtensionOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EinsumExtensionOp")
            .field("subscripts", &self.subscripts)
            .field("has_static_tree", &self.static_tree.is_some())
            .field("output_shape_hint", &self.output_shape_hint)
            .finish()
    }
}

impl EinsumExtensionOp {
    /// Create an einsum extension payload without a precomputed plan.
    #[must_use]
    #[cfg(any(feature = "autodiff", test))]
    pub(crate) fn new(subscripts: EinsumSubscripts) -> Self {
        Self {
            subscripts,
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
        Self {
            subscripts,
            static_tree: Some(tree),
            output_shape_hint: None,
        }
    }

    /// Create an einsum extension payload with an explicit output shape hint.
    #[must_use]
    pub(crate) fn with_output_shape_hint(
        subscripts: EinsumSubscripts,
        output_shape_hint: Vec<SymDim>,
    ) -> Self {
        Self {
            subscripts,
            static_tree: None,
            output_shape_hint: Some(output_shape_hint),
        }
    }

    /// Attach a precomputed contraction tree as an execution hint.
    #[must_use]
    pub(crate) fn with_static_tree_hint(mut self, tree: Arc<ContractionTree>) -> Self {
        self.static_tree = Some(tree);
        self
    }

    /// Return the canonical subscripts.
    #[must_use]
    pub(crate) fn subscripts(&self) -> &EinsumSubscripts {
        &self.subscripts
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
            self.subscripts == that.subscripts && self.output_shape_hint == that.output_shape_hint
        })
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn n_inputs(&self) -> usize {
        self.subscripts.inputs.len()
    }

    fn n_outputs(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        input_dtypes: &[DType],
        input_shapes: &[&[SymDim]],
    ) -> Vec<(DType, Vec<SymDim>)> {
        assert_eq!(
            input_shapes.len(),
            self.subscripts.inputs.len(),
            "einsum extension subscripts expect {} inputs, got {}",
            self.subscripts.inputs.len(),
            input_shapes.len()
        );
        assert_eq!(
            input_dtypes.len(),
            input_shapes.len(),
            "einsum extension expects dtype and shape arity to match"
        );

        let mut label_dims: HashMap<u32, SymDim> = HashMap::new();
        for (labels, shape) in self.subscripts.inputs.iter().zip(input_shapes.iter()) {
            assert_eq!(
                labels.len(),
                shape.len(),
                "einsum extension input rank mismatch: labels={}, shape={}",
                labels.len(),
                shape.len()
            );
            for (&label, dim) in labels.iter().zip(shape.iter()) {
                if let Some(existing) = label_dims.get(&label) {
                    if let (Some(lhs), Some(rhs)) =
                        (existing.constant_value(), dim.constant_value())
                    {
                        assert_eq!(
                            lhs, rhs,
                            "einsum extension label {label} has inconsistent concrete sizes {lhs} vs {rhs}"
                        );
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
                .map(|label| {
                    label_dims.get(label).cloned().unwrap_or_else(|| {
                        panic!("unknown size for label {label} in einsum extension output")
                    })
                })
                .collect(),
        };
        vec![(promote_dtypes(input_dtypes.iter().copied()), output_shape)]
    }

    fn eager_execute(&self, inputs: &[&Tensor]) -> tenferro_tensor::Result<Vec<Tensor>> {
        let mut backend = tenferro_cpu::CpuBackend::new();
        let subscripts = Subscripts::from(&self.subscripts);
        crate::eager::eager_einsum_subscripts(&mut backend, inputs, &subscripts)
            .map(|output| vec![output])
    }
}

#[cfg(feature = "autodiff")]
define_idempotent_rule_registration! {
    register_fn = ensure_einsum_extension_rule_registered,
    rule_type = EinsumAdRule,
    visibility = pub(crate),
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
        builder: &mut FragmentBuilder<StdTensorOp>,
        primal_in: &[GlobalValKey<StdTensorOp>],
        _primal_out: &[GlobalValKey<StdTensorOp>],
        tangent_in: &[Option<LocalValId>],
        _ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Linearize)?;
        let mut terms = Vec::new();

        for (active_idx, tangent) in tangent_in.iter().enumerate() {
            let Some(dt) = tangent else {
                continue;
            };

            let mut inputs = Vec::with_capacity(primal_in.len());
            for (input_idx, key) in primal_in.iter().enumerate() {
                if input_idx == active_idx {
                    inputs.push(ValRef::Local(*dt));
                } else {
                    inputs.push(ValRef::External(key.clone()));
                }
            }

            let out = builder.add_op(
                StdTensorOp::Extension(Arc::new(op.clone())),
                inputs,
                OpMode::Linear {
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
        emitter: &mut dyn OpEmitter<StdTensorOp>,
        cotangent_out: &[Option<LocalValId>],
        inputs: &[ValRef<StdTensorOp>],
        mode: &OpMode,
        ctx: &mut ShapeGuardContext,
    ) -> ADRuleResult<Vec<Option<LocalValId>>> {
        let op = downcast_ad_op(op, ADRuleKind::Transpose)?;
        let input_labels = &op.subscripts.inputs;
        let output_labels = &op.subscripts.output;
        let n_inputs = input_labels.len();

        let Some(ct) = cotangent_out.first().copied().flatten() else {
            return Ok(vec![None; n_inputs]);
        };
        let active_mask = match mode {
            OpMode::Linear { active_mask } => active_mask,
            OpMode::Primal => return Ok(vec![None; n_inputs]),
        };

        let mut result = Vec::with_capacity(n_inputs);
        for active_idx in 0..n_inputs {
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
            let mut vjp_input_labels = Vec::with_capacity(n_inputs);
            let mut vjp_inputs = Vec::with_capacity(n_inputs);
            vjp_input_labels.push(output_labels.clone());
            vjp_inputs.push(ValRef::Local(ct));

            for input_idx in 0..n_inputs {
                if input_idx == active_idx {
                    continue;
                }
                vjp_input_labels.push(input_labels[input_idx].clone());
                vjp_inputs.push(conjugate_primal_if_complex(
                    emitter,
                    inputs[input_idx].clone(),
                    ctx,
                ));
            }

            let output_shape_hint = ctx.shape_of(&inputs[active_idx]).to_vec();
            let vjp_op = EinsumExtensionOp::with_output_shape_hint(
                EinsumSubscripts {
                    inputs: vjp_input_labels,
                    output: vjp_output_labels.clone(),
                },
                output_shape_hint.clone(),
            );
            let out = emitter.add_op(
                StdTensorOp::Extension(Arc::new(vjp_op)),
                vjp_inputs,
                OpMode::Linear {
                    active_mask: std::iter::once(true)
                        .chain(std::iter::repeat(false).take(n_inputs.saturating_sub(1)))
                        .collect(),
                },
            );
            let mut cotangent = out[0];
            if vjp_output_labels != input_labels[active_idx] {
                cotangent = broadcast_einsum_vjp_to_input_shape(
                    emitter,
                    cotangent,
                    &vjp_output_labels,
                    &input_labels[active_idx],
                    inputs[active_idx].clone(),
                    &output_shape_hint,
                );
            }
            result.push(Some(cotangent));
        }

        Ok(result)
    }
}

#[cfg(feature = "autodiff")]
fn broadcast_einsum_vjp_to_input_shape(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent: LocalValId,
    cotangent_labels: &[u32],
    input_labels: &[u32],
    shape_source: ValRef<StdTensorOp>,
    input_shape: &[SymDim],
) -> LocalValId {
    let shape: Vec<DimExpr> = input_shape
        .iter()
        .enumerate()
        .map(|(axis, _)| DimExpr::InputDim { input_idx: 1, axis })
        .collect();
    let dims = map_label_occurrences(cotangent_labels, input_labels);
    let mut inputs = vec![ValRef::Local(cotangent)];
    if !shape.is_empty() {
        inputs.push(shape_source);
    }
    let broadcast = emitter.add_op(
        StdTensorOp::BroadcastInDim { shape, dims },
        inputs,
        OpMode::Linear {
            active_mask: vec![true, false],
        },
    )[0];
    project_repeated_labels_to_diagonal(emitter, broadcast, input_labels)
}

#[cfg(feature = "autodiff")]
fn map_label_occurrences(source_labels: &[u32], target_labels: &[u32]) -> Vec<usize> {
    let mut used = vec![false; target_labels.len()];
    source_labels
        .iter()
        .map(|label| {
            let axis = target_labels
                .iter()
                .enumerate()
                .find_map(|(axis, target)| (!used[axis] && target == label).then_some(axis))
                .unwrap_or_else(|| panic!("einsum VJP label {label} missing from input labels"));
            used[axis] = true;
            axis
        })
        .collect()
}

#[cfg(feature = "autodiff")]
fn project_repeated_labels_to_diagonal(
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    cotangent: LocalValId,
    labels: &[u32],
) -> LocalValId {
    let mut result = cotangent;
    let mut seen = HashSet::new();
    for (axis_a, label) in labels.iter().copied().enumerate() {
        if !seen.insert(label) {
            continue;
        }
        let Some(axis_b) = labels
            .iter()
            .enumerate()
            .skip(axis_a + 1)
            .find_map(|(axis, candidate)| (*candidate == label).then_some(axis))
        else {
            continue;
        };
        let extracted = emitter.add_op(
            StdTensorOp::ExtractDiag { axis_a, axis_b },
            vec![ValRef::Local(result)],
            OpMode::Linear {
                active_mask: vec![true],
            },
        )[0];
        result = emitter.add_op(
            StdTensorOp::EmbedDiag { axis_a, axis_b },
            vec![ValRef::Local(extracted)],
            OpMode::Linear {
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
        cached_runtime_tree(ctx, op.subscripts(), &shapes, || {
            ContractionTree::optimize(&subs, &shape_refs)
        })?
    };

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut input_vals = Vec::with_capacity(inputs.len());
    for input_idx in 0..inputs.len() {
        let local = builder.add_input(TensorInputKey::User {
            id: input_idx as u64,
        });
        input_vals.push(ValRef::Local(local));
    }

    let result_ref = build_einsum_fragment(&mut builder, tree.as_ref(), &input_vals, &shapes)
        .map_err(einsum_runtime_error)?;
    let result_local = match result_ref {
        ValRef::Local(local) => local,
        ValRef::External(_) => {
            return Err(tenferro_tensor::Error::backend_failure(
                "einsum_extension",
                "einsum builder returned an external value at runtime",
            ))
        }
    };
    builder.set_outputs(vec![result_local]);
    let fragment = Arc::new(builder.build());
    let output_key = fragment.vals()[result_local].key.clone();

    let view = resolve(vec![fragment]);
    let graph = materialize_merge(&view, &[output_key]);
    let compiled = compile(&graph);

    let mut program_inputs = Vec::with_capacity(graph.inputs.len());
    let mut input_dtypes = Vec::with_capacity(graph.inputs.len());
    let mut input_shapes = Vec::with_capacity(graph.inputs.len());
    for key in &graph.inputs {
        match key {
            GlobalValKey::Input(TensorInputKey::User { id }) => {
                let input_idx = *id as usize;
                let tensor = inputs.get(input_idx).ok_or_else(|| {
                    tenferro_tensor::Error::backend_failure(
                        "einsum_extension",
                        format!("runtime input {input_idx} missing"),
                    )
                })?;
                program_inputs.push((*tensor).clone());
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

    let program =
        tenferro_runtime::compiler::compile_std_to_exec(&compiled, &input_dtypes, &input_shapes);
    let mut outputs = tenferro_runtime::exec::eval_exec_ir_unsegmented(
        ctx.backend_mut(),
        &program,
        program_inputs,
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

fn cached_runtime_tree<B: TensorBackend>(
    ctx: &mut ExtensionExecutionContext<'_, B>,
    subscripts: &EinsumSubscripts,
    shapes: &[Vec<usize>],
    build: impl FnOnce() -> EinsumResult<ContractionTree>,
) -> tenferro_tensor::Result<Arc<ContractionTree>> {
    let key_data = (subscripts.clone(), shapes.to_vec());
    let key = ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_RUNTIME_PLANS_CACHE,
        hash_value(&key_data),
    );
    if let Some(cached) = ctx.caches_mut().get::<Arc<ContractionTree>>(&key) {
        return Ok(Arc::clone(cached));
    }

    let tree = Arc::new(build().map_err(einsum_runtime_error)?);
    let retained_bytes = einsum_subscripts_retained_bytes(subscripts)
        + shapes
            .iter()
            .map(|shape| shape.capacity() * std::mem::size_of::<usize>())
            .sum::<usize>()
        + tree.retained_bytes_for_cache_stats();
    ctx.caches_mut().put(key, Arc::clone(&tree), retained_bytes);
    Ok(tree)
}

fn einsum_runtime_error(error: EinsumError) -> tenferro_tensor::Error {
    error.to_tensor_error("einsum_extension")
}

fn hash_value<T: Hash>(value: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(feature = "autodiff")]
fn downcast_ad_op<'a>(
    op: &'a dyn ExtensionOp,
    kind: ADRuleKind,
) -> ADRuleResult<&'a EinsumExtensionOp> {
    op.as_any()
        .downcast_ref::<EinsumExtensionOp>()
        .ok_or_else(|| ADRuleError::unsupported("tenferro.einsum.v1 payload type mismatch", kind))
}

#[cfg(feature = "autodiff")]
fn sum_terms(
    builder: &mut FragmentBuilder<StdTensorOp>,
    terms: Vec<LocalValId>,
) -> Option<LocalValId> {
    match terms.as_slice() {
        [] => None,
        [only] => Some(*only),
        [head, tail @ ..] => {
            let mut result = *head;
            for &term in tail {
                let sum = builder.add_op(
                    StdTensorOp::Add,
                    vec![ValRef::Local(result), ValRef::Local(term)],
                    OpMode::Linear {
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
    emitter: &mut dyn OpEmitter<StdTensorOp>,
    input: ValRef<StdTensorOp>,
    ctx: &mut ShapeGuardContext,
) -> ValRef<StdTensorOp> {
    match ctx.dtype_of(&input) {
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::Bool => input,
        DType::C32 | DType::C64 => {
            ValRef::Local(emitter.add_op(StdTensorOp::Conj, vec![input], OpMode::Primal)[0])
        }
    }
}

fn promote_dtypes(dtypes: impl IntoIterator<Item = DType>) -> DType {
    dtypes
        .into_iter()
        .reduce(promote_dtype)
        .unwrap_or(DType::F64)
}

fn promote_dtype(lhs: DType, rhs: DType) -> DType {
    use DType::*;
    if lhs == rhs {
        return lhs;
    }
    let (a, b) = if promotion_rank(lhs) <= promotion_rank(rhs) {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    match (a, b) {
        (Bool, other) => other,
        (I32, I64) => I64,
        (I32 | I64, F32 | F64) => F64,
        (I32 | I64, C32 | C64) => C64,
        (F32, F64) => F64,
        (F32, C32) => C32,
        (F32, C64) => C64,
        (F64, C32) => C64,
        (F64, C64) => C64,
        (C32, C64) => C64,
        _ => unreachable!("promote_dtype: unhandled pair {:?} {:?}", lhs, rhs),
    }
}

fn promotion_rank(dt: DType) -> u8 {
    match dt {
        DType::Bool => 0,
        DType::I32 => 1,
        DType::I64 => 2,
        DType::F32 => 3,
        DType::F64 => 4,
        DType::C32 => 5,
        DType::C64 => 6,
    }
}

#[cfg(test)]
mod tests;
