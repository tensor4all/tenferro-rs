//! EagerTensor einsum extension API.

use std::collections::hash_map::DefaultHasher;
use std::error::Error as StdError;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::{Arc, OnceLock};

use computegraph::compile::{compile, CompiledProgram, Instruction};
use computegraph::graph::GraphBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{ValueKey, ValueRef};
use tenferro_ad::extension::{adopt_untracked_eager_value, apply_eager_with_extension_session};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{ErrorPhase, ExtensionCacheKey};
use tenferro_tensor::{ErrorKind, ShapeMismatch, Tensor, ValidationError, ValidationKind};

use crate::binary_dot::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};
use crate::builder::build_einsum_graph;
use crate::cache::{
    saturating_sum, vec_retained_bytes, EINSUM_EAGER_EXPANDED_PROGRAMS_CACHE,
    EINSUM_EXTENSION_FAMILY_ID,
};
use crate::extension::EinsumExtensionOp;
use crate::optimize::{
    default_auto_options, hash_einsum_plan_spec, plan_specs_equal, resolve_plan_spec,
    EinsumPlanSpec,
};
use crate::{parse_einsum_subscripts, EinsumSubscripts, Error, Result, Subscripts, TensorDotAxes};

/// Eager einsum extension methods for slices or arrays of [`EagerTensor`] refs.
pub trait EagerEinsumExt {
    /// Execute an einsum from string notation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] for rank/shape/dtype mismatches, or
    /// [`Error::Planning`] / [`Error::Runtime`] for contraction planning and
    /// execution failures.
    fn einsum(&self, subscripts: &str) -> Result<EagerTensor>;

    /// Execute an einsum from parsed integer labels.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for rank/shape/dtype mismatches,
    /// [`Error::Planning`] for an invalid contraction plan, or
    /// [`Error::Runtime`] for extension registration or backend execution
    /// failures.
    fn einsum_subscripts(&self, subscripts: &EinsumSubscripts) -> Result<EagerTensor>;
}

fn eager_cpu_extension_module(
) -> tenferro_runtime::Result<Arc<dyn tenferro_runtime::ExtensionModule>> {
    static MODULE: OnceLock<Arc<dyn tenferro_runtime::ExtensionModule>> = OnceLock::new();
    if let Some(module) = MODULE.get() {
        return Ok(Arc::clone(module));
    }

    let engine_id = tenferro_cpu::runtime_engine_id().map_err(|source| {
        tenferro_runtime::Error::runtime_state_source(
            "tenferro_einsum::eager_extension_module",
            ErrorPhase::Execution,
            source,
        )
    })?;
    let module = crate::extension::extension_module::<tenferro_cpu::CpuBackend>(engine_id)
        .map_err(|source| {
            tenferro_runtime::Error::runtime_state_source(
                "tenferro_einsum::eager_extension_module",
                ErrorPhase::Execution,
                source,
            )
        })?;
    let _ = MODULE.set(Arc::clone(&module));
    Ok(MODULE.get().cloned().unwrap_or(module))
}

impl EagerEinsumExt for [&EagerTensor] {
    fn einsum(&self, subscripts: &str) -> Result<EagerTensor> {
        einsum(self, subscripts)
    }

    fn einsum_subscripts(&self, subscripts: &EinsumSubscripts) -> Result<EagerTensor> {
        einsum_subscripts(self, subscripts)
    }
}

impl<const N: usize> EagerEinsumExt for [&EagerTensor; N] {
    fn einsum(&self, subscripts: &str) -> Result<EagerTensor> {
        einsum(self.as_slice(), subscripts)
    }

    fn einsum_subscripts(&self, subscripts: &EinsumSubscripts) -> Result<EagerTensor> {
        einsum_subscripts(self.as_slice(), subscripts)
    }
}

/// Eager tensor contraction-sugar methods.
pub trait EagerTensorEinsumExt {
    /// Contract two eager tensors over the requested axes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] with `AxisOutOfBounds`, `DuplicateAxis`,
    /// `RankMismatch`, or `ShapeMismatch` for invalid axes/shapes, or
    /// [`Error::Runtime`] for backend execution failures.
    fn tensordot(&self, rhs: &EagerTensor, axes: TensorDotAxes<'_>) -> Result<EagerTensor>;
}

impl EagerTensorEinsumExt for EagerTensor {
    fn tensordot(&self, rhs: &EagerTensor, axes: TensorDotAxes<'_>) -> Result<EagerTensor> {
        tensordot(self, rhs, axes)
    }
}

/// Execute an einsum eagerly on [`EagerTensor`] values.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::EagerEinsumExt;
/// use tenferro_tensor::Tensor;
///
/// let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
///     runtime.clone(),
/// ).unwrap();
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
///     runtime,
/// ).unwrap();
/// let out = [&a, &b].einsum("ij,jk->ik")?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::InvalidSubscripts`] for malformed notation,
/// [`Error::Validation`] for input count/rank/shape/dtype mismatches,
/// [`Error::Planning`] when no contraction path is valid, or [`Error::Runtime`]
/// for extension registration and backend execution failures.
pub fn einsum(inputs: &[&EagerTensor], subscripts: &str) -> Result<EagerTensor> {
    let subscripts = parse_einsum_subscripts(subscripts)?;
    einsum_subscripts(inputs, &subscripts)
}

/// Execute an einsum eagerly from integer labels.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::{EagerEinsumExt, parse_einsum_subscripts};
/// use tenferro_tensor::Tensor;
///
/// let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
///     runtime.clone(),
/// ).unwrap();
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
///     runtime,
/// ).unwrap();
/// let subscripts = parse_einsum_subscripts("ij,jk->ik").unwrap();
/// let out = [&a, &b].einsum_subscripts(&subscripts)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Validation`] for input count/rank/shape/dtype mismatches,
/// [`Error::Planning`] when no contraction path is valid, or [`Error::Runtime`]
/// for extension registration and backend execution failures.
pub fn einsum_subscripts(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<EagerTensor> {
    if let Some(result) = try_direct_binary_dot_general(inputs, subscripts) {
        return result;
    }

    if let Some(result) = try_whole_program_untracked(inputs, subscripts)? {
        return Ok(result);
    }

    let output_shape_hint = infer_eager_output_shape(subscripts, inputs)?;
    if let Some(result) = try_expand_eager_einsum(inputs, subscripts)? {
        return Ok(result);
    }

    let op = Arc::new(EinsumExtensionOp::with_output_shape_hint(
        subscripts.clone(),
        output_shape_hint,
        EinsumPlanSpec::Auto(default_auto_options()),
    ));
    let mut outputs =
        apply_eager_with_extension_session(op, inputs, eager_cpu_extension_module()?)?;
    outputs.pop().ok_or_else(|| {
        Error::Runtime(tenferro_runtime::Error::MissingInput(
            "einsum extension produced no eager output".into(),
        ))
    })
}

fn try_direct_binary_dot_general(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Option<Result<EagerTensor>> {
    if inputs.len() != 2 || subscripts.inputs.len() != 2 {
        return None;
    }

    let lhs_labels = &subscripts.inputs[0];
    let rhs_labels = &subscripts.inputs[1];
    if lhs_labels.len() != inputs[0].shape().len() || rhs_labels.len() != inputs[1].shape().len() {
        return None;
    }

    if let Some(plan) =
        try_build_exact_output_binary_dot_plan(lhs_labels, rhs_labels, &subscripts.output)
    {
        return Some(match plan.operand_order {
            BinaryDotOperandOrder::Original => inputs[0]
                .dot_general(inputs[1], plan.config)
                .map_err(Error::Runtime),
            BinaryDotOperandOrder::Swapped => inputs[1]
                .dot_general(inputs[0], plan.config)
                .map_err(Error::Runtime),
        });
    }
    None
}

/// Whether the untracked whole-program eager einsum executor is enabled.
///
/// Prototype gate (issue #1060 follow-up): when set, untracked N-ary eager
/// einsum runs the whole contraction in one backend session via
/// [`crate::eager::eager_einsum_subscripts_on_session`] instead of executing
/// the expanded program one standard op at a time. Tracked (`requires_grad`)
/// inputs keep the existing per-op path so eager AD recording semantics are
/// unchanged.
fn whole_program_untracked_enabled() -> bool {
    std::env::var_os("TENFERRO_EAGER_WHOLE_PROGRAM").is_some()
}

/// Run an untracked eager einsum as a single backend-session program.
///
/// Returns `None` (so the caller falls back to the per-op expanded path) when
/// the gate is off, there are no inputs, any input tracks gradients, or the
/// inputs do not all share one runtime.
fn try_whole_program_untracked(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<Option<EagerTensor>> {
    if !whole_program_untracked_enabled() {
        return Ok(None);
    }
    let Some(first) = inputs.first() else {
        return Ok(None);
    };
    if inputs.iter().any(|tensor| tensor.tracks_grad()) {
        return Ok(None);
    }
    let runtime = first.runtime();
    if inputs
        .iter()
        .any(|tensor| !Arc::ptr_eq(tensor.runtime(), runtime))
    {
        return Ok(None);
    }

    let subs = Subscripts::from(subscripts);
    let tensor_owners = inputs
        .iter()
        .map(|tensor| tensor.to_tensor().map_err(Error::Runtime))
        .collect::<Result<Vec<Tensor>>>()?;
    let tensors: Vec<_> = tensor_owners.iter().collect();
    let result = runtime.with_execution_session(|backend| {
        crate::eager::eager_einsum_subscripts_with_session(backend, &tensors, &subs)
    })??;
    Ok(Some(EagerTensor::from_tensor_in(result, runtime.clone())?))
}

/// Run an untracked whole-program eager einsum on an explicit contraction tree.
///
/// Prototype/benchmark entry (issue #1060 follow-up). Executes the whole
/// contraction in one backend session on the caller-provided path (e.g. an
/// externally optimized `opt_flops` order via [`crate::ContractionTree::from_pairs`]),
/// instead of one eager op per expanded step. All inputs must be untracked and
/// share one runtime; tracked inputs should use the per-op path to keep eager
/// AD semantics.
///
/// # Examples
///
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_einsum::{ContractionTree, Subscripts};
/// use tenferro_tensor::Tensor;
///
/// let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let a = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
///     runtime.clone(),
/// ).unwrap();
/// let b = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
///     runtime,
/// ).unwrap();
/// let subs = Subscripts::parse("ij,jk->ik").unwrap();
/// let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4]], &[(0, 1)]).unwrap();
/// let out = einsum_whole_program_untracked(&[&a, &b], &tree)?;
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_ad::error::Error>(())
/// ```
#[cfg(test)]
fn einsum_whole_program_untracked(
    inputs: &[&EagerTensor],
    tree: &crate::ContractionTree,
) -> Result<EagerTensor> {
    let first = inputs.first().ok_or_else(|| {
        Error::invalid_argument(
            "einsum",
            "inputs",
            "einsum requires at least one input tensor",
        )
    })?;
    if inputs.iter().any(|tensor| tensor.tracks_grad()) {
        return Err(Error::invalid_argument(
            "einsum",
            "inputs",
            "whole-program eager einsum requires untracked inputs",
        ));
    }
    let runtime = first.runtime();
    if inputs
        .iter()
        .any(|tensor| !Arc::ptr_eq(tensor.runtime(), runtime))
    {
        return Err(Error::invalid_argument(
            "einsum",
            "inputs",
            "whole-program eager einsum requires inputs from one runtime",
        ));
    }
    let tensor_owners = inputs
        .iter()
        .map(|tensor| tensor.to_tensor().map_err(Error::Runtime))
        .collect::<Result<Vec<Tensor>>>()?;
    let tensors: Vec<_> = tensor_owners.iter().collect();
    let result = runtime.with_execution_session(|backend| {
        crate::eager::eager_einsum_with_tree(backend, &tensors, tree)
    })??;
    EagerTensor::from_tensor_in(result, runtime.clone()).map_err(Error::Runtime)
}

fn try_expand_eager_einsum(
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
) -> Result<Option<EagerTensor>> {
    if inputs.len() <= 1 {
        return Ok(None);
    }

    let shapes: Vec<Vec<usize>> = inputs
        .iter()
        .map(|tensor| tensor.shape().to_vec())
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let subs = Subscripts::from(subscripts);
    let plan_spec = EinsumPlanSpec::Auto(default_auto_options());

    let program = cached_expanded_eager_program(
        inputs[0].runtime(),
        subscripts,
        &subs,
        &plan_spec,
        &shape_refs,
        &shapes,
    )?;
    execute_eager_einsum_program(inputs, &program)
}

struct ExpandedEagerProgram {
    compiled: CompiledProgram<StdTensorOp>,
    input_slots: Vec<(usize, usize)>,
}

#[derive(Clone)]
struct ExpandedEagerProgramCacheKeyData {
    subscripts: EinsumSubscripts,
    shapes: Vec<Vec<usize>>,
    plan_spec: EinsumPlanSpec,
}

impl ExpandedEagerProgramCacheKeyData {
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

    fn matches_expanded_eager_program(
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
            crate::cache::einsum_subscripts_retained_bytes(&self.subscripts),
            saturating_sum(self.shapes.iter().map(vec_retained_bytes)),
            plan_spec_retained_bytes(&self.plan_spec),
        ])
    }
}

struct CachedExpandedEagerProgram {
    key_data: ExpandedEagerProgramCacheKeyData,
    program: Arc<ExpandedEagerProgram>,
}

fn cached_expanded_eager_program(
    runtime: &Arc<EagerRuntime>,
    subscripts: &EinsumSubscripts,
    subs: &Subscripts,
    plan_spec: &EinsumPlanSpec,
    shape_refs: &[&[usize]],
    shapes: &[Vec<usize>],
) -> Result<Arc<ExpandedEagerProgram>> {
    runtime.with_extension_execution_context(|extension_ctx| {
        let caches = extension_ctx.caches_mut();
        let plan_hash = plan_spec_hash(plan_spec);
        let key = expanded_eager_program_cache_key(subscripts, shapes, plan_hash);
        if let Some(cached) = caches.get::<CachedExpandedEagerProgram>(&key) {
            let key_data = &cached.key_data;
            if key_data.matches_expanded_eager_program(subscripts, shapes, plan_spec) {
                return Ok(Arc::clone(&cached.program));
            }
        }

        let tree = resolve_plan_spec(plan_spec, subs, shape_refs)?;
        let program = Arc::new(build_expanded_eager_program(&tree, shapes)?);
        let key_data = ExpandedEagerProgramCacheKeyData::new(subscripts, shapes, plan_spec);
        let retained_bytes = saturating_sum([
            key_data.retained_bytes(),
            expanded_eager_program_retained_bytes(&program),
        ]);
        caches.put(
            key,
            CachedExpandedEagerProgram {
                key_data,
                program: Arc::clone(&program),
            },
            retained_bytes,
        );
        Ok(program)
    })?
}

fn expanded_eager_program_cache_key(
    subscripts: &EinsumSubscripts,
    shapes: &[Vec<usize>],
    plan_hash: u64,
) -> ExtensionCacheKey {
    let mut hasher = DefaultHasher::new();
    subscripts.hash(&mut hasher);
    shapes.hash(&mut hasher);
    plan_hash.hash(&mut hasher);
    ExtensionCacheKey::new(
        EINSUM_EXTENSION_FAMILY_ID,
        EINSUM_EAGER_EXPANDED_PROGRAMS_CACHE,
        hasher.finish(),
    )
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

fn build_expanded_eager_program(
    tree: &crate::ContractionTree,
    shapes: &[Vec<usize>],
) -> Result<ExpandedEagerProgram> {
    let mut builder = GraphBuilder::<StdTensorOp>::new();
    let mut input_vals = Vec::with_capacity(shapes.len());
    for input_idx in 0..shapes.len() {
        let local = builder.add_input(TensorInputKey::User {
            id: input_idx as u64,
        });
        input_vals.push(ValueRef::Local(local));
    }

    let result_ref = build_einsum_graph(&mut builder, tree, &input_vals, shapes)?;
    let ValueRef::Local(result_local) = result_ref else {
        return Err(Error::Runtime(tenferro_runtime::Error::Internal(
            "expanded eager einsum returned an external value".into(),
        )));
    };
    builder.set_outputs(vec![result_local]);
    let graph = Arc::new(builder.build());
    let output_key = graph.values()[result_local].key.clone();
    let view = resolve(vec![graph]);
    let graph = materialize_merge(&view, &[output_key]);
    let compiled = compile(&graph);
    let input_slots = compiled
        .input_slots
        .iter()
        .zip(graph.inputs.iter())
        .map(|(&slot, key)| {
            let ValueKey::Input(TensorInputKey::User { id }) = key else {
                return Err(runtime_internal(format!(
                    "expanded eager einsum saw unexpected input key: {key:?}"
                )));
            };
            Ok((slot, *id as usize))
        })
        .collect::<Result<_>>()?;

    Ok(ExpandedEagerProgram {
        compiled,
        input_slots,
    })
}

fn execute_eager_einsum_program(
    inputs: &[&EagerTensor],
    program: &ExpandedEagerProgram,
) -> Result<Option<EagerTensor>> {
    let mut slots: Vec<Option<EagerTensor>> = vec![None; program.compiled.n_slots];
    for &(slot, input_idx) in &program.input_slots {
        let tensor = inputs.get(input_idx).ok_or_else(|| {
            runtime_missing(format!(
                "expanded eager einsum input {input_idx} is missing"
            ))
        })?;
        slots[slot] = Some((*tensor).clone());
    }

    let mut instruction_idx = 0;
    while instruction_idx < program.compiled.instructions.len() {
        if let Some((output_slot, output)) = try_execute_eager_broadcast_multiply_pattern(
            &program.compiled.instructions,
            instruction_idx,
            &slots,
            &program.compiled.output_slots,
        )? {
            slots[output_slot] = Some(output);
            instruction_idx += 3;
            continue;
        }

        let instr = &program.compiled.instructions[instruction_idx];
        if instr.outputs.len() != 1 {
            return Err(runtime_internal(format!(
                "expanded eager einsum expected single-output op, got {} outputs",
                instr.outputs.len()
            )));
        }
        let input_refs: Vec<&EagerTensor> = instr
            .inputs
            .iter()
            .map(|&slot| slot_tensor(&slots, slot))
            .collect::<Result<_>>()?;
        let output =
            tenferro_ad::extension::apply_standard_op(instr.operation.clone(), &input_refs)?;
        slots[instr.outputs[0]] = Some(output);
        instruction_idx += 1;
    }

    let [output_slot] = program.compiled.output_slots.as_slice() else {
        return Err(runtime_internal(format!(
            "expanded eager einsum expected one graph output, got {}",
            program.compiled.output_slots.len()
        )));
    };
    slots
        .get_mut(*output_slot)
        .and_then(Option::take)
        .map(Some)
        .ok_or_else(|| runtime_missing("expanded eager einsum output slot is missing"))
}

fn expanded_eager_program_retained_bytes(program: &ExpandedEagerProgram) -> usize {
    saturating_sum([
        size_of::<ExpandedEagerProgram>(),
        vec_retained_bytes(&program.input_slots),
        compiled_program_retained_bytes(&program.compiled),
    ])
}

fn compiled_program_retained_bytes(program: &CompiledProgram<StdTensorOp>) -> usize {
    saturating_sum([
        size_of::<CompiledProgram<StdTensorOp>>(),
        vec_retained_bytes(&program.instructions),
        vec_retained_bytes(&program.input_slots),
        vec_retained_bytes(&program.output_slots),
        saturating_sum(program.instructions.iter().map(instruction_retained_bytes)),
    ])
}

fn instruction_retained_bytes(instruction: &Instruction<StdTensorOp>) -> usize {
    saturating_sum([
        size_of::<Instruction<StdTensorOp>>(),
        std_tensor_op_retained_bytes(&instruction.operation),
        vec_retained_bytes(&instruction.inputs),
        vec_retained_bytes(&instruction.outputs),
    ])
}

fn std_tensor_op_retained_bytes(op: &StdTensorOp) -> usize {
    match op {
        StdTensorOp::DotGeneral { config } => saturating_sum([
            vec_retained_bytes(&config.lhs_contracting_dims),
            vec_retained_bytes(&config.rhs_contracting_dims),
            vec_retained_bytes(&config.lhs_batch_dims),
            vec_retained_bytes(&config.rhs_batch_dims),
        ]),
        StdTensorOp::Transpose { perm } => vec_retained_bytes(perm),
        StdTensorOp::Reshape { to_shape } => vec_retained_bytes(to_shape),
        StdTensorOp::BroadcastInDim { shape, dims } => {
            saturating_sum([vec_retained_bytes(shape), vec_retained_bytes(dims)])
        }
        StdTensorOp::Constant { bytes, .. } => vec_retained_bytes(bytes),
        StdTensorOp::ReduceSum { axes }
        | StdTensorOp::ReduceProd { axes }
        | StdTensorOp::ReduceMax { axes }
        | StdTensorOp::ReduceMin { axes }
        | StdTensorOp::Reverse { axes } => vec_retained_bytes(axes),
        StdTensorOp::DynamicSlice { slice_sizes } => vec_retained_bytes(slice_sizes),
        StdTensorOp::GatherDynamicSliceSizes {
            offset_dims,
            collapsed_slice_dims,
            start_index_map,
            slice_sizes,
            ..
        } => saturating_sum([
            vec_retained_bytes(offset_dims),
            vec_retained_bytes(collapsed_slice_dims),
            vec_retained_bytes(start_index_map),
            vec_retained_bytes(slice_sizes),
        ]),
        _ => 0,
    }
}

fn try_execute_eager_broadcast_multiply_pattern(
    instructions: &[Instruction<StdTensorOp>],
    instruction_idx: usize,
    slots: &[Option<EagerTensor>],
    output_slots: &[usize],
) -> Result<Option<(usize, EagerTensor)>> {
    if instruction_idx + 2 >= instructions.len() {
        return Ok(None);
    }
    let lhs_bc = &instructions[instruction_idx];
    let rhs_bc = &instructions[instruction_idx + 1];
    let multiply = &instructions[instruction_idx + 2];

    let StdTensorOp::BroadcastInDim {
        shape: lhs_shape_exprs,
        dims: lhs_dims,
    } = &lhs_bc.operation
    else {
        return Ok(None);
    };
    let StdTensorOp::BroadcastInDim {
        shape: rhs_shape_exprs,
        dims: rhs_dims,
    } = &rhs_bc.operation
    else {
        return Ok(None);
    };
    if !matches!(multiply.operation, StdTensorOp::Mul)
        || lhs_bc.outputs.len() != 1
        || rhs_bc.outputs.len() != 1
        || multiply.outputs.len() != 1
        || multiply.inputs.len() != 2
        || lhs_bc.inputs.is_empty()
        || rhs_bc.inputs.is_empty()
        || multiply.inputs[0] != lhs_bc.outputs[0]
        || multiply.inputs[1] != rhs_bc.outputs[0]
    {
        return Ok(None);
    }

    let lhs_bc_slot = lhs_bc.outputs[0];
    let rhs_bc_slot = rhs_bc.outputs[0];
    if output_slots.contains(&lhs_bc_slot)
        || output_slots.contains(&rhs_bc_slot)
        || instructions[instruction_idx + 3..]
            .iter()
            .any(|instr| instr.inputs.contains(&lhs_bc_slot) || instr.inputs.contains(&rhs_bc_slot))
    {
        return Ok(None);
    }

    let lhs = slot_tensor(slots, lhs_bc.inputs[0])?;
    let rhs = slot_tensor(slots, rhs_bc.inputs[0])?;
    let lhs_shape = eval_shape_exprs(slots, &lhs_bc.inputs, lhs_shape_exprs)?;
    let rhs_shape = eval_shape_exprs(slots, &rhs_bc.inputs, rhs_shape_exprs)?;
    let Some(output) =
        backend_broadcast_multiply_untracked(lhs, &lhs_shape, lhs_dims, rhs, &rhs_shape, rhs_dims)?
    else {
        return Ok(None);
    };

    Ok(Some((multiply.outputs[0], output)))
}

#[allow(clippy::too_many_arguments)]
fn backend_broadcast_multiply_untracked(
    lhs: &EagerTensor,
    lhs_shape: &[usize],
    lhs_dims: &[usize],
    rhs: &EagerTensor,
    rhs_shape: &[usize],
    rhs_dims: &[usize],
) -> Result<Option<EagerTensor>> {
    if !Arc::ptr_eq(lhs.runtime(), rhs.runtime()) {
        return Err(tenferro_runtime::Error::ContextMismatch {
            lhs: lhs.ctx_id(),
            rhs: rhs.ctx_id(),
        }
        .into());
    }
    if lhs.tracks_grad() || rhs.tracks_grad() {
        return Ok(None);
    }

    let runtime = lhs.runtime();
    let value = runtime.with_execution_session(|backend| {
        backend.execute_broadcast_multiply_value(
            lhs.tensor_read(),
            lhs_shape,
            lhs_dims,
            rhs.tensor_read(),
            rhs_shape,
            rhs_dims,
        )
    })??;

    Ok(value
        .map(|value| adopt_untracked_eager_value(runtime.clone(), value))
        .transpose()?)
}

fn eval_shape_exprs(
    slots: &[Option<EagerTensor>],
    input_slots: &[usize],
    shape: &[DimExpr],
) -> Result<Vec<usize>> {
    let inputs = input_slots
        .iter()
        .map(|&slot| slot_tensor(slots, slot))
        .collect::<Result<Vec<_>>>()?;
    let input_shapes = inputs
        .iter()
        .map(|tensor| tensor.shape())
        .collect::<Vec<_>>();
    DimExpr::eval_all(shape, &input_shapes).map_err(|error| {
        runtime_extension_error(
            "einsum",
            ErrorKind::Validation(ValidationKind::InvalidArgument),
            error,
        )
    })
}

fn slot_tensor(slots: &[Option<EagerTensor>], slot: usize) -> Result<&EagerTensor> {
    slots.get(slot).and_then(Option::as_ref).ok_or_else(|| {
        Error::Runtime(tenferro_runtime::Error::MissingInput(format!(
            "expanded eager einsum missing value for slot {slot}"
        )))
    })
}

fn infer_eager_output_shape(
    subscripts: &EinsumSubscripts,
    inputs: &[&EagerTensor],
) -> Result<Vec<tenferro_runtime::SymDim>> {
    if inputs.is_empty() {
        return Err(Error::invalid_argument(
            "einsum",
            "inputs",
            "einsum requires at least one input tensor",
        ));
    }
    if subscripts.inputs.len() != inputs.len() {
        return Err(Error::invalid_argument(
            "einsum",
            "inputs",
            format!(
                "einsum subscripts expect {} inputs, got {}",
                subscripts.inputs.len(),
                inputs.len()
            ),
        ));
    }

    let mut label_dims = std::collections::HashMap::new();
    for (labels, tensor) in subscripts.inputs.iter().zip(inputs.iter()) {
        let shape = tensor.shape();
        if labels.len() != shape.len() {
            return Err(Error::validation(
                "einsum",
                ValidationError::RankMismatch {
                    expected: labels.len(),
                    actual: shape.len(),
                },
            ));
        }
        for (&label, &dim) in labels.iter().zip(shape.iter()) {
            if let Some(existing) = label_dims.insert(label, dim) {
                if existing != dim {
                    return Err(Error::validation(
                        "einsum",
                        ShapeMismatch::ExpectedActual {
                            expected: tenferro_tensor::ShapeVec::from_vec(vec![existing]),
                            actual: tenferro_tensor::ShapeVec::from_vec(vec![dim]),
                        }
                        .into(),
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
                .map(tenferro_runtime::SymDim::from)
                .ok_or_else(|| {
                    Error::invalid_argument(
                        "einsum",
                        "output",
                        format!("einsum output label {label} is missing from input labels"),
                    )
                })
        })
        .collect()
}

fn runtime_extension_error<E>(op: &'static str, kind: ErrorKind, source: E) -> Error
where
    E: StdError + Send + Sync + 'static,
{
    Error::Runtime(tenferro_runtime::Error::extension(
        op,
        ErrorPhase::Execution,
        EINSUM_EXTENSION_FAMILY_ID,
        kind,
        source,
    ))
}

fn runtime_internal(message: impl Into<String>) -> Error {
    Error::Runtime(tenferro_runtime::Error::Internal(message.into()))
}

fn runtime_missing(message: impl Into<String>) -> Error {
    Error::Runtime(tenferro_runtime::Error::MissingInput(message.into()))
}

/// Execute a NumPy-style tensor contraction on [`EagerTensor`] values.
///
/// This helper lives in the einsum extension trait surface because it is
/// contraction sugar over `dot_general`, not a linear algebra facade.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::Tensor;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_einsum::{EagerTensorEinsumExt, TensorDotAxes};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let lhs = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
///     ctx.clone(),
/// ).unwrap();
/// let rhs = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap(),
///     ctx,
/// ).unwrap();
/// let out = lhs.tensordot(&rhs, TensorDotAxes::Count(1)).unwrap();
///
/// assert_eq!(out.shape(), &[2, 4]);
/// # Ok::<(), tenferro_einsum::Error>(())
/// ```
///
/// # Errors
///
/// Returns [`Error::Validation`] with `AxisOutOfBounds`, `DuplicateAxis`,
/// `RankMismatch`, or `ShapeMismatch` for invalid contraction axes and shapes,
/// or [`Error::Runtime`] for eager backend execution failures.
pub fn tensordot(
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> Result<EagerTensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.shape().len(), rhs.shape().len())?;
    crate::tensordot::validate_concrete_contract_dims(lhs.shape(), rhs.shape(), &config)?;
    lhs.dot_general(rhs, config).map_err(Error::Runtime)
}

#[cfg(test)]
mod tests;
