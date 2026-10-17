//! Eager einsum and tensordot on a borrowed [`EagerSession`].

use std::collections::hash_map::DefaultHasher;
use std::error::Error as StdError;
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::sync::Arc;

use computegraph::compile::{compile, CompiledProgram, Instruction};
use computegraph::graph::GraphBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{ValueKey, ValueRef};
use tenferro_ad::extension::{
    adopt_untracked_eager_value, apply_eager_with_targeted_extension_in_session,
    EagerExtensionBackendKind, EagerExtensionTarget,
};
use tenferro_ad::{EagerSession, EagerTensor};
use tenferro_cpu::CpuBackend;
#[cfg(feature = "cuda")]
use tenferro_gpu::cuda::CudaBackend;
#[cfg(feature = "webgpu")]
use tenferro_gpu::webgpu::WebGpuBackend;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{ErrorPhase, ExtensionCacheKey, ExtensionModule};
use tenferro_tensor::{ErrorKind, ShapeMismatch, ValidationError, ValidationKind};

use crate::binary_dot::{try_build_exact_output_binary_dot_plan, BinaryDotOperandOrder};
use crate::builder::build_einsum_graph;
use crate::cache::{
    saturating_sum, vec_retained_bytes, EINSUM_EAGER_EXPANDED_PROGRAMS_CACHE,
    EINSUM_EXTENSION_FAMILY_ID,
};
use crate::ellipsis::resolve_einsum_notation;
use crate::extension::EinsumExtensionOp;
use crate::optimize::{
    default_auto_options, hash_einsum_plan_spec, plan_specs_equal, resolve_plan_spec,
    EinsumPlanSpec,
};
use crate::{
    parse_einsum_notation, EinsumNotation, EinsumSubscripts, Error, Result, Subscripts,
    TensorDotAxes,
};

/// Eager einsum and tensordot on a runtime-bound borrowed eager session.
///
/// Every operation runs inside the caller's session, so it composes with other
/// eager operations in the same [`tenferro_ad::EagerRuntime::with_eager_session`]
/// callback and never reopens the runtime. The calling thread's `no_grad` and
/// `capture_trace` modes govern it like any other eager operation.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_einsum::EagerSessionEinsumExt;
///
/// let ctx = EagerRuntime::new()?;
/// let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?, ctx.clone())?;
/// let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?, ctx.clone())?;
/// let product = ctx.with_eager_session(|session| {
///     let c = session.einsum(&[&a, &b], "ij,jk->ik")?;
///     session.einsum(&[&c], "ij->")
/// })?;
/// assert_eq!(product.value()?.as_slice::<f64>()?, &[24.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "autodiff")))]
pub trait EagerSessionEinsumExt {
    /// Execute an einsum from string notation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_einsum::EagerSessionEinsumExt;
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?, ctx.clone())?;
    /// let dot = ctx.with_eager_session(|session| session.einsum(&[&x, &x], "i,i->"))?;
    /// assert_eq!(dot.value()?.as_slice::<f64>()?, &[13.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSubscripts`] for malformed notation,
    /// [`Error::Validation`] for rank/shape/dtype mismatches, or
    /// [`Error::Planning`] / [`Error::Runtime`] for contraction planning and
    /// execution failures, including inputs owned by another runtime.
    fn einsum(&mut self, inputs: &[&EagerTensor], subscripts: &str) -> Result<EagerTensor>;

    /// Execute an einsum from rank-unresolved notation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_einsum::{EagerSessionEinsumExt, EinsumAxis, EinsumNotation};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?, ctx.clone())?;
    /// // `...->...` keeps every axis the ellipsis covers.
    /// let notation = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[EinsumAxis::Ellipsis]);
    /// let same = ctx.with_eager_session(|session| session.einsum_notation(&[&x], &notation))?;
    /// assert_eq!(same.value()?.as_slice::<f64>()?, &[2.0, 3.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a typed validation, planning, or runtime error when notation or
    /// execution is invalid.
    fn einsum_notation(
        &mut self,
        inputs: &[&EagerTensor],
        notation: &EinsumNotation,
    ) -> Result<EagerTensor>;

    /// Execute an einsum from parsed integer labels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_einsum::{EagerSessionEinsumExt, EinsumSubscripts};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let x = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 3.0])?, ctx.clone())?;
    /// let subscripts = EinsumSubscripts::new(&[&[0], &[0]], &[]);
    /// let dot = ctx.with_eager_session(|session| session.einsum_subscripts(&[&x, &x], &subscripts))?;
    /// assert_eq!(dot.value()?.as_slice::<f64>()?, &[13.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for rank/shape/dtype mismatches,
    /// [`Error::Planning`] for an invalid contraction plan, or
    /// [`Error::Runtime`] for extension registration or backend execution
    /// failures.
    fn einsum_subscripts(
        &mut self,
        inputs: &[&EagerTensor],
        subscripts: &EinsumSubscripts,
    ) -> Result<EagerTensor>;

    /// Contract two eager tensors over the requested axes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
    /// use tenferro_einsum::{EagerSessionEinsumExt, TensorDotAxes};
    ///
    /// let ctx = EagerRuntime::new()?;
    /// let a = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])?, ctx.clone())?;
    /// let b = EagerTensor::from_tensor_in(Tensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])?, ctx.clone())?;
    /// let c = ctx.with_eager_session(|session| session.tensordot(&a, &b, TensorDotAxes::Count(1)))?;
    /// assert_eq!(c.shape(), &[2, 4]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] for invalid axes or mismatched contracted
    /// extents, or [`Error::Runtime`] for execution failures.
    fn tensordot(
        &mut self,
        lhs: &EagerTensor,
        rhs: &EagerTensor,
        axes: TensorDotAxes<'_>,
    ) -> Result<EagerTensor>;
}

impl EagerSessionEinsumExt for EagerSession<'_> {
    fn einsum(&mut self, inputs: &[&EagerTensor], subscripts: &str) -> Result<EagerTensor> {
        einsum(self, inputs, subscripts)
    }

    fn einsum_notation(
        &mut self,
        inputs: &[&EagerTensor],
        notation: &EinsumNotation,
    ) -> Result<EagerTensor> {
        einsum_notation(self, inputs, notation)
    }

    fn einsum_subscripts(
        &mut self,
        inputs: &[&EagerTensor],
        subscripts: &EinsumSubscripts,
    ) -> Result<EagerTensor> {
        einsum_subscripts_with_broadcast(self, inputs, subscripts, false)
    }

    fn tensordot(
        &mut self,
        lhs: &EagerTensor,
        rhs: &EagerTensor,
        axes: TensorDotAxes<'_>,
    ) -> Result<EagerTensor> {
        tensordot(self, lhs, rhs, axes)
    }
}

fn eager_extension_module(
    target: EagerExtensionTarget,
) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>> {
    let EagerExtensionTarget {
        engine_id,
        backend_kind,
    } = target;
    match backend_kind {
        EagerExtensionBackendKind::Cpu => {
            crate::extension::extension_module::<CpuBackend>(engine_id)
                .map_err(eager_runtime_config_error)
        }
        #[cfg(feature = "cuda")]
        EagerExtensionBackendKind::Cuda => {
            crate::extension::extension_module::<CudaBackend>(engine_id)
                .map_err(eager_runtime_config_error)
        }
        #[cfg(feature = "webgpu")]
        EagerExtensionBackendKind::WebGpu => {
            crate::extension::extension_module::<WebGpuBackend>(engine_id)
                .map_err(eager_runtime_config_error)
        }
    }
}

fn eager_runtime_config_error(
    source: tenferro_runtime::RuntimeConfigError,
) -> tenferro_runtime::Error {
    tenferro_runtime::Error::runtime_state_source(
        "tenferro_einsum::eager_extension_module",
        ErrorPhase::Execution,
        source,
    )
}

fn einsum(
    session: &mut EagerSession<'_>,
    inputs: &[&EagerTensor],
    subscripts: &str,
) -> Result<EagerTensor> {
    let notation = parse_einsum_notation(subscripts)?;
    einsum_notation(session, inputs, &notation)
}

fn einsum_notation(
    session: &mut EagerSession<'_>,
    inputs: &[&EagerTensor],
    notation: &EinsumNotation,
) -> Result<EagerTensor> {
    let shapes: Vec<&[usize]> = inputs.iter().map(|tensor| tensor.shape()).collect();
    let subscripts = resolve_einsum_notation(notation, &shapes)?;
    let subscripts = EinsumSubscripts::from(subscripts);
    let allow_broadcast = notation
        .inputs
        .iter()
        .chain(std::iter::once(&notation.output))
        .any(|term| term.contains(&crate::EinsumAxis::Ellipsis))
        || requires_broadcast(inputs, &subscripts);
    einsum_subscripts_with_broadcast(session, inputs, &subscripts, allow_broadcast)
}

fn einsum_subscripts_with_broadcast(
    session: &mut EagerSession<'_>,
    inputs: &[&EagerTensor],
    subscripts: &EinsumSubscripts,
    allow_broadcast: bool,
) -> Result<EagerTensor> {
    if let Some(result) = try_direct_binary_dot_general(session, inputs, subscripts) {
        return result;
    }

    let output_shape_hint = infer_eager_output_shape(subscripts, inputs)?;
    if !requires_broadcast(inputs, subscripts) {
        if let Some(result) = try_expand_eager_einsum(session, inputs, subscripts)? {
            return Ok(result);
        }
    }

    let plan_spec = EinsumPlanSpec::Auto(default_auto_options());
    let op = Arc::new(if allow_broadcast {
        EinsumExtensionOp::with_output_shape_hint_and_broadcast(
            subscripts.clone(),
            output_shape_hint,
            plan_spec,
            true,
        )
    } else {
        EinsumExtensionOp::with_output_shape_hint(subscripts.clone(), output_shape_hint, plan_spec)
    });
    let mut outputs = apply_eager_with_targeted_extension_in_session(
        session,
        op,
        inputs,
        eager_extension_module,
    )?;
    outputs.pop().ok_or_else(|| {
        Error::Runtime(tenferro_runtime::Error::MissingInput(
            "einsum extension produced no eager output".into(),
        ))
    })
}

fn try_direct_binary_dot_general(
    session: &mut EagerSession<'_>,
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
        let (lhs, rhs) = match plan.operand_order {
            BinaryDotOperandOrder::Original => (inputs[0], inputs[1]),
            BinaryDotOperandOrder::Swapped => (inputs[1], inputs[0]),
        };
        if !exact_dot_shapes(lhs.shape(), rhs.shape(), &plan.config) {
            return None;
        }
        return Some(
            session
                .dot_general(lhs, rhs, plan.config)
                .map_err(Error::Runtime),
        );
    }
    None
}

fn requires_broadcast(inputs: &[&EagerTensor], subscripts: &EinsumSubscripts) -> bool {
    let mut sizes = std::collections::HashMap::<u32, usize>::new();
    for (tensor, labels) in inputs.iter().zip(&subscripts.inputs) {
        for (&label, &size) in labels.iter().zip(tensor.shape()) {
            if let Some(previous) = sizes.insert(label, size) {
                if previous != size && (previous == 1 || size == 1) {
                    return true;
                }
            }
        }
    }
    false
}

fn exact_dot_shapes(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &tenferro_tensor::DotGeneralConfig,
) -> bool {
    config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
        .all(|(&lhs, &rhs)| lhs_shape[lhs] == rhs_shape[rhs])
        && config
            .lhs_batch_dims
            .iter()
            .zip(&config.rhs_batch_dims)
            .all(|(&lhs, &rhs)| lhs_shape[lhs] == rhs_shape[rhs])
}

fn try_expand_eager_einsum(
    session: &mut EagerSession<'_>,
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
        session,
        subscripts,
        &subs,
        &plan_spec,
        &shape_refs,
        &shapes,
    )?;
    execute_eager_einsum_program_in_session(session, inputs, &program)
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
    session: &mut EagerSession<'_>,
    subscripts: &EinsumSubscripts,
    subs: &Subscripts,
    plan_spec: &EinsumPlanSpec,
    shape_refs: &[&[usize]],
    shapes: &[Vec<usize>],
) -> Result<Arc<ExpandedEagerProgram>> {
    session.with_extension_caches(|caches| {
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

fn execute_eager_einsum_program_in_session(
    session: &mut EagerSession<'_>,
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
            session,
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
        let output = session
            .apply_standard_op(instr.operation.clone(), &input_refs)
            .map_err(Error::Runtime)?;
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
        // Inline axes are already counted in Instruction<StdTensorOp>.
        StdTensorOp::DotGeneral { config } => saturating_sum(
            [
                &config.lhs_contracting_dims,
                &config.rhs_contracting_dims,
                &config.lhs_batch_dims,
                &config.rhs_batch_dims,
            ]
            .into_iter()
            .filter(|axes| axes.is_heap())
            .map(|axes| axes.capacity().saturating_mul(size_of::<usize>())),
        ),
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
    session: &mut EagerSession<'_>,
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
    let Some(output) = backend_broadcast_multiply_untracked(
        session, lhs, &lhs_shape, lhs_dims, rhs, &rhs_shape, rhs_dims,
    )?
    else {
        return Ok(None);
    };

    Ok(Some((multiply.outputs[0], output)))
}

#[allow(clippy::too_many_arguments)]
fn backend_broadcast_multiply_untracked(
    session: &mut EagerSession<'_>,
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
    let value = session.backend_session().execute_broadcast_multiply_value(
        lhs.tensor_read(),
        lhs_shape,
        lhs_dims,
        rhs.tensor_read(),
        rhs_shape,
        rhs_dims,
    )?;

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
            if let Some(existing) = label_dims.get_mut(&label) {
                if *existing != dim && *existing != 1 && dim != 1 {
                    return Err(Error::validation(
                        "einsum",
                        ShapeMismatch::ExpectedActual {
                            expected: vec![*existing].into_iter().collect(),
                            actual: vec![dim].into_iter().collect(),
                        }
                        .into(),
                    ));
                }
                if *existing == 1 {
                    *existing = dim;
                }
            } else {
                label_dims.insert(label, dim);
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

fn tensordot(
    session: &mut EagerSession<'_>,
    lhs: &EagerTensor,
    rhs: &EagerTensor,
    axes: TensorDotAxes<'_>,
) -> Result<EagerTensor> {
    let config = crate::tensordot::dot_general_config(axes, lhs.shape().len(), rhs.shape().len())?;
    crate::tensordot::validate_concrete_contract_dims(lhs.shape(), rhs.shape(), &config)?;
    session
        .dot_general(lhs, rhs, config)
        .map_err(Error::Runtime)
}

#[cfg(test)]
mod tests;
