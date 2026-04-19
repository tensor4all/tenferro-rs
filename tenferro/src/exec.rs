use std::sync::Arc;

use crate::compiler::compile_std_to_exec;
use crate::engine::NaryEinsumCache;
use crate::error::{Error, Result};
use computegraph::compile::compile;
use computegraph::fragment::FragmentBuilder;
use computegraph::materialize::materialize_merge;
use computegraph::resolve::resolve;
use computegraph::types::{GlobalValKey, ValRef};
use num_complex::{Complex32, Complex64};
use tenferro_algebra::Semiring;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::input_key::TensorInputKey;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::cpu::structural::{
    typed_broadcast_in_dim, typed_embed_diagonal, typed_extract_diagonal, typed_reshape,
    typed_transpose,
};
use tenferro_tensor::validate::validate_nonsingular_u;
use tenferro_tensor::Error as TensorError;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig,
    Tensor, TensorBackend, TensorExec, TypedTensor,
};
// `SemiringBackend` is the legacy algebra-generic backend trait, deprecated in
// design_v3 Stage 2 (see `docs/design/design_v3/30-algebra-and-tropical.md`)
// and scheduled for removal in Stage 6. It is only used by `eval_semiring_ir`
// below, which is itself deprecated; suppress the import-level warning at the
// narrow scope of the import.
#[allow(deprecated)]
use tenferro_tensor::SemiringBackend;

#[derive(Clone, Debug)]
pub enum ExecOp {
    Transpose {
        perm: Vec<usize>,
    },
    Reshape {
        shape: Vec<DimExpr>,
    },
    BroadcastInDim {
        shape: Vec<DimExpr>,
        dims: Vec<usize>,
    },
    Convert {
        to: DType,
    },
    Constant {
        dtype: DType,
        bytes: Vec<u8>,
    },
    DotGeneral(DotGeneralConfig),
    NaryEinsum {
        subscripts: String,
    },
    ReduceSum {
        axes: Vec<usize>,
    },
    ExtractDiag {
        axis_a: usize,
        axis_b: usize,
    },
    EmbedDiag {
        axis_a: usize,
        axis_b: usize,
    },
    Tril {
        k: i64,
    },
    Triu {
        k: i64,
    },
    Add,
    Multiply,
    Negate,
    Conj,
    Divide,
    Abs,
    Sign,
    Maximum,
    Minimum,
    Compare(CompareDir),
    Select,
    Clamp,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sqrt,
    Rsqrt,
    Pow,
    Expm1,
    Log1p,
    Gather(GatherConfig),
    Scatter(ScatterConfig),
    Slice(SliceConfig),
    DynamicSlice {
        slice_sizes: Vec<usize>,
    },
    Pad(PadConfig),
    Concatenate {
        axis: usize,
    },
    Reverse {
        axes: Vec<usize>,
    },
    ShapeOf {
        axis: usize,
    },
    DynamicTruncate {
        axis: usize,
    },
    PadToMatch {
        axis: usize,
    },
    ReduceProd {
        axes: Vec<usize>,
    },
    ReduceMax {
        axes: Vec<usize>,
    },
    ReduceMin {
        axes: Vec<usize>,
    },
    Cholesky,
    Svd {
        eps: f64,
    },
    Qr,
    Lu,
    Eigh {
        eps: f64,
    },
    Eig,
    ValidateNonsingular,
    TriangularSolve {
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    },
}

#[derive(Clone, Debug)]
pub struct ExecInstruction {
    pub op: ExecOp,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub dtype: tenferro_tensor::DType,
    pub output_shapes: Vec<Vec<tenferro_ops::dim_expr::DimExpr>>,
    pub last_use: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct ExecProgram {
    pub instructions: Vec<ExecInstruction>,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
    pub n_slots: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchMode {
    Unsegmented,
    Segmented,
}

pub(crate) fn get<'a, T>(
    slots: &'a [Option<T>],
    input_slots: &[usize],
    idx: usize,
) -> Result<&'a T> {
    let slot = input_slots[idx];
    slots[slot]
        .as_ref()
        .ok_or(TensorError::MissingValue { slot }.into())
}

pub(crate) fn initialize_slots(program: &ExecProgram, inputs: Vec<Tensor>) -> Vec<Option<Tensor>> {
    let mut slots: Vec<Option<Tensor>> = vec![None; program.n_slots];
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }
    slots
}

pub(crate) fn collect_outputs(
    program: &ExecProgram,
    mut slots: Vec<Option<Tensor>>,
) -> Result<Vec<Tensor>> {
    program
        .output_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .take()
                .ok_or(TensorError::MissingValue { slot }.into())
        })
        .collect()
}

pub(crate) fn is_host_instruction(inst: &ExecInstruction) -> bool {
    matches!(
        &inst.op,
        ExecOp::ShapeOf { .. }
            | ExecOp::DynamicTruncate { .. }
            | ExecOp::PadToMatch { .. }
            | ExecOp::Constant { .. }
            | ExecOp::ValidateNonsingular
    )
}

pub(crate) fn is_ffi_instruction(inst: &ExecInstruction) -> bool {
    matches!(
        &inst.op,
        ExecOp::DotGeneral(_)
            | ExecOp::NaryEinsum { .. }
            | ExecOp::Cholesky
            | ExecOp::Svd { .. }
            | ExecOp::Qr
            | ExecOp::Lu
            | ExecOp::Eigh { .. }
            | ExecOp::Eig
            | ExecOp::TriangularSolve { .. }
    )
}

pub(crate) fn resolve_tensor_shape_exprs(
    slots: &[Option<Tensor>],
    input_slots: &[usize],
    exprs: &[DimExpr],
) -> Result<Vec<usize>> {
    let mut input_shapes = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        input_shapes.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?
                .shape(),
        );
    }
    Ok(DimExpr::eval_all(exprs, &input_shapes))
}

fn resolve_semiring_shape_exprs<Alg: Semiring>(
    slots: &[Option<TypedTensor<Alg::Scalar>>],
    input_slots: &[usize],
    exprs: &[DimExpr],
) -> Result<Vec<usize>> {
    let mut input_shapes = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        input_shapes.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?
                .shape
                .as_slice(),
        );
    }
    Ok(DimExpr::eval_all(exprs, &input_shapes))
}

/// Evaluate an [`ExecProgram`] using segmented dispatch.
///
/// Consecutive fusible ops are executed within one backend execution session.
///
/// # Examples
///
/// ```
/// use tenferro::exec::{eval_exec_ir, ExecProgram};
/// use tenferro::CpuBackend;
///
/// let _eval: fn(&mut CpuBackend, &ExecProgram, Vec<tenferro::Tensor>) -> tenferro::error::Result<Vec<tenferro::Tensor>> =
///     eval_exec_ir::<CpuBackend>;
/// ```
pub fn eval_exec_ir<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    crate::segment::eval_exec_segmented(backend, program, inputs)
}

/// Evaluate an [`ExecProgram`] one instruction at a time.
///
/// This is retained for parity tests against segmented dispatch.
///
/// # Examples
///
/// ```
/// use tenferro::exec::{eval_exec_ir_unsegmented, ExecProgram};
/// use tenferro::CpuBackend;
///
/// let _eval: fn(&mut CpuBackend, &ExecProgram, Vec<tenferro::Tensor>) -> tenferro::error::Result<Vec<tenferro::Tensor>> =
///     eval_exec_ir_unsegmented::<CpuBackend>;
/// ```
pub fn eval_exec_ir_unsegmented<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
) -> Result<Vec<Tensor>> {
    let mut cache = NaryEinsumCache::new(
        std::num::NonZeroUsize::new(crate::engine::DEFAULT_EINSUM_CACHE_CAPACITY)
            .expect("DEFAULT_EINSUM_CACHE_CAPACITY must be non-zero"),
    );
    eval_exec_ir_unsegmented_with_cache(backend, program, inputs, &mut cache)
}

pub(crate) fn eval_exec_ir_unsegmented_with_cache<B: TensorBackend>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<Tensor>,
    cache: &mut NaryEinsumCache,
) -> Result<Vec<Tensor>> {
    let mut slots = initialize_slots(program, inputs);

    for inst in &program.instructions {
        if is_host_instruction(inst) {
            execute_host_instruction(backend, &mut slots, inst)?;
        } else if is_ffi_instruction(inst) {
            execute_ffi_instruction(backend, &mut slots, inst, DispatchMode::Unsegmented, cache)?;
        } else {
            let result =
                backend.with_exec_session(|exec| execute_backend_op(exec, &slots, inst))?;
            slots[inst.output_slots[0]] = Some(result);
        }
        reclaim_last_use_inputs_backend(&mut slots, inst, backend);
    }

    collect_outputs(program, slots)
}

pub(crate) fn execute_backend_op(
    exec: &mut dyn TensorExec,
    slots: &[Option<Tensor>],
    inst: &ExecInstruction,
) -> Result<Tensor> {
    let result = match &inst.op {
        ExecOp::Transpose { perm } => exec.transpose(get(slots, &inst.input_slots, 0)?, perm)?,
        ExecOp::Reshape { shape } => {
            let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
            exec.reshape(get(slots, &inst.input_slots, 0)?, &shape)?
        }
        ExecOp::BroadcastInDim { shape, dims } => {
            let shape = resolve_tensor_shape_exprs(slots, &inst.input_slots, shape)?;
            exec.broadcast_in_dim(get(slots, &inst.input_slots, 0)?, &shape, dims)?
        }
        ExecOp::Convert { to } => exec.convert(get(slots, &inst.input_slots, 0)?, *to)?,
        ExecOp::ReduceSum { axes } => exec.reduce_sum(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ExtractDiag { axis_a, axis_b } => {
            exec.extract_diagonal(get(slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?
        }
        ExecOp::EmbedDiag { axis_a, axis_b } => {
            exec.embed_diagonal(get(slots, &inst.input_slots, 0)?, *axis_a, *axis_b)?
        }
        ExecOp::Tril { k } => exec.tril(get(slots, &inst.input_slots, 0)?, *k)?,
        ExecOp::Triu { k } => exec.triu(get(slots, &inst.input_slots, 0)?, *k)?,
        ExecOp::Add => exec.add(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Multiply => exec.mul(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Negate => exec.neg(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Conj => exec.conj(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Divide => exec.div(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Abs => exec.abs(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Sign => exec.sign(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Maximum => exec.maximum(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Minimum => exec.minimum(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Compare(dir) => exec.compare(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            dir,
        )?,
        ExecOp::Select => exec.select(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            get(slots, &inst.input_slots, 2)?,
        )?,
        ExecOp::Clamp => exec.clamp(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            get(slots, &inst.input_slots, 2)?,
        )?,
        ExecOp::Exp => exec.exp(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Log => exec.log(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Sin => exec.sin(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Cos => exec.cos(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Tanh => exec.tanh(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Sqrt => exec.sqrt(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Rsqrt => exec.rsqrt(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Pow => exec.pow(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
        )?,
        ExecOp::Expm1 => exec.expm1(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Log1p => exec.log1p(get(slots, &inst.input_slots, 0)?)?,
        ExecOp::Gather(config) => exec.gather(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            config,
        )?,
        ExecOp::Scatter(config) => exec.scatter(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            get(slots, &inst.input_slots, 2)?,
            config,
        )?,
        ExecOp::Slice(config) => exec.slice(get(slots, &inst.input_slots, 0)?, config)?,
        ExecOp::DynamicSlice { slice_sizes } => exec.dynamic_slice(
            get(slots, &inst.input_slots, 0)?,
            get(slots, &inst.input_slots, 1)?,
            slice_sizes,
        )?,
        ExecOp::Pad(config) => exec.pad(get(slots, &inst.input_slots, 0)?, config)?,
        ExecOp::Concatenate { axis } => {
            let inputs = collect_tensor_refs(slots, &inst.input_slots)?;
            exec.concatenate(&inputs, *axis)?
        }
        ExecOp::Reverse { axes } => exec.reverse(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ReduceProd { axes } => exec.reduce_prod(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ReduceMax { axes } => exec.reduce_max(get(slots, &inst.input_slots, 0)?, axes)?,
        ExecOp::ReduceMin { axes } => exec.reduce_min(get(slots, &inst.input_slots, 0)?, axes)?,
        other => {
            return Err(Error::Internal(format!(
                "host or FFI op reached backend executor: {other:?}"
            )))
        }
    };
    Ok(result)
}

pub(crate) fn execute_host_instruction<B: TensorBackend>(
    backend: &mut B,
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
) -> Result<()> {
    match &inst.op {
        ExecOp::ShapeOf { axis } => {
            let input = get(slots, &inst.input_slots, 0)?;
            if *axis >= input.shape().len() {
                return Err(Error::Internal(format!(
                    "ShapeOf: axis {} out of bounds for rank {}",
                    axis,
                    input.shape().len()
                )));
            }
            let host = Tensor::F64(TypedTensor::from_vec(
                vec![],
                vec![input.shape()[*axis] as f64],
            ));
            slots[inst.output_slots[0]] = Some(backend.upload_host_tensor(&host)?);
        }
        ExecOp::DynamicTruncate { axis } => {
            let input = get(slots, &inst.input_slots, 0)?;
            if *axis >= input.shape().len() {
                return Err(Error::Internal(format!(
                    "DynamicTruncate: axis {} out of bounds for rank {}",
                    axis,
                    input.shape().len()
                )));
            }
            let size_tensor = backend.download_to_host(get(slots, &inst.input_slots, 1)?)?;
            let axis_extent = input.shape()[*axis];
            let size_f64 = scalar_size_value(&size_tensor)?;
            let rounded_size = if size_f64.is_finite() {
                size_f64.round()
            } else {
                0.0
            };
            let size = rounded_size.max(0.0).min(axis_extent as f64) as usize;
            let rank = input.shape().len();
            let mut limits = input.shape().to_vec();
            limits[*axis] = size;
            let config = SliceConfig {
                starts: vec![0; rank],
                limits,
                strides: vec![1; rank],
            };
            slots[inst.output_slots[0]] = Some(backend.slice(input, &config)?);
        }
        ExecOp::PadToMatch { axis } => {
            let input = get(slots, &inst.input_slots, 0)?;
            let reference = get(slots, &inst.input_slots, 1)?;
            if *axis >= input.shape().len() {
                return Err(Error::Internal(format!(
                    "PadToMatch: axis {} out of bounds for rank {}",
                    axis,
                    input.shape().len()
                )));
            }
            let target_size = reference.shape()[*axis];
            let current_size = input.shape()[*axis];
            if current_size >= target_size {
                slots[inst.output_slots[0]] = Some(input.clone());
            } else {
                let rank = input.shape().len();
                let mut high = vec![0i64; rank];
                high[*axis] = (target_size - current_size) as i64;
                let config = PadConfig {
                    edge_padding_low: vec![0i64; rank],
                    edge_padding_high: high,
                    interior_padding: vec![0i64; rank],
                };
                slots[inst.output_slots[0]] = Some(backend.pad(input, &config)?);
            }
        }
        ExecOp::Constant { dtype, bytes } => {
            let host = constant_tensor(*dtype, bytes);
            slots[inst.output_slots[0]] = Some(backend.upload_host_tensor(&host)?);
        }
        ExecOp::ValidateNonsingular => {
            let input = get(slots, &inst.input_slots, 0)?;
            let host_input = backend.download_to_host(input)?;
            validate_nonsingular_u(&host_input)?;
            slots[inst.output_slots[0]] = Some(input.clone());
        }
        other => {
            return Err(Error::Internal(format!(
                "non-host op reached host executor: {other:?}"
            )))
        }
    }
    Ok(())
}

pub(crate) fn execute_ffi_instruction<B: TensorBackend>(
    backend: &mut B,
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    mode: DispatchMode,
    cache: &mut NaryEinsumCache,
) -> Result<()> {
    match &inst.op {
        ExecOp::DotGeneral(config) => {
            let result = backend.dot_general(
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                config,
            )?;
            slots[inst.output_slots[0]] = Some(result);
        }
        ExecOp::NaryEinsum { subscripts } => {
            let inputs = collect_tensor_refs(slots, &inst.input_slots)?;
            let result = execute_nary_einsum(backend, &inputs, subscripts, mode, cache)?;
            slots[inst.output_slots[0]] = Some(result);
        }
        ExecOp::Cholesky => {
            let result = backend.cholesky(get(slots, &inst.input_slots, 0)?)?;
            slots[inst.output_slots[0]] = Some(result);
        }
        ExecOp::Svd { .. } => {
            let results = backend.svd(get(slots, &inst.input_slots, 0)?)?;
            assign_multi_output(slots, inst, results, "svd")?;
        }
        ExecOp::Qr => {
            let results = backend.qr(get(slots, &inst.input_slots, 0)?)?;
            assign_multi_output(slots, inst, results, "qr")?;
        }
        ExecOp::Lu => {
            let results = backend.lu(get(slots, &inst.input_slots, 0)?)?;
            assign_multi_output(slots, inst, results, "lu")?;
        }
        ExecOp::Eigh { .. } => {
            let results = backend.eigh(get(slots, &inst.input_slots, 0)?)?;
            assign_multi_output(slots, inst, results, "eigh")?;
        }
        ExecOp::Eig => {
            let results = backend.eig(get(slots, &inst.input_slots, 0)?)?;
            assign_multi_output(slots, inst, results, "eig")?;
        }
        ExecOp::TriangularSolve {
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        } => {
            let result = backend.triangular_solve(
                get(slots, &inst.input_slots, 0)?,
                get(slots, &inst.input_slots, 1)?,
                *left_side,
                *lower,
                *transpose_a,
                *unit_diagonal,
            )?;
            slots[inst.output_slots[0]] = Some(result);
        }
        other => {
            return Err(Error::Internal(format!(
                "non-ffi op reached ffi executor: {other:?}"
            )))
        }
    }
    Ok(())
}

fn assign_multi_output(
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    results: Vec<Tensor>,
    op_name: &str,
) -> Result<()> {
    if results.len() != inst.output_slots.len() {
        return Err(Error::Internal(format!(
            "{op_name} produced {} outputs for {} slots",
            results.len(),
            inst.output_slots.len()
        )));
    }
    for (slot, tensor) in inst.output_slots.iter().copied().zip(results.into_iter()) {
        slots[slot] = Some(tensor);
    }
    Ok(())
}

fn collect_tensor_refs<'a>(
    slots: &'a [Option<Tensor>],
    input_slots: &[usize],
) -> Result<Vec<&'a Tensor>> {
    let mut inputs = Vec::with_capacity(input_slots.len());
    for &slot in input_slots {
        inputs.push(
            slots[slot]
                .as_ref()
                .ok_or(TensorError::MissingValue { slot })?,
        );
    }
    Ok(inputs)
}

fn execute_nary_einsum<B: TensorBackend>(
    backend: &mut B,
    inputs: &[&Tensor],
    subscripts: &str,
    mode: DispatchMode,
    cache: &mut NaryEinsumCache,
) -> Result<Tensor> {
    use tenferro_einsum::{
        build_einsum_fragment, ContractionOptimizerOptions, ContractionTree, Subscripts,
    };

    if inputs.is_empty() {
        return Err(Error::ContractionError(
            "nary einsum requires at least one input tensor".into(),
        ));
    }

    let subs =
        Subscripts::parse(subscripts).map_err(|e| Error::InvalidSubscripts(format!("{e}")))?;
    let shapes: Vec<Vec<usize>> = inputs
        .iter()
        .map(|tensor| tensor.shape().to_vec())
        .collect();
    let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let cache_key = (subscripts.to_string(), shapes.clone());
    let tree_arc = if let Some(cached) = cache.get(&cache_key) {
        cached.clone()
    } else {
        let tree = Arc::new(
            ContractionTree::optimize_with_options(
                &subs,
                &shape_refs,
                &ContractionOptimizerOptions::default(),
            )
            .map_err(|e| Error::ContractionError(format!("{e}")))?,
        );
        cache.put(cache_key, tree.clone());
        tree
    };
    let tree = tree_arc.as_ref();

    let mut builder = FragmentBuilder::<StdTensorOp>::new();
    let mut input_vals = Vec::with_capacity(inputs.len());
    for input_idx in 0..inputs.len() {
        let local = builder.add_input(TensorInputKey::User {
            id: input_idx as u64,
        });
        input_vals.push(ValRef::Local(local));
    }

    let result_ref = build_einsum_fragment(&mut builder, &tree, &input_vals, &shapes);
    let result_local = match result_ref {
        ValRef::Local(local) => local,
        ValRef::External(_) => {
            return Err(Error::Internal(
                "runtime nary einsum builder returned an external value".into(),
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
                    Error::Internal(format!(
                        "runtime nary einsum input {input_idx} missing for subscripts {subscripts}"
                    ))
                })?;
                program_inputs.push((*tensor).clone());
                input_dtypes.push(tensor.dtype());
                input_shapes.push(DimExpr::from_concrete(tensor.shape()));
            }
            other => {
                return Err(Error::Internal(format!(
                    "unexpected runtime nary einsum input key: {other:?}"
                )))
            }
        }
    }
    let program = compile_std_to_exec(&compiled, &input_dtypes, &input_shapes);

    let mut outputs = match mode {
        DispatchMode::Unsegmented => {
            eval_exec_ir_unsegmented_with_cache(backend, &program, program_inputs, cache)?
        }
        DispatchMode::Segmented => crate::segment::eval_exec_segmented_with_cache(
            backend,
            &program,
            program_inputs,
            cache,
        )?,
    };
    if outputs.len() != 1 {
        return Err(Error::Internal(format!(
            "runtime nary einsum expected 1 output, got {}",
            outputs.len()
        )));
    }
    Ok(outputs.remove(0))
}

pub(crate) fn constant_tensor(dtype: DType, bytes: &[u8]) -> Tensor {
    match dtype {
        DType::F64 => Tensor::F64(TypedTensor::from_vec(
            vec![],
            vec![f64::from_le_bytes(exact_bytes::<8>(dtype, bytes))],
        )),
        DType::F32 => Tensor::F32(TypedTensor::from_vec(
            vec![],
            vec![f32::from_le_bytes(exact_bytes::<4>(dtype, bytes))],
        )),
        DType::C64 => {
            let data = exact_bytes::<16>(dtype, bytes);
            let mut re_bytes = [0u8; 8];
            let mut im_bytes = [0u8; 8];
            re_bytes.copy_from_slice(&data[..8]);
            im_bytes.copy_from_slice(&data[8..]);
            let re = f64::from_le_bytes(re_bytes);
            let im = f64::from_le_bytes(im_bytes);
            Tensor::C64(TypedTensor::from_vec(vec![], vec![Complex64::new(re, im)]))
        }
        DType::C32 => {
            let data = exact_bytes::<8>(dtype, bytes);
            let mut re_bytes = [0u8; 4];
            let mut im_bytes = [0u8; 4];
            re_bytes.copy_from_slice(&data[..4]);
            im_bytes.copy_from_slice(&data[4..]);
            let re = f32::from_le_bytes(re_bytes);
            let im = f32::from_le_bytes(im_bytes);
            Tensor::C32(TypedTensor::from_vec(vec![], vec![Complex32::new(re, im)]))
        }
    }
}

fn exact_bytes<const N: usize>(dtype: DType, bytes: &[u8]) -> [u8; N] {
    if bytes.len() != N {
        panic!(
            "constant {:?} expected {} bytes, got {}",
            dtype,
            N,
            bytes.len()
        );
    }
    let mut out = [0u8; N];
    out.copy_from_slice(bytes);
    out
}

fn scalar_size_value(size_tensor: &Tensor) -> Result<f64> {
    match size_tensor {
        Tensor::F64(inner) => Ok(inner.host_data()[0]),
        Tensor::F32(inner) => Ok(inner.host_data()[0] as f64),
        _ => Err(Error::Internal(
            "DynamicTruncate size must be an f32 or f64 scalar".into(),
        )),
    }
}

pub(crate) fn reclaim_last_use_inputs_exec(
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    exec: &mut dyn TensorExec,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(tensor) = slots[inst.input_slots[i]].take() {
                exec.reclaim_buffer(tensor);
            }
        }
    }
}

pub(crate) fn reclaim_last_use_inputs_backend<B: TensorBackend>(
    slots: &mut [Option<Tensor>],
    inst: &ExecInstruction,
    backend: &mut B,
) {
    for (i, &is_last) in inst.last_use.iter().enumerate() {
        if is_last {
            if let Some(tensor) = slots[inst.input_slots[i]].take() {
                backend.reclaim_buffer(tensor);
            }
        }
    }
}

/// Execute a semiring-lowered [`ExecProgram`] (legacy path).
///
/// **Deprecated**: non-mainline per
/// `docs/design/design_v3/30-algebra-and-tropical.md`; scheduled for removal
/// in Stage 6 of the `design_v3` migration plan.
#[deprecated(
    since = "design_v3-stage-2",
    note = "non-mainline per docs/design/design_v3/30-algebra-and-tropical.md; scheduled for removal in Stage 6"
)]
#[allow(deprecated)]
pub fn eval_semiring_ir<B, Alg>(
    backend: &mut B,
    program: &ExecProgram,
    inputs: Vec<TypedTensor<Alg::Scalar>>,
) -> Result<Vec<TypedTensor<Alg::Scalar>>>
where
    Alg: Semiring,
    B: SemiringBackend<Alg>,
{
    let mut slots: Vec<Option<TypedTensor<Alg::Scalar>>> = vec![None; program.n_slots];
    for (i, tensor) in inputs.into_iter().enumerate() {
        slots[program.input_slots[i]] = Some(tensor);
    }

    for inst in &program.instructions {
        let result = match &inst.op {
            ExecOp::Transpose { perm } => typed_transpose(get(&slots, &inst.input_slots, 0)?, perm),
            ExecOp::Reshape { shape } => {
                let shape = resolve_semiring_shape_exprs::<Alg>(&slots, &inst.input_slots, shape)?;
                typed_reshape(get(&slots, &inst.input_slots, 0)?, &shape)
            }
            ExecOp::BroadcastInDim { shape, dims } => {
                let shape = resolve_semiring_shape_exprs::<Alg>(&slots, &inst.input_slots, shape)?;
                typed_broadcast_in_dim(get(&slots, &inst.input_slots, 0)?, &shape, dims)
            }
            ExecOp::DotGeneral(config) => backend.batched_gemm(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
                config,
            ),
            ExecOp::ReduceSum { axes } => {
                backend.reduce_sum(get(&slots, &inst.input_slots, 0)?, axes)
            }
            ExecOp::ExtractDiag { axis_a, axis_b } => {
                typed_extract_diagonal(get(&slots, &inst.input_slots, 0)?, *axis_a, *axis_b)
            }
            ExecOp::EmbedDiag { axis_a, axis_b } => {
                typed_embed_diagonal(get(&slots, &inst.input_slots, 0)?, *axis_a, *axis_b)
            }
            ExecOp::Add => backend.add(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            ),
            ExecOp::Multiply => backend.mul(
                get(&slots, &inst.input_slots, 0)?,
                get(&slots, &inst.input_slots, 1)?,
            ),
            _ => panic!("non-semiring op in semiring program: {:?}", inst.op),
        };
        slots[inst.output_slots[0]] = Some(result?);
    }

    program
        .output_slots
        .iter()
        .map(|&slot| {
            slots[slot]
                .take()
                .ok_or(TensorError::MissingValue { slot }.into())
        })
        .collect()
}
