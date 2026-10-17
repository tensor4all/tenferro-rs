#![doc(hidden)]

//! Fused CPU elementwise execution.

/// Internal result alias for the fused CPU adapter.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu_fused::Result;
/// let result: Result<()> = Ok(());
/// assert!(result.is_ok());
/// ```
pub type Result<T> = tenferro_tensor::Result<T>;
pub use tenferro_tensor::{DType, Error, Tensor, TypedTensor};

use num_complex::{Complex32, Complex64};
use std::mem::size_of_val;
use strided_basic::{ErasedRawStridedPtr, ErasedRawStridedRef, ExecContext, KernelDType};
use strided_fused::{ErasedFusedPlan, FusedInst, FusedOp, FusedPlan};
use tenferro_cpu_basic::{
    erased_raw_strided_ref, erased_raw_strided_uninit_mut, typed_host_data, BufferPool, PoolScalar,
    PooledUninitOutput,
};
use tenferro_tensor::backend::{
    ElementwiseFusionInputView, ElementwiseFusionOp, ElementwiseFusionPlan,
};
use tenferro_tensor::col_major_strides;
use tenferro_tensor::{ShapeVec, StrideVec};

const ELEMENTWISE_FUSION_OP: &str = "execute_elementwise_fusion";

const ELEMENTWISE_FUSION_MIN_ELEMENTS: usize = 16 * 1024;

/// Largest input count `strided_fused::ErasedFusedPlan` accepts.
///
/// [`elementwise_fusion_with_pool`] declines larger plans, and the CPU runtime
/// engine reports this limit so the prepared planner does not plan such
/// regions at all.
///
/// # Examples
///
/// ```rust
/// assert_eq!(tenferro_cpu_fused::ERASED_FUSION_MAX_INPUTS, 4);
/// ```
pub const ERASED_FUSION_MAX_INPUTS: usize = 4;

fn validate_elementwise_fusion_inputs(
    inputs: &[&Tensor],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<bool> {
    if inputs.len() != plan.input_count() {
        return Err(crate::Error::invalid_argument(
            ELEMENTWISE_FUSION_OP,
            "inputs",
            format!(
                "plan expects {} inputs but backend received {}",
                plan.input_count(),
                inputs.len()
            ),
        ));
    }
    if plan.input_views().len() != plan.input_count() {
        return Err(crate::Error::invalid_argument(
            ELEMENTWISE_FUSION_OP,
            "input_views",
            format!(
                "plan has {} input views for {} inputs",
                plan.input_views().len(),
                plan.input_count()
            ),
        ));
    }
    if plan.outputs().is_empty() {
        return Ok(false);
    }
    for input in inputs {
        if input.dtype() != plan.dtype() {
            return Err(crate::Error::dtype_mismatch(
                ELEMENTWISE_FUSION_OP,
                input.dtype(),
                plan.dtype(),
            ));
        }
    }
    Ok(true)
}

fn strided_fused_op(op: ElementwiseFusionOp) -> FusedOp {
    match op {
        ElementwiseFusionOp::Add => FusedOp::Add,
        ElementwiseFusionOp::Multiply => FusedOp::Multiply,
        ElementwiseFusionOp::Negate => FusedOp::Negate,
        ElementwiseFusionOp::Conj => FusedOp::Conj,
        ElementwiseFusionOp::Divide => FusedOp::Divide,
        ElementwiseFusionOp::Abs => FusedOp::Abs,
        ElementwiseFusionOp::Maximum => FusedOp::Maximum,
        ElementwiseFusionOp::Minimum => FusedOp::Minimum,
        ElementwiseFusionOp::Clamp => FusedOp::Clamp,
        ElementwiseFusionOp::Exp => FusedOp::Exp,
        ElementwiseFusionOp::Log => FusedOp::Log,
        ElementwiseFusionOp::Sin => FusedOp::Sin,
        ElementwiseFusionOp::Cos => FusedOp::Cos,
        ElementwiseFusionOp::Tanh => FusedOp::Tanh,
        ElementwiseFusionOp::Sqrt => FusedOp::Sqrt,
        ElementwiseFusionOp::Rsqrt => FusedOp::Rsqrt,
        ElementwiseFusionOp::Pow => FusedOp::Pow,
        ElementwiseFusionOp::Expm1 => FusedOp::Expm1,
        ElementwiseFusionOp::Log1p => FusedOp::Log1p,
        // INVARIANT: `plan_uses_unfused_op` declines every plan containing
        // these ops before a strided plan is built; strided_fused has no
        // remainder or `erf` instruction.
        ElementwiseFusionOp::Remainder | ElementwiseFusionOp::Erf => {
            unreachable!("ops without a strided_fused instruction are declined before CPU fusion")
        }
    }
}

/// Whether the plan contains an op that `strided_fused` cannot replay.
///
/// `erf` has no `FusedOp` instruction yet, so a region containing it is
/// declined and its ops run unfused.
fn plan_uses_unfused_op(plan: &ElementwiseFusionPlan) -> bool {
    plan.ops().iter().any(|inst| {
        matches!(
            inst.op(),
            ElementwiseFusionOp::Remainder | ElementwiseFusionOp::Erf
        )
    })
}

fn plan_uses_ordered_op(plan: &ElementwiseFusionPlan) -> bool {
    plan.ops().iter().any(|inst| {
        matches!(
            inst.op(),
            ElementwiseFusionOp::Maximum
                | ElementwiseFusionOp::Minimum
                | ElementwiseFusionOp::Clamp
        )
    })
}

fn should_defer_to_broadcast_multiply_special_case(plan: &ElementwiseFusionPlan) -> bool {
    !plan.input_views().iter().all(|view| view.is_identity())
        && plan.ops().len() == 1
        && plan.outputs() == [plan.input_count()]
        && plan.ops()[0].op() == ElementwiseFusionOp::Multiply
}

fn strided_fused_plan(plan: &ElementwiseFusionPlan) -> FusedPlan {
    FusedPlan {
        input_count: plan.input_count(),
        outputs: plan.outputs().to_vec(),
        ops: plan
            .ops()
            .iter()
            .map(|inst| FusedInst {
                op: strided_fused_op(inst.op()),
                inputs: inst.inputs().to_vec(),
            })
            .collect(),
    }
}

fn kernel_dtype(dtype: DType) -> KernelDType {
    match dtype {
        DType::F32 => KernelDType::F32,
        DType::F64 => KernelDType::F64,
        DType::I32 => KernelDType::I32,
        DType::I64 => KernelDType::I64,
        DType::Bool => KernelDType::Bool,
        DType::C32 => KernelDType::C32,
        DType::C64 => KernelDType::C64,
        // INVARIANT: `KernelDType` is the fixed compiled-kernel vocabulary used by
        // the fused path; an externally defined scalar has no entry in it and is
        // rejected by `dtype_supports_erased_fusion` before this point.
        DType::External(_) => unreachable!("KernelDType covers the preset scalars"),
    }
}

fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: `data` is an aligned typed slice. The returned byte slice has
    // the same lifetime and exact byte length, and is read-only.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

struct ErasedFusionInput<'a> {
    data: &'a [u8],
    dims: ShapeVec,
    strides: StrideVec,
}

fn tensor_host_bytes<'a>(op: &'static str, input: &'a Tensor) -> crate::Result<&'a [u8]> {
    macro_rules! bytes {
        ($tensor:expr) => {
            typed_host_data(op, $tensor).map(typed_bytes)
        };
    }

    match input.dtype() {
        DType::F32 => bytes!(fused_host::<f32>(op, input)?),
        DType::F64 => bytes!(fused_host::<f64>(op, input)?),
        DType::I32 => bytes!(fused_host::<i32>(op, input)?),
        DType::I64 => bytes!(fused_host::<i64>(op, input)?),
        DType::Bool => bytes!(fused_host::<bool>(op, input)?),
        DType::C32 => bytes!(fused_host::<Complex32>(op, input)?),
        DType::C64 => bytes!(fused_host::<Complex64>(op, input)?),
        // A caller-owned payload is opaque here, so the fused path rejects it
        // instead of reading bytes it cannot interpret.
        DType::External(_) => Err(crate::Error::unsupported_dtype(
            op,
            input.dtype(),
            "an externally defined payload is not a fused input",
        )),
    }
}
/// The typed tensor behind `input`, or the refusal this path produces for one.
///
/// Callers reach this from a match on `input.dtype()`, so `None` means the tag table
/// and the runtime dtype disagree rather than a caller mistake.
fn fused_host<'a, T: tenferro_tensor::TensorScalar>(
    op: &'static str,
    input: &'a Tensor,
) -> crate::Result<&'a TypedTensor<T>> {
    input.as_typed::<T>().ok_or_else(|| {
        crate::Error::unsupported_dtype(
            op,
            input.dtype(),
            "an externally defined payload is not a fused input",
        )
    })
}

fn erased_fusion_input<'a>(
    input: &'a Tensor,
    view: &ElementwiseFusionInputView,
) -> crate::Result<ErasedFusionInput<'a>> {
    let data = tensor_host_bytes(ELEMENTWISE_FUSION_OP, input)?;
    let base_shape = input.shape();
    let base_strides = col_major_strides(base_shape)?;
    let ElementwiseFusionInputView::BroadcastInDim { shape, dims } = view else {
        return Ok(ErasedFusionInput {
            data,
            dims: ShapeVec::from_slice(base_shape),
            strides: base_strides,
        });
    };

    if dims.len() != base_shape.len() {
        return Err(crate::Error::invalid_argument(
            ELEMENTWISE_FUSION_OP,
            "configuration",
            format!(
                "broadcast dims length {} does not match input rank {}",
                dims.len(),
                base_shape.len()
            ),
        ));
    }

    let mut strides = StrideVec::from_elem(0, shape.len());
    let mut seen = vec![false; shape.len()];
    for (source_axis, &target_axis) in dims.iter().enumerate() {
        if target_axis >= shape.len() {
            return Err(crate::Error::axis_out_of_bounds(
                ELEMENTWISE_FUSION_OP,
                target_axis,
                shape.len(),
            ));
        }
        if seen[target_axis] {
            return Err(crate::Error::duplicate_axis(
                ELEMENTWISE_FUSION_OP,
                target_axis,
                "broadcast dims",
            ));
        }
        seen[target_axis] = true;
        let source_dim = base_shape[source_axis];
        let target_dim = shape[target_axis];
        if source_dim != target_dim && source_dim != 1 {
            return Err(crate::Error::shape_mismatch(
                ELEMENTWISE_FUSION_OP,
                shape.to_vec(),
                base_shape.to_vec(),
            ));
        }
        if source_dim == target_dim {
            strides[target_axis] = base_strides[source_axis];
        }
    }

    Ok(ErasedFusionInput {
        data,
        dims: ShapeVec::from_slice(shape),
        strides,
    })
}

#[doc(hidden)]
pub fn elementwise_fusion_with_pool(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    inputs: &[&Tensor],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<Option<Vec<Tensor>>> {
    if !validate_elementwise_fusion_inputs(inputs, plan)? {
        return Ok(None);
    }
    if inputs.is_empty() || inputs.len() > ERASED_FUSION_MAX_INPUTS {
        return Ok(None);
    }
    if plan_uses_unfused_op(plan) {
        return Ok(None);
    }
    if should_defer_to_broadcast_multiply_special_case(plan) {
        return Ok(None);
    }
    if !dtype_supports_erased_fusion(plan.dtype(), plan) {
        return Ok(None);
    }

    let input_layouts = inputs
        .iter()
        .zip(plan.input_views())
        .map(|(input, view)| erased_fusion_input(input, view))
        .collect::<crate::Result<Vec<_>>>()?;
    let shape = input_layouts[0].dims.clone();
    if input_layouts
        .iter()
        .skip(1)
        .any(|input| input.dims != shape)
    {
        return Ok(None);
    }
    let element_count =
        tenferro_tensor::validate::checked_shape_product(ELEMENTWISE_FUSION_OP, "shape", &shape)?;
    if element_count < ELEMENTWISE_FUSION_MIN_ELEMENTS {
        return Ok(None);
    }

    let dtype = kernel_dtype(plan.dtype());
    let input_refs = input_layouts
        .iter()
        .map(|input| {
            // SAFETY: fusion inputs are initialized typed storage with matching
            // dtype and alignment; validated layouts bound every reachable read
            // for the retained input borrow.
            unsafe { erased_raw_strided_ref(dtype, input.data, &input.dims, &input.strides, 0) }
                .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))
        })
        .collect::<crate::Result<Vec<_>>>()?;

    execute_erased_fused_outputs(buffers, exec_context, dtype, &input_refs, &shape, plan).map(Some)
}

/// Whether the CPU fused replay can be selected for this dtype and plan.
///
/// Only `F32` and `F64` are selected. `strided-fused 0.5.2`'s vectorized chunk
/// replay covers only those dtypes; every other preset dtype would run the
/// per-element interpreter, which is unmeasured and slower than the
/// per-instruction path, so it stays unfused.
///
/// An ordered op (`Maximum`/`Minimum`/`Clamp`) also declines for a real dtype:
/// the upstream kernels use Rust `.max`/`.min`, which do not propagate NaN,
/// while tenferro's ordinary kernels do (`R-numerics-01`).
fn dtype_supports_erased_fusion(dtype: DType, plan: &ElementwiseFusionPlan) -> bool {
    match dtype {
        DType::F32 | DType::F64 => !plan_uses_ordered_op(plan),
        _ => false,
    }
}

fn execute_erased_fused_outputs(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    dtype: KernelDType,
    input_refs: &[ErasedRawStridedRef<'_>],
    shape: &[usize],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<Vec<Tensor>> {
    let input_ptrs: Vec<_> = input_refs
        .iter()
        .map(ErasedRawStridedPtr::from_ref)
        .collect();
    match dtype {
        KernelDType::F32 => {
            execute_erased_fused_plan::<f32>(buffers, exec_context, dtype, &input_ptrs, shape, plan)
        }
        KernelDType::F64 => {
            execute_erased_fused_plan::<f64>(buffers, exec_context, dtype, &input_ptrs, shape, plan)
        }
        _ => Err(crate::Error::unsupported(
            ELEMENTWISE_FUSION_OP,
            format!(
                "unsupported dtype {}; the CPU fused replay covers F32/F64",
                dtype.label()
            ),
        )),
    }
}

/// Compile one strided plan for every live output and replay it once.
///
/// The plan's `outputs` are executed in order by
/// [`ErasedFusedPlan::execute_outputs_uninit`], so a region with several live
/// outputs replays its instruction list a single time instead of once per
/// output.
fn execute_erased_fused_plan<T>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    dtype: KernelDType,
    input_ptrs: &[ErasedRawStridedPtr<'_>],
    shape: &[usize],
    plan: &ElementwiseFusionPlan,
) -> crate::Result<Vec<Tensor>>
where
    T: Clone + PoolScalar,
{
    let erased_plan = ErasedFusedPlan::compile(dtype, strided_fused_plan(plan))
        .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    let output_strides = col_major_strides(shape)?;
    let mut outputs = plan
        .outputs()
        .iter()
        .map(|_| PooledUninitOutput::<T>::new(buffers, shape))
        .collect::<crate::Result<Vec<_>>>()?;
    // SAFETY: every pooled output exclusively owns its allocation with matching
    // dtype/alignment, the destination descriptors borrow each output mutably
    // for the exclusive lifetime of `dests`, and the fused plan overwrites
    // every reachable element before `assume_init_recycled` exposes typed
    // storage.
    let mut dests = outputs
        .iter_mut()
        .map(|output| {
            // SAFETY: as above.
            unsafe {
                erased_raw_strided_uninit_mut(
                    dtype,
                    output.as_uninit_bytes_mut(),
                    shape,
                    &output_strides,
                    0,
                )
            }
            .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))
        })
        .collect::<crate::Result<Vec<_>>>()?;
    erased_plan
        .execute_outputs_uninit(exec_context, &mut dests, input_ptrs)
        .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    drop(dests);
    outputs
        .drain(..)
        .map(|output| {
            // SAFETY: the fused replay wrote every logical destination element
            // and retains no destination view.
            let tensor = unsafe { output.assume_init_recycled()? };
            Ok(Tensor::from_typed::<T>(tensor))
        })
        .collect()
}

#[cfg(test)]
mod tests;
