#![doc(hidden)]

//! Fused CPU elementwise execution.

pub type Result<T> = tenferro_tensor::Result<T>;
pub use tenferro_tensor::{DType, Error, Tensor, TypedTensor};

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

const ELEMENTWISE_FUSION_OP: &str = "execute_elementwise_fusion";

const ELEMENTWISE_FUSION_MIN_ELEMENTS: usize = 16 * 1024;

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
        ElementwiseFusionOp::Remainder => {
            unreachable!("remainder must be filtered before CPU elementwise fusion")
        }
    }
}

fn plan_uses_unfused_op(plan: &ElementwiseFusionPlan) -> bool {
    plan.ops()
        .iter()
        .any(|inst| inst.op() == ElementwiseFusionOp::Remainder)
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

fn single_output_strided_fused_plan(plan: &ElementwiseFusionPlan, output: usize) -> FusedPlan {
    FusedPlan {
        input_count: plan.input_count(),
        outputs: vec![output],
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
    }
}

fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: `data` is an aligned typed slice. The returned byte slice has
    // the same lifetime and exact byte length, and is read-only.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) }
}

struct ErasedFusionInput<'a> {
    data: &'a [u8],
    dims: Vec<usize>,
    strides: Vec<isize>,
}

fn tensor_host_bytes<'a>(op: &'static str, input: &'a Tensor) -> crate::Result<&'a [u8]> {
    macro_rules! bytes {
        ($tensor:expr) => {
            typed_host_data(op, $tensor).map(typed_bytes)
        };
    }

    match input {
        Tensor::F32(tensor) => bytes!(tensor),
        Tensor::F64(tensor) => bytes!(tensor),
        Tensor::I32(tensor) => bytes!(tensor),
        Tensor::I64(tensor) => bytes!(tensor),
        Tensor::Bool(tensor) => bytes!(tensor),
        Tensor::C32(tensor) => bytes!(tensor),
        Tensor::C64(tensor) => bytes!(tensor),
    }
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
            dims: base_shape.to_vec(),
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

    let mut strides = vec![0; shape.len()];
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
        dims: shape.to_vec(),
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
    if inputs.is_empty() {
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

fn dtype_supports_erased_fusion(dtype: DType, plan: &ElementwiseFusionPlan) -> bool {
    match dtype {
        DType::F32 | DType::F64 => true,
        DType::C32 | DType::C64 => !plan_uses_ordered_op(plan),
        DType::I32 | DType::I64 => plan.ops().iter().all(|inst| {
            matches!(
                inst.op(),
                ElementwiseFusionOp::Add
                    | ElementwiseFusionOp::Multiply
                    | ElementwiseFusionOp::Negate
                    | ElementwiseFusionOp::Conj
                    | ElementwiseFusionOp::Abs
                    | ElementwiseFusionOp::Maximum
                    | ElementwiseFusionOp::Minimum
                    | ElementwiseFusionOp::Clamp
            )
        }),
        DType::Bool => plan
            .ops()
            .iter()
            .all(|inst| inst.op() == ElementwiseFusionOp::Conj),
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
        KernelDType::F32 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<f32>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::F32,
                )
            })
            .collect(),
        KernelDType::F64 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<f64>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::F64,
                )
            })
            .collect(),
        KernelDType::I32 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<i32>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::I32,
                )
            })
            .collect(),
        KernelDType::I64 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<i64>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::I64,
                )
            })
            .collect(),
        KernelDType::Bool => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<bool>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::Bool,
                )
            })
            .collect(),
        KernelDType::C32 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<num_complex::Complex32>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::C32,
                )
            })
            .collect(),
        KernelDType::C64 => plan
            .outputs()
            .iter()
            .map(|&output| {
                execute_erased_fused_output::<num_complex::Complex64>(
                    buffers,
                    exec_context,
                    dtype,
                    &input_ptrs,
                    shape,
                    plan,
                    output,
                    Tensor::C64,
                )
            })
            .collect(),
        _ => Err(crate::Error::unsupported(
            ELEMENTWISE_FUSION_OP,
            format!(
                "unsupported dtype {}; supported dtypes: F32/F64/I32/I64/Bool/C32/C64",
                dtype.label()
            ),
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_erased_fused_output<T>(
    buffers: &mut BufferPool,
    exec_context: &ExecContext,
    dtype: KernelDType,
    input_ptrs: &[ErasedRawStridedPtr<'_>],
    shape: &[usize],
    plan: &ElementwiseFusionPlan,
    output: usize,
    wrap: fn(TypedTensor<T>) -> Tensor,
) -> crate::Result<Tensor>
where
    T: Clone + PoolScalar,
{
    let fused_plan = single_output_strided_fused_plan(plan, output);
    let erased_plan = ErasedFusedPlan::compile(dtype, fused_plan)
        .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    let mut out = PooledUninitOutput::<T>::new(buffers, shape.to_vec())?;
    let output_strides = col_major_strides(shape)?;
    // SAFETY: `out` exclusively owns the output allocation with matching
    // dtype/alignment and the fused plan overwrites every reachable element
    // before `assume_init` exposes typed storage.
    let mut dest = unsafe {
        erased_raw_strided_uninit_mut(dtype, out.as_uninit_bytes_mut(), shape, &output_strides, 0)
    }
    .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    erased_plan
        .execute_uninit(exec_context, &mut dest, input_ptrs)
        .map_err(|err| crate::Error::backend_source(ELEMENTWISE_FUSION_OP, err))?;
    // SAFETY: the fused replay writes every logical destination element and retains no destination view.
    Ok(wrap(unsafe { out.assume_init()? }))
}

#[cfg(test)]
mod tests;
