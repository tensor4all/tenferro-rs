#![doc(hidden)]

//! CPU-owned one-shot elementwise read-into replay.

use smallvec::SmallVec;
use strided_kernel::{
    erased_map_into, erased_zip_into, ErasedMapOp, ErasedZipOp, ExecContext, KernelDType,
};
use tenferro_tensor::backend::{validate_read_into_destination, ElementwiseReadOp};
use tenferro_tensor::{
    DType, Tensor, TensorRead, TensorView, TensorViewMut, TensorWrite, TypedTensorView,
    TypedTensorViewMut,
};

use crate::Error;
use tenferro_cpu_basic::{erased_raw_strided_mut, erased_raw_strided_ptr};

fn validate_elementwise_output_disjoint(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: &TensorWrite<'_>,
) -> crate::Result<()> {
    validate_read_into_destination(op.label(), inputs, out)
}

fn read_is_host(input: &TensorRead<'_>) -> bool {
    match input {
        TensorRead::Tensor(tensor) => !tensor.is_backend_buffer(),
        TensorRead::View(view) => match view {
            TensorView::F32(view) => view.backend_buffer().is_none(),
            TensorView::F64(view) => view.backend_buffer().is_none(),
            TensorView::I32(view) => view.backend_buffer().is_none(),
            TensorView::I64(view) => view.backend_buffer().is_none(),
            TensorView::Bool(view) => view.backend_buffer().is_none(),
            TensorView::C32(view) => view.backend_buffer().is_none(),
            TensorView::C64(view) => view.backend_buffer().is_none(),
        },
    }
}

fn write_is_host(out: &TensorWrite<'_>) -> bool {
    read_is_host(&out.as_read())
}

fn one_shot_supports(op: ElementwiseReadOp, dtype: DType) -> bool {
    match op {
        ElementwiseReadOp::Conj => true,
        ElementwiseReadOp::Add
        | ElementwiseReadOp::Subtract
        | ElementwiseReadOp::Multiply
        | ElementwiseReadOp::Divide
        | ElementwiseReadOp::Negate => !matches!(dtype, DType::Bool),
        _ => false,
    }
}

fn one_shot_eligible(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: &TensorWrite<'_>,
) -> bool {
    let dtype = out.dtype();
    write_is_host(out)
        && one_shot_supports(op, dtype)
        && inputs.iter().all(|input| {
            read_is_host(input) && input.dtype() == dtype && input.shape() == out.shape()
        })
}

fn tensor_write_view(out: TensorWrite<'_>) -> TensorViewMut<'_> {
    match out {
        TensorWrite::Tensor(tensor) => match tensor {
            Tensor::F32(tensor) => TensorViewMut::F32(tensor.as_view_mut()),
            Tensor::F64(tensor) => TensorViewMut::F64(tensor.as_view_mut()),
            Tensor::I32(tensor) => TensorViewMut::I32(tensor.as_view_mut()),
            Tensor::I64(tensor) => TensorViewMut::I64(tensor.as_view_mut()),
            Tensor::Bool(tensor) => TensorViewMut::Bool(tensor.as_view_mut()),
            Tensor::C32(tensor) => TensorViewMut::C32(tensor.as_view_mut()),
            Tensor::C64(tensor) => TensorViewMut::C64(tensor.as_view_mut()),
        },
        TensorWrite::View(view) => view,
    }
}

fn typed_bytes<T>(data: &[T]) -> &[u8] {
    // SAFETY: u8 has alignment one and the returned bytes retain the shared
    // lifetime of the typed source slice.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn typed_bytes_mut<T>(data: &mut [T]) -> &mut [u8] {
    let len = std::mem::size_of_val(data);
    // SAFETY: u8 has alignment one and the returned bytes retain the unique
    // lifetime of the typed destination slice.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), len) }
}

fn execute_one_shot_map<T: 'static>(
    dtype: KernelDType,
    op: ErasedMapOp,
    ctx: &ExecContext,
    input: TypedTensorView<'_, T>,
    mut out: TypedTensorViewMut<'_, T>,
) -> crate::Result<()> {
    let input_data = input.host_storage()?;
    // INVARIANT: dtype and layout come from the same validated typed view, and
    // its host storage remains borrowed until replay returns.
    // SAFETY: input_data supplies the pointer and exact byte length; the view
    // owns the matching shape, signed strides, and in-bounds offset.
    let input_descriptor = unsafe {
        erased_raw_strided_ptr(
            dtype,
            typed_bytes(input_data),
            input.shape(),
            input.strides(),
            input.offset(),
        )
    }
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;

    let out_dims = SmallVec::<[usize; 8]>::from_slice(out.shape());
    let out_strides = SmallVec::<[isize; 8]>::from_slice(out.strides());
    let out_offset = out.offset();
    let out_data = out.host_storage_mut()?;
    // INVARIANT: the copied output layout describes this uniquely borrowed
    // host storage, already validated as disjoint from every input.
    // SAFETY: the host-storage borrow and validated layout satisfy the adapter contract.
    let mut out_descriptor = unsafe {
        erased_raw_strided_mut(
            dtype,
            typed_bytes_mut(out_data),
            &out_dims,
            &out_strides,
            out_offset,
        )
    }
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;
    erased_map_into(dtype, op, ctx, &mut out_descriptor, &input_descriptor)
        .map_err(|error| Error::backend_source("elementwise_read_into", error))
}

fn execute_one_shot_zip<T: 'static>(
    dtype: KernelDType,
    op: ErasedZipOp,
    ctx: &ExecContext,
    lhs: TypedTensorView<'_, T>,
    rhs: TypedTensorView<'_, T>,
    mut out: TypedTensorViewMut<'_, T>,
) -> crate::Result<()> {
    let lhs_data = lhs.host_storage()?;
    // INVARIANT: dtype and layout come from the same validated typed view, and
    // its host storage remains borrowed until replay returns.
    // SAFETY: lhs_data supplies the pointer and exact byte length; the view
    // owns the matching shape, signed strides, and in-bounds offset.
    let lhs_descriptor = unsafe {
        erased_raw_strided_ptr(
            dtype,
            typed_bytes(lhs_data),
            lhs.shape(),
            lhs.strides(),
            lhs.offset(),
        )
    }
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;
    let rhs_data = rhs.host_storage()?;
    // INVARIANT: dtype and layout come from the same validated typed view, and
    // its host storage remains borrowed until replay returns.
    // SAFETY: rhs_data supplies the pointer and exact byte length; the view
    // owns the matching shape, signed strides, and in-bounds offset.
    let rhs_descriptor = unsafe {
        erased_raw_strided_ptr(
            dtype,
            typed_bytes(rhs_data),
            rhs.shape(),
            rhs.strides(),
            rhs.offset(),
        )
    }
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;

    let out_dims = SmallVec::<[usize; 8]>::from_slice(out.shape());
    let out_strides = SmallVec::<[isize; 8]>::from_slice(out.strides());
    let out_offset = out.offset();
    let out_data = out.host_storage_mut()?;
    // INVARIANT: the copied output layout describes this uniquely borrowed
    // host storage, already validated as disjoint from every input.
    // SAFETY: the host-storage borrow and validated layout satisfy the adapter contract.
    let mut out_descriptor = unsafe {
        erased_raw_strided_mut(
            dtype,
            typed_bytes_mut(out_data),
            &out_dims,
            &out_strides,
            out_offset,
        )
    }
    .map_err(|error| Error::backend_source("elementwise_read_into", error))?;
    erased_zip_into(
        dtype,
        op,
        ctx,
        &mut out_descriptor,
        &lhs_descriptor,
        &rhs_descriptor,
    )
    .map_err(|error| Error::backend_source("elementwise_read_into", error))
}

fn execute_one_shot_elementwise(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
    ctx: &ExecContext,
) -> crate::Result<()> {
    let out = tensor_write_view(out);
    macro_rules! dispatch_map {
        ($map_op:expr) => {{
            let input = inputs[0].clone().tensor_view();
            match (input, out) {
                (TensorView::F32(input), TensorViewMut::F32(out)) => {
                    execute_one_shot_map(KernelDType::F32, $map_op, ctx, input, out)
                }
                (TensorView::F64(input), TensorViewMut::F64(out)) => {
                    execute_one_shot_map(KernelDType::F64, $map_op, ctx, input, out)
                }
                (TensorView::I32(input), TensorViewMut::I32(out)) => {
                    execute_one_shot_map(KernelDType::I32, $map_op, ctx, input, out)
                }
                (TensorView::I64(input), TensorViewMut::I64(out)) => {
                    execute_one_shot_map(KernelDType::I64, $map_op, ctx, input, out)
                }
                (TensorView::Bool(input), TensorViewMut::Bool(out)) => {
                    execute_one_shot_map(KernelDType::Bool, $map_op, ctx, input, out)
                }
                (TensorView::C32(input), TensorViewMut::C32(out)) => {
                    execute_one_shot_map(KernelDType::C32, $map_op, ctx, input, out)
                }
                (TensorView::C64(input), TensorViewMut::C64(out)) => {
                    execute_one_shot_map(KernelDType::C64, $map_op, ctx, input, out)
                }
                _ => unreachable!("one-shot eligibility validates matching dtypes"),
            }
        }};
    }
    macro_rules! dispatch_zip {
        ($zip_op:expr) => {{
            let lhs = inputs[0].clone().tensor_view();
            let rhs = inputs[1].clone().tensor_view();
            match (lhs, rhs, out) {
                (TensorView::F32(lhs), TensorView::F32(rhs), TensorViewMut::F32(out)) => {
                    execute_one_shot_zip(KernelDType::F32, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::F64(lhs), TensorView::F64(rhs), TensorViewMut::F64(out)) => {
                    execute_one_shot_zip(KernelDType::F64, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::I32(lhs), TensorView::I32(rhs), TensorViewMut::I32(out)) => {
                    execute_one_shot_zip(KernelDType::I32, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::I64(lhs), TensorView::I64(rhs), TensorViewMut::I64(out)) => {
                    execute_one_shot_zip(KernelDType::I64, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::C32(lhs), TensorView::C32(rhs), TensorViewMut::C32(out)) => {
                    execute_one_shot_zip(KernelDType::C32, $zip_op, ctx, lhs, rhs, out)
                }
                (TensorView::C64(lhs), TensorView::C64(rhs), TensorViewMut::C64(out)) => {
                    execute_one_shot_zip(KernelDType::C64, $zip_op, ctx, lhs, rhs, out)
                }
                _ => unreachable!("one-shot eligibility validates matching dtypes"),
            }
        }};
    }

    match op {
        ElementwiseReadOp::Add => dispatch_zip!(ErasedZipOp::Add),
        ElementwiseReadOp::Subtract => dispatch_zip!(ErasedZipOp::Subtract),
        ElementwiseReadOp::Multiply => dispatch_zip!(ErasedZipOp::Multiply),
        ElementwiseReadOp::Divide => dispatch_zip!(ErasedZipOp::Divide),
        ElementwiseReadOp::Negate => dispatch_map!(ErasedMapOp::Negate),
        ElementwiseReadOp::Conj => dispatch_map!(ErasedMapOp::Conj),
        _ => Err(Error::unsupported(
            "elementwise_read_into",
            "unsupported operation",
        )),
    }
}

/// Execute the shared elementwise-into path with an explicit replay context.
///
/// This is backend glue for implementations that own an execution context.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] when the input arity or tensor
/// metadata is invalid, or when the destination overlaps an input. Returns
/// [`crate::Error::BackendSource`] when an eligible strided replay fails.
/// Errors returned by `fallback` are preserved unchanged.
#[doc(hidden)]
pub fn elementwise_read_into_with_context(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
    ctx: &ExecContext,
    fallback: impl FnOnce(&[TensorRead<'_>], TensorWrite<'_>) -> crate::Result<()>,
) -> crate::Result<()> {
    if inputs.len() != op.arity() {
        return Err(Error::invalid_argument(
            op.label(),
            "inputs",
            format!("expected {} inputs, got {}", op.arity(), inputs.len()),
        ));
    }
    validate_elementwise_output_disjoint(op, inputs, &out)?;
    if one_shot_eligible(op, inputs, &out) {
        execute_one_shot_elementwise(op, inputs, out, ctx)
    } else {
        fallback(inputs, out)
    }
}

#[cfg(test)]
#[path = "read_into/tests.rs"]
mod tests;
