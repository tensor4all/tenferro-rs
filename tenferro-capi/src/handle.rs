use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext, SemiringCoreDescriptor, TensorSemiringCore};
use tenferro_tensor::{MemoryOrder, Tensor};

/// Opaque handle wrapping a `Tensor<f64>`.
///
/// Host languages hold a pointer to this type and pass it to all
/// `tfe_*` functions. The internal layout is private; only the C-API
/// functions can access the inner tensor.
///
/// # Examples (C)
///
/// ```c
/// tfe_status_t status;
/// size_t shape[] = {2, 3};
/// double data[] = {1, 2, 3, 4, 5, 6};
/// tfe_tensor_f64 *t = tfe_tensor_f64_from_data(data, 6, shape, 2, &status);
/// tfe_tensor_f64_release(t);
/// ```
#[repr(C)]
pub struct TfeTensorF64 {
    _private: [u8; 0],
}

/// Convert a `Tensor<f64>` into an opaque handle.
pub(crate) fn tensor_to_handle(tensor: Tensor<f64>) -> *mut TfeTensorF64 {
    Box::into_raw(Box::new(tensor)) as *mut TfeTensorF64
}

/// Ensure a tensor has column-major contiguous data layout.
pub(crate) fn ensure_col_major(
    ctx: &mut CpuContext,
    tensor: Tensor<f64>,
) -> std::result::Result<Tensor<f64>, tenferro_device::Error> {
    if tensor.is_col_major_contiguous() {
        return Ok(tensor);
    }
    let mut result = Tensor::<f64>::zeros(
        tensor.dims(),
        tensor.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    )?;
    let desc = SemiringCoreDescriptor::MakeContiguous;
    let shapes = [tensor.dims(), result.dims()];
    let plan = <CpuBackend as TensorSemiringCore<Standard<f64>>>::plan(ctx, &desc, &shapes)?;
    <CpuBackend as TensorSemiringCore<Standard<f64>>>::execute(
        ctx,
        &plan,
        1.0,
        &[&tensor],
        0.0,
        &mut result,
    )?;
    Ok(result)
}

/// Borrow the tensor behind an opaque handle.
///
/// # Safety
///
/// `handle` must be a valid, non-null pointer returned by `tensor_to_handle`.
pub(crate) unsafe fn handle_to_ref<'a>(handle: *const TfeTensorF64) -> &'a Tensor<f64> {
    &*(handle as *const Tensor<f64>)
}

/// Take ownership of the tensor behind an opaque handle.
///
/// # Safety
///
/// `handle` must be a valid, non-null pointer returned by `tensor_to_handle`.
/// Must not be used after this call.
pub(crate) unsafe fn handle_take(handle: *mut TfeTensorF64) -> Box<Tensor<f64>> {
    Box::from_raw(handle as *mut Tensor<f64>)
}
