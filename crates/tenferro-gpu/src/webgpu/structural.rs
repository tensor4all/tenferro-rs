use cubecl::prelude::{ArrayArg, CubeElement, CubePrimitive};
use cubecl_wgpu::WgpuRuntime;
use tenferro_tensor::validate::validate_permutation_axes;
use tenferro_tensor::{Tensor, TensorRank, TensorRead, TensorView, TypedTensor, TypedTensorView};

use super::{
    alloc_output, comptime_sequence, cube_count_for_len, cube_dim_1d,
    ensure_placement_resident_on_runtime, ensure_resident_on_runtime,
    typed_tensor_binding_with_layout, unsupported_dtype, WebGpuBackend, WebGpuBuffer,
};

const TRANSPOSE_OP: &str = "webgpu_transpose";
const MATERIALIZE_OP: &str = "WebGpuBackend::to_contiguous_read";

fn dense_strides(shape: &[usize], op: &'static str) -> crate::Result<Vec<usize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dim in shape {
        strides.push(stride);
        stride = stride.checked_mul(dim).ok_or_else(|| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("column-major stride overflow for shape {shape:?}"),
            )
        })?;
    }
    Ok(strides)
}

fn view_array_arg<T, R>(
    backend: &WebGpuBackend,
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<ArrayArg<WgpuRuntime>>
where
    T: CubeElement + Clone + Send + Sync + 'static,
    R: TensorRank,
{
    ensure_placement_resident_on_runtime(backend.runtime(), view.placement(), op)?;
    let buffer = view.backend_buffer().ok_or_else(|| {
        crate::Error::runtime_state(
            op,
            "expected WebGPU view, got host view; upload before materializing",
        )
    })?;
    let buffer = buffer
        .as_any()
        .downcast_ref::<WebGpuBuffer<T>>()
        .ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                format!(
                    "expected WebGPU backend buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            )
        })?;
    if let Some(expected) = backend.runtime().allocation_domain() {
        match buffer.domain.as_ref().map(|domain| domain.id) {
            Some(actual) if actual == expected.id => {}
            Some(actual) => {
                return Err(crate::Error::host_access(
                    op,
                    crate::HostAccessError::ForeignDomain {
                        expected: expected.id,
                        actual,
                    },
                ));
            }
            None => {
                return Err(crate::Error::runtime_state(
                    op,
                    "Apple runtime requires a managed allocation from its domain",
                ));
            }
        }
    }

    // SAFETY: TypedTensorView construction proves every reachable signed
    // offset is inside the backing allocation. The kernel receives that same
    // validated layout metadata and indexes only logical output elements.
    Ok(unsafe { ArrayArg::from_raw_parts(buffer.handle().clone(), buffer.element_len()) })
}

fn transpose_typed<T>(
    backend: &WebGpuBackend,
    input: &TypedTensor<T>,
    perm: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + CubePrimitive + Clone + Send + Sync + 'static,
{
    validate_permutation_axes(TRANSPOSE_OP, input.shape().len(), perm)?;
    ensure_resident_on_runtime(backend.runtime(), input, TRANSPOSE_OP)?;
    let output_shape: Vec<usize> = perm.iter().map(|&axis| input.shape()[axis]).collect();
    let output = alloc_output::<T>(backend.runtime(), &output_shape, TRANSPOSE_OP)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let input_strides = dense_strides(input.shape(), TRANSPOSE_OP)?;
    let output_strides = dense_strides(&output_shape, TRANSPOSE_OP)?;
    let input_arg =
        typed_tensor_binding_with_layout(input, input.shape(), &input_strides, TRANSPOSE_OP)?;
    let output_arg =
        typed_tensor_binding_with_layout(&output, &output_shape, &output_strides, TRANSPOSE_OP)?;

    unsafe {
        // SAFETY: The permutation is validated and both bindings cover their
        // complete compact allocations. The unchanged kernel guards every
        // output position and maps it through the full permutation.
        crate::kernels::structural::transpose_kernel::launch_unchecked::<T, WgpuRuntime>(
            backend.runtime().client(),
            cube_count_for_len(output.n_elements())?,
            cube_dim_1d(),
            output_arg.into_tensor_arg(),
            input_arg.into_tensor_arg(),
            comptime_sequence(perm),
        );
    }
    Ok(output)
}

fn materialize_typed<T, R>(
    backend: &WebGpuBackend,
    view: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + CubePrimitive + Clone + Send + Sync + 'static,
    R: TensorRank,
{
    let output = alloc_output::<T>(backend.runtime(), view.shape(), MATERIALIZE_OP)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    let output_strides = dense_strides(view.shape(), MATERIALIZE_OP)?;
    let output_arg =
        typed_tensor_binding_with_layout(&output, view.shape(), &output_strides, MATERIALIZE_OP)?;
    let input_arg = view_array_arg(backend, view, MATERIALIZE_OP)?;
    let strides = view
        .strides()
        .iter()
        .copied()
        .map(|stride| {
            i64::try_from(stride).map_err(|_| {
                crate::Error::invalid_argument(
                    MATERIALIZE_OP,
                    "strides",
                    format!("view stride {stride} cannot be represented as i64"),
                )
            })
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let base_offset = i64::try_from(view.offset()).map_err(|_| {
        crate::Error::invalid_argument(
            MATERIALIZE_OP,
            "offset",
            format!("view offset {} cannot be represented as i64", view.offset()),
        )
    })?;

    unsafe {
        // SAFETY: The view constructor validated all signed offsets against
        // the resident backing allocation. The unchanged kernel writes every
        // compact output element once and reads only those validated offsets.
        crate::kernels::structural::view_to_contiguous_kernel::launch_unchecked::<T, WgpuRuntime>(
            backend.runtime().client(),
            cube_count_for_len(output.n_elements())?,
            cube_dim_1d(),
            output_arg.into_tensor_arg(),
            input_arg,
            comptime_sequence(&strides),
            base_offset,
            view.shape().len(),
        );
    }
    Ok(output)
}

pub(super) fn to_contiguous_f32(
    backend: &WebGpuBackend,
    view: &TypedTensorView<'_, f32>,
) -> crate::Result<TypedTensor<f32>> {
    materialize_typed(backend, view)
}

pub(super) fn transpose(
    backend: &WebGpuBackend,
    input: &Tensor,
    perm: &[usize],
) -> crate::Result<Tensor> {
    match input {
        Tensor::F32(input) => transpose_typed(backend, input, perm).map(Tensor::F32),
        Tensor::I32(input) => transpose_typed(backend, input, perm).map(Tensor::I32),
        // CubeK has a dedicated complex WebGPU representation, but CubeCL's
        // generic WGSL compiler cannot lower CubePrimitive Complex32.
        other => Err(unsupported_dtype(TRANSPOSE_OP, other.dtype())),
    }
}

pub(super) fn to_contiguous_read(
    backend: &WebGpuBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    macro_rules! materialize {
        ($variant:ident, $view:expr) => {
            materialize_typed(backend, &$view).map(Tensor::$variant)
        };
    }

    match input {
        TensorRead::Tensor(Tensor::F32(input)) => materialize!(F32, input.as_view()),
        TensorRead::Tensor(Tensor::I32(input)) => materialize!(I32, input.as_view()),
        TensorRead::View(TensorView::F32(input)) => materialize!(F32, input),
        TensorRead::View(TensorView::I32(input)) => materialize!(I32, input),
        // Reject unsupported WGSL element types before asynchronous codegen.
        other => Err(unsupported_dtype(MATERIALIZE_OP, other.dtype())),
    }
}
