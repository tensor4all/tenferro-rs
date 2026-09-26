use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, CubeElement, CubePrimitive};
use cubecl_wgpu::WgpuRuntime;
use tenferro_tensor::{Tensor, TensorRank, TensorRead, TensorView, TypedTensor, TypedTensorView};

use crate::native_permutation::{
    NativePermutationKind, NativePermutationPlan, NativeTransposeTile,
};

use super::{
    alloc_output, comptime_sequence, cube_count_for_len, cube_dim_1d,
    ensure_placement_resident_on_runtime, prepared_webgpu_view, unsupported_dtype, WebGpuBackend,
};

/// The Rust scalar type behind a preset variant name a macro received.
macro_rules! preset_scalar {
    (F32) => {
        f32
    };
    (F64) => {
        f64
    };
    (I32) => {
        i32
    };
    (I64) => {
        i64
    };
    (Bool) => {
        bool
    };
    (C32) => {
        num_complex::Complex32
    };
    (C64) => {
        num_complex::Complex64
    };
}
const MATERIALIZE_OP: &str = "WebGpuBackend::to_contiguous_read";

fn view_allocation_len<T, R>(
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<usize>
where
    T: crate::TensorScalar + 'static,
    R: TensorRank,
{
    if view.backend_family().is_some() {
        Ok(view.backing_len())
    } else {
        Err(crate::Error::runtime_state(
            op,
            "expected WebGPU view, got host view; upload before materializing",
        ))
    }
}

fn strides_i64(strides: &[isize], op: &'static str) -> crate::Result<Vec<i64>> {
    strides
        .iter()
        .map(|&stride| {
            i64::try_from(stride).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "strides",
                    format!("stride {stride} cannot be represented as i64"),
                )
            })
        })
        .collect()
}

fn launch_materialization<T>(
    backend: &WebGpuBackend,
    output: &TypedTensor<T>,
    input: ArrayArg<WgpuRuntime>,
    plan: &NativePermutationPlan,
    op: &'static str,
) -> crate::Result<()>
where
    T: CubeElement + CubePrimitive + crate::TensorScalar + Clone + Send + Sync + 'static,
{
    if plan.len == 0 {
        return Ok(());
    }
    let output_arg = view_array_arg(backend, &output.as_view(), op)?;
    if output_arg.size() != plan.len {
        return Err(crate::Error::runtime_state(
            op,
            format!(
                "native permutation output binding has {} elements, plan requires {}",
                output_arg.size(),
                plan.len
            ),
        ));
    }
    if input.size() < plan.len {
        return Err(crate::Error::runtime_state(
            op,
            format!(
                "native permutation input binding has {} elements, plan requires at least {}",
                input.size(),
                plan.len
            ),
        ));
    }
    if plan.kind == NativePermutationKind::TiledTranspose {
        if let Some(config) = NativeTransposeTile::selected(op)? {
            let block_rows = config.block_rows as usize;
            let padding = config.padding as usize;
            let vector_width = config.vector_width as usize;
            let src_offset = usize::try_from(plan.src_offset).map_err(|_| {
                crate::Error::invalid_argument(
                    op,
                    "offset",
                    "tiled transpose requires a non-negative source offset",
                )
            })?;
            if let Some((cubes_x, cubes_y, cubes_z)) = config.dispatch_grid(
                op,
                plan.dims[0],
                plan.dims[1],
                plan.dims.get(2).copied().unwrap_or(1),
                65_535,
            )? {
                let batch_stride = plan.tiled_matrix_len(op)?;
                let cube_dim =
                    CubeDim::new_2d(config.tile / config.vector_width, config.block_rows);
                unsafe {
                    // SAFETY: The tiled classification proves a compact 2D
                    // transpose. Bounds guards cover edge tiles and every unit
                    // reaches the shared-memory barrier.
                    crate::kernels::structural::tiled_transpose_kernel::launch_unchecked::<
                        T,
                        WgpuRuntime,
                    >(
                        backend.runtime().client(),
                        CubeCount::Static(cubes_x, cubes_y, cubes_z),
                        cube_dim,
                        output_arg,
                        input,
                        src_offset,
                        batch_stride,
                        plan.dims[0],
                        plan.dims[1],
                        config.tile as usize,
                        block_rows,
                        padding,
                        vector_width,
                    );
                }
                return Ok(());
            }
        }
    }
    let src_strides = strides_i64(&plan.src_strides, op)?;
    let src_offset = i64::try_from(plan.src_offset).map_err(|_| {
        crate::Error::invalid_argument(
            op,
            "offset",
            format!(
                "source offset {} cannot be represented as i64",
                plan.src_offset
            ),
        )
    })?;
    unsafe {
        // SAFETY: `NativePermutationPlan` validated source/destination bounds,
        // destination non-overlap, shape products, and disjoint allocations.
        crate::kernels::structural::materialize_strided_kernel::launch_unchecked::<T, WgpuRuntime>(
            backend.runtime().client(),
            cube_count_for_len(plan.len)?,
            cube_dim_1d(),
            output_arg,
            input,
            comptime_sequence(&plan.dims),
            comptime_sequence(&src_strides),
            src_offset,
            plan.len,
            plan.dims.len(),
        );
    }
    Ok(())
}

fn view_array_arg<T, R>(
    backend: &WebGpuBackend,
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<ArrayArg<WgpuRuntime>>
where
    T: CubeElement + crate::TensorScalar + Clone + Send + Sync + 'static,
    R: TensorRank,
{
    ensure_placement_resident_on_runtime(backend.runtime(), view.placement(), op)?;
    let expected_allocation_domain = backend.runtime().allocation_domain_id();
    let Some(actual_allocation_domain) = view.allocation_domain() else {
        return Err(crate::Error::runtime_state(
            op,
            "expected WebGPU view, got host view; upload before materializing",
        ));
    };
    if actual_allocation_domain != expected_allocation_domain {
        return Err(crate::Error::host_access(
            op,
            crate::HostAccessError::ForeignDomain {
                expected: expected_allocation_domain,
                actual: actual_allocation_domain,
            },
        ));
    }
    if !matches!(view.backend_family(), Some("webgpu" | "cubecl-webgpu")) {
        return Err(crate::Error::runtime_state(
            op,
            "expected a WebGPU view from the selected provider",
        ));
    }
    let prepared = prepared_webgpu_view(view, op)?;

    // SAFETY: the shared storage root prepared this exact checked view before
    // the binding is constructed. The kernel receives the same validated
    // layout metadata and indexes only logical output elements.
    Ok(unsafe {
        ArrayArg::from_raw_parts(
            prepared.handle,
            prepared.byte_len / core::mem::size_of::<T>(),
        )
    })
}

fn materialize_typed<T, R>(
    backend: &WebGpuBackend,
    view: &TypedTensorView<'_, T, R>,
) -> crate::Result<TypedTensor<T>>
where
    T: CubeElement + CubePrimitive + crate::TensorScalar + Clone + Send + Sync + 'static,
    R: TensorRank,
{
    let len = view
        .shape()
        .iter()
        .try_fold(1usize, |len, &dim| len.checked_mul(dim))
        .ok_or_else(|| {
            crate::Error::invalid_argument(
                MATERIALIZE_OP,
                "shape",
                format!("shape product overflow for {:?}", view.shape()),
            )
        })?;
    let plan = NativePermutationPlan::for_contiguous_output(
        MATERIALIZE_OP,
        view.shape(),
        view.strides(),
        view.offset(),
        view_allocation_len(view, MATERIALIZE_OP)?,
        len,
        false,
    )?;
    let output = alloc_output::<T>(backend.runtime(), view.shape(), MATERIALIZE_OP)?;
    let input_arg = view_array_arg(backend, view, MATERIALIZE_OP)?;
    launch_materialization(backend, &output, input_arg, &plan, MATERIALIZE_OP)?;
    Ok(output)
}

pub(super) fn to_contiguous_f32(
    backend: &WebGpuBackend,
    view: &TypedTensorView<'_, f32>,
) -> crate::Result<TypedTensor<f32>> {
    materialize_typed(backend, view)
}

/// The typed tensor behind a transpose operand, or this module's refusal for one.
pub(super) fn to_contiguous_read(
    backend: &WebGpuBackend,
    input: TensorRead<'_>,
) -> crate::Result<Tensor> {
    macro_rules! materialize {
        ($variant:ident, $view:expr) => {
            materialize_typed(backend, &$view).map(Tensor::from_typed::<preset_scalar!($variant)>)
        };
    }

    match input {
        TensorRead::Tensor(tensor) => match tensor.dtype() {
            crate::DType::F32 => materialize!(
                F32,
                webgpu_read_operand::<f32>(tensor, MATERIALIZE_OP)?.as_view()
            ),
            crate::DType::I32 => materialize!(
                I32,
                webgpu_read_operand::<i32>(tensor, MATERIALIZE_OP)?.as_view()
            ),
            // Reject unsupported WGSL element types before asynchronous codegen.
            other => Err(unsupported_dtype(MATERIALIZE_OP, other)),
        },
        TensorRead::View(TensorView::F32(input)) => materialize!(F32, input),
        TensorRead::View(TensorView::I32(input)) => materialize!(I32, input),
        // Reject unsupported WGSL element types before asynchronous codegen.
        other => Err(unsupported_dtype(MATERIALIZE_OP, other.dtype())),
    }
}

/// The typed tensor behind a read adapter's tensor, or the refusal this op reports for one.
fn webgpu_read_operand<'a, T: crate::TensorScalar>(
    tensor: &'a Tensor,
    op: &'static str,
) -> crate::Result<&'a TypedTensor<T>> {
    tensor
        .as_typed::<T>()
        .ok_or_else(|| unsupported_dtype(op, tensor.dtype()))
}
