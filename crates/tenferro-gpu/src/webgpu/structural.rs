use cubecl::prelude::{ArrayArg, CubeCount, CubeDim, CubeElement, CubePrimitive};
use cubecl_wgpu::WgpuRuntime;
use tenferro_tensor::validate::validate_permutation_axes;
use tenferro_tensor::{Tensor, TensorRank, TensorRead, TensorView, TypedTensor, TypedTensorView};

use crate::native_permutation::{
    NativePermutationKind, NativePermutationPlan, NativeTransposeTile,
};

use super::{
    alloc_output, comptime_sequence, cube_count_for_len, cube_dim_1d,
    ensure_placement_resident_on_runtime, unsupported_dtype, WebGpuBackend, WebGpuBuffer,
};

const TRANSPOSE_OP: &str = "webgpu_transpose";
const MATERIALIZE_OP: &str = "WebGpuBackend::to_contiguous_read";

fn dense_strides(shape: &[usize], op: &'static str) -> crate::Result<Vec<isize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &dim in shape {
        strides.push(stride);
        let dim = isize::try_from(dim).map_err(|_| {
            crate::Error::invalid_argument(
                op,
                "shape",
                format!("dimension {dim} cannot be represented as isize"),
            )
        })?;
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

fn view_allocation_len<T, R>(
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<usize>
where
    T: 'static,
    R: TensorRank,
{
    view.backend_buffer()
        .map(|buffer| buffer.len())
        .ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                "expected WebGPU view, got host view; upload before materializing",
            )
        })
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
            if let Some((cubes_x, cubes_y, cubes_z)) =
                config.dispatch_grid(op, &plan.dims, 65_535)?
            {
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
        .downcast_ref::<WebGpuBuffer>()
        .ok_or_else(|| {
            crate::Error::runtime_state(
                op,
                format!(
                    "expected WebGPU backend buffer, got `{}` backend buffer",
                    buffer.backend_family()
                ),
            )
        })?;
    let expected_allocation_domain = backend.runtime().allocation_domain_id();
    if buffer.allocation_domain() != expected_allocation_domain {
        return Err(crate::Error::host_access(
            op,
            crate::HostAccessError::ForeignDomain {
                expected: expected_allocation_domain,
                actual: buffer.allocation_domain(),
            },
        ));
    }
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
    T: CubeElement + CubePrimitive + crate::TensorScalar + Clone + Send + Sync + 'static,
{
    validate_permutation_axes(TRANSPOSE_OP, input.shape().len(), perm)?;
    let output_shape: Vec<usize> = perm.iter().map(|&axis| input.shape()[axis]).collect();
    let input_strides = dense_strides(input.shape(), TRANSPOSE_OP)?;
    let plan = NativePermutationPlan::for_transpose(
        TRANSPOSE_OP,
        input.shape(),
        &input_strides,
        perm,
        0,
        input.n_elements(),
        input.n_elements(),
        false,
    )?;
    let output = alloc_output::<T>(backend.runtime(), &output_shape, TRANSPOSE_OP)?;
    let input_arg = view_array_arg(backend, &input.as_view(), TRANSPOSE_OP)?;
    launch_materialization(backend, &output, input_arg, &plan, TRANSPOSE_OP)?;
    Ok(output)
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
