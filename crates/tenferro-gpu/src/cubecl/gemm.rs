use std::ffi::c_void;

use cubecl::prelude::{CubeElement, CubePrimitive};
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use super::dispatch::{
    alloc_output, cube_count_for_len, cube_dim_1d, cubecl_buffer, dtype_mismatch,
    ensure_resident_on_runtime, launch_nullary_into, typed_tensor_array_arg,
};
use super::ffi::cutensor::{
    CudaDataType, CutensorComputeDescriptor, CutensorCudaStream, CutensorHandle, CutensorOperator,
    CutensorWorksizePreference, OperationDescriptor, Plan, PlanPreference, TensorDescriptor,
};
use super::interop::cuda_device_ptr_from_addr;
use super::memory::upload_tensor;
use super::{CudaBackend, CudaRuntime};
use crate::config::DotGeneralConfig;
use crate::kernels::structural;
use crate::{col_major_strides, Error, Tensor, TypedTensor};
use tenferro_tensor::{ContractionScalar, DotGeneralAccumulation};

const OP: &str = "dot_general";
const CUDA_ALLOCATION_ALIGNMENT: u32 = 256;

trait CutensorScalar: CubeElement + CubePrimitive + Clone + One + Zero {
    const DATA_TYPE: CudaDataType;
    const IS_COMPLEX: bool;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor;

    /// Wrap/unwrap between the typed tensor and the dtype-erased [`Tensor`],
    /// used to materialize scalar device constants.
    fn wrap_tensor(tensor: TypedTensor<Self>) -> Tensor;
    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>>;

    /// Launch the in-place scale kernel for this scalar type.
    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    );
}

impl CutensorScalar for f32 {
    const DATA_TYPE: CudaDataType = CudaDataType::R32F;
    const IS_COMPLEX: bool = false;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_32f()
    }
    fn wrap_tensor(tensor: TypedTensor<Self>) -> Tensor {
        Tensor::F32(tensor)
    }

    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::F32(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_float_kernel::launch_unchecked::<f32, CubeclCudaRuntime>(
                client, count, dim, out, factor,
            );
        }
    }
}

impl CutensorScalar for f64 {
    const DATA_TYPE: CudaDataType = CudaDataType::R64F;
    const IS_COMPLEX: bool = false;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_64f()
    }
    fn wrap_tensor(tensor: TypedTensor<Self>) -> Tensor {
        Tensor::F64(tensor)
    }

    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_float_kernel::launch_unchecked::<f64, CubeclCudaRuntime>(
                client, count, dim, out, factor,
            );
        }
    }
}

impl CutensorScalar for Complex32 {
    const DATA_TYPE: CudaDataType = CudaDataType::C32F;
    const IS_COMPLEX: bool = true;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_32f()
    }
    fn wrap_tensor(tensor: TypedTensor<Self>) -> Tensor {
        Tensor::C32(tensor)
    }

    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::C32(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_complex_kernel::launch_unchecked::<
                Complex32,
                CubeclCudaRuntime,
            >(client, count, dim, out, factor);
        }
    }
}

impl CutensorScalar for Complex64 {
    const DATA_TYPE: CudaDataType = CudaDataType::C64F;
    const IS_COMPLEX: bool = true;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_64f()
    }
    fn wrap_tensor(tensor: TypedTensor<Self>) -> Tensor {
        Tensor::C64(tensor)
    }

    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::C64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_complex_kernel::launch_unchecked::<
                Complex64,
                CubeclCudaRuntime,
            >(client, count, dim, out, factor);
        }
    }
}

struct DotGeneralLayout {
    lhs_modes: Vec<i32>,
    rhs_modes: Vec<i32>,
    output_modes: Vec<i32>,
    output_shape: Vec<usize>,
    lhs_extents: Vec<i64>,
    rhs_extents: Vec<i64>,
    output_extents: Vec<i64>,
    lhs_strides: Vec<i64>,
    rhs_strides: Vec<i64>,
    output_strides: Vec<i64>,
    contracting_elements: usize,
}

struct Workspace {
    _handle: Option<cubecl_runtime::server::Handle>,
    ptr: *mut c_void,
    size: u64,
}

impl Workspace {
    fn none() -> Self {
        Self {
            _handle: None,
            ptr: std::ptr::null_mut(),
            size: 0,
        }
    }
}

pub(super) fn dot_general(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(lhs), Tensor::F32(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::F32)
        }
        (Tensor::F64(lhs), Tensor::F64(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::F64)
        }
        (Tensor::C32(lhs), Tensor::C32(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::C32)
        }
        (Tensor::C64(lhs), Tensor::C64(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::C64)
        }
        _ => Err(dtype_mismatch(OP, lhs, rhs)),
    }
}

pub(super) fn dot_general_with_conj(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(lhs), Tensor::F32(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::F32)
        }
        (Tensor::F64(lhs), Tensor::F64(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::F64)
        }
        (Tensor::C32(lhs), Tensor::C32(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::C32)
        }
        (Tensor::C64(lhs), Tensor::C64(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::C64)
        }
        _ => Err(dtype_mismatch(OP, lhs, rhs)),
    }
}

/// Local extraction of typed accumulation coefficients; dtype mismatches
/// between the coefficient and the operand dtype are explicit errors.
trait FromContractionScalar: Sized {
    fn from_contraction_scalar(value: ContractionScalar) -> crate::Result<Self>;
}

macro_rules! impl_from_contraction_scalar {
    ($ty:ty, $variant:ident) => {
        impl FromContractionScalar for $ty {
            fn from_contraction_scalar(value: ContractionScalar) -> crate::Result<Self> {
                match value {
                    ContractionScalar::$variant(value) => Ok(value),
                    other => Err(Error::DTypeMismatch {
                        op: OP,
                        lhs: <$ty as tenferro_tensor::TensorScalar>::dtype(),
                        rhs: other.dtype(),
                    }),
                }
            }
        }
    };
}

impl_from_contraction_scalar!(f32, F32);
impl_from_contraction_scalar!(f64, F64);
impl_from_contraction_scalar!(Complex32, C32);
impl_from_contraction_scalar!(Complex64, C64);

/// CUDA-native accumulate-form contraction:
/// `out = alpha * op(lhs) * op(rhs) + beta * out` executed by a single
/// cuTENSOR contraction with `C = D = out` (no temporary result tensor).
///
/// Stage-1 scope (tensor4all/tenferro-rs#1287): compact GPU-resident owned
/// tensors on all three slots; anything else is an explicit error — no hidden
/// host transfer and no silent fallback.
pub(super) fn dot_general_with_conj_into_accum(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    out: &mut Tensor,
) -> crate::Result<()> {
    match (lhs, rhs, out) {
        (Tensor::F32(lhs), Tensor::F32(rhs), Tensor::F32(out)) => dot_general_typed_into_accum(
            backend,
            lhs,
            rhs,
            config,
            accumulation.lhs_conj,
            accumulation.rhs_conj,
            f32::from_contraction_scalar(accumulation.alpha)?,
            f32::from_contraction_scalar(accumulation.beta)?,
            out,
        ),
        (Tensor::F64(lhs), Tensor::F64(rhs), Tensor::F64(out)) => dot_general_typed_into_accum(
            backend,
            lhs,
            rhs,
            config,
            accumulation.lhs_conj,
            accumulation.rhs_conj,
            f64::from_contraction_scalar(accumulation.alpha)?,
            f64::from_contraction_scalar(accumulation.beta)?,
            out,
        ),
        (Tensor::C32(lhs), Tensor::C32(rhs), Tensor::C32(out)) => dot_general_typed_into_accum(
            backend,
            lhs,
            rhs,
            config,
            accumulation.lhs_conj,
            accumulation.rhs_conj,
            Complex32::from_contraction_scalar(accumulation.alpha)?,
            Complex32::from_contraction_scalar(accumulation.beta)?,
            out,
        ),
        (Tensor::C64(lhs), Tensor::C64(rhs), Tensor::C64(out)) => dot_general_typed_into_accum(
            backend,
            lhs,
            rhs,
            config,
            accumulation.lhs_conj,
            accumulation.rhs_conj,
            Complex64::from_contraction_scalar(accumulation.alpha)?,
            Complex64::from_contraction_scalar(accumulation.beta)?,
            out,
        ),
        (lhs, rhs, out) => {
            if lhs.dtype() != rhs.dtype() || lhs.dtype() != out.dtype() {
                return Err(dtype_mismatch(OP, lhs, rhs));
            }
            Err(Error::backend_failure(
                OP,
                "CUDA dot-general accumulation supports f32/f64/c32/c64 operands",
            ))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dot_general_typed_into_accum<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
    alpha: T,
    beta: T,
    out: &mut TypedTensor<T>,
) -> crate::Result<()>
where
    T: CutensorScalar + PartialEq + tenferro_tensor::TensorScalar,
{
    backend.runtime().set_current_cuda_context(OP)?;
    validate_dot_general(lhs, rhs, config)?;
    let layout = build_layout(lhs, rhs, config)?;
    if out.shape() != layout.output_shape.as_slice() {
        return Err(Error::ShapeMismatch {
            op: OP,
            lhs: out.shape().to_vec(),
            rhs: layout.output_shape.clone(),
        });
    }
    ensure_resident_on_runtime(backend.runtime(), lhs, OP)?;
    ensure_resident_on_runtime(backend.runtime(), rhs, OP)?;
    ensure_resident_on_runtime(backend.runtime(), out, OP)?;
    if out.n_elements() == 0 {
        return Ok(());
    }
    if layout.contracting_elements == 0 {
        // The contraction sum is empty: out = beta * out.
        return scale_in_place(backend.runtime(), out, beta);
    }

    let cutensor = backend.cutensor_handle()?;
    let desc_a = TensorDescriptor::new(
        cutensor,
        &layout.lhs_extents,
        &layout.lhs_strides,
        T::DATA_TYPE,
        CUDA_ALLOCATION_ALIGNMENT,
        OP,
    )?;
    let desc_b = TensorDescriptor::new(
        cutensor,
        &layout.rhs_extents,
        &layout.rhs_strides,
        T::DATA_TYPE,
        CUDA_ALLOCATION_ALIGNMENT,
        OP,
    )?;
    let desc_out = TensorDescriptor::new(
        cutensor,
        &layout.output_extents,
        &layout.output_strides,
        T::DATA_TYPE,
        CUDA_ALLOCATION_ALIGNMENT,
        OP,
    )?;
    let op_desc = OperationDescriptor::new_contraction_with_ops(
        cutensor,
        &desc_a,
        &layout.lhs_modes,
        cutensor_conj_op::<T>(lhs_conj),
        &desc_b,
        &layout.rhs_modes,
        cutensor_conj_op::<T>(rhs_conj),
        &desc_out,
        &layout.output_modes,
        &desc_out,
        &layout.output_modes,
        T::compute_descriptor(cutensor),
        OP,
    )?;
    let pref = PlanPreference::new_default(cutensor, OP)?;
    let workspace_size = cutensor.estimate_workspace_size(
        &op_desc,
        &pref,
        CutensorWorksizePreference::Default,
        OP,
    )?;
    let plan = Plan::new(cutensor, &op_desc, &pref, workspace_size, OP)?;
    let workspace = alloc_workspace(backend.runtime(), workspace_size)?;

    let lhs_ptr = typed_device_ptr(backend.runtime(), lhs)?;
    let rhs_ptr = typed_device_ptr(backend.runtime(), rhs)?;
    let out_ptr = typed_device_ptr(backend.runtime(), out)?;

    let stream = raw_stream(backend.runtime())?;
    // C = D = out: cuTENSOR reads the destination as the accumulator (skipped
    // by cuTENSOR itself when beta == 0) and writes the result in place.
    unsafe {
        cutensor.contract(
            &plan,
            &alpha as *const T as *const c_void,
            lhs_ptr as *const c_void,
            rhs_ptr as *const c_void,
            &beta as *const T as *const c_void,
            out_ptr as *const c_void,
            out_ptr,
            workspace.ptr,
            workspace.size,
            stream,
            OP,
        )?;
    }
    Ok(())
}

/// Device-side `out *= beta` for the degenerate zero-contraction case. The
/// factor is materialized as an explicit one-element device constant; user
/// operand tensors are never transferred.
fn scale_in_place<T>(rt: &CudaRuntime, out: &mut TypedTensor<T>, beta: T) -> crate::Result<()>
where
    T: CutensorScalar + PartialEq + tenferro_tensor::TensorScalar,
{
    if beta == T::one() {
        return Ok(());
    }
    if beta == T::zero() {
        return launch_nullary_into(
            rt,
            out,
            OP,
            cube_count_for_len(out.n_elements())?,
            cube_dim_1d(),
            |client, count, dim, out| unsafe {
                structural::fill_zero_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client, count, dim, out,
                );
            },
        );
    }
    let factor_host = T::wrap_tensor(TypedTensor::from_vec_col_major(vec![1], vec![beta])?);
    let factor_device = upload_tensor(rt, &factor_host)?;
    let factor_typed = T::unwrap_tensor(&factor_device)
        .ok_or_else(|| Error::backend_failure(OP, "scale factor upload changed dtype"))?;
    ensure_resident_on_runtime(rt, out, OP)?;
    ensure_resident_on_runtime(rt, factor_typed, OP)?;
    let out_arg = typed_tensor_array_arg(out, OP)?;
    let factor_arg = typed_tensor_array_arg(factor_typed, OP)?;
    T::launch_scale_in_place(
        rt.client(),
        cube_count_for_len(out.n_elements())?,
        cube_dim_1d(),
        out_arg,
        factor_arg,
    );
    Ok(())
}

fn dot_general_typed<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: CutensorScalar,
{
    dot_general_typed_with_conj(backend, lhs, rhs, config, false, false)
}

fn dot_general_typed_with_conj<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<T>>
where
    T: CutensorScalar,
{
    backend.runtime().set_current_cuda_context(OP)?;
    validate_dot_general(lhs, rhs, config)?;
    let layout = build_layout(lhs, rhs, config)?;
    let output = alloc_output::<T>(backend.runtime(), &layout.output_shape)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    if layout.contracting_elements == 0 {
        return zero_alloc::<T>(backend.runtime(), &layout.output_shape);
    }

    let cutensor = backend.cutensor_handle()?;
    let desc_a = TensorDescriptor::new(
        cutensor,
        &layout.lhs_extents,
        &layout.lhs_strides,
        T::DATA_TYPE,
        CUDA_ALLOCATION_ALIGNMENT,
        OP,
    )?;
    let desc_b = TensorDescriptor::new(
        cutensor,
        &layout.rhs_extents,
        &layout.rhs_strides,
        T::DATA_TYPE,
        CUDA_ALLOCATION_ALIGNMENT,
        OP,
    )?;
    let desc_out = TensorDescriptor::new(
        cutensor,
        &layout.output_extents,
        &layout.output_strides,
        T::DATA_TYPE,
        CUDA_ALLOCATION_ALIGNMENT,
        OP,
    )?;
    let op_desc = OperationDescriptor::new_contraction_with_ops(
        cutensor,
        &desc_a,
        &layout.lhs_modes,
        cutensor_conj_op::<T>(lhs_conj),
        &desc_b,
        &layout.rhs_modes,
        cutensor_conj_op::<T>(rhs_conj),
        &desc_out,
        &layout.output_modes,
        &desc_out,
        &layout.output_modes,
        T::compute_descriptor(cutensor),
        OP,
    )?;
    let pref = PlanPreference::new_default(cutensor, OP)?;
    let workspace_size = cutensor.estimate_workspace_size(
        &op_desc,
        &pref,
        CutensorWorksizePreference::Default,
        OP,
    )?;
    let plan = Plan::new(cutensor, &op_desc, &pref, workspace_size, OP)?;
    let workspace = alloc_workspace(backend.runtime(), workspace_size)?;

    let lhs_ptr = typed_device_ptr(backend.runtime(), lhs)?;
    let rhs_ptr = typed_device_ptr(backend.runtime(), rhs)?;
    let output_ptr = typed_device_ptr(backend.runtime(), &output)?;

    let accumulator = alloc_output::<T>(backend.runtime(), &layout.output_shape)?;
    let accumulator_ptr = typed_device_ptr(backend.runtime(), &accumulator)?;

    let alpha = T::one();
    let beta = T::zero();
    let stream = raw_stream(backend.runtime())?;
    unsafe {
        cutensor.contract(
            &plan,
            &alpha as *const T as *const c_void,
            lhs_ptr as *const c_void,
            rhs_ptr as *const c_void,
            &beta as *const T as *const c_void,
            accumulator_ptr as *const c_void,
            output_ptr,
            workspace.ptr,
            workspace.size,
            stream,
            OP,
        )?;
    }

    Ok(output)
}

fn cutensor_conj_op<T: CutensorScalar>(conj: bool) -> CutensorOperator {
    if conj && T::IS_COMPLEX {
        CutensorOperator::Conj
    } else {
        CutensorOperator::Identity
    }
}

fn raw_stream(rt: &CudaRuntime) -> crate::Result<CutensorCudaStream> {
    Ok(rt.raw_cuda_stream()? as usize as CutensorCudaStream)
}

fn alloc_workspace(rt: &CudaRuntime, workspace_size: u64) -> crate::Result<Workspace> {
    if workspace_size == 0 {
        return Ok(Workspace::none());
    }
    let workspace_len = usize::try_from(workspace_size).map_err(|_| {
        crate::Error::backend_failure(
            OP,
            format!("workspace size {workspace_size} does not fit in usize"),
        )
    })?;
    let handle = rt.client().empty(workspace_len);
    let resource = rt.client().get_resource(handle.clone()).map_err(|err| {
        crate::Error::backend_failure(OP, format!("failed to obtain workspace resource: {err:?}"))
    })?;
    Ok(Workspace {
        _handle: Some(handle),
        ptr: cuda_device_ptr_from_addr(resource.resource().ptr, OP)?,
        size: workspace_size,
    })
}

fn typed_device_ptr<T: 'static>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T>,
) -> crate::Result<*mut c_void> {
    ensure_resident_on_runtime(rt, tensor, OP)?;
    let buffer = cubecl_buffer(tensor, OP)?;
    let resource = rt
        .client()
        .get_resource(buffer.handle().clone())
        .map_err(|err| {
            crate::Error::backend_failure(OP, format!("failed to obtain CubeCL resource: {err:?}"))
        })?;
    // The residency check above ties this raw FFI pointer to the caller's runtime/device.
    cuda_device_ptr_from_addr(resource.resource().ptr, OP)
}

fn zero_alloc<T>(rt: &CudaRuntime, shape: &[usize]) -> crate::Result<TypedTensor<T>>
where
    T: CutensorScalar,
{
    let output = alloc_output::<T>(rt, shape)?;
    launch_nullary_into(
        rt,
        &output,
        OP,
        cube_count_for_len(output.n_elements())?,
        cube_dim_1d(),
        |client, count, dim, out| unsafe {
            structural::fill_zero_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                client, count, dim, out,
            );
        },
    )?;
    Ok(output)
}

fn build_layout<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<DotGeneralLayout> {
    let lhs_free = free_axes(
        lhs.shape().len(),
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_axes(
        rhs.shape().len(),
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );

    let mut lhs_modes = vec![-1i32; lhs.shape().len()];
    let mut rhs_modes = vec![-1i32; rhs.shape().len()];
    let mut output_modes =
        Vec::with_capacity(lhs_free.len() + rhs_free.len() + config.lhs_batch_dims.len());
    let mut output_shape = Vec::with_capacity(output_modes.capacity());
    let mut batch_modes = Vec::with_capacity(config.lhs_batch_dims.len());
    let mut batch_shape = Vec::with_capacity(config.lhs_batch_dims.len());
    let mut next_mode = 0i32;
    let mut contracting_elements = 1usize;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        let mode = next_mode;
        next_mode += 1;
        lhs_modes[lhs_axis] = mode;
        rhs_modes[rhs_axis] = mode;
        contracting_elements = contracting_elements
            .checked_mul(lhs.shape()[lhs_axis])
            .ok_or_else(|| Error::InvalidConfig {
                op: OP,
                message: format!(
                    "contracting dimension product overflows usize for lhs shape {:?}",
                    lhs.shape()
                ),
            })?;
    }

    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        let mode = next_mode;
        next_mode += 1;
        lhs_modes[lhs_axis] = mode;
        rhs_modes[rhs_axis] = mode;
        batch_modes.push(mode);
        batch_shape.push(lhs.shape()[lhs_axis]);
    }

    for &lhs_axis in &lhs_free {
        let mode = next_mode;
        next_mode += 1;
        lhs_modes[lhs_axis] = mode;
        output_modes.push(mode);
        output_shape.push(lhs.shape()[lhs_axis]);
    }

    for &rhs_axis in &rhs_free {
        let mode = next_mode;
        next_mode += 1;
        rhs_modes[rhs_axis] = mode;
        output_modes.push(mode);
        output_shape.push(rhs.shape()[rhs_axis]);
    }

    output_modes.extend_from_slice(&batch_modes);
    output_shape.extend_from_slice(&batch_shape);

    let lhs_extents = dims_to_i64(lhs.shape())?;
    let rhs_extents = dims_to_i64(rhs.shape())?;
    let output_extents = dims_to_i64(&output_shape)?;
    let lhs_strides = strides_to_i64(&col_major_strides(lhs.shape())?)?;
    let rhs_strides = strides_to_i64(&col_major_strides(rhs.shape())?)?;
    let output_strides = strides_to_i64(&col_major_strides(&output_shape)?)?;

    Ok(DotGeneralLayout {
        lhs_modes,
        rhs_modes,
        output_modes,
        output_shape,
        lhs_extents,
        rhs_extents,
        output_extents,
        lhs_strides,
        rhs_strides,
        output_strides,
        contracting_elements,
    })
}

fn dims_to_i64(dims: &[usize]) -> crate::Result<Vec<i64>> {
    dims.iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| Error::InvalidConfig {
                op: OP,
                message: format!("extent {dim} exceeds cuTENSOR i64 limit"),
            })
        })
        .collect()
}

fn strides_to_i64(strides: &[isize]) -> crate::Result<Vec<i64>> {
    strides
        .iter()
        .map(|&stride| {
            i64::try_from(stride).map_err(|_| Error::InvalidConfig {
                op: OP,
                message: format!("stride {stride} exceeds cuTENSOR i64 limit"),
            })
        })
        .collect()
}

fn free_axes(rank: usize, contracting: &[usize], batch: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn validate_axis_list(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::AxisOutOfBounds { op, axis, rank });
        }
        if seen[axis] {
            return Err(Error::DuplicateAxis { op, axis, role });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_role_disjoint(
    op: &'static str,
    first_role: &'static str,
    first_axes: &[usize],
    second_role: &'static str,
    second_axes: &[usize],
) -> crate::Result<()> {
    for &axis in first_axes {
        if second_axes.contains(&axis) {
            return Err(Error::AxisRoleConflict {
                op,
                axis,
                first_role,
                second_role,
            });
        }
    }
    Ok(())
}

fn validate_dot_general<T>(
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<()> {
    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs contracting dim counts differ".into(),
        });
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs batch dim counts differ".into(),
        });
    }

    let lhs_rank = lhs.shape().len();
    let rhs_rank = rhs.shape().len();

    validate_axis_list(
        OP,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        lhs_rank,
    )?;
    validate_axis_list(
        OP,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        rhs_rank,
    )?;
    validate_axis_list(OP, "lhs_batch", &config.lhs_batch_dims, lhs_rank)?;
    validate_axis_list(OP, "rhs_batch", &config.rhs_batch_dims, rhs_rank)?;
    validate_role_disjoint(
        OP,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        "lhs_batch",
        &config.lhs_batch_dims,
    )?;
    validate_role_disjoint(
        OP,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        "rhs_batch",
        &config.rhs_batch_dims,
    )?;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        if lhs.shape()[lhs_axis] != rhs.shape()[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: format!(
                    "contracting dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs.shape()[lhs_axis],
                    rhs.shape()[rhs_axis]
                ),
            });
        }
    }

    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs.shape()[lhs_axis] != rhs.shape()[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: format!(
                    "batch dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs.shape()[lhs_axis],
                    rhs.shape()[rhs_axis]
                ),
            });
        }
    }

    Ok(())
}
