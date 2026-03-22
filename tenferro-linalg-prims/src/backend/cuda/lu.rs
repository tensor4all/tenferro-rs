#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{compile_ptx, Ptx};
#[cfg(feature = "cuda")]
use tenferro_device::Error;
use tenferro_device::Result;
#[cfg(feature = "cuda")]
use tenferro_tensor::MemoryOrder;
use tenferro_tensor::Tensor;

#[cfg(feature = "cuda")]
use super::runtime::{context_device_ptr, copy_device_to_host, load_runtime, DeviceAllocation};
#[cfg(feature = "cuda")]
use super::scalar_type::CudaDataType;
use super::scalar_type::CudaLinalgScalar;
#[cfg(feature = "cuda")]
use crate::backend::linalg_utils::clone_batched_column_major;
#[cfg(feature = "cuda")]
use crate::backend::tensor_helpers::{batch_count, validate_matrix_shape};
use crate::{LuTensorExResult, LuTensorResult};

#[cfg(feature = "cuda")]
const LU_SPLIT_LOWER_KERNEL_NAME_F32: &str = "lu_split_lower_f32";
#[cfg(feature = "cuda")]
const LU_SPLIT_LOWER_KERNEL_NAME_F64: &str = "lu_split_lower_f64";
#[cfg(feature = "cuda")]
const LU_SPLIT_LOWER_KERNEL_NAME_C32: &str = "lu_split_lower_complex32";
#[cfg(feature = "cuda")]
const LU_SPLIT_LOWER_KERNEL_NAME_C64: &str = "lu_split_lower_complex64";
#[cfg(feature = "cuda")]
const LU_SPLIT_UPPER_KERNEL_NAME_F32: &str = "lu_split_upper_f32";
#[cfg(feature = "cuda")]
const LU_SPLIT_UPPER_KERNEL_NAME_F64: &str = "lu_split_upper_f64";
#[cfg(feature = "cuda")]
const LU_SPLIT_UPPER_KERNEL_NAME_C32: &str = "lu_split_upper_complex32";
#[cfg(feature = "cuda")]
const LU_SPLIT_UPPER_KERNEL_NAME_C64: &str = "lu_split_upper_complex64";

#[cfg(feature = "cuda")]
const LU_SPLIT_CUDA_SRC: &str = r#"
extern "C" __global__ void lu_split_lower_f32(
    const float* packed,
    float* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % m;
    unsigned long long col = within / m;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row < col) {
        out[idx] = 0.0f;
    } else if (row == col) {
        out[idx] = 1.0f;
    } else {
        out[idx] = packed[src];
    }
}

extern "C" __global__ void lu_split_lower_f64(
    const double* packed,
    double* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % m;
    unsigned long long col = within / m;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row < col) {
        out[idx] = 0.0;
    } else if (row == col) {
        out[idx] = 1.0;
    } else {
        out[idx] = packed[src];
    }
}

typedef struct { float re; float im; } complex32_t;
typedef struct { double re; double im; } complex64_t;

extern "C" __global__ void lu_split_lower_complex32(
    const complex32_t* packed,
    complex32_t* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % m;
    unsigned long long col = within / m;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row < col) {
        out[idx].re = 0.0f;
        out[idx].im = 0.0f;
    } else if (row == col) {
        out[idx].re = 1.0f;
        out[idx].im = 0.0f;
    } else {
        out[idx] = packed[src];
    }
}

extern "C" __global__ void lu_split_lower_complex64(
    const complex64_t* packed,
    complex64_t* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % m;
    unsigned long long col = within / m;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row < col) {
        out[idx].re = 0.0;
        out[idx].im = 0.0;
    } else if (row == col) {
        out[idx].re = 1.0;
        out[idx].im = 0.0;
    } else {
        out[idx] = packed[src];
    }
}

extern "C" __global__ void lu_split_upper_f32(
    const float* packed,
    float* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % k;
    unsigned long long col = within / k;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row <= col) {
        out[idx] = packed[src];
    } else {
        out[idx] = 0.0f;
    }
}

extern "C" __global__ void lu_split_upper_f64(
    const double* packed,
    double* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % k;
    unsigned long long col = within / k;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row <= col) {
        out[idx] = packed[src];
    } else {
        out[idx] = 0.0;
    }
}

extern "C" __global__ void lu_split_upper_complex32(
    const complex32_t* packed,
    complex32_t* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % k;
    unsigned long long col = within / k;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row <= col) {
        out[idx] = packed[src];
    } else {
        out[idx].re = 0.0f;
        out[idx].im = 0.0f;
    }
}

extern "C" __global__ void lu_split_upper_complex64(
    const complex64_t* packed,
    complex64_t* out,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k,
    unsigned long long matrix_size,
    unsigned long long out_matrix_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long batch = idx / out_matrix_size;
    unsigned long long within = idx % out_matrix_size;
    unsigned long long row = within % k;
    unsigned long long col = within / k;
    unsigned long long src = batch * matrix_size + col * m + row;

    if (row <= col) {
        out[idx] = packed[src];
    } else {
        out[idx].re = 0.0;
        out[idx].im = 0.0;
    }
}
"#;

#[cfg(feature = "cuda")]
fn lu_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(LU_SPLIT_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for LU split kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

#[cfg(feature = "cuda")]
fn checked_mul(lhs: usize, rhs: usize, label: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidArgument(format!("{label} overflow: {lhs} * {rhs}")))
}

#[cfg(feature = "cuda")]
fn as_i32(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{label} does not fit in i32: {value}")))
}

#[cfg(feature = "cuda")]
fn to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value)
        .map_err(|_| Error::InvalidArgument(format!("{label} does not fit in u64: {value}")))
}

#[cfg(feature = "cuda")]
pub(super) fn has_lu_support<T: CudaLinalgScalar>() -> bool {
    matches!(
        T::cuda_data_type(),
        CudaDataType::F32 | CudaDataType::F64 | CudaDataType::Complex32 | CudaDataType::Complex64
    )
}

#[cfg(not(feature = "cuda"))]
pub(super) fn has_lu_support<T: CudaLinalgScalar>() -> bool {
    let _ = T::cuda_data_type();
    false
}

#[cfg(feature = "cuda")]
fn load_lu_kernel(
    ctx: &tenferro_prims::CudaContext,
    kernel_name: &str,
) -> Result<(
    cudarc::driver::CudaFunction,
    std::sync::Arc<cudarc::driver::CudaStream>,
)> {
    ctx.bind_to_device()?;
    let cuda_ctx = ctx.shared_runtime().clone().context();
    let module = cuda_ctx
        .load_module(lu_ptx()?)
        .map_err(|err| Error::DeviceError(format!("CUDA module load failed: {err:?}")))?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|err| Error::DeviceError(format!("CUDA load function failed: {err:?}")))?;
    Ok((kernel, cuda_ctx.default_stream()))
}

#[cfg(feature = "cuda")]
fn check_getrf_info(info: i32, allow_positive: bool, op: &str) -> Result<()> {
    if info == 0 || (allow_positive && info > 0) {
        return Ok(());
    }
    if info < 0 {
        return Err(Error::DeviceError(format!(
            "{op} reported an invalid parameter at position {}",
            -info
        )));
    }
    Err(Error::DeviceError(format!(
        "{op} failed with unexpected status info={info}"
    )))
}

#[cfg(feature = "cuda")]
fn pivots_to_forward_perm(m: usize, pivots: &[i32]) -> Result<Vec<usize>> {
    let mut perm: Vec<usize> = (0..m).collect();
    for (i, &p) in pivots.iter().enumerate() {
        if p <= 0 {
            return Err(Error::DeviceError(
                "lu: cuSOLVER returned non-positive pivot index".into(),
            ));
        }
        let j = usize::try_from(p - 1)
            .map_err(|_| Error::DeviceError("lu: pivot index underflow".into()))?;
        if j >= m {
            return Err(Error::DeviceError(format!(
                "lu: cuSOLVER pivot index {p} out of range for m={m}"
            )));
        }
        perm.swap(i, j);
    }
    Ok(perm)
}

#[cfg(feature = "cuda")]
fn launch_lu_split<T: CudaLinalgScalar>(
    ctx: &mut tenferro_prims::CudaContext,
    packed: &Tensor<T>,
    output: &Tensor<T>,
    m: usize,
    n: usize,
    k: usize,
    kernel_name: &str,
) -> Result<()>
where
    T: CudaLinalgScalar,
{
    let (kernel, stream) = load_lu_kernel(ctx, kernel_name)?;
    let packed_ptr = packed
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("packed LU tensor buffer is not on GPU".into()))?
        as u64;
    let out_ptr = output
        .buffer()
        .as_device_ptr()
        .ok_or_else(|| Error::DeviceError("LU output tensor buffer is not on GPU".into()))?
        as *mut T as u64;
    let matrix_size = to_u64(checked_mul(m, n, "lu matrix size")?, "lu matrix size")?;
    let out_matrix_size = to_u64(
        checked_mul(output.dims()[0], output.dims()[1], "lu output matrix size")?,
        "lu output matrix size",
    )?;
    let numel = to_u64(output.len(), "lu output numel")?;
    let numel_u32 = u32::try_from(output.len()).map_err(|_| {
        Error::InvalidArgument("lu split currently requires len <= u32::MAX".into())
    })?;
    let config = LaunchConfig {
        grid_dim: (numel_u32.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let m_u64 = to_u64(m, "lu m")?;
    let n_u64 = to_u64(n, "lu n")?;
    let k_u64 = to_u64(k, "lu k")?;

    unsafe {
        stream
            .launch_builder(&kernel)
            .arg(&packed_ptr)
            .arg(&out_ptr)
            .arg(&m_u64)
            .arg(&n_u64)
            .arg(&k_u64)
            .arg(&matrix_size)
            .arg(&out_matrix_size)
            .arg(&numel)
            .launch(config)
            .map_err(|err| Error::DeviceError(format!("CUDA LU split launch failed: {err:?}")))?;
    }
    stream
        .synchronize()
        .map_err(|err| Error::DeviceError(format!("CUDA stream synchronize failed: {err:?}")))?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn lu_factor_common<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
    collect_info: bool,
) -> Result<(Tensor<T>, Tensor<T>, Vec<i32>, Vec<i32>)>
where
    T: CudaLinalgScalar,
{
    if !has_lu_support::<T>() {
        return Err(Error::DeviceError(format!(
            "CUDA lu_factor currently supports only f32/f64/complex32/complex64, got {:?}",
            T::cuda_data_type()
        )));
    }

    let (m, n, batch_dims) = validate_matrix_shape(a)?;
    let k = m.min(n);
    let bc = batch_count(batch_dims);
    let l_dims = {
        let mut dims = vec![m, k];
        dims.extend_from_slice(batch_dims);
        dims
    };
    let u_dims = {
        let mut dims = vec![k, n];
        dims.extend_from_slice(batch_dims);
        dims
    };

    let a_work = clone_batched_column_major(ctx, a)?;
    let l = Tensor::zeros(
        &l_dims,
        a_work.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let u = Tensor::zeros(
        &u_dims,
        a_work.logical_memory_space(),
        MemoryOrder::ColumnMajor,
    );
    let mut pivots_out = vec![0i32; m * bc];
    let mut info = vec![0i32; bc];

    if m == 0 || n == 0 || bc == 0 {
        for batch in 0..bc {
            for i in 0..m {
                pivots_out[batch * m + i] = i as i32;
            }
        }
        return Ok((l, u, pivots_out, info));
    }

    let dtype = T::cuda_data_type();
    let runtime = load_runtime(ctx)?;
    let a_base = context_device_ptr(ctx, &a_work, "lu_factor a")?.cast::<T>();
    let a_offset = a_work.offset() as usize;
    let a_stride = checked_mul(m, n, "lu factor a_stride")?;
    let lda = as_i32(m, "lu_factor lda")?;
    let m_i32 = as_i32(m, "lu_factor m")?;
    let n_i32 = as_i32(n, "lu_factor n")?;

    let lwork = runtime.cusolver_api().getrf_buffer_size(
        dtype,
        runtime.cusolver_handle.raw,
        m_i32,
        n_i32,
        a_base.cast::<c_void>(),
        lda,
    )?;
    let workspace = DeviceAllocation::alloc(
        ctx,
        checked_mul(
            usize::try_from(lwork).map_err(|_| {
                Error::DeviceError(format!(
                    "lu_factor getrf workspace size was negative: {lwork}"
                ))
            })?,
            std::mem::size_of::<T>(),
            "lu_factor workspace bytes",
        )?,
        "cudaMalloc(lu_factor workspace)",
    )?;
    let pivots = DeviceAllocation::alloc(
        ctx,
        checked_mul(k, std::mem::size_of::<i32>(), "lu_factor pivots bytes")?,
        "cudaMalloc(lu_factor pivots)",
    )?;
    let info_dev = DeviceAllocation::alloc(
        ctx,
        std::mem::size_of::<i32>(),
        "cudaMalloc(lu_factor info)",
    )?;
    let mut host_pivots = vec![0i32; k];
    let mut host_info = [0i32; 1];

    for batch in 0..bc {
        let a_ptr = unsafe { a_base.add(a_offset + batch * a_stride) }.cast::<c_void>();
        runtime.cusolver_api().getrf(
            dtype,
            runtime.cusolver_handle.raw,
            m_i32,
            n_i32,
            a_ptr,
            lda,
            workspace.as_mut_ptr(),
            pivots.as_mut_ptr().cast::<i32>(),
            info_dev.as_mut_ptr().cast::<i32>(),
        )?;
        ctx.shared_runtime()
            .clone()
            .context()
            .default_stream()
            .synchronize()
            .map_err(|err| {
                Error::DeviceError(format!("CUDA stream synchronize failed: {err:?}"))
            })?;
        copy_device_to_host(
            ctx,
            pivots.as_mut_ptr().cast::<c_void>(),
            &mut host_pivots,
            "cudaMemcpyDtoH(lu_factor pivots)",
        )?;
        copy_device_to_host(
            ctx,
            info_dev.as_mut_ptr().cast::<c_void>(),
            &mut host_info,
            "cudaMemcpyDtoH(lu_factor info)",
        )?;
        check_getrf_info(host_info[0], true, "lu_factor/getrf")?;
        if collect_info {
            info[batch] = host_info[0];
        }

        let perm = pivots_to_forward_perm(m, &host_pivots)?;
        for (i, &p) in perm.iter().enumerate() {
            pivots_out[batch * m + i] = p as i32;
        }
    }

    let (lower_kernel, upper_kernel) = match dtype {
        CudaDataType::F32 => (
            LU_SPLIT_LOWER_KERNEL_NAME_F32,
            LU_SPLIT_UPPER_KERNEL_NAME_F32,
        ),
        CudaDataType::F64 => (
            LU_SPLIT_LOWER_KERNEL_NAME_F64,
            LU_SPLIT_UPPER_KERNEL_NAME_F64,
        ),
        CudaDataType::Complex32 => (
            LU_SPLIT_LOWER_KERNEL_NAME_C32,
            LU_SPLIT_UPPER_KERNEL_NAME_C32,
        ),
        CudaDataType::Complex64 => (
            LU_SPLIT_LOWER_KERNEL_NAME_C64,
            LU_SPLIT_UPPER_KERNEL_NAME_C64,
        ),
    };

    ctx.shared_runtime()
        .clone()
        .context()
        .default_stream()
        .synchronize()
        .map_err(|err| Error::DeviceError(format!("CUDA stream synchronize failed: {err:?}")))?;
    launch_lu_split(ctx, &a_work, &l, m, n, k, lower_kernel)?;
    launch_lu_split(ctx, &a_work, &u, m, n, k, upper_kernel)?;

    if collect_info {
        Ok((l, u, pivots_out, info))
    } else {
        Ok((l, u, pivots_out, Vec::new()))
    }
}

#[cfg(feature = "cuda")]
pub(super) fn lu_factor<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
) -> Result<LuTensorResult<T>>
where
    T: CudaLinalgScalar,
{
    let (l, u, pivots, _info) = lu_factor_common(ctx, a, false)?;
    Ok(LuTensorResult { l, u, pivots })
}

#[cfg(feature = "cuda")]
pub(super) fn lu_factor_ex<T>(
    ctx: &mut tenferro_prims::CudaContext,
    a: &Tensor<T>,
) -> Result<LuTensorExResult<T>>
where
    T: CudaLinalgScalar,
{
    let (l, u, pivots, info) = lu_factor_common(ctx, a, true)?;
    Ok(LuTensorExResult { l, u, pivots, info })
}

#[cfg(not(feature = "cuda"))]
pub(super) fn lu_factor<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
) -> Result<LuTensorResult<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("lu_factor")
}

#[cfg(not(feature = "cuda"))]
pub(super) fn lu_factor_ex<T>(
    _ctx: &mut tenferro_prims::CudaContext,
    _a: &Tensor<T>,
) -> Result<LuTensorExResult<T>>
where
    T: CudaLinalgScalar,
{
    super::runtime::unsupported("lu_factor_ex")
}
