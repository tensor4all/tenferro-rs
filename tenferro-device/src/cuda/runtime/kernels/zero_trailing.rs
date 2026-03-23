use std::sync::{Arc, OnceLock};

use cudarc::{
    driver::{CudaFunction, CudaStream},
    nvrtc::Ptx,
};

use super::super::state::CudaRuntime;
use super::helpers::{compile_ptx_once, load_kernel_from_ptx};
use crate::{Error, Result};

pub const ZERO_TRAILING_VALIDATE_KERNEL_NAME_F32: &str = "validate_keep_counts_f32";
pub const ZERO_TRAILING_VALIDATE_KERNEL_NAME_F64: &str = "validate_keep_counts_f64";
pub const ZERO_TRAILING_KERNEL_NAME_F32: &str = "zero_trailing_by_counts_f32";
pub const ZERO_TRAILING_KERNEL_NAME_F64: &str = "zero_trailing_by_counts_f64";
pub const ZERO_TRAILING_CUDA_SRC: &str = r#"
__device__ int classify_keep_count_f32(float value, long long axis_len) {
    if (!isfinite(value)) {
        return 1;
    }
    if (value < 0.0f) {
        return 2;
    }
    float rounded = nearbyintf(value);
    if (rounded != value) {
        return 3;
    }
    if (rounded > (float)axis_len) {
        return 4;
    }
    return 0;
}

__device__ int classify_keep_count_f64(double value, long long axis_len) {
    if (!isfinite(value)) {
        return 1;
    }
    if (value < 0.0) {
        return 2;
    }
    double rounded = nearbyint(value);
    if (rounded != value) {
        return 3;
    }
    if (rounded > (double)axis_len) {
        return 4;
    }
    return 0;
}

extern "C" __global__ void validate_keep_counts_f32(
    const float* keep_counts,
    const long long* batch_dims,
    const long long* keep_count_strides,
    long long keep_count_offset,
    int batch_rank,
    long long axis_len,
    unsigned long long count_numel,
    int* status_out
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= count_numel) {
        return;
    }

    unsigned long long remainder = idx;
    long long count_index = keep_count_offset;
    for (int axis = 0; axis < batch_rank; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)batch_dims[axis]);
        remainder /= (unsigned long long)batch_dims[axis];
        count_index += coord * keep_count_strides[axis];
    }

    int code = classify_keep_count_f32(keep_counts[count_index], axis_len);
    if (code != 0) {
        atomicCAS(status_out, 0, code);
    }
}

extern "C" __global__ void validate_keep_counts_f64(
    const double* keep_counts,
    const long long* batch_dims,
    const long long* keep_count_strides,
    long long keep_count_offset,
    int batch_rank,
    long long axis_len,
    unsigned long long count_numel,
    int* status_out
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= count_numel) {
        return;
    }

    unsigned long long remainder = idx;
    long long count_index = keep_count_offset;
    for (int axis = 0; axis < batch_rank; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)batch_dims[axis]);
        remainder /= (unsigned long long)batch_dims[axis];
        count_index += coord * keep_count_strides[axis];
    }

    int code = classify_keep_count_f64(keep_counts[count_index], axis_len);
    if (code != 0) {
        atomicCAS(status_out, 0, code);
    }
}

extern "C" __global__ void zero_trailing_by_counts_f32(
    const unsigned char* src,
    unsigned char* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    const float* keep_counts,
    const long long* keep_count_strides,
    long long keep_count_offset,
    int ndim,
    int axis,
    int structural_rank,
    unsigned long long elem_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long remainder = idx;
    long long src_index = src_offset;
    long long dst_index = dst_offset;
    long long count_index = keep_count_offset;
    long long axis_coord = 0;

    for (int dim_axis = 0; dim_axis < ndim; ++dim_axis) {
        long long coord = (long long)(remainder % (unsigned long long)dims[dim_axis]);
        remainder /= (unsigned long long)dims[dim_axis];
        src_index += coord * src_strides[dim_axis];
        dst_index += coord * dst_strides[dim_axis];
        if (dim_axis == axis) {
            axis_coord = coord;
        }
        if (dim_axis >= structural_rank) {
            count_index += coord * keep_count_strides[dim_axis - structural_rank];
        }
    }

    long long keep = (long long)keep_counts[count_index];
    unsigned long long src_byte = (unsigned long long)src_index * elem_size;
    unsigned long long dst_byte = (unsigned long long)dst_index * elem_size;
    if (axis_coord < keep) {
        for (unsigned long long byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
            dst[dst_byte + byte_idx] = src[src_byte + byte_idx];
        }
    } else {
        for (unsigned long long byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
            dst[dst_byte + byte_idx] = 0;
        }
    }
}

extern "C" __global__ void zero_trailing_by_counts_f64(
    const unsigned char* src,
    unsigned char* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    const double* keep_counts,
    const long long* keep_count_strides,
    long long keep_count_offset,
    int ndim,
    int axis,
    int structural_rank,
    unsigned long long elem_size,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    unsigned long long remainder = idx;
    long long src_index = src_offset;
    long long dst_index = dst_offset;
    long long count_index = keep_count_offset;
    long long axis_coord = 0;

    for (int dim_axis = 0; dim_axis < ndim; ++dim_axis) {
        long long coord = (long long)(remainder % (unsigned long long)dims[dim_axis]);
        remainder /= (unsigned long long)dims[dim_axis];
        src_index += coord * src_strides[dim_axis];
        dst_index += coord * dst_strides[dim_axis];
        if (dim_axis == axis) {
            axis_coord = coord;
        }
        if (dim_axis >= structural_rank) {
            count_index += coord * keep_count_strides[dim_axis - structural_rank];
        }
    }

    long long keep = (long long)keep_counts[count_index];
    unsigned long long src_byte = (unsigned long long)src_index * elem_size;
    unsigned long long dst_byte = (unsigned long long)dst_index * elem_size;
    if (axis_coord < keep) {
        for (unsigned long long byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
            dst[dst_byte + byte_idx] = src[src_byte + byte_idx];
        }
    } else {
        for (unsigned long long byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
            dst[dst_byte + byte_idx] = 0;
        }
    }
}
"#;

pub fn map_keep_count_status(status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        1 => Err(Error::InvalidArgument(
            "keep_counts values must be finite".into(),
        )),
        2 => Err(Error::InvalidArgument(
            "keep_counts values must be non-negative".into(),
        )),
        3 => Err(Error::InvalidArgument(
            "keep_counts values must be integer-valued".into(),
        )),
        4 => Err(Error::InvalidArgument(
            "keep_counts values exceed axis length".into(),
        )),
        _ => Err(Error::DeviceError(format!(
            "unknown keep-count validation status {status}"
        ))),
    }
}

pub fn zero_trailing_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, ZERO_TRAILING_CUDA_SRC, "zero-trailing kernel")
}

pub fn load_zero_trailing_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    load_kernel_from_ptx(runtime, zero_trailing_ptx()?, kernel_name)
}
