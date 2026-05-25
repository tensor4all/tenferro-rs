use std::sync::OnceLock;

use cudarc::nvrtc::Ptx;

use super::helpers::compile_ptx_once;
use crate::Result;

pub const TRIANGULAR_PART_KERNEL_NAME: &str = "triangular_part_kernel";
pub const TRIANGULAR_PART_CUDA_SRC: &str = r#"
extern "C" __global__ void triangular_part_kernel(
    const unsigned char* src,
    unsigned char* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    long long diagonal,
    int half,
    unsigned long long elem_size,
    unsigned long long numel
) {
    unsigned long long linear_idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (linear_idx >= numel) {
        return;
    }

    unsigned long long remainder = linear_idx;
    long long src_index = src_offset;
    long long dst_index = dst_offset;
    long long row = 0;
    long long col = 0;

    for (int axis = 0; axis < ndim; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)dims[axis]);
        remainder /= (unsigned long long)dims[axis];
        src_index += coord * src_strides[axis];
        dst_index += coord * dst_strides[axis];
        if (axis == 0) {
            row = coord;
        } else if (axis == 1) {
            col = coord;
        }
    }

    bool keep = half == 0 ? ((col - row) <= diagonal) : ((col - row) >= diagonal);
    unsigned long long src_byte = (unsigned long long)src_index * elem_size;
    unsigned long long dst_byte = (unsigned long long)dst_index * elem_size;
    if (keep) {
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

pub const TRIANGULAR_MERGE_KERNEL_NAME: &str = "triangular_merge_kernel";
pub const TRIANGULAR_MERGE_CUDA_SRC: &str = r#"
extern "C" __global__ void triangular_merge_kernel(
    const unsigned char* lower_src,
    const unsigned char* upper_src,
    unsigned char* dst,
    const long long* dims,
    const long long* lower_strides,
    long long lower_offset,
    const long long* upper_strides,
    long long upper_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long elem_size,
    unsigned long long numel
) {
    unsigned long long linear_idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (linear_idx >= numel) {
        return;
    }

    unsigned long long remainder = linear_idx;
    long long lower_index = lower_offset;
    long long upper_index = upper_offset;
    long long dst_index = dst_offset;
    long long row = 0;
    long long col = 0;

    for (int axis = 0; axis < ndim; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)dims[axis]);
        remainder /= (unsigned long long)dims[axis];
        lower_index += coord * lower_strides[axis];
        upper_index += coord * upper_strides[axis];
        dst_index += coord * dst_strides[axis];
        if (axis == 0) {
            row = coord;
        } else if (axis == 1) {
            col = coord;
        }
    }

    const unsigned char* src = row > col ? lower_src : upper_src;
    unsigned long long src_byte =
        (unsigned long long)(row > col ? lower_index : upper_index) * elem_size;
    unsigned long long dst_byte = (unsigned long long)dst_index * elem_size;
    for (unsigned long long byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
        dst[dst_byte + byte_idx] = src[src_byte + byte_idx];
    }
}
"#;

pub fn triangular_part_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, TRIANGULAR_PART_CUDA_SRC, "triangular-part kernel")
}

pub fn triangular_merge_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, TRIANGULAR_MERGE_CUDA_SRC, "triangular-merge kernel")
}
