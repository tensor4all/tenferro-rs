use std::sync::OnceLock;

use cudarc::nvrtc::Ptx;

use super::super::shared::StridedCopyTransform;
use super::helpers::*;
use crate::Result;

pub const STRIDED_COPY_KERNEL_NAME: &str = "strided_copy_kernel";
pub const STRIDED_COPY_TRANSFORM_NONE: i32 = 0;
pub const STRIDED_COPY_TRANSFORM_CONJ: i32 = 1;
pub const STRIDED_COPY_CUDA_SRC: &str = r#"
extern "C" __global__ void strided_copy_kernel(
    const unsigned char* src,
    unsigned char* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    int source_transform,
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
    long long src_index = src_offset;
    long long dst_index = dst_offset;

    for (int axis = 0; axis < ndim; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)dims[axis]);
        remainder /= (unsigned long long)dims[axis];
        src_index += coord * src_strides[axis];
        dst_index += coord * dst_strides[axis];
    }

    unsigned long long src_byte = (unsigned long long)src_index * elem_size;
    unsigned long long dst_byte = (unsigned long long)dst_index * elem_size;
    if (source_transform == 0) {
        for (unsigned long long byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
            dst[dst_byte + byte_idx] = src[src_byte + byte_idx];
        }
    } else if (elem_size == 8ull) {
        const float* src_elem = reinterpret_cast<const float*>(src + src_byte);
        float* dst_elem = reinterpret_cast<float*>(dst + dst_byte);
        dst_elem[0] = src_elem[0];
        dst_elem[1] = -src_elem[1];
    } else if (elem_size == 16ull) {
        const double* src_elem = reinterpret_cast<const double*>(src + src_byte);
        double* dst_elem = reinterpret_cast<double*>(dst + dst_byte);
        dst_elem[0] = src_elem[0];
        dst_elem[1] = -src_elem[1];
    } else {
        for (unsigned long long byte_idx = 0; byte_idx < elem_size; ++byte_idx) {
            dst[dst_byte + byte_idx] = src[src_byte + byte_idx];
        }
    }
}
"#;

pub fn strided_copy_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, STRIDED_COPY_CUDA_SRC, "strided-copy kernel")
}

pub fn strided_copy_transform_code(transform: StridedCopyTransform) -> i32 {
    match transform {
        StridedCopyTransform::None => STRIDED_COPY_TRANSFORM_NONE,
        StridedCopyTransform::Conj => STRIDED_COPY_TRANSFORM_CONJ,
    }
}
