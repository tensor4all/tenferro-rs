use std::sync::{Arc, OnceLock};

use cudarc::{
    driver::{CudaFunction, CudaStream},
    nvrtc::Ptx,
};

use super::super::state::CudaRuntime;
use super::helpers::{compile_ptx_once, load_kernel_from_ptx};
use crate::Result;

pub const METADATA_GENERATE_IOTA_I32_KERNEL_NAME: &str = "metadata_generate_iota_i32";
pub const METADATA_BINARY_I32_I32_KERNEL_NAME: &str = "metadata_binary_i32_i32";
pub const METADATA_BINARY_I32_BOOL_KERNEL_NAME: &str = "metadata_binary_i32_bool";
pub const METADATA_BINARY_BOOL_BOOL_KERNEL_NAME: &str = "metadata_binary_bool_bool";
pub const METADATA_TERNARY_I32_KERNEL_NAME: &str = "metadata_where_i32";
pub const METADATA_TERNARY_BOOL_KERNEL_NAME: &str = "metadata_where_bool";
pub const METADATA_REDUCE_SUM_I32_KERNEL_NAME: &str = "metadata_reduce_sum_i32";
pub const METADATA_REDUCE_SUM_BOOL_KERNEL_NAME: &str = "metadata_reduce_sum_bool";
pub const METADATA_REDUCE_ALL_BOOL_KERNEL_NAME: &str = "metadata_reduce_all_bool";
pub const METADATA_REDUCE_ANY_BOOL_KERNEL_NAME: &str = "metadata_reduce_any_bool";

pub const METADATA_SCALAR_CUDA_SRC: &str = r#"
__device__ long long linear_offset(
    unsigned long long linear_idx,
    const long long* dims,
    const long long* strides,
    long long base_offset,
    int ndim
) {
    long long offset = base_offset;
    unsigned long long remainder = linear_idx;
    for (int axis = 0; axis < ndim; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)dims[axis]);
        remainder /= (unsigned long long)dims[axis];
        offset += coord * strides[axis];
    }
    return offset;
}

extern "C" __global__ void metadata_generate_iota_i32(
    int* dst,
    const long long* dims,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = (int)idx;
}

extern "C" __global__ void metadata_binary_i32_bool(
    const int* lhs,
    const int* rhs,
    unsigned char* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    long long lhs_idx = linear_offset(idx, dims, lhs_strides, lhs_offset, ndim);
    long long rhs_idx = linear_offset(idx, dims, rhs_strides, rhs_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    int value = lhs[lhs_idx];
    int other = rhs[rhs_idx];
    unsigned char mapped = op_code == 0 ? (value == other) : (value != other);
    dst[dst_idx] = mapped;
}

extern "C" __global__ void metadata_binary_i32_i32(
    const int* lhs,
    const int* rhs,
    int* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    long long lhs_idx = linear_offset(idx, dims, lhs_strides, lhs_offset, ndim);
    long long rhs_idx = linear_offset(idx, dims, rhs_strides, rhs_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    int value = lhs[lhs_idx];
    int other = rhs[rhs_idx];
    int mapped = 0;
    switch (op_code) {
        case 0:
            mapped = (value == other) ? 1 : 0;
            break;
        case 1:
            mapped = (value != other) ? 1 : 0;
            break;
        case 2:
            mapped = value + other;
            break;
        case 3:
            mapped = value - other;
            break;
        case 4:
            mapped = value * other;
            break;
    }
    dst[dst_idx] = mapped;
}

extern "C" __global__ void metadata_binary_bool_bool(
    const unsigned char* lhs,
    const unsigned char* rhs,
    unsigned char* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    long long lhs_idx = linear_offset(idx, dims, lhs_strides, lhs_offset, ndim);
    long long rhs_idx = linear_offset(idx, dims, rhs_strides, rhs_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    unsigned char value = lhs[lhs_idx];
    unsigned char other = rhs[rhs_idx];
    unsigned char mapped = 0;
    switch (op_code) {
        case 0:
            mapped = (value == other) ? 1 : 0;
            break;
        case 1:
            mapped = (value != other) ? 1 : 0;
            break;
        case 2:
            mapped = (value != 0 && other != 0) ? 1 : 0;
            break;
    }
    dst[dst_idx] = mapped;
}

extern "C" __global__ void metadata_where_i32(
    const unsigned char* cond,
    const int* on_true,
    const int* on_false,
    int* dst,
    const long long* dims,
    const long long* cond_strides,
    long long cond_offset,
    const long long* true_strides,
    long long true_offset,
    const long long* false_strides,
    long long false_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    long long cond_idx = linear_offset(idx, dims, cond_strides, cond_offset, ndim);
    long long true_idx = linear_offset(idx, dims, true_strides, true_offset, ndim);
    long long false_idx = linear_offset(idx, dims, false_strides, false_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = cond[cond_idx] != 0 ? on_true[true_idx] : on_false[false_idx];
}

extern "C" __global__ void metadata_where_bool(
    const unsigned char* cond,
    const unsigned char* on_true,
    const unsigned char* on_false,
    unsigned char* dst,
    const long long* dims,
    const long long* cond_strides,
    long long cond_offset,
    const long long* true_strides,
    long long true_offset,
    const long long* false_strides,
    long long false_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }

    long long cond_idx = linear_offset(idx, dims, cond_strides, cond_offset, ndim);
    long long true_idx = linear_offset(idx, dims, true_strides, true_offset, ndim);
    long long false_idx = linear_offset(idx, dims, false_strides, false_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = cond[cond_idx] != 0 ? on_true[true_idx] : on_false[false_idx];
}

extern "C" __global__ void metadata_reduce_sum_i32(
    const int* input,
    int* output,
    const long long* input_strides,
    long long input_offset,
    const long long* output_dims,
    const long long* output_strides,
    long long output_offset,
    const int* kept_axes,
    int kept_rank,
    const int* reduced_axes,
    const long long* reduced_dims,
    int reduced_rank,
    unsigned long long output_numel,
    unsigned long long reduced_total
) {
    unsigned long long out_idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (out_idx >= output_numel) {
        return;
    }

    unsigned long long remainder = out_idx;
    long long out_offset = output_offset;
    long long base_input = input_offset;
    for (int axis = 0; axis < kept_rank; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)output_dims[axis]);
        remainder /= (unsigned long long)output_dims[axis];
        out_offset += coord * output_strides[axis];
        base_input += coord * input_strides[kept_axes[axis]];
    }

    int acc = 0;
    for (unsigned long long red_idx = 0; red_idx < reduced_total; ++red_idx) {
        unsigned long long red_rem = red_idx;
        long long input_index = base_input;
        for (int axis = 0; axis < reduced_rank; ++axis) {
            long long coord = (long long)(red_rem % (unsigned long long)reduced_dims[axis]);
            red_rem /= (unsigned long long)reduced_dims[axis];
            input_index += coord * input_strides[reduced_axes[axis]];
        }
        acc += input[input_index];
    }

    output[out_offset] = acc;
}

extern "C" __global__ void metadata_reduce_sum_bool(
    const unsigned char* input,
    int* output,
    const long long* input_strides,
    long long input_offset,
    const long long* output_dims,
    const long long* output_strides,
    long long output_offset,
    const int* kept_axes,
    int kept_rank,
    const int* reduced_axes,
    const long long* reduced_dims,
    int reduced_rank,
    unsigned long long output_numel,
    unsigned long long reduced_total
) {
    unsigned long long out_idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (out_idx >= output_numel) {
        return;
    }

    unsigned long long remainder = out_idx;
    long long out_offset = output_offset;
    long long base_input = input_offset;
    for (int axis = 0; axis < kept_rank; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)output_dims[axis]);
        remainder /= (unsigned long long)output_dims[axis];
        out_offset += coord * output_strides[axis];
        base_input += coord * input_strides[kept_axes[axis]];
    }

    int acc = 0;
    for (unsigned long long red_idx = 0; red_idx < reduced_total; ++red_idx) {
        unsigned long long red_rem = red_idx;
        long long input_index = base_input;
        for (int axis = 0; axis < reduced_rank; ++axis) {
            long long coord = (long long)(red_rem % (unsigned long long)reduced_dims[axis]);
            red_rem /= (unsigned long long)reduced_dims[axis];
            input_index += coord * input_strides[reduced_axes[axis]];
        }
        acc += input[input_index] != 0 ? 1 : 0;
    }

    output[out_offset] = acc;
}

extern "C" __global__ void metadata_reduce_all_bool(
    const unsigned char* input,
    unsigned char* output,
    const long long* input_strides,
    long long input_offset,
    const long long* output_dims,
    const long long* output_strides,
    long long output_offset,
    const int* kept_axes,
    int kept_rank,
    const int* reduced_axes,
    const long long* reduced_dims,
    int reduced_rank,
    unsigned long long output_numel,
    unsigned long long reduced_total
) {
    unsigned long long out_idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (out_idx >= output_numel) {
        return;
    }

    unsigned long long remainder = out_idx;
    long long out_offset = output_offset;
    long long base_input = input_offset;
    for (int axis = 0; axis < kept_rank; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)output_dims[axis]);
        remainder /= (unsigned long long)output_dims[axis];
        out_offset += coord * output_strides[axis];
        base_input += coord * input_strides[kept_axes[axis]];
    }

    unsigned char acc = 1;
    for (unsigned long long red_idx = 0; red_idx < reduced_total; ++red_idx) {
        unsigned long long red_rem = red_idx;
        long long input_index = base_input;
        for (int axis = 0; axis < reduced_rank; ++axis) {
            long long coord = (long long)(red_rem % (unsigned long long)reduced_dims[axis]);
            red_rem /= (unsigned long long)reduced_dims[axis];
            input_index += coord * input_strides[reduced_axes[axis]];
        }
        acc = (input[input_index] != 0) ? acc : 0;
    }

    output[out_offset] = acc;
}

extern "C" __global__ void metadata_reduce_any_bool(
    const unsigned char* input,
    unsigned char* output,
    const long long* input_strides,
    long long input_offset,
    const long long* output_dims,
    const long long* output_strides,
    long long output_offset,
    const int* kept_axes,
    int kept_rank,
    const int* reduced_axes,
    const long long* reduced_dims,
    int reduced_rank,
    unsigned long long output_numel,
    unsigned long long reduced_total
) {
    unsigned long long out_idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (out_idx >= output_numel) {
        return;
    }

    unsigned long long remainder = out_idx;
    long long out_offset = output_offset;
    long long base_input = input_offset;
    for (int axis = 0; axis < kept_rank; ++axis) {
        long long coord = (long long)(remainder % (unsigned long long)output_dims[axis]);
        remainder /= (unsigned long long)output_dims[axis];
        out_offset += coord * output_strides[axis];
        base_input += coord * input_strides[kept_axes[axis]];
    }

    unsigned char acc = 0;
    for (unsigned long long red_idx = 0; red_idx < reduced_total; ++red_idx) {
        unsigned long long red_rem = red_idx;
        long long input_index = base_input;
        for (int axis = 0; axis < reduced_rank; ++axis) {
            long long coord = (long long)(red_rem % (unsigned long long)reduced_dims[axis]);
            red_rem /= (unsigned long long)reduced_dims[axis];
            input_index += coord * input_strides[reduced_axes[axis]];
        }
        acc = (input[input_index] != 0) ? 1 : acc;
    }

    output[out_offset] = acc;
}
"#;

pub fn metadata_scalar_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, METADATA_SCALAR_CUDA_SRC, "metadata-scalar kernel")
}

pub fn load_metadata_scalar_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    load_kernel_from_ptx(runtime, metadata_scalar_ptx()?, kernel_name)
}
