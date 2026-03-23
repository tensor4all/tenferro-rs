use std::{
    any::TypeId,
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::{
    driver::CudaStream,
    nvrtc::{compile_ptx, Ptx},
};
use num_complex::{Complex32, Complex64};

use super::shared::*;
use super::state::CudaRuntime;
use crate::{Error, Result};

pub(super) const STRIDED_COPY_KERNEL_NAME: &str = "strided_copy_kernel";
pub(super) const STRIDED_COPY_TRANSFORM_NONE: i32 = 0;
pub(super) const STRIDED_COPY_TRANSFORM_CONJ: i32 = 1;
pub(super) const STRIDED_COPY_CUDA_SRC: &str = r#"
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

pub(super) const TRIANGULAR_PART_KERNEL_NAME: &str = "triangular_part_kernel";
pub(super) const TRIANGULAR_PART_CUDA_SRC: &str = r#"
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

pub(super) const TRIANGULAR_MERGE_KERNEL_NAME: &str = "triangular_merge_kernel";
pub(super) const TRIANGULAR_MERGE_CUDA_SRC: &str = r#"
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

pub(super) const ZERO_TRAILING_VALIDATE_KERNEL_NAME_F32: &str = "validate_keep_counts_f32";
pub(super) const ZERO_TRAILING_VALIDATE_KERNEL_NAME_F64: &str = "validate_keep_counts_f64";
pub(super) const ZERO_TRAILING_KERNEL_NAME_F32: &str = "zero_trailing_by_counts_f32";
pub(super) const ZERO_TRAILING_KERNEL_NAME_F64: &str = "zero_trailing_by_counts_f64";
pub(super) const ZERO_TRAILING_CUDA_SRC: &str = r#"
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

pub(super) const REAL_UNARY_KERNEL_NAME_F32: &str = "pointwise_unary_real_f32";
pub(super) const REAL_UNARY_KERNEL_NAME_F64: &str = "pointwise_unary_real_f64";
pub(super) const REAL_BINARY_KERNEL_NAME_F32: &str = "pointwise_binary_real_f32";
pub(super) const REAL_BINARY_KERNEL_NAME_F64: &str = "pointwise_binary_real_f64";
pub(super) const REAL_TERNARY_KERNEL_NAME_F32: &str = "pointwise_ternary_real_f32";
pub(super) const REAL_TERNARY_KERNEL_NAME_F64: &str = "pointwise_ternary_real_f64";
pub(super) const REAL_REDUCTION_KERNEL_NAME_F32: &str = "reduce_real_f32";
pub(super) const REAL_REDUCTION_KERNEL_NAME_F64: &str = "reduce_real_f64";
pub(super) const REAL_SCALAR_CUDA_SRC: &str = r#"
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

extern "C" __global__ void pointwise_unary_real_f32(
    const float* src,
    float* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code,
    float alpha,
    float beta
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }
    long long src_idx = linear_offset(idx, dims, src_strides, src_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    float value = src[src_idx];
    float mapped = value;
    if (op_code == 1) {
        mapped = value < 0.0f ? -value : value;
    } else if (op_code == 2) {
        mapped = 1.0f / value;
    } else if (op_code == 3) {
        mapped = logf(value);
    } else if (op_code == 4) {
        mapped = sqrtf(value);
    }
    dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
}

extern "C" __global__ void pointwise_unary_real_f64(
    const double* src,
    double* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code,
    double alpha,
    double beta
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }
    long long src_idx = linear_offset(idx, dims, src_strides, src_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    double value = src[src_idx];
    double mapped = value;
    if (op_code == 1) {
        mapped = value < 0.0 ? -value : value;
    } else if (op_code == 2) {
        mapped = 1.0 / value;
    } else if (op_code == 3) {
        mapped = log(value);
    } else if (op_code == 4) {
        mapped = sqrt(value);
    }
    dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
}

extern "C" __global__ void pointwise_binary_real_f32(
    const float* lhs,
    const float* rhs,
    float* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code,
    float alpha,
    float beta
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
    float x = lhs[lhs_idx];
    float y = rhs[rhs_idx];
    float mapped = x + y;
    if (op_code == 1) {
        mapped = x - y;
    } else if (op_code == 2) {
        mapped = x * y;
    } else if (op_code == 3) {
        mapped = x / y;
    } else if (op_code == 4) {
        mapped = x >= y ? x : y;
    } else if (op_code == 5) {
        mapped = x <= y ? x : y;
    } else if (op_code == 6) {
        mapped = x > y ? 1.0f : 0.0f;
    } else if (op_code == 7) {
        mapped = x >= y ? 1.0f : 0.0f;
    } else if (op_code == 8) {
        mapped = powf(x, y);
    }
    dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
}

extern "C" __global__ void pointwise_binary_real_f64(
    const double* lhs,
    const double* rhs,
    double* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code,
    double alpha,
    double beta
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
    double x = lhs[lhs_idx];
    double y = rhs[rhs_idx];
    double mapped = x + y;
    if (op_code == 1) {
        mapped = x - y;
    } else if (op_code == 2) {
        mapped = x * y;
    } else if (op_code == 3) {
        mapped = x / y;
    } else if (op_code == 4) {
        mapped = x >= y ? x : y;
    } else if (op_code == 5) {
        mapped = x <= y ? x : y;
    } else if (op_code == 6) {
        mapped = x > y ? 1.0 : 0.0;
    } else if (op_code == 7) {
        mapped = x >= y ? 1.0 : 0.0;
    } else if (op_code == 8) {
        mapped = pow(x, y);
    }
    dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
}

extern "C" __global__ void pointwise_ternary_real_f32(
    const float* cond,
    const float* on_true,
    const float* on_false,
    float* dst,
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
    unsigned long long numel,
    int op_code,
    float alpha,
    float beta
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
    float mapped = cond[cond_idx] != 0.0f ? on_true[true_idx] : on_false[false_idx];
    if (op_code == 0) {
        dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
    }
}

extern "C" __global__ void pointwise_ternary_real_f64(
    const double* cond,
    const double* on_true,
    const double* on_false,
    double* dst,
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
    unsigned long long numel,
    int op_code,
    double alpha,
    double beta
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
    double mapped = cond[cond_idx] != 0.0 ? on_true[true_idx] : on_false[false_idx];
    if (op_code == 0) {
        dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
    }
}

extern "C" __global__ void reduce_real_f32(
    const float* input,
    float* output,
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
    unsigned long long reduced_total,
    int op_code,
    float alpha,
    float beta
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

    float acc = op_code == 3 ? 1.0f : 0.0f;
    for (unsigned long long red_idx = 0; red_idx < reduced_total; ++red_idx) {
        unsigned long long red_rem = red_idx;
        long long input_index = base_input;
        for (int axis = 0; axis < reduced_rank; ++axis) {
            long long coord = (long long)(red_rem % (unsigned long long)reduced_dims[axis]);
            red_rem /= (unsigned long long)reduced_dims[axis];
            input_index += coord * input_strides[reduced_axes[axis]];
        }
        float value = input[input_index];
        if (red_idx == 0 && op_code != 3) {
            acc = value;
        } else if (op_code == 1) {
            acc = acc >= value ? acc : value;
        } else if (op_code == 2) {
            acc = acc <= value ? acc : value;
        } else if (op_code == 3) {
            acc *= value;
        } else {
            acc += value;
        }
    }

    output[out_offset] = alpha * acc + beta * output[out_offset];
}

extern "C" __global__ void reduce_real_f64(
    const double* input,
    double* output,
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
    unsigned long long reduced_total,
    int op_code,
    double alpha,
    double beta
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

    double acc = op_code == 3 ? 1.0 : 0.0;
    for (unsigned long long red_idx = 0; red_idx < reduced_total; ++red_idx) {
        unsigned long long red_rem = red_idx;
        long long input_index = base_input;
        for (int axis = 0; axis < reduced_rank; ++axis) {
            long long coord = (long long)(red_rem % (unsigned long long)reduced_dims[axis]);
            red_rem /= (unsigned long long)reduced_dims[axis];
            input_index += coord * input_strides[reduced_axes[axis]];
        }
        double value = input[input_index];
        if (red_idx == 0 && op_code != 3) {
            acc = value;
        } else if (op_code == 1) {
            acc = acc >= value ? acc : value;
        } else if (op_code == 2) {
            acc = acc <= value ? acc : value;
        } else if (op_code == 3) {
            acc *= value;
        } else {
            acc += value;
        }
    }

    output[out_offset] = alpha * acc + beta * output[out_offset];
}
"#;

pub(super) const COMPLEX_REAL_UNARY_KERNEL_NAME_F32: &str = "pointwise_unary_complex32_to_real_f32";
pub(super) const COMPLEX_REAL_UNARY_KERNEL_NAME_F64: &str = "pointwise_unary_complex64_to_real_f64";
pub(super) const COMPLEX_SCALE_KERNEL_NAME_F32: &str = "pointwise_mul_complex32_real_f32";
pub(super) const COMPLEX_SCALE_KERNEL_NAME_F64: &str = "pointwise_mul_complex64_real_f64";

#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelComplex32 {
    re: f32,
    im: f32,
}

#[doc(hidden)]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernelComplex64 {
    re: f64,
    im: f64,
}

unsafe impl cudarc::driver::DeviceRepr for KernelComplex32 {}
unsafe impl cudarc::driver::DeviceRepr for KernelComplex64 {}

impl From<Complex32> for KernelComplex32 {
    fn from(value: Complex32) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

impl From<Complex64> for KernelComplex64 {
    fn from(value: Complex64) -> Self {
        Self {
            re: value.re,
            im: value.im,
        }
    }
}

pub(super) trait ComplexScaleSrc {
    type Real;
}

impl ComplexScaleSrc for Complex32 {
    type Real = f32;
}

impl ComplexScaleSrc for Complex64 {
    type Real = f64;
}

pub(super) const COMPLEX_REAL_CUDA_SRC: &str = r#"
typedef struct { float re; float im; } complex32_t;
typedef struct { double re; double im; } complex64_t;

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

extern "C" __global__ void pointwise_unary_complex32_to_real_f32(
    const complex32_t* src,
    float* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code,
    float alpha,
    float beta
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }
    long long src_idx = linear_offset(idx, dims, src_strides, src_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    complex32_t value = src[src_idx];
    float mapped;
    if (op_code == 0) {
        mapped = sqrtf(value.re * value.re + value.im * value.im);
    } else if (op_code == 1) {
        mapped = value.re;
    } else {
        mapped = value.im;
    }
    dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
}

extern "C" __global__ void pointwise_unary_complex64_to_real_f64(
    const complex64_t* src,
    double* dst,
    const long long* dims,
    const long long* src_strides,
    long long src_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    int op_code,
    double alpha,
    double beta
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= numel) {
        return;
    }
    long long src_idx = linear_offset(idx, dims, src_strides, src_offset, ndim);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    complex64_t value = src[src_idx];
    double mapped;
    if (op_code == 0) {
        mapped = sqrt(value.re * value.re + value.im * value.im);
    } else if (op_code == 1) {
        mapped = value.re;
    } else {
        mapped = value.im;
    }
    dst[dst_idx] = alpha * mapped + beta * dst[dst_idx];
}

__device__ inline complex32_t complex32_add(complex32_t x, complex32_t y) {
    complex32_t out;
    out.re = x.re + y.re;
    out.im = x.im + y.im;
    return out;
}

__device__ inline complex64_t complex64_add(complex64_t x, complex64_t y) {
    complex64_t out;
    out.re = x.re + y.re;
    out.im = x.im + y.im;
    return out;
}

__device__ inline complex32_t complex32_mul(complex32_t x, complex32_t y) {
    complex32_t out;
    out.re = x.re * y.re - x.im * y.im;
    out.im = x.re * y.im + x.im * y.re;
    return out;
}

__device__ inline complex64_t complex64_mul(complex64_t x, complex64_t y) {
    complex64_t out;
    out.re = x.re * y.re - x.im * y.im;
    out.im = x.re * y.im + x.im * y.re;
    return out;
}

extern "C" __global__ void pointwise_scale_complex32_real_f32(
    const complex32_t* lhs,
    const float* rhs,
    complex32_t* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    complex32_t alpha,
    complex32_t beta
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
    complex32_t scaled;
    scaled.re = lhs[lhs_idx].re * rhs[rhs_idx];
    scaled.im = lhs[lhs_idx].im * rhs[rhs_idx];
    dst[dst_idx] = complex32_add(complex32_mul(alpha, scaled), complex32_mul(beta, dst[dst_idx]));
}

extern "C" __global__ void pointwise_scale_complex64_real_f64(
    const complex64_t* lhs,
    const double* rhs,
    complex64_t* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    complex64_t alpha,
    complex64_t beta
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
    complex64_t scaled;
    scaled.re = lhs[lhs_idx].re * rhs[rhs_idx];
    scaled.im = lhs[lhs_idx].im * rhs[rhs_idx];
    dst[dst_idx] = complex64_add(complex64_mul(alpha, scaled), complex64_mul(beta, dst[dst_idx]));
}
"#;

pub(super) const COMPLEX_SCALE_CUDA_SRC: &str = r#"
typedef struct { float re; float im; } complex32_t;
typedef struct { double re; double im; } complex64_t;

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

__device__ complex32_t complex32_mul(complex32_t lhs, complex32_t rhs) {
    complex32_t out;
    out.re = lhs.re * rhs.re - lhs.im * rhs.im;
    out.im = lhs.re * rhs.im + lhs.im * rhs.re;
    return out;
}

__device__ complex32_t complex32_add(complex32_t lhs, complex32_t rhs) {
    complex32_t out;
    out.re = lhs.re + rhs.re;
    out.im = lhs.im + rhs.im;
    return out;
}

__device__ complex64_t complex64_mul(complex64_t lhs, complex64_t rhs) {
    complex64_t out;
    out.re = lhs.re * rhs.re - lhs.im * rhs.im;
    out.im = lhs.re * rhs.im + lhs.im * rhs.re;
    return out;
}

__device__ complex64_t complex64_add(complex64_t lhs, complex64_t rhs) {
    complex64_t out;
    out.re = lhs.re + rhs.re;
    out.im = lhs.im + rhs.im;
    return out;
}

extern "C" __global__ void pointwise_mul_complex32_real_f32(
    const complex32_t* lhs,
    const float* rhs,
    complex32_t* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    complex32_t alpha,
    complex32_t beta
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
    complex32_t lhs_value = lhs[lhs_idx];
    float rhs_value = rhs[rhs_idx];
    complex32_t scaled = {lhs_value.re * rhs_value, lhs_value.im * rhs_value};
    dst[dst_idx] = complex32_add(complex32_mul(alpha, scaled), complex32_mul(beta, dst[dst_idx]));
}

extern "C" __global__ void pointwise_mul_complex64_real_f64(
    const complex64_t* lhs,
    const double* rhs,
    complex64_t* dst,
    const long long* dims,
    const long long* lhs_strides,
    long long lhs_offset,
    const long long* rhs_strides,
    long long rhs_offset,
    const long long* dst_strides,
    long long dst_offset,
    int ndim,
    unsigned long long numel,
    complex64_t alpha,
    complex64_t beta
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
    complex64_t lhs_value = lhs[lhs_idx];
    double rhs_value = rhs[rhs_idx];
    complex64_t scaled = {lhs_value.re * rhs_value, lhs_value.im * rhs_value};
    dst[dst_idx] = complex64_add(complex64_mul(alpha, scaled), complex64_mul(beta, dst[dst_idx]));
}
"#;

pub(super) fn cuda_error(operation: &str, err: impl std::fmt::Debug) -> Error {
    Error::DeviceError(format!("{operation} failed: {err:?}"))
}

pub(super) fn runtime_cache() -> &'static Mutex<HashMap<usize, Arc<CudaRuntime>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaRuntime>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(super) fn strided_copy_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(STRIDED_COPY_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for strided-copy kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

pub(super) fn real_scalar_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(REAL_SCALAR_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for real-scalar kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

pub(super) fn complex_real_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(COMPLEX_REAL_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for complex-real kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

pub(super) fn complex_scale_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(COMPLEX_SCALE_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for complex-scale kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

pub(super) fn zero_trailing_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(ZERO_TRAILING_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for zero-trailing kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

pub(super) fn triangular_part_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(TRIANGULAR_PART_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for triangular-part kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

pub(super) fn triangular_merge_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(TRIANGULAR_MERGE_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for triangular-merge kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

pub(super) fn checked_num_bytes<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::DeviceError("CUDA allocation size overflow".into()))
}

pub(super) fn checked_numel(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidArgument("strided copy numel overflow".into()))
    })
}

pub(super) fn map_keep_count_status(status: i32) -> Result<()> {
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

pub(super) fn contiguous_strides(dims: &[usize], order: ContiguousOrder) -> Result<Vec<isize>> {
    let mut strides = vec![0isize; dims.len()];
    let mut stride = 1isize;

    match order {
        ContiguousOrder::ColumnMajor => {
            for (axis, &dim) in dims.iter().enumerate() {
                strides[axis] = stride;
                let dim = isize::try_from(dim).map_err(|_| {
                    Error::InvalidArgument(format!("dimension {dim} exceeds isize range"))
                })?;
                stride = stride
                    .checked_mul(dim)
                    .ok_or_else(|| Error::InvalidArgument("contiguous stride overflow".into()))?;
            }
        }
        ContiguousOrder::RowMajor => {
            for axis in (0..dims.len()).rev() {
                strides[axis] = stride;
                let dim = isize::try_from(dims[axis]).map_err(|_| {
                    Error::InvalidArgument(format!("dimension {} exceeds isize range", dims[axis]))
                })?;
                stride = stride
                    .checked_mul(dim)
                    .ok_or_else(|| Error::InvalidArgument("contiguous stride overflow".into()))?;
            }
        }
    }

    Ok(strides)
}

pub(super) fn to_i64_vec(values: &[isize], label: &str) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| {
                Error::InvalidArgument(format!("{label} value {value} exceeds i64 range"))
            })
        })
        .collect()
}

pub(super) fn dims_to_i64(dims: &[usize]) -> Result<Vec<i64>> {
    dims.iter()
        .map(|&dim| {
            i64::try_from(dim)
                .map_err(|_| Error::InvalidArgument(format!("dimension {dim} exceeds i64 range")))
        })
        .collect()
}

pub(super) fn axes_to_i32(axes: &[usize], label: &str) -> Result<Vec<i32>> {
    axes.iter()
        .map(|&axis| {
            i32::try_from(axis).map_err(|_| {
                Error::InvalidArgument(format!("{label} axis {axis} exceeds i32 range"))
            })
        })
        .collect()
}

pub(super) fn supports_conj_strided_copy<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<Complex32>() || TypeId::of::<T>() == TypeId::of::<Complex64>()
}

pub(super) fn strided_copy_transform_code(transform: StridedCopyTransform) -> i32 {
    match transform {
        StridedCopyTransform::None => STRIDED_COPY_TRANSFORM_NONE,
        StridedCopyTransform::Conj => STRIDED_COPY_TRANSFORM_CONJ,
    }
}

pub(super) fn unary_opcode(op: RealUnaryOp) -> i32 {
    match op {
        RealUnaryOp::Conj => 0,
        RealUnaryOp::Abs => 1,
        RealUnaryOp::Reciprocal => 2,
        RealUnaryOp::Log => 3,
        RealUnaryOp::Sqrt => 4,
    }
}

pub(super) fn complex_real_opcode(op: ComplexRealUnaryOp) -> i32 {
    match op {
        ComplexRealUnaryOp::Abs => 0,
        ComplexRealUnaryOp::Real => 1,
        ComplexRealUnaryOp::Imag => 2,
    }
}

pub(super) fn binary_opcode(op: RealBinaryOp) -> i32 {
    match op {
        RealBinaryOp::Add => 0,
        RealBinaryOp::Sub => 1,
        RealBinaryOp::Mul => 2,
        RealBinaryOp::Div => 3,
        RealBinaryOp::Maximum => 4,
        RealBinaryOp::Minimum => 5,
        RealBinaryOp::Greater => 6,
        RealBinaryOp::GreaterEqual => 7,
        RealBinaryOp::Pow => 8,
    }
}

pub(super) fn ternary_opcode(op: RealTernaryOp) -> i32 {
    match op {
        RealTernaryOp::Where => 0,
    }
}

pub(super) fn reduction_opcode(op: RealReductionOp) -> i32 {
    match op {
        RealReductionOp::Sum => 0,
        RealReductionOp::Max => 1,
        RealReductionOp::Min => 2,
        RealReductionOp::Prod => 3,
    }
}

pub(super) fn load_real_scalar_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(cudarc::driver::CudaFunction, Arc<CudaStream>)> {
    runtime.bind_context()?;
    let ctx = runtime.context();
    let module = ctx
        .load_module(real_scalar_ptx()?)
        .map_err(|err| cuda_error("CUDA module load", err))?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|err| cuda_error("CUDA load function", err))?;
    Ok((kernel, ctx.default_stream()))
}

pub(super) fn load_complex_real_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(cudarc::driver::CudaFunction, Arc<CudaStream>)> {
    runtime.bind_context()?;
    let ctx = runtime.context();
    let module = ctx
        .load_module(complex_real_ptx()?)
        .map_err(|err| cuda_error("CUDA module load", err))?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|err| cuda_error("CUDA load function", err))?;
    Ok((kernel, ctx.default_stream()))
}

pub(super) fn load_complex_scale_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(cudarc::driver::CudaFunction, Arc<CudaStream>)> {
    runtime.bind_context()?;
    let ctx = runtime.context();
    let module = ctx
        .load_module(complex_scale_ptx()?)
        .map_err(|err| cuda_error("CUDA module load", err))?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|err| cuda_error("CUDA load function", err))?;
    Ok((kernel, ctx.default_stream()))
}

pub(super) fn load_zero_trailing_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(cudarc::driver::CudaFunction, Arc<CudaStream>)> {
    runtime.bind_context()?;
    let ctx = runtime.context();
    let module = ctx
        .load_module(zero_trailing_ptx()?)
        .map_err(|err| cuda_error("CUDA module load", err))?;
    let kernel = module
        .load_function(kernel_name)
        .map_err(|err| cuda_error("CUDA load function", err))?;
    Ok((kernel, ctx.default_stream()))
}

pub(super) fn validate_pointwise_rank(
    dims: &[usize],
    lhs_or_src_strides: &[isize],
    rhs_strides: Option<&[isize]>,
    dst_strides: &[isize],
) -> Result<()> {
    if dims.len() != lhs_or_src_strides.len() || dims.len() != dst_strides.len() {
        return Err(Error::InvalidArgument(format!(
            "pointwise rank mismatch: dims={} src={} dst={}",
            dims.len(),
            lhs_or_src_strides.len(),
            dst_strides.len()
        )));
    }
    if let Some(rhs_strides) = rhs_strides {
        if dims.len() != rhs_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "pointwise rank mismatch: dims={} rhs={}",
                dims.len(),
                rhs_strides.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn validate_ternary_pointwise_rank(
    dims: &[usize],
    cond_strides: &[isize],
    true_strides: &[isize],
    false_strides: &[isize],
    dst_strides: &[isize],
) -> Result<()> {
    if dims.len() != cond_strides.len()
        || dims.len() != true_strides.len()
        || dims.len() != false_strides.len()
        || dims.len() != dst_strides.len()
    {
        return Err(Error::InvalidArgument(format!(
            "pointwise ternary rank mismatch: dims={} cond={} true={} false={} dst={}",
            dims.len(),
            cond_strides.len(),
            true_strides.len(),
            false_strides.len(),
            dst_strides.len()
        )));
    }
    Ok(())
}

pub(super) fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}
