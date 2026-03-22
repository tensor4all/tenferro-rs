use std::{
    any::TypeId,
    collections::HashMap,
    ffi::c_void,
    marker::PhantomData,
    mem::MaybeUninit,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::{
    driver::{CudaContext, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{compile_ptx, Ptx},
    runtime::result as cuda_result,
};
use num_complex::{Complex32, Complex64};

use crate::{Error, Result};

const STRIDED_COPY_KERNEL_NAME: &str = "strided_copy_kernel";
const STRIDED_COPY_TRANSFORM_NONE: i32 = 0;
const STRIDED_COPY_TRANSFORM_CONJ: i32 = 1;
const STRIDED_COPY_CUDA_SRC: &str = r#"
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

const TRIANGULAR_PART_KERNEL_NAME: &str = "triangular_part_kernel";
const TRIANGULAR_PART_CUDA_SRC: &str = r#"
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

const TRIANGULAR_MERGE_KERNEL_NAME: &str = "triangular_merge_kernel";
const TRIANGULAR_MERGE_CUDA_SRC: &str = r#"
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

const ZERO_TRAILING_VALIDATE_KERNEL_NAME_F32: &str = "validate_keep_counts_f32";
const ZERO_TRAILING_VALIDATE_KERNEL_NAME_F64: &str = "validate_keep_counts_f64";
const ZERO_TRAILING_KERNEL_NAME_F32: &str = "zero_trailing_by_counts_f32";
const ZERO_TRAILING_KERNEL_NAME_F64: &str = "zero_trailing_by_counts_f64";
const ZERO_TRAILING_CUDA_SRC: &str = r#"
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

const REAL_UNARY_KERNEL_NAME_F32: &str = "pointwise_unary_real_f32";
const REAL_UNARY_KERNEL_NAME_F64: &str = "pointwise_unary_real_f64";
const REAL_BINARY_KERNEL_NAME_F32: &str = "pointwise_binary_real_f32";
const REAL_BINARY_KERNEL_NAME_F64: &str = "pointwise_binary_real_f64";
const REAL_TERNARY_KERNEL_NAME_F32: &str = "pointwise_ternary_real_f32";
const REAL_TERNARY_KERNEL_NAME_F64: &str = "pointwise_ternary_real_f64";
const REAL_REDUCTION_KERNEL_NAME_F32: &str = "reduce_real_f32";
const REAL_REDUCTION_KERNEL_NAME_F64: &str = "reduce_real_f64";
const REAL_SCALAR_CUDA_SRC: &str = r#"
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

const COMPLEX_REAL_UNARY_KERNEL_NAME_F32: &str = "pointwise_unary_complex32_to_real_f32";
const COMPLEX_REAL_UNARY_KERNEL_NAME_F64: &str = "pointwise_unary_complex64_to_real_f64";
const COMPLEX_SCALE_KERNEL_NAME_F32: &str = "pointwise_mul_complex32_real_f32";
const COMPLEX_SCALE_KERNEL_NAME_F64: &str = "pointwise_mul_complex64_real_f64";

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

trait ComplexScaleSrc {
    type Real;
}

impl ComplexScaleSrc for Complex32 {
    type Real = f32;
}

impl ComplexScaleSrc for Complex64 {
    type Real = f64;
}

const COMPLEX_REAL_CUDA_SRC: &str = r#"
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

const COMPLEX_SCALE_CUDA_SRC: &str = r#"
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

fn cuda_error(operation: &str, err: impl std::fmt::Debug) -> Error {
    Error::DeviceError(format!("{operation} failed: {err:?}"))
}

fn runtime_cache() -> &'static Mutex<HashMap<usize, Arc<CudaRuntime>>> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Arc<CudaRuntime>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn strided_copy_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(STRIDED_COPY_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for strided-copy kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

fn real_scalar_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(REAL_SCALAR_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for real-scalar kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

fn complex_real_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(COMPLEX_REAL_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for complex-real kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

fn complex_scale_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(COMPLEX_SCALE_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for complex-scale kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

fn zero_trailing_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(ZERO_TRAILING_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for zero-trailing kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

fn triangular_part_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(TRIANGULAR_PART_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for triangular-part kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

fn triangular_merge_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(TRIANGULAR_MERGE_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for triangular-merge kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}

fn checked_num_bytes<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::DeviceError("CUDA allocation size overflow".into()))
}

fn checked_numel(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| Error::InvalidArgument("strided copy numel overflow".into()))
    })
}

fn map_keep_count_status(status: i32) -> Result<()> {
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

fn contiguous_strides(dims: &[usize], order: ContiguousOrder) -> Result<Vec<isize>> {
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

fn to_i64_vec(values: &[isize], label: &str) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| {
                Error::InvalidArgument(format!("{label} value {value} exceeds i64 range"))
            })
        })
        .collect()
}

fn dims_to_i64(dims: &[usize]) -> Result<Vec<i64>> {
    dims.iter()
        .map(|&dim| {
            i64::try_from(dim)
                .map_err(|_| Error::InvalidArgument(format!("dimension {dim} exceeds i64 range")))
        })
        .collect()
}

fn axes_to_i32(axes: &[usize], label: &str) -> Result<Vec<i32>> {
    axes.iter()
        .map(|&axis| {
            i32::try_from(axis).map_err(|_| {
                Error::InvalidArgument(format!("{label} axis {axis} exceeds i32 range"))
            })
        })
        .collect()
}

fn supports_conj_strided_copy<T: 'static>() -> bool {
    TypeId::of::<T>() == TypeId::of::<Complex32>() || TypeId::of::<T>() == TypeId::of::<Complex64>()
}

fn strided_copy_transform_code(transform: StridedCopyTransform) -> i32 {
    match transform {
        StridedCopyTransform::None => STRIDED_COPY_TRANSFORM_NONE,
        StridedCopyTransform::Conj => STRIDED_COPY_TRANSFORM_CONJ,
    }
}

fn unary_opcode(op: RealUnaryOp) -> i32 {
    match op {
        RealUnaryOp::Conj => 0,
        RealUnaryOp::Abs => 1,
        RealUnaryOp::Reciprocal => 2,
        RealUnaryOp::Log => 3,
        RealUnaryOp::Sqrt => 4,
    }
}

fn complex_real_opcode(op: ComplexRealUnaryOp) -> i32 {
    match op {
        ComplexRealUnaryOp::Abs => 0,
        ComplexRealUnaryOp::Real => 1,
        ComplexRealUnaryOp::Imag => 2,
    }
}

fn binary_opcode(op: RealBinaryOp) -> i32 {
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

fn ternary_opcode(op: RealTernaryOp) -> i32 {
    match op {
        RealTernaryOp::Where => 0,
    }
}

fn reduction_opcode(op: RealReductionOp) -> i32 {
    match op {
        RealReductionOp::Sum => 0,
        RealReductionOp::Max => 1,
        RealReductionOp::Min => 2,
        RealReductionOp::Prod => 3,
    }
}

fn load_real_scalar_kernel(
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

fn load_complex_real_kernel(
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

fn load_complex_scale_kernel(
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

fn load_zero_trailing_kernel(
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

fn validate_pointwise_rank(
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

fn validate_ternary_pointwise_rank(
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

fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), std::mem::size_of_val(slice)) }
}

/// Destination layout for materialized contiguous buffers.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::ContiguousOrder;
///
/// let order = ContiguousOrder::ColumnMajor;
/// assert_eq!(order, ContiguousOrder::ColumnMajor);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContiguousOrder {
    /// Column-major / Fortran order.
    ColumnMajor,
    /// Row-major / C order.
    RowMajor,
}

/// Which triangular half to keep when materializing a matrix or batched matrix.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::TriangularHalf;
///
/// let half = TriangularHalf::Lower;
/// assert_eq!(half, TriangularHalf::Lower);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularHalf {
    /// Keep the lower triangle.
    Lower,
    /// Keep the upper triangle.
    Upper,
}

impl TriangularHalf {
    fn as_i32(self) -> i32 {
        match self {
            TriangularHalf::Lower => 0,
            TriangularHalf::Upper => 1,
        }
    }
}

/// Low-level specification for copying a strided source layout into a destination layout.
///
/// The `dims`, `src_strides`, and `dst_strides` arrays describe the same logical tensor
/// shape. Offsets are measured in elements, not bytes.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
///
/// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
/// assert_eq!(spec.dims(), &[4, 2, 3]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedCopySpec {
    dims: Vec<usize>,
    src_strides: Vec<isize>,
    src_offset: isize,
    dst_strides: Vec<isize>,
    dst_offset: isize,
}

impl StridedCopySpec {
    /// Build a strided-copy spec whose destination is contiguous in the requested order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
    ///
    /// let spec = StridedCopySpec::to_contiguous(&[2, 3], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(spec.dst_strides(), &[1, 2]);
    /// ```
    pub fn to_contiguous(
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        order: ContiguousOrder,
    ) -> Result<Self> {
        if dims.len() != src_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "strided copy rank mismatch: dims={} src_strides={}",
                dims.len(),
                src_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            src_offset,
            dst_strides: contiguous_strides(dims, order)?,
            dst_offset: 0,
        })
    }

    /// Returns the logical dimensions.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
    ///
    /// let spec = StridedCopySpec::to_contiguous(&[2, 3], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(spec.dims(), &[2, 3]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the destination strides in elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{ContiguousOrder, StridedCopySpec};
    ///
    /// let spec = StridedCopySpec::to_contiguous(&[2, 3], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(spec.dst_strides(), &[1, 2]);
    /// ```
    pub fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }
}

/// Source-side transforms supported by the Layer 0 strided-copy helper.
///
/// Phase 1 supports plain copy and complex conjugation only.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::StridedCopyTransform;
///
/// assert_eq!(StridedCopyTransform::None, StridedCopyTransform::None);
/// assert_eq!(StridedCopyTransform::Conj, StridedCopyTransform::Conj);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StridedCopyTransform {
    None,
    Conj,
}

/// Low-level specification for materializing a triangular matrix view on the GPU.
///
/// The first two dimensions are interpreted as the matrix rows and columns.
/// Any remaining dimensions are treated as batch dimensions and copied
/// elementwise. The output shape matches the input shape.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
///
/// let spec = TriangularPartSpec::new(
///     &[3, 2, 4],
///     &[1, 3, 6],
///     0,
///     &[1, 3, 6],
///     0,
///     -1,
///     TriangularHalf::Lower,
/// ).unwrap();
/// assert_eq!(spec.diagonal(), -1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriangularPartSpec {
    dims: Vec<usize>,
    src_strides: Vec<isize>,
    src_offset: isize,
    dst_strides: Vec<isize>,
    dst_offset: isize,
    diagonal: isize,
    half: TriangularHalf,
}

impl TriangularPartSpec {
    /// Build a triangular-copy specification.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(
    ///     &[2, 3],
    ///     &[1, 2],
    ///     0,
    ///     &[1, 2],
    ///     0,
    ///     0,
    ///     TriangularHalf::Upper,
    /// ).unwrap();
    /// assert_eq!(spec.half(), TriangularHalf::Upper);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        diagonal: isize,
        half: TriangularHalf,
    ) -> Result<Self> {
        if dims.len() < 2 {
            return Err(Error::InvalidArgument(
                "triangular copy requires rank >= 2".into(),
            ));
        }
        if dims.len() != src_strides.len() || dims.len() != dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "triangular copy rank mismatch: dims={} src_strides={} dst_strides={}",
                dims.len(),
                src_strides.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            src_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
            diagonal,
            half,
        })
    }

    /// Returns the triangular diagonal offset.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 2], &[1, 2], 0, &[1, 2], 0, 1, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.diagonal(), 1);
    /// ```
    pub fn diagonal(&self) -> isize {
        self.diagonal
    }

    /// Returns which half is preserved.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 2], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Upper).unwrap();
    /// assert_eq!(spec.half(), TriangularHalf::Upper);
    /// ```
    pub fn half(&self) -> TriangularHalf {
        self.half
    }

    /// Returns the logical dimensions described by this triangular-copy spec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.dims(), &[2, 3]);
    /// ```
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the source strides described by this triangular-copy spec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.src_strides(), &[1, 2]);
    /// ```
    pub fn src_strides(&self) -> &[isize] {
        &self.src_strides
    }

    /// Returns the source element offset.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 4, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.src_offset(), 4);
    /// ```
    pub fn src_offset(&self) -> isize {
        self.src_offset
    }

    /// Returns the destination strides described by this triangular-copy spec.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 0, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.dst_strides(), &[1, 2]);
    /// ```
    pub fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }

    /// Returns the destination element offset.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{TriangularHalf, TriangularPartSpec};
    ///
    /// let spec = TriangularPartSpec::new(&[2, 3], &[1, 2], 0, &[1, 2], 5, 0, TriangularHalf::Lower).unwrap();
    /// assert_eq!(spec.dst_offset(), 5);
    /// ```
    pub fn dst_offset(&self) -> isize {
        self.dst_offset
    }
}

/// Low-level specification for merging a strict-lower source and an upper-with-diagonal source.
///
/// The logical output shape is `dims`. The first source is read when `row > col`,
/// and the second source is read otherwise. The first two dimensions are
/// interpreted as matrix rows and columns; trailing dimensions are batch dims.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::TriangularMergeSpec;
///
/// let spec = TriangularMergeSpec::new(
///     &[3, 2, 4],
///     &[1, 3, 6],
///     0,
///     &[1, 3, 6],
///     0,
///     &[1, 3, 6],
///     0,
/// ).unwrap();
/// assert_eq!(spec.dims(), &[3, 2, 4]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriangularMergeSpec {
    dims: Vec<usize>,
    lower_strides: Vec<isize>,
    lower_offset: isize,
    upper_strides: Vec<isize>,
    upper_offset: isize,
    dst_strides: Vec<isize>,
    dst_offset: isize,
}

impl TriangularMergeSpec {
    /// Build a triangular-merge specification.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::TriangularMergeSpec;
    ///
    /// let spec = TriangularMergeSpec::new(
    ///     &[2, 3],
    ///     &[1, 2],
    ///     0,
    ///     &[1, 2],
    ///     0,
    ///     &[1, 2],
    ///     0,
    /// ).unwrap();
    /// assert_eq!(spec.dims(), &[2, 3]);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: &[usize],
        lower_strides: &[isize],
        lower_offset: isize,
        upper_strides: &[isize],
        upper_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<Self> {
        if dims.len() < 2 {
            return Err(Error::InvalidArgument(
                "triangular merge requires rank >= 2".into(),
            ));
        }
        if dims.len() != lower_strides.len()
            || dims.len() != upper_strides.len()
            || dims.len() != dst_strides.len()
        {
            return Err(Error::InvalidArgument(format!(
                "triangular merge rank mismatch: dims={} lower_strides={} upper_strides={} dst_strides={}",
                dims.len(),
                lower_strides.len(),
                upper_strides.len(),
                dst_strides.len()
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            lower_strides: lower_strides.to_vec(),
            lower_offset,
            upper_strides: upper_strides.to_vec(),
            upper_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
        })
    }

    /// Returns the logical output dimensions.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the strict-lower source strides.
    pub fn lower_strides(&self) -> &[isize] {
        &self.lower_strides
    }

    /// Returns the strict-lower source offset.
    pub fn lower_offset(&self) -> isize {
        self.lower_offset
    }

    /// Returns the upper-with-diagonal source strides.
    pub fn upper_strides(&self) -> &[isize] {
        &self.upper_strides
    }

    /// Returns the upper-with-diagonal source offset.
    pub fn upper_offset(&self) -> isize {
        self.upper_offset
    }

    /// Returns the destination strides.
    pub fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }

    /// Returns the destination offset.
    pub fn dst_offset(&self) -> isize {
        self.dst_offset
    }
}

/// Real unary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealUnaryOp;
///
/// let op = RealUnaryOp::Abs;
/// assert_eq!(op, RealUnaryOp::Abs);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealUnaryOp {
    Conj,
    Abs,
    Reciprocal,
    Log,
    Sqrt,
}

/// Complex-to-real unary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::ComplexRealUnaryOp;
///
/// let op = ComplexRealUnaryOp::Abs;
/// assert_eq!(op, ComplexRealUnaryOp::Abs);
/// let op = ComplexRealUnaryOp::Real;
/// assert_eq!(op, ComplexRealUnaryOp::Real);
/// let op = ComplexRealUnaryOp::Imag;
/// assert_eq!(op, ComplexRealUnaryOp::Imag);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexRealUnaryOp {
    Abs,
    Real,
    Imag,
}

/// Real binary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealBinaryOp;
///
/// let op = RealBinaryOp::Add;
/// assert_eq!(op, RealBinaryOp::Add);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Greater,
    GreaterEqual,
    Pow,
}

/// Real ternary operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealTernaryOp;
///
/// let op = RealTernaryOp::Where;
/// assert_eq!(op, RealTernaryOp::Where);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealTernaryOp {
    Where,
}

/// Real reduction operations exposed by the Layer 0 CUDA runtime.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RealReductionOp;
///
/// let op = RealReductionOp::Sum;
/// assert_eq!(op, RealReductionOp::Sum);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealReductionOp {
    Sum,
    Max,
    Min,
    Prod,
}

/// Low-level specification for zero-filling trailing regions by batch-local keep counts.
///
/// The trailing batch dims are `dims[structural_rank..]`. `axis` is interpreted
/// within the structural prefix `[0, structural_rank)`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec;
///
/// let spec = ZeroTrailingByCountsSpec::new(
///     &[2, 2, 2],
///     &[1, 2, 4],
///     0,
///     &[1, 2, 4],
///     0,
///     &[1],
///     0,
///     1,
///     2,
/// ).unwrap();
/// assert_eq!(spec.axis(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZeroTrailingByCountsSpec {
    dims: Vec<usize>,
    src_strides: Vec<isize>,
    src_offset: isize,
    dst_strides: Vec<isize>,
    dst_offset: isize,
    keep_count_strides: Vec<isize>,
    keep_count_offset: isize,
    axis: usize,
    structural_rank: usize,
}

impl ZeroTrailingByCountsSpec {
    /// Build a zero-trailing specification.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::ZeroTrailingByCountsSpec;
    ///
    /// let spec = ZeroTrailingByCountsSpec::new(
    ///     &[3, 2, 2],
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1],
    ///     0,
    ///     0,
    ///     2,
    /// ).unwrap();
    /// assert_eq!(spec.structural_rank(), 2);
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        dst_strides: &[isize],
        dst_offset: isize,
        keep_count_strides: &[isize],
        keep_count_offset: isize,
        axis: usize,
        structural_rank: usize,
    ) -> Result<Self> {
        if dims.len() != src_strides.len() || dims.len() != dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "zero-trailing rank mismatch: dims={} src_strides={} dst_strides={}",
                dims.len(),
                src_strides.len(),
                dst_strides.len()
            )));
        }
        if structural_rank == 0 || structural_rank > dims.len() {
            return Err(Error::InvalidArgument(format!(
                "structural_rank {structural_rank} must be in 1..={}",
                dims.len()
            )));
        }
        if axis >= structural_rank {
            return Err(Error::InvalidArgument(format!(
                "axis {axis} out of range for structural_rank {structural_rank}"
            )));
        }
        let batch_rank = dims.len() - structural_rank;
        if keep_count_strides.len() != batch_rank {
            return Err(Error::InvalidArgument(format!(
                "keep_count_strides rank {} does not match batch rank {}",
                keep_count_strides.len(),
                batch_rank
            )));
        }

        Ok(Self {
            dims: dims.to_vec(),
            src_strides: src_strides.to_vec(),
            src_offset,
            dst_strides: dst_strides.to_vec(),
            dst_offset,
            keep_count_strides: keep_count_strides.to_vec(),
            keep_count_offset,
            axis,
            structural_rank,
        })
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    #[cfg(test)]
    pub(crate) fn src_strides(&self) -> &[isize] {
        &self.src_strides
    }

    #[cfg(test)]
    pub(crate) fn src_offset(&self) -> isize {
        self.src_offset
    }

    #[cfg(test)]
    pub(crate) fn keep_count_strides(&self) -> &[isize] {
        &self.keep_count_strides
    }

    #[cfg(test)]
    pub(crate) fn keep_count_offset(&self) -> isize {
        self.keep_count_offset
    }

    #[cfg(test)]
    pub(crate) fn axis(&self) -> usize {
        self.axis
    }

    #[cfg(test)]
    pub(crate) fn structural_rank(&self) -> usize {
        self.structural_rank
    }
}

trait RuntimeRealScalar: cudarc::driver::DeviceRepr + Copy + 'static {
    const UNARY_KERNEL_NAME: &'static str;
    const BINARY_KERNEL_NAME: &'static str;
    const TERNARY_KERNEL_NAME: &'static str;
    const REDUCTION_KERNEL_NAME: &'static str;
}

/// Marker trait for keep-count scalars supported by CUDA trailing zero-fill.
///
/// Implemented for `f32` and `f64`.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime::RuntimeKeepCountScalar;
///
/// fn needs_counts<T: RuntimeKeepCountScalar>() {}
/// needs_counts::<f32>();
/// needs_counts::<f64>();
/// ```
pub trait RuntimeKeepCountScalar: cudarc::driver::DeviceRepr + Copy + 'static {
    const VALIDATE_KERNEL_NAME: &'static str;
    const ZERO_TRAILING_KERNEL_NAME: &'static str;
}

impl RuntimeRealScalar for f32 {
    const UNARY_KERNEL_NAME: &'static str = REAL_UNARY_KERNEL_NAME_F32;
    const BINARY_KERNEL_NAME: &'static str = REAL_BINARY_KERNEL_NAME_F32;
    const TERNARY_KERNEL_NAME: &'static str = REAL_TERNARY_KERNEL_NAME_F32;
    const REDUCTION_KERNEL_NAME: &'static str = REAL_REDUCTION_KERNEL_NAME_F32;
}

impl RuntimeRealScalar for f64 {
    const UNARY_KERNEL_NAME: &'static str = REAL_UNARY_KERNEL_NAME_F64;
    const BINARY_KERNEL_NAME: &'static str = REAL_BINARY_KERNEL_NAME_F64;
    const TERNARY_KERNEL_NAME: &'static str = REAL_TERNARY_KERNEL_NAME_F64;
    const REDUCTION_KERNEL_NAME: &'static str = REAL_REDUCTION_KERNEL_NAME_F64;
}

impl RuntimeKeepCountScalar for f32 {
    const VALIDATE_KERNEL_NAME: &'static str = ZERO_TRAILING_VALIDATE_KERNEL_NAME_F32;
    const ZERO_TRAILING_KERNEL_NAME: &'static str = ZERO_TRAILING_KERNEL_NAME_F32;
}

impl RuntimeKeepCountScalar for f64 {
    const VALIDATE_KERNEL_NAME: &'static str = ZERO_TRAILING_VALIDATE_KERNEL_NAME_F64;
    const ZERO_TRAILING_KERNEL_NAME: &'static str = ZERO_TRAILING_KERNEL_NAME_F64;
}

/// Shared CUDA runtime handle for one device ordinal.
///
/// The handle retains the CUDA primary context and exposes low-level memory
/// allocation and copy primitives that higher-level crates can reuse.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// assert_eq!(runtime.device_id(), 0);
/// ```
#[derive(Debug)]
pub struct CudaRuntime {
    context: Arc<CudaContext>,
}

impl CudaRuntime {
    fn new(device_id: usize) -> Result<Arc<Self>> {
        let context =
            CudaContext::new(device_id).map_err(|err| cuda_error("CUDA device init", err))?;
        Ok(Arc::new(Self { context }))
    }

    fn bind_context(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|err| cuda_error("CUDA context bind", err))
    }

    /// Returns the CUDA device ordinal this runtime is bound to.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// assert_eq!(runtime.device_id(), 0);
    /// ```
    pub fn device_id(&self) -> usize {
        self.context.ordinal()
    }

    /// Returns a clone of the shared CUDA context handle.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ctx = runtime.context();
    /// assert_eq!(ctx.ordinal(), 0);
    /// ```
    pub fn context(&self) -> Arc<CudaContext> {
        Arc::clone(&self.context)
    }

    /// Allocates a raw device pointer for `len` elements of `T`.
    ///
    /// # Safety
    ///
    /// The returned pointer must eventually be passed to [`CudaRuntime::free_raw`]
    /// on the same runtime.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub fn alloc_raw<T>(&self, len: usize) -> Result<*mut T> {
        self.bind_context()?;
        if len == 0 {
            return Ok(std::ptr::null_mut());
        }

        let ptr = unsafe { cuda_result::malloc_sync(checked_num_bytes::<T>(len)?) }
            .map_err(|err| cuda_error("cudaMalloc", err))?;
        Ok(ptr.cast::<T>())
    }

    /// Frees a raw device pointer previously allocated by [`CudaRuntime::alloc_raw`].
    ///
    /// # Safety
    ///
    /// `ptr` must either be null or a live allocation returned by this runtime.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub unsafe fn free_raw<T>(&self, ptr: *mut T) -> Result<()> {
        self.bind_context()?;
        if ptr.is_null() {
            return Ok(());
        }

        unsafe { cuda_result::free_sync(ptr.cast::<c_void>()) }
            .map_err(|err| cuda_error("cudaFree", err))
    }

    /// Copies a host slice into a raw device allocation.
    ///
    /// # Safety
    ///
    /// `dst` must point to a live device allocation holding at least `dst_len` elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.copy_htod_raw(&[1.0_f32, 2.0, 3.0, 4.0], ptr, 4).unwrap();
    ///     runtime.free_raw(ptr).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_htod_raw<T>(&self, src: &[T], dst: *mut T, dst_len: usize) -> Result<()> {
        if src.len() != dst_len {
            return Err(Error::InvalidArgument(format!(
                "host/device length mismatch: src={} dst={dst_len}",
                src.len()
            )));
        }

        self.bind_context()?;
        if src.is_empty() {
            return Ok(());
        }

        unsafe { cuda_result::memcpy_htod_sync(dst.cast::<c_void>(), as_byte_slice(src)) }
            .map_err(|err| cuda_error("cudaMemcpyHtoD", err))
    }

    /// Copies a raw device allocation into a host vector.
    ///
    /// # Safety
    ///
    /// `src` must point to a live device allocation holding at least `len` elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let ptr = runtime.alloc_raw::<f32>(4).unwrap();
    /// let host = unsafe { runtime.copy_dtoh_raw(ptr, 4).unwrap() };
    /// assert_eq!(host.len(), 4);
    /// unsafe { runtime.free_raw(ptr).unwrap(); }
    /// ```
    pub unsafe fn copy_dtoh_raw<T>(&self, src: *const T, len: usize) -> Result<Vec<T>> {
        self.bind_context()?;
        if len == 0 {
            return Ok(Vec::new());
        }

        let num_bytes = checked_num_bytes::<T>(len)?;
        let mut host = Vec::<MaybeUninit<T>>::with_capacity(len);
        unsafe { host.set_len(len) };
        let host_bytes =
            unsafe { std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), num_bytes) };

        unsafe { cuda_result::memcpy_dtoh_sync(host_bytes, src.cast::<c_void>()) }
            .map_err(|err| cuda_error("cudaMemcpyDtoH", err))?;

        let ptr = host.as_mut_ptr().cast::<T>();
        let len = host.len();
        let cap = host.capacity();
        std::mem::forget(host);
        Ok(unsafe { Vec::from_raw_parts(ptr, len, cap) })
    }

    /// Copies one raw device allocation into another.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations holding at least `len` elements.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<f32>(4).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.copy_dtod_raw(src, dst, 4).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_dtod_raw<T>(&self, src: *const T, dst: *mut T, len: usize) -> Result<()> {
        self.bind_context()?;
        if len == 0 {
            return Ok(());
        }

        unsafe {
            cuda_result::memcpy_dtod_sync(
                dst.cast::<c_void>(),
                src.cast::<c_void>(),
                checked_num_bytes::<T>(len)?,
            )
        }
        .map_err(|err| cuda_error("cudaMemcpyDtoD", err))
    }

    /// Launches the generic strided-copy kernel from a raw device source to a raw device destination.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with `spec`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<f32>(24).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// unsafe {
    ///     runtime.copy_strided_raw(src, dst, &spec).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_strided_raw<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
    ) -> Result<()> {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let stream =
            self.launch_strided_copy_raw_impl(src, dst, spec, STRIDED_COPY_TRANSFORM_NONE)?;
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the generic strided-copy kernel with a source-side transform from a raw device source to a raw device destination.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with `spec`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec, StridedCopyTransform};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<num_complex::Complex64>(24).unwrap();
    /// let dst = runtime.alloc_raw::<num_complex::Complex64>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// unsafe {
    ///     runtime.copy_strided_raw_with_transform(src, dst, &spec, StridedCopyTransform::Conj).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn copy_strided_raw_with_transform<T: 'static>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
        transform: StridedCopyTransform,
    ) -> Result<()> {
        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let stream = self.launch_strided_copy_raw_with_transform(src, dst, spec, transform)?;
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    unsafe fn launch_strided_copy_raw_impl<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
        source_transform: i32,
    ) -> Result<Arc<CudaStream>> {
        if spec.dims.len() != spec.src_strides.len() || spec.dims.len() != spec.dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "strided copy rank mismatch: dims={} src_strides={} dst_strides={}",
                spec.dims.len(),
                spec.src_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;

        self.bind_context()?;
        let ctx = self.context();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(strided_copy_ptx()?)
            .map_err(|err| cuda_error("CUDA module load", err))?;
        let kernel = module
            .load_function(STRIDED_COPY_KERNEL_NAME)
            .map_err(|err| cuda_error("CUDA load function", err))?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("strided copy rank exceeds i32 range".into()))?;
        let src_offset = i64::try_from(spec.src_offset)
            .map_err(|_| Error::InvalidArgument("source offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("strided copy numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("strided copy currently requires len <= u32::MAX".into())
        })?;
        let src_ptr = src as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&src_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&src_strides_dev)
                .arg(&src_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&source_transform)
                .arg(&ndim)
                .arg(&elem_size)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA strided-copy kernel launch", err))?;
        }

        Ok(stream)
    }

    unsafe fn launch_strided_copy_raw<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
    ) -> Result<Arc<CudaStream>> {
        self.launch_strided_copy_raw_impl(src, dst, spec, STRIDED_COPY_TRANSFORM_NONE)
    }

    unsafe fn launch_strided_copy_raw_with_transform<T: 'static>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &StridedCopySpec,
        transform: StridedCopyTransform,
    ) -> Result<Arc<CudaStream>> {
        if matches!(transform, StridedCopyTransform::Conj) && !supports_conj_strided_copy::<T>() {
            return Err(Error::InvalidArgument(
                "strided copy conj transform requires Complex32 or Complex64 element type".into(),
            ));
        }

        self.launch_strided_copy_raw_impl(src, dst, spec, strided_copy_transform_code(transform))
    }

    /// Launches the keep-count-driven trailing zero-fill kernel on raw device allocations.
    ///
    /// # Safety
    ///
    /// `src`, `dst`, and `keep_counts` must point to live device allocations compatible
    /// with `spec`.
    pub unsafe fn zero_trailing_by_counts_raw<T, R>(
        &self,
        src: *const T,
        dst: *mut T,
        keep_counts: *const R,
        spec: &ZeroTrailingByCountsSpec,
    ) -> Result<()>
    where
        R: RuntimeKeepCountScalar,
    {
        if spec.dims.len() != spec.src_strides.len() || spec.dims.len() != spec.dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "zero-trailing rank mismatch: dims={} src_strides={} dst_strides={}",
                spec.dims.len(),
                spec.src_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        let batch_dims = &spec.dims[spec.structural_rank..];
        let count_numel = checked_numel(batch_dims)?;
        let batch_rank = i32::try_from(batch_dims.len())
            .map_err(|_| Error::InvalidArgument("batch rank exceeds i32 range".into()))?;
        let axis_len = i64::try_from(spec.dims[spec.axis])
            .map_err(|_| Error::InvalidArgument("axis length exceeds i64 range".into()))?;
        let keep_count_offset = i64::try_from(spec.keep_count_offset)
            .map_err(|_| Error::InvalidArgument("keep-count offset exceeds i64 range".into()))?;
        let src_offset = i64::try_from(spec.src_offset)
            .map_err(|_| Error::InvalidArgument("source offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("zero-trailing rank exceeds i32 range".into()))?;
        let axis = i32::try_from(spec.axis)
            .map_err(|_| Error::InvalidArgument("axis exceeds i32 range".into()))?;
        let structural_rank = i32::try_from(spec.structural_rank)
            .map_err(|_| Error::InvalidArgument("structural_rank exceeds i32 range".into()))?;
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("zero-trailing numel exceeds u64 range".into()))?;
        let count_numel_u64 = u64::try_from(count_numel)
            .map_err(|_| Error::InvalidArgument("keep-count numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("zero-trailing currently requires len <= u32::MAX".into())
        })?;
        let count_numel_u32 = u32::try_from(count_numel).map_err(|_| {
            Error::InvalidArgument(
                "keep-count validation currently requires len <= u32::MAX".into(),
            )
        })?;

        let (validate_kernel, stream) = load_zero_trailing_kernel(self, R::VALIDATE_KERNEL_NAME)?;
        let batch_dims_dev = stream
            .clone_htod(&dims_to_i64(batch_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD batch dims", err))?;
        let keep_count_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.keep_count_strides, "keep-count stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD keep-count strides", err))?;
        let status = self.alloc::<i32>(1)?;
        self.copy_htod(&[0i32], &status)?;
        let keep_counts_ptr = keep_counts as u64;
        let status_ptr = status.device_ptr() as u64;
        let validate_config = LaunchConfig {
            grid_dim: (count_numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&validate_kernel)
                .arg(&keep_counts_ptr)
                .arg(&batch_dims_dev)
                .arg(&keep_count_strides_dev)
                .arg(&keep_count_offset)
                .arg(&batch_rank)
                .arg(&axis_len)
                .arg(&count_numel_u64)
                .arg(&status_ptr)
                .launch(validate_config)
                .map_err(|err| cuda_error("CUDA keep-count validation launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))?;
        let status_host = self.copy_dtoh(&status)?;
        map_keep_count_status(status_host[0])?;

        let (zero_kernel, stream) = load_zero_trailing_kernel(self, R::ZERO_TRAILING_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let src_ptr = src as u64;
        let dst_ptr = dst as u64;
        let launch_config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&zero_kernel)
                .arg(&src_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&src_strides_dev)
                .arg(&src_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&keep_counts_ptr)
                .arg(&keep_count_strides_dev)
                .arg(&keep_count_offset)
                .arg(&ndim)
                .arg(&axis)
                .arg(&structural_rank)
                .arg(&elem_size)
                .arg(&numel_u64)
                .launch(launch_config)
                .map_err(|err| cuda_error("CUDA zero-trailing launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the triangular-copy kernel on raw device allocations.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with `spec`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, TriangularHalf, TriangularPartSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc_raw::<f32>(24).unwrap();
    /// let dst = runtime.alloc_raw::<f32>(24).unwrap();
    /// let spec = TriangularPartSpec::new(&[3, 2, 4], &[1, 3, 6], 0, &[1, 3, 6], 0, 0, TriangularHalf::Lower).unwrap();
    /// unsafe {
    ///     runtime.triangular_part_raw(src, dst, &spec).unwrap();
    ///     runtime.free_raw(src).unwrap();
    ///     runtime.free_raw(dst).unwrap();
    /// }
    /// ```
    pub unsafe fn triangular_part_raw<T>(
        &self,
        src: *const T,
        dst: *mut T,
        spec: &TriangularPartSpec,
    ) -> Result<()> {
        if spec.dims.len() != spec.src_strides.len() || spec.dims.len() != spec.dst_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "triangular copy rank mismatch: dims={} src_strides={} dst_strides={}",
                spec.dims.len(),
                spec.src_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        self.bind_context()?;
        let ctx = self.context();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(triangular_part_ptx()?)
            .map_err(|err| cuda_error("CUDA module load", err))?;
        let kernel = module
            .load_function(TRIANGULAR_PART_KERNEL_NAME)
            .map_err(|err| cuda_error("CUDA load function", err))?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len())
            .map_err(|_| Error::InvalidArgument("triangular copy rank exceeds i32 range".into()))?;
        let src_offset = i64::try_from(spec.src_offset)
            .map_err(|_| Error::InvalidArgument("source offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let diagonal = i64::try_from(spec.diagonal)
            .map_err(|_| Error::InvalidArgument("diagonal exceeds i64 range".into()))?;
        let half = spec.half.as_i32();
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular copy numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular copy currently requires len <= u32::MAX".into())
        })?;
        let src_ptr = src as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&src_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&src_strides_dev)
                .arg(&src_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&diagonal)
                .arg(&half)
                .arg(&elem_size)
                .arg(&numel_u64)
                .launch(config)
                .map_err(|err| cuda_error("CUDA triangular-part kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the triangular-merge kernel on raw device allocations.
    ///
    /// # Safety
    ///
    /// `lower_src`, `upper_src`, and `dst` must point to live device allocations
    /// compatible with `spec`.
    pub unsafe fn triangular_merge_raw<T>(
        &self,
        lower_src: *const T,
        upper_src: *const T,
        dst: *mut T,
        spec: &TriangularMergeSpec,
    ) -> Result<()> {
        if spec.dims.len() != spec.lower_strides.len()
            || spec.dims.len() != spec.upper_strides.len()
            || spec.dims.len() != spec.dst_strides.len()
        {
            return Err(Error::InvalidArgument(format!(
                "triangular merge rank mismatch: dims={} lower_strides={} upper_strides={} dst_strides={}",
                spec.dims.len(),
                spec.lower_strides.len(),
                spec.upper_strides.len(),
                spec.dst_strides.len()
            )));
        }

        let numel = checked_numel(&spec.dims)?;
        if numel == 0 {
            return Ok(());
        }

        self.bind_context()?;
        let ctx = self.context();
        let stream = ctx.default_stream();
        let module = ctx
            .load_module(triangular_merge_ptx()?)
            .map_err(|err| cuda_error("CUDA module load", err))?;
        let kernel = module
            .load_function(TRIANGULAR_MERGE_KERNEL_NAME)
            .map_err(|err| cuda_error("CUDA load function", err))?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(&spec.dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let lower_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.lower_strides, "lower stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD lower strides", err))?;
        let upper_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.upper_strides, "upper stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD upper strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(&spec.dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(spec.dims.len()).map_err(|_| {
            Error::InvalidArgument("triangular merge rank exceeds i32 range".into())
        })?;
        let lower_offset = i64::try_from(spec.lower_offset)
            .map_err(|_| Error::InvalidArgument("lower offset exceeds i64 range".into()))?;
        let upper_offset = i64::try_from(spec.upper_offset)
            .map_err(|_| Error::InvalidArgument("upper offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(spec.dst_offset)
            .map_err(|_| Error::InvalidArgument("destination offset exceeds i64 range".into()))?;
        let elem_size = u64::try_from(std::mem::size_of::<T>())
            .map_err(|_| Error::InvalidArgument("element size exceeds u64 range".into()))?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular merge numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("triangular merge currently requires len <= u32::MAX".into())
        })?;
        let lower_src_ptr = lower_src as u64;
        let upper_src_ptr = upper_src as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        stream
            .launch_builder(&kernel)
            .arg(&lower_src_ptr)
            .arg(&upper_src_ptr)
            .arg(&dst_ptr)
            .arg(&dims_dev)
            .arg(&lower_strides_dev)
            .arg(&lower_offset)
            .arg(&upper_strides_dev)
            .arg(&upper_offset)
            .arg(&dst_strides_dev)
            .arg(&dst_offset)
            .arg(&ndim)
            .arg(&elem_size)
            .arg(&numel_u64)
            .launch(config)
            .map_err(|err| cuda_error("CUDA triangular-merge kernel launch", err))?;
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_unary_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealUnaryOp,
        alpha: T,
        src: *const T,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: T,
        dst: *mut T,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_pointwise_rank(dims, src_strides, None, dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_real_scalar_kernel(self, T::UNARY_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("pointwise unary rank exceeds i32 range".into()))?;
        let src_offset = i64::try_from(src_offset).map_err(|_| {
            Error::InvalidArgument("pointwise unary source offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("pointwise unary destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise unary numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise unary currently requires len <= u32::MAX".into())
        })?;
        let opcode = unary_opcode(op);
        let src_ptr = src as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&src_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&src_strides_dev)
                .arg(&src_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real unary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_binary_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealBinaryOp,
        alpha: T,
        lhs: *const T,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const T,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: T,
        dst: *mut T,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_pointwise_rank(dims, lhs_strides, Some(rhs_strides), dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_real_scalar_kernel(self, T::BINARY_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let lhs_strides_dev = stream
            .clone_htod(&to_i64_vec(lhs_strides, "lhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD lhs strides", err))?;
        let rhs_strides_dev = stream
            .clone_htod(&to_i64_vec(rhs_strides, "rhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD rhs strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len()).map_err(|_| {
            Error::InvalidArgument("pointwise binary rank exceeds i32 range".into())
        })?;
        let lhs_offset = i64::try_from(lhs_offset).map_err(|_| {
            Error::InvalidArgument("pointwise binary lhs offset exceeds i64 range".into())
        })?;
        let rhs_offset = i64::try_from(rhs_offset).map_err(|_| {
            Error::InvalidArgument("pointwise binary rhs offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("pointwise binary destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise binary numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise binary currently requires len <= u32::MAX".into())
        })?;
        let opcode = binary_opcode(op);
        let lhs_ptr = lhs as u64;
        let rhs_ptr = rhs as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&lhs_ptr)
                .arg(&rhs_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&lhs_strides_dev)
                .arg(&lhs_offset)
                .arg(&rhs_strides_dev)
                .arg(&rhs_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real binary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_ternary_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealTernaryOp,
        alpha: T,
        cond: *const T,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const T,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const T,
        false_strides: &[isize],
        false_offset: isize,
        beta: T,
        dst: *mut T,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        validate_ternary_pointwise_rank(
            dims,
            cond_strides,
            true_strides,
            false_strides,
            dst_strides,
        )?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_real_scalar_kernel(self, T::TERNARY_KERNEL_NAME)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let cond_strides_dev = stream
            .clone_htod(&to_i64_vec(cond_strides, "cond stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD cond strides", err))?;
        let true_strides_dev = stream
            .clone_htod(&to_i64_vec(true_strides, "true stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD true strides", err))?;
        let false_strides_dev = stream
            .clone_htod(&to_i64_vec(false_strides, "false stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD false strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len()).map_err(|_| {
            Error::InvalidArgument("pointwise ternary rank exceeds i32 range".into())
        })?;
        let cond_offset = i64::try_from(cond_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary condition offset exceeds i64 range".into())
        })?;
        let true_offset = i64::try_from(true_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary true offset exceeds i64 range".into())
        })?;
        let false_offset = i64::try_from(false_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary false offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("pointwise ternary destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise ternary numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise ternary currently requires len <= u32::MAX".into())
        })?;
        let opcode = ternary_opcode(op);
        let cond_ptr = cond as u64;
        let true_ptr = on_true as u64;
        let false_ptr = on_false as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&cond_ptr)
                .arg(&true_ptr)
                .arg(&false_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&cond_strides_dev)
                .arg(&cond_offset)
                .arg(&true_strides_dev)
                .arg(&true_offset)
                .arg(&false_strides_dev)
                .arg(&false_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real ternary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn reduce_real_raw_impl<T: RuntimeRealScalar>(
        &self,
        op: RealReductionOp,
        alpha: T,
        input: *const T,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: T,
        output: *mut T,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        if input_dims.len() != input_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "reduction rank mismatch: input dims={} input strides={}",
                input_dims.len(),
                input_strides.len()
            )));
        }
        if output_dims.len() != output_strides.len() {
            return Err(Error::InvalidArgument(format!(
                "reduction rank mismatch: output dims={} output strides={}",
                output_dims.len(),
                output_strides.len()
            )));
        }
        if output_dims.len() != kept_axes.len() {
            return Err(Error::InvalidArgument(format!(
                "reduction kept-axis mismatch: output dims={} kept_axes={}",
                output_dims.len(),
                kept_axes.len()
            )));
        }
        for (output_axis, &input_axis) in kept_axes.iter().enumerate() {
            let Some(&expected_dim) = input_dims.get(input_axis) else {
                return Err(Error::InvalidArgument(format!(
                    "reduction kept axis {input_axis} out of bounds"
                )));
            };
            if output_dims[output_axis] != expected_dim {
                return Err(Error::InvalidArgument(format!(
                    "reduction output dim mismatch at axis {output_axis}: expected {expected_dim}, got {}",
                    output_dims[output_axis]
                )));
            }
        }

        let output_numel = checked_numel(output_dims)?;
        if output_numel == 0 {
            return Ok(());
        }

        let reduced_dims: Vec<usize> = reduced_axes
            .iter()
            .map(|&axis| {
                input_dims.get(axis).copied().ok_or_else(|| {
                    Error::InvalidArgument(format!("reduction axis {axis} out of bounds"))
                })
            })
            .collect::<Result<_>>()?;
        let reduced_total = checked_numel(&reduced_dims)?;
        if reduced_total == 0 && matches!(op, RealReductionOp::Max | RealReductionOp::Min) {
            return Err(Error::InvalidArgument(
                "extrema reduction requires a non-empty reduction domain".into(),
            ));
        }
        let (kernel, stream) = load_real_scalar_kernel(self, T::REDUCTION_KERNEL_NAME)?;
        let input_strides_dev = stream
            .clone_htod(&to_i64_vec(input_strides, "input stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD input strides", err))?;
        let output_dims_dev = stream
            .clone_htod(&dims_to_i64(output_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD output dims", err))?;
        let output_strides_dev = stream
            .clone_htod(&to_i64_vec(output_strides, "output stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD output strides", err))?;
        let kept_axes_dev = stream
            .clone_htod(&axes_to_i32(kept_axes, "kept")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD kept axes", err))?;
        let reduced_axes_dev = stream
            .clone_htod(&axes_to_i32(reduced_axes, "reduced")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD reduced axes", err))?;
        let reduced_dims_dev = stream
            .clone_htod(&dims_to_i64(&reduced_dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD reduced dims", err))?;
        let kept_rank = i32::try_from(kept_axes.len())
            .map_err(|_| Error::InvalidArgument("reduction kept rank exceeds i32 range".into()))?;
        let reduced_rank = i32::try_from(reduced_axes.len()).map_err(|_| {
            Error::InvalidArgument("reduction reduced rank exceeds i32 range".into())
        })?;
        let input_offset = i64::try_from(input_offset).map_err(|_| {
            Error::InvalidArgument("reduction input offset exceeds i64 range".into())
        })?;
        let output_offset = i64::try_from(output_offset).map_err(|_| {
            Error::InvalidArgument("reduction output offset exceeds i64 range".into())
        })?;
        let output_numel_u64 = u64::try_from(output_numel).map_err(|_| {
            Error::InvalidArgument("reduction output numel exceeds u64 range".into())
        })?;
        let output_numel_u32 = u32::try_from(output_numel).map_err(|_| {
            Error::InvalidArgument("reduction currently requires len <= u32::MAX".into())
        })?;
        let reduced_total_u64 = u64::try_from(reduced_total).map_err(|_| {
            Error::InvalidArgument("reduction reduced total exceeds u64 range".into())
        })?;
        let opcode = reduction_opcode(op);
        let input_ptr = input as u64;
        let output_ptr = output as u64;
        let config = LaunchConfig {
            grid_dim: (output_numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&input_ptr)
                .arg(&output_ptr)
                .arg(&input_strides_dev)
                .arg(&input_offset)
                .arg(&output_dims_dev)
                .arg(&output_strides_dev)
                .arg(&output_offset)
                .arg(&kept_axes_dev)
                .arg(&kept_rank)
                .arg(&reduced_axes_dev)
                .arg(&reduced_dims_dev)
                .arg(&reduced_rank)
                .arg(&output_numel_u64)
                .arg(&reduced_total_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA real reduction kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the Layer 0 real unary kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_real_f32_raw(
    ///         RealUnaryOp::Abs,
    ///         1.0,
    ///         src.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_unary_real_f32_raw(
        &self,
        op: RealUnaryOp,
        alpha: f32,
        src: *const f32,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_real_raw_impl(
            op,
            alpha,
            src,
            dims,
            src_strides,
            src_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real unary kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_real_f64_raw(
    ///         RealUnaryOp::Abs,
    ///         1.0,
    ///         src.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_unary_real_f64_raw(
        &self,
        op: RealUnaryOp,
        alpha: f64,
        src: *const f64,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_real_raw_impl(
            op,
            alpha,
            src,
            dims,
            src_strides,
            src_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 complex-by-real pointwise multiply kernel for `Complex32 * f32` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex32;
    /// use tenferro_device::cuda::runtime::{self, KernelComplex32};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<Complex32>(4).unwrap();
    /// let rhs = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<Complex32>(4).unwrap();
    /// let alpha = KernelComplex32 { re: 1.0, im: 0.0 };
    /// let beta = KernelComplex32 { re: 0.0, im: 0.0 };
    /// unsafe {
    ///     runtime.pointwise_scale_complex32_real_f32_raw(
    ///         alpha,
    ///         lhs.device_ptr().cast_const(),
    ///         rhs.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///         beta,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_scale_complex32_real_f32_raw(
        &self,
        alpha: KernelComplex32,
        lhs: *const Complex32,
        rhs: *const f32,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Complex32,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: KernelComplex32,
    ) -> Result<()> {
        self.pointwise_scale_complex_real_raw_impl(
            COMPLEX_SCALE_KERNEL_NAME_F32,
            alpha,
            lhs,
            rhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            dst,
            dst_strides,
            dst_offset,
            beta,
        )
    }

    /// Launches the Layer 0 complex-by-real pointwise multiply kernel for `Complex64 * f64` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro_device::cuda::runtime::{self, KernelComplex64};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<Complex64>(4).unwrap();
    /// let rhs = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<Complex64>(4).unwrap();
    /// let alpha = KernelComplex64 { re: 1.0, im: 0.0 };
    /// let beta = KernelComplex64 { re: 0.0, im: 0.0 };
    /// unsafe {
    ///     runtime.pointwise_scale_complex64_real_f64_raw(
    ///         alpha,
    ///         lhs.device_ptr().cast_const(),
    ///         rhs.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///         beta,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_scale_complex64_real_f64_raw(
        &self,
        alpha: KernelComplex64,
        lhs: *const Complex64,
        rhs: *const f64,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Complex64,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: KernelComplex64,
    ) -> Result<()> {
        self.pointwise_scale_complex_real_raw_impl(
            COMPLEX_SCALE_KERNEL_NAME_F64,
            alpha,
            lhs,
            rhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            dst,
            dst_strides,
            dst_offset,
            beta,
        )
    }

    fn pointwise_scale_complex_real_raw_impl<Dst, Src>(
        &self,
        kernel_name: &str,
        alpha: Dst,
        lhs: *const Src,
        rhs: *const <Src as ComplexScaleSrc>::Real,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs_strides: &[isize],
        rhs_offset: isize,
        dst: *mut Src,
        dst_strides: &[isize],
        dst_offset: isize,
        beta: Dst,
    ) -> Result<()>
    where
        Dst: cudarc::driver::DeviceRepr,
        Src: ComplexScaleSrc,
    {
        validate_pointwise_rank(dims, lhs_strides, Some(rhs_strides), dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_complex_real_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let lhs_strides_dev = stream
            .clone_htod(&to_i64_vec(lhs_strides, "lhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD lhs strides", err))?;
        let rhs_strides_dev = stream
            .clone_htod(&to_i64_vec(rhs_strides, "rhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD rhs strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("complex scale rank exceeds i32 range".into()))?;
        let lhs_offset = i64::try_from(lhs_offset).map_err(|_| {
            Error::InvalidArgument("complex scale lhs offset exceeds i64 range".into())
        })?;
        let rhs_offset = i64::try_from(rhs_offset).map_err(|_| {
            Error::InvalidArgument("complex scale rhs offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("complex scale destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("complex scale numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("complex scale currently requires len <= u32::MAX".into())
        })?;
        let lhs_ptr = lhs as u64;
        let rhs_ptr = rhs as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&lhs_ptr)
                .arg(&rhs_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&lhs_strides_dev)
                .arg(&lhs_offset)
                .arg(&rhs_strides_dev)
                .arg(&rhs_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA complex scale kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_unary_complex_real_raw_impl<Dst, Src>(
        &self,
        kernel_name: &str,
        op: ComplexRealUnaryOp,
        alpha: Dst,
        src: *const Src,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: Dst,
        dst: *mut Dst,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>
    where
        Dst: RuntimeRealScalar,
        Src: Copy + 'static,
    {
        validate_pointwise_rank(dims, src_strides, None, dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_complex_real_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let src_strides_dev = stream
            .clone_htod(&to_i64_vec(src_strides, "src stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD src strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("pointwise unary rank exceeds i32 range".into()))?;
        let src_offset = i64::try_from(src_offset).map_err(|_| {
            Error::InvalidArgument("pointwise unary source offset exceeds i64 range".into())
        })?;
        let dst_offset = i64::try_from(dst_offset).map_err(|_| {
            Error::InvalidArgument("pointwise unary destination offset exceeds i64 range".into())
        })?;
        let numel_u64 = u64::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise unary numel exceeds u64 range".into())
        })?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise unary currently requires len <= u32::MAX".into())
        })?;
        let opcode = complex_real_opcode(op);
        let src_ptr = src as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&src_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&src_strides_dev)
                .arg(&src_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&opcode)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA complex-real unary kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    fn pointwise_complex_scale_raw_impl<KernelComplex, Complex>(
        &self,
        kernel_name: &str,
        alpha: KernelComplex,
        lhs: *const Complex,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const <Complex as ComplexScaleSrc>::Real,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: KernelComplex,
        dst: *mut Complex,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()>
    where
        KernelComplex: Copy + cudarc::driver::DeviceRepr + 'static,
        Complex: ComplexScaleSrc + Copy + 'static,
    {
        validate_pointwise_rank(dims, lhs_strides, Some(rhs_strides), dst_strides)?;
        let numel = checked_numel(dims)?;
        if numel == 0 {
            return Ok(());
        }

        let (kernel, stream) = load_complex_scale_kernel(self, kernel_name)?;
        let dims_dev = stream
            .clone_htod(&dims_to_i64(dims)?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dims", err))?;
        let lhs_strides_dev = stream
            .clone_htod(&to_i64_vec(lhs_strides, "lhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD lhs strides", err))?;
        let rhs_strides_dev = stream
            .clone_htod(&to_i64_vec(rhs_strides, "rhs stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD rhs strides", err))?;
        let dst_strides_dev = stream
            .clone_htod(&to_i64_vec(dst_strides, "dst stride")?)
            .map_err(|err| cuda_error("cudaMemcpyHtoD dst strides", err))?;
        let ndim = i32::try_from(dims.len())
            .map_err(|_| Error::InvalidArgument("pointwise rank exceeds i32 range".into()))?;
        let lhs_offset = i64::try_from(lhs_offset)
            .map_err(|_| Error::InvalidArgument("pointwise lhs offset exceeds i64 range".into()))?;
        let rhs_offset = i64::try_from(rhs_offset)
            .map_err(|_| Error::InvalidArgument("pointwise rhs offset exceeds i64 range".into()))?;
        let dst_offset = i64::try_from(dst_offset)
            .map_err(|_| Error::InvalidArgument("pointwise dst offset exceeds i64 range".into()))?;
        let numel_u64 = u64::try_from(numel)
            .map_err(|_| Error::InvalidArgument("pointwise numel exceeds u64 range".into()))?;
        let numel_u32 = u32::try_from(numel).map_err(|_| {
            Error::InvalidArgument("pointwise currently requires len <= u32::MAX".into())
        })?;
        let lhs_ptr = lhs as u64;
        let rhs_ptr = rhs as u64;
        let dst_ptr = dst as u64;
        let config = LaunchConfig {
            grid_dim: (numel_u32.div_ceil(256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream
                .launch_builder(&kernel)
                .arg(&lhs_ptr)
                .arg(&rhs_ptr)
                .arg(&dst_ptr)
                .arg(&dims_dev)
                .arg(&lhs_strides_dev)
                .arg(&lhs_offset)
                .arg(&rhs_strides_dev)
                .arg(&rhs_offset)
                .arg(&dst_strides_dev)
                .arg(&dst_offset)
                .arg(&ndim)
                .arg(&numel_u64)
                .arg(&alpha)
                .arg(&beta)
                .launch(config)
                .map_err(|err| cuda_error("CUDA complex-scale kernel launch", err))?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))
    }

    /// Launches the Layer 0 complex-to-real unary kernel for `Complex32 -> f32` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex32;
    /// use tenferro_device::cuda::runtime::{self, ComplexRealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<Complex32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_complex32_to_real_f32_raw(
    ///         ComplexRealUnaryOp::Abs,
    ///         1.0,
    ///         src.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_unary_complex32_to_real_f32_raw(
        &self,
        op: ComplexRealUnaryOp,
        alpha: f32,
        src: *const Complex32,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_complex_real_raw_impl(
            COMPLEX_REAL_UNARY_KERNEL_NAME_F32,
            op,
            alpha,
            src,
            dims,
            src_strides,
            src_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 complex-to-real unary kernel for `Complex64 -> f64` data.
    ///
    /// # Safety
    ///
    /// `src` and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use num_complex::Complex64;
    /// use tenferro_device::cuda::runtime::{self, ComplexRealUnaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<Complex64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_unary_complex64_to_real_f64_raw(
    ///         ComplexRealUnaryOp::Abs,
    ///         1.0,
    ///         src.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_unary_complex64_to_real_f64_raw(
        &self,
        op: ComplexRealUnaryOp,
        alpha: f64,
        src: *const Complex64,
        dims: &[usize],
        src_strides: &[isize],
        src_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_unary_complex_real_raw_impl(
            COMPLEX_REAL_UNARY_KERNEL_NAME_F64,
            op,
            alpha,
            src,
            dims,
            src_strides,
            src_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 complex-scale kernel for `Complex32 × f32 -> Complex32` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible
    /// with the provided layout metadata.
    #[allow(private_interfaces)]
    pub unsafe fn pointwise_mul_complex32_real_f32_raw(
        &self,
        alpha: Complex32,
        lhs: *const Complex32,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f32,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: Complex32,
        dst: *mut Complex32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_complex_scale_raw_impl::<KernelComplex32, Complex32>(
            COMPLEX_SCALE_KERNEL_NAME_F32,
            alpha.into(),
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta.into(),
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 complex-scale kernel for `Complex64 × f64 -> Complex64` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible
    /// with the provided layout metadata.
    #[allow(private_interfaces)]
    pub unsafe fn pointwise_mul_complex64_real_f64_raw(
        &self,
        alpha: Complex64,
        lhs: *const Complex64,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f64,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: Complex64,
        dst: *mut Complex64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_complex_scale_raw_impl::<KernelComplex64, Complex64>(
            COMPLEX_SCALE_KERNEL_NAME_F64,
            alpha.into(),
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta.into(),
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real binary kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealBinaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<f32>(4).unwrap();
    /// let rhs = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_binary_real_f32_raw(
    ///         RealBinaryOp::Add,
    ///         1.0,
    ///         lhs.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         rhs.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_binary_real_f32_raw(
        &self,
        op: RealBinaryOp,
        alpha: f32,
        lhs: *const f32,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f32,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_binary_real_raw_impl(
            op,
            alpha,
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real binary kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `lhs`, `rhs`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealBinaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lhs = runtime.alloc::<f64>(4).unwrap();
    /// let rhs = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_binary_real_f64_raw(
    ///         RealBinaryOp::Add,
    ///         1.0,
    ///         lhs.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         rhs.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_binary_real_f64_raw(
        &self,
        op: RealBinaryOp,
        alpha: f64,
        lhs: *const f64,
        dims: &[usize],
        lhs_strides: &[isize],
        lhs_offset: isize,
        rhs: *const f64,
        rhs_strides: &[isize],
        rhs_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_binary_real_raw_impl(
            op,
            alpha,
            lhs,
            dims,
            lhs_strides,
            lhs_offset,
            rhs,
            rhs_strides,
            rhs_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real ternary kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `cond`, `on_true`, `on_false`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealTernaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let cond = runtime.alloc::<f32>(4).unwrap();
    /// let on_true = runtime.alloc::<f32>(4).unwrap();
    /// let on_false = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_ternary_real_f32_raw(
    ///         RealTernaryOp::Where,
    ///         1.0,
    ///         cond.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         on_true.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         on_false.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_ternary_real_f32_raw(
        &self,
        op: RealTernaryOp,
        alpha: f32,
        cond: *const f32,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const f32,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const f32,
        false_strides: &[isize],
        false_offset: isize,
        beta: f32,
        dst: *mut f32,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_ternary_real_raw_impl(
            op,
            alpha,
            cond,
            dims,
            cond_strides,
            cond_offset,
            on_true,
            true_strides,
            true_offset,
            on_false,
            false_strides,
            false_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real ternary kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `cond`, `on_true`, `on_false`, and `dst` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealTernaryOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let cond = runtime.alloc::<f64>(4).unwrap();
    /// let on_true = runtime.alloc::<f64>(4).unwrap();
    /// let on_false = runtime.alloc::<f64>(4).unwrap();
    /// let dst = runtime.alloc::<f64>(4).unwrap();
    /// unsafe {
    ///     runtime.pointwise_ternary_real_f64_raw(
    ///         RealTernaryOp::Where,
    ///         1.0,
    ///         cond.device_ptr().cast_const(),
    ///         &[4],
    ///         &[1],
    ///         0,
    ///         on_true.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         on_false.device_ptr().cast_const(),
    ///         &[1],
    ///         0,
    ///         0.0,
    ///         dst.device_ptr(),
    ///         &[1],
    ///         0,
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn pointwise_ternary_real_f64_raw(
        &self,
        op: RealTernaryOp,
        alpha: f64,
        cond: *const f64,
        dims: &[usize],
        cond_strides: &[isize],
        cond_offset: isize,
        on_true: *const f64,
        true_strides: &[isize],
        true_offset: isize,
        on_false: *const f64,
        false_strides: &[isize],
        false_offset: isize,
        beta: f64,
        dst: *mut f64,
        dst_strides: &[isize],
        dst_offset: isize,
    ) -> Result<()> {
        self.pointwise_ternary_real_raw_impl(
            op,
            alpha,
            cond,
            dims,
            cond_strides,
            cond_offset,
            on_true,
            true_strides,
            true_offset,
            on_false,
            false_strides,
            false_offset,
            beta,
            dst,
            dst_strides,
            dst_offset,
        )
    }

    /// Launches the Layer 0 real reduction kernel for `f32` data.
    ///
    /// # Safety
    ///
    /// `input` and `output` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealReductionOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc::<f32>(4).unwrap();
    /// let output = runtime.alloc::<f32>(2).unwrap();
    /// unsafe {
    ///     runtime.reduce_real_f32_raw(
    ///         RealReductionOp::Sum,
    ///         1.0,
    ///         input.device_ptr().cast_const(),
    ///         &[2, 2],
    ///         &[1, 2],
    ///         0,
    ///         0.0,
    ///         output.device_ptr(),
    ///         &[2],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         &[0],
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn reduce_real_f32_raw(
        &self,
        op: RealReductionOp,
        alpha: f32,
        input: *const f32,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: f32,
        output: *mut f32,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        self.reduce_real_raw_impl(
            op,
            alpha,
            input,
            input_dims,
            input_strides,
            input_offset,
            beta,
            output,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )
    }

    /// Launches the Layer 0 real reduction kernel for `f64` data.
    ///
    /// # Safety
    ///
    /// `input` and `output` must point to live device allocations compatible with the provided layout metadata.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, RealReductionOp};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let input = runtime.alloc::<f64>(4).unwrap();
    /// let output = runtime.alloc::<f64>(2).unwrap();
    /// unsafe {
    ///     runtime.reduce_real_f64_raw(
    ///         RealReductionOp::Sum,
    ///         1.0,
    ///         input.device_ptr().cast_const(),
    ///         &[2, 2],
    ///         &[1, 2],
    ///         0,
    ///         0.0,
    ///         output.device_ptr(),
    ///         &[2],
    ///         &[1],
    ///         0,
    ///         &[1],
    ///         &[0],
    ///     ).unwrap();
    /// }
    /// ```
    pub unsafe fn reduce_real_f64_raw(
        &self,
        op: RealReductionOp,
        alpha: f64,
        input: *const f64,
        input_dims: &[usize],
        input_strides: &[isize],
        input_offset: isize,
        beta: f64,
        output: *mut f64,
        output_dims: &[usize],
        output_strides: &[isize],
        output_offset: isize,
        kept_axes: &[usize],
        reduced_axes: &[usize],
    ) -> Result<()> {
        self.reduce_real_raw_impl(
            op,
            alpha,
            input,
            input_dims,
            input_strides,
            input_offset,
            beta,
            output,
            output_dims,
            output_strides,
            output_offset,
            kept_axes,
            reduced_axes,
        )
    }

    /// Allocates a device buffer for `len` elements of `T`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.len(), 4);
    /// ```
    pub fn alloc<T>(&self, len: usize) -> Result<CudaBuffer<T>> {
        let ptr = self.alloc_raw::<T>(len)?;
        Ok(CudaBuffer::new(
            Arc::clone(&self.context),
            ptr.cast::<c_void>(),
            len,
        ))
    }

    /// Copies a host slice into a device buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// runtime.copy_htod(&[1.0_f32, 2.0, 3.0, 4.0], &buffer).unwrap();
    /// ```
    pub fn copy_htod<T>(&self, src: &[T], dst: &CudaBuffer<T>) -> Result<()> {
        self.ensure_same_device(dst.device_id())?;
        unsafe { self.copy_htod_raw(src, dst.ptr.cast::<T>(), dst.len()) }
    }

    /// Copies a device buffer into a host vector.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// let host = runtime.copy_dtoh::<f32>(&buffer).unwrap();
    /// assert_eq!(host.len(), 4);
    /// ```
    pub fn copy_dtoh<T>(&self, src: &CudaBuffer<T>) -> Result<Vec<T>> {
        self.ensure_same_device(src.device_id())?;
        unsafe { self.copy_dtoh_raw(src.ptr.cast::<T>(), src.len()) }
    }

    /// Copies the contents of one device buffer into another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(4).unwrap();
    /// let dst = runtime.alloc::<f32>(4).unwrap();
    /// runtime.copy_dtod(&src, &dst).unwrap();
    /// ```
    pub fn copy_dtod<T>(&self, src: &CudaBuffer<T>, dst: &CudaBuffer<T>) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        if src.len() != dst.len() {
            return Err(Error::InvalidArgument(format!(
                "device/device length mismatch: src={} dst={}",
                src.len(),
                dst.len()
            )));
        }

        unsafe { self.copy_dtod_raw(src.ptr.cast::<T>(), dst.ptr.cast::<T>(), src.len()) }
    }

    /// Launches the generic strided-copy kernel from one device buffer to another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(24).unwrap();
    /// let dst = runtime.alloc::<f32>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// runtime.copy_strided(&src, &dst, &spec).unwrap();
    /// ```
    pub fn copy_strided<T>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &StridedCopySpec,
    ) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe { self.copy_strided_raw(src.ptr.cast::<T>(), dst.ptr.cast::<T>(), spec) }
    }

    /// Launches the generic strided-copy kernel while applying a source-side transform.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec, StridedCopyTransform};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<num_complex::Complex64>(24).unwrap();
    /// let dst = runtime.alloc::<num_complex::Complex64>(24).unwrap();
    /// let spec = StridedCopySpec::to_contiguous(&[4, 2, 3], &[6, 1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// runtime.copy_strided_with_transform(&src, &dst, &spec, StridedCopyTransform::Conj).unwrap();
    /// ```
    pub fn copy_strided_with_transform<T: 'static>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &StridedCopySpec,
        transform: StridedCopyTransform,
    ) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe {
            self.copy_strided_raw_with_transform(
                src.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
                spec,
                transform,
            )
        }
    }

    /// Packs two source views into a freshly allocated contiguous destination buffer.
    ///
    /// The source views must live on the same device, have the same rank, and match on every
    /// dimension except `axis`. The destination is allocated on the same device and is laid out
    /// contiguously in the requested order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ContiguousOrder, StridedCopySpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let left = runtime.alloc::<f32>(2).unwrap();
    /// let right = runtime.alloc::<f32>(4).unwrap();
    /// let left_spec = StridedCopySpec::to_contiguous(&[1, 2], &[1, 1], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// let right_spec = StridedCopySpec::to_contiguous(&[2, 2], &[1, 2], 0, ContiguousOrder::ColumnMajor).unwrap();
    /// let packed = runtime.pack_concat_sources(&left, &left_spec, &right, &right_spec, 0, ContiguousOrder::ColumnMajor).unwrap();
    /// assert_eq!(packed.len(), 6);
    /// ```
    pub fn pack_concat_sources<T>(
        &self,
        left: &CudaBuffer<T>,
        left_spec: &StridedCopySpec,
        right: &CudaBuffer<T>,
        right_spec: &StridedCopySpec,
        axis: usize,
        order: ContiguousOrder,
    ) -> Result<CudaBuffer<T>> {
        self.ensure_same_device(left.device_id())?;
        self.ensure_same_device(right.device_id())?;
        if left_spec.dims.len() != left_spec.src_strides.len()
            || right_spec.dims.len() != right_spec.src_strides.len()
        {
            if left_spec.dims.len() != left_spec.src_strides.len() {
                return Err(Error::InvalidArgument(format!(
                    "concat pack left spec rank mismatch: dims={} src_strides={}",
                    left_spec.dims.len(),
                    left_spec.src_strides.len()
                )));
            }
            return Err(Error::InvalidArgument(format!(
                "concat pack right spec rank mismatch: dims={} src_strides={}",
                right_spec.dims.len(),
                right_spec.src_strides.len()
            )));
        }
        if left_spec.dims.len() != right_spec.dims.len() {
            return Err(Error::InvalidArgument(format!(
                "concat pack source rank mismatch: left={} right={}",
                left_spec.dims.len(),
                right_spec.dims.len()
            )));
        }
        if axis >= left_spec.dims.len() {
            return Err(Error::InvalidArgument(format!(
                "concat axis {axis} out of range for rank {}",
                left_spec.dims.len()
            )));
        }
        for dim_axis in 0..left_spec.dims.len() {
            if dim_axis != axis && left_spec.dims[dim_axis] != right_spec.dims[dim_axis] {
                return Err(Error::InvalidArgument(format!(
                    "concat dimension mismatch at axis {dim_axis}: left={} right={}",
                    left_spec.dims[dim_axis], right_spec.dims[dim_axis]
                )));
            }
        }

        let mut dst_dims = left_spec.dims.clone();
        dst_dims[axis] = dst_dims[axis]
            .checked_add(right_spec.dims[axis])
            .ok_or_else(|| Error::InvalidArgument("concat dimension overflow".into()))?;
        let dst_len = checked_numel(&dst_dims)?;
        let dst = self.alloc::<T>(dst_len)?;
        if dst_len == 0 {
            return Ok(dst);
        }
        let dst_strides = contiguous_strides(&dst_dims, order)?;
        let axis_stride = dst_strides[axis];
        let right_axis_len = isize::try_from(left_spec.dims[axis]).map_err(|_| {
            Error::InvalidArgument(format!(
                "concat axis length {} exceeds isize range",
                left_spec.dims[axis]
            ))
        })?;
        let right_dst_offset = right_axis_len
            .checked_mul(axis_stride)
            .ok_or_else(|| Error::InvalidArgument("concat destination offset overflow".into()))?;

        let left_dst_spec = StridedCopySpec {
            dims: left_spec.dims.clone(),
            src_strides: left_spec.src_strides.clone(),
            src_offset: left_spec.src_offset,
            dst_strides: dst_strides.clone(),
            dst_offset: 0,
        };
        let right_dst_spec = StridedCopySpec {
            dims: right_spec.dims.clone(),
            src_strides: right_spec.src_strides.clone(),
            src_offset: right_spec.src_offset,
            dst_strides,
            dst_offset: right_dst_offset,
        };

        let stream = unsafe {
            self.launch_strided_copy_raw(left.ptr.cast::<T>(), dst.ptr.cast::<T>(), &left_dst_spec)?
        };
        unsafe {
            self.launch_strided_copy_raw(
                right.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
                &right_dst_spec,
            )?;
        }
        stream
            .synchronize()
            .map_err(|err| cuda_error("CUDA stream synchronize", err))?;
        Ok(dst)
    }

    /// Launches the keep-count-driven trailing zero-fill kernel from one device buffer to another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, ZeroTrailingByCountsSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(8).unwrap();
    /// let dst = runtime.alloc::<f32>(8).unwrap();
    /// let keep_counts = runtime.alloc::<f32>(2).unwrap();
    /// let spec = ZeroTrailingByCountsSpec::new(&[2, 2, 2], &[1, 2, 4], 0, &[1, 2, 4], 0, &[1], 0, 1, 2).unwrap();
    /// runtime.zero_trailing_by_counts(&src, &dst, &keep_counts, &spec).unwrap();
    /// ```
    pub fn zero_trailing_by_counts<T, R>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        keep_counts: &CudaBuffer<R>,
        spec: &ZeroTrailingByCountsSpec,
    ) -> Result<()>
    where
        R: RuntimeKeepCountScalar,
    {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        self.ensure_same_device(keep_counts.device_id())?;
        let expected_keep_count_len = checked_numel(&spec.dims[spec.structural_rank..])?;
        if keep_counts.len() != expected_keep_count_len {
            return Err(Error::InvalidArgument(format!(
                "keep-count buffer length mismatch: expected {} got {}",
                expected_keep_count_len,
                keep_counts.len()
            )));
        }
        unsafe {
            self.zero_trailing_by_counts_raw(
                src.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
                keep_counts.ptr.cast::<R>(),
                spec,
            )
        }
    }

    /// Launches the triangular-copy kernel from one device buffer to another.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, TriangularHalf, TriangularPartSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let src = runtime.alloc::<f32>(24).unwrap();
    /// let dst = runtime.alloc::<f32>(24).unwrap();
    /// let spec = TriangularPartSpec::new(&[3, 2, 4], &[1, 3, 6], 0, &[1, 3, 6], 0, 0, TriangularHalf::Upper).unwrap();
    /// runtime.triangular_part(&src, &dst, &spec).unwrap();
    /// ```
    pub fn triangular_part<T>(
        &self,
        src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &TriangularPartSpec,
    ) -> Result<()> {
        self.ensure_same_device(src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe { self.triangular_part_raw(src.ptr.cast::<T>(), dst.ptr.cast::<T>(), spec) }
    }

    /// Launches the triangular-merge kernel from two device buffers into a destination buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime::{self, TriangularMergeSpec};
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let lower = runtime.alloc::<f32>(24).unwrap();
    /// let upper = runtime.alloc::<f32>(24).unwrap();
    /// let dst = runtime.alloc::<f32>(24).unwrap();
    /// let spec = TriangularMergeSpec::new(
    ///     &[3, 2, 4],
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1, 3, 6],
    ///     0,
    ///     &[1, 3, 6],
    ///     0,
    /// ).unwrap();
    /// runtime.triangular_merge(&lower, &upper, &dst, &spec).unwrap();
    /// ```
    pub fn triangular_merge<T>(
        &self,
        lower_src: &CudaBuffer<T>,
        upper_src: &CudaBuffer<T>,
        dst: &CudaBuffer<T>,
        spec: &TriangularMergeSpec,
    ) -> Result<()> {
        self.ensure_same_device(lower_src.device_id())?;
        self.ensure_same_device(upper_src.device_id())?;
        self.ensure_same_device(dst.device_id())?;
        unsafe {
            self.triangular_merge_raw(
                lower_src.ptr.cast::<T>(),
                upper_src.ptr.cast::<T>(),
                dst.ptr.cast::<T>(),
                spec,
            )
        }
    }

    fn ensure_same_device(&self, device_id: usize) -> Result<()> {
        if self.device_id() == device_id {
            Ok(())
        } else {
            Err(Error::InvalidArgument(format!(
                "buffer belongs to device {device_id}, runtime is bound to device {}",
                self.device_id()
            )))
        }
    }
}

/// Owning CUDA device buffer allocated by [`CudaRuntime`].
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// let buffer = runtime.alloc::<f32>(4).unwrap();
/// assert_eq!(buffer.len(), 4);
/// ```
#[derive(Debug)]
pub struct CudaBuffer<T> {
    context: Arc<CudaContext>,
    ptr: *mut c_void,
    len: usize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for CudaBuffer<T> {}
unsafe impl<T: Sync> Sync for CudaBuffer<T> {}

impl<T> CudaBuffer<T> {
    fn new(context: Arc<CudaContext>, ptr: *mut c_void, len: usize) -> Self {
        Self {
            context,
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Returns the number of elements in this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the raw device pointer for this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert!(!buffer.device_ptr().is_null());
    /// ```
    pub fn device_ptr(&self) -> *mut T {
        self.ptr.cast::<T>()
    }

    /// Returns the device ordinal that owns this buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_device::cuda::runtime;
    ///
    /// let runtime = runtime::get_or_init(0).unwrap();
    /// let buffer = runtime.alloc::<f32>(4).unwrap();
    /// assert_eq!(buffer.device_id(), 0);
    /// ```
    pub fn device_id(&self) -> usize {
        self.context.ordinal()
    }
}

impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        if !self.ptr.is_null() {
            let _ = unsafe { cuda_result::free_sync(self.ptr) };
        }
    }
}

/// Returns the shared runtime handle for one CUDA device ordinal.
///
/// # Examples
///
/// ```ignore
/// use tenferro_device::cuda::runtime;
///
/// let runtime = runtime::get_or_init(0).unwrap();
/// assert_eq!(runtime.device_id(), 0);
/// ```
pub fn get_or_init(device_id: usize) -> Result<Arc<CudaRuntime>> {
    let mut cache = runtime_cache()
        .lock()
        .map_err(|_| Error::DeviceError("CUDA runtime cache mutex poisoned".into()))?;
    if let Some(runtime) = cache.get(&device_id) {
        return Ok(Arc::clone(runtime));
    }

    let runtime = CudaRuntime::new(device_id)?;
    cache.insert(device_id, Arc::clone(&runtime));
    Ok(runtime)
}
