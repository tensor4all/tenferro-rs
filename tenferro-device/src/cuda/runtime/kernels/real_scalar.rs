use std::sync::{Arc, OnceLock};

use cudarc::{
    driver::{CudaFunction, CudaStream},
    nvrtc::Ptx,
};

use super::super::{shared::*, state::CudaRuntime};
use super::helpers::{compile_ptx_once, load_kernel_from_ptx};
use crate::Result;

pub const REAL_UNARY_KERNEL_NAME_F32: &str = "pointwise_unary_real_f32";
pub const REAL_UNARY_KERNEL_NAME_F64: &str = "pointwise_unary_real_f64";
pub const REAL_BINARY_KERNEL_NAME_F32: &str = "pointwise_binary_real_f32";
pub const REAL_BINARY_KERNEL_NAME_F64: &str = "pointwise_binary_real_f64";
pub const REAL_TERNARY_KERNEL_NAME_F32: &str = "pointwise_ternary_real_f32";
pub const REAL_TERNARY_KERNEL_NAME_F64: &str = "pointwise_ternary_real_f64";
pub const REAL_REDUCTION_KERNEL_NAME_F32: &str = "reduce_real_f32";
pub const REAL_REDUCTION_KERNEL_NAME_F64: &str = "reduce_real_f64";
pub const REAL_SCALAR_CUDA_SRC: &str = r#"
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
    } else if (op_code == 5) {
        mapped = ceilf(value);
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
    } else if (op_code == 5) {
        mapped = ceil(value);
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

pub fn real_scalar_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, REAL_SCALAR_CUDA_SRC, "real-scalar kernel")
}

pub fn unary_opcode(op: RealUnaryOp) -> i32 {
    match op {
        RealUnaryOp::Conj => 0,
        RealUnaryOp::Abs => 1,
        RealUnaryOp::Reciprocal => 2,
        RealUnaryOp::Log => 3,
        RealUnaryOp::Sqrt => 4,
        RealUnaryOp::Ceil => 5,
    }
}

pub fn binary_opcode(op: RealBinaryOp) -> i32 {
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

pub fn ternary_opcode(op: RealTernaryOp) -> i32 {
    match op {
        RealTernaryOp::Where => 0,
    }
}

pub fn reduction_opcode(op: RealReductionOp) -> i32 {
    match op {
        RealReductionOp::Sum => 0,
        RealReductionOp::Max => 1,
        RealReductionOp::Min => 2,
        RealReductionOp::Prod => 3,
    }
}

pub fn load_real_scalar_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    load_kernel_from_ptx(runtime, real_scalar_ptx()?, kernel_name)
}
