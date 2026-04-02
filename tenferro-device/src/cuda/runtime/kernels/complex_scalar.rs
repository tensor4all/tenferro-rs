use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaFunction, CudaStream, DeviceRepr};
use cudarc::nvrtc::Ptx;
use num_complex::{Complex32, Complex64};

use super::super::shared::*;
use super::super::state::CudaRuntime;
use super::helpers::{compile_ptx_once, load_kernel_from_ptx};
use crate::Result;

pub const COMPLEX_REAL_UNARY_KERNEL_NAME_F32: &str = "pointwise_unary_complex32_to_real_f32";
pub const COMPLEX_REAL_UNARY_KERNEL_NAME_F64: &str = "pointwise_unary_complex64_to_real_f64";
pub const COMPLEX_SCALE_KERNEL_NAME_F32: &str = "pointwise_mul_complex32_real_f32";
pub const COMPLEX_SCALE_KERNEL_NAME_F64: &str = "pointwise_mul_complex64_real_f64";

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

unsafe impl DeviceRepr for KernelComplex32 {}
unsafe impl DeviceRepr for KernelComplex64 {}

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

pub trait ComplexScaleSrc {
    type Real;
}

impl ComplexScaleSrc for Complex32 {
    type Real = f32;
}

impl ComplexScaleSrc for Complex64 {
    type Real = f64;
}

pub const COMPLEX_REAL_CUDA_SRC: &str = r#"
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

pub const COMPLEX_SCALE_CUDA_SRC: &str = r#"
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

pub fn complex_real_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, COMPLEX_REAL_CUDA_SRC, "complex-real kernel")
}

pub fn complex_scale_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    compile_ptx_once(&PTX, COMPLEX_SCALE_CUDA_SRC, "complex-scale kernel")
}

pub fn complex_real_opcode(op: ComplexRealUnaryOp) -> i32 {
    match op {
        ComplexRealUnaryOp::Abs => 0,
        ComplexRealUnaryOp::Real => 1,
        ComplexRealUnaryOp::Imag => 2,
    }
}

pub fn load_complex_real_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    load_kernel_from_ptx(runtime, complex_real_ptx()?, kernel_name)
}

pub fn load_complex_scale_kernel(
    runtime: &CudaRuntime,
    kernel_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    load_kernel_from_ptx(runtime, complex_scale_ptx()?, kernel_name)
}
