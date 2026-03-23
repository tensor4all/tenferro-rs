use std::sync::{Arc, OnceLock};

use cudarc::{
    driver::{CudaFunction, CudaStream},
    nvrtc::Ptx,
};

use super::super::state::CudaRuntime;
use super::helpers::{compile_ptx_once, load_kernel_from_ptx};
use crate::Result;

pub const RNG_UNIFORM_F64_KERNEL_NAME: &str = "rng_fill_uniform_f64";
pub const RNG_NORMAL_F64_KERNEL_NAME: &str = "rng_fill_normal_f64";
pub const RNG_INT_I32_KERNEL_NAME: &str = "rng_fill_i32";

pub const RNG_CUDA_SRC: &str = r#"
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

__device__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ double uniform53(unsigned long long seed, unsigned long long counter) {
    unsigned long long bits = splitmix64(seed ^ counter);
    unsigned long long mantissa = bits >> 11;
    return (double)mantissa * (1.0 / 9007199254740992.0);
}

extern "C" __global__ void rng_fill_uniform_f64(
    unsigned long long seed,
    unsigned long long offset_counter,
    unsigned long long dst_ptr,
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
    double* dst = (double*)dst_ptr;
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = uniform53(seed, offset_counter + idx);
}

extern "C" __global__ void rng_fill_normal_f64(
    unsigned long long seed,
    unsigned long long offset_counter,
    unsigned long long dst_ptr,
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
    double* dst = (double*)dst_ptr;
    unsigned long long base_counter = offset_counter + 2ULL * idx;
    double u1 = uniform53(seed, base_counter);
    if (u1 <= 0.0) {
        u1 = 0x1.0p-53;
    }
    double u2 = uniform53(seed, base_counter + 1ULL);
    double radius = sqrt(-2.0 * log(u1));
    double theta = 6.28318530717958647692 * u2;
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = radius * cos(theta);
}

extern "C" __global__ void rng_fill_i32(
    unsigned long long seed,
    unsigned long long offset_counter,
    int low,
    int high,
    unsigned long long dst_ptr,
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
    int* dst = (int*)dst_ptr;
    unsigned long long value = splitmix64(seed ^ (offset_counter + idx));
    unsigned int span = (unsigned int)(high - low);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = low + (int)(value % (unsigned long long)span);
}
"#;

fn load_rng_kernel_impl(
    runtime: &CudaRuntime,
    kernel_name: &'static str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    let ptx = compile_ptx_once(&PTX, RNG_CUDA_SRC, "rng")?;
    load_kernel_from_ptx(runtime, ptx, kernel_name)
}

pub fn load_rng_kernel(
    runtime: &CudaRuntime,
    kernel_name: &'static str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    load_rng_kernel_impl(runtime, kernel_name)
}
