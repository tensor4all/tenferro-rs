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

__device__ uint2 split_u64(unsigned long long value) {
    return make_uint2(
        (unsigned int)(value & 0xffffffffULL),
        (unsigned int)(value >> 32)
    );
}

__device__ uint2 mulhilo32(unsigned int a, unsigned int b) {
    unsigned long long product = (unsigned long long)a * (unsigned long long)b;
    return make_uint2((unsigned int)product, (unsigned int)(product >> 32));
}

__device__ uint4 philox_single_round(uint4 ctr, uint2 key) {
    const unsigned int kPhiloxSA = 0xD2511F53U;
    const unsigned int kPhiloxSB = 0xCD9E8D57U;
    uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
    return make_uint4(
        res1.y ^ ctr.y ^ key.x,
        res1.x,
        res0.y ^ ctr.w ^ key.y,
        res0.x
    );
}

__device__ uint4 philox4x32_10(unsigned long long seed, unsigned long long subsequence) {
    const unsigned int kPhilox10A = 0x9E3779B9U;
    const unsigned int kPhilox10B = 0xBB67AE85U;
    uint2 key = split_u64(seed);
    uint4 ctr = make_uint4(0U, 0U, (unsigned int)subsequence, (unsigned int)(subsequence >> 32));
    #pragma unroll
    for (int round = 0; round < 9; ++round) {
        ctr = philox_single_round(ctr, key);
        key.x += kPhilox10A;
        key.y += kPhilox10B;
    }
    return philox_single_round(ctr, key);
}

__device__ double uniform_from_uint32(unsigned int value) {
    const double scale = 4.6566127342e-10;
    return (double)(value & 0x7fffffffU) * scale;
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
    uint4 value = philox4x32_10(seed, offset_counter + idx);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = uniform_from_uint32(value.x);
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
    uint4 value = philox4x32_10(seed, offset_counter + idx);
    double u1 = 1.0 - uniform_from_uint32(value.x);
    if (u1 <= 0.0) {
        u1 = 4.6566127342e-10;
    }
    double u2 = 1.0 - uniform_from_uint32(value.y);
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
    uint4 value = philox4x32_10(seed, offset_counter + idx);
    unsigned int span = (unsigned int)(high - low);
    long long dst_idx = linear_offset(idx, dims, dst_strides, dst_offset, ndim);
    dst[dst_idx] = low + (int)(uniform_from_uint32(value.x) * (double)span);
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
