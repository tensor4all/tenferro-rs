use std::sync::OnceLock;

use cudarc::nvrtc::{compile_ptx, Ptx};

use tenferro_device::{Error, Result};

pub(super) const RESOLVE_CONJ_KERNEL_NAME_C32: &str = "resolve_conj_complex32";
pub(super) const RESOLVE_CONJ_KERNEL_NAME_C64: &str = "resolve_conj_complex64";

const RESOLVE_CONJ_CUDA_SRC: &str = r#"
typedef struct { float re; float im; } complex32_t;
typedef struct { double re; double im; } complex64_t;

extern "C" __global__ void resolve_conj_complex32(
    const complex32_t* src,
    complex32_t* dst,
    unsigned long long len
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= len) {
        return;
    }
    dst[idx].re = src[idx].re;
    dst[idx].im = -src[idx].im;
}

extern "C" __global__ void resolve_conj_complex64(
    const complex64_t* src,
    complex64_t* dst,
    unsigned long long len
) {
    unsigned long long idx =
        (unsigned long long)blockIdx.x * (unsigned long long)blockDim.x +
        (unsigned long long)threadIdx.x;
    if (idx >= len) {
        return;
    }
    dst[idx].re = src[idx].re;
    dst[idx].im = -src[idx].im;
}
"#;

pub(super) fn resolve_conj_ptx() -> Result<Ptx> {
    static PTX: OnceLock<std::result::Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| {
        compile_ptx(RESOLVE_CONJ_CUDA_SRC)
            .map_err(|err| format!("NVRTC compile failed for resolve_conj kernel: {err:?}"))
    })
    .clone()
    .map_err(Error::DeviceError)
}
