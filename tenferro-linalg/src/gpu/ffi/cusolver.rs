use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::Arc;

use libloading::Library;
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{Error, Result};

use super::library_search_paths;

/// Default cuSOLVER search paths. Override with `TENFERRO_CUSOLVER_PATH` env var.
const CUSOLVER_DEFAULT_PATHS: &[&str] = &[
    "libcusolver.so.12",
    "libcusolver.so.11",
    "/usr/lib/x86_64-linux-gnu/libcusolver.so.11",
    "libcusolver.so",
];
/// Default cuBLAS search paths. Override with `TENFERRO_CUBLAS_PATH` env var.
const CUBLAS_DEFAULT_PATHS: &[&str] = &[
    "libcublas.so.12",
    "libcublas.so",
    "/usr/lib/x86_64-linux-gnu/libcublas.so",
];

pub type CusolverDnHandleRaw = *mut c_void;
pub type CublasHandleRaw = *mut c_void;
pub type CudaStream = *mut c_void;
type GesvdjInfoRaw = *mut c_void;

type CusolverStatus = i32;
type CublasStatus = i32;

const CUSOLVER_STATUS_SUCCESS: CusolverStatus = 0;
const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CudaDataType {
    F32,
    F64,
    Complex32,
    Complex64,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CublasFillMode {
    Lower = 0,
    Upper = 1,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CublasDiagType {
    NonUnit = 0,
    Unit = 1,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CublasSideMode {
    Left = 0,
    Right = 1,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CublasOperation {
    N = 0,
    T = 1,
    C = 2,
}

#[repr(i32)]
#[derive(Clone, Copy)]
pub enum CusolverEigMode {
    NoVector = 0,
    Vector = 1,
}

type CusolverCreateFn = unsafe extern "C" fn(*mut CusolverDnHandleRaw) -> CusolverStatus;
type CusolverDestroyFn = unsafe extern "C" fn(CusolverDnHandleRaw) -> CusolverStatus;
type CusolverSetStreamFn = unsafe extern "C" fn(CusolverDnHandleRaw, CudaStream) -> CusolverStatus;

type PotrfBufferSizeF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type PotrfBufferSizeF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type PotrfBufferSizeC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type PotrfBufferSizeC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type PotrfF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type PotrfF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type PotrfC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type PotrfC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CublasFillMode,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;

type GetrfBufferSizeF32Fn =
    unsafe extern "C" fn(CusolverDnHandleRaw, i32, i32, *mut f32, i32, *mut i32) -> CusolverStatus;
type GetrfBufferSizeF64Fn =
    unsafe extern "C" fn(CusolverDnHandleRaw, i32, i32, *mut f64, i32, *mut i32) -> CusolverStatus;
type GetrfBufferSizeC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type GetrfBufferSizeC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type GetrfF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut i32,
    *mut i32,
) -> CusolverStatus;
type GetrfF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut i32,
    *mut i32,
) -> CusolverStatus;
type GetrfC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    *mut i32,
    *mut i32,
) -> CusolverStatus;
type GetrfC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    *mut i32,
    *mut i32,
) -> CusolverStatus;

type GeqrfBufferSizeF32Fn =
    unsafe extern "C" fn(CusolverDnHandleRaw, i32, i32, *mut f32, i32, *mut i32) -> CusolverStatus;
type GeqrfBufferSizeF64Fn =
    unsafe extern "C" fn(CusolverDnHandleRaw, i32, i32, *mut f64, i32, *mut i32) -> CusolverStatus;
type GeqrfBufferSizeC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type GeqrfBufferSizeC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;
type GeqrfF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type GeqrfF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type GeqrfC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type GeqrfC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;

type OrgqrBufferSizeF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *const f32,
    i32,
    *const f32,
    *mut i32,
) -> CusolverStatus;
type OrgqrBufferSizeF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *const f64,
    i32,
    *const f64,
    *mut i32,
) -> CusolverStatus;
type OrgqrBufferSizeC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *const Complex32,
    i32,
    *const Complex32,
    *mut i32,
) -> CusolverStatus;
type OrgqrBufferSizeC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *const Complex64,
    i32,
    *const Complex64,
    *mut i32,
) -> CusolverStatus;
type OrgqrF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *mut f32,
    i32,
    *const f32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type OrgqrF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *mut f64,
    i32,
    *const f64,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type OrgqrC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *mut Complex32,
    i32,
    *const Complex32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type OrgqrC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    i32,
    i32,
    i32,
    *mut Complex64,
    i32,
    *const Complex64,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;

type GesvdBufferSizeFn =
    unsafe extern "C" fn(CusolverDnHandleRaw, i32, i32, *mut i32) -> CusolverStatus;
type GesvdF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    c_char,
    c_char,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut i32,
) -> CusolverStatus;
type GesvdF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    c_char,
    c_char,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut i32,
) -> CusolverStatus;
type GesvdC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    c_char,
    c_char,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut f32,
    *mut Complex32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    i32,
    *mut f32,
    *mut i32,
) -> CusolverStatus;
type GesvdC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    c_char,
    c_char,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut f64,
    *mut Complex64,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    i32,
    *mut f64,
    *mut i32,
) -> CusolverStatus;

type CreateGesvdjInfoFn = unsafe extern "C" fn(*mut GesvdjInfoRaw) -> CusolverStatus;
type DestroyGesvdjInfoFn = unsafe extern "C" fn(GesvdjInfoRaw) -> CusolverStatus;
type GesvdjBufferSizeF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *const f32,
    i32,
    *const f32,
    *const f32,
    i32,
    *const f32,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;
type GesvdjBufferSizeF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *const f64,
    i32,
    *const f64,
    *const f64,
    i32,
    *const f64,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;
type GesvdjBufferSizeC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *const Complex32,
    i32,
    *const f32,
    *const Complex32,
    i32,
    *const Complex32,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;
type GesvdjBufferSizeC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *const Complex64,
    i32,
    *const f64,
    *const Complex64,
    i32,
    *const Complex64,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;
type GesvdjF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut f32,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;
type GesvdjF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut f64,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;
type GesvdjC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *mut Complex32,
    i32,
    *mut f32,
    *mut Complex32,
    i32,
    *mut Complex32,
    i32,
    *mut Complex32,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;
type GesvdjC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    i32,
    i32,
    i32,
    *mut Complex64,
    i32,
    *mut f64,
    *mut Complex64,
    i32,
    *mut Complex64,
    i32,
    *mut Complex64,
    i32,
    *mut i32,
    GesvdjInfoRaw,
) -> CusolverStatus;

type SyevdBufferSizeF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *const f32,
    i32,
    *const f32,
    *mut i32,
) -> CusolverStatus;
type SyevdBufferSizeF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *const f64,
    i32,
    *const f64,
    *mut i32,
) -> CusolverStatus;
type SyevdBufferSizeC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *const Complex32,
    i32,
    *const f32,
    *mut i32,
) -> CusolverStatus;
type SyevdBufferSizeC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *const Complex64,
    i32,
    *const f64,
    *mut i32,
) -> CusolverStatus;
type SyevdF32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *mut f32,
    i32,
    *mut f32,
    *mut f32,
    i32,
    *mut i32,
) -> CusolverStatus;
type SyevdF64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *mut f64,
    i32,
    *mut f64,
    *mut f64,
    i32,
    *mut i32,
) -> CusolverStatus;
type SyevdC32Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *mut Complex32,
    i32,
    *mut f32,
    *mut Complex32,
    i32,
    *mut i32,
) -> CusolverStatus;
type SyevdC64Fn = unsafe extern "C" fn(
    CusolverDnHandleRaw,
    CusolverEigMode,
    CublasFillMode,
    i32,
    *mut Complex64,
    i32,
    *mut f64,
    *mut Complex64,
    i32,
    *mut i32,
) -> CusolverStatus;

type CublasCreateFn = unsafe extern "C" fn(*mut CublasHandleRaw) -> CublasStatus;
type CublasDestroyFn = unsafe extern "C" fn(CublasHandleRaw) -> CublasStatus;
type CublasSetStreamFn = unsafe extern "C" fn(CublasHandleRaw, CudaStream) -> CublasStatus;
type TrsmF32Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    *mut f32,
    i32,
) -> CublasStatus;
type TrsmF64Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const f64,
    *const f64,
    i32,
    *mut f64,
    i32,
) -> CublasStatus;
type TrsmC32Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const Complex32,
    *const Complex32,
    i32,
    *mut Complex32,
    i32,
) -> CublasStatus;
type TrsmC64Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const Complex64,
    *const Complex64,
    i32,
    *mut Complex64,
    i32,
) -> CublasStatus;
type TrsmBatchedF32Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const f32,
    *const *const f32,
    i32,
    *mut *mut f32,
    i32,
    i32,
) -> CublasStatus;
type TrsmBatchedF64Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const f64,
    *const *const f64,
    i32,
    *mut *mut f64,
    i32,
    i32,
) -> CublasStatus;
type TrsmBatchedC32Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const Complex32,
    *const *const Complex32,
    i32,
    *mut *mut Complex32,
    i32,
    i32,
) -> CublasStatus;
type TrsmBatchedC64Fn = unsafe extern "C" fn(
    CublasHandleRaw,
    CublasSideMode,
    CublasFillMode,
    CublasOperation,
    CublasDiagType,
    i32,
    i32,
    *const Complex64,
    *const *const Complex64,
    i32,
    *mut *mut Complex64,
    i32,
    i32,
) -> CublasStatus;

struct CusolverVtable {
    create: CusolverCreateFn,
    destroy: CusolverDestroyFn,
    set_stream: CusolverSetStreamFn,
    spotrf_buffer_size: PotrfBufferSizeF32Fn,
    dpotrf_buffer_size: PotrfBufferSizeF64Fn,
    cpotrf_buffer_size: PotrfBufferSizeC32Fn,
    zpotrf_buffer_size: PotrfBufferSizeC64Fn,
    spotrf: PotrfF32Fn,
    dpotrf: PotrfF64Fn,
    cpotrf: PotrfC32Fn,
    zpotrf: PotrfC64Fn,
    sgetrf_buffer_size: GetrfBufferSizeF32Fn,
    dgetrf_buffer_size: GetrfBufferSizeF64Fn,
    cgetrf_buffer_size: GetrfBufferSizeC32Fn,
    zgetrf_buffer_size: GetrfBufferSizeC64Fn,
    sgetrf: GetrfF32Fn,
    dgetrf: GetrfF64Fn,
    cgetrf: GetrfC32Fn,
    zgetrf: GetrfC64Fn,
    sgeqrf_buffer_size: GeqrfBufferSizeF32Fn,
    dgeqrf_buffer_size: GeqrfBufferSizeF64Fn,
    cgeqrf_buffer_size: GeqrfBufferSizeC32Fn,
    zgeqrf_buffer_size: GeqrfBufferSizeC64Fn,
    sgeqrf: GeqrfF32Fn,
    dgeqrf: GeqrfF64Fn,
    cgeqrf: GeqrfC32Fn,
    zgeqrf: GeqrfC64Fn,
    sorgqr_buffer_size: OrgqrBufferSizeF32Fn,
    dorgqr_buffer_size: OrgqrBufferSizeF64Fn,
    cungqr_buffer_size: OrgqrBufferSizeC32Fn,
    zungqr_buffer_size: OrgqrBufferSizeC64Fn,
    sorgqr: OrgqrF32Fn,
    dorgqr: OrgqrF64Fn,
    cungqr: OrgqrC32Fn,
    zungqr: OrgqrC64Fn,
    sgesvd_buffer_size: GesvdBufferSizeFn,
    dgesvd_buffer_size: GesvdBufferSizeFn,
    cgesvd_buffer_size: GesvdBufferSizeFn,
    zgesvd_buffer_size: GesvdBufferSizeFn,
    sgesvd: GesvdF32Fn,
    dgesvd: GesvdF64Fn,
    cgesvd: GesvdC32Fn,
    zgesvd: GesvdC64Fn,
    create_gesvdj_info: CreateGesvdjInfoFn,
    destroy_gesvdj_info: DestroyGesvdjInfoFn,
    sgesvdj_buffer_size: GesvdjBufferSizeF32Fn,
    dgesvdj_buffer_size: GesvdjBufferSizeF64Fn,
    cgesvdj_buffer_size: GesvdjBufferSizeC32Fn,
    zgesvdj_buffer_size: GesvdjBufferSizeC64Fn,
    sgesvdj: GesvdjF32Fn,
    dgesvdj: GesvdjF64Fn,
    cgesvdj: GesvdjC32Fn,
    zgesvdj: GesvdjC64Fn,
    ssyevd_buffer_size: SyevdBufferSizeF32Fn,
    dsyevd_buffer_size: SyevdBufferSizeF64Fn,
    cheevd_buffer_size: SyevdBufferSizeC32Fn,
    zheevd_buffer_size: SyevdBufferSizeC64Fn,
    ssyevd: SyevdF32Fn,
    dsyevd: SyevdF64Fn,
    cheevd: SyevdC32Fn,
    zheevd: SyevdC64Fn,
}

impl CusolverVtable {
    unsafe fn load(lib: &Library) -> Result<Self> {
        Ok(Self {
            create: load_symbol(lib, b"cusolverDnCreate\0", "cuSOLVER")?,
            destroy: load_symbol(lib, b"cusolverDnDestroy\0", "cuSOLVER")?,
            set_stream: load_symbol(lib, b"cusolverDnSetStream\0", "cuSOLVER")?,
            spotrf_buffer_size: load_symbol(lib, b"cusolverDnSpotrf_bufferSize\0", "cuSOLVER")?,
            dpotrf_buffer_size: load_symbol(lib, b"cusolverDnDpotrf_bufferSize\0", "cuSOLVER")?,
            cpotrf_buffer_size: load_symbol(lib, b"cusolverDnCpotrf_bufferSize\0", "cuSOLVER")?,
            zpotrf_buffer_size: load_symbol(lib, b"cusolverDnZpotrf_bufferSize\0", "cuSOLVER")?,
            spotrf: load_symbol(lib, b"cusolverDnSpotrf\0", "cuSOLVER")?,
            dpotrf: load_symbol(lib, b"cusolverDnDpotrf\0", "cuSOLVER")?,
            cpotrf: load_symbol(lib, b"cusolverDnCpotrf\0", "cuSOLVER")?,
            zpotrf: load_symbol(lib, b"cusolverDnZpotrf\0", "cuSOLVER")?,
            sgetrf_buffer_size: load_symbol(lib, b"cusolverDnSgetrf_bufferSize\0", "cuSOLVER")?,
            dgetrf_buffer_size: load_symbol(lib, b"cusolverDnDgetrf_bufferSize\0", "cuSOLVER")?,
            cgetrf_buffer_size: load_symbol(lib, b"cusolverDnCgetrf_bufferSize\0", "cuSOLVER")?,
            zgetrf_buffer_size: load_symbol(lib, b"cusolverDnZgetrf_bufferSize\0", "cuSOLVER")?,
            sgetrf: load_symbol(lib, b"cusolverDnSgetrf\0", "cuSOLVER")?,
            dgetrf: load_symbol(lib, b"cusolverDnDgetrf\0", "cuSOLVER")?,
            cgetrf: load_symbol(lib, b"cusolverDnCgetrf\0", "cuSOLVER")?,
            zgetrf: load_symbol(lib, b"cusolverDnZgetrf\0", "cuSOLVER")?,
            sgeqrf_buffer_size: load_symbol(lib, b"cusolverDnSgeqrf_bufferSize\0", "cuSOLVER")?,
            dgeqrf_buffer_size: load_symbol(lib, b"cusolverDnDgeqrf_bufferSize\0", "cuSOLVER")?,
            cgeqrf_buffer_size: load_symbol(lib, b"cusolverDnCgeqrf_bufferSize\0", "cuSOLVER")?,
            zgeqrf_buffer_size: load_symbol(lib, b"cusolverDnZgeqrf_bufferSize\0", "cuSOLVER")?,
            sgeqrf: load_symbol(lib, b"cusolverDnSgeqrf\0", "cuSOLVER")?,
            dgeqrf: load_symbol(lib, b"cusolverDnDgeqrf\0", "cuSOLVER")?,
            cgeqrf: load_symbol(lib, b"cusolverDnCgeqrf\0", "cuSOLVER")?,
            zgeqrf: load_symbol(lib, b"cusolverDnZgeqrf\0", "cuSOLVER")?,
            sorgqr_buffer_size: load_symbol(lib, b"cusolverDnSorgqr_bufferSize\0", "cuSOLVER")?,
            dorgqr_buffer_size: load_symbol(lib, b"cusolverDnDorgqr_bufferSize\0", "cuSOLVER")?,
            cungqr_buffer_size: load_symbol(lib, b"cusolverDnCungqr_bufferSize\0", "cuSOLVER")?,
            zungqr_buffer_size: load_symbol(lib, b"cusolverDnZungqr_bufferSize\0", "cuSOLVER")?,
            sorgqr: load_symbol(lib, b"cusolverDnSorgqr\0", "cuSOLVER")?,
            dorgqr: load_symbol(lib, b"cusolverDnDorgqr\0", "cuSOLVER")?,
            cungqr: load_symbol(lib, b"cusolverDnCungqr\0", "cuSOLVER")?,
            zungqr: load_symbol(lib, b"cusolverDnZungqr\0", "cuSOLVER")?,
            sgesvd_buffer_size: load_symbol(lib, b"cusolverDnSgesvd_bufferSize\0", "cuSOLVER")?,
            dgesvd_buffer_size: load_symbol(lib, b"cusolverDnDgesvd_bufferSize\0", "cuSOLVER")?,
            cgesvd_buffer_size: load_symbol(lib, b"cusolverDnCgesvd_bufferSize\0", "cuSOLVER")?,
            zgesvd_buffer_size: load_symbol(lib, b"cusolverDnZgesvd_bufferSize\0", "cuSOLVER")?,
            sgesvd: load_symbol(lib, b"cusolverDnSgesvd\0", "cuSOLVER")?,
            dgesvd: load_symbol(lib, b"cusolverDnDgesvd\0", "cuSOLVER")?,
            cgesvd: load_symbol(lib, b"cusolverDnCgesvd\0", "cuSOLVER")?,
            zgesvd: load_symbol(lib, b"cusolverDnZgesvd\0", "cuSOLVER")?,
            create_gesvdj_info: load_symbol(lib, b"cusolverDnCreateGesvdjInfo\0", "cuSOLVER")?,
            destroy_gesvdj_info: load_symbol(lib, b"cusolverDnDestroyGesvdjInfo\0", "cuSOLVER")?,
            sgesvdj_buffer_size: load_symbol(lib, b"cusolverDnSgesvdj_bufferSize\0", "cuSOLVER")?,
            dgesvdj_buffer_size: load_symbol(lib, b"cusolverDnDgesvdj_bufferSize\0", "cuSOLVER")?,
            cgesvdj_buffer_size: load_symbol(lib, b"cusolverDnCgesvdj_bufferSize\0", "cuSOLVER")?,
            zgesvdj_buffer_size: load_symbol(lib, b"cusolverDnZgesvdj_bufferSize\0", "cuSOLVER")?,
            sgesvdj: load_symbol(lib, b"cusolverDnSgesvdj\0", "cuSOLVER")?,
            dgesvdj: load_symbol(lib, b"cusolverDnDgesvdj\0", "cuSOLVER")?,
            cgesvdj: load_symbol(lib, b"cusolverDnCgesvdj\0", "cuSOLVER")?,
            zgesvdj: load_symbol(lib, b"cusolverDnZgesvdj\0", "cuSOLVER")?,
            ssyevd_buffer_size: load_symbol(lib, b"cusolverDnSsyevd_bufferSize\0", "cuSOLVER")?,
            dsyevd_buffer_size: load_symbol(lib, b"cusolverDnDsyevd_bufferSize\0", "cuSOLVER")?,
            cheevd_buffer_size: load_symbol(lib, b"cusolverDnCheevd_bufferSize\0", "cuSOLVER")?,
            zheevd_buffer_size: load_symbol(lib, b"cusolverDnZheevd_bufferSize\0", "cuSOLVER")?,
            ssyevd: load_symbol(lib, b"cusolverDnSsyevd\0", "cuSOLVER")?,
            dsyevd: load_symbol(lib, b"cusolverDnDsyevd\0", "cuSOLVER")?,
            cheevd: load_symbol(lib, b"cusolverDnCheevd\0", "cuSOLVER")?,
            zheevd: load_symbol(lib, b"cusolverDnZheevd\0", "cuSOLVER")?,
        })
    }
}

struct CublasVtable {
    create: CublasCreateFn,
    destroy: CublasDestroyFn,
    set_stream: CublasSetStreamFn,
    strsm: TrsmF32Fn,
    dtrsm: TrsmF64Fn,
    ctrsm: TrsmC32Fn,
    ztrsm: TrsmC64Fn,
    strsm_batched: TrsmBatchedF32Fn,
    dtrsm_batched: TrsmBatchedF64Fn,
    ctrsm_batched: TrsmBatchedC32Fn,
    ztrsm_batched: TrsmBatchedC64Fn,
}

impl CublasVtable {
    unsafe fn load(lib: &Library) -> Result<Self> {
        Ok(Self {
            create: load_symbol(lib, b"cublasCreate_v2\0", "cuBLAS")?,
            destroy: load_symbol(lib, b"cublasDestroy_v2\0", "cuBLAS")?,
            set_stream: load_symbol(lib, b"cublasSetStream_v2\0", "cuBLAS")?,
            strsm: load_symbol(lib, b"cublasStrsm_v2\0", "cuBLAS")?,
            dtrsm: load_symbol(lib, b"cublasDtrsm_v2\0", "cuBLAS")?,
            ctrsm: load_symbol(lib, b"cublasCtrsm_v2\0", "cuBLAS")?,
            ztrsm: load_symbol(lib, b"cublasZtrsm_v2\0", "cuBLAS")?,
            strsm_batched: load_symbol(lib, b"cublasStrsmBatched\0", "cuBLAS")?,
            dtrsm_batched: load_symbol(lib, b"cublasDtrsmBatched\0", "cuBLAS")?,
            ctrsm_batched: load_symbol(lib, b"cublasCtrsmBatched\0", "cuBLAS")?,
            ztrsm_batched: load_symbol(lib, b"cublasZtrsmBatched\0", "cuBLAS")?,
        })
    }
}

unsafe fn load_symbol<T: Copy>(
    lib: &Library,
    name: &[u8],
    library_name: &'static str,
) -> Result<T> {
    let symbol = lib.get::<T>(name).map_err(|err| {
        Error::backend_failure(
            "cubecl_linalg",
            format!(
                "failed to load {library_name} symbol {}: {err}",
                String::from_utf8_lossy(name).trim_end_matches('\0')
            ),
        )
    })?;
    Ok(*symbol)
}

type GetVersionFn = unsafe extern "C" fn(*mut i32) -> CusolverStatus;

struct CusolverLibrary {
    _lib: Library,
    vtable: CusolverVtable,
    /// cuSOLVER library version (e.g. 11403 for 11.4.3).
    _version: i32,
}

unsafe impl Send for CusolverLibrary {}
unsafe impl Sync for CusolverLibrary {}

impl CusolverLibrary {
    fn load() -> Result<Arc<Self>> {
        let paths = library_search_paths("TENFERRO_CUSOLVER_PATH", CUSOLVER_DEFAULT_PATHS);
        let mut errors = Vec::new();
        for path in &paths {
            let lib = match unsafe { Library::new(path) } {
                Ok(lib) => lib,
                Err(err) => {
                    errors.push(format!("{path}: {err}"));
                    continue;
                }
            };
            let vtable = unsafe { CusolverVtable::load(&lib) }?;
            // Query version if available
            let version = unsafe {
                if let Ok(get_ver) = lib.get::<GetVersionFn>(b"cusolverGetVersion\0") {
                    let mut ver = 0i32;
                    (*get_ver)(&mut ver);
                    ver
                } else {
                    0
                }
            };
            // Version is retained for diagnostics while keeping the loaded
            // library alive with the vtable.
            return Ok(Arc::new(Self {
                _lib: lib,
                vtable,
                _version: version,
            }));
        }

        Err(Error::backend_failure(
            "cubecl_linalg",
            format!(
                "failed to load cuSOLVER library (tried {}): {}",
                paths.join(", "),
                errors.join("; ")
            ),
        ))
    }

    fn check_status(
        &self,
        status: CusolverStatus,
        op: &'static str,
        call: &'static str,
    ) -> Result<()> {
        if status == CUSOLVER_STATUS_SUCCESS {
            return Ok(());
        }
        Err(Error::backend_failure(
            op,
            format!(
                "{call} failed with cuSOLVER {} ({status})",
                cusolver_status_name(status)
            ),
        ))
    }
}

struct CublasLibrary {
    _lib: Library,
    vtable: CublasVtable,
}

unsafe impl Send for CublasLibrary {}
unsafe impl Sync for CublasLibrary {}

impl CublasLibrary {
    fn load() -> Result<Arc<Self>> {
        let paths = library_search_paths("TENFERRO_CUBLAS_PATH", CUBLAS_DEFAULT_PATHS);
        let mut errors = Vec::new();
        for path in &paths {
            let lib = match unsafe { Library::new(path) } {
                Ok(lib) => lib,
                Err(err) => {
                    errors.push(format!("{path}: {err}"));
                    continue;
                }
            };
            let vtable = unsafe { CublasVtable::load(&lib) }?;
            return Ok(Arc::new(Self { _lib: lib, vtable }));
        }

        Err(Error::backend_failure(
            "triangular_solve",
            format!(
                "failed to load cuBLAS library (tried {}): {}",
                paths.join(", "),
                errors.join("; ")
            ),
        ))
    }

    fn check_status(
        &self,
        status: CublasStatus,
        op: &'static str,
        call: &'static str,
    ) -> Result<()> {
        if status == CUBLAS_STATUS_SUCCESS {
            return Ok(());
        }
        Err(Error::backend_failure(
            op,
            format!(
                "{call} failed with cuBLAS {} ({status})",
                cublas_status_name(status)
            ),
        ))
    }
}

fn cusolver_status_name(status: CusolverStatus) -> &'static str {
    match status {
        0 => "SUCCESS",
        1 => "NOT_INITIALIZED",
        2 => "ALLOC_FAILED",
        3 => "INVALID_VALUE",
        4 => "ARCH_MISMATCH",
        5 => "MAPPING_ERROR",
        6 => "EXECUTION_FAILED",
        7 => "INTERNAL_ERROR",
        8 => "MATRIX_TYPE_NOT_SUPPORTED",
        9 => "NOT_SUPPORTED",
        10 => "ZERO_PIVOT",
        11 => "INVALID_LICENSE",
        12 => "IRS_PARAMS_NOT_INITIALIZED",
        13 => "IRS_PARAMS_INVALID",
        14 => "IRS_INTERNAL_ERROR",
        15 => "IRS_NOT_SUPPORTED",
        16 => "IRS_OUT_OF_RANGE",
        17 => "IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES",
        _ => "UNKNOWN_STATUS",
    }
}

fn cublas_status_name(status: CublasStatus) -> &'static str {
    match status {
        0 => "SUCCESS",
        1 => "NOT_INITIALIZED",
        3 => "ALLOC_FAILED",
        7 => "INVALID_VALUE",
        8 => "ARCH_MISMATCH",
        11 => "MAPPING_ERROR",
        13 => "EXECUTION_FAILED",
        14 => "INTERNAL_ERROR",
        15 => "NOT_SUPPORTED",
        16 => "LICENSE_ERROR",
        _ => "UNKNOWN_STATUS",
    }
}

pub struct CusolverDnHandle {
    lib: Arc<CusolverLibrary>,
    raw: CusolverDnHandleRaw,
}

unsafe impl Send for CusolverDnHandle {}

pub struct GesvdjInfo<'a> {
    handle: &'a CusolverDnHandle,
    raw: GesvdjInfoRaw,
}

impl<'a> GesvdjInfo<'a> {
    fn raw(&self) -> GesvdjInfoRaw {
        self.raw
    }
}

impl Drop for GesvdjInfo<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.handle.lib.vtable.destroy_gesvdj_info)(self.raw);
        }
    }
}

impl CusolverDnHandle {
    pub fn load() -> Result<Self> {
        let lib = CusolverLibrary::load()?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe { (lib.vtable.create)(&mut raw) };
        lib.check_status(status, "cubecl_linalg", "cusolverDnCreate")?;
        Ok(Self { lib, raw })
    }

    pub fn set_stream(&self, stream: CudaStream, op: &'static str) -> Result<()> {
        let status = unsafe { (self.lib.vtable.set_stream)(self.raw, stream) };
        self.lib.check_status(status, op, "cusolverDnSetStream")
    }

    pub fn create_gesvdj_info(&self, op: &'static str) -> Result<GesvdjInfo<'_>> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { (self.lib.vtable.create_gesvdj_info)(&mut raw) };
        self.lib
            .check_status(status, op, "cusolverDnCreateGesvdjInfo")?;
        Ok(GesvdjInfo { handle: self, raw })
    }

    pub fn potrf_buffer_size(
        &self,
        dtype: CudaDataType,
        uplo: CublasFillMode,
        n: i32,
        a: *mut c_void,
        lda: i32,
        op: &'static str,
    ) -> Result<i32> {
        let mut lwork = 0;
        let status = unsafe {
            match dtype {
                CudaDataType::F32 => (self.lib.vtable.spotrf_buffer_size)(
                    self.raw,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    &mut lwork,
                ),
                CudaDataType::F64 => (self.lib.vtable.dpotrf_buffer_size)(
                    self.raw,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    &mut lwork,
                ),
                CudaDataType::Complex32 => (self.lib.vtable.cpotrf_buffer_size)(
                    self.raw,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    &mut lwork,
                ),
                CudaDataType::Complex64 => (self.lib.vtable.zpotrf_buffer_size)(
                    self.raw,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    &mut lwork,
                ),
            }
        };
        self.lib
            .check_status(status, op, "cusolverDn*potrf_bufferSize")?;
        Ok(lwork)
    }

    pub unsafe fn potrf(
        &self,
        dtype: CudaDataType,
        uplo: CublasFillMode,
        n: i32,
        a: *mut c_void,
        lda: i32,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.spotrf)(
                self.raw,
                uplo,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::F64 => (self.lib.vtable.dpotrf)(
                self.raw,
                uplo,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.cpotrf)(
                self.raw,
                uplo,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.zpotrf)(
                self.raw,
                uplo,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                lwork,
                info,
            ),
        };
        self.lib.check_status(status, op, "cusolverDn*potrf")
    }

    pub fn getrf_buffer_size(
        &self,
        dtype: CudaDataType,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        op: &'static str,
    ) -> Result<i32> {
        let mut lwork = 0;
        let status = unsafe {
            match dtype {
                CudaDataType::F32 => {
                    (self.lib.vtable.sgetrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
                CudaDataType::F64 => {
                    (self.lib.vtable.dgetrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
                CudaDataType::Complex32 => {
                    (self.lib.vtable.cgetrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
                CudaDataType::Complex64 => {
                    (self.lib.vtable.zgetrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
            }
        };
        self.lib
            .check_status(status, op, "cusolverDn*getrf_bufferSize")?;
        Ok(lwork)
    }

    pub unsafe fn getrf(
        &self,
        dtype: CudaDataType,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        workspace: *mut c_void,
        pivots: *mut i32,
        info: *mut i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.sgetrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                pivots,
                info,
            ),
            CudaDataType::F64 => (self.lib.vtable.dgetrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                pivots,
                info,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.cgetrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                pivots,
                info,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.zgetrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                workspace.cast(),
                pivots,
                info,
            ),
        };
        self.lib.check_status(status, op, "cusolverDn*getrf")
    }

    pub fn geqrf_buffer_size(
        &self,
        dtype: CudaDataType,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        op: &'static str,
    ) -> Result<i32> {
        let mut lwork = 0;
        let status = unsafe {
            match dtype {
                CudaDataType::F32 => {
                    (self.lib.vtable.sgeqrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
                CudaDataType::F64 => {
                    (self.lib.vtable.dgeqrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
                CudaDataType::Complex32 => {
                    (self.lib.vtable.cgeqrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
                CudaDataType::Complex64 => {
                    (self.lib.vtable.zgeqrf_buffer_size)(self.raw, m, n, a.cast(), lda, &mut lwork)
                }
            }
        };
        self.lib
            .check_status(status, op, "cusolverDn*geqrf_bufferSize")?;
        Ok(lwork)
    }

    pub unsafe fn geqrf(
        &self,
        dtype: CudaDataType,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        tau: *mut c_void,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.sgeqrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::F64 => (self.lib.vtable.dgeqrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.cgeqrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.zgeqrf)(
                self.raw,
                m,
                n,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
        };
        self.lib.check_status(status, op, "cusolverDn*geqrf")
    }

    pub fn orgqr_buffer_size(
        &self,
        dtype: CudaDataType,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i32,
        tau: *const c_void,
        op: &'static str,
    ) -> Result<i32> {
        let mut lwork = 0;
        let status = unsafe {
            match dtype {
                CudaDataType::F32 => (self.lib.vtable.sorgqr_buffer_size)(
                    self.raw,
                    m,
                    n,
                    k,
                    a.cast(),
                    lda,
                    tau.cast(),
                    &mut lwork,
                ),
                CudaDataType::F64 => (self.lib.vtable.dorgqr_buffer_size)(
                    self.raw,
                    m,
                    n,
                    k,
                    a.cast(),
                    lda,
                    tau.cast(),
                    &mut lwork,
                ),
                CudaDataType::Complex32 => (self.lib.vtable.cungqr_buffer_size)(
                    self.raw,
                    m,
                    n,
                    k,
                    a.cast(),
                    lda,
                    tau.cast(),
                    &mut lwork,
                ),
                CudaDataType::Complex64 => (self.lib.vtable.zungqr_buffer_size)(
                    self.raw,
                    m,
                    n,
                    k,
                    a.cast(),
                    lda,
                    tau.cast(),
                    &mut lwork,
                ),
            }
        };
        self.lib
            .check_status(status, op, "cusolverDn*orgqr_bufferSize")?;
        Ok(lwork)
    }

    pub unsafe fn orgqr(
        &self,
        dtype: CudaDataType,
        m: i32,
        n: i32,
        k: i32,
        a: *mut c_void,
        lda: i32,
        tau: *const c_void,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.sorgqr)(
                self.raw,
                m,
                n,
                k,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::F64 => (self.lib.vtable.dorgqr)(
                self.raw,
                m,
                n,
                k,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.cungqr)(
                self.raw,
                m,
                n,
                k,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.zungqr)(
                self.raw,
                m,
                n,
                k,
                a.cast(),
                lda,
                tau.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
        };
        self.lib.check_status(status, op, "cusolverDn*orgqr")
    }

    pub fn gesvd_buffer_size(
        &self,
        dtype: CudaDataType,
        m: i32,
        n: i32,
        op: &'static str,
    ) -> Result<i32> {
        let mut lwork = 0;
        let status = unsafe {
            match dtype {
                CudaDataType::F32 => {
                    (self.lib.vtable.sgesvd_buffer_size)(self.raw, m, n, &mut lwork)
                }
                CudaDataType::F64 => {
                    (self.lib.vtable.dgesvd_buffer_size)(self.raw, m, n, &mut lwork)
                }
                CudaDataType::Complex32 => {
                    (self.lib.vtable.cgesvd_buffer_size)(self.raw, m, n, &mut lwork)
                }
                CudaDataType::Complex64 => {
                    (self.lib.vtable.zgesvd_buffer_size)(self.raw, m, n, &mut lwork)
                }
            }
        };
        self.lib
            .check_status(status, op, "cusolverDn*gesvd_bufferSize")?;
        Ok(lwork)
    }

    pub fn gesvdj_buffer_size(
        &self,
        dtype: CudaDataType,
        jobz: CusolverEigMode,
        econ: i32,
        m: i32,
        n: i32,
        a: *const c_void,
        lda: i32,
        s: *const c_void,
        u: *const c_void,
        ldu: i32,
        v: *const c_void,
        ldv: i32,
        params: &GesvdjInfo<'_>,
        op: &'static str,
    ) -> Result<i32> {
        let mut lwork = 0;
        let status = unsafe {
            match dtype {
                CudaDataType::F32 => (self.lib.vtable.sgesvdj_buffer_size)(
                    self.raw,
                    jobz,
                    econ,
                    m,
                    n,
                    a.cast(),
                    lda,
                    s.cast(),
                    u.cast(),
                    ldu,
                    v.cast(),
                    ldv,
                    &mut lwork,
                    params.raw(),
                ),
                CudaDataType::F64 => (self.lib.vtable.dgesvdj_buffer_size)(
                    self.raw,
                    jobz,
                    econ,
                    m,
                    n,
                    a.cast(),
                    lda,
                    s.cast(),
                    u.cast(),
                    ldu,
                    v.cast(),
                    ldv,
                    &mut lwork,
                    params.raw(),
                ),
                CudaDataType::Complex32 => (self.lib.vtable.cgesvdj_buffer_size)(
                    self.raw,
                    jobz,
                    econ,
                    m,
                    n,
                    a.cast(),
                    lda,
                    s.cast(),
                    u.cast(),
                    ldu,
                    v.cast(),
                    ldv,
                    &mut lwork,
                    params.raw(),
                ),
                CudaDataType::Complex64 => (self.lib.vtable.zgesvdj_buffer_size)(
                    self.raw,
                    jobz,
                    econ,
                    m,
                    n,
                    a.cast(),
                    lda,
                    s.cast(),
                    u.cast(),
                    ldu,
                    v.cast(),
                    ldv,
                    &mut lwork,
                    params.raw(),
                ),
            }
        };
        self.lib
            .check_status(status, op, "cusolverDn*gesvdj_bufferSize")?;
        Ok(lwork)
    }

    pub unsafe fn gesvd(
        &self,
        dtype: CudaDataType,
        jobu: c_char,
        jobvt: c_char,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        s: *mut c_void,
        u: *mut c_void,
        ldu: i32,
        vt: *mut c_void,
        ldvt: i32,
        workspace: *mut c_void,
        lwork: i32,
        rwork: *mut c_void,
        info: *mut i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.sgesvd)(
                self.raw,
                jobu,
                jobvt,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                vt.cast(),
                ldvt,
                workspace.cast(),
                lwork,
                rwork.cast(),
                info,
            ),
            CudaDataType::F64 => (self.lib.vtable.dgesvd)(
                self.raw,
                jobu,
                jobvt,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                vt.cast(),
                ldvt,
                workspace.cast(),
                lwork,
                rwork.cast(),
                info,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.cgesvd)(
                self.raw,
                jobu,
                jobvt,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                vt.cast(),
                ldvt,
                workspace.cast(),
                lwork,
                rwork.cast(),
                info,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.zgesvd)(
                self.raw,
                jobu,
                jobvt,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                vt.cast(),
                ldvt,
                workspace.cast(),
                lwork,
                rwork.cast(),
                info,
            ),
        };
        self.lib.check_status(status, op, "cusolverDn*gesvd")
    }

    pub unsafe fn gesvdj(
        &self,
        dtype: CudaDataType,
        jobz: CusolverEigMode,
        econ: i32,
        m: i32,
        n: i32,
        a: *mut c_void,
        lda: i32,
        s: *mut c_void,
        u: *mut c_void,
        ldu: i32,
        v: *mut c_void,
        ldv: i32,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
        params: &GesvdjInfo<'_>,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.sgesvdj)(
                self.raw,
                jobz,
                econ,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                v.cast(),
                ldv,
                workspace.cast(),
                lwork,
                info,
                params.raw(),
            ),
            CudaDataType::F64 => (self.lib.vtable.dgesvdj)(
                self.raw,
                jobz,
                econ,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                v.cast(),
                ldv,
                workspace.cast(),
                lwork,
                info,
                params.raw(),
            ),
            CudaDataType::Complex32 => (self.lib.vtable.cgesvdj)(
                self.raw,
                jobz,
                econ,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                v.cast(),
                ldv,
                workspace.cast(),
                lwork,
                info,
                params.raw(),
            ),
            CudaDataType::Complex64 => (self.lib.vtable.zgesvdj)(
                self.raw,
                jobz,
                econ,
                m,
                n,
                a.cast(),
                lda,
                s.cast(),
                u.cast(),
                ldu,
                v.cast(),
                ldv,
                workspace.cast(),
                lwork,
                info,
                params.raw(),
            ),
        };
        self.lib.check_status(status, op, "cusolverDn*gesvdj")
    }

    pub fn syevd_buffer_size(
        &self,
        dtype: CudaDataType,
        jobz: CusolverEigMode,
        uplo: CublasFillMode,
        n: i32,
        a: *const c_void,
        lda: i32,
        w: *const c_void,
        op: &'static str,
    ) -> Result<i32> {
        let mut lwork = 0;
        let status = unsafe {
            match dtype {
                CudaDataType::F32 => (self.lib.vtable.ssyevd_buffer_size)(
                    self.raw,
                    jobz,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    w.cast(),
                    &mut lwork,
                ),
                CudaDataType::F64 => (self.lib.vtable.dsyevd_buffer_size)(
                    self.raw,
                    jobz,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    w.cast(),
                    &mut lwork,
                ),
                CudaDataType::Complex32 => (self.lib.vtable.cheevd_buffer_size)(
                    self.raw,
                    jobz,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    w.cast(),
                    &mut lwork,
                ),
                CudaDataType::Complex64 => (self.lib.vtable.zheevd_buffer_size)(
                    self.raw,
                    jobz,
                    uplo,
                    n,
                    a.cast(),
                    lda,
                    w.cast(),
                    &mut lwork,
                ),
            }
        };
        self.lib
            .check_status(status, op, "cusolverDn*syevd_bufferSize")?;
        Ok(lwork)
    }

    pub unsafe fn syevd(
        &self,
        dtype: CudaDataType,
        jobz: CusolverEigMode,
        uplo: CublasFillMode,
        n: i32,
        a: *mut c_void,
        lda: i32,
        w: *mut c_void,
        workspace: *mut c_void,
        lwork: i32,
        info: *mut i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.ssyevd)(
                self.raw,
                jobz,
                uplo,
                n,
                a.cast(),
                lda,
                w.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::F64 => (self.lib.vtable.dsyevd)(
                self.raw,
                jobz,
                uplo,
                n,
                a.cast(),
                lda,
                w.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.cheevd)(
                self.raw,
                jobz,
                uplo,
                n,
                a.cast(),
                lda,
                w.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.zheevd)(
                self.raw,
                jobz,
                uplo,
                n,
                a.cast(),
                lda,
                w.cast(),
                workspace.cast(),
                lwork,
                info,
            ),
        };
        self.lib.check_status(status, op, "cusolverDn*syevd")
    }
}

impl Drop for CusolverDnHandle {
    fn drop(&mut self) {
        let _ = unsafe { (self.lib.vtable.destroy)(self.raw) };
    }
}

pub struct CublasHandle {
    lib: Arc<CublasLibrary>,
    raw: CublasHandleRaw,
}

unsafe impl Send for CublasHandle {}

impl CublasHandle {
    pub fn load() -> Result<Self> {
        let lib = CublasLibrary::load()?;
        let mut raw = std::ptr::null_mut();
        let status = unsafe { (lib.vtable.create)(&mut raw) };
        lib.check_status(status, "triangular_solve", "cublasCreate_v2")?;
        Ok(Self { lib, raw })
    }

    pub fn set_stream(&self, stream: CudaStream, op: &'static str) -> Result<()> {
        let status = unsafe { (self.lib.vtable.set_stream)(self.raw, stream) };
        self.lib.check_status(status, op, "cublasSetStream_v2")
    }

    pub unsafe fn trsm(
        &self,
        dtype: CudaDataType,
        side: CublasSideMode,
        uplo: CublasFillMode,
        trans: CublasOperation,
        diag: CublasDiagType,
        m: i32,
        n: i32,
        alpha: *const c_void,
        a: *const c_void,
        lda: i32,
        b: *mut c_void,
        ldb: i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.strsm)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
            ),
            CudaDataType::F64 => (self.lib.vtable.dtrsm)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.ctrsm)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.ztrsm)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
            ),
        };
        self.lib.check_status(status, op, "cublas*trsm_v2")
    }

    pub unsafe fn trsm_batched(
        &self,
        dtype: CudaDataType,
        side: CublasSideMode,
        uplo: CublasFillMode,
        trans: CublasOperation,
        diag: CublasDiagType,
        m: i32,
        n: i32,
        alpha: *const c_void,
        a_array: *const c_void,
        lda: i32,
        b_array: *mut c_void,
        ldb: i32,
        batch_count: i32,
        op: &'static str,
    ) -> Result<()> {
        let status = match dtype {
            CudaDataType::F32 => (self.lib.vtable.strsm_batched)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a_array.cast(),
                lda,
                b_array.cast(),
                ldb,
                batch_count,
            ),
            CudaDataType::F64 => (self.lib.vtable.dtrsm_batched)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a_array.cast(),
                lda,
                b_array.cast(),
                ldb,
                batch_count,
            ),
            CudaDataType::Complex32 => (self.lib.vtable.ctrsm_batched)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a_array.cast(),
                lda,
                b_array.cast(),
                ldb,
                batch_count,
            ),
            CudaDataType::Complex64 => (self.lib.vtable.ztrsm_batched)(
                self.raw,
                side,
                uplo,
                trans,
                diag,
                m,
                n,
                alpha.cast(),
                a_array.cast(),
                lda,
                b_array.cast(),
                ldb,
                batch_count,
            ),
        };
        self.lib.check_status(status, op, "cublas*trsmBatched")
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        let _ = unsafe { (self.lib.vtable.destroy)(self.raw) };
    }
}

pub struct CudaLinalgHandles {
    cusolver: CusolverDnHandle,
    cublas: CublasHandle,
}

impl CudaLinalgHandles {
    pub fn load() -> Result<Self> {
        Ok(Self {
            cusolver: CusolverDnHandle::load()?,
            cublas: CublasHandle::load()?,
        })
    }

    pub fn cusolver(&self) -> &CusolverDnHandle {
        &self.cusolver
    }

    pub fn cublas(&self) -> &CublasHandle {
        &self.cublas
    }
}
