//! Dynamic cuFFT bindings owned by the FFT operation family.
//!
//! The function signatures and status values follow NVIDIA's cuFFT API
//! reference and the installed `cufft.h` ABI (`cufftCreate`,
//! `cufftMakePlanMany64`, `cufftSetWorkArea`, `cufftSetStream`, and the six
//! execution entry points). The dynamic library is retained next to the
//! copied function table so no symbol can outlive its library handle.

use std::ffi::{c_void, OsStr, OsString};
use std::sync::Arc;

use libloading::Library;

use super::error::CudaFftError;

const DEFAULT_CUFFT_PATHS: &[&str] = &["libcufft.so.12", "libcufft.so.11", "libcufft.so"];

pub(crate) type CufftHandle = i32;
pub(crate) type CufftStatus = i32;

pub(crate) const CUFFT_SUCCESS: CufftStatus = 0;
pub(crate) const CUFFT_INVALID_PLAN: CufftStatus = 1;
pub(crate) const CUFFT_ALLOC_FAILED: CufftStatus = 2;
pub(crate) const CUFFT_INVALID_TYPE: CufftStatus = 3;
pub(crate) const CUFFT_INVALID_VALUE: CufftStatus = 4;
pub(crate) const CUFFT_INTERNAL_ERROR: CufftStatus = 5;
pub(crate) const CUFFT_EXEC_FAILED: CufftStatus = 6;
pub(crate) const CUFFT_SETUP_FAILED: CufftStatus = 7;
pub(crate) const CUFFT_INVALID_SIZE: CufftStatus = 8;
pub(crate) const CUFFT_UNALIGNED_DATA: CufftStatus = 9;

pub(crate) const CUFFT_FORWARD: i32 = -1;
pub(crate) const CUFFT_INVERSE: i32 = 1;
pub(crate) const CUFFT_R2C: i32 = 0x2a;
pub(crate) const CUFFT_C2R: i32 = 0x2c;
pub(crate) const CUFFT_C2C: i32 = 0x29;
pub(crate) const CUFFT_D2Z: i32 = 0x6a;
pub(crate) const CUFFT_Z2D: i32 = 0x6c;
pub(crate) const CUFFT_Z2Z: i32 = 0x69;

pub(crate) type CufftCreateFn = unsafe extern "C" fn(*mut CufftHandle) -> CufftStatus;
pub(crate) type CufftSetAutoAllocationFn = unsafe extern "C" fn(CufftHandle, i32) -> CufftStatus;
pub(crate) type CufftMakePlanMany64Fn = unsafe extern "C" fn(
    CufftHandle,
    i32,
    *mut i64,
    *mut i64,
    i64,
    i64,
    *mut i64,
    i64,
    i64,
    i32,
    i64,
    *mut usize,
) -> CufftStatus;
pub(crate) type CufftSetWorkAreaFn = unsafe extern "C" fn(CufftHandle, *mut c_void) -> CufftStatus;
pub(crate) type CufftSetStreamFn = unsafe extern "C" fn(CufftHandle, *mut c_void) -> CufftStatus;
pub(crate) type CufftExecC2cFn =
    unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void, i32) -> CufftStatus;
pub(crate) type CufftExecR2cFn =
    unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void) -> CufftStatus;
pub(crate) type CufftExecC2rFn =
    unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void) -> CufftStatus;
pub(crate) type CufftExecZ2zFn =
    unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void, i32) -> CufftStatus;
pub(crate) type CufftExecD2zFn =
    unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void) -> CufftStatus;
pub(crate) type CufftExecZ2dFn =
    unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void) -> CufftStatus;
pub(crate) type CufftDestroyFn = unsafe extern "C" fn(CufftHandle) -> CufftStatus;

/// The complete cuFFT function table used by one dynamically loaded library.
#[derive(Clone, Copy)]
pub(crate) struct CufftApi {
    pub(crate) create: CufftCreateFn,
    pub(crate) set_auto_allocation: CufftSetAutoAllocationFn,
    pub(crate) make_plan_many_64: CufftMakePlanMany64Fn,
    pub(crate) set_work_area: CufftSetWorkAreaFn,
    pub(crate) set_stream: CufftSetStreamFn,
    pub(crate) exec_c2c: CufftExecC2cFn,
    pub(crate) exec_r2c: CufftExecR2cFn,
    pub(crate) exec_c2r: CufftExecC2rFn,
    pub(crate) exec_z2z: CufftExecZ2zFn,
    pub(crate) exec_d2z: CufftExecD2zFn,
    pub(crate) exec_z2d: CufftExecZ2dFn,
    pub(crate) destroy: CufftDestroyFn,
}

impl CufftApi {
    fn load(library: &Library) -> Result<Self, CudaFftError> {
        Ok(Self {
            create: load_symbol(library, b"cufftCreate\0")?,
            set_auto_allocation: load_symbol(library, b"cufftSetAutoAllocation\0")?,
            make_plan_many_64: load_symbol(library, b"cufftMakePlanMany64\0")?,
            set_work_area: load_symbol(library, b"cufftSetWorkArea\0")?,
            set_stream: load_symbol(library, b"cufftSetStream\0")?,
            exec_c2c: load_symbol(library, b"cufftExecC2C\0")?,
            exec_r2c: load_symbol(library, b"cufftExecR2C\0")?,
            exec_c2r: load_symbol(library, b"cufftExecC2R\0")?,
            exec_z2z: load_symbol(library, b"cufftExecZ2Z\0")?,
            exec_d2z: load_symbol(library, b"cufftExecD2Z\0")?,
            exec_z2d: load_symbol(library, b"cufftExecZ2D\0")?,
            destroy: load_symbol(library, b"cufftDestroy\0")?,
        })
    }
}

fn load_symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, CudaFftError> {
    let symbol_name = String::from_utf8_lossy(name)
        .trim_end_matches('\0')
        .to_owned();
    // SAFETY: `name` is a NUL-terminated cuFFT symbol and `T` is the exact
    // function-pointer type declared for that symbol above. `library` remains
    // alive in the returned `CufftLibrary`.
    let symbol = unsafe { library.get::<T>(name) }.map_err(|source| CudaFftError::SymbolLoad {
        name: symbol_name,
        source,
    })?;
    Ok(*symbol)
}

/// Dynamically loaded cuFFT symbols and the library handle that owns them.
pub(crate) struct CufftLibrary {
    _library: Option<Library>,
    pub(crate) api: CufftApi,
}

// SAFETY: all copied function pointers remain valid because `library` is
// retained by this value; calls carry explicit plan, pointer, and stream state.
unsafe impl Send for CufftLibrary {}
// SAFETY: the table is immutable and cuFFT state is synchronized by the owning
// FFT execution session before mutable plan calls.
unsafe impl Sync for CufftLibrary {}

impl CufftLibrary {
    /// Load one cuFFT library and all symbols required by the FFT backend.
    pub(crate) fn load() -> Result<Arc<Self>, CudaFftError> {
        Self::load_from_paths(cufft_library_candidates(
            std::env::var_os("TENFERRO_CUFFT_PATH").as_deref(),
        ))
    }

    #[cfg(test)]
    pub(crate) fn load_from_paths_for_tests(
        paths: Vec<OsString>,
    ) -> Result<Arc<Self>, CudaFftError> {
        Self::load_from_paths(paths)
    }

    fn load_from_paths(paths: Vec<OsString>) -> Result<Arc<Self>, CudaFftError> {
        let mut attempts = Vec::new();
        let mut last_source = None;
        for path in &paths {
            // SAFETY: `Library` owns the handle and remains stored in
            // `CufftLibrary` for the lifetime of every loaded symbol.
            let library = match unsafe { Library::new(path) } {
                Ok(library) => library,
                Err(source) => {
                    attempts.push(format!("{}: {source}", path.to_string_lossy()));
                    last_source = Some(source);
                    continue;
                }
            };
            let api = CufftApi::load(&library)?;
            return Ok(Arc::new(Self {
                _library: Some(library),
                api,
            }));
        }

        let paths_text = paths
            .iter()
            .map(|path| path.to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ");
        let attempts_text = attempts.join("; ");
        let Some(source) = last_source else {
            return Err(CudaFftError::NoLibraryCandidates);
        };
        Err(CudaFftError::LibraryLoad {
            paths: paths_text,
            attempts: attempts_text,
            source,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_api_for_tests(api: CufftApi) -> Arc<Self> {
        Arc::new(Self {
            _library: None,
            api,
        })
    }
}

/// Build the ordered cuFFT soname/path candidates without reading process
/// environment state in callers or tests.
pub(crate) fn cufft_library_candidates(override_path: Option<&OsStr>) -> Vec<OsString> {
    let mut candidates = override_path
        .into_iter()
        .flat_map(std::env::split_paths)
        .filter(|path| !path.as_os_str().is_empty())
        .map(|path| path.into_os_string())
        .collect::<Vec<_>>();
    candidates.extend(DEFAULT_CUFFT_PATHS.iter().map(OsString::from));
    candidates
}

pub(crate) fn map_cufft_status(
    function: &'static str,
    status: CufftStatus,
) -> Result<(), CudaFftError> {
    if status == CUFFT_SUCCESS {
        Ok(())
    } else {
        Err(CudaFftError::CufftStatus { function, status })
    }
}
