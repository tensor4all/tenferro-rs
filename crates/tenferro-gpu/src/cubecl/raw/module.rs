//! CUDA module loading, function lookup, and launch validation helpers
//! (issue #1597).
//!
//! The public [`Module`](super::Module), [`Function`](super::Function), and
//! [`LaunchConfig`](super::LaunchConfig) types are defined in the parent
//! [`super`] module; this file holds the driver-bridge helpers that construct
//! them.
//!
//! Launch safety model
//! -------------------
//!
//! `Session::launch` is an `unsafe fn` (see
//! [`crate::cubecl::raw::Session::launch`]): it validates grid/block/shared-
//! memory limits, symbol existence, and argument address sanity as safe
//! checks, but the ABI, argument order, read/write ranges, aliasing,
//! initialization, and async-liveness invariants are the caller's contract.
//! The success path does not synchronize; `Session::synchronize` is the only
//! host barrier.

use std::ffi::CString;
use std::rc::Rc;

use super::{Function, Module, ModuleInner};

/// Load a CUDA module from a raw data image and wrap it.
///
/// `image` must point to valid PTX (NUL-terminated text) or CUBIN (binary)
/// data. The driver copies the image, so the source can be freed afterwards.
///
/// # Errors
///
/// Returns the driver's typed error via [`crate::Error::BackendSource`] when
/// the image cannot be loaded (unsupported format, architecture mismatch, or
/// corrupt data).
pub(crate) fn load_module_data(
    image: *const std::ffi::c_void,
    runtime: crate::cubecl::CudaRuntime,
    op: &'static str,
) -> crate::Result<Module> {
    // SAFETY: `image` points to a well-formed PTX/CUBIN image and the tenferro
    // primary context is current on this thread (the raw session guarantees it).
    let handle = unsafe { cudarc::driver::result::module::load_data(image) }
        .map_err(|err| crate::Error::backend_source(op, err))?;
    Module::from_handle(op, handle, runtime)
}

/// Look up a kernel entry point within a loaded module.
///
/// # Errors
///
/// Returns the driver's typed error (missing symbol / module invalid) via
/// [`crate::Error::BackendSource`].
pub(crate) fn module_function(
    inner: &Rc<ModuleInner>,
    name: &str,
    op: &'static str,
) -> crate::Result<Function> {
    let name_c = CString::new(name)
        .map_err(|_| crate::Error::invalid_argument(op, "symbol", "symbol name contains NUL"))?;
    // SAFETY: `inner.handle` is a valid, loaded module handle kept alive by
    // the `Arc` this function retains.
    let handle = unsafe { cudarc::driver::result::module::get_function(inner.handle, name_c) }
        .map_err(|err| crate::Error::backend_source(op, err))?;
    Ok(Function {
        handle,
        module: Rc::clone(inner),
    })
}

/// Validate launch geometry against the CUDA hardware limits.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] when any dimension is zero, the block
/// exceeds 1024 threads, or a grid dimension exceeds `2^31 - 1`.
pub(crate) fn validate_launch_config(config: &super::LaunchConfig) -> crate::Result<()> {
    const MAX_BLOCK_THREADS: u32 = 1024;
    const MAX_GRID_X: u32 = 2_147_483_647;

    if config.grid[0] == 0 || config.grid[1] == 0 || config.grid[2] == 0 {
        return Err(crate::Error::invalid_argument(
            "launch.validate",
            "grid",
            "grid dimensions must be non-zero",
        ));
    }
    if config.block[0] == 0 || config.block[1] == 0 || config.block[2] == 0 {
        return Err(crate::Error::invalid_argument(
            "launch.validate",
            "block",
            "block dimensions must be non-zero",
        ));
    }
    let threads = config.block[0]
        .checked_mul(config.block[1])
        .and_then(|v| v.checked_mul(config.block[2]))
        .ok_or_else(|| {
            crate::Error::invalid_argument("launch.validate", "block", "block product overflow")
        })?;
    if threads > MAX_BLOCK_THREADS {
        return Err(crate::Error::invalid_argument(
            "launch.validate",
            "block",
            format!("block has {threads} threads; CUDA allows at most {MAX_BLOCK_THREADS}"),
        ));
    }
    if config.grid[0] > MAX_GRID_X {
        return Err(crate::Error::invalid_argument(
            "launch.validate",
            "grid",
            format!(
                "grid.x {} exceeds the CUDA limit {MAX_GRID_X}",
                config.grid[0]
            ),
        ));
    }
    Ok(())
}
