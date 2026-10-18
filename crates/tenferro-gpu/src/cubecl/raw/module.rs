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
use std::sync::Arc;

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
    inner: &Arc<ModuleInner>,
    name: &str,
    op: &'static str,
) -> crate::Result<Function> {
    let name_c = CString::new(name)
        .map_err(|_| crate::Error::invalid_argument(op, "symbol", "symbol name contains NUL"))?;
    // SAFETY: `inner` owns a valid, loaded module handle kept alive by the
    // `Arc` this function retains, and `Module::function` activated its owning
    // context for this lookup.
    let handle = unsafe { cudarc::driver::result::module::get_function(inner.handle(), name_c) }
        .map_err(|err| crate::Error::backend_source(op, err))?;
    Ok(Function {
        handle: handle as usize,
        module: Arc::clone(inner),
    })
}

/// Dynamic shared memory a launch may request without the per-function opt-in
/// (48 KiB, the driver default since Volta).
const DEFAULT_MAX_DYNAMIC_SHARED_BYTES: u32 = 48 * 1024;

/// Whether a launch requesting `bytes` of dynamic shared memory must raise the
/// per-function opt-in, given `current`, the value already recorded on the
/// handle.
///
/// The attribute only grows, so a request at or below the recorded value needs
/// no driver call; requests at or below the 48 KiB default never do.
pub(crate) fn needs_shared_memory_opt_in(current: u32, bytes: u32) -> bool {
    bytes > DEFAULT_MAX_DYNAMIC_SHARED_BYTES && current < bytes
}

/// Raise `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` to `bytes` when a
/// launch asks for more dynamic shared memory than the 48 KiB default.
///
/// The attribute only grows, so the last value recorded for the underlying
/// `CUfunction` is reused: a repeated or smaller request above the default
/// still needs no driver call (#2085). The record lives on the module owner and
/// is keyed by the function handle, so two [`Module::function`] lookups of the
/// same symbol share one budget and cannot shrink each other's attribute. The
/// requested value is not clamped; a value above the device's
/// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` is rejected by the
/// driver and returned as a typed backend error.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] when `bytes` does not fit the driver's
/// `i32` attribute value, or [`crate::Error::BackendSource`] when the context
/// cannot be activated or the driver rejects the attribute.
pub(crate) fn ensure_dynamic_shared_memory(function: &Function, bytes: u32) -> crate::Result<()> {
    // Below the no-opt-in default there is nothing to raise, so the common
    // case pays no lock.
    if !needs_shared_memory_opt_in(0, bytes) {
        return Ok(());
    }
    let value = i32::try_from(bytes).map_err(|_| {
        crate::Error::invalid_argument(
            "raw.launch",
            "shared_mem_bytes",
            format!("dynamic shared memory of {bytes} bytes does not fit the driver's i32 attribute value"),
        )
    })?;

    // A function handle is shared by every lookup of the same symbol and across
    // threads, so the module owner's table holds the recorded value and its
    // lock serializes the read-decide-raise sequence. Without that, a smaller
    // concurrent request could lower the attribute after a larger one raised
    // it, and a later large launch would fail.
    let function_handle = function.handle;
    let mut opt_in = function.module.lock_shared_memory_opt_in();
    let recorded = opt_in.get(&function_handle).copied().unwrap_or(0);
    if !needs_shared_memory_opt_in(recorded, bytes) {
        return Ok(());
    }

    // Activate the context that owns the module instead of relying on the raw
    // scope that loaded the function still being current: a `Function` may be
    // used outside its creating scope and thread later (#2085). The guard
    // restores the previous device/context, including "no current context".
    let _guard = function.module.enter_context_if_needed("raw.launch")?;
    // SAFETY: the function handle belongs to the loaded module kept alive by
    // `function`, and its owning context is current on this thread.
    let set = unsafe {
        cudarc::driver::result::function::set_function_attribute(
            function.handle(),
            cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            value,
        )
    };
    set.map_err(|err| crate::Error::backend_source("raw.launch", err))?;
    opt_in.insert(function_handle, bytes);
    Ok(())
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
