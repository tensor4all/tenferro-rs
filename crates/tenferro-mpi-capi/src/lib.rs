//! C entry points for tenferro's MPI transport on a host-owned communicator.
//!
//! See `docs/design/mpi-native-abi.md` (§2 handles, §3 ownership, D4, D6, D10-D12).
//!
//! # Contract in one paragraph
//!
//! The host passes its raw `MPI_Comm` handle widened to `uint64_t`. tenferro never
//! initializes or finalizes MPI, never duplicates or frees a communicator and never changes
//! its error handler. It does not claim to verify that the handle is the communicator the host
//! meant: no local check can tell two live communicators of one size apart.
//!
//! Every `extern "C"` entry runs inside `catch_unwind`. A panic returns
//! `TFE_INTERNAL_ERROR` with its message in the thread-local last error, and **poisons** the
//! communicator object.

#![allow(non_camel_case_types)]
#![deny(missing_docs)]
// Empty without the `runtime` feature, so the root workspace lanes build no MPI binding.
#![cfg(feature = "runtime")]

#[cfg(panic = "abort")]
compile_error!(
    "tenferro-mpi-capi must be built with `panic = \"unwind\"`: every panic stub of the \
     native MPI surface would otherwise abort the host process (D10)"
);

use std::any::Any;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use tenferro_mpi::mpi::ffi;
use tenferro_mpi::mpi::raw::FromRaw;
use tenferro_mpi::mpi::topology::SimpleCommunicator;
use tenferro_mpi::mpi::traits::Communicator;
use tenferro_mpi::{Error, MAX_WIRE_RANK};
use tenferro_tensor::{Complex32, Complex64, DType, DynRank, Host, Tensor, TypedTensor};

/// Status of a C entry point. `0` is success; see the README for the codes.
pub type tfe_status_t = i32;
/// Success.
pub const TFE_SUCCESS: tfe_status_t = 0;
/// A null pointer, a malformed argument, or a value the transport refuses.
pub const TFE_INVALID_ARGUMENT: tfe_status_t = -1;
/// A shape that disagrees with the data or with the peer.
pub const TFE_SHAPE_MISMATCH: tfe_status_t = -2;
/// A bug, or a caught panic. The communicator object is poisoned.
pub const TFE_INTERNAL_ERROR: tfe_status_t = -3;
/// The output buffer is too small; `needed` says how large it must be.
pub const TFE_BUFFER_TOO_SMALL: tfe_status_t = -4;
/// The MPI library is outside the supported families, ambiguous, or mismatched.
pub const TFE_MPI_LIBRARY: tfe_status_t = -10;
/// `MPI_Init` has not run in the host.
pub const TFE_MPI_NOT_INITIALIZED: tfe_status_t = -11;
/// The host finalized MPI; the communicator is no longer usable.
pub const TFE_MPI_FINALIZED: tfe_status_t = -12;
/// An MPI call returned an error code (the communicator's handler returns).
pub const TFE_MPI_ERROR: tfe_status_t = -13;
/// The two ranks disagree on the transport protocol (peer rejected, aborted, bad frame).
pub const TFE_MPI_PROTOCOL: tfe_status_t = -14;
/// The call came from a thread the MPI thread level does not allow.
pub const TFE_MPI_THREAD: tfe_status_t = -15;
/// No MPI library is loaded in the process yet. Not cached: retry after the host initializes MPI.
pub const TFE_MPI_NO_LIBRARY: tfe_status_t = -16;

struct CErr {
    code: tfe_status_t,
    message: String,
}

fn err(code: tfe_status_t, message: impl Into<String>) -> CErr {
    CErr {
        code,
        message: message.into(),
    }
}

thread_local! {
    static LAST_ERROR: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

fn set_last_error(message: &str) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message.to_string());
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(hit) = payload.downcast_ref::<ffi::UnsupportedMpiCall>() {
        return hit.to_string();
    }
    if let Some(text) = payload.downcast_ref::<&str>() {
        return (*text).to_string();
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return text.clone();
    }
    "a panic with a non-string payload".to_string()
}

fn finish(status: *mut tfe_status_t, result: Result<(), CErr>) -> tfe_status_t {
    let code = match result {
        Ok(()) => TFE_SUCCESS,
        Err(e) => {
            set_last_error(&e.message);
            e.code
        }
    };
    if !status.is_null() {
        // SAFETY: the host passed a writable status pointer (checked non-null).
        unsafe { *status = code };
    }
    code
}

/// Run `f` so that no panic crosses the C boundary (D10).
fn guarded(status: *mut tfe_status_t, f: impl FnOnce() -> Result<(), CErr>) -> tfe_status_t {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => finish(status, result),
        Err(payload) => finish(
            status,
            Err(err(TFE_INTERNAL_ERROR, panic_message(&*payload))),
        ),
    }
}

fn load_error(e: ffi::detect::LoadError) -> CErr {
    let code = match e {
        ffi::detect::LoadError::NoMpiInProcess | ffi::detect::LoadError::NotFound { .. } => {
            TFE_MPI_NO_LIBRARY
        }
        _ => TFE_MPI_LIBRARY,
    };
    err(code, e.to_string())
}

fn mpi_error_text(code: c_int) -> String {
    let mut buffer = [0 as c_char; 1024];
    let mut length: c_int = 0;
    // SAFETY: only called at a failure site, after `MPI_Finalized` was checked at entry.
    let rc = unsafe { ffi::MPI_Error_string(code, buffer.as_mut_ptr(), &mut length) };
    if rc != 0 {
        return String::from("(no error string)");
    }
    let length = usize::try_from(length).unwrap_or(0).min(buffer.len());
    let bytes: Vec<u8> = buffer[..length].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn mpi_ok(code: c_int, call: &str) -> Result<(), CErr> {
    if code == 0 {
        Ok(())
    } else {
        Err(err(
            TFE_MPI_ERROR,
            format!(
                "{call} failed: {} (MPI error code {code})",
                mpi_error_text(code)
            ),
        ))
    }
}

fn transport_error(e: Error) -> CErr {
    let code = match &e {
        Error::Mpi { .. } => TFE_MPI_ERROR,
        Error::InvalidTag { .. }
        | Error::SelfMessage
        | Error::NotHostResident
        | Error::NonCompact
        | Error::UnsupportedDType { .. }
        | Error::RankTooLarge { .. }
        | Error::TooManyElements { .. }
        | Error::DTypeMismatch { .. }
        | Error::Tensor(_) => TFE_INVALID_ARGUMENT,
        Error::RankMismatch { .. } | Error::ShapeMismatch { .. } => TFE_SHAPE_MISMATCH,
        _ => TFE_MPI_PROTOCOL,
    };
    err(code, e.to_string())
}

/// Copy `text` and its NUL into `buf`, report the needed size (with the NUL) in `*needed`.
unsafe fn write_text(
    text: &str,
    buf: *mut c_char,
    len: usize,
    needed: *mut usize,
) -> Result<(), CErr> {
    let required = text.len() + 1;
    if !needed.is_null() {
        *needed = required;
    }
    if buf.is_null() {
        return Ok(());
    }
    if len < required {
        return Err(err(
            TFE_BUFFER_TOO_SMALL,
            format!("buffer of {len} bytes is too small; {required} are needed"),
        ));
    }
    std::ptr::copy_nonoverlapping(text.as_ptr(), buf.cast::<u8>(), text.len());
    *buf.add(text.len()) = 0;
    Ok(())
}

/// Library, family and version string, as `key=value` lines. Callable before `MPI_Init`;
/// a failure (for example no MPI loaded yet) is returned and not cached (D12).
///
/// # Safety
///
/// `buf` is null or valid for `len` bytes; `needed` and `status` are null or writable.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_library_info(
    buf: *mut c_char,
    len: usize,
    needed: *mut usize,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    guarded(status, || {
        let lib = ffi::loader::library().map_err(load_error)?;
        let first_line = lib.version.lines().next().unwrap_or("");
        let text = format!(
            "family={}\nlibrary={}\nversion={}\n",
            lib.family.name(),
            lib.path,
            first_line
        );
        write_text(&text, buf, len, needed)
    })
}

/// A borrowed communicator. Owned by tenferro; never frees the MPI communicator.
pub struct tfe_mpi_comm {
    comm: ManuallyDrop<SimpleCommunicator>,
    rank: i32,
    size: i32,
    tag_ub: i32,
    thread_level: &'static str,
    funneled_only: bool,
    fatal_errors: bool,
    poisoned: AtomicBool,
    pending: Mutex<Option<Tensor>>,
}

impl tfe_mpi_comm {
    /// Checks every entry makes before any other MPI call (D6, D10, D11).
    fn enter(&self) -> Result<(), CErr> {
        if self.poisoned.load(Ordering::SeqCst) {
            return Err(err(
                TFE_INTERNAL_ERROR,
                "this communicator is poisoned by an earlier panic; detach it",
            ));
        }
        let mut finalized: c_int = 0;
        // SAFETY: MPI_Finalized is callable at any time.
        mpi_ok(
            unsafe { ffi::MPI_Finalized(&mut finalized) },
            "MPI_Finalized",
        )?;
        if finalized != 0 {
            return Err(err(
                TFE_MPI_FINALIZED,
                "the host finalized MPI; this communicator is no longer usable",
            ));
        }
        if self.funneled_only {
            let mut main: c_int = 0;
            // SAFETY: MPI is initialized and not finalized.
            mpi_ok(
                unsafe { ffi::MPI_Is_thread_main(&mut main) },
                "MPI_Is_thread_main",
            )?;
            if main == 0 {
                return Err(err(
                    TFE_MPI_THREAD,
                    format!(
                        "the MPI thread level is {}: calls are allowed from the main thread only",
                        self.thread_level
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn thread_level_name(lib: &ffi::loader::Loaded, provided: c_int) -> (&'static str, bool) {
    let c = &lib.consts;
    if provided == c.thread_multiple {
        ("MPI_THREAD_MULTIPLE", false)
    } else if provided == c.thread_serialized {
        ("MPI_THREAD_SERIALIZED", false)
    } else if provided == c.thread_funneled {
        ("MPI_THREAD_FUNNELED", true)
    } else {
        ("MPI_THREAD_SINGLE", true)
    }
}

/// Decode `comm_handle` into the family's neutral handle (D4).
///
/// For `int` handles (MPICH ABI) the low 32 bits are the handle and the upper 32 bits must be
/// all zero or all one (mpi4py's `comm.handle` sign-extends a negative `int`). For pointer
/// families the value is the address.
fn decode_handle(lib: &ffi::loader::Loaded, handle: u64) -> Result<usize, CErr> {
    decode_handle_for(lib.family, &lib.consts, handle)
}

fn decode_handle_for(
    family: ffi::detect::Family,
    consts: &ffi::consts::Consts,
    handle: u64,
) -> Result<usize, CErr> {
    use ffi::detect::Family;
    let raw = match family {
        Family::Mpich => {
            let upper = (handle >> 32) as u32;
            if upper != 0 && upper != u32::MAX {
                return Err(err(
                    TFE_INVALID_ARGUMENT,
                    format!(
                        "communicator handle {handle:#x} is not a 32-bit MPICH-ABI handle: the \
                         upper 32 bits must be all zero or all one"
                    ),
                ));
            }
            (handle as u32) as usize
        }
        Family::OpenMpi | Family::StandardAbi => usize::try_from(handle).map_err(|_| {
            err(
                TFE_INVALID_ARGUMENT,
                format!("handle {handle:#x} does not fit a pointer"),
            )
        })?,
    };
    if !consts.comm_handle_is_plausible(raw) {
        return Err(err(
            TFE_INVALID_ARGUMENT,
            format!("communicator handle {handle:#x} is null or MPI_COMM_NULL"),
        ));
    }
    Ok(raw)
}

unsafe fn attach(handle: u64) -> Result<Box<tfe_mpi_comm>, CErr> {
    // Detection first, and before rsmpi-rt is touched: a loader failure never panics.
    let lib = ffi::loader::library().map_err(load_error)?;

    let (mut initialized, mut finalized): (c_int, c_int) = (0, 0);
    mpi_ok(ffi::MPI_Initialized(&mut initialized), "MPI_Initialized")?;
    mpi_ok(ffi::MPI_Finalized(&mut finalized), "MPI_Finalized")?;
    if initialized == 0 {
        return Err(err(
            TFE_MPI_NOT_INITIALIZED,
            "MPI is not initialized; initialize it in the host before attaching",
        ));
    }
    if finalized != 0 {
        return Err(err(TFE_MPI_FINALIZED, "the host finalized MPI"));
    }

    // The thread level comes before any call on the communicator: under FUNNELED/SINGLE a
    // non-main thread must not make even the attach queries.
    let mut provided: c_int = 0;
    mpi_ok(ffi::MPI_Query_thread(&mut provided), "MPI_Query_thread")?;
    let (thread_level, funneled_only) = thread_level_name(lib, provided);
    if funneled_only {
        let mut main: c_int = 0;
        mpi_ok(ffi::MPI_Is_thread_main(&mut main), "MPI_Is_thread_main")?;
        if main == 0 {
            return Err(err(
                TFE_MPI_THREAD,
                format!("the MPI thread level is {thread_level}: attach from the main thread"),
            ));
        }
    }

    let raw = decode_handle(lib, handle)?;

    // First MPI call on the handle: a dead or garbage handle reaches the host's error
    // handler here (D4). There is no identity check, by design.
    let mut inter: c_int = 0;
    mpi_ok(
        ffi::MPI_Comm_test_inter(raw, &mut inter),
        "MPI_Comm_test_inter",
    )?;
    if inter != 0 {
        return Err(err(
            TFE_INVALID_ARGUMENT,
            "an inter-communicator cannot be attached",
        ));
    }
    let (mut rank, mut size): (c_int, c_int) = (0, 0);
    mpi_ok(ffi::MPI_Comm_rank(raw, &mut rank), "MPI_Comm_rank")?;
    mpi_ok(ffi::MPI_Comm_size(raw, &mut size), "MPI_Comm_size")?;

    let mut handler = lib.consts.errors_return;
    mpi_ok(
        ffi::MPI_Comm_get_errhandler(raw, &mut handler),
        "MPI_Comm_get_errhandler",
    )?;
    let fatal_errors = handler == lib.consts.errors_are_fatal;
    // The standard requires freeing the reference `MPI_Comm_get_errhandler` returned.
    mpi_ok(
        ffi::MPI_Errhandler_free(&mut handler),
        "MPI_Errhandler_free",
    )?;

    let mut tag_ub_ptr: *mut c_void = std::ptr::null_mut();
    let mut flag: c_int = 0;
    mpi_ok(
        ffi::MPI_Comm_get_attr(
            raw,
            lib.consts.tag_ub,
            (&mut tag_ub_ptr as *mut *mut c_void).cast(),
            &mut flag,
        ),
        "MPI_Comm_get_attr(MPI_TAG_UB)",
    )?;
    // The standard guarantees at least 32767.
    let tag_ub = if flag != 0 && !tag_ub_ptr.is_null() {
        *(tag_ub_ptr as *const c_int)
    } else {
        32767
    };

    // Held in ManuallyDrop: rsmpi-rt's user communicator would `MPI_Comm_free` on drop.
    let comm = SimpleCommunicator::from_raw(raw);
    Ok(Box::new(tfe_mpi_comm {
        comm: ManuallyDrop::new(comm),
        rank,
        size,
        tag_ub,
        thread_level,
        funneled_only,
        fatal_errors,
        poisoned: AtomicBool::new(false),
        pending: Mutex::new(None),
    }))
}

/// Borrow the host's communicator. Process-local: no message is sent.
///
/// `comm_handle` is the host library's `MPI_Comm`, widened to 64 bits the way C converts it
/// to an unsigned integer (for `int` handles the upper half may be zero- or sign-extension).
///
/// # Safety
///
/// `out` and `status` are null or writable. `comm_handle` must be a live communicator of the
/// MPI library the host loaded, or the host's error handler decides what happens.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_comm_attach(
    comm_handle: u64,
    out: *mut *mut tfe_mpi_comm,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    guarded(status, || {
        if out.is_null() {
            return Err(err(TFE_INVALID_ARGUMENT, "`out` is null"));
        }
        *out = std::ptr::null_mut();
        let comm = attach(comm_handle)?;
        *out = Box::into_raw(comm);
        Ok(())
    })
}

/// Release a communicator object. Frees only Rust memory: no MPI call, so it is safe after the
/// host finalized MPI (a Julia finalizer may run after MPI.jl's atexit hook).
///
/// # Safety
///
/// `comm` is null or a pointer returned by [`tfe_mpi_comm_attach`], detached at most once.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_comm_detach(
    comm: *mut tfe_mpi_comm,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    guarded(status, || {
        if !comm.is_null() {
            drop(Box::from_raw(comm));
        }
        Ok(())
    })
}

/// Run `f` on an attached communicator. A panic poisons the object (D10).
unsafe fn with_comm(
    comm: *const tfe_mpi_comm,
    status: *mut tfe_status_t,
    f: impl FnOnce(&tfe_mpi_comm) -> Result<(), CErr>,
) -> tfe_status_t {
    if comm.is_null() {
        return finish(status, Err(err(TFE_INVALID_ARGUMENT, "`comm` is null")));
    }
    let c = &*comm;
    match catch_unwind(AssertUnwindSafe(|| {
        c.enter()?;
        f(c)
    })) {
        Ok(result) => finish(status, result),
        Err(payload) => {
            c.poisoned.store(true, Ordering::SeqCst);
            finish(
                status,
                Err(err(TFE_INTERNAL_ERROR, panic_message(&*payload))),
            )
        }
    }
}

/// Library information plus, for the attached communicator, its rank and size, the provided
/// thread level and whether its error handler is `MPI_ERRORS_ARE_FATAL`.
///
/// # Safety
///
/// `comm` is attached; `buf` is null or valid for `len` bytes; `needed`, `status` null or writable.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_comm_info(
    comm: *const tfe_mpi_comm,
    buf: *mut c_char,
    len: usize,
    needed: *mut usize,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    with_comm(comm, status, |c| {
        let lib = ffi::loader::library().map_err(load_error)?;
        let first_line = lib.version.lines().next().unwrap_or("");
        let text = format!(
            "family={}\nlibrary={}\nversion={}\nrank={}\nsize={}\nthread_level={}\n\
             errhandler_fatal={}\ntag_ub={}\n",
            lib.family.name(),
            lib.path,
            first_line,
            c.rank,
            c.size,
            c.thread_level,
            u8::from(c.fatal_errors),
            c.tag_ub
        );
        write_text(&text, buf, len, needed)
    })
}

/// Validate a peer rank and a tag locally, so the host's error handler never sees them (D6).
fn check_peer(c: &tfe_mpi_comm, peer: i32, tag: i32) -> Result<(), CErr> {
    if peer < 0 || peer >= c.size {
        return Err(err(
            TFE_INVALID_ARGUMENT,
            format!("rank {peer} is outside 0..{}", c.size),
        ));
    }
    if tag < 0 || tag > c.tag_ub {
        return Err(err(
            TFE_INVALID_ARGUMENT,
            format!("tag {tag} is outside 0..={} (MPI_TAG_UB)", c.tag_ub),
        ));
    }
    Ok(())
}

fn element_bytes(code: u8) -> Option<usize> {
    match code {
        1 => Some(1),
        2 | 4 => Some(4),
        3 | 5 | 6 => Some(8),
        7 => Some(16),
        _ => None,
    }
}

fn from_bytes<T: Copy>(bytes: &[u8]) -> Vec<T> {
    let n = bytes.len() / std::mem::size_of::<T>();
    let mut out: Vec<T> = Vec::with_capacity(n);
    // SAFETY: `T` is a plain scalar (checked at the call sites: no padding, any bit pattern
    // valid except `bool`, which the caller validates first); the destination has room for `n`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr().cast::<u8>(),
            n * std::mem::size_of::<T>(),
        );
        out.set_len(n);
    }
    out
}

fn to_bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: reading the bytes of a slice of plain scalars.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

macro_rules! send_typed {
    ($c:expr, $dest:expr, $tag:expr, $ty:ty, $shape:expr, $bytes:expr) => {{
        let values: Vec<$ty> = from_bytes($bytes);
        let tensor = TypedTensor::<$ty, DynRank, Host>::from_host_vec_col_major($shape, values)
            .map_err(|e| err(TFE_INVALID_ARGUMENT, e.to_string()))?;
        tenferro_mpi::send($c.comm.process_at_rank($dest), $tag, &tensor).map_err(transport_error)
    }};
}

/// Send a compact column-major host buffer. `dtype` is the wire code (`1` bool, `2` i32,
/// `3` i64, `4` f32, `5` f64, `6` complex64, `7` complex128); `shape`/`rank` give the extents;
/// `nbytes` must equal the element count times the element size. Blocks until the receiver
/// accepted the header and the payload was handed to MPI.
///
/// # Safety
///
/// `shape` is valid for `rank` elements and `data` for `nbytes` bytes; `comm` is attached.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_send(
    comm: *const tfe_mpi_comm,
    dest: i32,
    tag: i32,
    dtype: u8,
    shape: *const usize,
    rank: usize,
    data: *const c_void,
    nbytes: usize,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    with_comm(comm, status, |c| {
        check_peer(c, dest, tag)?;
        if dest == c.rank {
            return Err(err(
                TFE_INVALID_ARGUMENT,
                "a send to the calling rank would block forever",
            ));
        }
        let Some(width) = element_bytes(dtype) else {
            return Err(err(
                TFE_INVALID_ARGUMENT,
                format!("unknown dtype code {dtype}"),
            ));
        };
        if rank > MAX_WIRE_RANK {
            return Err(err(
                TFE_INVALID_ARGUMENT,
                format!("rank {rank} exceeds {MAX_WIRE_RANK}"),
            ));
        }
        if (shape.is_null() && rank > 0) || (data.is_null() && nbytes > 0) {
            return Err(err(TFE_INVALID_ARGUMENT, "`shape` or `data` is null"));
        }
        let dims: Vec<usize> = if rank == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(shape, rank).to_vec()
        };
        let elements = dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| err(TFE_SHAPE_MISMATCH, "the element count overflows"))?;
        if elements.checked_mul(width) != Some(nbytes) {
            return Err(err(
                TFE_SHAPE_MISMATCH,
                format!(
                    "shape {dims:?} of {width}-byte elements needs {} bytes, got {nbytes}",
                    elements.saturating_mul(width)
                ),
            ));
        }
        let bytes: &[u8] = if nbytes == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(data.cast::<u8>(), nbytes)
        };
        match dtype {
            1 => {
                if let Some(bad) = bytes.iter().find(|&&b| b > 1) {
                    return Err(err(
                        TFE_INVALID_ARGUMENT,
                        format!("a bool element holds {bad}, not 0 or 1"),
                    ));
                }
                // `bool` has the size and valid bit patterns 0 and 1 of the validated bytes.
                let values: Vec<bool> = bytes.iter().map(|&b| b == 1).collect();
                let tensor =
                    TypedTensor::<bool, DynRank, Host>::from_host_vec_col_major(dims, values)
                        .map_err(|e| err(TFE_INVALID_ARGUMENT, e.to_string()))?;
                tenferro_mpi::send(c.comm.process_at_rank(dest), tag, &tensor)
                    .map_err(transport_error)
            }
            2 => send_typed!(c, dest, tag, i32, dims, bytes),
            3 => send_typed!(c, dest, tag, i64, dims, bytes),
            4 => send_typed!(c, dest, tag, f32, dims, bytes),
            5 => send_typed!(c, dest, tag, f64, dims, bytes),
            6 => send_typed!(c, dest, tag, Complex32, dims, bytes),
            7 => send_typed!(c, dest, tag, Complex64, dims, bytes),
            _ => unreachable!("element_bytes accepted the code"),
        }
    })
}

fn dtype_code(dtype: DType) -> Option<u8> {
    match dtype {
        DType::Bool => Some(1),
        DType::I32 => Some(2),
        DType::I64 => Some(3),
        DType::F32 => Some(4),
        DType::F64 => Some(5),
        DType::C32 => Some(6),
        DType::C64 => Some(7),
        _ => None,
    }
}

/// Receive one tensor from `source` on `tag` into a buffer **held by the communicator object**,
/// and report its dtype code, rank, shape and byte count. Nothing is lost if `shape_cap` is
/// too small: the object keeps the tensor, `*rank` and `*nbytes` are set, and the call returns
/// `TFE_BUFFER_TOO_SMALL`; read the full shape with [`tfe_mpi_recv_shape`] and copy the data out with [`tfe_mpi_recv_take`]. At most one received tensor
/// is held at a time.
///
/// # Safety
///
/// `dtype`, `rank`, `nbytes` are writable; `shape_out` is valid for `shape_cap` elements.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_recv(
    comm: *const tfe_mpi_comm,
    source: i32,
    tag: i32,
    dtype: *mut u8,
    shape_out: *mut usize,
    shape_cap: usize,
    rank: *mut usize,
    nbytes: *mut usize,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    with_comm(comm, status, |c| {
        check_peer(c, source, tag)?;
        if source == c.rank {
            return Err(err(
                TFE_INVALID_ARGUMENT,
                "a receive from the calling rank would block forever",
            ));
        }
        if dtype.is_null()
            || rank.is_null()
            || nbytes.is_null()
            || (shape_out.is_null() && shape_cap > 0)
        {
            return Err(err(TFE_INVALID_ARGUMENT, "an output pointer is null"));
        }
        let mut pending = c.pending.lock().unwrap_or_else(|e| e.into_inner());
        if pending.is_some() {
            return Err(err(
                TFE_INVALID_ARGUMENT,
                "a received tensor is still held; call tfe_mpi_recv_take first",
            ));
        }
        let tensor: Tensor =
            tenferro_mpi::recv(c.comm.process_at_rank(source), tag).map_err(transport_error)?;
        let code = dtype_code(tensor.dtype()).ok_or_else(|| {
            err(
                TFE_INTERNAL_ERROR,
                "the transport delivered an unknown dtype",
            )
        })?;
        let dims = tensor.shape().to_vec();
        let width = element_bytes(code).unwrap_or(0);
        *dtype = code;
        *rank = dims.len();
        *nbytes = dims.iter().product::<usize>() * width;
        let copy = dims.len().min(shape_cap);
        if copy > 0 {
            std::ptr::copy_nonoverlapping(dims.as_ptr(), shape_out, copy);
        }
        let too_small = dims.len() > shape_cap;
        *pending = Some(tensor);
        if too_small {
            return Err(err(
                TFE_BUFFER_TOO_SMALL,
                format!(
                    "shape buffer of {shape_cap} is too small for rank {}",
                    dims.len()
                ),
            ));
        }
        Ok(())
    })
}

/// The shape of the tensor held by the last [`tfe_mpi_recv`], for a caller whose shape buffer was
/// too small there. Receives nothing. `*rank` is always set; `TFE_BUFFER_TOO_SMALL` if
/// `shape_cap` is still too small.
///
/// # Safety
///
/// `rank` is writable and `shape_out` is valid for `shape_cap` elements.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_recv_shape(
    comm: *const tfe_mpi_comm,
    shape_out: *mut usize,
    shape_cap: usize,
    rank: *mut usize,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    with_comm(comm, status, |c| {
        if rank.is_null() || (shape_out.is_null() && shape_cap > 0) {
            return Err(err(TFE_INVALID_ARGUMENT, "an output pointer is null"));
        }
        let pending = c.pending.lock().unwrap_or_else(|e| e.into_inner());
        let Some(tensor) = pending.as_ref() else {
            return Err(err(TFE_INVALID_ARGUMENT, "no received tensor is held"));
        };
        let dims = tensor.shape();
        *rank = dims.len();
        if dims.len() > shape_cap {
            return Err(err(
                TFE_BUFFER_TOO_SMALL,
                format!(
                    "shape buffer of {shape_cap} is too small for rank {}",
                    dims.len()
                ),
            ));
        }
        if !dims.is_empty() {
            std::ptr::copy_nonoverlapping(dims.as_ptr(), shape_out, dims.len());
        }
        Ok(())
    })
}

macro_rules! take_typed {
    ($tensor:expr, $ty:ty, $out:expr) => {{
        let values = $tensor
            .as_slice::<$ty>()
            .map_err(|e| err(TFE_INTERNAL_ERROR, e.to_string()))?;
        $out.extend_from_slice(to_bytes(values));
    }};
}

/// Copy the tensor held by the last [`tfe_mpi_recv`] into `data` (column-major, the dtype and
/// shape that call reported) and release it. `TFE_BUFFER_TOO_SMALL` leaves it held.
///
/// # Safety
///
/// `data` is valid for `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_recv_take(
    comm: *const tfe_mpi_comm,
    data: *mut c_void,
    cap: usize,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    with_comm(comm, status, |c| {
        let mut pending = c.pending.lock().unwrap_or_else(|e| e.into_inner());
        let Some(tensor) = pending.as_ref() else {
            return Err(err(TFE_INVALID_ARGUMENT, "no received tensor is held"));
        };
        let mut bytes: Vec<u8> = Vec::new();
        match tensor.dtype() {
            DType::Bool => {
                let values = tensor
                    .as_slice::<bool>()
                    .map_err(|e| err(TFE_INTERNAL_ERROR, e.to_string()))?;
                bytes.extend(values.iter().map(|&b| u8::from(b)));
            }
            DType::I32 => take_typed!(tensor, i32, &mut bytes),
            DType::I64 => take_typed!(tensor, i64, &mut bytes),
            DType::F32 => take_typed!(tensor, f32, &mut bytes),
            DType::F64 => take_typed!(tensor, f64, &mut bytes),
            DType::C32 => take_typed!(tensor, Complex32, &mut bytes),
            DType::C64 => take_typed!(tensor, Complex64, &mut bytes),
            other => {
                return Err(err(
                    TFE_INTERNAL_ERROR,
                    format!("unsupported held dtype {other:?}"),
                ))
            }
        }
        if bytes.len() > cap {
            return Err(err(
                TFE_BUFFER_TOO_SMALL,
                format!(
                    "buffer of {cap} bytes is too small; {} are needed",
                    bytes.len()
                ),
            ));
        }
        if data.is_null() && !bytes.is_empty() {
            return Err(err(TFE_INVALID_ARGUMENT, "`data` is null"));
        }
        if !bytes.is_empty() {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data.cast::<u8>(), bytes.len());
        }
        *pending = None;
        Ok(())
    })
}

/// The last failure's message on this thread (UTF-8, NUL-terminated). With `buf == NULL` only
/// the required length is written to `*out_len`.
///
/// # Safety
///
/// `buf` is null or valid for `buf_len` bytes; `out_len` is writable.
#[no_mangle]
pub unsafe extern "C" fn tfe_last_error_message(
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> tfe_status_t {
    if out_len.is_null() {
        return TFE_INVALID_ARGUMENT;
    }
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        let required = message.len() + 1;
        *out_len = required;
        if buf.is_null() {
            return TFE_SUCCESS;
        }
        if buf_len < required {
            return TFE_BUFFER_TOO_SMALL;
        }
        std::ptr::copy_nonoverlapping(message.as_ptr(), buf, message.len());
        *buf.add(message.len()) = 0;
        TFE_SUCCESS
    })
}

/// Test hook (feature `test-hooks`): enter a panic stub through the normal entry path.
///
/// # Safety
///
/// `comm` is attached.
#[cfg(feature = "test-hooks")]
#[no_mangle]
pub unsafe extern "C" fn tfe_mpi_test_hit_stub(
    comm: *const tfe_mpi_comm,
    status: *mut tfe_status_t,
) -> tfe_status_t {
    with_comm(comm, status, |_| {
        // MPI_Buffer_detach is outside the carried surface, so this is a panic stub.
        ffi::MPI_Buffer_detach(std::ptr::null_mut(), std::ptr::null_mut());
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffi::consts::Consts;
    use ffi::detect::Family;

    fn consts(family: Family) -> Consts {
        // Open MPI constants are addresses; any distinct non-zero values do for an encoding test.
        let mut next = 0x1000usize;
        Consts::for_family(family, |_| {
            next += 0x10;
            next
        })
    }

    #[test]
    fn mpich_user_communicator_zero_and_sign_extended_are_the_same_handle() {
        let c = consts(Family::Mpich);
        // 0x84000002 is a user communicator; mpi4py's `comm.handle` sign-extends it.
        let zero_extended = 0x8400_0002u64;
        let sign_extended = 0xffff_ffff_8400_0002u64;
        assert_eq!(
            decode_handle_for(Family::Mpich, &c, zero_extended).ok(),
            Some(0x8400_0002)
        );
        assert_eq!(
            decode_handle_for(Family::Mpich, &c, sign_extended).ok(),
            Some(0x8400_0002)
        );
    }

    #[test]
    fn mpich_mixed_upper_bits_are_refused() {
        let c = consts(Family::Mpich);
        for handle in [
            0x1_8400_0002u64,
            0x7fff_ffff_8400_0002,
            0x0000_0001_0000_0000,
        ] {
            let e = decode_handle_for(Family::Mpich, &c, handle).expect_err("refused");
            assert_eq!(e.code, TFE_INVALID_ARGUMENT, "{handle:#x}");
        }
    }

    #[test]
    fn zero_and_each_familys_comm_null_are_refused_before_any_mpi_call() {
        for family in [Family::Mpich, Family::OpenMpi, Family::StandardAbi] {
            let c = consts(family);
            assert_eq!(
                decode_handle_for(family, &c, 0).expect_err("zero").code,
                TFE_INVALID_ARGUMENT
            );
            let null = c.comm_null as u64;
            let e = decode_handle_for(family, &c, null).expect_err("MPI_COMM_NULL");
            assert_eq!(e.code, TFE_INVALID_ARGUMENT, "{family:?}");
            assert!(
                decode_handle_for(family, &c, c.comm_world as u64).is_ok(),
                "{family:?}"
            );
        }
    }

    #[test]
    fn a_pointer_family_takes_the_value_as_the_address() {
        let c = consts(Family::OpenMpi);
        assert_eq!(
            decode_handle_for(Family::OpenMpi, &c, 0xdead_beef).ok(),
            Some(0xdead_beef)
        );
        let a = consts(Family::StandardAbi);
        assert_eq!(
            decode_handle_for(Family::StandardAbi, &a, 0x101).ok(),
            Some(0x101)
        );
    }

    #[test]
    fn entries_reject_null_pointers_without_panicking() {
        let mut status = 0;
        // SAFETY: null arguments are exactly what is under test.
        unsafe {
            assert_eq!(
                tfe_mpi_comm_attach(1, std::ptr::null_mut(), &mut status),
                TFE_INVALID_ARGUMENT
            );
            assert_eq!(
                tfe_mpi_comm_info(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null_mut(),
                    &mut status
                ),
                TFE_INVALID_ARGUMENT
            );
            assert_eq!(
                tfe_mpi_comm_detach(std::ptr::null_mut(), &mut status),
                TFE_SUCCESS
            );
            assert_eq!(
                tfe_last_error_message(std::ptr::null_mut(), 0, std::ptr::null_mut()),
                TFE_INVALID_ARGUMENT
            );
        }
    }

    #[test]
    fn a_panic_inside_an_entry_is_caught_and_reported() {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let mut status = 0;
        let code = guarded(&mut status, || panic!("boom"));
        std::panic::set_hook(hook);
        assert_eq!(code, TFE_INTERNAL_ERROR);
        assert_eq!(status, TFE_INTERNAL_ERROR);
        assert!(last_message().contains("boom"));
    }

    fn last_message() -> String {
        let mut needed = 0usize;
        // SAFETY: a size query, then a buffer of that size.
        unsafe {
            tfe_last_error_message(std::ptr::null_mut(), 0, &mut needed);
            let mut buf = vec![0u8; needed];
            tfe_last_error_message(buf.as_mut_ptr(), needed, &mut needed);
            String::from_utf8_lossy(&buf[..needed - 1]).into_owned()
        }
    }
}
