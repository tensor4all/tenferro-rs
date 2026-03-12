use std::cell::RefCell;

/// Status code type returned by all C-API functions.
pub type tfe_status_t = i32;

/// Operation completed successfully.
pub const TFE_SUCCESS: tfe_status_t = 0;

/// Invalid argument (null pointer, bad subscript string, etc.).
pub const TFE_INVALID_ARGUMENT: tfe_status_t = -1;

/// Tensor shape mismatch for the requested operation.
pub const TFE_SHAPE_MISMATCH: tfe_status_t = -2;

/// Internal error (Rust panic or unexpected failure).
pub const TFE_INTERNAL_ERROR: tfe_status_t = -3;

/// Output buffer is too small for the requested data.
pub const TFE_BUFFER_TOO_SMALL: tfe_status_t = -4;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

pub(crate) type StatusResult<T> = Result<T, tfe_status_t>;

/// Store an error message in thread-local storage.
pub(crate) fn set_last_error(msg: &str) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = msg.to_string();
    });
}

/// Extract a human-readable message from a panic payload.
pub(crate) fn panic_message(payload: &dyn std::any::Any) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Retrieve the last error message (UTF-8, null-terminated).
///
/// - `buf == NULL`: query required length only (written to `*out_len`).
/// - `buf != NULL`: copy message into buffer.
///
/// `out_len` receives the required buffer size including the null terminator.
///
/// # Returns
///
/// - `TFE_SUCCESS` on success (or query-only mode).
/// - `TFE_INVALID_ARGUMENT` if `out_len` is null.
/// - `TFE_BUFFER_TOO_SMALL` if `buf_len` is too small (required size in `*out_len`).
///
/// # Safety
///
/// - `out_len` must be a valid, non-null pointer.
/// - If `buf` is non-null, it must point to a buffer of at least `buf_len` bytes.
///
/// # Examples (C)
///
/// ```c
/// size_t len;
/// tfe_last_error_message(NULL, 0, &len);
/// if (len > 0) {
///     char *buf = malloc(len);
///     tfe_last_error_message((uint8_t *)buf, len, &len);
///     printf("Error: %s\n", buf);
///     free(buf);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn tfe_last_error_message(
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> tfe_status_t {
    if out_len.is_null() {
        return TFE_INVALID_ARGUMENT;
    }

    LAST_ERROR.with(|cell| {
        let msg = cell.borrow();
        let required = msg.len() + 1;
        *out_len = required;

        if buf.is_null() {
            return TFE_SUCCESS;
        }
        if buf_len < required {
            return TFE_BUFFER_TOO_SMALL;
        }

        std::ptr::copy_nonoverlapping(msg.as_ptr(), buf, msg.len());
        *buf.add(msg.len()) = 0;
        TFE_SUCCESS
    })
}

/// Map `tenferro_device::Error` to the appropriate status code.
pub(crate) fn map_device_error(err: &tenferro_device::Error) -> tfe_status_t {
    set_last_error(&err.to_string());
    use tenferro_device::Error;
    match err {
        Error::InvalidArgument(_)
        | Error::StrideError(_)
        | Error::CrossMemorySpaceOperation { .. } => TFE_INVALID_ARGUMENT,
        Error::ShapeMismatch { .. } | Error::RankMismatch { .. } => TFE_SHAPE_MISMATCH,
        Error::DeviceError(_) | Error::NoCompatibleComputeDevice { .. } => TFE_INTERNAL_ERROR,
    }
}

/// Map `chainrules_core::AutodiffError` to the appropriate status code.
pub(crate) fn map_ad_error(err: &chainrules_core::AutodiffError) -> tfe_status_t {
    set_last_error(&err.to_string());
    use chainrules_core::AutodiffError;
    match err {
        AutodiffError::InvalidArgument(_)
        | AutodiffError::ModeNotSupported { .. }
        | AutodiffError::NonScalarLoss { .. }
        | AutodiffError::HvpNotSupported
        | AutodiffError::GraphFreed => TFE_INVALID_ARGUMENT,
        AutodiffError::TangentShapeMismatch { .. } => TFE_SHAPE_MISMATCH,
        AutodiffError::MissingNode => TFE_INTERNAL_ERROR,
    }
}

pub(crate) unsafe fn finalize_ptr<T>(
    result: std::thread::Result<StatusResult<*mut T>>,
    status: *mut tfe_status_t,
) -> *mut T {
    match result {
        Ok(Ok(ptr)) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
            ptr
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                *status = code;
            }
            std::ptr::null_mut()
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
            std::ptr::null_mut()
        }
    }
}

pub(crate) unsafe fn finalize_void(
    result: std::thread::Result<StatusResult<()>>,
    status: *mut tfe_status_t,
) {
    match result {
        Ok(Ok(())) => {
            if !status.is_null() {
                *status = TFE_SUCCESS;
            }
        }
        Ok(Err(code)) => {
            if !status.is_null() {
                *status = code;
            }
        }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            if !status.is_null() {
                *status = TFE_INTERNAL_ERROR;
            }
        }
    }
}
