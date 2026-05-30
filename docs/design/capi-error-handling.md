# C-API Error Handling Policy

Error handling policy for `tenferro-capi` FFI functions.
Aligns with patterns from `tensor4all-rs` (`crates/tensor4all-capi/src/lib.rs`)
while adapting to tenferro's error hierarchy.

Tracking issue: #110

---

## Status Codes

```rust
pub type tfe_status_t = i32;

pub const TFE_SUCCESS:          tfe_status_t =  0;
pub const TFE_INVALID_ARGUMENT: tfe_status_t = -1;
pub const TFE_SHAPE_MISMATCH:   tfe_status_t = -2;
pub const TFE_INTERNAL_ERROR:   tfe_status_t = -3;
pub const TFE_BUFFER_TOO_SMALL: tfe_status_t = -4;  // NEW (for last-error API)
```

Null pointer errors use `TFE_INVALID_ARGUMENT` (not a separate code)
to keep the status code set small.

---

## Error Mapping

### Crate-local tensor and operation errors -> `tfe_status_t`

The workspace no longer has a shared device error crate. C-API bindings should
map the concrete error type at the crate boundary where the FFI call invokes
Rust code.

| Rust Error | Status Code | Rationale |
|---|---|---|
| `tenferro_tensor::Error::InvalidConfig { .. }` | `TFE_INVALID_ARGUMENT` | Bad caller input |
| `tenferro_tensor::Error::ShapeMismatch { .. }` | `TFE_SHAPE_MISMATCH` | Operand dimensions incompatible |
| `tenferro_tensor::Error::RankMismatch { .. }` | `TFE_SHAPE_MISMATCH` | Rank is a shape constraint |
| `tenferro_tensor::Error::DTypeMismatch { .. }` | `TFE_INVALID_ARGUMENT` | Caller mixed incompatible dtypes |
| `tenferro_tensor::Error::AxisOutOfBounds { .. }` | `TFE_INVALID_ARGUMENT` | Bad axis argument |
| `tenferro_tensor::Error::DuplicateAxis { .. }` | `TFE_INVALID_ARGUMENT` | Bad axis argument |
| `tenferro_tensor::Error::AxisRoleConflict { .. }` | `TFE_INVALID_ARGUMENT` | Bad axis argument |
| `tenferro_tensor::Error::MissingValue { .. }` | `TFE_INTERNAL_ERROR` | Runtime graph/value failure |
| `tenferro_tensor::Error::BackendFailure { .. }` | `TFE_INTERNAL_ERROR` | Backend failure |
| `tenferro_einsum::Error::InvalidArgument(_)` | `TFE_INVALID_ARGUMENT` | Bad einsum notation or path |
| `tenferro_einsum::Error::ShapeMismatch { .. }` | `TFE_SHAPE_MISMATCH` | Operand dimensions incompatible |

### `chainrules_core::AutodiffError` → `tfe_status_t`

| Rust Variant | Status Code | Rationale |
|---|---|---|
| `InvalidArgument(_)` | `TFE_INVALID_ARGUMENT` | Bad AD argument |
| `ModeNotSupported { .. }` | `TFE_INVALID_ARGUMENT` | Unsupported AD mode (e.g. tropical frule) |
| `TangentShapeMismatch { .. }` | `TFE_SHAPE_MISMATCH` | Tangent/primal shape mismatch |
| `NonScalarLoss { .. }` | `TFE_INVALID_ARGUMENT` | Non-scalar loss for pullback |
| `HvpNotSupported` | `TFE_INVALID_ARGUMENT` | HVP not available |
| `MissingNode` | `TFE_INTERNAL_ERROR` | AD graph error |

### Null Pointers

| Condition | Behavior |
|---|---|
| Required pointer is null | Return `TFE_INVALID_ARGUMENT` |
| Array pointer null with `len > 0` | Return `TFE_INVALID_ARGUMENT` |
| Array pointer null with `len == 0` | OK (use `&[]`) |
| Optional pointer null (cotangent/tangent) | OK (treated as `None`) |
| Status pointer null | Undefined (no way to report; early return) |

### Panics

Caught by `catch_unwind` → `TFE_INTERNAL_ERROR`.
Panic message stored in thread-local last-error buffer.

---

## Shared Helpers

### Error Mapping Functions

```rust
fn map_tensor_error(err: &tenferro_tensor::Error) -> tfe_status_t {
    use tenferro_tensor::Error;
    match err {
        Error::InvalidConfig { .. }
        | Error::DTypeMismatch { .. }
        | Error::AxisOutOfBounds { .. }
        | Error::DuplicateAxis { .. }
        | Error::AxisRoleConflict { .. } => TFE_INVALID_ARGUMENT,
        Error::ShapeMismatch { .. } | Error::RankMismatch { .. } => TFE_SHAPE_MISMATCH,
        Error::MissingValue { .. } | Error::BackendFailure { .. } => TFE_INTERNAL_ERROR,
    }
}

fn map_einsum_error(err: &tenferro_einsum::Error) -> tfe_status_t {
    use tenferro_einsum::Error;
    match err {
        Error::InvalidArgument(_) => TFE_INVALID_ARGUMENT,
        Error::ShapeMismatch { .. } => TFE_SHAPE_MISMATCH,
    }
}

fn map_ad_error(err: &chainrules_core::AutodiffError) -> tfe_status_t {
    use chainrules_core::AutodiffError;
    match err {
        AutodiffError::InvalidArgument(_)
        | AutodiffError::ModeNotSupported { .. }
        | AutodiffError::NonScalarLoss { .. }
        | AutodiffError::HvpNotSupported => TFE_INVALID_ARGUMENT,
        AutodiffError::TangentShapeMismatch { .. } => TFE_SHAPE_MISMATCH,
        AutodiffError::MissingNode => TFE_INTERNAL_ERROR,
    }
}
```

### Catch/Dispatch Helpers

Modeled after `tensor4all-rs` (`err_status`, `unwrap_catch`, `unwrap_catch_ptr`):

```rust
/// Store error message in thread-local buffer and return status code.
fn err_status<E: std::fmt::Display>(err: E, code: tfe_status_t) -> tfe_status_t {
    set_last_error(&err.to_string());
    code
}

/// Unwrap catch_unwind for status-returning functions.
fn unwrap_catch(result: std::thread::Result<tfe_status_t>) -> tfe_status_t {
    match result {
        Ok(code) => code,
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            TFE_INTERNAL_ERROR
        }
    }
}

/// Unwrap catch_unwind for pointer-returning functions.
fn unwrap_catch_ptr<T>(result: std::thread::Result<*mut T>) -> *mut T {
    match result {
        Ok(ptr) => ptr,
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            std::ptr::null_mut()
        }
    }
}
```

### Boilerplate Reduction

Replace the repeated `match result { Ok(Ok(..)) => ..., Ok(Err(..)) => ..., Err(_) => ... }`
pattern with a `finalize_ptr` helper:

```rust
/// Finalize a catch_unwind result for functions returning a pointer via status.
unsafe fn finalize_ptr(
    result: std::thread::Result<Result<*mut TfeTensorF64, tfe_status_t>>,
    status: *mut tfe_status_t,
) -> *mut TfeTensorF64 {
    match result {
        Ok(Ok(ptr)) => { *status = TFE_SUCCESS; ptr }
        Ok(Err(code)) => { *status = code; std::ptr::null_mut() }
        Err(panic) => {
            set_last_error(&panic_message(&*panic));
            *status = TFE_INTERNAL_ERROR;
            std::ptr::null_mut()
        }
    }
}
```

---

## Thread-Local Last-Error API

### Storage

```rust
use std::cell::RefCell;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_last_error(msg: &str) {
    let bt = std::backtrace::Backtrace::capture();
    let full = match bt.status() {
        std::backtrace::BacktraceStatus::Captured =>
            format!("{msg}\n\nRust backtrace:\n{bt}"),
        _ => msg.to_string(),
    };
    LAST_ERROR.with(|cell| *cell.borrow_mut() = full);
}

fn panic_message(payload: &dyn std::any::Any) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}
```

### Retrieval Function

```rust
/// Retrieve the last error message (UTF-8, null-terminated).
///
/// - `buf == NULL`: query required length only (written to `*out_len`).
/// - `buf != NULL`: copy message into buffer.
///
/// Returns `TFE_SUCCESS`, `TFE_INVALID_ARGUMENT` (out_len is null),
/// or `TFE_BUFFER_TOO_SMALL`.
#[no_mangle]
pub unsafe extern "C" fn tfe_last_error_message(
    buf: *mut u8,
    buf_len: usize,
    out_len: *mut usize,
) -> tfe_status_t;
```

### Host Language Usage (Julia)

```julia
function check_tfe_status(status::Cint)
    status == TFE_SUCCESS && return
    len = Ref{Csize_t}(0)
    ccall(:tfe_last_error_message, Cint,
          (Ptr{UInt8}, Csize_t, Ptr{Csize_t}), C_NULL, 0, len)
    buf = Vector{UInt8}(undef, len[])
    ccall(:tfe_last_error_message, Cint,
          (Ptr{UInt8}, Csize_t, Ptr{Csize_t}), buf, length(buf), len)
    msg = unsafe_string(pointer(buf))
    error("tenferro error ($status): $msg")
end
```

---

## Query Function Safety

Current query functions (`tfe_tensor_f64_ndim`, `_shape`, `_len`, `_data`)
lack `catch_unwind` and null guards.

**Policy**: Add null guards to these functions. The performance cost of a
null check is negligible for query operations.

---

## Implementation Sequence

1. **#112** — SVD AD null+len guards (standalone, no API changes)
2. **Status mapping refactor** — Add mapping functions + shared helpers,
   refactor all FFI functions, add null guards to query functions
3. **Last-error message API** — Thread-local storage, `tfe_last_error_message`,
   `TFE_BUFFER_TOO_SMALL` code

Items 2 and 3 can be combined into a single PR.

## Related Issues

- **#110** — Tracking umbrella
- **#111** — SVD AD layout restoration
- **#112** — SVD AD null+len guards
- **#113** — Numerical-oracle AD tests
