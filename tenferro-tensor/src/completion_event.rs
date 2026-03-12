/// Synchronization event for asynchronous accelerator operations.
///
/// Tracks completion of asynchronous operations on accelerator devices,
/// enabling operation chaining without CPU synchronization.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::CompletionEvent;
///
/// // CompletionEvent is typically created by GPU backends.
/// let _event: Option<CompletionEvent> = None;
/// ```
#[derive(Clone)]
pub struct CompletionEvent {
    #[allow(dead_code)]
    inner: CompletionEventInner,
}

#[derive(Clone)]
#[allow(dead_code)]
enum CompletionEventInner {
    Noop,
    Cuda { _event: *mut std::ffi::c_void },
    Rocm { _event: *mut std::ffi::c_void },
}

unsafe impl Send for CompletionEvent {}
unsafe impl Sync for CompletionEvent {}
