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

impl CompletionEvent {
    /// Create a no-op completion event.
    ///
    /// This is useful for testing and for cases where no synchronization is needed.
    pub fn noop() -> Self {
        Self {
            inner: CompletionEventInner::Noop,
        }
    }
}

#[derive(Clone)]
#[allow(dead_code)]
enum CompletionEventInner {
    Noop,
    Cuda { _event: *mut std::ffi::c_void },
    Rocm { _event: *mut std::ffi::c_void },
}

/// # Safety
///
/// `CompletionEvent` can be safely sent across threads because:
/// - The raw pointer in `CompletionEventInner` is only used as an opaque handle
/// - CUDA/ROCm events are internally thread-safe when used with proper synchronization
/// - The event is only used for synchronization and does not expose mutable state
/// - The pointer is never dereferenced directly by this crate
unsafe impl Send for CompletionEvent {}

/// # Safety
///
/// `CompletionEvent` can be safely shared across threads because:
/// - The raw pointer is an opaque handle to a GPU event object
/// - CUDA/ROCm event APIs are thread-safe for query operations
/// - The event does not contain any Rust-managed mutable state
/// - Concurrent access to the event is managed by the GPU driver
unsafe impl Sync for CompletionEvent {}
