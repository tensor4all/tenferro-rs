//! The runtime handle: one `t4a-cuda-runtime` runtime, one device, one allocation
//! domain, one submission lock for explicit host transfers.

use std::sync::{Arc, Mutex};

use tenferro_tensor::AllocationDomainId;

use crate::cuda::{CudaDeviceError, CudaDeviceId, CudaDeviceInfo};

/// Identity of one opened runtime; equal only for clones of the same handle.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct CudaRuntimeIdentity {
    marker: Arc<u8>,
}

impl CudaRuntimeIdentity {
    fn fresh() -> Self {
        Self {
            marker: Arc::new(0),
        }
    }
}

impl PartialEq for CudaRuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.marker, &other.marker)
    }
}

impl Eq for CudaRuntimeIdentity {}

impl std::hash::Hash for CudaRuntimeIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // The marker allocation is retained by every clone, so its address is
        // move- and clone-invariant while an identity is witnessed.
        state.write_usize(Arc::as_ptr(&self.marker) as usize);
    }
}

std::thread_local! {
    /// Addresses of the runtimes whose context this thread has already activated.
    ///
    /// The raw scope takes the runtime's submission lock, so a second activation on the same
    /// thread would block forever; nesting is legal because the context is already current.
    static ACTIVATED_CONTEXTS: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// One activation level of a runtime's context on this thread.
pub(crate) struct ActivatedContext {
    key: usize,
    /// Whether this call found the context already active on this thread.
    nested: bool,
    /// `Some` when this call pushed a level; dropping it pops the level again, including on a
    /// panic, so a leaked level cannot make a later activation silently skip the context.
    pushed: bool,
}

impl ActivatedContext {
    /// Enter `runtime`'s context, reporting whether this thread was already inside it.
    fn enter(runtime: &CudaRuntime) -> ActivatedContext {
        let key = Arc::as_ptr(&runtime.inner) as usize;
        let nested = ACTIVATED_CONTEXTS.with(|active| {
            let mut active = active.borrow_mut();
            if active.contains(&key) {
                true
            } else {
                active.push(key);
                false
            }
        });
        ActivatedContext {
            key,
            nested,
            pushed: !nested,
        }
    }
}

impl Drop for ActivatedContext {
    fn drop(&mut self) {
        if self.pushed {
            ACTIVATED_CONTEXTS.with(|active| {
                active.borrow_mut().retain(|entry| *entry != self.key);
            });
        }
    }
}

/// Default bound on the retained download staging: 8 MiB.
///
/// Pinning costs about 1.9 ms, so the buffer is kept — but it is page-locked memory the allocator
/// cannot reclaim. 8 MiB covers the readbacks this path exists for (solver `info` words, norms,
/// ranks, a 1024x1024 f64 matrix) with room to spare; a larger download stages through a temporary
/// buffer and leaves nothing resident. Raise it with
/// [`CudaRuntime::set_pinned_staging_limit_bytes`] when large downloads repeat.
pub const DEFAULT_PINNED_STAGING_LIMIT_BYTES: usize = 8 * 1024 * 1024;

/// What a runtime's download staging retains, and the bound it retains it under.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), tenferro_gpu::cuda::CudaEngineError> {
/// use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaRuntime};
///
/// # if gpu_available() {
/// let runtime = CudaRuntime::new(CudaDeviceId::from_ordinal(0))?;
/// let stats = runtime.pinned_staging_stats();
/// if stats.retained_bytes > stats.limit_bytes {
///     runtime.clear_pinned_staging();
/// }
/// println!("{stats:?}");
/// # }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedStagingStats {
    /// Bytes the cached staging buffer holds right now.
    pub retained_bytes: usize,
    /// Pinned host allocations this runtime made for staging, cumulative: the first download, each
    /// later growth, and each temporary buffer used above the bound or re-pinned after a clear.
    pub allocations: u64,
    /// Largest download whose staging is retained.
    pub limit_bytes: usize,
}

/// State shared by every handle to one runtime.
pub(crate) struct CudaRuntimeState {
    /// Pinned host staging reused by explicit downloads. Allocating and freeing one per download
    /// cost far more than the copy (a 4-byte `cuMemHostAlloc` measured 1.9 ms beside a 19 us
    /// device-to-host copy), so one buffer lives here and grows on demand.
    pub(crate) pinned_staging: Mutex<Option<crate::cuda::raw::PinnedStaging>>,
    /// What that buffer retains, the bound it retains it under, and how many pinned allocations
    /// this runtime made for staging.
    pub(crate) pinned_staging_accounting: crate::cuda::raw::StagingAccounting,

    /// The execution owner. Explicit transfers take this lock for one admission;
    /// numerical submission borrows it mutably through the engine.
    pub(crate) runtime: Mutex<t4a_cuda_runtime::Runtime>,
    pub(crate) device_id: CudaDeviceId,
    pub(crate) device_info: CudaDeviceInfo,
    pub(crate) allocation_domain: AllocationDomainId,
    pub(crate) identity: CudaRuntimeIdentity,
    /// The allocation pool's retention budget this runtime opened with, in bytes.
    pub(crate) pool_release_threshold: usize,
    /// Handles dropped while their work was still queued. The runtime's retirement
    /// already owns the staging; this queue owns only the caller's packages (a pinned
    /// source, a tensor handle) until their completion is observed.
    pub(crate) detached: Mutex<Vec<Detached>>,
    /// How many scheduled runs are currently enqueueing nodes on this runtime.
    ///
    /// While it is non-zero a run owns the completion proof: every node it enqueues records a
    /// stream-ordered checkpoint, and the run's retirement is the barrier, so an operation's own
    /// launch does not have to wait. Zero means a standalone call, which proves its own launch.
    pub(crate) deferred_completions: std::sync::atomic::AtomicUsize,
    /// Completion checkpoints recorded on this runtime's stream, per node of a scheduled run or
    /// by a transfer. A coalesced run records none: submission order is its order (#2396).
    pub(crate) completion_checkpoints: std::sync::atomic::AtomicU64,
    /// Stream barriers this runtime ran, whoever asked: an operation proving its own launch, a
    /// download, or a caller's explicit synchronize.
    pub(crate) stream_waits: std::sync::atomic::AtomicU64,
}

/// One dropped handle retained until its completion is observed.
pub(crate) struct Detached {
    /// The caller's package: the queued copy still reads or writes it, so it stays owned
    /// here until the completion below proves the work done.
    #[allow(dead_code)]
    pub(crate) keepalive: Box<dyn std::any::Any + Send>,
    pub(crate) completion: Option<t4a_cuda_runtime::Completion>,
}

/// The runtime configuration this build opens with.
///
/// The artifact directory, its bound and the allocation pool's retention budget are read from
/// the environment on every open: `TENFERRO_CUDA_ARTIFACT_CACHE`,
/// `TENFERRO_CUDA_ARTIFACT_CACHE_MIB` and `TENFERRO_CUDA_POOL_RELEASE_MIB`.
///
/// `TENFERRO_CUDA_ARTIFACT_CACHE` names the directory that persists compiled kernel images, so a
/// later process reads them back instead of paying NVRTC again: a whole module is about 8 s per
/// process on an A100, and a warm read is a few milliseconds. The value wins over the default;
/// `none`, `off` and `memory` select the in-memory cache only; unset selects
/// `$XDG_CACHE_HOME/tenferro/t4a-cuda-artifacts`, else `$HOME/.cache/tenferro/t4a-cuda-artifacts`
/// (`%LOCALAPPDATA%\tenferro\t4a-cuda-artifacts` on Windows), and falls back to the in-memory
/// cache when no absolute per-user root is set or the directory cannot be created. A newly created
/// directory is mode 0700 on Unix.
///
/// The directory is bounded, never unlimited: at most [`artifact_cache_max_bytes`] bytes of
/// images, 256 MiB unless `TENFERRO_CUDA_ARTIFACT_CACHE_MIB` says otherwise. Publishing past the
/// bound removes the least recently used images first, best effort: a removal that fails is
/// dropped, so the directory can stay above its bound. [`CudaRuntime::artifact_dir_evictions`]
/// counts what was actually removed, and [`CudaRuntime::artifact_dir_stats`] and
/// [`CudaRuntime::clear_artifact_dir`] read and empty it.
///
/// `TENFERRO_CUDA_POOL_RELEASE_MIB` overrides the allocation pool's retention budget, the idle
/// device memory the pool keeps mapped instead of handing back to the driver. See
/// [`pool_release_threshold_bytes`].
fn default_config(total_memory_bytes: u64) -> t4a_cuda_runtime::Config {
    let artifact_cache_dir = artifact_cache_dir(
        std::env::var_os("TENFERRO_CUDA_ARTIFACT_CACHE").as_deref(),
        |name| std::env::var_os(name),
    )
    .and_then(usable_artifact_cache_dir);
    t4a_cuda_runtime::Config {
        artifact_cache_dir,
        pool_release_threshold: pool_release_threshold_bytes(total_memory_bytes),
        ..t4a_cuda_runtime::Config::default()
    }
}

/// The least idle device memory the pool keeps mapped, whatever the device size.
const MIN_POOL_RELEASE_BYTES: usize = 256 << 20;
/// The most idle device memory the pool keeps mapped, whatever the device size.
const MAX_POOL_RELEASE_BYTES: usize = 4 << 30;
/// The share of the device's memory the pool may keep mapped.
const POOL_RELEASE_SHARE: u64 = 8;

/// Idle device memory the allocation pool keeps mapped instead of returning to the driver.
///
/// The driver's own default is a fixed 256 MiB, which is smaller than the working set of a single
/// large operation: an eager 3072^3 f64 matmul holds about 430 MiB of 72 MiB buffers, so every
/// call's output allocation missed the pool and re-mapped its pages — 864 us of
/// `cuMemAllocFromPoolAsync` per call, 87 of 87 calls on the A100 host, reproducing in plain CUDA
/// once six buffers are live (#2381). A budget scaled to the device keeps those pages.
///
/// `TENFERRO_CUDA_POOL_RELEASE_MIB` overrides the result, in MiB. The budget is a ceiling on
/// retained *idle* memory, not a reservation: nothing is allocated before it is needed.
fn pool_release_threshold_bytes(total_memory_bytes: u64) -> usize {
    let override_mib = std::env::var("TENFERRO_CUDA_POOL_RELEASE_MIB")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok());
    pool_release_threshold_from(override_mib, total_memory_bytes)
}

/// [`pool_release_threshold_bytes`] with the environment's override supplied explicitly.
fn pool_release_threshold_from(override_mib: Option<u64>, total_memory_bytes: u64) -> usize {
    if let Some(mib) = override_mib {
        return usize::try_from(mib.saturating_mul(1 << 20))
            .unwrap_or(usize::MAX)
            .clamp(MIN_POOL_RELEASE_BYTES, MAX_POOL_RELEASE_BYTES);
    }
    let share = usize::try_from(total_memory_bytes / POOL_RELEASE_SHARE).unwrap_or(usize::MAX);
    share.clamp(MIN_POOL_RELEASE_BYTES, MAX_POOL_RELEASE_BYTES)
}

/// The bound on the persistent artifact directory, in bytes.
///
/// `TENFERRO_CUDA_ARTIFACT_CACHE_MIB` (`value`) sets it in MiB. `0` is a bound like any other, not
/// a read-only mode: the runtime applies a bound through
/// [`t4a_cuda_runtime::Runtime::set_artifact_dir_max_bytes`], which trims the directory as it
/// records the bound. That trim is best effort — a removal that fails is dropped — so `0` stops
/// new images being published and does not guarantee the directory is empty; an image a failed
/// trim left behind is still served. A process that wants no persistent directory asks for that
/// with `TENFERRO_CUDA_ARTIFACT_CACHE=none`. Unset, or a value that is not a whole number of MiB,
/// keeps [`t4a_cuda_runtime::DEFAULT_ARTIFACT_DIR_BYTES`] (256 MiB, the in-memory artifact
/// budget): there is no spelling of "unlimited".
fn artifact_cache_max_bytes(value: Option<&std::ffi::OsStr>) -> u64 {
    value
        .and_then(std::ffi::OsStr::to_str)
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(t4a_cuda_runtime::DEFAULT_ARTIFACT_DIR_BYTES, |mib| {
            mib.saturating_mul(1 << 20)
        })
}

/// The one operation opening a runtime performs on its artifact cache.
///
/// The cache lives behind a device, so the open applies the bound through this trait and a test
/// drives a fake instead: a compile of the decision alone would not show that it is applied.
trait ArtifactDirBound {
    fn set_artifact_dir_max_bytes(&mut self, max_bytes: u64);
}

impl ArtifactDirBound for t4a_cuda_runtime::Runtime {
    fn set_artifact_dir_max_bytes(&mut self, max_bytes: u64) {
        // The dependency's own method, not this trait's: no recursion.
        t4a_cuda_runtime::Runtime::set_artifact_dir_max_bytes(self, max_bytes);
    }
}

/// Apply `TENFERRO_CUDA_ARTIFACT_CACHE_MIB` to `cache`, the open's only cache mutation.
fn apply_artifact_cache_bound(
    cache: &mut dyn ArtifactDirBound,
    read: impl Fn(&str) -> Option<std::ffi::OsString>,
) {
    cache.set_artifact_dir_max_bytes(artifact_cache_max_bytes(
        read("TENFERRO_CUDA_ARTIFACT_CACHE_MIB").as_deref(),
    ));
}

/// `tenferro`'s directory under the user's cache root, which holds compiled kernel images.
const ARTIFACT_CACHE_PATH: &str = "tenferro/t4a-cuda-artifacts";

/// The persistent artifact directory the environment selects, or `None` for memory only.
///
/// `read` supplies the environment so the fallback chain is testable without one.
fn artifact_cache_dir(
    value: Option<&std::ffi::OsStr>,
    read: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Option<std::path::PathBuf> {
    if let Some(value) = value {
        return match value.to_str() {
            Some(value) => {
                let value = value.trim();
                if value.is_empty() || matches!(value, "none" | "off" | "memory") {
                    None
                } else {
                    Some(std::path::PathBuf::from(value))
                }
            }
            // Not UTF-8, so it cannot be one of the tokens: it names a directory as it is.
            None => Some(std::path::PathBuf::from(value)),
        };
    }
    cache_root(read).map(|root| root.join(ARTIFACT_CACHE_PATH))
}

/// This user's cache root.
///
/// `$XDG_CACHE_HOME` when it is absolute, else `$HOME/.cache`; a relative `XDG_CACHE_HOME` is
/// ignored, as the specification requires, so the cache location never depends on the working
/// directory. Windows has no XDG and uses `%LOCALAPPDATA%`, which is already a per-user root.
fn cache_root(read: impl Fn(&str) -> Option<std::ffi::OsString>) -> Option<std::path::PathBuf> {
    let root = |name: &str| {
        read(name)
            .filter(|value| !value.is_empty())
            .map(std::path::PathBuf::from)
            .filter(|dir| dir.is_absolute())
    };
    if cfg!(windows) {
        return root("LOCALAPPDATA");
    }
    root("XDG_CACHE_HOME").or_else(|| root("HOME").map(|home| home.join(".cache")))
}

/// The same directory, once this process has created it or found it present.
///
/// A home it cannot create the directory under degrades to the in-memory cache with one warning,
/// never to a failed operation, so nothing here returns an error into the launch path.
fn usable_artifact_cache_dir(dir: std::path::PathBuf) -> Option<std::path::PathBuf> {
    match create_private_dir(&dir) {
        Ok(()) => Some(dir),
        Err(error) => {
            ARTIFACT_CACHE_WARNING.call_once(|| {
                write_best_effort_warning(
                    &mut std::io::stderr(),
                    format_args!(
                        "tenferro-gpu: cannot create the artifact cache directory {}: {error}; \
                         compiled kernels stay in memory for this process",
                        dir.display()
                    ),
                );
            });
            None
        }
    }
}

/// Write one warning line to `sink`, dropping it if the write fails.
///
/// `eprintln!` panics when stderr is closed, and a diagnostic on a best-effort path must never
/// turn into a panic in the library.
pub(crate) fn write_best_effort_warning(
    sink: &mut impl std::io::Write,
    message: std::fmt::Arguments<'_>,
) {
    let _ = writeln!(sink, "{message}");
}

/// One warning per process, however many runtimes open against an unusable directory.
static ARTIFACT_CACHE_WARNING: std::sync::Once = std::sync::Once::new();

/// Create `dir` and any missing parent. A directory this creates is mode 0700 on Unix; an existing
/// one keeps its own mode, so a shared read-only cache stays readable and its writes stay dropped.
fn create_private_dir(dir: &std::path::Path) -> std::io::Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir)
}

/// A cheap, cloneable handle to one opened CUDA runtime.
///
/// The handle carries no submission authority of its own: explicit host transfers
/// take the runtime's submission lock for the duration of one admission, and
/// numerical submission belongs to the [`CudaEngine`](super::CudaEngine) that owns
/// the same state.
#[derive(Clone)]
#[doc(hidden)]
pub struct CudaRuntime {
    pub(crate) inner: Arc<CudaRuntimeState>,
}

impl CudaRuntime {
    /// Open the CUDA runtime on `device_id` with the default configuration.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError`] when the device is unknown or the driver cannot open it.
    #[doc(hidden)]
    pub fn new(device_id: CudaDeviceId) -> Result<CudaRuntime, CudaDeviceError> {
        let total_memory = crate::cuda::cuda_devices()?
            .iter()
            .find(|info| info.id() == device_id)
            .map(crate::cuda::CudaDeviceInfo::total_memory_bytes)
            .unwrap_or(0);
        let runtime = Self::with_config(device_id, default_config(total_memory))?;
        apply_artifact_cache_bound(&mut *runtime.submit(), |name| std::env::var_os(name));
        Ok(runtime)
    }

    /// Open the CUDA runtime on `device_id` with an explicit configuration.
    ///
    /// The configuration bounds the runtime's staging chunk and pending budget, which is
    /// what a caller uses to force chunked transfers.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError`] when the device is unknown, the configuration is invalid, or
    /// the driver cannot open the device.
    #[doc(hidden)]
    pub fn with_config(
        device_id: CudaDeviceId,
        config: t4a_cuda_runtime::Config,
    ) -> Result<CudaRuntime, CudaDeviceError> {
        let devices = crate::cuda::cuda_devices()?;
        let info = devices
            .iter()
            .find(|info| info.id() == device_id)
            .cloned()
            .ok_or_else(|| CudaDeviceError::Unavailable {
                requested: device_id,
                discovered: devices.into_boxed_slice(),
            })?;
        let pool_release_threshold = config.pool_release_threshold;
        let runtime = t4a_cuda_runtime::Runtime::open(device_id.ordinal() as usize, config)
            .map_err(|error| CudaDeviceError::Initialization {
                device: device_id,
                operation: "t4a_cuda_runtime::Runtime::open",
                source: Box::new(error),
            })?;
        Ok(CudaRuntime {
            inner: Arc::new(CudaRuntimeState {
                runtime: Mutex::new(runtime),
                device_id,
                device_info: info,
                allocation_domain: AllocationDomainId::fresh(),
                pool_release_threshold,
                identity: CudaRuntimeIdentity::fresh(),
                detached: Mutex::new(Vec::new()),
                deferred_completions: std::sync::atomic::AtomicUsize::new(0),
                stream_waits: std::sync::atomic::AtomicU64::new(0),
                completion_checkpoints: std::sync::atomic::AtomicU64::new(0),
                pinned_staging: Mutex::new(None),
                pinned_staging_accounting: crate::cuda::raw::StagingAccounting::new(
                    DEFAULT_PINNED_STAGING_LIMIT_BYTES,
                ),
            }),
        })
    }

    /// The device this runtime owns.
    #[doc(hidden)]
    pub fn device_id(&self) -> CudaDeviceId {
        self.inner.device_id
    }

    /// The device metadata this runtime was opened with.
    #[doc(hidden)]
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.inner.device_info
    }

    /// The ordinal of the owning device.
    #[doc(hidden)]
    pub fn device_ordinal(&self) -> usize {
        self.inner.device_id.ordinal() as usize
    }

    /// The identity of this runtime; distinct for every opened runtime.
    #[doc(hidden)]
    pub fn identity(&self) -> &CudaRuntimeIdentity {
        &self.inner.identity
    }

    /// The shared-allocation domain every buffer of this runtime belongs to.
    #[doc(hidden)]
    pub fn allocation_domain_id(&self) -> AllocationDomainId {
        self.inner.allocation_domain
    }

    /// NVRTC compilation attempts this runtime's artifact cache performed.
    pub fn compile_attempts(&self) -> u64 {
        self.submit().compile_attempts()
    }

    /// Module loads this runtime performed.
    pub fn module_loads(&self) -> u64 {
        self.submit().module_loads()
    }

    /// Artifact-cache hits this runtime served.
    pub fn artifact_cache_hits(&self) -> u64 {
        self.submit().artifact_cache_hits()
    }

    /// Images and bytes in the persistent artifact directory; empty for an in-memory cache.
    ///
    /// # Errors
    ///
    /// The [`std::io::Error`] from listing a directory that exists but cannot be read, such as
    /// `ErrorKind::PermissionDenied`. A directory that does not exist yet is not an error; it
    /// reports zero entries.
    pub fn artifact_dir_stats(&self) -> std::io::Result<t4a_cuda_runtime::ArtifactDirStats> {
        self.submit().artifact_dir_stats()
    }

    /// The bound on the persistent artifact directory, in bytes.
    pub fn artifact_dir_max_bytes(&self) -> u64 {
        self.submit().artifact_dir_max_bytes()
    }

    /// Remove every compiled image from the persistent artifact directory and return what went.
    ///
    /// Images already loaded by this process stay usable; only later processes recompile.
    ///
    /// # Errors
    ///
    /// The [`std::io::Error`] from listing the directory or removing an entry, such as
    /// `ErrorKind::PermissionDenied` on a read-only cache. An entry another process removed
    /// first (`ErrorKind::NotFound`) is not an error.
    pub fn clear_artifact_dir(&self) -> std::io::Result<t4a_cuda_runtime::ArtifactDirStats> {
        self.submit().clear_artifact_dir()
    }

    /// Images this runtime actually removed from the persistent artifact directory to stay within
    /// its bound, least recently used first.
    ///
    /// Trimming is best effort, so a removal that failed is not counted here and the directory can
    /// stay above its bound; [`CudaRuntime::artifact_dir_stats`] reports what is really there.
    pub fn artifact_dir_evictions(&self) -> u64 {
        self.submit().artifact_dir_evictions()
    }

    /// Run `closure` inside a raw scope on this runtime's stream, under the submission
    /// lock.
    ///
    /// # Safety
    ///
    /// As [`crate::Runtime::with_raw`](t4a_cuda_runtime::Runtime::with_raw): the closure
    /// must use this runtime's stream, keep its enqueues within the registered resources
    /// and obey each kernel's or vendor library's contract.
    pub(crate) unsafe fn with_raw_locked<R>(
        &self,
        resources: Vec<t4a_cuda_runtime::RawResource<'_>>,
        closure: impl FnOnce(t4a_cuda_runtime::RawScope<'_>) -> R,
    ) -> crate::Result<R> {
        let mut guard = self.submit();
        // SAFETY: forwarded to the caller's contract.
        unsafe { guard.with_raw(resources, closure) }
            .map_err(|error| crate::Error::runtime_state_source("cuda raw scope", error))
    }

    /// Run `f` with this runtime's CUDA context current, for a vendor call that only sets up state
    /// (a cuFFT or cuSOLVER plan) and enqueues nothing.
    ///
    /// # Errors
    ///
    /// The closure's own failure, and the runtime's when the context cannot be activated.
    #[doc(hidden)]
    pub fn with_current_context<R>(
        &self,
        op: &'static str,
        f: impl FnOnce() -> R,
    ) -> crate::Result<R> {
        // The scope holds the runtime's submission lock, so a nested call would deadlock on it (a
        // cuFFT plan's rollback runs inside the same context). Nesting is tracked per thread:
        // inside an activated context the closure simply runs, and the level is popped when this
        // call leaves, panic included.
        let activation = ActivatedContext::enter(self);
        if activation.nested {
            return Ok(f());
        }
        // SAFETY: the scope activates this runtime's context for the closure's duration; it declares
        // no resources, so nothing is registered against the closure's work.
        unsafe { self.with_raw_locked(Vec::new(), |_scope| f()) }
            .map_err(|error| crate::Error::runtime_state_source(op, error))
    }

    /// Resource counters of this runtime.
    ///
    /// The snapshot is the runtime's own: live and pooled device bytes, pending
    /// host/module bytes, retirement records and the transfer counters. The
    /// compatibility package adds its detached-handle count beside them
    /// ([`CudaRuntime::pending_transfer_retirements`]).
    #[doc(hidden)]
    pub fn stats(&self) -> t4a_cuda_runtime::Stats {
        let stats = self.submit().stats();
        self.retire_detached();
        stats
    }

    /// Copy `value` from `self` into a fresh allocation of `destination`.
    ///
    /// The copy is ordered on both runtimes' streams: the producer may be dropped and
    /// the value reused immediately, and a later read on the destination stream observes
    /// the copied bytes without a host wait.
    ///
    /// # Errors
    ///
    /// A typed refusal when the two handles are the same runtime, when `value` is not a
    /// CUDA allocation of this runtime, or when the driver cannot admit the copy.
    #[doc(hidden)]
    pub fn copy_to(
        &self,
        destination: &CudaRuntime,
        value: &tenferro_tensor::Tensor,
    ) -> crate::Result<tenferro_tensor::Tensor> {
        use tenferro_tensor::{Complex32, Complex64, DType, TensorScalar, TypedTensor};

        if self.identity() == destination.identity() {
            return Err(crate::Error::runtime_state(
                "cuda copy",
                "the source and destination runtimes are the same",
            ));
        }
        let storage = super::pending::cuda_storage(value).ok_or_else(|| {
            crate::Error::runtime_state("cuda copy", "expected a CUDA allocation")
        })?;
        if storage.device_ordinal() != self.device_ordinal() {
            return Err(crate::Error::runtime_state(
                "cuda copy",
                "the value belongs to another runtime",
            ));
        }

        // Both runtimes participate in admission. Take their submission locks in a
        // stable order (by allocation domain) so two opposing copies cannot invert them,
        // while the copy direction stays source -> destination in both branches.
        let source_is_lower = self.inner.allocation_domain < destination.inner.allocation_domain;
        let (buffer, _completion) = if source_is_lower {
            let mut source_guard = self.submit();
            let mut destination_guard = destination.submit();
            let (buffer, completion, _route) = destination_guard
                .copy_between(&mut source_guard, storage.allocation())
                .map_err(|error| crate::Error::runtime_state_source("cuda copy", error))?;
            (buffer, completion)
        } else {
            let mut destination_guard = destination.submit();
            let mut source_guard = self.submit();
            let (buffer, completion, _route) = destination_guard
                .copy_between(&mut source_guard, storage.allocation())
                .map_err(|error| crate::Error::runtime_state_source("cuda copy", error))?;
            (buffer, completion)
        };
        fn rebuild<T: TensorScalar + Send + Sync + 'static>(
            value: &tenferro_tensor::Tensor,
            buffer: t4a_cuda_runtime::Buffer,
            destination: &CudaRuntime,
        ) -> crate::Result<tenferro_tensor::Tensor> {
            let typed: &TypedTensor<T> = value.as_typed::<T>().ok_or_else(|| {
                crate::Error::runtime_state("cuda copy", "expected a preset scalar")
            })?;
            let storage = super::storage::fresh_storage(destination, buffer);
            TypedTensor::<T>::from_buffer_col_major(
                typed.shape().to_vec(),
                tenferro_tensor::StorageBuffer::Backend(Box::new(storage)),
                super::transfer::placement(destination),
            )
            .map(tenferro_tensor::Tensor::from_typed)
        }
        match value.dtype() {
            DType::F64 => rebuild::<f64>(value, buffer, destination),
            DType::F32 => rebuild::<f32>(value, buffer, destination),
            DType::I32 => rebuild::<i32>(value, buffer, destination),
            DType::I64 => rebuild::<i64>(value, buffer, destination),
            DType::C32 => rebuild::<Complex32>(value, buffer, destination),
            DType::C64 => rebuild::<Complex64>(value, buffer, destination),
            DType::Bool | DType::External(_) => Err(crate::Error::unsupported(
                "cuda copy",
                "the transfer route does not serve this dtype",
            )),
        }
    }

    /// Read a runtime allocation back into a byte vector.
    ///
    /// # Errors
    ///
    /// [`cuda::CudaDeviceError::Initialization`] when the copy cannot be admitted or
    /// completed.
    ///
    /// [`cuda::CudaDeviceError::Initialization`]: crate::cuda::CudaDeviceError::Initialization
    pub fn download_bytes(&self, buffer: &t4a_cuda_runtime::Buffer) -> crate::Result<Vec<u8>> {
        let mut guard = self.submit();
        let mut pending = guard
            .download(buffer)
            .map_err(|error| crate::Error::runtime_state_source("cuda download", error))?;
        pending
            .wait()
            .map_err(|error| crate::Error::runtime_state_source("cuda download", error))
    }

    /// Admit one raw host-to-device upload and return its allocation.
    ///
    /// This is the seam's own upload path, used by callers that already hold bytes (the
    /// publication test queues work with it); a typed [`upload_tensor`] wraps it with
    /// the tensor and placement metadata.
    ///
    /// [`upload_tensor`]: super::upload_tensor
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the runtime cannot allocate or enqueue
    /// the copy.
    #[doc(hidden)]
    pub fn upload_bytes(&self, bytes: &[u8]) -> Result<t4a_cuda_runtime::Buffer, CudaDeviceError> {
        let mut guard = self.submit();
        guard
            .upload(bytes)
            .map_err(|error| CudaDeviceError::Initialization {
                device: self.inner.device_id,
                operation: "t4a_cuda_runtime::Runtime::upload",
                source: Box::new(error),
            })
    }

    /// Allocate a pinned host buffer owned by this runtime.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the driver cannot allocate pinned
    /// memory.
    #[doc(hidden)]
    pub fn alloc_pinned(
        &self,
        bytes: usize,
    ) -> Result<t4a_cuda_runtime::PinnedHostBuffer, CudaDeviceError> {
        let mut guard = self.submit();
        guard
            .alloc_pinned(bytes)
            .map_err(|error| CudaDeviceError::Initialization {
                device: self.inner.device_id,
                operation: "t4a_cuda_runtime::Runtime::alloc_pinned",
                source: Box::new(error),
            })
    }

    /// Pinned staging allocations this runtime made, whoever asked.
    ///
    /// One for the first download and one per later growth of the staging buffer: a download that
    /// allocated per call would count every download instead.
    #[doc(hidden)]
    #[must_use]
    pub fn pinned_staging_allocations(&self) -> u64 {
        self.inner.pinned_staging_accounting.allocations()
    }

    /// What the download staging currently retains: bytes, pinned allocations, and the bound.
    ///
    /// Downloads reuse one page-locked host buffer, because pinning it costs about 1.9 ms beside a
    /// 19 us copy. The buffer is memory the OS cannot reclaim, so it is bounded (see
    /// [`CudaRuntime::set_pinned_staging_limit_bytes`]) and can be released early with
    /// [`CudaRuntime::clear_pinned_staging`].
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), tenferro_gpu::cuda::CudaEngineError> {
    /// use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaRuntime};
    ///
    /// # if gpu_available() {
    /// let runtime = CudaRuntime::new(CudaDeviceId::from_ordinal(0))?;
    /// let stats = runtime.pinned_staging_stats();
    /// println!("{} bytes retained, {} allocations", stats.retained_bytes, stats.allocations);
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn pinned_staging_stats(&self) -> PinnedStagingStats {
        let accounting = &self.inner.pinned_staging_accounting;
        PinnedStagingStats {
            retained_bytes: accounting.retained(),
            allocations: accounting.allocations(),
            limit_bytes: accounting.limit(),
        }
    }

    /// Bound the download staging, releasing a larger buffer at once.
    ///
    /// A download above this size still stages through host memory, but through a buffer released
    /// when that download returns, so nothing above the bound stays resident. Lowering the bound
    /// below what is retained releases the cached buffer; `usize::MAX` keeps every download.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), tenferro_gpu::cuda::CudaEngineError> {
    /// use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaRuntime};
    ///
    /// # if gpu_available() {
    /// let runtime = CudaRuntime::new(CudaDeviceId::from_ordinal(0))?;
    /// runtime.set_pinned_staging_limit_bytes(1024 * 1024);
    /// assert_eq!(runtime.pinned_staging_stats().limit_bytes, 1024 * 1024);
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_pinned_staging_limit_bytes(&self, limit: usize) {
        self.inner.pinned_staging_accounting.set_limit(limit);
        self.release_staging_above(limit);
    }

    /// Release the retained download staging now.
    ///
    /// The next download pins a buffer again. Returns the bytes released, so a caller can tell the
    /// call apart from one that found nothing to release.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), tenferro_gpu::cuda::CudaEngineError> {
    /// use tenferro_gpu::cuda::{gpu_available, CudaDeviceId, CudaRuntime};
    ///
    /// # if gpu_available() {
    /// let runtime = CudaRuntime::new(CudaDeviceId::from_ordinal(0))?;
    /// let released = runtime.clear_pinned_staging();
    /// assert_eq!(runtime.pinned_staging_stats().retained_bytes, 0);
    /// # let _ = released;
    /// # }
    /// # Ok(())
    /// # }
    /// ```
    pub fn clear_pinned_staging(&self) -> usize {
        self.release_staging_above(0)
    }

    /// Drop the cached buffer when it exceeds `limit`, and report the bytes actually released.
    ///
    /// The free runs inside this runtime's raw scope, which makes the owning context current on
    /// this thread; without it `cuMemFreeHost` can fail with `CUDA_ERROR_INVALID_CONTEXT` on a
    /// thread that never activated the context. A failed free keeps the buffer and its accounting,
    /// because dropping the pointer would leak the allocation while reporting it released.
    fn release_staging_above(&self, limit: usize) -> usize {
        let Ok(mut slot) = self.inner.pinned_staging.lock() else {
            return 0;
        };
        let Some(entry) = slot.as_ref() else {
            return 0;
        };
        let released = entry.bytes();
        if released == 0 || released <= limit {
            return 0;
        }
        // No download is in flight: it would hold this lock, and it synchronizes before releasing
        // it, so nothing queued still reads this buffer.
        let pointer = entry.pointer();
        let freed = unsafe {
            self.with_raw_locked(Vec::new(), |_raw| {
                cudarc::driver::sys::cuMemFreeHost(pointer)
            })
        };
        match freed {
            Ok(cudarc::driver::sys::CUresult::CUDA_SUCCESS) => {
                *slot = None;
                self.inner.pinned_staging_accounting.record_retained(0);
                released
            }
            other => {
                // Keep the buffer: it is still allocated, so the accounting stays truthful and a
                // later clear can retry.
                let _ = other;
                0
            }
        }
    }

    /// Wait for every queued operation and retire the detached handles it completes.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the stream reports a failure.
    #[doc(hidden)]
    pub fn synchronize(&self) -> Result<(), CudaDeviceError> {
        self.inner
            .stream_waits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        {
            // Hold the submission lock through the reclamation: a concurrent clone cannot
            // enqueue a transfer it then detaches in the gap between the stream wait and
            // the queue drain, which would free its pinned source while still queued.
            let mut guard = self.submit();
            guard
                .synchronize()
                .map_err(|error| CudaDeviceError::Initialization {
                    device: self.inner.device_id,
                    operation: "t4a_cuda_runtime::Runtime::synchronize",
                    source: Box::new(error),
                })?;
            self.retire_detached_locked();
        }
        Ok(())
    }

    /// Completion checkpoints this runtime recorded on its stream.
    ///
    /// A scheduled run records one per node, unless it took the single-domain coalesced path,
    /// where submission order is the ordering and the run's retirement is the only barrier.
    /// Exposed so a regression test can pin that (#2396).
    #[doc(hidden)]
    pub fn completion_checkpoints(&self) -> u64 {
        self.inner
            .completion_checkpoints
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record one completion checkpoint here, for callers that own the submission lock.
    pub(crate) fn note_completion_checkpoint(&self) {
        self.inner
            .completion_checkpoints
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stream barriers this runtime ran, including the ones a scheduled run's operations skip.
    ///
    /// A prepared run adds exactly the caller's own barrier; an operation that drains its own
    /// launch adds one per launch. Exposed so the regression test can count real waits.
    #[doc(hidden)]
    pub fn stream_waits(&self) -> u64 {
        self.inner
            .stream_waits
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether a scheduled run currently owns the completion of the launches it enqueues.
    ///
    /// A run that holds the deferral records a stream-ordered checkpoint after every node and is
    /// the barrier for its own work (see the CUDA engine's event domain), so an operation's launch
    /// does not prove its own completion while it is zero. A standalone call sees zero and waits.
    #[doc(hidden)]
    pub fn completion_deferred(&self) -> bool {
        self.inner
            .deferred_completions
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    }

    /// Take a scheduled run's completion ownership for as long as the guard lives.
    ///
    /// Nested guards are counted, so a run enqueueing a node that itself starts a run keeps the
    /// deferral in force until the outermost guard is dropped.
    pub(crate) fn defer_completion(&self) -> CompletionDeferral<'_> {
        self.inner
            .deferred_completions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        CompletionDeferral { runtime: self }
    }

    /// Idle device memory the allocation pool keeps mapped before returning it to the driver.
    ///
    /// Reported next to [`Self::stats`] because it is the bound the pool's retained bytes obey
    /// (`PERFORMANCE_TIPS.md`, Cache Ownership): raise it with
    /// `TENFERRO_CUDA_POOL_RELEASE_MIB` when a working set of live buffers exceeds it, which is
    /// what made every eager 3072^3 output allocation re-map its pages (#2381).
    #[doc(hidden)]
    pub fn pool_release_threshold_bytes(&self) -> usize {
        self.inner.pool_release_threshold
    }

    /// Return idle pool memory above `keep_bytes` to the driver now.
    ///
    /// The pool trims itself at the threshold recorded by
    /// [`Self::pool_release_threshold_bytes`]; this is the explicit clear that a caller uses to
    /// give device memory back, for example before handing the device to another library.
    ///
    /// # Errors
    ///
    /// [`CudaDeviceError::Initialization`] when the driver refuses the trim.
    #[doc(hidden)]
    pub fn trim_pool(&self, keep_bytes: usize) -> Result<(), CudaDeviceError> {
        self.submit()
            .trim_pool(keep_bytes)
            .map_err(|error| CudaDeviceError::Initialization {
                device: self.inner.device_id,
                operation: "t4a_cuda_runtime::Runtime::trim_pool",
                source: Box::new(error),
            })
    }

    /// Detached pending handles that have not been retired yet, the compatibility
    /// package's own count (upstream `pending_records` counts retirement epochs, a
    /// different quantity).
    #[doc(hidden)]
    pub fn pending_transfer_retirements(&self) -> u64 {
        self.inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .len() as u64
    }

    /// Retain a dropped handle until its completion is observed.
    pub(crate) fn detach(
        &self,
        keepalive: Box<dyn std::any::Any + Send>,
        completion: Option<t4a_cuda_runtime::Completion>,
    ) {
        self.inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push(Detached {
                keepalive,
                completion,
            });
    }

    /// Release the completed detached packages while the caller already holds the
    /// submission lock.
    fn retire_detached_locked(&self) {
        let mut detached = self
            .inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        detached.retain_mut(|entry| match entry.completion.as_mut() {
            Some(completion) => !matches!(completion.poll(), Ok(Some(()))),
            None => false,
        });
    }

    /// Release the detached packages whose completion is observed, keeping the rest.
    ///
    /// Reclamation is explicit: a package is only released once its own completion says
    /// the work is done, so a queued transfer never loses its source or destination.
    pub(crate) fn retire_detached(&self) {
        let mut detached = self
            .inner
            .detached
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        detached.retain_mut(|entry| match entry.completion.as_mut() {
            Some(completion) => match completion.poll() {
                Ok(Some(())) => false,
                Ok(None) => true,
                // A failed completion is kept: the failure surfaces through the next
                // explicit observation rather than being dropped silently here.
                Err(_) => true,
            },
            None => false,
        });
    }

    /// Synchronise and then reclaim what completed, for teardown paths that cannot
    /// report an error.
    pub(crate) fn synchronize_best_effort(&self) {
        {
            let mut guard = self.submit();
            let _ = guard.synchronize();
        }
        self.retire_detached();
    }

    /// Take the submission lock for one explicit transfer.
    pub(crate) fn submit(&self) -> std::sync::MutexGuard<'_, t4a_cuda_runtime::Runtime> {
        self.inner
            .runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl Drop for CudaRuntimeState {
    fn drop(&mut self) {
        // Last handle: drain the stream, then release the detached packages. A failure
        // leaves their completion unproved, so the queue is leaked instead of freeing
        // pinned memory a queued copy may still read or write.
        let proved = self
            .runtime
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .synchronize()
            .is_ok();
        if proved {
            let mut detached = self
                .detached
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            detached.clear();
        } else {
            let leaked = std::mem::take(
                &mut *self
                    .detached
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner()),
            );
            std::mem::forget(leaked);
        }
    }
}

/// One scheduled run's ownership of its nodes' launch completion, for as long as it is enqueued.
///
/// Dropping the guard hands completion back to the operations, which is what a standalone call
/// expects; it is deliberately trivial so no failure path can leave the count raised.
pub(crate) struct CompletionDeferral<'a> {
    runtime: &'a CudaRuntime,
}

impl Drop for CompletionDeferral<'_> {
    fn drop(&mut self) {
        self.runtime
            .inner
            .deferred_completions
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl std::fmt::Debug for CompletionDeferral<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CompletionDeferral")
    }
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("device", &self.inner.device_id)
            .field("domain", &self.inner.allocation_domain)
            .finish()
    }
}

#[cfg(test)]
mod tests;
