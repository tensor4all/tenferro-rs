//! The Metal context: one device, one command queue, one shared allocation domain, and
//! the paired CPU engine that shares it.
//!
//! Every Metal tensor this context hands out lives in the domain it owns, so f32 work runs
//! on the GPU and f64 work runs on the paired [`tenferro_cpu::CpuEngine`] over the same
//! tensors, with no transfer and no second storage path (design §4.2, §4.3). Upload and
//! download are the only copies that exist, both explicit and both counted.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use t4a_metal_runtime::{Config, Runtime};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    AllocationDomainId, AllocationId, BackendStorage, DType, Error, MemoryKind, Placement,
    StorageBuffer, Tensor, TensorRead, TensorScalar, TypedTensor,
};

use super::device::{metal_devices, MetalDeviceError, MetalDeviceInfo};
use super::domain::MetalAllocationDomain;
use super::engine::MetalEngine;
use super::pending::PendingTouches;
use super::storage::MetalStorage;
use super::MetalBackend;

/// Runtime configuration for one Metal context.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MetalConfig {
    /// The runtime's bounds: in-flight submissions, the completion timeout, and the
    /// artifact and pipeline cache limits.
    pub runtime: Config,
}

/// A failure of the Metal backend itself.
#[derive(Debug)]
#[non_exhaustive]
pub enum MetalError {
    /// This host has no usable Metal device.
    ///
    /// The detail is what the probe reported, so a caller can print it and continue with
    /// another backend rather than panicking.
    Unavailable {
        /// What was probed, from the runtime.
        what: String,
        /// Why it failed, from the runtime.
        detail: String,
    },
    /// The runtime reported a failure this backend does not translate further.
    Runtime {
        /// The operation that failed.
        op: &'static str,
        /// The runtime's own message.
        detail: String,
    },
}

impl MetalError {
    fn runtime(op: &'static str, error: t4a_metal_runtime::Error) -> Self {
        match error {
            t4a_metal_runtime::Error::Unavailable { what, detail } => {
                MetalError::Unavailable { what, detail }
            }
            other => MetalError::Runtime {
                op,
                detail: other.to_string(),
            },
        }
    }
}

impl From<MetalDeviceError> for MetalError {
    fn from(error: MetalDeviceError) -> Self {
        match error {
            MetalDeviceError::Unavailable { what, detail } => {
                MetalError::Unavailable { what, detail }
            }
        }
    }
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::Unavailable { what, detail } => {
                write!(formatter, "no usable Metal device: {what}: {detail}")
            }
            MetalError::Runtime { op, detail } => {
                write!(formatter, "{op} on the Metal backend failed: {detail}")
            }
        }
    }
}

impl std::error::Error for MetalError {}

impl From<MetalError> for Error {
    fn from(error: MetalError) -> Self {
        match error {
            MetalError::Unavailable { what, detail } => Error::unsupported(
                "metal",
                format!("no usable Metal device ({what}: {detail})"),
            ),
            MetalError::Runtime { op, detail } => Error::runtime_state(op, detail),
        }
    }
}

/// What one context's shared domain has done.
///
/// The transfer counters are the measured form of "no transfer": an operation that copies
/// between host and device raises them, and an operation that does not leaves them alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetalStats {
    /// Allocations handed out by the domain, including zero-length ones.
    pub allocations: u64,
    /// Bytes held by live tensors allocated in this domain.
    ///
    /// A storage counts once when it is created and subtracts the same bytes when it is
    /// dropped, so the value returns to its baseline after the last tensor is released. A
    /// buffer allocated inside an operation but not yet wrapped (a launch destination) is
    /// not counted; it is not reachable by the caller.
    pub live_bytes: u64,
    /// Explicit uploads.
    pub uploads: u64,
    /// Bytes copied by explicit uploads.
    pub upload_bytes: u64,
    /// Explicit downloads.
    pub downloads: u64,
    /// Bytes copied by explicit downloads.
    pub download_bytes: u64,
    /// Dispatches encoded on this domain's queue.
    pub launches: u64,
    /// Command buffers committed.
    pub submissions: u64,
    /// Host waits for pending device work, across every allocation.
    pub waits: u64,
}

/// The runtime behind the domain's mutex, and the domain's bookkeeping.
///
/// Every use of the runtime — allocation, compile, bind, launch encoding, commit,
/// synchronize and the host-access admission — takes this one lock, so host access and
/// submission are serialised and a host guard cannot appear between a submission's
/// admission check and its commit.
pub(crate) struct RuntimeCell {
    runtime: Mutex<Runtime>,
    domain: AllocationDomainId,
    next_allocation: AtomicU64,
    allocations: AtomicU64,
    live_bytes: AtomicU64,
    uploads: AtomicU64,
    upload_bytes: AtomicU64,
    downloads: AtomicU64,
    download_bytes: AtomicU64,
    waits: Arc<AtomicU64>,
}

impl std::fmt::Debug for RuntimeCell {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeCell")
            .field("domain", &self.domain)
            .finish_non_exhaustive()
    }
}

impl RuntimeCell {
    fn new(runtime: Runtime) -> Self {
        Self {
            runtime: Mutex::new(runtime),
            domain: AllocationDomainId::fresh(),
            next_allocation: AtomicU64::new(1),
            allocations: AtomicU64::new(0),
            live_bytes: AtomicU64::new(0),
            uploads: AtomicU64::new(0),
            upload_bytes: AtomicU64::new(0),
            downloads: AtomicU64::new(0),
            download_bytes: AtomicU64::new(0),
            waits: Arc::new(AtomicU64::new(0)),
        }
    }

    /// The allocation domain every allocation of this context belongs to.
    pub(crate) fn domain(&self) -> AllocationDomainId {
        self.domain
    }

    /// The shared wait counter, handed to each allocation's bookkeeping.
    pub(crate) fn waits(&self) -> &Arc<AtomicU64> {
        &self.waits
    }

    /// Take the one runtime lock.
    ///
    /// A poisoned lock is recovered: the guarded value is the runtime, and every operation
    /// this crate performs on it either completes or returns before touching it again, so
    /// refusing all later work would strand the context for no safety.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Runtime> {
        self.runtime
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Run `f` with the runtime locked.
    pub(crate) fn with_runtime<T>(
        &self,
        f: impl FnOnce(&mut Runtime) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut runtime = self.lock();
        f(&mut runtime)
    }

    /// Number of bytes one allocation of `shape` of `T` needs.
    fn element_count(shape: &[usize], op: &'static str) -> Result<usize, Error> {
        shape
            .iter()
            .try_fold(1usize, |product, &extent| product.checked_mul(extent))
            .ok_or_else(|| Error::runtime_state(op, "the shape's element count overflows usize"))
    }

    /// Allocate a zeroed tensor of `dtype` in this domain.
    ///
    /// Every `DType` is served here, not only the ones the GPU computes: the paired CPU
    /// engine allocates its outputs in this domain too (design §4.3), which is what makes a
    /// CPU result chain back into a Metal operation without an upload. It is the engine,
    /// not the allocator, that refuses f64 and Complex64 operations.
    pub(crate) fn allocate(
        self: &Arc<Self>,
        op: &'static str,
        dtype: DType,
        shape: &[usize],
    ) -> Result<Tensor, Error> {
        match dtype {
            DType::F32 => self.allocate_typed::<f32>(op, shape),
            DType::F64 => self.allocate_typed::<f64>(op, shape),
            DType::I32 => self.allocate_typed::<i32>(op, shape),
            DType::I64 => self.allocate_typed::<i64>(op, shape),
            DType::Bool => self.allocate_typed::<bool>(op, shape),
            DType::C32 => self.allocate_typed::<num_complex::Complex32>(op, shape),
            DType::C64 => self.allocate_typed::<num_complex::Complex64>(op, shape),
            other => Err(Error::unsupported_dtype(
                op,
                other,
                "the Metal shared domain allocates the tensor dtypes only",
            )),
        }
    }

    /// Allocate `bytes` in this domain without initialising them.
    ///
    /// A launch that fully overwrites the allocation uses this instead of a zeroed
    /// allocation, which would write every byte twice.
    pub(crate) fn allocate_uninit(
        &self,
        op: &'static str,
        bytes: usize,
    ) -> Result<t4a_metal_runtime::UninitBuffer, Error> {
        let buffer = self.with_runtime(|runtime| {
            runtime.alloc_uninit(bytes).map_err(|error| {
                Error::runtime_state(op, format!("the shared allocation failed: {error}"))
            })
        })?;
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(buffer)
    }

    /// Wrap a runtime buffer as an in-domain tensor of `shape`.
    ///
    /// The buffer must hold exactly the elements `shape` names; the caller produced it
    /// from that shape, so the check is a debug assertion rather than a policy.
    pub(crate) fn wrap<T>(
        self: &Arc<Self>,
        op: &'static str,
        shape: Vec<usize>,
        buffer: t4a_metal_runtime::Buffer,
        touches: Arc<PendingTouches>,
    ) -> Result<Tensor, Error>
    where
        T: TensorScalar + Copy + Send + Sync + 'static,
    {
        let elements = Self::element_count(&shape, op)?;
        let expected = elements
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::runtime_state(op, "the allocation size overflows usize"))?;
        if buffer.len() != expected {
            return Err(Error::runtime_state(
                op,
                format!(
                    "the allocation holds {} bytes but the shape needs {expected}",
                    buffer.len()
                ),
            ));
        }
        let identity = crate::gpu::StorageIdentity {
            domain: self.domain,
            allocation: AllocationId::from_backend_id(
                self.next_allocation.fetch_add(1, Ordering::Relaxed),
            ),
            device: 0,
        };
        let seam =
            crate::gpu::Buffer::<MetalBackend>::new(identity, buffer.len(), Arc::new(buffer));
        let storage: Box<dyn BackendStorage<T>> = Box::new(MetalStorage::<T>::new(
            seam,
            touches,
            Arc::clone(self),
            elements,
        ));
        wrap_typed::<T>(shape, storage)
    }

    fn allocate_typed<T>(
        self: &Arc<Self>,
        op: &'static str,
        shape: &[usize],
    ) -> Result<Tensor, Error>
    where
        T: TensorScalar + Send + Sync + 'static,
    {
        let elements = Self::element_count(shape, op)?;
        let bytes = elements
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::runtime_state(op, "the allocation size overflows usize"))?;
        let buffer = self.with_runtime(|runtime| {
            runtime.alloc_zeroed(bytes).map_err(|error| {
                Error::runtime_state(op, format!("the shared allocation failed: {error}"))
            })
        })?;
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.wrap::<T>(
            op,
            shape.to_vec(),
            buffer,
            Arc::new(PendingTouches::new(Arc::clone(&self.waits))),
        )
    }

    /// Record the bytes of a storage just created in this domain.
    pub(crate) fn note_live_allocation(&self, bytes: usize) {
        self.live_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Record the bytes released with a storage that just dropped.
    pub(crate) fn note_live_deallocation(&self, bytes: usize) {
        let previous = self.live_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
        debug_assert!(
            previous >= bytes as u64,
            "live_bytes would underflow: {previous} < {bytes}"
        );
    }

    fn note_upload(&self, bytes: usize) {
        self.uploads.fetch_add(1, Ordering::Relaxed);
        self.upload_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn note_download(&self, bytes: usize) {
        self.downloads.fetch_add(1, Ordering::Relaxed);
        self.download_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// Build an in-domain tensor from a Metal storage box.
fn wrap_typed<T>(shape: Vec<usize>, storage: Box<dyn BackendStorage<T>>) -> Result<Tensor, Error>
where
    T: TensorScalar + Copy + Send + Sync + 'static,
{
    let placement = Placement {
        memory_kind: MemoryKind::Managed,
        device: None,
        cpu_affinity: None,
    };
    let typed =
        TypedTensor::<T>::from_buffer_col_major(shape, StorageBuffer::Backend(storage), placement)?;
    Ok(Tensor::from_typed(typed))
}

/// One Metal device, command queue and shared allocation domain, with its paired CPU
/// engine.
///
/// Cloning a context shares the one device, queue and domain; independently opened
/// contexts get distinct domains (design §4.2).
#[derive(Debug, Clone)]
pub struct MetalContext {
    cell: Arc<RuntimeCell>,
    domain: Arc<MetalAllocationDomain>,
    cpu: Arc<CpuEngine>,
    device: MetalDeviceInfo,
}

impl MetalContext {
    /// Probe, then open the device, its queue and its shared allocation domain.
    ///
    /// Nothing is constructed before the probe answers, and a host without Metal gets
    /// [`MetalError::Unavailable`] naming what failed rather than a panic.
    ///
    /// # Errors
    ///
    /// [`MetalError::Unavailable`] when the frameworks, a required symbol or a device is
    /// missing, and [`MetalError::Runtime`] when the runtime refuses to open.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_gpu::metal::MetalContext;
    ///
    /// match MetalContext::open(Default::default()) {
    ///     Ok(context) => assert!(!context.device().name.is_empty()),
    ///     Err(error) => assert!(!error.to_string().is_empty()),
    /// }
    /// ```
    pub fn open(config: MetalConfig) -> Result<Self, MetalError> {
        let device =
            metal_devices()?
                .into_iter()
                .next()
                .ok_or_else(|| MetalError::Unavailable {
                    what: "MTLCreateSystemDefaultDevice".to_string(),
                    detail: "the probe listed no device".to_string(),
                })?;
        let runtime = Runtime::open(config.runtime)
            .map_err(|error| MetalError::runtime("MetalContext::open", error))?;
        let cell = Arc::new(RuntimeCell::new(runtime));
        let domain = Arc::new(MetalAllocationDomain::new(Arc::clone(&cell)));
        let cpu = Arc::new(CpuEngine::serial().with_allocation_domain(domain.clone()));
        Ok(Self {
            cell,
            domain,
            cpu,
            device,
        })
    }

    /// Facts about the device the probe reported.
    ///
    /// The memory facts and the name come from Metal; the dtype fields are what the probe
    /// measured on this host, so `f64` is what the compiler answered rather than a
    /// declaration.
    pub fn device(&self) -> MetalDeviceInfo {
        self.device.clone()
    }

    /// The identity of the shared allocation domain every Metal tensor lives in.
    pub fn allocation_domain(&self) -> AllocationDomainId {
        self.cell.domain()
    }

    /// The domain owner, as the CPU engine sees it.
    pub fn domain(&self) -> &Arc<MetalAllocationDomain> {
        &self.domain
    }

    /// The paired CPU engine, bound to the same domain.
    ///
    /// f64 and Complex64 work belongs here: it reads and writes the very tensors the Metal
    /// engine produced, with no transfer (design §2 requirement 2).
    pub fn cpu_engine(&self) -> &CpuEngine {
        &self.cpu
    }

    /// An engine over this context.
    pub fn engine(&self) -> MetalEngine {
        MetalEngine::new(self.clone())
    }

    /// Copy a host tensor into a fresh shared allocation.
    ///
    /// This is one of the two copies that exist; nothing else in the Metal path copies
    /// between host and device, and the byte count is recorded in [`Self::stats`].
    ///
    /// # Errors
    ///
    /// A typed `Unsupported` when the source is not an owned compact host tensor,
    /// `Validation` when its shape and data disagree, and the allocation or host-access
    /// failure otherwise.
    pub fn upload(&self, tensor: TensorRead<'_>) -> Result<Tensor, Error> {
        const OP: &str = "MetalContext::upload";
        let source = tensor
            .as_tensor()
            .ok_or_else(|| Error::unsupported(OP, "the source must be an owned host tensor"))?;
        if !source.is_host_memory() {
            return Err(Error::unsupported(
                OP,
                "the source is not a host tensor; a shared allocation needs no upload",
            ));
        }
        require_compact(source, OP)?;
        let destination = self.cell.allocate(OP, source.dtype(), source.shape())?;
        macro_rules! copy {
            ($t:ty) => {
                copy_host_to_shared::<$t>(source, &destination, OP)
            };
        }
        match source.dtype() {
            DType::F32 => copy!(f32),
            DType::F64 => copy!(f64),
            DType::I32 => copy!(i32),
            DType::I64 => copy!(i64),
            DType::Bool => copy!(bool),
            DType::C32 => copy!(num_complex::Complex32),
            DType::C64 => copy!(num_complex::Complex64),
            other => Err(Error::unsupported_dtype(
                OP,
                other,
                "the shared domain allocates the tensor dtypes only",
            )),
        }?;
        self.cell.note_upload(element_bytes(source));
        Ok(destination)
    }

    /// Copy a shared tensor into a fresh host tensor.
    ///
    /// Waits for the submissions that named the allocation, then copies; the byte count is
    /// recorded in [`Self::stats`].
    ///
    /// # Errors
    ///
    /// `Error::Unsupported` when the source is not an owned shared tensor or is not
    /// compact, `Error::UnsupportedDType` for a dtype the shared domain does not
    /// allocate, and `Error::RuntimeState` when another context owns the allocation or
    /// the host copy fails.
    pub fn download(&self, tensor: TensorRead<'_>) -> Result<Tensor, Error> {
        const OP: &str = "MetalContext::download";
        let source = tensor
            .as_tensor()
            .ok_or_else(|| Error::unsupported(OP, "the source must be an owned shared tensor"))?;
        check_own_domain(self, source, OP)?;
        require_compact(source, OP)?;
        macro_rules! copy {
            ($t:ty) => {
                copy_shared_to_host::<$t>(source, OP)
            };
        }
        let host = match source.dtype() {
            DType::F32 => copy!(f32),
            DType::F64 => copy!(f64),
            DType::I32 => copy!(i32),
            DType::I64 => copy!(i64),
            DType::Bool => copy!(bool),
            DType::C32 => copy!(num_complex::Complex32),
            DType::C64 => copy!(num_complex::Complex64),
            other => Err(Error::unsupported_dtype(
                OP,
                other,
                "the shared domain allocates the tensor dtypes only",
            )),
        }?;
        self.cell.note_download(element_bytes(source));
        Ok(host)
    }

    /// Wait for every submission this context has committed.
    ///
    /// # Errors
    ///
    /// The runtime's own failure, including a submission that completed with an error.
    pub fn synchronize(&self) -> Result<(), Error> {
        self.cell.with_runtime(|runtime| {
            runtime.synchronize().map_err(|error| {
                Error::from(MetalError::runtime("MetalContext::synchronize", error))
            })
        })
    }

    /// What the domain has done so far.
    pub fn stats(&self) -> MetalStats {
        let runtime = self.cell.lock();
        let runtime_stats = runtime.stats();
        MetalStats {
            allocations: self.cell.allocations.load(Ordering::Relaxed),
            live_bytes: self.cell.live_bytes.load(Ordering::Relaxed),
            uploads: self.cell.uploads.load(Ordering::Relaxed),
            upload_bytes: self.cell.upload_bytes.load(Ordering::Relaxed),
            downloads: self.cell.downloads.load(Ordering::Relaxed),
            download_bytes: self.cell.download_bytes.load(Ordering::Relaxed),
            launches: runtime_stats.launches,
            submissions: runtime_stats.submissions,
            waits: self.cell.waits.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn cell(&self) -> &Arc<RuntimeCell> {
        &self.cell
    }
}

/// Refuse an operand that is not compact.
pub(crate) fn require_compact(tensor: &Tensor, op: &'static str) -> Result<(), Error> {
    if tensor.is_col_major_contiguous()? {
        Ok(())
    } else {
        Err(Error::unsupported(
            op,
            "a non-compact shared operand would need a materialization kernel (U4 slice 2)",
        ))
    }
}

/// Bytes the tensor's elements need, for the transfer counters.
///
/// Every dtype counts here, including the ones only the paired CPU engine computes: the
/// counters report what moved, not what the GPU was willing to touch.
pub(crate) fn element_bytes(tensor: &Tensor) -> usize {
    let dtype = tensor.dtype();
    let components = if matches!(dtype, DType::C32 | DType::C64) {
        2
    } else {
        1
    };
    // `DType::Bool` declares width 0 (its kind carries the fact), but its storage element
    // is one byte, which is what the transfer counters moved.
    let component_bytes = match dtype {
        DType::Bool => std::mem::size_of::<bool>(),
        other => other.spec().width as usize / 8,
    };
    tensor
        .shape()
        .iter()
        .product::<usize>()
        .saturating_mul(component_bytes.saturating_mul(components))
}

/// Refuse a tensor that is not in this context's domain.
pub(crate) fn check_own_domain(
    context: &MetalContext,
    tensor: &Tensor,
    op: &'static str,
) -> Result<(), Error> {
    match allocation_domain_of(tensor) {
        Some(domain) if domain == context.allocation_domain() => Ok(()),
        Some(_) => Err(Error::runtime_state(
            op,
            "the tensor belongs to another Metal context's allocation domain",
        )),
        None => Err(Error::unsupported(
            op,
            "the tensor is not a shared allocation; upload it first",
        )),
    }
}

/// The domain of an owned backend tensor, by dtype.
fn allocation_domain_of(tensor: &Tensor) -> Option<AllocationDomainId> {
    macro_rules! domain {
        ($t:ty) => {
            tensor
                .as_typed::<$t>()
                .and_then(|typed| typed.backend_buffer()?.allocation_domain())
        };
    }
    match tensor.dtype() {
        DType::F32 => domain!(f32),
        DType::F64 => domain!(f64),
        DType::I32 => domain!(i32),
        DType::I64 => domain!(i64),
        DType::Bool => domain!(bool),
        DType::C32 => domain!(num_complex::Complex32),
        DType::C64 => domain!(num_complex::Complex64),
        _ => None,
    }
}

/// The Metal storage of an owned tensor with the same dtype as `source`.
fn storage_of_same_dtype<'a, T>(
    tensor: &'a Tensor,
    op: &'static str,
) -> Result<&'a MetalStorage<T>, Error>
where
    T: TensorScalar + 'static,
{
    tensor
        .as_typed::<T>()
        .and_then(|typed| typed.backend_buffer())
        .ok_or_else(|| Error::unsupported(op, "the destination is not a shared allocation"))?
        .as_any()
        .downcast_ref::<MetalStorage<T>>()
        .ok_or_else(|| Error::unsupported(op, "the destination is not a Metal allocation"))
}

/// Run `f` with the host elements of a compact host tensor.
fn host_slice<'a, T>(source: &'a Tensor, op: &'static str) -> Result<&'a [T], Error>
where
    T: TensorScalar + 'static,
{
    source
        .as_typed::<T>()
        .ok_or_else(|| Error::unsupported(op, "the source dtype changed"))?
        .host_data()
        .map_err(|error| Error::runtime_state(op, error.to_string()))
}

/// Copy one compact host tensor's elements into a fresh shared allocation.
fn copy_host_to_shared<T>(
    source: &Tensor,
    destination: &Tensor,
    op: &'static str,
) -> Result<(), Error>
where
    T: TensorScalar + 'static,
{
    let values = host_slice::<T>(source, op)?;
    let storage = storage_of_same_dtype::<T>(destination, op)?;
    storage.write_from_slice(values).map_err(|error| {
        Error::runtime_state(op, format!("writing the shared allocation failed: {error}"))
    })
}

/// Copy one shared allocation's elements into a fresh host tensor.
fn copy_shared_to_host<T>(source: &Tensor, op: &'static str) -> Result<Tensor, Error>
where
    T: TensorScalar + 'static,
{
    let storage = storage_of_same_dtype::<T>(source, op)?;
    let values = storage
        .host_slice()
        .map_err(|error| Error::runtime_state(op, format!("sharing the source failed: {error}")))?;
    TypedTensor::<T>::from_vec_col_major(source.shape().to_vec(), values.to_vec())
        .map(Tensor::from_typed)
}
