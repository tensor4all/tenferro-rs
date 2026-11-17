//! The Metal backend: one engine over `t4a-metal-runtime` and the `t4a-cuda-kernels`
//! Metal source set (tensor4all/tenferro-rs#2402, design unit U4).
//!
//! # What this engine is
//!
//! One [`MetalContext`] owns one Metal device, one command queue and one **shared
//! allocation domain**. Every Metal tensor lives in that domain: its storage is a
//! `MTLResourceStorageModeShared` buffer, so the CPU and the GPU read and write the same
//! address and no Metal tensor is ever copied to or from the host for an operation. The
//! context also hands out a paired [`tenferro_cpu::CpuEngine`] bound to the same domain,
//! so f64 and Complex64 work runs on the CPU over the very same tensors
//! (`REPOSITORY_RULES.md`, "Device Transfer And Backend Buffer Errors").
//!
//! # What is served, and what refuses
//!
//! This is the first slice of U4, following the shape the CUDA engine's own first slice
//! used (`crate::cuda` `structural.rs`: the materialization boundary plus what the
//! catalogue serves, everything else a typed `Unsupported`). Served here:
//!
//! - probe-first discovery ([`metal_devices`], [`metal_available`]);
//! - the shared allocation domain ([`MetalAllocationDomain`]) and its host admission;
//! - explicit [`MetalContext::upload`] and [`MetalContext::download`], with byte
//!   accounting;
//! - the flat elementwise family the Metal catalogue carries for f32 and Complex32:
//!   `add`, `sub`, `mul`, `div` (f32), `neg`, `abs`, `sign`, `conj`, and the in-place form
//!   over a caller-owned output;
//! - a same-dtype `cast` (one exact copy).
//!
//! Refused with a typed error, never a fallback: **f64 and Complex64 operands of any
//! Metal operation** (the message names the paired `CpuEngine`), reductions, indexing,
//! contractions, the analytic family beyond the unary entries the catalogue carries,
//! `tril`/`triu`, the diagonal entries, cross-dtype casts, and non-compact operands. The
//! structural kernels, reductions through MPS, MPS GEMM and the MSL fusion emitter are
//! the following slices (design §10: U4 slices 2 and 3, then U5 and U6); each refusal
//! says which one serves it.
//!
//! # Synchronization
//!
//! One in-order command queue. Every submission records a `Completion` on each
//! allocation it named, **read or write**, and touching a shared allocation from the CPU
//! waits for the last one: waiting on the newest submission on an in-order queue is
//! exactly waiting for everything earlier on it. A CPU read therefore waits for pending
//! GPU writes and a CPU write also waits for pending GPU reads, as design §5 requires.
//!
//! Those completion waits run outside the context lock. `MetalEngine::submit` does hold
//! the lock across `Runtime::commit`, which can itself wait when the runtime is at its
//! in-flight bound; the lock therefore serialises back-pressure as well as encoding, and
//! it is not held across a host completion wait. A `commit` error is therefore handled
//! before the caller sees it: the engine synchronises the runtime, and if that cannot
//! prove the queue idle it poisons every allocation the call named, so a later host access
//! reports a backend failure instead of waiting on a stale completion.
//!
//! A GPU submission that names an allocation holding a live `BackendStorage::map_write`
//! or `BackendStorage::map_read` guard returns [`tenferro_tensor::Error`] with the
//! `MappedForHost` cause and commits nothing. The shared borrow that
//! `BackendStorage::shared_host_slice` hands out cannot raise that count, because its
//! signature returns no guard to drop; this slice removes the window structurally instead
//! — every GPU **write** targets an allocation created inside the same call and returned
//! only after the commit, so no caller can hold a borrow of it yet, and a GPU **read**
//! concurrent with a host borrow is harmless. Slice 2's in-place materialization kernel
//! has to revisit that.
//!
//! # No build-time linking
//!
//! Nothing here links Metal, MetalPerformanceShaders or Accelerate: the frameworks are
//! opened through `t4a-metal-sys` with `dlopen` on first use, and the MSL sources are
//! `&'static str` compiled by Metal at run time. `otool -L` on a binary that uses this
//! engine shows no such entry, which the macOS lane checks mechanically.
//!
//! # Examples
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tenferro_gpu::metal::{metal_available, MetalContext};
//!
//! if metal_available() {
//!     let context = MetalContext::open(Default::default())?;
//!     let engine = context.engine();
//!     assert_eq!(context.device().name.is_empty(), false);
//!     let _ = engine;
//! }
//! # Ok(())
//! # }
//! ```

mod capability;
mod context;
mod device;
mod domain;
mod engine;
mod pending;
mod storage;

pub use capability::metal_capabilities;
pub use context::{MetalConfig, MetalContext, MetalError, MetalStats};
pub use device::{metal_available, metal_devices, MetalDeviceError, MetalDeviceInfo};
pub use domain::MetalAllocationDomain;
pub use engine::MetalEngine;

/// The Metal backend's identity for the backend-neutral [`crate::gpu`] seam.
///
/// Its allocation is the `t4a-metal-runtime` buffer the context hands out; the seam
/// carries it with the domain and allocation identity, so no consumer reaches into tensor
/// storage for a Metal handle.
#[doc(hidden)]
pub struct MetalBackend;

impl crate::gpu::Backend for MetalBackend {
    type Allocation = std::sync::Arc<t4a_metal_runtime::Buffer>;
}

#[cfg(test)]
mod tests;
