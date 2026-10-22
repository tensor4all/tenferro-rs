//! Prepare a readback before reserving its host destination (#2247).

use cubecl_runtime::{server::Handle, storage::ManagedResource};

use super::CudaRuntimeState;

/// A resolved source whose allocation is retained until the copy completes.
pub(crate) struct PreparedHostDownload<'a> {
    runtime: &'a CudaRuntimeState,
    resource: ManagedResource<cubecl_cuda::ffi_interop::GpuResource>,
    len: usize,
    op: &'static str,
}

impl<'a> PreparedHostDownload<'a> {
    pub(super) fn new(
        runtime: &'a CudaRuntimeState,
        handle: Handle,
        len: usize,
        op: &'static str,
    ) -> crate::Result<Self> {
        // Preserve the existing stream ordering: flush queued launches, then
        // resolve the source and its cross-stream dependencies. A refused
        // allocation must surface here, before a host destination is reserved.
        runtime.flush_cubecl(op)?;
        let resource = runtime
            .client
            .get_resource::<cubecl_cuda::ffi_interop::CudaServer>(handle)
            .map_err(|err| crate::Error::backend_source(op, err))?;
        let available = usize::try_from(resource.resource().size).unwrap_or(usize::MAX);
        if available < len {
            return Err(crate::Error::Internal(format!(
                "{op}: download of {len} bytes exceeds the {available}-byte allocation"
            )));
        }
        Ok(Self {
            runtime,
            resource,
            len,
            op,
        })
    }

    /// Copy directly into the final host destination and wait for this stream.
    ///
    /// # Safety
    ///
    /// `dst` must be valid for writes of the prepared byte length until this
    /// returns. On error it must not be freed or reused: the device may still
    /// be writing it. The caller must leak the destination in that case.
    pub(crate) unsafe fn copy_into(self, dst: *mut u8) -> crate::Result<()> {
        let Self {
            runtime,
            resource,
            len,
            op,
        } = self;
        runtime.set_current_cuda_context(op)?;
        let stream = runtime.raw_cuda_stream()? as usize as cudarc::driver::sys::CUstream;
        let src = resource.resource().ptr;
        // SAFETY: preparation validated `len` against the resolved allocation,
        // which `resource` retains. The caller provides the live destination;
        // get_resource ordered source writes on the current CubeCL stream.
        let completed = unsafe {
            cudarc::driver::sys::cuMemcpyDtoHAsync_v2(dst.cast(), src, len, stream).result()
        }
        // SAFETY: `stream` is the current runtime's live CUDA stream. Waiting
        // for it proves the destination is initialized before it is published.
        .and_then(|()| unsafe { cudarc::driver::result::stream::synchronize(stream) });
        if let Err(err) = completed {
            std::mem::forget(resource);
            return Err(crate::Error::backend_source(op, err));
        }
        Ok(())
    }
}
