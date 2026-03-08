use tenferro_device::{Error, Result};

use crate::cpu::CpuBackend;

#[cfg(feature = "cuda")]
use crate::cuda::CudaBackend;
#[cfg(not(feature = "cuda"))]
use crate::gpu_stubs::CudaBackend;

use crate::gpu_stubs::RocmBackend;

// ===========================================================================
// Backend Registry
// ===========================================================================

/// Registry of available compute backends.
///
/// **Current behavior:** Only the CPU backend is available.
/// [`load_cutensor`](BackendRegistry::load_cutensor) and
/// [`load_hiptensor`](BackendRegistry::load_hiptensor) always return
/// errors. GPU backend loading is not yet implemented.
///
/// When GPU support is implemented, this registry will hold the CPU
/// backend (always available) and optional GPU backends loaded at
/// runtime.
///
/// # Examples
///
/// ```ignore
/// // Aspirational API — GPU loading not yet functional.
/// use tenferro_prims::BackendRegistry;
///
/// let mut registry = BackendRegistry::new(); // CPU only
/// registry.load_cutensor("/usr/lib/libcutensor.so").unwrap();
/// assert!(registry.cuda().is_some());
/// ```
pub struct BackendRegistry {
    cpu: CpuBackend,
    cuda: Option<CudaBackend>,
    rocm: Option<RocmBackend>,
}

impl BackendRegistry {
    /// Create a registry with CPU backend only.
    pub fn new() -> Self {
        Self {
            cpu: CpuBackend,
            cuda: None,
            rocm: None,
        }
    }

    /// Load the cuTENSOR library from the given path.
    ///
    /// When the `cuda` feature is enabled, delegates to
    /// [`CudaBackend::load`] which initializes cudarc and populates
    /// the cuTENSOR vtable. Without the `cuda` feature, always returns
    /// `Err(DeviceError)`.
    ///
    /// The caller (Julia, Python, or standalone Rust) provides the path
    /// to the shared library. No auto-search.
    #[cfg(feature = "cuda")]
    pub fn load_cutensor(&mut self, path: &str) -> Result<()> {
        let (backend, _ctx) = CudaBackend::load(path)?;
        self.cuda = Some(backend);
        Ok(())
    }

    /// Load the cuTENSOR library from the given path.
    ///
    /// **Status: Not available.** The `cuda` feature is not enabled.
    /// Rebuild with `--features cuda` to enable cuTENSOR support.
    #[cfg(not(feature = "cuda"))]
    pub fn load_cutensor(&mut self, _path: &str) -> Result<()> {
        Err(Error::DeviceError(
            "cuTENSOR runtime loading not available: rebuild with --features cuda".into(),
        ))
    }

    /// Load the hipTENSOR library from the given path.
    ///
    /// **Status: Not yet implemented.** Always returns
    /// `Err(DeviceError)`.
    ///
    /// When implemented, the caller (Julia, Python, or standalone Rust)
    /// will provide the path to the shared library. No auto-search.
    pub fn load_hiptensor(&mut self, _path: &str) -> Result<()> {
        Err(Error::DeviceError(
            "hipTENSOR runtime loading not yet implemented".into(),
        ))
    }

    /// Returns a reference to the CPU backend.
    pub fn cpu(&self) -> &CpuBackend {
        &self.cpu
    }

    /// Returns a reference to the CUDA backend, if loaded.
    pub fn cuda(&self) -> Option<&CudaBackend> {
        self.cuda.as_ref()
    }

    /// Returns a reference to the ROCm backend, if loaded.
    pub fn rocm(&self) -> Option<&RocmBackend> {
        self.rocm.as_ref()
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
