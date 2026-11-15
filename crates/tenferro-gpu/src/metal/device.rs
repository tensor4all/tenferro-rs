//! Metal device discovery: the probe, and what this host answered.
//!
//! Nothing is constructed before the probe answers. `t4a-metal-runtime::probe` loads
//! `libobjc` and `Metal.framework` through the system dynamic loader, checks that a
//! system default device exists, compiles one kernel per dtype and runs one small real
//! dispatch; a host without Metal gets a typed error, never a panic (design §6, "probe
//! before constructing"). `metal_devices()` follows the CUDA side's shape
//! ([`crate::cuda::cuda_devices`]): probe, then report.

/// Everything this host answered about its Apple GPU.
///
/// A field is what the probe measured, not a declaration: `f64` is `false` because this
/// host's MSL compiler rejected a `double` kernel, which is the measurement design §2
/// requirement 2 rests on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    /// The device's own name, such as `Apple M5 Max`.
    pub name: String,
    /// Whether the CPU and the GPU address one memory space, which is what makes the
    /// shared allocation domain transfer-free.
    pub unified_memory: bool,
    /// Largest single buffer the device accepts, in bytes.
    pub max_buffer_length: u64,
    /// The device's recommended working set, in bytes.
    pub recommended_max_working_set: u64,
    /// Whether the MSL compiler accepted an f32 kernel and the dispatch round-tripped.
    pub f32: bool,
    /// Whether the MSL compiler accepted a Complex32 (`float2`) kernel.
    pub c32: bool,
    /// Whether the MSL compiler accepted an fp16 kernel.
    pub fp16: bool,
    /// Whether the MSL compiler accepted a bf16 kernel.
    pub bf16: bool,
    /// Whether a `double` kernel compiled. Measured, not declared.
    pub f64: bool,
    /// The compiler's diagnostic when it rejected the `double` kernel.
    pub f64_diagnostic: Option<String>,
    /// Whether `MetalPerformanceShaders.framework` opened.
    pub mps_framework: bool,
    /// Whether `MetalPerformanceShadersGraph.framework` opened.
    pub mpsgraph_framework: bool,
    /// Whether `MPSMatrixMultiplication` is visible; U5's GEMM path needs it.
    pub mps_matrix_multiplication: bool,
    /// Whether `MPSGraph` is visible once its framework is open (the re-check
    /// tenferro-rs#2402 asked for).
    pub mpsgraph_class: bool,
}

/// Why device discovery listed nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetalDeviceError {
    /// This host has no usable Metal device. The detail is the runtime's own report of
    /// what failed to load or which dtype the compiler refused.
    Unavailable {
        /// What was probed, from the runtime.
        what: String,
        /// Why it failed, from the runtime.
        detail: String,
    },
}

impl std::fmt::Display for MetalDeviceError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalDeviceError::Unavailable { what, detail } => {
                write!(formatter, "no usable Metal device: {what}: {detail}")
            }
        }
    }
}

impl std::error::Error for MetalDeviceError {}

/// List the usable Apple GPUs.
///
/// Metal has one system default device per process, so the list is empty or holds one
/// entry. An absent device is a typed [`MetalDeviceError::Unavailable`] naming what
/// failed, so a multi-backend binary can report it.
///
/// # Errors
///
/// [`MetalDeviceError::Unavailable`] when the frameworks, a required symbol, a device or
/// a probe kernel is missing.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::metal::metal_devices;
///
/// match metal_devices() {
///     Ok(devices) => assert!(devices.len() <= 1),
///     Err(error) => assert!(!error.to_string().is_empty()),
/// }
/// ```
pub fn metal_devices() -> Result<Vec<MetalDeviceInfo>, MetalDeviceError> {
    // A non-Apple build compiles this module but has no device to list (design §2
    // requirement 6). The probe would report the same thing with a loader detail; this
    // short-circuit keeps the discovery result the same on a host whose loader has no
    // framework to look for at all.
    if !cfg!(target_vendor = "apple") {
        return Err(MetalDeviceError::Unavailable {
            what: "Metal.framework".to_string(),
            detail: "this target has no Apple GPU".to_string(),
        });
    }
    let capabilities = t4a_metal_runtime::probe().map_err(|error| match error {
        t4a_metal_runtime::Error::Unavailable { what, detail } => {
            MetalDeviceError::Unavailable { what, detail }
        }
        other => MetalDeviceError::Unavailable {
            what: "probe".to_string(),
            detail: other.to_string(),
        },
    })?;
    Ok(vec![MetalDeviceInfo {
        name: capabilities.device.name,
        unified_memory: capabilities.device.unified_memory,
        max_buffer_length: capabilities.device.max_buffer_length,
        recommended_max_working_set: capabilities.device.recommended_max_working_set,
        f32: true,
        c32: capabilities.c32,
        fp16: capabilities.fp16,
        bf16: capabilities.bf16,
        f64: capabilities.f64_supported,
        f64_diagnostic: capabilities.f64_diagnostic,
        mps_framework: capabilities.mps_framework,
        mpsgraph_framework: capabilities.mpsgraph_framework,
        mps_matrix_multiplication: capabilities.mps_matrix_multiplication,
        mpsgraph_class: capabilities.mpsgraph_class,
    }])
}

/// Whether any Metal device is usable, probed through the runtime.
///
/// The macOS lane sets `TENFERRO_REQUIRE_METAL=1`: there a missing device must fail rather
/// than skip, so a hardware lane cannot report success without hardware (the same rule
/// `crate::cuda::gpu_available` follows with `TENFERRO_REQUIRE_GPU`).
///
/// # Panics
///
/// Panics when `TENFERRO_REQUIRE_METAL` is set and no device is usable, which is how a
/// hardware lane refuses to report success without hardware.
///
/// # Examples
///
/// ```
/// use tenferro_gpu::metal::metal_available;
///
/// let available = metal_available();
/// assert_eq!(available, metal_available());
/// ```
pub fn metal_available() -> bool {
    let available = metal_devices().is_ok();
    assert!(
        available || std::env::var_os("TENFERRO_REQUIRE_METAL").is_none(),
        "TENFERRO_REQUIRE_METAL is set but no Metal device is usable"
    );
    available
}
