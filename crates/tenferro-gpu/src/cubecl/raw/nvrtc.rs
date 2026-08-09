//! NVRTC compilation surface (issue #1597).
//!
//! [`NvrtcOptions`] is the narrow, typed option set forwarded to the NVRTC
//! compiler through `cudarc`. [`compile_nvrtc`] compiles CUDA source on the
//! host and returns the resulting PTX image without touching the GPU.

/// Narrow, typed NVRTC compile options.
///
/// Only a conservative subset of NVRTC flags is exposed. The `arch` option is
/// forwarded as `--gpu-architecture=...` (e.g. `"compute_80"`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NvrtcOptions {
    /// Pass `--gpu-architecture=<arch>` when set (e.g. `Some("compute_80")`).
    pub arch: Option<String>,
    /// Pass `--std=<std>` when set (e.g. `Some("c++17")`).
    pub std: Option<String>,
    /// Extra raw flags forwarded verbatim.
    pub extra: Vec<String>,
}

impl NvrtcOptions {
    /// Build the cudarc equivalent without leaking: every flag funnels into
    /// the raw `options` vector (cudarc passes those to the compiler as-is).
    fn to_cudarc(&self) -> cudarc::nvrtc::CompileOptions {
        let mut options = cudarc::nvrtc::CompileOptions::default();
        if let Some(arch) = &self.arch {
            options.options.push(format!("--gpu-architecture={arch}"));
        }
        if let Some(std_flag) = &self.std {
            options.options.push(format!("--std={std_flag}"));
        }
        options.options.extend(self.extra.iter().cloned());
        options
    }
}

/// Compile CUDA source to PTX on the host using NVRTC.
///
/// # Errors
///
/// Returns the compiler's typed error via [`crate::Error::BackendSource`]
/// (with the NVRTC log) when compilation fails, or a validation error when the
/// source contains a NUL byte.
pub fn compile_nvrtc(src: &str, opts: &NvrtcOptions) -> crate::Result<cudarc::nvrtc::Ptx> {
    if src.as_bytes().contains(&0) {
        return Err(crate::Error::invalid_argument(
            "nvrtc.compile",
            "source",
            "CUDA source cannot contain NUL bytes",
        ));
    }
    cudarc::nvrtc::compile_ptx_with_opts(src, opts.to_cudarc())
        .map_err(|err| crate::Error::backend_source("nvrtc.compile", err))
}
