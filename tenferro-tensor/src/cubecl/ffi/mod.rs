pub(crate) mod cusolver;
pub(crate) mod cutensor;

/// Build the library search path list for a CUDA FFI library.
///
/// If the environment variable `env_var` is set, its value is used as the
/// sole search path (supporting `:` separated lists). Otherwise, the
/// `default_paths` are used in order.
///
/// # Environment Variables
///
/// | Variable | Library |
/// |----------|---------|
/// | `TENFERRO_CUSOLVER_PATH` | cuSOLVER |
/// | `TENFERRO_CUBLAS_PATH` | cuBLAS |
/// | `TENFERRO_CUTENSOR_PATH` | cuTENSOR |
pub(crate) fn library_search_paths(env_var: &str, default_paths: &[&str]) -> Vec<String> {
    if let Ok(val) = std::env::var(env_var) {
        val.split(':')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    } else {
        default_paths.iter().map(|s| s.to_string()).collect()
    }
}
