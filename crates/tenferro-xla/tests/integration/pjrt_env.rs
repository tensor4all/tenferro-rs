#![cfg(feature = "pjrt")]

use std::ffi::OsString;

use tenferro_xla::{
    Error, PjrtPlugin, XlaExecutor, TENFERRO_PJRT_GPU_PLUGIN_ENV, TENFERRO_PJRT_PLUGIN_ENV,
};

fn with_env_var<T>(var: &'static str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _guard = super::pjrt_env_lock();
    let previous = std::env::var_os(var);
    match value {
        Some(value) => std::env::set_var(var, value),
        None => std::env::remove_var(var),
    }
    let result = f();
    match previous {
        Some(value) => std::env::set_var(var, value),
        None => std::env::remove_var(var),
    }
    result
}

#[test]
fn plugin_path_from_env_requires_explicit_env_var() {
    with_env_var(TENFERRO_PJRT_PLUGIN_ENV, None, || {
        let err = XlaExecutor::from_env_var(TENFERRO_PJRT_PLUGIN_ENV).unwrap_err();

        assert!(matches!(
            err,
            Error::MissingEnv {
                var: TENFERRO_PJRT_PLUGIN_ENV
            }
        ));
    });
}

#[test]
fn plugin_path_from_env_accepts_gpu_specific_env_var() {
    with_env_var(
        TENFERRO_PJRT_GPU_PLUGIN_ENV,
        Some("/definitely/missing/pjrt_gpu.so"),
        || {
            let err = XlaExecutor::from_env_var(TENFERRO_PJRT_GPU_PLUGIN_ENV).unwrap_err();

            assert!(matches!(
                err,
                Error::PluginLoad { path, .. }
                    if path == std::path::Path::new("/definitely/missing/pjrt_gpu.so")
            ));
        },
    );
}

#[test]
fn pjrt_plugin_reports_dynamic_load_errors() {
    let err = PjrtPlugin::load_path("/definitely/missing/pjrt.so").unwrap_err();

    assert!(matches!(err, Error::PluginLoad { .. }));
}

#[test]
fn xla_executor_from_env_loads_only_through_configured_path() {
    with_env_var(
        TENFERRO_PJRT_PLUGIN_ENV,
        Some("/definitely/missing/pjrt.so"),
        || {
            let err = XlaExecutor::from_env().unwrap_err();

            assert!(matches!(err, Error::PluginLoad { .. }));
        },
    );
}

#[test]
fn empty_plugin_env_var_is_treated_as_missing() {
    with_env_var(TENFERRO_PJRT_PLUGIN_ENV, Some(""), || {
        let err = XlaExecutor::from_env().unwrap_err();

        assert!(matches!(
            err,
            Error::MissingEnv {
                var: TENFERRO_PJRT_PLUGIN_ENV
            }
        ));
    });
}

#[test]
fn env_restore_handles_non_utf8_values() {
    let _guard = super::pjrt_env_lock();
    let var = "__TENFERRO_XLA_PJRT_NON_UTF8";
    let previous = std::env::var_os(var);
    std::env::set_var(var, OsString::from("/tmp/non_utf8_marker"));
    assert!(matches!(
        XlaExecutor::from_env_var(var).unwrap_err(),
        Error::PluginLoad { .. }
    ));
    match previous {
        Some(value) => std::env::set_var(var, value),
        None => std::env::remove_var(var),
    }
}
