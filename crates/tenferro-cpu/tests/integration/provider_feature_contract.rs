use std::path::{Path, PathBuf};

const PROVIDER_FEATURES: &[(&str, &[&str])] = &[
    (
        "blas-openblas",
        &["provider-src", "blas-src/openblas", "lapack-src/openblas"],
    ),
    (
        "blas-accelerate",
        &[
            "provider-src",
            "blas-src/accelerate",
            "lapack-src/accelerate",
        ],
    ),
    (
        "blas-mkl",
        &[
            "provider-src",
            "blas-src/intel-mkl-dynamic-parallel",
            "lapack-src/intel-mkl-dynamic-parallel",
        ],
    ),
];

const PASSTHROUGH_CRATES: &[&str] = &[
    "tenferro-runtime",
    "tenferro-ad",
    "tenferro-einsum",
    "tenferro-linalg",
    "tenferro-fft",
    "tenferro-gpu",
];

const PROVIDER_PASSTHROUGH_FEATURES: &[&str] = &["blas-openblas", "blas-accelerate", "blas-mkl"];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tenferro-cpu should live under crates/")
        .to_path_buf()
}

fn manifest(crate_name: &str) -> String {
    std::fs::read_to_string(
        workspace_root()
            .join("crates")
            .join(crate_name)
            .join("Cargo.toml"),
    )
    .unwrap_or_else(|err| panic!("{crate_name} manifest should be readable: {err}"))
}

fn source(path: &str) -> String {
    std::fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|err| panic!("{path} should be readable: {err}"))
}

fn feature_values(manifest: &str, feature: &str) -> String {
    let features_section = manifest
        .split_once("[features]")
        .expect("manifest should define [features]")
        .1
        .split_once("\n[")
        .map_or_else(
            || manifest.split_once("[features]").unwrap().1,
            |(section, _)| section,
        );
    let lines: Vec<_> = features_section.lines().collect();
    let prefix = format!("{feature} = ");

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with(&prefix) {
            let mut values = trimmed.to_owned();
            if trimmed.contains(']') {
                return values;
            }
            for continuation in &lines[index + 1..] {
                let trimmed = continuation.trim();
                values.push('\n');
                values.push_str(trimmed);
                if trimmed == "]" {
                    return values;
                }
            }
        }
    }

    panic!("missing feature `{feature}`");
}

#[test]
fn cpu_provider_features_select_matching_source_provider() {
    let manifest = manifest("tenferro-cpu");

    for (feature, required_values) in PROVIDER_FEATURES {
        let values = feature_values(&manifest, feature);
        for required in *required_values {
            assert!(
                values.contains(required),
                "`{feature}` should include `{required}`, got:\n{values}"
            );
        }
    }
}

#[test]
fn public_cpu_user_crates_expose_provider_passthrough_features() {
    for crate_name in PASSTHROUGH_CRATES {
        let manifest = manifest(crate_name);
        for feature in PROVIDER_PASSTHROUGH_FEATURES {
            let values = feature_values(&manifest, feature);
            assert!(
                values.contains(&format!("tenferro-cpu/{feature}")),
                "`{crate_name}` feature `{feature}` should forward to tenferro-cpu, got:\n{values}"
            );
        }
    }
}

#[test]
fn cpu_crate_rejects_ambiguous_explicit_provider_features() {
    let lib = source("crates/tenferro-cpu/src/lib.rs");

    for pair in [
        r#"all(feature = "blas-openblas", feature = "blas-accelerate")"#,
        r#"all(feature = "blas-openblas", feature = "blas-mkl")"#,
        r#"all(feature = "blas-accelerate", feature = "blas-mkl")"#,
    ] {
        assert!(
            lib.contains(pair),
            "missing provider conflict guard: {pair}"
        );
    }
    assert!(
        lib.contains("enable at most one explicit BLAS provider feature"),
        "missing explicit provider conflict diagnostic"
    );
    assert!(
        lib.contains("provider-inject cannot be combined with explicit BLAS provider features"),
        "provider-inject should not be combinable with source provider features"
    );
}

#[test]
fn tenferro_cpu_does_not_ship_tblis_provider_features_or_dependencies() {
    let manifest = manifest("tenferro-cpu");
    let lib = source("crates/tenferro-cpu/src/lib.rs");
    let backend = source("crates/tenferro-cpu/src/backend.rs");
    let provider = source("crates/tenferro-cpu/src/provider.rs");

    for forbidden in [
        "cpu-tblis",
        "cpu-tblis-runtime",
        "cpu-tblis-linked",
        "cpu-tblis-provider",
        "tblis-ffi",
        "tblis-src",
        "t4a-tblis-src",
    ] {
        assert!(
            !manifest.contains(forbidden),
            "tenferro-cpu manifest must not retain `{forbidden}` after TBLIS moves to ext/"
        );
    }
    assert!(
        !lib.contains("tblis_src") && !lib.contains("TBLIS"),
        "tenferro-cpu lib root must not link or gate an external TBLIS implementation"
    );
    assert!(
        !backend.contains("DotGeneralProvider")
            && !backend.contains("TblisGeneralContractionProvider"),
        "built-in backend shims must not expose a TBLIS-specific selector"
    );
    assert!(
        !provider.contains("TblisGeneralContractionProvider"),
        "TBLIS provider implementation must live outside tenferro-cpu"
    );
}

#[test]
fn provider_inject_call_through_is_owned_by_the_registered_integration_fixture() {
    let lib = source("crates/tenferro-cpu/src/lib.rs");
    let gemm_tests = source("crates/tenferro-cpu/src/gemm/tests.rs");
    let provider_tests = source("crates/tenferro-cpu/src/provider/tests.rs");
    let profile = source("scripts/ci/run_profile.py");

    assert!(
        lib.contains(r#"#[cfg(all(test, not(feature = "provider-inject")))]"#),
        "the broad unit suite must not call unregistered injected BLAS symbols"
    );
    assert!(
        lib.contains(r#"all(test, feature = "provider-inject")"#)
            && lib.contains("allow(dead_code, unused_imports)"),
        "the intentionally smaller provider-inject unit target must document its expected unused helpers"
    );
    for direct_test_source in [gemm_tests, provider_tests] {
        assert!(
            direct_test_source.contains(
                r#"#[cfg(all(feature = "cpu-blas", not(feature = "provider-inject")))]"#
            ),
            "direct BLAS unit tests must defer provider-inject call-through coverage to its fixture"
        );
    }
    assert!(
        profile
            .contains(r#"--features "cpu-blas,provider-inject" --test integration inject_tests"#),
        "the provider-inject CI profile must run the fixture that registers every FFI symbol"
    );
}

#[test]
fn provider_capabilities_require_wired_per_call_controls() {
    let provider = source("crates/tenferro-cpu/src/provider.rs");
    let capabilities = source("crates/tenferro-cpu/src/provider_capability.rs");

    assert_eq!(
        provider.matches("fn execution_capabilities(&self)").count(),
        6,
        "three provider traits and three built-in providers must classify execution explicitly",
    );
    assert!(
        capabilities.contains("thread_local_setter_wired")
            && capabilities.contains("process_global_set_restore_wired")
            && capabilities.contains("binary_thread_local_control_wired"),
        "probe fixtures must distinguish local controls from OpenBLAS global set-and-restore",
    );
    let openblas = capabilities
        .split_once("fn classify_openblas")
        .expect("OpenBLAS classification should be explicit")
        .1
        .split_once("fn classify_accelerate")
        .expect("Accelerate classification should follow OpenBLAS")
        .0;
    assert!(openblas.contains("uncontrolled_external_capabilities()"));
    assert!(
        !openblas.contains("PerCallUpperBound"),
        "OpenBLAS global set-and-restore must never claim per-call count control",
    );
    let builtin = capabilities
        .split_once("fn builtin_blas_execution_capabilities")
        .expect("built-in BLAS capability function should exist")
        .1
        .split_once("pub(crate) fn validate_provider_for_domain")
        .expect("domain validation should follow built-in classification")
        .0;
    assert!(
        builtin.contains("uncontrolled_external_capabilities()"),
        "until sound provider-specific controls are implemented, built-in BLAS must remain conservative",
    );
    assert!(
        capabilities.contains("domain_cpus == process_allowed_cpus"),
        "exact external-worker placement may use only the process-wide domain exception",
    );
}

#[test]
fn external_tblis_provider_example_owns_unpublished_source_build_path() {
    let root_manifest = source("Cargo.toml");
    let ext_manifest = source("ext/tenferro-cpu-tblis/Cargo.toml");
    let ext_lib = source("ext/tenferro-cpu-tblis/src/lib.rs");

    assert!(
        root_manifest.contains("exclude = [")
            && root_manifest.contains("\"third_party/t4a-tblis-src\""),
        "the unpublished source-build crate must stay outside the root workspace"
    );
    assert!(
        !root_manifest.contains("tblis-src = { package = \"t4a-tblis-src\""),
        "the root workspace must not resolve unpublished t4a-tblis-src for packaged crates"
    );
    assert!(
        ext_manifest.contains("publish = false"),
        "the external TBLIS provider example must not be published from this repository"
    );
    assert!(
        ext_manifest.contains("tblis-src = { package = \"t4a-tblis-src\"")
            && ext_manifest.contains("path = \"../../third_party/t4a-tblis-src\""),
        "source-build support must remain a local path dependency until t4a-tblis-src is published"
    );
    assert!(
        ext_manifest.contains("[workspace]"),
        "the external provider example must terminate parent workspace discovery"
    );
    assert!(
        ext_lib.contains("std::panic::catch_unwind")
            && !ext_lib.contains("std::panic::take_hook")
            && !ext_lib.contains("std::panic::set_hook"),
        "the temporary dynamic-loader bridge belongs in the external provider and must not mutate the global panic hook"
    );
}
