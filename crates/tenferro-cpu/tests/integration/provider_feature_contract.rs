use std::path::{Path, PathBuf};

const PROVIDER_FEATURES: &[(&str, &[&str])] = &[
    ("cpu-tblis", &["cpu-tblis-runtime"]),
    (
        "cpu-tblis-runtime",
        &[
            "cpu-tblis-provider",
            "dep:tblis-ffi",
            "tblis-ffi/dynamic_loading",
        ],
    ),
    (
        "cpu-tblis-linked",
        &[
            "cpu-tblis-provider",
            "dep:tblis-ffi",
            "dep:tblis-src",
            "tblis-src/build_from_source",
            "tblis-src/static",
        ],
    ),
    (
        "blas-openblas",
        &[
            "provider-src",
            "dep:strided-einsum2",
            "blas-src/openblas",
            "lapack-src/openblas",
            "strided-einsum2/blas-openblas",
        ],
    ),
    (
        "blas-accelerate",
        &[
            "provider-src",
            "dep:strided-einsum2",
            "blas-src/accelerate",
            "lapack-src/accelerate",
            "strided-einsum2/blas-accelerate",
        ],
    ),
    (
        "blas-mkl",
        &[
            "provider-src",
            "dep:strided-einsum2",
            "blas-src/intel-mkl-dynamic-parallel",
            "lapack-src/intel-mkl-dynamic-parallel",
            "strided-einsum2/blas-mkl",
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

const PROVIDER_PASSTHROUGH_FEATURES: &[&str] = &[
    "cpu-tblis",
    "cpu-tblis-linked",
    "blas-openblas",
    "blas-accelerate",
    "blas-mkl",
];

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
fn cpu_provider_features_select_matching_source_and_einsum_provider() {
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
    assert!(
        lib.contains(r#"all(feature = "cpu-tblis-runtime", feature = "cpu-tblis-linked")"#),
        "cpu-tblis runtime and linked provider modes should be mutually exclusive"
    );
    assert!(
        lib.contains("enable at most one TBLIS provider mode"),
        "missing TBLIS provider conflict diagnostic"
    );
    assert!(
        lib.contains("cpu-tblis-provider is an internal marker"),
        "missing internal TBLIS marker diagnostic"
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
        7,
        "three provider traits and four built-in providers must classify execution explicitly",
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
fn tblis_runtime_panic_bridge_is_temporary_and_does_not_replace_the_global_hook() {
    let tblis = source("crates/tenferro-cpu/src/gemm/tblis_gemm.rs");

    assert!(
        tblis.contains("// INVARIANT: `tblis-ffi` 0.2.6 exposes only panic-based"),
        "the temporary panic bridge needs an auditable invariant marker"
    );
    assert!(
        tblis.contains("RESTGroup/tblis-rs/pull/4"),
        "the temporary panic bridge must name the upstream removal condition"
    );
    assert!(
        tblis.contains("std::panic::catch_unwind"),
        "the temporary bridge should catch only the upstream loader panic"
    );
    assert!(
        !tblis.contains("std::panic::take_hook") && !tblis.contains("std::panic::set_hook"),
        "TBLIS availability probing must not mutate the process-global panic hook"
    );
}

#[test]
fn tblis_source_provider_is_an_independent_excluded_package() {
    let root_manifest = source("Cargo.toml");
    let provider_manifest = source("third_party/t4a-tblis-src/Cargo.toml");
    let cpu_lib = source("crates/tenferro-cpu/src/lib.rs");

    assert!(
        root_manifest.contains(r#"exclude = ["third_party/t4a-tblis-src"]"#),
        "the independently released source provider must stay outside the tenferro workspace"
    );
    assert!(
        root_manifest.contains("tblis-src = { package = \"t4a-tblis-src\""),
        "tenferro should use a neutral dependency alias for the t4a package"
    );
    assert!(
        root_manifest.contains("path = \"third_party/t4a-tblis-src\"")
            && root_manifest.contains("version = \"=0.1.0\""),
        "local development must use the in-repository path with the exact registry version"
    );
    assert!(
        provider_manifest.contains("name = \"t4a-tblis-src\"")
            && provider_manifest.contains("version = \"0.1.0\"")
            && provider_manifest.contains("rust-version = \"1.78\"")
            && provider_manifest.contains("license = \"Apache-2.0\"")
            && provider_manifest.contains("links = \"tblis\""),
        "the source provider needs its independent package identity and native link owner"
    );
    assert!(
        provider_manifest.contains("documentation = \"https://docs.rs/t4a-tblis-src\"")
            && provider_manifest.contains("keywords = [")
            && provider_manifest.contains("categories = [")
            && provider_manifest.contains("\"licenses/**\""),
        "the independently published package needs complete registry metadata and notices"
    );
    assert!(
        !provider_manifest.contains(".workspace = true"),
        "an excluded package must not inherit tenferro workspace metadata"
    );
    assert!(
        provider_manifest.contains("[workspace]"),
        "the independently packaged provider must terminate parent workspace discovery"
    );
    assert!(
        cpu_lib.contains("extern crate tblis_src as _;"),
        "tenferro-cpu should link through the neutral dependency alias"
    );
}
