use std::path::{Path, PathBuf};

const PROVIDER_FEATURES: &[(&str, &[&str])] = &[
    (
        "cpu-tblis",
        &[
            "dep:tblis-ffi",
            "dep:tenferro-tblis-src",
            "tenferro-tblis-src/build_from_source",
            "tenferro-tblis-src/static",
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
        for (feature, _) in PROVIDER_FEATURES {
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
