use std::{fs, path::Path};

fn cubecl_source(file: &str) -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("cubecl")
            .join(file),
    )
    .unwrap_or_else(|err| panic!("CubeCL source {file} should be readable: {err}"))
}

fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_idx = source
        .find(start)
        .unwrap_or_else(|| panic!("source should contain section start {start:?}"));
    let remaining = &source[start_idx..];
    let end_idx = remaining
        .find(end)
        .map(|offset| start_idx + offset)
        .unwrap_or(source.len());
    &source[start_idx..end_idx]
}

#[test]
fn cubecl_scatter_does_not_use_single_thread_launch_fallback() {
    let mod_source = cubecl_source("mod.rs");
    let scatter_source = source_section(&mod_source, "    fn scatter(", "    fn slice(");
    let dispatch_source = cubecl_source("dispatch.rs");
    let sources = [
        ("cubecl/mod.rs scatter body", scatter_source),
        ("cubecl/dispatch.rs", dispatch_source.as_str()),
    ];
    let banned = ["single_thread_launch_config", "CubeCount::new_single()"];

    let mut violations = Vec::new();
    for (name, source) in sources {
        for needle in banned {
            if source.contains(needle) {
                violations.push(format!("{name} contains {needle}"));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL scatter launch must not use a single-thread fallback:\n{}",
        violations.join("\n")
    );
}

#[test]
fn cubecl_i64_index_conversion_does_not_roundtrip_through_host() {
    let mod_source = cubecl_source("mod.rs");
    let banned = [
        "fn i64_indices_as_f64",
        "download_tensor(self.runtime(), &Tensor::I64",
        "upload_tensor(self.runtime(), &converted",
    ];

    let mut violations = Vec::new();
    for needle in banned {
        if mod_source.contains(needle) {
            violations.push(format!("cubecl/mod.rs contains {needle}"));
        }
    }

    assert!(
        violations.is_empty(),
        "CubeCL I64 index conversion must stay on device; host roundtrips in indexing paths are performance regressions:\n{}",
        violations.join("\n")
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cubecl_runtime_exposes_explicit_synchronize() {
    let _sync: fn(&tenferro_gpu::CubeclRuntime) -> tenferro_tensor::Result<()> =
        tenferro_gpu::CubeclRuntime::synchronize;
}
