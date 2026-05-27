use std::{fs, path::Path};

fn linalg_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("gpu")
            .join("linalg.rs"),
    )
    .unwrap_or_else(|err| panic!("GPU linalg source should be readable: {err}"))
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
fn gpu_lu_outputs_are_not_rebuilt_by_host_roundtrip() {
    let source = linalg_source();
    let lu_source = source_section(&source, "fn lu_typed", "fn svd_typed");
    let banned = [
        "download_device_tensor(backend.runtime(), &work, OP)",
        "build_lu_outputs_host(&host_lu",
        "upload_host_tensor(backend.runtime(), p)",
        "upload_host_tensor(backend.runtime(), l)",
        "upload_host_tensor(backend.runtime(), u)",
        "upload_host_tensor(backend.runtime(), parity)",
    ];

    let mut violations = Vec::new();
    for needle in banned {
        if lu_source.contains(needle) {
            violations.push(format!("gpu/linalg.rs lu_typed contains {needle}"));
        }
    }

    assert!(
        violations.is_empty(),
        "GPU LU must not rebuild P/L/U/parity through a full device-to-host-to-device roundtrip:\n{}",
        violations.join("\n")
    );
}
