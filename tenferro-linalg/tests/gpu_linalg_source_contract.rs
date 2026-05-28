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

fn gpu_mod_source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("gpu")
            .join("mod.rs"),
    )
    .unwrap_or_else(|err| panic!("GPU linalg module source should be readable: {err}"))
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tenferro-linalg should live inside the workspace")
}

fn read_workspace_source(path: &str) -> String {
    fs::read_to_string(workspace_root().join(path))
        .unwrap_or_else(|err| panic!("workspace source {path} should be readable: {err}"))
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

fn assert_before(section: &str, earlier: &str, later: &str) {
    let earlier_idx = section
        .find(earlier)
        .unwrap_or_else(|| panic!("source section should contain {earlier:?}"));
    let later_idx = section
        .find(later)
        .unwrap_or_else(|| panic!("source section should contain {later:?}"));
    assert!(
        earlier_idx < later_idx,
        "{earlier:?} should appear before {later:?}"
    );
}

#[test]
fn tenferro_gpu_no_longer_owns_linalg_specific_ffi_or_kernels() {
    for path in [
        "tenferro-gpu/src/cubecl/ffi/cusolver.rs",
        "tenferro-gpu/src/kernels/linalg.rs",
    ] {
        assert!(
            !workspace_root().join(path).exists(),
            "{path} should be owned by tenferro-linalg, not tenferro-gpu"
        );
    }

    let cubecl_mod = read_workspace_source("tenferro-gpu/src/cubecl/mod.rs");
    for needle in ["CudaLinalgHandles", "linalg_handles", "cusolver", "cublas"] {
        assert!(
            !cubecl_mod.contains(needle),
            "CubeclBackend should not expose linalg-specific state: found {needle}"
        );
    }
}

#[test]
fn linalg_ad_rules_use_internal_ops_conjugation_helpers() {
    let source = read_workspace_source("tenferro-linalg/src/ad/rules/mod.rs");
    for needle in [
        "fn is_real_dtype",
        "fn conjugate_primal_if_dtype_complex",
        "fn conjugate_linear_if_dtype_complex",
    ] {
        assert!(
            !source.contains(needle),
            "linalg AD should import canonical helper instead of redefining {needle}"
        );
    }
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

#[test]
fn cubecl_linalg_overrides_svd_view_with_backend_canonicalization() {
    let source = gpu_mod_source();
    let svd_view_source = source_section(&source, "fn svd_view", "fn qr");

    for needle in [
        "self.to_contiguous(&view)?",
        "let input = Tensor::F64(compact);",
        "self.svd(&input)",
    ] {
        assert!(
            svd_view_source.contains(needle),
            "CubeCL svd_view should canonicalize borrowed GPU views on the backend: missing {needle}"
        );
    }
}

#[test]
fn gpu_linalg_zero_dim_fast_paths_validate_residency_before_allocating_outputs() {
    let source = linalg_source();

    for (start, end, residency_check) in [
        (
            "fn cholesky_typed",
            "fn triangular_solve_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn lu_typed",
            "fn svd_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn svd_typed",
            "fn qr_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn qr_typed",
            "fn eigh_typed",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
        (
            "fn eigh_typed",
            "fn validate_nonsingular_gpu",
            "ensure_cubecl_resident_typed(OP, input)?;",
        ),
    ] {
        let section = source_section(&source, start, end);
        assert_before(section, residency_check, "if has_zero_dim");
    }

    let triangular = source_section(&source, "fn triangular_solve_typed", "fn lu_typed");
    assert_before(
        triangular,
        "ensure_cubecl_resident_typed(OP, a)?;",
        "if has_zero_dim",
    );
    assert_before(
        triangular,
        "ensure_cubecl_resident_typed(OP, b)?;",
        "if has_zero_dim",
    );

    let solve = source_section(&source, "pub(super) fn solve", "fn cholesky_typed");
    assert_before(
        solve,
        "ensure_supported_linalg_pair(OP, a, b)?;",
        "if has_zero_dim",
    );
    assert_before(
        solve,
        "ensure_cubecl_resident_tensor(OP, a)?;",
        "if has_zero_dim",
    );
    assert_before(
        solve,
        "ensure_cubecl_resident_tensor(OP, b)?;",
        "if has_zero_dim",
    );
    assert!(
        solve.contains("zero_like_linalg_device_tensor(backend.runtime(), b, OP)"),
        "GPU solve zero-dim fast path should allocate the result on the GPU"
    );
    assert!(
        !source.contains("fn zeros_like_tensor"),
        "GPU linalg should not build host zero tensors for device fast paths"
    );
}
