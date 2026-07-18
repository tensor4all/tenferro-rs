use std::{fs, path::Path};

fn crate_source(path: &str) -> String {
    fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path))
        .unwrap_or_else(|err| panic!("crate source {path} should be readable: {err}"))
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
fn eager_extension_registration_preserves_typed_source_errors() {
    let source = crate_source("src/eager_ext.rs");
    let registration = source_section(
        &source,
        ".register_extension(register_runtime)",
        "apply_eager(Arc::new",
    );

    assert!(registration.contains("Error::runtime_state_source("));
    assert!(!registration.contains("Error::Internal"));
    assert!(!registration.contains("to_string()"));
}

#[test]
fn traced_solve_builds_factor_then_prepared_solve() {
    let source = crate_source("src/traced.rs");
    let solve_source = source_section(
        &source,
        "pub fn solve",
        "/// Build a traced full-pivot LU solve op",
    );

    assert!(
        solve_source.contains("LinalgOp::LuFactor"),
        "traced solve should emit an internal packed LU factor op"
    );
    assert!(
        solve_source.contains("LinalgOp::LuSolvePrepared"),
        "traced solve should emit an internal prepared LU solve op"
    );
    assert!(
        !solve_source.contains("LinalgOp::Solve"),
        "traced solve should not emit the legacy monolithic solve op"
    );
}

#[test]
fn traced_slogdet_consumes_packed_lu_instead_of_public_lu_outputs() {
    let source = crate_source("src/traced.rs");
    let slogdet_source = source_section(
        &source,
        "pub fn slogdet",
        "/// Build a traced determinant op",
    );

    assert!(
        slogdet_source.contains("LinalgOp::LuFactor"),
        "slogdet should use packed LU factors"
    );
    assert!(
        !slogdet_source.contains("lu(a)?"),
        "slogdet should not call public lu(), which materializes P/L/U outputs"
    );
}

#[test]
fn matrix_norm_uses_singular_values_only_path() {
    let source = crate_source("src/traced.rs");
    let matrix_norm_source = source_section(&source, "fn matrix_norm", "fn count_nonzero");

    assert!(
        matrix_norm_source.contains("svd_values("),
        "matrix norm ord=2/-2 should use a singular-values-only internal op"
    );
    assert!(
        !matrix_norm_source.contains("svd(&matrix)?.1"),
        "matrix norm should not materialize U/V when only singular values are needed"
    );
}

#[test]
fn traced_eigvalsh_uses_hermitian_values_only_path() {
    let source = crate_source("src/traced.rs");
    let eigvalsh_source = source_section(
        &source,
        "pub fn eigvalsh",
        "/// Build a traced general eigenvalue-only op",
    );

    assert!(
        eigvalsh_source.contains("eigh_values("),
        "eigvalsh should emit a Hermitian eigenvalues-only internal op"
    );
    assert!(
        !eigvalsh_source.contains("eigh(a)?.0"),
        "eigvalsh should not materialize eigenvectors and then discard them"
    );

    let extension_source = crate_source("src/extension.rs");
    assert!(
        extension_source.contains("LinalgOp::EighVals"),
        "linalg extension should define an internal EighVals op"
    );
    assert!(
        extension_source.contains("backend.eigh_values"),
        "EighVals execution should dispatch to a backend values-only hook"
    );
}

#[test]
fn traced_eigvals_uses_general_values_only_path() {
    let source = crate_source("src/traced.rs");
    let eigvals_source = source_section(
        &source,
        "pub fn eigvals",
        "/// Build a traced Moore-Penrose pseudoinverse op",
    );

    assert!(
        eigvals_source.contains("eig_values("),
        "eigvals should emit a general eigenvalues-only internal op"
    );
    assert!(
        !eigvals_source.contains("eig(a)?.0"),
        "eigvals should not materialize eigenvectors and then discard them"
    );

    let extension_source = crate_source("src/extension.rs");
    assert!(
        extension_source.contains("LinalgOp::EigVals"),
        "linalg extension should define an internal EigVals op"
    );
    assert!(
        extension_source.contains("backend.eig_values"),
        "EigVals execution should dispatch to a backend values-only hook"
    );
}

#[test]
fn traced_pinv_scales_svd_vectors_without_dense_diagonal_materialization() {
    let source = crate_source("src/traced.rs");
    let pinv_source = source_section(
        &source,
        "pub fn pinv_with_rtol",
        "/// Build a traced vector, matrix, or tensor norm op",
    );

    assert!(
        pinv_source.contains("scale_matrix_columns"),
        "pinv should broadcast singular-value reciprocals across V columns"
    );
    assert!(
        !pinv_source.contains("s_inv.embed_diag"),
        "pinv should not materialize a dense diagonal matrix for singular values"
    );
}

#[test]
fn cpu_backend_overrides_prepared_lu_and_values_only_paths() {
    let source = crate_source("src/cpu/backend.rs");
    let lu_factor_source = source_section(&source, "fn lu_factor", "fn full_piv_lu");

    assert!(
        !lu_factor_source.contains("self.lu(input)?"),
        "CPU lu_factor should factor directly instead of rebuilding from public LU outputs"
    );
    assert!(
        !lu_factor_source.contains("identity_pivots"),
        "CPU lu_factor should return real pivot metadata, not identity pivots"
    );
    for needle in [
        "fn lu_solve_prepared",
        "fn svd_values",
        "fn eigh_values",
        "fn eig_values",
    ] {
        assert!(
            source.contains(needle),
            "CPU backend should override {needle}"
        );
    }
}

#[test]
fn backend_surface_has_hidden_hermitian_values_only_hook() {
    let source = crate_source("src/backend.rs");

    assert!(
        source.contains("fn eigh_values"),
        "backend surface should include a hidden eigh_values hook"
    );
    assert!(
        source.contains(
            "backend {} does not implement internal Hermitian eigenvalues-only decomposition"
        ),
        "default eigh_values should fail explicitly instead of silently using full eigh"
    );
    assert!(
        source.contains("fn eig_values"),
        "backend surface should include a hidden eig_values hook"
    );
    assert!(
        source.contains(
            "backend {} does not implement internal general eigenvalues-only decomposition"
        ),
        "default eig_values should fail explicitly instead of silently using full eig"
    );
}

#[test]
fn cpu_general_eig_values_only_paths_do_not_request_vectors() {
    let lapack_source = crate_source("src/cpu/linalg/lapack_linalg/eig.rs");
    let lapack_real_values = source_section(
        &lapack_source,
        "macro_rules! impl_eig_values_real_2d",
        "macro_rules! impl_eig_complex_2d",
    );
    let lapack_complex_values = source_section(
        &lapack_source,
        "macro_rules! impl_eig_values_complex_2d",
        "impl_real_eig_to_complex_outputs!",
    );

    for section in [lapack_real_values, lapack_complex_values] {
        assert!(
            section.contains("b'N'"),
            "LAPACK eig_values should request no left or right eigenvectors"
        );
        assert!(
            !section.contains("b'V'"),
            "LAPACK eig_values should not request eigenvectors"
        );
    }

    let faer_source = crate_source("src/cpu/linalg/faer_linalg.rs");
    let faer_real_values = source_section(
        &faer_source,
        "macro_rules! impl_eig_values_real_2d",
        "macro_rules! impl_eig_complex_2d",
    );
    let faer_complex_values = source_section(
        &faer_source,
        "macro_rules! impl_eig_values_complex_2d",
        "impl_eig_real_2d!",
    );

    for section in [faer_real_values, faer_complex_values] {
        assert!(
            section.contains("ComputeEigenvectors::No"),
            "Faer eig_values should request no eigenvectors"
        );
        assert!(
            !section.contains("ComputeEigenvectors::Yes"),
            "Faer eig_values should not request eigenvectors"
        );
    }
}

#[test]
fn linalg_ad_has_prepared_solve_rules() {
    let source = crate_source("src/ad/rules/solve.rs");

    assert!(
        source.contains("linearize_lu_solve_prepared"),
        "linalg AD should define a linearize rule for prepared LU solve"
    );
    assert!(
        source.contains("transpose_lu_solve_prepared"),
        "linalg AD should define a transpose rule for prepared LU solve"
    );
}

#[test]
fn prepared_solve_transpose_rule_keeps_matrix_cotangent_path() {
    let source = crate_source("src/ad/rules/solve.rs");
    let transpose_rule = source_section(
        &source,
        "pub(crate) fn transpose_lu_solve_prepared",
        "pub(crate) fn transpose_full_piv_lu_solve",
    );

    assert!(
        transpose_rule.contains("ADRuleResult<Vec<Option<LocalValueId>>>"),
        "prepared LU solve transpose rule must be able to report unsupported active inputs"
    );
    assert!(
        transpose_rule.contains("active_mask[0]"),
        "prepared LU solve transpose rule must inspect matrix A activity"
    );
    assert!(
        transpose_rule.contains("solve_matrix_cotangent"),
        "prepared LU solve transpose rule must build the matrix A cotangent"
    );
}

#[test]
fn backend_surface_does_not_expose_internal_lu_solve_mode_type() {
    let source = crate_source("src/backend.rs");

    assert!(
        !source.contains("LuSolveMode"),
        "internal prepared-solve mode state should not be exposed as a backend public type"
    );
}

#[test]
fn faer_batched_paths_reuse_pooled_scratch_inputs() {
    let source = crate_source("src/cpu/linalg/faer_linalg.rs");

    assert!(
        source.contains("tensor_from_pooled_slice_with_template"),
        "Faer batched paths should construct batch inputs from pooled scratch buffers"
    );
    assert!(
        source.contains("refill_tensor_from_slice"),
        "Faer batched paths should refill scratch tensors instead of reallocating per batch"
    );
    assert!(
        !source.contains("host_data()[range].to_vec()"),
        "Faer batched paths should not allocate a new Vec for every batch slice"
    );
}
