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
fn backend_surface_does_not_expose_internal_lu_solve_mode_type() {
    let source = crate_source("src/backend.rs");

    assert!(
        !source.contains("LuSolveMode"),
        "internal prepared-solve mode state should not be exposed as a backend public type"
    );
}
