use std::path::PathBuf;

fn repo_file(path: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(root.join(path)).unwrap()
}

fn file_section<'a>(contents: &'a str, start: &str, end: &str) -> &'a str {
    let start_idx = contents
        .find(start)
        .unwrap_or_else(|| panic!("missing section start marker: {start}"));
    let end_idx = contents[start_idx..]
        .find(end)
        .map(|idx| start_idx + idx)
        .unwrap_or(contents.len());
    &contents[start_idx..end_idx]
}

// IMPORTANT: Do not delete or weaken these tests.
// They are the regression guard that keeps linalg on capability-based runtime
// checks instead of slipping back to CPU/backend-name special cases.

#[test]
fn capability_checked_composite_paths_do_not_require_cpu_type_checks() {
    let frules = repo_file("src/frules/mod.rs");
    let rrules = repo_file("src/rrules/mod.rs");

    assert!(
        !frules.contains("ensure_cpu_backend::<"),
        "forward rules should gate on capability-driven backend contracts rather than ensure_cpu_backend(...)"
    );
    assert!(
        !rrules.contains("ensure_cpu_backend::<"),
        "reverse rules should gate on capability-driven backend contracts rather than ensure_cpu_backend(...)"
    );
}

#[test]
fn cpu_only_kernel_paths_fail_through_capability_not_backend_name() {
    let primal = repo_file("src/primal/mod.rs");

    assert!(
        !primal.contains("TypeId::of::<C::Backend>()"),
        "primal linalg paths should not identify unsupported runtimes through direct backend type checks"
    );
}

#[test]
fn public_linalg_layers_do_not_spell_cpu_scalar_contracts() {
    let public_layers = [
        "src/lib.rs",
        "src/primal/mod.rs",
        "src/frules/mod.rs",
        "src/rrules/mod.rs",
        "src/ad_helpers/mod.rs",
        "src/backend/tensor_api.rs",
        "src/backend/tensor_context.rs",
    ]
    .into_iter()
    .map(repo_file)
    .collect::<Vec<_>>()
    .join("\n");

    assert!(
        !public_layers.contains("CpuLinalgScalar"),
        "public linalg layers should depend on backend-generic kernel scalar contracts rather than CPU-specific scalar names"
    );
}

#[test]
fn semiring_bridge_stays_context_driven_and_avoids_thread_local_cpu_state() {
    let prims_bridge = repo_file("src/prims_bridge.rs");

    assert!(
        !prims_bridge.contains("thread_local!"),
        "prims_bridge should use explicit semiring context dispatch instead of hiding a thread-local CpuContext"
    );
    assert!(
        !prims_bridge.contains("CpuContext::try_new("),
        "prims_bridge should not allocate ad hoc CpuContext state inside library helpers"
    );
    assert!(
        prims_bridge.contains("TensorSemiringContextFor"),
        "prims_bridge should route matmul helpers through the shared semiring context bridge"
    );
}

#[test]
fn public_and_ad_linalg_layers_do_not_fall_back_to_cpu_slice_helpers() {
    let layers = [
        "src/primal/mod.rs",
        "src/frules/mod.rs",
        "src/rrules/mod.rs",
        "src/ad_helpers/mod.rs",
        "src/prims_bridge.rs",
    ]
    .into_iter()
    .map(repo_file)
    .collect::<Vec<_>>()
    .join("\n");

    assert!(
        !layers.contains("backend::cpu::"),
        "public/composite linalg layers should stay generic over runtime contexts instead of calling CPU slice helpers directly"
    );
}

#[test]
fn solve_ex_and_inv_ex_sections_do_not_extract_cpu_slices() {
    let linear_systems = repo_file("src/primal/linear_systems.rs");
    let solve_ex = file_section(&linear_systems, "pub fn solve_ex", "pub fn inv");
    let inv_ex = file_section(&linear_systems, "pub fn inv_ex", "pub fn det");

    for (name, section) in [("solve_ex", solve_ex), ("inv_ex", inv_ex)] {
        assert!(
            !section.contains("extract_slice("),
            "{name} should stay tensor-native and avoid extract_slice(...)"
        );
        assert!(
            !section.contains("backend::slice_bridge::"),
            "{name} should stay tensor-native and avoid slice_bridge helpers"
        );
    }
}

#[test]
fn inv_ex_section_requires_inverse_capability() {
    let linear_systems = repo_file("src/primal/linear_systems.rs");
    let inv_ex = file_section(&linear_systems, "pub fn inv_ex", "pub fn det");

    assert!(
        inv_ex.contains("LinalgCapabilityOp::Inv"),
        "inv_ex should remain gated by inverse capability rather than inheriting solve_ex exposure"
    );
}

#[test]
fn cholesky_ex_section_does_not_extract_cpu_slices() {
    let least_squares = repo_file("src/primal/least_squares.rs");
    let cholesky_ex = file_section(&least_squares, "pub fn cholesky_ex", "#[cfg(test)]");

    assert!(
        !cholesky_ex.contains("extract_slice("),
        "cholesky_ex should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !cholesky_ex.contains("backend::slice_bridge::"),
        "cholesky_ex should stay tensor-native and avoid slice_bridge helpers"
    );
}

#[test]
fn lu_factor_ex_section_does_not_extract_cpu_slices() {
    let decompositions = repo_file("src/primal/decompositions.rs");
    let lu_factor_ex = file_section(&decompositions, "pub fn lu_factor_ex", "pub fn lu_solve");

    assert!(
        !lu_factor_ex.contains("extract_slice("),
        "lu_factor_ex should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !lu_factor_ex.contains("backend::slice_bridge::"),
        "lu_factor_ex should stay tensor-native and avoid slice_bridge helpers"
    );
}

#[test]
fn lu_factor_section_does_not_extract_cpu_slices() {
    let decompositions = repo_file("src/primal/decompositions.rs");
    let lu_factor = file_section(&decompositions, "pub fn lu_factor", "pub fn lu_factor_ex");

    assert!(
        !lu_factor.contains("extract_slice("),
        "lu_factor should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !lu_factor.contains("backend::slice_bridge::"),
        "lu_factor should stay tensor-native and avoid slice_bridge helpers"
    );
}

#[test]
fn svd_max_rank_path_uses_tensor_views_before_cutoff_logic() {
    let decompositions = repo_file("src/primal/decompositions.rs");
    let svd = file_section(&decompositions, "pub fn svd", "pub fn svdvals");

    assert!(
        svd.contains(".narrow("),
        "svd max_rank-only truncation should use Tensor::narrow(...) views"
    );
}

#[test]
fn svd_cutoff_path_stays_tensor_native() {
    let decompositions = repo_file("src/primal/decompositions.rs");
    let svd = file_section(&decompositions, "pub fn svd", "pub fn svdvals");

    assert!(
        !svd.contains("extract_slice("),
        "svd cutoff truncation should avoid extract_slice(...)"
    );
    assert!(
        svd.contains("zero_trailing_by_counts"),
        "svd cutoff truncation should route through zero_trailing_by_counts"
    );
}

#[test]
fn nuclear_norm_branch_stays_tensor_native_after_svdvals() {
    let norms = repo_file("src/primal/norms.rs");
    let nuclear = file_section(
        &norms,
        "        NormKind::Nuclear => {\n            let singular_values = svdvals(ctx, tensor)?;",
        "        NormKind::Spectral => {",
    );

    assert!(
        nuclear.contains("svdvals(ctx, tensor)?"),
        "nuclear norm should still derive from svdvals(...)"
    );
    assert!(
        !nuclear.contains("extract_slice("),
        "nuclear norm should avoid extract_slice(...) after svdvals"
    );
}

#[test]
fn spectral_norm_branch_stays_tensor_native_after_svdvals() {
    let norms = repo_file("src/primal/norms.rs");
    let spectral = file_section(
        &norms,
        "        NormKind::Spectral => {\n            let singular_values = svdvals(ctx, tensor)?;",
        "        NormKind::L1 => {",
    );

    assert!(
        spectral.contains("svdvals(ctx, tensor)?"),
        "spectral norm should still derive from svdvals(...)"
    );
    assert!(
        !spectral.contains("extract_slice("),
        "spectral norm should avoid extract_slice(...) after svdvals"
    );
}

#[test]
fn vector_l1_inf_norms_use_scalar_bridge() {
    let norms = repo_file("src/primal/norms.rs");
    let vector_match = file_section(
        &norms,
        "    if tensor.ndim() == 1 {",
        "        let input = ensure_col_major(tensor);",
    );

    assert!(
        vector_match.contains("scalar_unary_same_shape"),
        "vector L1/Inf norms should route through the scalar unary bridge"
    );
}

#[test]
fn matrix_l1_inf_norms_use_scalar_bridge() {
    let norms = repo_file("src/primal/norms.rs");
    let matrix_l1 = file_section(
        &norms,
        "        NormKind::L1 => {\n            if m == 0 || n == 0 {",
        "        NormKind::Inf => {",
    );
    let matrix_inf = file_section(
        &norms,
        "        NormKind::Inf => {\n            if m == 0 || n == 0 {",
        "        NormKind::Fro => {",
    );

    assert!(
        matrix_l1.contains("scalar_unary_same_shape"),
        "matrix L1 norm should route through the scalar unary bridge"
    );
    assert!(
        matrix_inf.contains("scalar_unary_same_shape"),
        "matrix Inf norm should route through the scalar unary bridge"
    );
}

#[test]
fn cond_path_multiplies_norms_tensor_natively() {
    let norms = repo_file("src/primal/norms.rs");
    let cond = file_section(&norms, "pub fn cond", "#[cfg(test)]");

    assert!(
        !cond.contains("extract_slice("),
        "cond should avoid extract_slice(...) when combining norm outputs"
    );
    assert!(
        cond.contains("scalar_binary_same_shape"),
        "cond should combine norm outputs through the scalar bridge"
    );
}

#[test]
fn det_path_uses_tensor_lu_and_prod_reduction() {
    let linear_systems = repo_file("src/primal/linear_systems.rs");
    let det = file_section(&linear_systems, "pub fn det", "pub fn slogdet");

    assert!(
        !det.contains("extract_slice("),
        "det should avoid extract_slice(...)"
    );
    assert!(
        !det.contains("backend::slice_bridge::"),
        "det should avoid slice_bridge helpers"
    );
    assert!(
        det.contains("ScalarReductionOp::Prod"),
        "det should derive its diagonal product through scalar reduction"
    );
}

#[test]
fn slogdet_path_uses_tensor_lu_and_log_without_slice_bridge() {
    let linear_systems = repo_file("src/primal/linear_systems.rs");
    let slogdet = file_section(&linear_systems, "pub fn slogdet", "#[cfg(test)]");

    assert!(
        slogdet.contains("TensorLinalgBackend<T>>::lu_factor(ctx, tensor)?;"),
        "slogdet should derive from backend tensor LU without packed host cleanup"
    );
    assert!(
        slogdet.contains("scalar_unary_same_shape"),
        "slogdet should use tensor-native unary helpers for abs/sign work"
    );
    assert!(
        slogdet.contains("AnalyticUnaryOp::Log"),
        "slogdet should use the analytic log substrate"
    );
    assert!(
        slogdet.contains("ScalarReductionOp::Sum"),
        "slogdet should reduce log singular values by summation"
    );
    assert!(
        !slogdet.contains("extract_slice("),
        "slogdet should avoid extract_slice(...)"
    );
    assert!(
        !slogdet.contains("backend::slice_bridge::"),
        "slogdet should avoid slice_bridge helpers"
    );
}
