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

fn file_section_after<'a>(contents: &'a str, anchor: &str, start: &str, end: &str) -> &'a str {
    let anchor_idx = contents
        .find(anchor)
        .unwrap_or_else(|| panic!("missing anchor marker: {anchor}"));
    let anchored = &contents[anchor_idx..];
    let start_idx = anchored
        .find(start)
        .map(|idx| anchor_idx + idx)
        .unwrap_or_else(|| panic!("missing section start marker after anchor: {start}"));
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
fn complex_real_bridge_stays_context_driven_and_avoids_host_extraction() {
    let prims_bridge = repo_file("src/prims_bridge.rs");
    let complex_real_unary = file_section(
        &prims_bridge,
        "pub(crate) fn complex_real_unary_same_shape",
        "pub(crate) fn analytic_unary_same_shape",
    );
    let complex_real_reduce = file_section(
        &prims_bridge,
        "pub(crate) fn complex_real_reduce_keep_axes",
        "#[cfg(test)]",
    );

    for (name, section) in [
        ("complex_real_unary_same_shape", complex_real_unary),
        ("complex_real_reduce_keep_axes", complex_real_reduce),
    ] {
        assert!(
            section.contains("TensorComplexRealContextFor"),
            "{name} should route through the shared complex-real context bridge"
        );
        assert!(
            !section.contains("extract_slice("),
            "{name} should stay tensor-native and avoid extract_slice(...)"
        );
        assert!(
            !section.contains("CpuContext::"),
            "{name} should not allocate ad hoc CpuContext state"
        );
    }
}

#[test]
fn complex_real_reduce_bridge_uses_family_level_reduction_descriptor() {
    let prims_bridge = repo_file("src/prims_bridge.rs");
    let complex_real_reduce = file_section(
        &prims_bridge,
        "pub(crate) fn complex_real_reduce_keep_axes",
        "#[cfg(test)]",
    );

    assert!(
        complex_real_reduce.contains("ComplexRealPrimsDescriptor::Reduction"),
        "complex_real_reduce_keep_axes should lower through a family-level complex-real reduction descriptor"
    );
    assert!(
        !complex_real_reduce.contains("scalar_reduce_keep_axes(ctx, &unary"),
        "complex_real_reduce_keep_axes should not expand reduction through a separate scalar_reduce_keep_axes(...) bridge"
    );
}

#[test]
fn scalar_where_bridge_stays_context_driven_and_avoids_host_extraction() {
    let prims_bridge = repo_file("src/prims_bridge.rs");
    let scalar_where = file_section(
        &prims_bridge,
        "pub(crate) fn scalar_where_same_shape",
        "pub(crate) fn scalar_unary_same_shape",
    );

    assert!(
        scalar_where.contains("TensorScalarContextFor"),
        "scalar_where_same_shape should route through the shared scalar context bridge"
    );
    assert!(
        scalar_where.contains("ScalarTernaryOp::Where"),
        "scalar_where_same_shape should lower through the ternary scalar family rather than open-coded branch logic"
    );
    assert!(
        !scalar_where.contains("extract_slice("),
        "scalar_where_same_shape should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !scalar_where.contains("CpuContext::"),
        "scalar_where_same_shape should not allocate ad hoc CpuContext state"
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
fn vander_stays_tensor_native_and_avoids_host_extraction() {
    let tensor_ops = repo_file("src/primal/tensor_ops.rs");
    let vander = file_section(&tensor_ops, "pub fn vander", "pub fn tensorinv");

    assert!(
        vander.contains("scalar_binary_same_shape_into("),
        "vander should be composed from tensor-native scalar primitives rather than host loops"
    );
    assert!(
        !vander.contains("extract_slice("),
        "vander should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !vander.contains("tensor_from_data("),
        "vander should stay tensor-native and avoid tensor_from_data(...)"
    );
    assert!(
        !vander.contains("require_linalg_support("),
        "vander should not gate tensor-native composition behind a backend linalg capability check"
    );
}

#[test]
fn cross_stays_tensor_native_and_avoids_host_extraction() {
    let tensor_ops = repo_file("src/primal/tensor_ops.rs");
    let cross = file_section(&tensor_ops, "pub fn cross", "pub fn householder_product");

    assert!(
        cross.contains("select(0,"),
        "cross should be composed from leading-axis tensor views rather than host extraction"
    );
    assert!(
        cross.contains("broadcast("),
        "cross should use tensor broadcast views rather than host-side expansion"
    );
    assert!(
        cross.contains("Tensor::stack("),
        "cross should reassemble the result with tensor stacking rather than tensor_from_data(...)"
    );
    assert!(
        !cross.contains("extract_slice("),
        "cross should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !cross.contains("tensor_from_data("),
        "cross should stay tensor-native and avoid tensor_from_data(...)"
    );
    assert!(
        !cross.contains("require_linalg_support("),
        "cross should not gate tensor-native composition behind a backend linalg capability check"
    );
}

#[test]
fn householder_product_stays_tensor_native_and_avoids_host_extraction() {
    let tensor_ops = repo_file("src/primal/tensor_ops.rs");
    let householder = file_section(&tensor_ops, "pub fn householder_product", "pub fn vander");

    assert!(
        householder.contains("batched_gemm_with_semiring_tensors("),
        "householder_product should use semiring GEMM substrate rather than host loops"
    );
    assert!(
        householder.contains("resolve_conj("),
        "householder_product should resolve lazy conjugation before adjoint GEMM operands"
    );
    assert!(
        householder.contains("Tensor::cat("),
        "householder_product should rebuild updated row blocks with tensor combine helpers"
    );
    assert!(
        !householder.contains("extract_slice("),
        "householder_product should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !householder.contains("tensor_from_data("),
        "householder_product should stay tensor-native and avoid tensor_from_data(...)"
    );
    assert!(
        !householder.contains("require_linalg_support("),
        "householder_product should not gate tensor-native composition behind backend linalg capability checks"
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
fn solve_triangular_section_does_not_extract_cpu_slices() {
    let norms = repo_file("src/primal/norms.rs");
    let solve_triangular = file_section(&norms, "pub fn solve_triangular", "pub fn norm");

    assert!(
        !solve_triangular.contains("extract_slice("),
        "solve_triangular should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !solve_triangular.contains("backend::slice_bridge::"),
        "solve_triangular should stay tensor-native and avoid slice_bridge helpers"
    );
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
fn eigen_section_requires_capability_and_avoids_host_hermitian_validation() {
    let decompositions = repo_file("src/primal/decompositions.rs");
    let eigen = file_section(&decompositions, "pub fn eigen", "#[cfg(test)]");
    let capability_gate =
        "require_linalg_support::<T, C>(backend::LinalgCapabilityOp::EigenSym, \"eigen\")?";

    assert!(
        eigen.contains(capability_gate),
        "eigen should gate through capability checks rather than direct backend-name or host-slice heuristics"
    );
    assert!(
        !eigen.contains("extract_slice("),
        "eigen should not extract host slices"
    );
    assert!(
        !eigen.contains("validate_hermitian_batches("),
        "eigen should not validate Hermitian structure internally"
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
fn lstsq_section_does_not_extract_cpu_slices() {
    let least_squares = repo_file("src/primal/least_squares.rs");
    let lstsq = file_section(&least_squares, "pub fn lstsq", "pub fn cholesky");

    assert!(
        !lstsq.contains("extract_slice("),
        "lstsq should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !lstsq.contains("backend::slice_bridge::"),
        "lstsq should stay tensor-native and avoid slice_bridge helpers"
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
fn pinv_path_stays_tensor_native() {
    let spectral = repo_file("src/primal/spectral.rs");
    let pinv = file_section(&spectral, "pub fn pinv", "pub fn matrix_exp");

    assert!(
        !pinv.contains("extract_slice("),
        "pinv should avoid extract_slice(...)"
    );
    assert!(
        !pinv.contains("backend::slice_bridge::"),
        "pinv should avoid slice_bridge helpers"
    );
    assert!(
        pinv.contains("complex_scale_same_shape"),
        "pinv should route complex scaling through the complex-scale bridge"
    );
    assert!(
        pinv.contains("resolve_conj(ctx"),
        "pinv should resolve lazy conjugation before semiring GEMM"
    );
}

#[test]
fn nuclear_norm_branch_stays_tensor_native_after_svdvals() {
    let norms = repo_file("src/primal/norms/norm_impl.rs");
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
    let norms = repo_file("src/primal/norms/norm_impl.rs");
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
    let norms = repo_file("src/primal/norms/norm_impl.rs");
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
    let norms = repo_file("src/primal/norms/norm_impl.rs");
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
fn fro_norm_path_is_tensor_native_and_uses_sqrt_bridge() {
    let norms = repo_file("src/primal/norms/norm_impl.rs");
    let vector_fro = file_section(
        &norms,
        "            NormKind::Fro => {",
        "            NormKind::Lp(_) => {}",
    );
    let matrix_fro = file_section_after(
        &norms,
        "    let (m, n, batch_dims) = validate_2d(tensor)?;",
        "        NormKind::Fro => {",
        "        _ => Err(",
    );

    assert!(
        !vector_fro.contains("extract_slice("),
        "vector Fro norm should avoid extract_slice(...)"
    );
    assert!(
        !matrix_fro.contains("extract_slice("),
        "matrix Fro norm should avoid extract_slice(...)"
    );
    assert!(
        vector_fro.contains("AnalyticUnaryOp::Sqrt"),
        "vector Fro norm should route through the analytic sqrt bridge"
    );
    assert!(
        matrix_fro.contains("AnalyticUnaryOp::Sqrt"),
        "matrix Fro norm should route through the analytic sqrt bridge"
    );
}

#[test]
fn vector_lp_norm_path_is_tensor_native_and_uses_pow_bridge() {
    let norms = repo_file("src/primal/norms/norm_impl.rs");
    let vector_lp = file_section(
        &norms,
        "        let NormKind::Lp(p) = kind else {",
        "    let (m, n, batch_dims) = validate_2d(tensor)?;",
    );

    assert!(
        !vector_lp.contains("extract_slice("),
        "vector Lp norm should avoid host slice extraction"
    );
    assert!(
        vector_lp.contains("analytic_binary_same_shape"),
        "vector Lp norm should route through the analytic binary Pow bridge"
    );
}

#[test]
fn matrix_power_path_is_tensor_native_and_uses_tensor_power_logic() {
    let matrix_functions = repo_file("src/primal/matrix_functions.rs");
    let matrix_power = file_section(&matrix_functions, "pub fn matrix_power", "#[cfg(test)]");

    assert!(
        !matrix_power.contains("require_main_memory_tensor(tensor, \"matrix_power\")?"),
        "matrix_power should not reject non-main-memory tensors before tensor-native power logic"
    );
    assert!(
        !matrix_power.contains("extract_slice("),
        "matrix_power should stay tensor-native and avoid extract_slice(...)"
    );
    assert!(
        !matrix_power.contains("backend::slice_bridge::"),
        "matrix_power should stay tensor-native and avoid slice_bridge helpers"
    );
    assert!(
        matrix_power.contains("batched_gemm_with_semiring_tensors"),
        "matrix_power should use tensor-native batched GEMM for exponentiation by squaring"
    );
    assert!(
        matrix_power.contains("batched_identity"),
        "matrix_power should build the identity tensor natively"
    );
    assert!(
        matrix_power.contains("inv(ctx, tensor)?"),
        "matrix_power should use tensor-native inverse logic for negative exponents"
    );
    assert!(
        matrix_power.contains("if exponent == 1"),
        "matrix_power should fast-path exponent == 1"
    );
    assert!(
        matrix_power.contains("if exponent == -1"),
        "matrix_power should fast-path exponent == -1"
    );
    assert!(
        matrix_power.contains("positive_exponent == 2"),
        "matrix_power should fast-path exponent == 2 after normalizing negatives"
    );
    assert!(
        matrix_power.contains("positive_exponent == 3"),
        "matrix_power should fast-path exponent == 3 after normalizing negatives"
    );
}

#[test]
fn matrix_exp_path_is_tensor_native_with_scalar_host_loop_bound_sync() {
    let spectral = repo_file("src/primal/spectral.rs");
    let matrix_exp = file_section(&spectral, "pub fn matrix_exp", "#[cfg(test)]");
    let matrix_exp_helper = repo_file("src/ad_helpers/matrix_exp.rs");
    let tensor_native_helper = file_section(
        &matrix_exp_helper,
        "pub(crate) fn matrix_exp_batch_1_norms_tensor",
        "pub(crate) fn matrix_1_norm",
    );

    assert!(
        !matrix_exp.contains("require_main_memory_tensor(tensor, \"matrix_exp\")?"),
        "matrix_exp should not reject non-main-memory tensors before tensor-native logic"
    );
    assert!(
        !matrix_exp.contains("extract_slice("),
        "matrix_exp should avoid extract_slice(...)"
    );
    assert!(
        !matrix_exp.contains("matrix_exp_single("),
        "matrix_exp should avoid the old host-slice helper"
    );
    assert!(
        !matrix_exp.contains("batch_norms.to_memory_space_async("),
        "matrix_exp should keep batch norms on-device"
    );
    assert!(
        matrix_exp.contains("s_max_tensor.to_memory_space_async("),
        "matrix_exp may sync only the reduced scalar loop bound to host"
    );
    assert!(
        matrix_exp.contains("matrix_exp_tensor_native"),
        "matrix_exp should hand off Padé evaluation to the tensor-native helper"
    );
    assert!(
        matrix_exp.contains("matrix_exp_batch_1_norms_tensor"),
        "matrix_exp should build batch-wise norms through tensor-native helper logic"
    );
    assert!(
        matrix_exp.contains("matrix_exp_batch_squaring_counts_tensor"),
        "matrix_exp should derive squaring counts through a tensor-native helper"
    );
    assert!(
        !matrix_exp.contains("control_tensor_from_host_values"),
        "matrix_exp should not rebuild control tensors from host vectors"
    );
    assert!(
        tensor_native_helper.contains("ScalarReductionOp::Sum"),
        "matrix_exp helper should build the batch-wise norm through tensor-native reductions"
    );
    assert!(
        tensor_native_helper.contains("ScalarReductionOp::Max"),
        "matrix_exp helper should reduce each batch independently"
    );
    assert!(
        tensor_native_helper.contains("blend_tensor_by_real_mask_same_shape"),
        "matrix_exp helper should blend squared batches through a reusable mask helper"
    );
    assert!(
        tensor_native_helper.contains("batched_gemm_with_semiring_tensors"),
        "matrix_exp helper should use tensor-native batched GEMM for Padé evaluation"
    );
    assert!(
        tensor_native_helper.contains("crate::solve(ctx, &lhs, &rhs)?"),
        "matrix_exp helper should use tensor-native solve for the Padé denominator"
    );
    assert!(
        !tensor_native_helper.contains("backend::slice_bridge::"),
        "matrix_exp helper should avoid slice_bridge helpers"
    );
    assert!(
        tensor_native_helper.contains("AnalyticUnaryOp::Ceil"),
        "matrix_exp helper should derive squaring counts through tensor-native rounding"
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
        !det.contains("backend_pivots_to_forward_perm("),
        "det should avoid host pivot reconstruction"
    );
    assert!(
        !det.contains("permutation_sign_from_forward_pivots("),
        "det should avoid host pivot sign reconstruction"
    );
    assert!(
        !det.contains("tensor_from_data("),
        "det should avoid materializing sign tensors from host data"
    );
    assert!(
        !det.contains("sign_data"),
        "det should avoid host sign reconstruction buffers"
    );
    assert!(
        det.contains("ScalarReductionOp::Prod"),
        "det should derive its diagonal product through scalar reduction"
    );
}

#[test]
fn slogdet_path_uses_tensor_lu_and_log_without_slice_bridge() {
    let linear_systems = repo_file("src/primal/linear_systems.rs");
    let slogdet = file_section(
        &linear_systems,
        "fn slogdet_real_impl",
        "fn slogdet_complex_impl",
    );

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
    assert!(
        !slogdet.contains("backend_pivots_to_forward_perm("),
        "slogdet should avoid host pivot reconstruction"
    );
    assert!(
        !slogdet.contains("permutation_sign_from_forward_pivots("),
        "slogdet should avoid host pivot sign reconstruction"
    );
    assert!(
        !slogdet.contains("tensor_from_data("),
        "slogdet should avoid materializing sign tensors from host data"
    );
    assert!(
        !slogdet.contains("sign_data"),
        "slogdet should avoid host sign reconstruction buffers"
    );
}

#[test]
fn complex_slogdet_path_uses_complex_real_and_complex_scale_without_slice_bridge() {
    let linear_systems = repo_file("src/primal/linear_systems.rs");
    let slogdet = file_section(
        &linear_systems,
        "fn slogdet_complex_impl",
        "impl<C> SlogdetDispatch<C> for f32",
    );

    assert!(
        slogdet.contains("complex_real_unary_same_shape"),
        "complex slogdet should derive abs(diag) through the complex-real unary bridge"
    );
    assert!(
        slogdet.contains("scalar_where_same_shape"),
        "complex slogdet should use tensor-native where to keep reciprocal(abs) safe"
    );
    assert!(
        slogdet.contains("complex_scale_same_shape"),
        "complex slogdet should scale phases and permutation signs through the complex-scale bridge"
    );
    assert!(
        !slogdet.contains("extract_slice("),
        "complex slogdet should avoid extract_slice(...)"
    );
    assert!(
        !slogdet.contains("backend::slice_bridge::"),
        "complex slogdet should avoid slice_bridge helpers"
    );
    assert!(
        !slogdet.contains("backend_pivots_to_forward_perm("),
        "complex slogdet should avoid host pivot reconstruction"
    );
    assert!(
        !slogdet.contains("permutation_sign_from_forward_pivots("),
        "complex slogdet should avoid host pivot sign reconstruction"
    );
    assert!(
        !slogdet.contains("tensor_from_data("),
        "complex slogdet should avoid materializing sign tensors from host data"
    );
    assert!(
        !slogdet.contains("sign_data"),
        "complex slogdet should avoid host sign reconstruction buffers"
    );
}

#[test]
fn complex_norm_branch_uses_complex_real_bridge_and_avoids_host_slicing() {
    let norm_impl = repo_file("src/primal/norms/norm_impl.rs");
    let complex_norm = file_section(
        &norm_impl,
        "fn norm_complex_impl",
        "macro_rules! impl_norm_primal_real",
    );

    assert!(
        complex_norm.contains("complex_real_unary_same_shape")
            || complex_norm.contains("complex_real_reduce_keep_axes"),
        "complex norm path should use complex-real bridge helpers"
    );
    assert!(
        !complex_norm.contains("extract_slice("),
        "norm should avoid extract_slice(...)"
    );
    assert!(
        !complex_norm.contains("backend::slice_bridge::"),
        "norm should avoid slice_bridge helpers"
    );
}
