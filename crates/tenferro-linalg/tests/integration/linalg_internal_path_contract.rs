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
fn norm_fro_and_p2_norm_use_fused_backend_hook_without_generic_pow() {
    let eager_source = crate_source("src/eager_composites.rs");
    let eager_frobenius = source_section(
        &eager_source,
        "fn frobenius_norm(abs: &EagerTensor",
        "fn p_norm(abs: &EagerTensor",
    );
    let eager_p_norm = source_section(
        &eager_source,
        "fn p_norm(abs: &EagerTensor",
        "fn default_pinv_rtol",
    );
    assert!(
        eager_frobenius.contains(".reduce_sum_squares(") && !eager_frobenius.contains(".pow("),
        "eager Frobenius norm should use the fused backend hook, not generic pow"
    );
    assert!(
        eager_p_norm.contains("p == 2.0") && eager_p_norm.contains("frobenius_norm(abs, axes)"),
        "eager p-norm should special-case p=2.0 through the Frobenius mul path"
    );

    let concrete_source = crate_source("src/tensor_ext.rs");
    let concrete_frobenius = source_section(
        &concrete_source,
        "fn frobenius_norm<B: LinalgBackend + ?Sized>",
        "fn p_norm<B: LinalgBackend + ?Sized>",
    );
    let concrete_p_norm = source_section(
        &concrete_source,
        "fn p_norm<B: LinalgBackend + ?Sized>",
        "fn count_nonzero<B: LinalgBackend>",
    );
    assert!(
        concrete_frobenius.contains("backend.reduce_sum_squares_read")
            && !concrete_frobenius.contains("backend.pow_read"),
        "concrete Frobenius norm should use the fused backend hook, not generic pow_read"
    );
    assert!(
        concrete_p_norm.contains("p == 2.0")
            && concrete_p_norm.contains("frobenius_norm(abs, axes, backend)"),
        "concrete p-norm should special-case p=2.0 through the Frobenius mul_read path"
    );

    let traced_source = crate_source("src/traced.rs");
    let traced_frobenius = source_section(
        &traced_source,
        "fn frobenius_norm(abs: &TracedTensor",
        "fn p_norm(abs: &TracedTensor",
    );
    let traced_p_norm = source_section(
        &traced_source,
        "fn p_norm(abs: &TracedTensor",
        "fn reduced_axes_have_zero_extent",
    );
    assert!(
        traced_frobenius.contains("reduce_sum_squares") && !traced_frobenius.contains(".pow("),
        "traced Frobenius norm should emit the fused core reduction primitive"
    );
    assert!(
        traced_p_norm.contains("p == 2.0") && traced_p_norm.contains("frobenius_norm(abs, axes)"),
        "traced p-norm should special-case p=2.0 through the Frobenius mul path"
    );
}

#[test]
fn real_sum_of_squares_norm_skips_abs_materialization_before_square() {
    let eager_source = crate_source("src/eager_composites.rs");
    let eager_norm = source_section(
        &eager_source,
        "pub(crate) fn norm",
        "fn scalar_real(anchor: &EagerTensor",
    );
    assert!(
        eager_norm.contains("can_square_without_abs(a.dtype(), axes.len(), ord)")
            && eager_norm.contains("frobenius_norm(a, &axes)"),
        "eager real Frobenius and p=2 norms should dispatch to the square path before abs"
    );

    let concrete_source = crate_source("src/tensor_ext.rs");
    let concrete_norm = source_section(
        &concrete_source,
        "fn norm_from_read<B: LinalgBackend + ?Sized>",
        "fn scalar_real(dtype: DType",
    );
    assert!(
        concrete_norm.contains("can_square_without_abs(input.dtype(), axes.len(), ord)")
            && concrete_norm.contains("frobenius_norm_read(input.clone(), &axes, backend)"),
        "concrete real Frobenius and p=2 norms should dispatch to the square path before abs"
    );

    let traced_source = crate_source("src/traced.rs");
    let traced_norm = source_section(&traced_source, "pub fn norm", "fn unexpected_output_count");
    assert!(
        traced_norm.contains("can_square_without_abs(a.dtype, axes.len(), ord)")
            && traced_norm.contains("frobenius_norm(a, &axes)"),
        "traced real Frobenius and p=2 norms should dispatch to the square path before abs"
    );

    for (label, source) in [
        ("eager", eager_source.as_str()),
        ("concrete", concrete_source.as_str()),
        ("traced", traced_source.as_str()),
    ] {
        let helper = source_section(source, "fn can_square_without_abs", "\n}\n\nfn");
        assert!(
            helper.contains("DType::F32 | DType::F64")
                && helper.contains("ord.is_none()")
                && helper.contains("ord == Some(2.0) && axes_len != 2"),
            "{label} helper should skip abs only for real Frobenius or non-matrix p=2 norms"
        );
    }
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
fn concrete_values_only_surfaces_use_backend_values_only_hooks() {
    let source = crate_source("src/tensor_ext.rs");

    // Owned SVD values: values-only hook, not compute-and-discard via svd.
    let owned_svdvals = source_section(
        &source,
        "fn svdvals(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
        "fn svd_with_options(",
    );
    assert!(
        owned_svdvals.contains("backend.svd_values(self)"),
        "svdvals should call the singular-values-only backend hook"
    );
    assert!(
        !owned_svdvals.contains("svd(backend)"),
        "svdvals should not compute singular vectors and discard them"
    );

    let read_svdvals = source_section(
        &source,
        "fn svdvals_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
        "fn svd_with_options_read(",
    );
    assert!(
        read_svdvals.contains("backend.svd_values_read(self)"),
        "svdvals_read should call the borrowed singular-values-only hook"
    );

    // Owned eigvalsh: values-only hook, not compute-and-discard via eigh.
    let owned_eigvalsh = source_section(
        &source,
        "fn eigvalsh(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
        "fn eigvals(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
    );
    assert!(
        owned_eigvalsh.contains("backend.eigh_values(self)"),
        "owned eigvalsh should call the Hermitian values-only backend hook"
    );
    assert!(
        !owned_eigvalsh.contains("eigh(backend)?.0"),
        "owned eigvalsh should not compute eigenvectors and discard them"
    );

    // Owned eigvals already uses the general values-only hook (kept).
    let owned_eigvals = source_section(
        &source,
        "fn eigvals(&self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
        "fn pinv(",
    );
    assert!(
        owned_eigvals.contains("backend.eig_values(self)"),
        "owned eigvals should call the general values-only backend hook"
    );
    assert!(
        !owned_eigvals.contains("eig(backend)?.0"),
        "owned eigvals should not compute eigenvectors and discard them"
    );

    // Read surfaces route directly through the borrowed values-only hook.
    let read_eigvalsh = source_section(
        &source,
        "fn eigvalsh_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
        "fn eigvals_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
    );
    assert!(
        read_eigvalsh.contains("eigh_values_read(self)"),
        "eigvalsh_read should call the borrowed Hermitian values-only hook"
    );
    assert!(
        !read_eigvalsh.contains("eigh_read(backend)?.0"),
        "eigvalsh_read should not compute eigenvectors and discard them"
    );

    let read_eigvals = source_section(
        &source,
        "fn eigvals_read(self, session: &mut dyn BackendSession) -> tenferro_tensor::Result<Tensor> {",
        "fn pinv_read",
    );
    assert!(
        read_eigvals.contains("to_contiguous_read") && read_eigvals.contains("eig_values("),
        "eigvals_read should materialize the read then call eig_values"
    );
    assert!(
        !read_eigvals.contains("eig_read(backend)?.0"),
        "eigvals_read should not compute eigenvectors and discard them"
    );
}

#[test]
fn eager_values_only_composites_emit_values_only_ops() {
    let source = crate_source("src/eager_composites.rs");
    let values_only = source_section(&source, "pub(crate) fn eigvalsh", "pub(crate) fn pinv");

    assert!(
        values_only.contains("LinalgOp::EighVals") && values_only.contains("LinalgOp::EigVals"),
        "eager eigvalsh/eigvals should emit the values-only EighVals/EigVals ops"
    );
    assert!(
        values_only.contains("one_output(") && values_only.contains("apply_linalg_eager("),
        "eager eigvalsh/eigvals should use the apply_linalg_eager + one_output pattern"
    );
    assert!(
        !values_only.contains("eigh(a)?.0") && !values_only.contains("eig(a)?.0"),
        "eager eigvalsh/eigvals should not compute eigenvectors and discard them"
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
        source.contains("fn svd_values_read") && source.contains("fn eigh_values_read"),
        "backend surface should include borrowed values-only hooks"
    );
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
fn semantic_linalg_linearize_does_not_replay_recorded_legacy_fragments() {
    let source = crate_source("src/ad/semantic.rs");
    let linearize_impl = source_section(
        &source,
        "impl SemanticLinearizeRule for LinalgAdRule",
        "impl SemanticLinearTransposeRule for LinalgAdRule",
    );

    assert!(
        linearize_impl.contains("SemanticRuleBuilder::with_seeds"),
        "semantic linearize should emit directly into SemanticProgramBuilder"
    );
    assert!(
        !linearize_impl.contains("RecordedBuilder::with_seed_count"),
        "semantic linearize must not replay recorded legacy fragments"
    );
}

#[test]
fn semantic_linalg_transpose_does_not_use_recorded_builder_family() {
    let source = crate_source("src/ad/semantic.rs");

    assert!(
        !source.contains("RecordedBuilder"),
        "semantic linalg transpose must not use the legacy RecordedBuilder family"
    );
    assert!(
        !source.contains("RecordedOperation"),
        "semantic linalg transpose must not store legacy RecordedOperation fragments"
    );
    assert!(
        !source.contains("RecordedTransposeContext"),
        "semantic linalg transpose must not depend on recorded-fragment transpose context"
    );
    assert!(
        source.contains("General linalg decomposition VJPs remain"),
        "remaining op-local linear fragment must document why linearize-then-transpose is retained"
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

#[test]
fn traced_lstsq_validates_once_without_symbolic_shape_probe_allocation() {
    let source = crate_source("src/traced.rs");
    let lstsq_source = source_section(
        &source,
        "pub fn lstsq",
        "/// Build a traced full-pivot LU solve op",
    );

    assert!(
        lstsq_source.contains("validate_lstsq"),
        "traced lstsq should use the shared validator"
    );
    assert_eq!(
        lstsq_source.matches("require_concrete_shape").count(),
        1,
        "traced lstsq should extract a concrete shape exactly once"
    );
    assert!(
        !lstsq_source.contains("try_concrete_shape") && !lstsq_source.contains("vec![0; a.rank]"),
        "symbolic lstsq validation must not probe or allocate a dummy shape"
    );

    let validation = crate_source("src/validation.rs");
    assert!(
        validation.contains("shape: impl FnOnce() -> Result<(usize, usize)>")
            && validation.contains("ensure_float_or_complex(op, dtype)?")
            && validation.contains("ensure_min_rank(op, a_rank, 2)?"),
        "shared lstsq validation must keep dtype/rank checks ahead of lazy shape extraction"
    );
}

#[test]
fn cpu_solve_read_into_reaches_direct_write_for_eligible_outputs() {
    // The public solve_read_into surface must dispatch eligible calls to the
    // direct in-place write (`solve_read_into_entered`) and reserve the
    // allocate-then-copy fallback (`solve_read_into_default`) for ineligible
    // destinations. `concrete_surface.rs` exercises the eligible runtime path
    // (contiguous host output, matching shape/dtype/placement), so this source
    // contract closes the gap: the direct path is provably taken, not just
    // value-identical.
    let source = crate_source("src/cpu/backend.rs");
    let solve_read_into = source_section(
        &source,
        "fn solve_read_into(",
        "fn solve_read_into_direct_eligible",
    );

    assert!(
        solve_read_into.contains("solve_read_into_entered(provider, context, buffers, a, b, out)"),
        "eligible solve_read_into should reach the direct in-place solver"
    );
    assert_eq!(
        solve_read_into.matches("solve_read_into_default").count(),
        1,
        "the allocate+copy fallback should appear exactly once, behind the eligibility guard"
    );
    let guarded_fallback = source_section(
        solve_read_into,
        "if has_zero_dim(a.shape())",
        "}\n\n        let a = a.tensor_view();",
    );
    assert!(
        guarded_fallback.contains("solve_read_into_default(self, a, b, out)"),
        "the allocate+copy fallback must be the guarded ineligible branch, not the main path"
    );
}

#[test]
fn cpu_solve_read_into_entered_writes_into_the_caller_buffer() {
    // The direct path must solve into the caller's destination buffer without
    // an allocate-then-copy round trip.
    let source = crate_source("src/cpu/backend.rs");
    let entered = source_section(
        &source,
        "fn solve_read_into_entered(",
        "fn tensor_write_view(out: TensorWrite<'_>) -> TensorViewMut<'_> {",
    );

    assert!(
        entered.contains("&mut out") && entered.contains("solve_into("),
        "the direct path should solve straight into the caller's out buffer"
    );
    assert!(
        !entered.contains("copy_read_into"),
        "the direct path should not allocate-then-copy"
    );
}
