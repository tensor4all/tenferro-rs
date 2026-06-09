#![cfg(feature = "autodiff")]

use tenferro_linalg::{
    all_linalg_ad_support, linalg_ad_support, LinalgAdOpKind, LinalgAdRuleSupport,
};

#[test]
fn linalg_ad_support_manifest_covers_all_dispatch_arms_in_order() {
    let expected = [
        LinalgAdOpKind::Cholesky,
        LinalgAdOpKind::Lu,
        LinalgAdOpKind::LuFactor,
        LinalgAdOpKind::LuSolvePrepared,
        LinalgAdOpKind::FullPivLu,
        LinalgAdOpKind::FullPivLuSolve,
        LinalgAdOpKind::Svd,
        LinalgAdOpKind::SvdVals,
        LinalgAdOpKind::Qr,
        LinalgAdOpKind::Eigh,
        LinalgAdOpKind::EighVals,
        LinalgAdOpKind::Eig,
        LinalgAdOpKind::TriangularSolve,
    ];

    let manifest = all_linalg_ad_support();
    assert_eq!(manifest.len(), LinalgAdOpKind::COUNT);
    assert_eq!(manifest.len(), expected.len());

    for (index, &kind) in expected.iter().enumerate() {
        let entry = linalg_ad_support(kind);
        assert_eq!(entry.kind, kind);
        assert_eq!(manifest[index], *entry);
        for (output_index, output) in entry.outputs.iter().enumerate() {
            assert_eq!(output.index, output_index);
        }
    }
}

#[test]
fn linalg_ad_support_manifest_marks_partial_decomposition_outputs() {
    let lu = linalg_ad_support(LinalgAdOpKind::Lu);
    assert_eq!(lu.linearize, LinalgAdRuleSupport::PartiallySupported);
    assert_output_status(lu, "p", LinalgAdRuleSupport::NonDifferentiable);
    assert_output_status(lu, "l", LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(lu, "u", LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(lu, "parity", LinalgAdRuleSupport::NonDifferentiable);

    let lu_factor = linalg_ad_support(LinalgAdOpKind::LuFactor);
    assert_eq!(lu_factor.linearize, LinalgAdRuleSupport::Unsupported);
    assert_output_status(lu_factor, "packed_lu", LinalgAdRuleSupport::Unsupported);
    assert_output_status(lu_factor, "pivots", LinalgAdRuleSupport::NonDifferentiable);
    assert_output_status(lu_factor, "parity", LinalgAdRuleSupport::NonDifferentiable);

    let eig = linalg_ad_support(LinalgAdOpKind::Eig);
    assert_eq!(eig.linearize, LinalgAdRuleSupport::PartiallySupported);
    assert_output_status(
        eig,
        "eigenvalues",
        LinalgAdRuleSupport::SupportedViaLinearize,
    );
    assert_output_status(eig, "eigenvectors", LinalgAdRuleSupport::Unsupported);
}

#[test]
fn linalg_ad_support_manifest_marks_vector_outputs_explicitly() {
    let svd = linalg_ad_support(LinalgAdOpKind::Svd);
    assert_eq!(svd.linearize, LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(svd, "u", LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(
        svd,
        "singular_values",
        LinalgAdRuleSupport::SupportedViaLinearize,
    );
    assert_output_status(svd, "vt", LinalgAdRuleSupport::SupportedViaLinearize);

    let eigh = linalg_ad_support(LinalgAdOpKind::Eigh);
    assert_eq!(eigh.linearize, LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(
        eigh,
        "eigenvalues",
        LinalgAdRuleSupport::SupportedViaLinearize,
    );
    assert_output_status(
        eigh,
        "eigenvectors",
        LinalgAdRuleSupport::SupportedViaLinearize,
    );
}

#[test]
fn linalg_ad_support_manifest_keeps_full_pivot_lu_pending_oracle() {
    let full_piv_lu = linalg_ad_support(LinalgAdOpKind::FullPivLu);
    assert_eq!(full_piv_lu.linearize, LinalgAdRuleSupport::PendingOracle);
    assert_output_status(full_piv_lu, "p", LinalgAdRuleSupport::NonDifferentiable);
    assert_output_status(full_piv_lu, "l", LinalgAdRuleSupport::PendingOracle);
    assert_output_status(full_piv_lu, "u", LinalgAdRuleSupport::PendingOracle);
    assert_output_status(full_piv_lu, "q", LinalgAdRuleSupport::NonDifferentiable);
    assert_output_status(
        full_piv_lu,
        "parity",
        LinalgAdRuleSupport::NonDifferentiable,
    );

    let full_piv_lu_solve = linalg_ad_support(LinalgAdOpKind::FullPivLuSolve);
    assert_eq!(
        full_piv_lu_solve.linearize,
        LinalgAdRuleSupport::PendingOracle
    );
    assert_eq!(
        full_piv_lu_solve.transpose,
        LinalgAdRuleSupport::PendingOracle
    );
    assert_output_status(
        full_piv_lu_solve,
        "solution",
        LinalgAdRuleSupport::PendingOracle,
    );
}

fn assert_output_status(
    entry: &tenferro_linalg::LinalgAdSupport,
    name: &str,
    expected: LinalgAdRuleSupport,
) {
    let output = entry
        .outputs
        .iter()
        .find(|output| output.name == name)
        .unwrap_or_else(|| panic!("missing output {name} for {:?}", entry.kind));
    assert_eq!(output.status, expected);
}
