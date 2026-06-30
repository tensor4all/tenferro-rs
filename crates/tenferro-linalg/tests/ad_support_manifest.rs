#![cfg(feature = "autodiff")]

use std::{fs, path::Path};

use tenferro_linalg::{
    all_linalg_ad_support, linalg_ad_support, LinalgAdOpKind, LinalgAdRuleSupport,
};

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tenferro-linalg should live inside workspace/crates")
}

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
        LinalgAdOpKind::EigVals,
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
    assert_eq!(lu.transpose, LinalgAdRuleSupport::Supported);
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
    let qr = linalg_ad_support(LinalgAdOpKind::Qr);
    assert_eq!(qr.linearize, LinalgAdRuleSupport::SupportedViaLinearize);
    assert_eq!(qr.transpose, LinalgAdRuleSupport::Supported);
    assert_output_status(qr, "q", LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(qr, "r", LinalgAdRuleSupport::SupportedViaLinearize);

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
fn linalg_ad_support_manifest_marks_values_only_rules_finite_diff_backed() {
    let svd_vals = linalg_ad_support(LinalgAdOpKind::SvdVals);
    assert_eq!(
        svd_vals.linearize,
        LinalgAdRuleSupport::SupportedViaLinearize
    );
    assert_output_status(
        svd_vals,
        "singular_values",
        LinalgAdRuleSupport::SupportedViaLinearize,
    );

    let eigh_vals = linalg_ad_support(LinalgAdOpKind::EighVals);
    assert_eq!(
        eigh_vals.linearize,
        LinalgAdRuleSupport::SupportedViaLinearize
    );
    assert_output_status(
        eigh_vals,
        "eigenvalues",
        LinalgAdRuleSupport::SupportedViaLinearize,
    );

    let eig_vals = linalg_ad_support(LinalgAdOpKind::EigVals);
    assert_eq!(
        eig_vals.linearize,
        LinalgAdRuleSupport::SupportedViaLinearize
    );
    assert_output_status(
        eig_vals,
        "eigenvalues",
        LinalgAdRuleSupport::SupportedViaLinearize,
    );
}

#[test]
fn linalg_values_only_finite_diff_support_is_documented_next_to_oracle_snapshot() {
    let docs = fs::read_to_string(
        workspace_root()
            .join("docs")
            .join("oracle")
            .join("tensor-ad-oracles-support.md"),
    )
    .expect("oracle support docs should be readable");

    for needle in [
        "| Lu | finite-difference | `grad(sum(l) + sum(u))` |",
        "| Qr | finite-difference | `grad(sum(q) + sum(r))` |",
        "| SvdVals | finite-difference | `norm(..., ord=2)` |",
        "| EighVals | finite-difference | `eigvalsh` |",
        "| EigVals | finite-difference | `eigvals` |",
    ] {
        assert!(
            docs.contains(needle),
            "oracle support docs should include local linalg AD coverage row {needle}"
        );
    }
}

#[test]
fn linalg_ad_support_manifest_marks_full_pivot_lu_oracle_backed() {
    let full_piv_lu = linalg_ad_support(LinalgAdOpKind::FullPivLu);
    assert_eq!(
        full_piv_lu.linearize,
        LinalgAdRuleSupport::SupportedViaLinearize
    );
    assert_output_status(full_piv_lu, "p", LinalgAdRuleSupport::NonDifferentiable);
    assert_output_status(full_piv_lu, "l", LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(full_piv_lu, "u", LinalgAdRuleSupport::SupportedViaLinearize);
    assert_output_status(full_piv_lu, "q", LinalgAdRuleSupport::NonDifferentiable);
    assert_output_status(
        full_piv_lu,
        "parity",
        LinalgAdRuleSupport::NonDifferentiable,
    );

    let full_piv_lu_solve = linalg_ad_support(LinalgAdOpKind::FullPivLuSolve);
    assert_eq!(
        full_piv_lu_solve.linearize,
        LinalgAdRuleSupport::SupportedViaLinearize
    );
    assert_eq!(full_piv_lu_solve.transpose, LinalgAdRuleSupport::Supported);
    assert_output_status(
        full_piv_lu_solve,
        "solution",
        LinalgAdRuleSupport::SupportedViaLinearize,
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
