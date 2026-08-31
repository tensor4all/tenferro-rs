use tenferro_tensor::DType;

use crate::extension::{EighGauge, QrGauge, SvdGauge, DEFAULT_DECOMPOSITION_DERIVATIVE_EPS};

use super::*;

#[test]
fn manifest_internal_mapping_covers_linalg_op_variants() {
    let samples = [
        (LinalgOp::Cholesky, LinalgAdOpKind::Cholesky),
        (LinalgOp::Lu, LinalgAdOpKind::Lu),
        (LinalgOp::LuFactor, LinalgAdOpKind::LuFactor),
        (
            LinalgOp::LuSolvePrepared {
                transpose_a: false,
                conjugate_a: false,
            },
            LinalgAdOpKind::LuSolvePrepared,
        ),
        (LinalgOp::FullPivLu, LinalgAdOpKind::FullPivLu),
        (
            LinalgOp::FullPivLuSolve { transpose_a: false },
            LinalgAdOpKind::FullPivLuSolve,
        ),
        // Single-op partial-pivot solve shares the solve-family manifest kind.
        (LinalgOp::Solve, LinalgAdOpKind::FullPivLuSolve),
        (
            LinalgOp::Svd {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                gauge: SvdGauge::Raw,
            },
            LinalgAdOpKind::Svd,
        ),
        (
            LinalgOp::SvdVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
            },
            LinalgAdOpKind::SvdVals,
        ),
        (LinalgOp::SvdFull, LinalgAdOpKind::SvdFull),
        (
            LinalgOp::Qr {
                gauge: QrGauge::Raw,
            },
            LinalgAdOpKind::Qr,
        ),
        (
            LinalgOp::HouseholderQrFactor,
            LinalgAdOpKind::HouseholderQrFactor,
        ),
        (
            LinalgOp::HouseholderQrFromFactors,
            LinalgAdOpKind::HouseholderQrFromFactors,
        ),
        (
            LinalgOp::HouseholderQrAppend,
            LinalgAdOpKind::HouseholderQrAppend,
        ),
        (
            LinalgOp::HouseholderQrR {
                gauge: QrGauge::Raw,
            },
            LinalgAdOpKind::HouseholderQrR,
        ),
        (
            LinalgOp::HouseholderQrQColumns {
                start: 0,
                end: 1,
                gauge: QrGauge::Raw,
            },
            LinalgAdOpKind::HouseholderQrQColumns,
        ),
        (
            LinalgOp::Eigh {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
                gauge: EighGauge::Raw,
            },
            LinalgAdOpKind::Eigh,
        ),
        (
            LinalgOp::EighVals {
                derivative_eps: DEFAULT_DECOMPOSITION_DERIVATIVE_EPS,
            },
            LinalgAdOpKind::EighVals,
        ),
        (
            LinalgOp::Eig {
                input_dtype: DType::F64,
            },
            LinalgAdOpKind::Eig,
        ),
        (
            LinalgOp::EigVals {
                input_dtype: DType::F64,
            },
            LinalgAdOpKind::EigVals,
        ),
        (
            LinalgOp::TriangularSolve {
                left_side: true,
                lower: true,
                transpose_a: false,
                unit_diagonal: false,
            },
            LinalgAdOpKind::TriangularSolve,
        ),
    ];

    for (op, kind) in samples {
        assert_eq!(linalg_ad_support_for_op(op).kind, kind);
    }
}

#[test]
fn helper_routes_cover_all_rule_statuses() {
    let supported_statuses = [
        LinalgAdRuleSupport::Supported,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::PartiallySupported,
    ];
    for status in supported_statuses {
        assert_eq!(jvp_route(status), LinalgAdRoute::Linearize);
        assert_eq!(mode(status, jvp_route(status)).status, status);
    }

    let inactive_statuses = [
        LinalgAdRuleSupport::Unsupported,
        LinalgAdRuleSupport::NonDifferentiable,
        LinalgAdRuleSupport::PendingOracle,
    ];
    for status in inactive_statuses {
        assert_eq!(jvp_route(status), LinalgAdRoute::Unsupported);
        assert_eq!(
            mode(status, jvp_route(status)).route,
            LinalgAdRoute::Unsupported
        );
    }
}

#[test]
fn support_entry_helper_preserves_manifest_fields() {
    static OUTPUTS: [LinalgAdOutputSupport; 1] = [output(
        0,
        "solution",
        LinalgAdRuleSupport::SupportedViaLinearize,
    )];
    static CAVEATS: [&str; 1] = ["test caveat"];

    let vjp = mode(
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRoute::LinearizeThenCustomLinearTranspose,
    );
    let entry = support_entry(
        LinalgAdOpKind::TriangularSolve,
        LinalgAdRuleSupport::SupportedViaLinearize,
        LinalgAdRuleSupport::Supported,
        vjp,
        LinalgAdRuleSupport::Supported,
        &OUTPUTS,
        &CAVEATS,
    );

    assert_eq!(entry.kind, LinalgAdOpKind::TriangularSolve);
    assert_eq!(entry.jvp.status, LinalgAdRuleSupport::SupportedViaLinearize);
    assert_eq!(entry.jvp.route, LinalgAdRoute::Linearize);
    assert_eq!(entry.vjp, vjp);
    assert_eq!(
        entry.linearize_rule,
        LinalgAdRuleSupport::SupportedViaLinearize
    );
    assert_eq!(entry.custom_vjp_rule, LinalgAdRuleSupport::Unsupported);
    assert_eq!(
        entry.custom_linear_transpose_rule,
        LinalgAdRuleSupport::Supported
    );
    assert_eq!(entry.linearize, LinalgAdRuleSupport::SupportedViaLinearize);
    assert_eq!(entry.transpose, LinalgAdRuleSupport::Supported);
    assert_eq!(entry.outputs, &OUTPUTS);
    assert_eq!(entry.caveats, &CAVEATS);
}
