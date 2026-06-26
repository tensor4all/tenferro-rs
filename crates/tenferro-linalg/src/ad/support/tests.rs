use tenferro_tensor::DType;

use crate::extension::DEFAULT_DECOMPOSITION_AD_EPS;

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
        (
            LinalgOp::Svd {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
            },
            LinalgAdOpKind::Svd,
        ),
        (
            LinalgOp::SvdVals {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
            },
            LinalgAdOpKind::SvdVals,
        ),
        (LinalgOp::Qr, LinalgAdOpKind::Qr),
        (
            LinalgOp::Eigh {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
            },
            LinalgAdOpKind::Eigh,
        ),
        (
            LinalgOp::EighVals {
                eps: DEFAULT_DECOMPOSITION_AD_EPS,
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
