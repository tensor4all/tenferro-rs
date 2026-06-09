use tenferro_tensor::DType;

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
        (LinalgOp::Svd { eps: 1.0e-12 }, LinalgAdOpKind::Svd),
        (LinalgOp::SvdVals { eps: 1.0e-12 }, LinalgAdOpKind::SvdVals),
        (LinalgOp::Qr, LinalgAdOpKind::Qr),
        (LinalgOp::Eigh { eps: 1.0e-12 }, LinalgAdOpKind::Eigh),
        (
            LinalgOp::EighVals { eps: 1.0e-12 },
            LinalgAdOpKind::EighVals,
        ),
        (
            LinalgOp::Eig {
                input_dtype: DType::F64,
            },
            LinalgAdOpKind::Eig,
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
