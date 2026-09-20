use crate::extension::{execute_linalg_extension_reads_in_session, LinalgExtensionOp, LinalgOp};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorView, TypedTensorView};

#[test]
fn prepared_solve_preserves_mixed_owned_and_strided_inputs() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let a = Tensor::from_vec_col_major([2, 2], vec![2.0_f64, 0.0, 0.0, 3.0]).unwrap();
    let factors = backend
        .with_backend_session(|session| {
            with_cpu_exec_session(session, |session| {
                execute_linalg_extension_reads_in_session(
                    &LinalgExtensionOp::new(LinalgOp::LuFactor),
                    &[TensorRead::from_tensor(&a)],
                    session,
                )
            })
            .unwrap()
        })
        .unwrap();
    let b = Tensor::from_vec_col_major([2, 1], vec![4.0_f64, 9.0]).unwrap();
    let strided_data = [99.0_f64, 4.0, 99.0, 9.0, 99.0];
    let strided_rhs =
        TensorView::F64(TypedTensorView::from_slice([2, 1], [2, 4], 1, &strided_data).unwrap());
    let base = [
        TensorRead::from_tensor(&a),
        TensorRead::from_tensor(&factors[0]),
        TensorRead::from_tensor(&factors[1]),
        TensorRead::from_tensor(&b),
    ];
    let original_lu = factors[0].as_slice::<f64>().unwrap().to_vec();
    for mask in 0..16 {
        let reads = base
            .iter()
            .enumerate()
            .map(|(i, read)| {
                if mask & (1 << i) == 0 {
                    read.clone()
                } else if i == 3 {
                    TensorRead::from_view(strided_rhs.clone())
                } else {
                    TensorRead::from_view(read.clone().tensor_view())
                }
            })
            .collect::<Vec<_>>();
        for transpose_a in [false, true] {
            let op = LinalgExtensionOp::new(LinalgOp::LuSolvePrepared {
                transpose_a,
                conjugate_a: true,
            });
            let result = backend
                .with_backend_session(|session| {
                    with_cpu_exec_session(session, |session| {
                        execute_linalg_extension_reads_in_session(&op, &reads, session)
                    })
                    .unwrap()
                })
                .unwrap();
            assert_eq!(result[0].as_slice::<f64>().unwrap(), &[2.0, 3.0]);
        }
    }
    assert_eq!(a.as_slice::<f64>().unwrap(), &[2.0, 0.0, 0.0, 3.0]);
    assert_eq!(factors[0].as_slice::<f64>().unwrap(), original_lu);
    assert_eq!(b.as_slice::<f64>().unwrap(), &[4.0, 9.0]);
    assert_eq!(strided_data, [99.0, 4.0, 99.0, 9.0, 99.0]);
}

#[test]
fn prepared_solve_does_not_copy_owned_factors_or_conjugate_real_lu() {
    // Data-movement contract: numerical equality alone cannot detect these copies.
    let extension = include_str!("../../extension.rs");
    let fallback = extension
        .split("let materialized_inputs = inputs")
        .nth(1)
        .unwrap()
        .split("execute_linalg(op.op()")
        .next()
        .unwrap();
    assert!(fallback.contains(".filter(|input| input.as_tensor().is_none())"));
    assert!(fallback.contains("input.as_tensor().or_else(|| views.next())"));
    let backend = include_str!("../../cpu/backend.rs");
    let solve = backend
        .split("const OP: &str = \"lu_solve_prepared\";")
        .nth(1)
        .unwrap()
        .split("fn solve(")
        .next()
        .unwrap();
    assert!(solve.contains("matches!(packed_lu.dtype(), DType::C32 | DType::C64)"));
    assert!(solve.contains("conjugated_lu.as_ref().unwrap_or(packed_lu)"));
    assert!(!solve.contains("packed_lu.duplicate()"));
}
