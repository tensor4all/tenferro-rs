use super::*;
use tenferro_tensor::{MemoryOrder, Tensor};

#[test]
fn structured_ex_apis_expose_expected_fields() {
    let mut ctx = CpuContext::new(1);

    let spd =
        Tensor::from_slice(&[4.0_f64, 2.0, 2.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let chol = cholesky_ex(&mut ctx, &spd).unwrap();
    assert_eq!(chol.l.dims(), &[2, 2]);
    assert_eq!(chol.info, vec![0]);

    let eye =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let inv = inv_ex(&mut ctx, &eye).unwrap();
    assert_eq!(inv.inverse.dims(), &[2, 2]);
    assert_eq!(inv.info, vec![0]);

    let b = Tensor::from_slice(&[3.0_f64, -1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let solved = solve_ex(&mut ctx, &eye, &b).unwrap();
    assert_eq!(solved.solution.dims(), &[2]);
    assert_eq!(solved.info, vec![0]);
}

#[test]
fn lu_factor_contract_uses_packed_factors_and_forward_pivots() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[2.0_f64, 1.0, 1.0, 3.0, 0.0, 4.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let factored = lu_factor(&mut ctx, &a).unwrap();
    assert_eq!(factored.factors.dims(), &[2, 3]);
    assert_eq!(factored.pivots.len(), 2);

    let factored_ex = lu_factor_ex(&mut ctx, &a).unwrap();
    assert_eq!(factored_ex.factors.dims(), &[2, 3]);
    assert_eq!(factored_ex.pivots.len(), 2);
    assert_eq!(factored_ex.info, vec![0]);
}

#[test]
fn lu_solve_uses_lu_factor_output() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[3.0_f64, 1.0, 1.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let b =
        Tensor::from_slice(&[9.0_f64, 8.0, 4.0, 5.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let factored = lu_factor(&mut ctx, &a).unwrap();
    let x_lu = lu_solve(&mut ctx, &factored.factors, &factored.pivots, &b).unwrap();
    let x_direct = solve(&mut ctx, &a, &b).unwrap();

    assert_eq!(x_lu.dims(), &[2, 2]);
    for (lhs, rhs) in tensor_data(&x_lu).iter().zip(tensor_data(&x_direct).iter()) {
        assert!((lhs - rhs).abs() < 1e-12);
    }
}

#[test]
fn structured_ex_and_lu_solve_reject_invalid_contracts() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let bad_rhs = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let solve_err = solve_ex(&mut ctx, &a, &bad_rhs).unwrap_err();
    assert!(matches!(
        solve_err,
        tenferro_device::Error::InvalidArgument(_)
    ));

    let factored = lu_factor(&mut ctx, &a).unwrap();
    let lu_err = lu_solve(&mut ctx, &factored.factors, &[0], &bad_rhs).unwrap_err();
    assert!(matches!(lu_err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn cond_accepts_supported_norm_kinds() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 0.5], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let cond_fro = cond(&mut ctx, &a, NormKind::Fro).unwrap();
    assert!(cond_fro.dims().is_empty());
    assert!((tensor_data(&cond_fro)[0] - 4.25).abs() < 1e-12);

    let cond_l1 = cond(&mut ctx, &a, NormKind::L1).unwrap();
    assert!(cond_l1.dims().is_empty());
    assert!((tensor_data(&cond_l1)[0] - 4.0).abs() < 1e-12);
}

#[test]
fn cond_rejects_unsupported_shapes_and_norms() {
    let mut ctx = CpuContext::new(1);
    let rect = Tensor::from_slice(
        &[1.0_f64, 0.0, 0.0, 1.0, 2.0, 3.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    assert!(cond(&mut ctx, &rect, NormKind::Fro).is_err());

    let square =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    assert!(cond(&mut ctx, &square, NormKind::Nuclear).is_err());
}

#[test]
fn matrix_power_supports_zero_positive_and_negative_exponents() {
    let mut ctx = CpuContext::new(1);
    let diag =
        Tensor::from_slice(&[2.0_f64, 0.0, 0.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let pow0 = matrix_power(&mut ctx, &diag, 0).unwrap();
    assert_eq!(tensor_data(&pow0), vec![1.0, 0.0, 0.0, 1.0]);

    let pow3 = matrix_power(&mut ctx, &diag, 3).unwrap();
    assert_eq!(tensor_data(&pow3), vec![8.0, 0.0, 0.0, 27.0]);

    let pow_neg1 = matrix_power(&mut ctx, &diag, -1).unwrap();
    let data = tensor_data(&pow_neg1);
    assert!((data[0] - 0.5).abs() < 1e-12);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert!((data[3] - (1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn matrix_power_rejects_non_square_input() {
    let mut ctx = CpuContext::new(1);
    let rect = Tensor::from_slice(
        &[1.0_f64, 0.0, 0.0, 1.0, 2.0, 3.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    assert!(matrix_power(&mut ctx, &rect, 2).is_err());
}
