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
fn solve_ex_mixed_batches_preserve_successful_solution_and_report_zero_pivot() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, //
            1.0, 2.0, 2.0, 4.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::from_slice(
        &[3.0_f64, -1.0, 1.0, 1.0],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = solve_ex(&mut ctx, &a, &b).unwrap();
    assert_eq!(result.info, vec![0, 2]);

    let payload = tensor_data(&result.solution);
    assert_eq!(&payload[..2], &[3.0, -1.0]);
}

#[test]
fn inv_ex_mixed_batches_preserve_successful_inverse_and_report_zero_pivot() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, //
            1.0, 2.0, 2.0, 4.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = inv_ex(&mut ctx, &a).unwrap();
    assert_eq!(result.info, vec![0, 2]);

    let payload = tensor_data(&result.inverse);
    assert_eq!(&payload[..4], &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn inv_0x0_returns_empty_inverse() {
    let mut ctx = CpuContext::new(1);
    let a: Tensor<f64> = Tensor::from_vec(vec![], &[0, 0], &[1, 0], 0).unwrap();

    let inverse = inv(&mut ctx, &a).unwrap();
    let contiguous = inverse.contiguous(MemoryOrder::ColumnMajor);

    assert_eq!(inverse.dims(), &[0, 0]);
    assert!(contiguous.buffer().as_slice().unwrap().is_empty());
}

#[test]
fn inv_ex_0x0_batches_return_empty_inverse_and_batch_shaped_zero_info() {
    let mut ctx = CpuContext::new(1);
    let two_batch: Tensor<f64> = Tensor::from_vec(vec![], &[0, 0, 2], &[1, 0, 0], 0).unwrap();
    let zero_batch: Tensor<f64> = Tensor::from_vec(vec![], &[0, 0, 0], &[1, 0, 0], 0).unwrap();

    let two_batch_result = inv_ex(&mut ctx, &two_batch).unwrap();
    let zero_batch_result = inv_ex(&mut ctx, &zero_batch).unwrap();

    let two_batch_contiguous = two_batch_result
        .inverse
        .contiguous(MemoryOrder::ColumnMajor);
    assert_eq!(two_batch_result.inverse.dims(), &[0, 0, 2]);
    assert!(two_batch_contiguous.buffer().as_slice().unwrap().is_empty());
    assert_eq!(two_batch_result.info, vec![0, 0]);

    let zero_batch_contiguous = zero_batch_result
        .inverse
        .contiguous(MemoryOrder::ColumnMajor);
    assert_eq!(zero_batch_result.inverse.dims(), &[0, 0, 0]);
    assert!(zero_batch_contiguous
        .buffer()
        .as_slice()
        .unwrap()
        .is_empty());
    assert!(zero_batch_result.info.is_empty());
}

#[test]
fn cholesky_ex_mixed_batches_preserve_successful_factor_and_report_failing_minor() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            4.0_f64, 2.0, 2.0, 3.0, //
            1.0, 2.0, 2.0, 1.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = cholesky_ex(&mut ctx, &a).unwrap();
    assert_eq!(result.info, vec![0, 2]);

    let payload = tensor_data(&result.l);
    assert_eq!(&payload[..3], &[2.0, 1.0, 0.0]);
    assert!((payload[3] - (2.0_f64).sqrt()).abs() < 1e-12);
}

#[test]
fn cholesky_ex_multi_axis_batches_follow_column_major_batch_order() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            4.0_f64, 2.0, 2.0, 3.0, //
            1.0, 2.0, 2.0, 1.0, //
            -1.0, 0.0, 0.0, 1.0, //
            9.0, 0.0, 0.0, 4.0,
        ],
        &[2, 2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = cholesky_ex(&mut ctx, &a).unwrap();
    assert_eq!(result.info, vec![0, 2, 1, 0]);

    let payload = tensor_data(&result.l);
    assert_eq!(&payload[..3], &[2.0, 1.0, 0.0]);
    assert!((payload[3] - (2.0_f64).sqrt()).abs() < 1e-12);
    assert_eq!(&payload[12..16], &[3.0, 0.0, 0.0, 2.0]);
}

#[test]
fn lu_factor_ex_does_not_treat_small_nonzero_pivot_as_zero() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[1.0e-20_f64, 0.0, 0.0, 1.0],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let result = lu_factor_ex(&mut ctx, &a).unwrap();
    assert_eq!(result.info, vec![0]);
}

#[test]
fn lu_factor_ex_mixed_batches_preserve_successful_packed_factors_and_info() {
    let mut ctx = CpuContext::new(1);
    let good =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let batched = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, //
            1.0, 2.0, 2.0, 4.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let expected = lu_factor(&mut ctx, &good).unwrap();
    let result = lu_factor_ex(&mut ctx, &batched).unwrap();

    assert_eq!(result.info, vec![0, 2]);
    assert_eq!(&result.pivots[..2], expected.pivots.as_slice());
    assert_eq!(
        &tensor_data(&result.factors)[..4],
        tensor_data(&expected.factors).as_slice()
    );
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
fn matrix_power_zero_batched_returns_identity_per_batch() {
    let mut ctx = CpuContext::new(1);
    let batched = Tensor::from_slice(
        &[
            2.0_f64, 0.0, 0.0, 3.0, //
            4.0, 0.0, 0.0, 5.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let pow0 = matrix_power(&mut ctx, &batched, 0).unwrap();
    assert_eq!(pow0.dims(), &[2, 2, 2]);
    assert_eq!(
        tensor_data(&pow0),
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    );
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
