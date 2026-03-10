use super::*;
use tenferro_tensor::{MemoryOrder, Tensor};

#[test]
fn cross_matches_right_hand_rule_with_trailing_batches() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::from_slice(
        &[
            0.0_f64, 1.0, 0.0, //
            0.0, 0.0, 1.0,
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let out = cross(&mut ctx, &a, &b).unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert_eq!(
        tensor_data(&out),
        vec![
            0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0,
        ]
    );
}

#[test]
fn cross_rejects_non_three_vector_axis() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let err = cross(&mut ctx, &a, &a).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn cross_supports_singleton_broadcasting_and_rejects_rank_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, //
            0.0, 1.0, 0.0,
        ],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3, 1], MemoryOrder::ColumnMajor).unwrap();
    let out = cross(&mut ctx, &a, &b).unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert_eq!(tensor_data(&out), vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);

    let rank_mismatch =
        Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = cross(&mut ctx, &a, &rank_mismatch).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn cross_rejects_rhs_vector_axis_and_broadcast_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let bad_rhs =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let rhs_err = cross(&mut ctx, &a, &bad_rhs).unwrap_err();
    assert!(matches!(
        rhs_err,
        tenferro_device::Error::InvalidArgument(_)
    ));

    let mismatch = Tensor::from_slice(
        &[0.0_f64, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mismatch_err = cross(&mut ctx, &a, &mismatch).unwrap_err();
    assert!(matches!(
        mismatch_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
}

#[test]
fn householder_product_zero_tau_returns_identity_columns() {
    let mut ctx = CpuContext::new(1);
    let reflectors = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, 0.0, //
            3.0, 4.0, 1.0, 0.0,
        ],
        &[4, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tau = Tensor::from_slice(&[0.0_f64, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();

    let q = householder_product(&mut ctx, &reflectors, &tau).unwrap();
    assert_eq!(q.dims(), &[4, 3]);
    assert_eq!(
        tensor_data(&q),
        vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ]
    );
}

#[test]
fn householder_product_applies_nonzero_reflector_and_rejects_oversized_k() {
    let mut ctx = CpuContext::new(1);
    let reflectors =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let tau = Tensor::from_slice(&[2.0_f64], &[1], MemoryOrder::ColumnMajor).unwrap();
    let out = householder_product(&mut ctx, &reflectors, &tau).unwrap();
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(tensor_data(&out), vec![-1.0, 0.0, 0.0, 1.0]);

    let skinny_reflectors =
        Tensor::from_slice(&[1.0_f64, 0.0], &[2, 1], MemoryOrder::ColumnMajor).unwrap();
    let oversized_tau =
        Tensor::from_slice(&[1.0_f64, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let err = householder_product(&mut ctx, &skinny_reflectors, &oversized_tau).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn householder_product_rejects_invalid_tau_shape() {
    let mut ctx = CpuContext::new(1);
    let reflectors =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let tau = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = householder_product(&mut ctx, &reflectors, &tau).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn householder_product_supports_batches_and_rejects_batch_mismatch() {
    let mut ctx = CpuContext::new(1);
    let reflectors = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, //
            1.0, 0.0, 0.0, 1.0,
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tau = Tensor::from_slice(&[0.0_f64, 0.0], &[1, 2], MemoryOrder::ColumnMajor).unwrap();
    let out = householder_product(&mut ctx, &reflectors, &tau).unwrap();
    assert_eq!(out.dims(), &[2, 2, 2]);

    let bad_tau = Tensor::from_slice(&[0.0_f64, 0.0], &[2, 1], MemoryOrder::ColumnMajor).unwrap();
    let err = householder_product(&mut ctx, &reflectors, &bad_tau).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn vander_supports_default_and_custom_column_counts() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap();

    let default = vander(&mut ctx, &x, None, false).unwrap();
    assert_eq!(default.dims(), &[2, 2]);
    assert_eq!(tensor_data(&default), vec![2.0, 3.0, 1.0, 1.0]);

    let increasing = vander(&mut ctx, &x, Some(4), true).unwrap();
    assert_eq!(increasing.dims(), &[2, 4]);
    assert_eq!(
        tensor_data(&increasing),
        vec![
            1.0, 1.0, //
            2.0, 3.0, //
            4.0, 9.0, //
            8.0, 27.0,
        ]
    );
}

#[test]
fn vander_handles_scalar_inputs() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::from_slice(&[2.0_f64], &[], MemoryOrder::ColumnMajor).unwrap();
    let out = vander(&mut ctx, &x, Some(4), false).unwrap();
    assert_eq!(out.dims(), &[1, 4]);
    assert_eq!(tensor_data(&out), vec![8.0, 4.0, 2.0, 1.0]);
}

#[test]
fn vander_supports_batched_vectors() {
    let mut ctx = CpuContext::new(1);
    let x =
        Tensor::from_slice(&[2.0_f64, 3.0, 4.0, 5.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let out = vander(&mut ctx, &x, Some(3), false).unwrap();
    assert_eq!(out.dims(), &[2, 3, 2]);
    assert_eq!(
        tensor_data(&out),
        vec![
            4.0, 9.0, 2.0, 3.0, 1.0, 1.0, //
            16.0, 25.0, 4.0, 5.0, 1.0, 1.0,
        ]
    );
}

#[test]
fn tensorinv_inverts_tensorized_identity() {
    let mut ctx = CpuContext::new(1);
    let eye = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tensor = eye.reshape(&[2, 2, 2, 2]).unwrap();

    let inverse = tensorinv(&mut ctx, &tensor, 2).unwrap();
    assert_eq!(inverse.dims(), &[2, 2, 2, 2]);
    assert_eq!(tensor_data(&inverse), tensor_data(&tensor));
}

#[test]
fn tensorinv_rejects_non_square_partition() {
    let mut ctx = CpuContext::new(1);
    let tensor = Tensor::from_slice(
        &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let err = tensorinv(&mut ctx, &tensor, 1).unwrap_err();
    assert!(matches!(err, tenferro_device::Error::InvalidArgument(_)));
}

#[test]
fn tensorinv_rejects_zero_and_terminal_split_points() {
    let mut ctx = CpuContext::new(1);
    let tensor =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let zero_err = tensorinv(&mut ctx, &tensor, 0).unwrap_err();
    assert!(matches!(
        zero_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
    let terminal_err = tensorinv(&mut ctx, &tensor, 2).unwrap_err();
    assert!(matches!(
        terminal_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
}

#[test]
fn tensorsolve_matches_identity_tensor_operator() {
    let mut ctx = CpuContext::new(1);
    let eye = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a = eye.reshape(&[2, 2, 2, 2]).unwrap();
    let b =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let x = tensorsolve(&mut ctx, &a, &b, None).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(tensor_data(&x), tensor_data(&b));
}

#[test]
fn tensorsolve_respects_solution_axis_order() {
    let mut ctx = CpuContext::new(1);
    let eye = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let a = eye.reshape(&[2, 2, 2, 2]).unwrap();
    let b =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let x = tensorsolve(&mut ctx, &a, &b, Some(&[3, 2])).unwrap();
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(tensor_data(&x), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn tensorsolve_rejects_rank_and_shape_contract_violations() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let b_rank =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[1, 1, 3], MemoryOrder::ColumnMajor).unwrap();
    let rank_err = tensorsolve(&mut ctx, &a, &b_rank, None).unwrap_err();
    assert!(matches!(
        rank_err,
        tenferro_device::Error::InvalidArgument(_)
    ));

    let reshaped = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &[2, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_lead = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let lead_err = tensorsolve(&mut ctx, &reshaped, &b_lead, Some(&[0, 2])).unwrap_err();
    assert!(matches!(
        lead_err,
        tenferro_device::Error::InvalidArgument(_)
    ));

    let a_size = Tensor::from_slice(
        &[
            1.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        ],
        &[2, 2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b_size =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let size_err = tensorsolve(&mut ctx, &a_size, &b_size, None).unwrap_err();
    assert!(matches!(
        size_err,
        tenferro_device::Error::InvalidArgument(_)
    ));
}

#[test]
fn batch_b_ops_reject_cuda_context() {
    let mut ctx = tenferro_prims::CudaContext::new();

    let a_vec = Tensor::from_slice(&[1.0_f64, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let b_vec = Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    assert!(matches!(
        cross(&mut ctx, &a_vec, &b_vec),
        Err(tenferro_device::Error::DeviceError(_))
    ));

    let reflectors =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let tau = Tensor::from_slice(&[0.0_f64, 0.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    assert!(matches!(
        householder_product(&mut ctx, &reflectors, &tau),
        Err(tenferro_device::Error::DeviceError(_))
    ));

    let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    assert!(matches!(
        vander(&mut ctx, &x, Some(3), true),
        Err(tenferro_device::Error::DeviceError(_))
    ));

    let eye =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    assert!(matches!(
        tensorinv(&mut ctx, &eye, 1),
        Err(tenferro_device::Error::DeviceError(_))
    ));

    let rhs = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    assert!(matches!(
        tensorsolve(&mut ctx, &eye, &rhs, None),
        Err(tenferro_device::Error::DeviceError(_))
    ));
}
