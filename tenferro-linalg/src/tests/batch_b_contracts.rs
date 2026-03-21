use super::*;
use tenferro_tensor::{MemoryOrder, Tensor};

#[cfg(feature = "cuda")]
fn with_cuda_ctx<T>(f: impl FnOnce(&mut tenferro_prims::CudaContext) -> T) -> Option<T> {
    let path = [
        "/usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so",
        "/usr/lib/x86_64-linux-gnu/libcutensor.so",
        "/usr/lib/libcutensor.so",
    ]
    .into_iter()
    .find(|path| std::path::Path::new(path).exists())?;
    let (_backend, mut ctx) = tenferro_prims::CudaBackend::load(path).ok()?;
    Some(f(&mut ctx))
}

#[cfg(not(feature = "cuda"))]
fn with_cuda_ctx<T>(f: impl FnOnce(&mut tenferro_prims::CudaContext) -> T) -> Option<T> {
    let mut ctx = tenferro_prims::CudaContext::new();
    Some(f(&mut ctx))
}

#[cfg(feature = "cuda")]
fn tensor_data_on_cpu(tensor: &Tensor<f64>) -> Vec<f64> {
    let cpu = tensor
        .to_memory_space_async(tenferro_device::LogicalMemorySpace::MainMemory)
        .unwrap();
    let contiguous = cpu.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[cfg(feature = "cuda")]
fn assert_close_slice(label: &str, got: &[f64], expected: &[f64], atol: f64) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch {} vs {}",
        got.len(),
        expected.len()
    );
    for (idx, (&got_v, &exp_v)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (got_v - exp_v).abs();
        assert!(
            diff <= atol,
            "{label}[{idx}] diff {diff} exceeded tolerance {atol}; got {got_v}, expected {exp_v}"
        );
    }
}

#[cfg(feature = "cuda")]
fn matmul_col_major(lhs: &[f64], m: usize, k: usize, rhs: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for col in 0..n {
        for row in 0..m {
            let mut acc = 0.0;
            for inner in 0..k {
                acc += lhs[row + inner * m] * rhs[inner + col * k];
            }
            out[row + col * m] = acc;
        }
    }
    out
}

#[cfg(feature = "cuda")]
fn reconstruct_thin_svd_col_major(
    u: &[f64],
    m: usize,
    k: usize,
    s: &[f64],
    vt: &[f64],
    n: usize,
) -> Vec<f64> {
    let mut scaled_u = u.to_vec();
    for col in 0..k {
        for row in 0..m {
            scaled_u[row + col * m] *= s[col];
        }
    }
    matmul_col_major(&scaled_u, m, k, vt, n)
}

#[cfg(feature = "cuda")]
fn public_cuda_svd_matrix(wide: bool) -> Tensor<f64> {
    if wide {
        Tensor::from_slice(
            &[3.0_f64, 1.0, 1.0, 2.0, 0.0, 1.0],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    } else {
        Tensor::from_slice(
            &[3.0_f64, 1.0, 0.0, 1.0, 2.0, 1.0],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap()
    }
}

#[cfg(feature = "cuda")]
fn cuda_public_svdvals_matches_cpu_generic(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = public_cuda_svd_matrix(wide);
        let expected = svdvals(&mut cpu_ctx, &a_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svdvals(ctx, &a_gpu).unwrap();
        assert_eq!(
            got.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.dims(), expected.dims());
        assert_close_slice(
            if wide {
                "cuda public svdvals wide"
            } else {
                "cuda public svdvals tall"
            },
            &tensor_data_on_cpu(&got),
            &tensor_data(&expected),
            2048.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svd_reconstructs_generic(wide: bool) {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = public_cuda_svd_matrix(wide);
        let expected = svd(&mut cpu_ctx, &a_cpu, None).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = svd(ctx, &a_gpu, None).unwrap();
        assert_eq!(
            got.u.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.s.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            got.vt.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.u.dims(), expected.u.dims());
        assert_eq!(got.s.dims(), expected.s.dims());
        assert_eq!(got.vt.dims(), expected.vt.dims());
        assert_close_slice(
            if wide {
                "cuda public svd singular values wide"
            } else {
                "cuda public svd singular values tall"
            },
            &tensor_data_on_cpu(&got.s),
            &tensor_data(&expected.s),
            2048.0 * f64::EPSILON,
        );

        let reconstructed = reconstruct_thin_svd_col_major(
            &tensor_data_on_cpu(&got.u),
            a_cpu.dims()[0],
            got.s.dims()[0],
            &tensor_data_on_cpu(&got.s),
            &tensor_data_on_cpu(&got.vt),
            a_cpu.dims()[1],
        );
        assert_close_slice(
            if wide {
                "cuda public svd reconstruction wide"
            } else {
                "cuda public svd reconstruction tall"
            },
            &reconstructed,
            &tensor_data(&a_cpu),
            4096.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_svd_cutoff_preserves_zero_fill_semantics() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let a_cpu = Tensor::from_slice(
            &[3.0_f64, 0.0, 0.0, 0.25],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let opts = SvdOptions {
            max_rank: Some(2),
            cutoff: Some(0.5),
        };

        let result = svd(ctx, &a_gpu, Some(&opts)).unwrap();
        assert_eq!(
            result.u.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            result.s.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(
            result.vt.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(result.u.dims(), &[2, 2]);
        assert_eq!(result.s.dims(), &[2]);
        assert_eq!(result.vt.dims(), &[2, 2]);
        assert_eq!(tensor_data_on_cpu(&result.s), vec![3.0, 0.0]);

        let u = tensor_data_on_cpu(&result.u);
        let vt = tensor_data_on_cpu(&result.vt);
        assert_eq!(u[2], 0.0);
        assert_eq!(u[3], 0.0);
        assert_eq!(vt[1], 0.0);
        assert_eq!(vt[3], 0.0);
    }) else {
        return;
    };
}

#[cfg(feature = "cuda")]
fn cuda_public_solve_ex_matches_cpu_generic() {
    let Some(()) = with_cuda_ctx(|ctx| {
        let mut cpu_ctx = CpuContext::new(1);
        let a_cpu = Tensor::from_slice(
            &[
                1.0_f64, 0.0, 0.0, 1.0, //
                1.0, 2.0, 2.0, 4.0,
            ],
            &[2, 2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let b_cpu = Tensor::from_slice(
            &[3.0_f64, -1.0, 1.0, 1.0],
            &[2, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap();
        let expected = solve_ex(&mut cpu_ctx, &a_cpu, &b_cpu).unwrap();
        let a_gpu = a_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();
        let b_gpu = b_cpu
            .to_memory_space_async(tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 })
            .unwrap();

        let got = solve_ex(ctx, &a_gpu, &b_gpu).unwrap();
        assert_eq!(
            got.solution.logical_memory_space(),
            tenferro_device::LogicalMemorySpace::GpuMemory { device_id: 0 }
        );
        assert_eq!(got.info, expected.info);
        assert_close_slice(
            "cuda public solve_ex",
            &tensor_data_on_cpu(&got.solution),
            &tensor_data(&expected.solution),
            256.0 * f64::EPSILON,
        );
    }) else {
        return;
    };
}

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
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_tall_matrix() {
    cuda_public_svdvals_matches_cpu_generic(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svdvals_matches_cpu_for_wide_matrix() {
    cuda_public_svdvals_matches_cpu_generic(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_tall_matrix() {
    cuda_public_svd_reconstructs_generic(false);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_reconstructs_wide_matrix() {
    cuda_public_svd_reconstructs_generic(true);
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_svd_cutoff_preserves_zero_fill_semantics() {
    cuda_public_svd_cutoff_preserves_zero_fill_semantics();
}

#[test]
#[cfg(feature = "cuda")]
fn public_cuda_solve_ex_matches_cpu_for_mixed_batch() {
    cuda_public_solve_ex_matches_cpu_generic();
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
    with_cuda_ctx(|ctx| {
        let a_vec =
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        let b_vec =
            Tensor::from_slice(&[0.0_f64, 1.0, 0.0], &[3], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            cross(ctx, &a_vec, &b_vec),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        let reflectors =
            Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
                .unwrap();
        let tau = Tensor::from_slice(&[0.0_f64, 0.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            householder_product(ctx, &reflectors, &tau),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        let x = Tensor::from_slice(&[2.0_f64, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            vander(ctx, &x, Some(3), true),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        let eye = Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap();
        assert!(matches!(
            tensorinv(ctx, &eye, 1),
            Err(tenferro_device::Error::DeviceError(_))
        ));

        let rhs = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
        assert!(matches!(
            tensorsolve(ctx, &eye, &rhs, None),
            Err(tenferro_device::Error::DeviceError(_))
        ));
    });
}
