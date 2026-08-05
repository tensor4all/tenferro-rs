// CUDA-native dot-general accumulation (tensor4all/tenferro-rs#1287).
// Run with: cargo test --features cuda -- --ignored
use num_complex::Complex64;

use crate::DotGeneralConfig;
use crate::Tensor;
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, TensorDot, TensorRead, TensorView, TensorViewMut,
    TensorWrite, TypedTensorView,
};

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c64, tensor_f32, tensor_f64,
    upload,
};

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

/// Reference on CPU, native accumulation on GPU, elementwise comparison.
fn run_accum_case(
    lhs: Tensor,
    rhs: Tensor,
    out_init: Tensor,
    config: DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    tol: f64,
) {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let mut expected = out_init.duplicate().expect("host output duplication");
    cpu.dot_general_read_into_accum(
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        accumulation,
        TensorWrite::from_tensor(&mut expected),
    )
    .unwrap();

    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let mut gpu_out = upload(&gpu, &out_init);
    gpu.dot_general_read_into_accum(
        TensorRead::from_tensor(&gpu_lhs),
        TensorRead::from_tensor(&gpu_rhs),
        &config,
        accumulation,
        TensorWrite::from_tensor(&mut gpu_out),
    )
    .unwrap();
    let actual = download(&gpu, &gpu_out);

    assert_eq!(actual.shape(), expected.shape());
    assert_tensor_close(&actual, &expected, tol);
}

#[test]
#[ignore]
fn test_accum_overwrite_compatible_f32() {
    run_accum_case(
        tensor_f32(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        tensor_f32(vec![3, 2], vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
        tensor_f32(vec![2, 2], vec![9.0, 9.0, 9.0, 9.0]),
        matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F32(1.0),
            beta: ContractionScalar::F32(0.0),
        },
        1e-4,
    );
}

#[test]
#[ignore]
fn test_accum_nontrivial_alpha_beta_f64() {
    run_accum_case(
        tensor_f64(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        tensor_f64(vec![3, 2], vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
        tensor_f64(vec![2, 2], vec![0.5, -1.5, 2.5, -3.5]),
        matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(2.0),
            beta: ContractionScalar::F64(-0.5),
        },
        1e-10,
    );
}

#[test]
#[ignore]
fn test_accum_lhs_conj_c64() {
    let lhs: Vec<Complex64> = (0..6)
        .map(|i| Complex64::new(i as f64 - 2.0, 0.5 * i as f64 + 1.0))
        .collect();
    let rhs: Vec<Complex64> = (0..6)
        .map(|i| Complex64::new(0.25 * i as f64 + 0.5, -0.75 * i as f64))
        .collect();
    let out: Vec<Complex64> = (0..4)
        .map(|i| Complex64::new(1.0 - i as f64, 2.0 * i as f64))
        .collect();
    run_accum_case(
        tensor_c64(vec![2, 3], lhs),
        tensor_c64(vec![3, 2], rhs),
        tensor_c64(vec![2, 2], out),
        matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: true,
            rhs_conj: false,
            alpha: ContractionScalar::C64(Complex64::new(2.0, 0.0)),
            beta: ContractionScalar::C64(Complex64::new(1.0, 0.0)),
        },
        1e-10,
    );
}

#[test]
#[ignore]
fn test_accum_zero_contraction_scales_destination_f64() {
    // k = 0: the contraction sum is empty, out = beta * out.
    run_accum_case(
        tensor_f64(vec![2, 0], vec![]),
        tensor_f64(vec![0, 2], vec![]),
        tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(-2.0),
        },
        1e-12,
    );
}

#[test]
#[ignore]
fn test_accum_dtype_mismatch_rejected() {
    let mut gpu = gpu_backend();
    let lhs = upload(&gpu, &tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let rhs = upload(&gpu, &tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let mut out = upload(&gpu, &tensor_f64(vec![2, 2], vec![0.0; 4]));
    let result = gpu.dot_general_read_into_accum(
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F32(1.0),
            beta: ContractionScalar::F64(0.0),
        },
        TensorWrite::from_tensor(&mut out),
    );
    assert!(
        result.is_err(),
        "f32 alpha with f64 operands must be rejected"
    );
}

#[test]
#[ignore]
fn test_accum_view_output_is_explicit_error() {
    // A borrowed view output over HOST memory must produce an explicit
    // backend error, never a silent fallback or hidden upload.
    let mut gpu = gpu_backend();
    let lhs = upload(&gpu, &tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let rhs = upload(&gpu, &tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let mut out_host = vec![0.0_f64; 4];
    let shape = [2usize, 2usize];
    let view = tenferro_tensor::TensorViewMut::f64(&shape, &mut out_host).unwrap();
    let result = gpu.dot_general_read_into_accum(
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(0.0),
        },
        TensorWrite::from_view(view),
    );
    assert!(result.is_err(), "view output must be an explicit error");
}

#[test]
#[ignore]
fn test_accum_zero_contraction_rejects_cpu_tensors() {
    // Even when k = 0 and beta = 1 makes the mathematical update a no-op,
    // the CUDA backend contract still requires GPU-resident owned tensors.
    let mut gpu = gpu_backend();
    let lhs = tensor_f64(vec![2, 0], vec![]);
    let rhs = tensor_f64(vec![0, 2], vec![]);
    let mut gpu_out = upload(&gpu, &tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let result = gpu.dot_general_read_into_accum(
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(1.0),
        },
        TensorWrite::from_tensor(&mut gpu_out),
    );
    assert!(result.is_err(), "CPU operands must be rejected");

    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let mut cpu_out = tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let result = gpu.dot_general_read_into_accum(
        TensorRead::from_tensor(&gpu_lhs),
        TensorRead::from_tensor(&gpu_rhs),
        &matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(1.0),
        },
        TensorWrite::from_tensor(&mut cpu_out),
    );
    assert!(result.is_err(), "CPU output must be rejected");
}

// --- Stage 2 (tensor4all/tenferro-rs#1287): borrowed views over device
// buffers on all three slots. Region layout metadata comes from
// `TypedTensor::backend_region_view{,_mut}`.

/// Read an element of a strided f64 region embedded in a flat host vector.
fn region_get(flat: &[f64], offset: usize, strides: &[usize], indices: &[usize]) -> f64 {
    let linear = offset
        + indices
            .iter()
            .zip(strides)
            .map(|(&index, &stride)| index * stride)
            .sum::<usize>();
    flat[linear]
}

/// Host reference for `out = alpha * lhs * rhs + beta * out` on strided
/// matrix regions of flat col-major buffers; updates `out_flat` in place.
#[allow(clippy::too_many_arguments)]
fn host_region_gemm_accum(
    lhs_flat: &[f64],
    lhs_off: usize,
    lhs_strides: &[usize],
    rhs_flat: &[f64],
    rhs_off: usize,
    rhs_strides: &[usize],
    out_flat: &mut [f64],
    out_off: usize,
    out_strides: &[usize],
    (rows, k, cols): (usize, usize, usize),
    alpha: f64,
    beta: f64,
) {
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = 0.0;
            for inner in 0..k {
                acc += region_get(lhs_flat, lhs_off, lhs_strides, &[row, inner])
                    * region_get(rhs_flat, rhs_off, rhs_strides, &[inner, col]);
            }
            let linear = out_off + row * out_strides[0] + col * out_strides[1];
            out_flat[linear] = alpha * acc + beta * out_flat[linear];
        }
    }
}

fn flat_f64(len: usize, seed: f64) -> Vec<f64> {
    (0..len).map(|i| seed + 0.25 * i as f64).collect()
}

fn download_f64(gpu: &crate::cubecl::CudaBackend, tensor: &Tensor) -> Vec<f64> {
    match download(gpu, tensor) {
        Tensor::F64(host) => host.as_slice().unwrap().to_vec(),
        other => panic!("expected f64 tensor, got {:?}", other.dtype()),
    }
}

#[test]
#[ignore]
fn test_accum_view_operands_offset_regions_f64() {
    // lhs: [2,3] region at offset 5 with leading dimension 4 in a flat len-32
    // device buffer; rhs: [3,2] region at offset 7, ld 3 (contiguous);
    // out: [2,2] region at offset 3, ld 4, in a flat len-20 device buffer.
    let mut gpu = gpu_backend();
    let lhs_host = flat_f64(32, -3.0);
    let rhs_host = flat_f64(32, 1.5);
    let out_host = flat_f64(20, -0.5);
    let (alpha, beta) = (2.0, -0.5);

    let mut expected = out_host.clone();
    host_region_gemm_accum(
        &lhs_host,
        5,
        &[1, 4],
        &rhs_host,
        7,
        &[1, 3],
        &mut expected,
        3,
        &[1, 4],
        (2, 3, 2),
        alpha,
        beta,
    );

    let lhs_gpu = upload(&gpu, &tensor_f64(vec![32], lhs_host));
    let rhs_gpu = upload(&gpu, &tensor_f64(vec![32], rhs_host));
    let mut out_gpu = upload(&gpu, &tensor_f64(vec![20], out_host));
    let (Tensor::F64(lhs_t), Tensor::F64(rhs_t)) = (&lhs_gpu, &rhs_gpu) else {
        unreachable!()
    };
    let Tensor::F64(out_t) = &mut out_gpu else {
        unreachable!()
    };
    let lhs_view = lhs_t
        .backend_region_view(vec![2, 3], vec![1, 4], 5)
        .unwrap();
    let rhs_view = rhs_t
        .backend_region_view(vec![3, 2], vec![1, 3], 7)
        .unwrap();
    let out_view = out_t
        .backend_region_view_mut(vec![2, 2], vec![1, 4], 3)
        .unwrap();
    gpu.dot_general_read_into_accum(
        TensorRead::from_view(TensorView::F64(lhs_view)),
        TensorRead::from_view(TensorView::F64(rhs_view)),
        &matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(alpha),
            beta: ContractionScalar::F64(beta),
        },
        TensorWrite::from_view(TensorViewMut::F64(out_view)),
    )
    .unwrap();

    // The updated region matches the host reference and every element outside
    // the region is preserved bit-for-bit.
    let actual = download_f64(&gpu, &out_gpu);
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "flat index {index}: {actual} != {expected}"
        );
    }
}

#[test]
#[ignore]
fn test_accum_block_diagonal_regions_of_one_buffer_f64() {
    // Two successive accumulations into disjoint diagonal blocks of ONE flat
    // device buffer holding a col-major [4,4] matrix; off-block elements and
    // the beta-scaled block contents must both be exact.
    let mut gpu = gpu_backend();
    let out_host = flat_f64(16, 10.0);
    let lhs_a = flat_f64(4, -1.0);
    let rhs_a = flat_f64(4, 0.5);
    let lhs_b = flat_f64(4, 2.0);
    let rhs_b = flat_f64(4, -0.75);
    let (alpha, beta) = (1.5, 1.0);

    let mut expected = out_host.clone();
    // Block (0..2)x(0..2): offset 0. Block (2..4)x(2..4): offset 2 + 2*4 = 10.
    host_region_gemm_accum(
        &lhs_a,
        0,
        &[1, 2],
        &rhs_a,
        0,
        &[1, 2],
        &mut expected,
        0,
        &[1, 4],
        (2, 2, 2),
        alpha,
        beta,
    );
    host_region_gemm_accum(
        &lhs_b,
        0,
        &[1, 2],
        &rhs_b,
        0,
        &[1, 2],
        &mut expected,
        10,
        &[1, 4],
        (2, 2, 2),
        alpha,
        beta,
    );

    let lhs_a_gpu = upload(&gpu, &tensor_f64(vec![2, 2], lhs_a));
    let rhs_a_gpu = upload(&gpu, &tensor_f64(vec![2, 2], rhs_a));
    let lhs_b_gpu = upload(&gpu, &tensor_f64(vec![2, 2], lhs_b));
    let rhs_b_gpu = upload(&gpu, &tensor_f64(vec![2, 2], rhs_b));
    let mut out_gpu = upload(&gpu, &tensor_f64(vec![16], out_host));

    for (lhs, rhs, offset) in [(&lhs_a_gpu, &rhs_a_gpu, 0), (&lhs_b_gpu, &rhs_b_gpu, 10)] {
        let Tensor::F64(out_t) = &mut out_gpu else {
            unreachable!()
        };
        let out_view = out_t
            .backend_region_view_mut(vec![2, 2], vec![1, 4], offset)
            .unwrap();
        gpu.dot_general_read_into_accum(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            &matmul_config(),
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F64(alpha),
                beta: ContractionScalar::F64(beta),
            },
            TensorWrite::from_view(TensorViewMut::F64(out_view)),
        )
        .unwrap();
    }

    let actual = download_f64(&gpu, &out_gpu);
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "flat index {index}: {actual} != {expected}"
        );
    }
}

#[test]
#[ignore]
fn test_accum_host_view_operand_rejected() {
    // A borrowed view over HOST memory must be an explicit backend error on
    // the CUDA accumulation path (no hidden upload).
    let mut gpu = gpu_backend();
    let lhs_host = [1.0_f64, 2.0, 3.0, 4.0];
    let lhs_view = TypedTensorView::from_col_major(&[2, 2], &lhs_host).unwrap();
    let rhs = upload(&gpu, &tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let mut out = upload(&gpu, &tensor_f64(vec![2, 2], vec![0.0; 4]));
    let result = gpu.dot_general_read_into_accum(
        TensorRead::from_view(TensorView::F64(lhs_view)),
        TensorRead::from_tensor(&rhs),
        &matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(0.0),
        },
        TensorWrite::from_tensor(&mut out),
    );
    assert!(result.is_err(), "host view operand must be rejected");
}

#[test]
#[ignore]
fn test_accum_negative_stride_view_rejected() {
    // Reversed device regions are valid tensor views but outside the cuTENSOR
    // nonnegative-stride contract: explicit error, no silent canonicalization.
    let mut gpu = gpu_backend();
    let lhs_gpu = upload(&gpu, &tensor_f64(vec![8], flat_f64(8, 0.0)));
    let Tensor::F64(lhs_t) = &lhs_gpu else {
        unreachable!()
    };
    let lhs_view = lhs_t
        .backend_region_view(vec![2, 2], vec![1, -2], 2)
        .unwrap();
    let rhs = upload(&gpu, &tensor_f64(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let mut out = upload(&gpu, &tensor_f64(vec![2, 2], vec![0.0; 4]));
    let result = gpu.dot_general_read_into_accum(
        TensorRead::from_view(TensorView::F64(lhs_view)),
        TensorRead::from_tensor(&rhs),
        &matmul_config(),
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(1.0),
            beta: ContractionScalar::F64(0.0),
        },
        TensorWrite::from_tensor(&mut out),
    );
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("stride"),
        "negative view stride must be an explicit stride error, got: {err}"
    );
}

#[test]
#[ignore]
fn test_accum_zero_contraction_view_output_beta_error() {
    // k = 0 with a view output: beta == 1 is a validated no-op; any other
    // beta needs a strided scale kernel that does not exist yet, so it must
    // be an explicit error (documented stage-2 limitation).
    let mut gpu = gpu_backend();
    let lhs = upload(&gpu, &tensor_f64(vec![2, 0], vec![]));
    let rhs = upload(&gpu, &tensor_f64(vec![0, 2], vec![]));
    let out_host = flat_f64(8, 1.0);
    let mut out_gpu = upload(&gpu, &tensor_f64(vec![8], out_host.clone()));

    for (beta, expect_ok) in [(1.0, true), (-2.0, false)] {
        let Tensor::F64(out_t) = &mut out_gpu else {
            unreachable!()
        };
        let out_view = out_t
            .backend_region_view_mut(vec![2, 2], vec![1, 4], 1)
            .unwrap();
        let result = gpu.dot_general_read_into_accum(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &matmul_config(),
            DotGeneralAccumulation {
                lhs_conj: false,
                rhs_conj: false,
                alpha: ContractionScalar::F64(1.0),
                beta: ContractionScalar::F64(beta),
            },
            TensorWrite::from_view(TensorViewMut::F64(out_view)),
        );
        assert_eq!(result.is_ok(), expect_ok, "beta = {beta}: {result:?}");
    }

    // Both calls must leave the buffer untouched (no-op or rejected).
    let actual = download_f64(&gpu, &out_gpu);
    assert_eq!(actual, out_host);
}
