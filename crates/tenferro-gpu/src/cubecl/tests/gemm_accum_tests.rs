// CUDA-native dot-general accumulation (tensor4all/tenferro-rs#1287).
// Run with: cargo test --features cuda -- --ignored
use num_complex::Complex64;

use crate::DotGeneralConfig;
use crate::Tensor;
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, TensorDot, TensorRead, TensorWrite,
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

    let mut expected = out_init.clone();
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
    // Stage 1 supports compact owned tensors only; a borrowed view output
    // must produce an explicit backend error, never a silent fallback.
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
