// Run with: cargo test --features cuda -- --ignored
use num_complex::{Complex32, Complex64};
use std::num::NonZeroUsize;

use crate::DotGeneralConfig;
use crate::Tensor;
use tenferro_tensor::TensorDot;

use super::super::gemm::cutensor_plan_cache_workspace_bytes;
use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32, tensor_c64, tensor_f32,
    tensor_f64, upload,
};

fn run_dot_general_case(lhs: Tensor, rhs: Tensor, config: DotGeneralConfig, tol: f64) {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let expected = cpu.dot_general(&lhs, &rhs, &config).unwrap();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let actual_gpu = gpu.dot_general(&gpu_lhs, &gpu_rhs, &config).unwrap();
    let actual = download(&gpu, &actual_gpu);

    assert_eq!(actual.shape(), expected.shape());
    assert_tensor_close(&actual, &expected, tol);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_cutensor_cache_eviction_keeps_inflight_workspace_valid() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    gpu.set_cutensor_plan_cache_max_entries(NonZeroUsize::new(1).unwrap())
        .unwrap();

    let lhs_a = tensor_f32(vec![64, 64], vec![1.0; 64 * 64]);
    let rhs_a = tensor_f32(vec![64, 64], vec![1.0; 64 * 64]);
    let lhs_b = tensor_f32(vec![65, 64], vec![1.0; 65 * 64]);
    let rhs_b = tensor_f32(vec![64, 65], vec![1.0; 64 * 65]);
    let expected_a = cpu.dot_general(&lhs_a, &rhs_a, &matmul_config()).unwrap();
    let expected_b = cpu.dot_general(&lhs_b, &rhs_b, &matmul_config()).unwrap();

    let gpu_lhs_a = upload(&gpu, &lhs_a);
    let gpu_rhs_a = upload(&gpu, &rhs_a);
    let gpu_lhs_b = upload(&gpu, &lhs_b);
    let gpu_rhs_b = upload(&gpu, &rhs_b);
    let actual_a = gpu
        .dot_general(&gpu_lhs_a, &gpu_rhs_a, &matmul_config())
        .unwrap();
    assert!(
        cutensor_plan_cache_workspace_bytes(&gpu).unwrap() > 0,
        "first contraction must retain a nonzero cuTENSOR workspace"
    );

    let actual_b = gpu
        .dot_general(&gpu_lhs_b, &gpu_rhs_b, &matmul_config())
        .unwrap();
    let cache_stats = gpu.cutensor_plan_cache_stats().unwrap();
    assert_eq!(cache_stats.entries, 1);
    assert!(
        cache_stats.evictions > 0,
        "second contraction must evict the first plan"
    );

    // Eviction retires the workspace's owning stream before releasing its
    // CubeCL allocation handle, so the first queued launch remains valid.
    gpu.runtime().synchronize().unwrap();
    assert_tensor_close(&download(&gpu, &actual_a), &expected_a, 1e-4);
    assert_tensor_close(&download(&gpu, &actual_b), &expected_b, 1e-4);
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

#[test]
#[ignore]
fn test_dot_general_matmul_f32() {
    run_dot_general_case(
        tensor_f32(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        tensor_f32(
            vec![3, 4],
            vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ],
        ),
        matmul_config(),
        1e-4,
    );
}

#[test]
#[ignore]
fn test_dot_general_matmul_f64() {
    run_dot_general_case(
        tensor_f64(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        tensor_f64(
            vec![3, 4],
            vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ],
        ),
        matmul_config(),
        1e-9,
    );
}

#[test]
#[ignore]
fn test_dot_general_batched_matmul_f32() {
    run_dot_general_case(
        tensor_f32(
            vec![2, 3, 5],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            ],
        ),
        tensor_f32(
            vec![3, 4, 5],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                10.0, 11.0, 12.0, 13.0, 14.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
        ),
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
        },
        1e-4,
    );
}

#[test]
#[ignore]
fn test_dot_general_complex_matmul_c32() {
    run_dot_general_case(
        tensor_c32(
            vec![2, 3],
            vec![
                Complex32::new(1.0, 0.5),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 0.25),
                Complex32::new(4.0, 1.0),
                Complex32::new(5.0, -0.5),
                Complex32::new(6.0, 0.75),
            ],
        ),
        tensor_c32(
            vec![3, 4],
            vec![
                Complex32::new(1.0, -1.0),
                Complex32::new(2.0, 0.25),
                Complex32::new(3.0, 0.5),
                Complex32::new(4.0, -0.75),
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, 1.0),
                Complex32::new(7.0, -0.5),
                Complex32::new(8.0, 0.75),
                Complex32::new(9.0, -1.25),
                Complex32::new(10.0, 0.5),
                Complex32::new(11.0, 1.5),
                Complex32::new(12.0, -0.25),
            ],
        ),
        matmul_config(),
        1e-4,
    );
}

#[test]
#[ignore]
fn test_dot_general_complex_matmul_c64() {
    run_dot_general_case(
        tensor_c64(
            vec![2, 3],
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.25),
                Complex64::new(4.0, 1.0),
                Complex64::new(5.0, -0.5),
                Complex64::new(6.0, 0.75),
            ],
        ),
        tensor_c64(
            vec![3, 4],
            vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, 0.25),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, -0.75),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 1.0),
                Complex64::new(7.0, -0.5),
                Complex64::new(8.0, 0.75),
                Complex64::new(9.0, -1.25),
                Complex64::new(10.0, 0.5),
                Complex64::new(11.0, 1.5),
                Complex64::new(12.0, -0.25),
            ],
        ),
        matmul_config(),
        1e-9,
    );
}
