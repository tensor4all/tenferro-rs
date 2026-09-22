// Run with: cargo test --features cuda -- --ignored
use num_complex::{Complex32, Complex64};
use std::num::NonZeroUsize;

use crate::cuda::CutensorWorkspaceStats;
use crate::DotGeneralConfig;
use crate::Tensor;
use tenferro_tensor::TensorDot;

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32, tensor_c64, tensor_f32,
    tensor_f64, upload, CpuBackend, CudaBackend,
};

/// Run one `rows x 64` by `64 x 63` f64 matmul on both backends.
fn compare_rows_matmul(
    cpu: &mut CpuBackend,
    gpu: &mut CudaBackend,
    rows: usize,
) -> (Tensor, Tensor) {
    let lhs = tensor_f64(
        vec![rows, 64],
        (0..rows * 64)
            .map(|i| (i % 13) as f64 * 0.1 - 0.7)
            .collect(),
    );
    let rhs = tensor_f64(
        vec![64, 63],
        (0..64 * 63).map(|i| (i % 7) as f64 * 0.2 - 0.3).collect(),
    );
    let expected = cpu.dot_general(&lhs, &rhs, &matmul_config()).unwrap();
    let lhs_gpu = upload(gpu, &lhs);
    let rhs_gpu = upload(gpu, &rhs);
    let actual = gpu
        .dot_general(&lhs_gpu, &rhs_gpu, &matmul_config())
        .unwrap();
    (actual, expected)
}

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
        gpu.cutensor_workspace_bytes().unwrap() > 0,
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

    // Eviction keeps shared scratch alive; same-stream execution orders reuse
    // after the first launch without requiring a barrier at each eviction.
    gpu.runtime().synchronize().unwrap();
    assert_tensor_close(&download(&gpu, &actual_a), &expected_a, 1e-4);
    assert_tensor_close(&download(&gpu, &actual_b), &expected_b, 1e-4);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_cutensor_shared_workspace_does_not_evict_plan_cache_by_bytes() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    gpu.set_cutensor_plan_cache_max_entries(NonZeroUsize::new(2).unwrap())
        .unwrap();
    // Enough for two plans, but less than the 1 MiB shared-workspace floor.
    gpu.set_cuda_extension_cache_max_retained_bytes(NonZeroUsize::new(64 * 1024).unwrap())
        .unwrap();
    let mut results = Vec::new();
    for rows in [64, 65, 64] {
        results.push(compare_rows_matmul(&mut cpu, &mut gpu, rows));
    }
    let stats = gpu.cutensor_plan_cache_stats().unwrap();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits, stats.evictions),
        (2, 2, 1, 0)
    );
    assert_eq!(gpu.cutensor_plan_cache_max_entries().unwrap().get(), 2);
    assert!(gpu.cutensor_workspace_bytes().unwrap() >= 1 << 20);
    assert!(gpu.cuda_extension_cache_stats().unwrap().retained_bytes < 64 * 1024);
    gpu.clear_cuda_extension_cache().unwrap();
    assert_eq!(gpu.cutensor_workspace_bytes().unwrap(), 0);
    for (actual, expected) in results {
        assert_tensor_close(&download(&gpu, &actual), &expected, 1e-9);
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_cutensor_default_retention_cap_survives_cache_clear() {
    let gpu = gpu_backend();
    assert_eq!(
        gpu.cutensor_workspace_max_retained_bytes(),
        1 << 30,
        "the default retention cap is 1 GiB"
    );
    gpu.set_cutensor_workspace_max_retained_bytes(3 * 1024 * 1024)
        .unwrap();
    gpu.clear_cuda_extension_cache().unwrap();
    assert_eq!(
        gpu.cutensor_workspace_max_retained_bytes(),
        3 * 1024 * 1024,
        "the cap is backend-level state and must survive a cache clear"
    );
    assert_eq!(
        gpu.cutensor_workspace_stats().unwrap(),
        CutensorWorkspaceStats::default()
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_cutensor_zero_retention_cap_runs_without_retaining_scratch() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    gpu.set_cutensor_workspace_max_retained_bytes(0).unwrap();

    let mut results = Vec::new();
    for rows in [64, 65, 64] {
        results.push(compare_rows_matmul(&mut cpu, &mut gpu, rows));
    }
    assert_eq!(
        gpu.cutensor_workspace_stats().unwrap(),
        CutensorWorkspaceStats::default(),
        "a zero cap must retain no scratch"
    );
    let stats = gpu.cutensor_plan_cache_stats().unwrap();
    assert_eq!(
        (stats.misses, stats.hits),
        (2, 1),
        "plan reuse must survive a zero retention cap"
    );
    for (actual, expected) in results {
        assert_tensor_close(&download(&gpu, &actual), &expected, 1e-9);
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_cutensor_small_retention_cap_is_respected_and_keeps_plans() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    gpu.set_cutensor_plan_cache_max_entries(NonZeroUsize::new(2).unwrap())
        .unwrap();
    gpu.set_cutensor_workspace_max_retained_bytes(1 << 20)
        .unwrap();

    let mut results = Vec::new();
    for rows in [64, 65, 64] {
        results.push(compare_rows_matmul(&mut cpu, &mut gpu, rows));
    }
    let workspace = gpu.cutensor_workspace_stats().unwrap();
    assert!(
        workspace.retained_bytes <= 1 << 20,
        "retained scratch must respect the cap: {workspace:?}"
    );
    let stats = gpu.cutensor_plan_cache_stats().unwrap();
    assert_eq!(
        (stats.entries, stats.misses, stats.hits, stats.evictions),
        (2, 2, 1, 0)
    );
    for (actual, expected) in results {
        assert_tensor_close(&download(&gpu, &actual), &expected, 1e-9);
    }
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_cutensor_shrinking_retention_cap_releases_scratch_and_keeps_plans() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let (actual, expected) = compare_rows_matmul(&mut cpu, &mut gpu, 64);
    assert_tensor_close(&download(&gpu, &actual), &expected, 1e-9);
    assert!(
        gpu.cutensor_workspace_stats().unwrap().retained_bytes > 0,
        "the first contraction is expected to retain scratch"
    );
    let before = gpu.cutensor_plan_cache_stats().unwrap();

    gpu.set_cutensor_workspace_max_retained_bytes(0).unwrap();
    assert_eq!(
        gpu.cutensor_workspace_stats().unwrap(),
        CutensorWorkspaceStats::default(),
        "shrinking the cap must release retained scratch"
    );
    let after = gpu.cutensor_plan_cache_stats().unwrap();
    assert_eq!(
        (after.entries, after.evictions),
        (before.entries, before.evictions),
        "a cap change must not evict plans"
    );

    // The plan stays reusable with retention disabled.
    let (actual, expected) = compare_rows_matmul(&mut cpu, &mut gpu, 64);
    assert_tensor_close(&download(&gpu, &actual), &expected, 1e-9);
    assert_eq!(
        gpu.cutensor_plan_cache_stats().unwrap().hits,
        before.hits + 1
    );
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
