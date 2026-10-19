// Run with: cargo test --features cuda -- --ignored
use num_complex::{Complex32, Complex64};
use std::num::NonZeroUsize;

use crate::cuda::CutensorWorkspaceStats;
use crate::DotGeneralConfig;
use crate::Tensor;

use super::super::batched_gemm::{
    batched_gemm_launches_for_test, reset_batched_gemm_launches_for_test,
};
use super::super::cublaslt::{lt_gemm_launches_for_test, reset_lt_gemm_launches_for_test};
use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32, tensor_c64, tensor_f32,
    tensor_f64, upload, CpuBackend, CudaBackend,
};
use tenferro_tensor::BackendSessionHost;
use tenferro_tensor::{TensorRead, TensorView};

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
    let expected = cpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &matmul_config(),
            )
        })
        .unwrap()
        .unwrap();
    let lhs_gpu = upload(gpu, &lhs);
    let rhs_gpu = upload(gpu, &rhs);
    let actual = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs_gpu),
                TensorRead::from_tensor(&rhs_gpu),
                &matmul_config(),
            )
        })
        .unwrap()
        .unwrap();
    (actual, expected)
}

fn run_dot_general_case(lhs: Tensor, rhs: Tensor, config: DotGeneralConfig, tol: f64) {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let expected = cpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let actual_gpu = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&gpu_lhs),
                TensorRead::from_tensor(&gpu_rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();
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
    let expected_a = cpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs_a),
                TensorRead::from_tensor(&rhs_a),
                &matmul_config(),
            )
        })
        .unwrap()
        .unwrap();
    let expected_b = cpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs_b),
                TensorRead::from_tensor(&rhs_b),
                &matmul_config(),
            )
        })
        .unwrap()
        .unwrap();

    let gpu_lhs_a = upload(&gpu, &lhs_a);
    let gpu_rhs_a = upload(&gpu, &rhs_a);
    let gpu_lhs_b = upload(&gpu, &lhs_b);
    let gpu_rhs_b = upload(&gpu, &rhs_b);
    let actual_a = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&gpu_lhs_a),
                TensorRead::from_tensor(&gpu_rhs_a),
                &matmul_config(),
            )
        })
        .unwrap()
        .unwrap();
    assert!(
        gpu.cutensor_workspace_bytes().unwrap() > 0,
        "first contraction must retain a nonzero cuTENSOR workspace"
    );

    let actual_b = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&gpu_lhs_b),
                TensorRead::from_tensor(&gpu_rhs_b),
                &matmul_config(),
            )
        })
        .unwrap()
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
    assert_eq!(
        gpu.cutensor_workspace_temporary_uses(),
        0,
        "the default cap must not be binding for these shapes"
    );
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
        10 << 30,
        "the default retention cap is 10 GiB"
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
    assert_eq!(
        gpu.cutensor_workspace_temporary_uses(),
        3,
        "a zero cap must route every scratch-requiring contraction to a temporary workspace"
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
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
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
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [2].as_slice().into(),
            rhs_batch_dims: [2].as_slice().into(),
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

/// The plan path depends on the effective `(pointer, strides, alignment)`, not
/// on the operand's representation: an owned operand and a view over the same
/// region share one cuTENSOR plan (#2110).
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_cutensor_plan_is_shared_by_an_owned_and_an_equivalent_view_operand() {
    let mut gpu = gpu_backend();
    let lhs = tensor_f64(
        vec![8, 64],
        (0..8 * 64).map(|i| (i % 11) as f64 * 0.1 - 0.4).collect(),
    );
    let rhs = tensor_f64(
        vec![64, 63],
        (0..64 * 63).map(|i| (i % 7) as f64 * 0.2 - 0.3).collect(),
    );
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);

    let owned = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&gpu_lhs),
                TensorRead::from_tensor(&gpu_rhs),
                &matmul_config(),
            )
        })
        .unwrap()
        .unwrap();
    let after_owned = gpu.cutensor_plan_cache_stats().unwrap();
    assert_eq!(after_owned.entries, 1);

    // The view covers the whole region with the same effective strides the
    // owned operand is described with (the layout's compact column-major
    // strides), so only the representation differs.
    let typed = gpu_lhs.as_typed::<f64>().expect("f64 lhs");
    let region = typed
        .backend_region_view(vec![8, 64], vec![1, 8], 0)
        .unwrap();
    let viewed = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_view(TensorView::F64(region)),
                TensorRead::from_tensor(&gpu_rhs),
                &matmul_config(),
            )
        })
        .unwrap()
        .unwrap();

    let after_view = gpu.cutensor_plan_cache_stats().unwrap();
    assert_eq!(
        after_view.entries, 1,
        "an equivalent view must share the owned operand's plan"
    );
    assert!(
        after_view.hits > after_owned.hits,
        "the view must hit the owned plan"
    );
    assert_tensor_close(&download(&gpu, &viewed), &download(&gpu, &owned), 1e-9);
}

/// Deterministic values, so a mismatch is reproducible and the two operands see
/// different patterns.
fn ramp(shape: &[usize], scale: f64) -> Vec<f64> {
    let len: usize = shape.iter().product();
    (0..len)
        .map(|i| (i * 7 % 23) as f64 * scale - 0.5)
        .collect()
}

fn as_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&value| value as f32).collect()
}

fn as_c32(values: &[f64]) -> Vec<Complex32> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| Complex32::new(value as f32, (i % 5) as f32 * 0.25 - 0.5))
        .collect()
}

fn as_c64(values: &[f64]) -> Vec<Complex64> {
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| Complex64::new(value, (i % 5) as f64 * 0.25 - 0.5))
        .collect()
}

/// Fill a compact column-major tensor of `shape` from a linear index.
fn fill(shape: &[usize], value: impl Fn(usize) -> f64) -> Vec<f64> {
    (0..shape.iter().product::<usize>()).map(value).collect()
}

/// `lhs[m, k, batch] * rhs[k, n, batch] -> out[m, n, batch]`: the standard
/// batched matmul whose batch axes are the trailing ones on both sides.
fn batched_matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    }
}

/// Which contraction path a case must take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GemmPath {
    Cutensor,
    Strided,
    CublasLt,
}

impl GemmPath {
    fn of(strided: usize, lt: usize) -> Self {
        match (strided, lt) {
            (0, 0) => Self::Cutensor,
            (1, 0) => Self::Strided,
            (0, 1) => Self::CublasLt,
            _ => panic!("unexpected launch counts: {strided} strided, {lt} cuBLASLt"),
        }
    }
}

/// Run one case against the CPU reference and assert which path served it.
///
/// One launch on the expected path proves the fast path is taken; a cuTENSOR
/// expectation proves the case fell back instead of matching only numerically.
fn assert_case_launches(
    lhs: Tensor,
    rhs: Tensor,
    config: DotGeneralConfig,
    tol: f64,
    expected: GemmPath,
) {
    reset_batched_gemm_launches_for_test();
    reset_lt_gemm_launches_for_test();
    run_dot_general_case(lhs, rhs, config, tol);
    assert_eq!(
        GemmPath::of(
            batched_gemm_launches_for_test(),
            lt_gemm_launches_for_test()
        ),
        expected,
        "the case must be served by the expected contraction path"
    );
}

/// Every supported dtype takes the cuBLASLt batched GEMM for a small standard
/// batched matmul.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_standard_batched_matmul_uses_one_cublaslt_gemm() {
    let lhs_shape = vec![8, 5, 3];
    let rhs_shape = vec![5, 7, 3];
    let lhs = ramp(&lhs_shape, 0.1);
    let rhs = ramp(&rhs_shape, -0.05);
    // The f32/complex32 tolerances are tight enough that a compute type
    // drifting to TF32 (~1e-3 here) would fail.
    let cases = [
        (
            tensor_f32(lhs_shape.clone(), as_f32(&lhs)),
            tensor_f32(rhs_shape.clone(), as_f32(&rhs)),
            1e-5,
        ),
        (
            tensor_f64(lhs_shape.clone(), lhs.clone()),
            tensor_f64(rhs_shape.clone(), rhs.clone()),
            1e-12,
        ),
        (
            tensor_c32(lhs_shape.clone(), as_c32(&lhs)),
            // Tight enough that a compute type drifting to TF32 would fail.
            tensor_c32(rhs_shape.clone(), as_c32(&rhs)),
            1e-5,
        ),
        (
            tensor_c64(lhs_shape.clone(), as_c64(&lhs)),
            tensor_c64(rhs_shape.clone(), as_c64(&rhs)),
            1e-12,
        ),
    ];
    for (lhs, rhs, tol) in cases {
        assert_case_launches(lhs, rhs, batched_matmul_config(), tol, GemmPath::CublasLt);
    }
}

/// A two-matrix rectangular product takes the cuBLASLt path at both the low
/// end of the batch range and a non-square matrix shape.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_standard_batched_matmul_covers_batch_two_and_rectangular_slices() {
    let cases = [
        (vec![5, 3, 2], vec![3, 7, 2]),
        (vec![3, 5, 2], vec![7, 3, 2]),
    ];
    for (lhs_shape, rhs_shape) in cases {
        let lhs_contracting = if lhs_shape[0] == 3 { 0 } else { 1 };
        let rhs_contracting = if rhs_shape[0] == 7 { 1 } else { 0 };
        let config = DotGeneralConfig {
            lhs_contracting_dims: [lhs_contracting].as_slice().into(),
            rhs_contracting_dims: [rhs_contracting].as_slice().into(),
            lhs_batch_dims: [2].as_slice().into(),
            rhs_batch_dims: [2].as_slice().into(),
        };
        assert_case_launches(
            tensor_f64(lhs_shape.clone(), ramp(&lhs_shape, 0.1)),
            tensor_f64(rhs_shape.clone(), ramp(&rhs_shape, -0.05)),
            config,
            1e-12,
            GemmPath::CublasLt,
        );
    }
}

/// A matrix extent above the cuBLASLt threshold keeps the cuBLAS strided-batched
/// path, so the threshold does not silently capture the large batched case.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_batched_matmul_above_the_small_threshold_stays_strided() {
    let (m, n, k, batch) = (65usize, 63usize, 67usize, 2usize);
    let lhs_shape = vec![m, k, batch];
    let rhs_shape = vec![k, n, batch];
    assert_case_launches(
        tensor_f64(lhs_shape.clone(), ramp(&lhs_shape, 0.1)),
        tensor_f64(rhs_shape.clone(), ramp(&rhs_shape, -0.05)),
        batched_matmul_config(),
        1e-11,
        GemmPath::Strided,
    );
}

/// Both transpose combinations per operand are expressed with `opA`/`opB`, so
/// none of them costs a transpose copy or leaves the fast path.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_standard_batched_matmul_covers_both_transpose_combinations() {
    // (lhs shape, rhs shape, lhs contracting axis, rhs contracting axis):
    // lhs `[m, k, b]`/`[k, m, b]` and rhs `[k, n, b]`/`[n, k, b]`.
    let cases = [
        (vec![5, 3, 4], vec![3, 7, 4], 1, 0),
        (vec![3, 5, 4], vec![3, 7, 4], 0, 0),
        (vec![5, 3, 4], vec![7, 3, 4], 1, 1),
        (vec![3, 5, 4], vec![7, 3, 4], 0, 1),
    ];
    for (lhs_shape, rhs_shape, lhs_contracting, rhs_contracting) in cases {
        let config = DotGeneralConfig {
            lhs_contracting_dims: [lhs_contracting].as_slice().into(),
            rhs_contracting_dims: [rhs_contracting].as_slice().into(),
            lhs_batch_dims: [2].as_slice().into(),
            rhs_batch_dims: [2].as_slice().into(),
        };
        assert_case_launches(
            tensor_f64(lhs_shape.clone(), ramp(&lhs_shape, 0.1)),
            tensor_f64(rhs_shape.clone(), ramp(&rhs_shape, -0.05)),
            config,
            1e-12,
            GemmPath::CublasLt,
        );
    }

    // The two non-real transposes as complex64: a stray `OP_T -> OP_C` would
    // pass every real case above while conjugating these.
    let complex_cases = [
        (vec![3, 5, 4], vec![3, 7, 4], 0, 0),
        (vec![5, 3, 4], vec![7, 3, 4], 1, 1),
    ];
    for (lhs_shape, rhs_shape, lhs_contracting, rhs_contracting) in complex_cases {
        let config = DotGeneralConfig {
            lhs_contracting_dims: [lhs_contracting].as_slice().into(),
            rhs_contracting_dims: [rhs_contracting].as_slice().into(),
            lhs_batch_dims: [2].as_slice().into(),
            rhs_batch_dims: [2].as_slice().into(),
        };
        assert_case_launches(
            tensor_c64(lhs_shape.clone(), as_c64(&ramp(&lhs_shape, 0.1))),
            tensor_c64(rhs_shape.clone(), as_c64(&ramp(&rhs_shape, -0.05))),
            config,
            1e-12,
            GemmPath::CublasLt,
        );
    }
}

/// A rank-four operand pair with two trailing batch axes stays on the fast
/// path: one batch stride covers the flattened batch index.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_standard_batched_matmul_flattens_two_batch_axes() {
    let lhs_shape = vec![6, 3, 2, 4];
    let rhs_shape = vec![3, 5, 2, 4];
    assert_case_launches(
        tensor_f64(lhs_shape.clone(), ramp(&lhs_shape, 0.1)),
        tensor_f64(rhs_shape.clone(), ramp(&rhs_shape, -0.05)),
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [2, 3].as_slice().into(),
            rhs_batch_dims: [2, 3].as_slice().into(),
        },
        1e-12,
        GemmPath::CublasLt,
    );
}

/// The `16 x 32 x 32` f64 batched product in the two physical layouts the
/// contraction paths accept.
///
/// Both layouts are filled from the same logical `(row, inner, batch)`
/// function, so the two paths contract identical matrices and only the
/// physical axis order differs: `lhs_standard`/`rhs_standard` are the trailing
/// batch layout the cuBLAS/cuBLASLt fast path detects, and
/// `lhs_outer`/`rhs_outer` are the batch-outer layout only cuTENSOR serves.
struct Batch16Layouts {
    lhs_standard: Tensor,
    rhs_standard: Tensor,
    lhs_outer: Tensor,
    rhs_outer: Tensor,
}

fn batch_16_layouts() -> Batch16Layouts {
    let (m, n, k, batch) = (32usize, 32usize, 32usize, 16usize);
    let lhs_value = |row: usize, inner: usize, b: usize| {
        ((row * 7 + inner * 11 + b * 3) % 23) as f64 * 0.01 - 0.5
    };
    let rhs_value = |inner: usize, col: usize, b: usize| {
        ((inner * 5 + col * 13 + b * 17) % 23) as f64 * -0.005 - 0.5
    };
    // Column-major `[m, k, batch]`: `i = row + inner * m + b * m * k`.
    let lhs_standard = fill(&[m, k, batch], |i| {
        lhs_value(i % m, (i / m) % k, i / (m * k))
    });
    let rhs_standard = fill(&[k, n, batch], |i| {
        rhs_value(i % k, (i / k) % n, i / (k * n))
    });
    // Column-major `[batch, m, k]`: `i = b + row * batch + inner * batch * m`.
    let lhs_outer = fill(&[batch, m, k], |i| {
        lhs_value((i / batch) % m, i / (batch * m), i % batch)
    });
    let rhs_outer = fill(&[batch, k, n], |i| {
        rhs_value((i / batch) % k, i / (batch * k), i % batch)
    });
    Batch16Layouts {
        lhs_standard: tensor_f64(vec![m, k, batch], lhs_standard),
        rhs_standard: tensor_f64(vec![k, n, batch], rhs_standard),
        lhs_outer: tensor_f64(vec![batch, m, k], lhs_outer),
        rhs_outer: tensor_f64(vec![batch, k, n], rhs_outer),
    }
}

fn cpu_dot_general(lhs: &Tensor, rhs: &Tensor, config: &DotGeneralConfig) -> Tensor {
    let mut cpu = cpu_backend();
    cpu.with_backend_session(|__s| {
        __s.dot_general_read(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            config,
        )
    })
    .unwrap()
    .unwrap()
}

/// The benchmark shape (16 batches of 32x32) takes the fast path and agrees
/// with both the CPU reference and the cuTENSOR contraction of the same
/// algebra, reached through the batch-outer layout cuTENSOR still serves.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_standard_batched_matmul_batch_16_matches_the_cutensor_path() {
    let layouts = batch_16_layouts();
    let config = batched_matmul_config();
    // `lhs_outer` is `[batch, m, k]`, so its contraction axis is 2; `rhs_outer`
    // is `[batch, k, n]`, so its contraction axis is 1.
    let outer_config = DotGeneralConfig {
        lhs_contracting_dims: [2].as_slice().into(),
        rhs_contracting_dims: [1].as_slice().into(),
        lhs_batch_dims: [0].as_slice().into(),
        rhs_batch_dims: [0].as_slice().into(),
    };
    let expected = cpu_dot_general(&layouts.lhs_standard, &layouts.rhs_standard, &config);

    let mut gpu = gpu_backend();
    let lhs = upload(&gpu, &layouts.lhs_standard);
    let rhs = upload(&gpu, &layouts.rhs_standard);
    let lhs_outer = upload(&gpu, &layouts.lhs_outer);
    let rhs_outer = upload(&gpu, &layouts.rhs_outer);

    reset_batched_gemm_launches_for_test();
    reset_lt_gemm_launches_for_test();
    let fast = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        lt_gemm_launches_for_test(),
        1,
        "the 16-batch 32x32 case must be one cuBLASLt batched GEMM"
    );
    assert_eq!(
        batched_gemm_launches_for_test(),
        0,
        "the cuBLAS strided-batched path must not also run"
    );

    reset_batched_gemm_launches_for_test();
    reset_lt_gemm_launches_for_test();
    let general = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs_outer),
                TensorRead::from_tensor(&rhs_outer),
                &outer_config,
            )
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        (
            batched_gemm_launches_for_test(),
            lt_gemm_launches_for_test()
        ),
        (0, 0),
        "the batch-outer form must not take either batched-GEMM fast path"
    );

    let fast = download(&gpu, &fast);
    let general = download(&gpu, &general);
    assert_eq!(fast.shape(), expected.shape());
    assert_eq!(general.shape(), expected.shape());
    assert_tensor_close(&fast, &expected, 1e-9);
    assert_tensor_close(&general, &expected, 1e-9);
}

/// A100 in-process timing of the same logical `16 x 32 x 32` f64 batched
/// product through the cuBLASLt small-batch path and the cuTENSOR fallback.
///
/// The external benchmark harness cannot compile against this tree, so this
/// measures both backends through the public `dot_general_read` surface with a
/// host-side timer and an explicit device synchronization, exactly like the
/// existing GPU benchmarks. Run with:
/// `cargo test -p tenferro-gpu --features cuda --lib -- --ignored --nocapture
/// cuda_batched_matmul_16_a100_fast_path_timing`.
#[test]
#[ignore = "requires CUDA 12.8+ GPU; prints A100 in-process timings"]
fn cuda_batched_matmul_16_a100_fast_path_timing() {
    use std::hint::black_box;
    use std::time::Instant;

    fn measure(
        gpu: &mut CudaBackend,
        lhs: &Tensor,
        rhs: &Tensor,
        config: &DotGeneralConfig,
    ) -> (f64, usize, usize) {
        let mut run = || {
            reset_batched_gemm_launches_for_test();
            reset_lt_gemm_launches_for_test();
            let start = Instant::now();
            let result = gpu
                .with_backend_session(|__s| {
                    __s.dot_general_read(
                        TensorRead::from_tensor(black_box(lhs)),
                        TensorRead::from_tensor(black_box(rhs)),
                        config,
                    )
                })
                .unwrap();
            black_box(result).unwrap();
            gpu.runtime().synchronize().unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        for _ in 0..3 {
            black_box(run());
        }
        let mut samples: Vec<f64> = (0..7).map(|_| run()).collect();
        samples.sort_by(f64::total_cmp);
        (
            samples[samples.len() / 2],
            batched_gemm_launches_for_test(),
            lt_gemm_launches_for_test(),
        )
    }

    let layouts = batch_16_layouts();
    let config = batched_matmul_config();
    let outer_config = DotGeneralConfig {
        lhs_contracting_dims: [2].as_slice().into(),
        rhs_contracting_dims: [1].as_slice().into(),
        lhs_batch_dims: [0].as_slice().into(),
        rhs_batch_dims: [0].as_slice().into(),
    };
    let expected = cpu_dot_general(&layouts.lhs_standard, &layouts.rhs_standard, &config);

    let mut gpu = gpu_backend();
    let lhs = upload(&gpu, &layouts.lhs_standard);
    let rhs = upload(&gpu, &layouts.rhs_standard);
    let lhs_outer = upload(&gpu, &layouts.lhs_outer);
    let rhs_outer = upload(&gpu, &layouts.rhs_outer);

    let (cublas_ms, cublas_strided_launches, cublas_lt_launches) =
        measure(&mut gpu, &lhs, &rhs, &config);
    let (cutensor_ms, cutensor_launches, _) =
        measure(&mut gpu, &lhs_outer, &rhs_outer, &outer_config);

    let fast = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
        })
        .unwrap()
        .unwrap();
    let general = gpu
        .with_backend_session(|__s| {
            __s.dot_general_read(
                TensorRead::from_tensor(&lhs_outer),
                TensorRead::from_tensor(&rhs_outer),
                &outer_config,
            )
        })
        .unwrap()
        .unwrap();
    let fast = download(&gpu, &fast);
    let general = download(&gpu, &general);
    assert_eq!(fast.shape(), expected.shape());
    assert_eq!(general.shape(), expected.shape());
    assert_tensor_close(&fast, &expected, 1e-9);
    assert_tensor_close(&general, &expected, 1e-9);
    assert_tensor_close(&fast, &general, 1e-9);

    println!(
        "16x32x32 f64 batched GEMM medians of 7 runs after 3 warmups (host API + sync): \
         cuBLASLt {cublas_ms:.4} ms ({cublas_lt_launches} launch(es), \
         {cublas_strided_launches} strided launches), \
         cuTENSOR {cutensor_ms:.4} ms ({cutensor_launches} launch(es))"
    );
}

/// Threshold probe for the small-batch cuBLASLt routing.
///
/// Prints one median per square extent with 16 batches; the launch columns show
/// which path served each extent. Run with:
/// `cargo test -p tenferro-gpu --features cuda --lib -- --ignored --nocapture
/// cuda_batched_matmul_cublaslt_threshold_timing`.
#[test]
#[ignore = "requires CUDA 12.8+ GPU; prints A100 threshold timings"]
fn cuda_batched_matmul_cublaslt_threshold_timing() {
    use std::hint::black_box;
    use std::time::Instant;

    let batch = 16usize;
    let config = batched_matmul_config();
    let mut gpu = gpu_backend();
    for dim in [16usize, 32, 48, 64, 96, 128] {
        let lhs_shape = vec![dim, dim, batch];
        let rhs_shape = vec![dim, dim, batch];
        let lhs = upload(&gpu, &tensor_f64(lhs_shape.clone(), ramp(&lhs_shape, 0.1)));
        let rhs = upload(
            &gpu,
            &tensor_f64(rhs_shape.clone(), ramp(&rhs_shape, -0.05)),
        );
        let mut run = || {
            reset_batched_gemm_launches_for_test();
            reset_lt_gemm_launches_for_test();
            let start = Instant::now();
            let result = gpu
                .with_backend_session(|__s| {
                    __s.dot_general_read(
                        TensorRead::from_tensor(black_box(&lhs)),
                        TensorRead::from_tensor(black_box(&rhs)),
                        &config,
                    )
                })
                .unwrap();
            black_box(result).unwrap();
            gpu.runtime().synchronize().unwrap();
            start.elapsed().as_secs_f64() * 1e3
        };
        for _ in 0..3 {
            black_box(run());
        }
        let mut samples: Vec<f64> = (0..7).map(|_| run()).collect();
        samples.sort_by(f64::total_cmp);
        println!(
            "batch={batch} dim={dim}: {:.4} ms ({} cuBLASLt, {} strided)",
            samples[samples.len() / 2],
            lt_gemm_launches_for_test(),
            batched_gemm_launches_for_test(),
        );
    }
}

/// A single matrix keeps the cuTENSOR path and still matches.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_batched_matmul_with_one_batch_stays_on_the_cutensor_path() {
    assert_case_launches(
        tensor_f64(vec![8, 5, 1], ramp(&[8, 5, 1], 0.1)),
        tensor_f64(vec![5, 7, 1], ramp(&[5, 7, 1], -0.05)),
        batched_matmul_config(),
        1e-12,
        GemmPath::Cutensor,
    );
}

/// A dot-general that is a real contraction but not a standard batched matmul
/// keeps cuTENSOR and still matches.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_non_standard_dot_general_stays_on_the_cutensor_path() {
    // Two free axes on each side plus a batch axis: not a matrix pair.
    let lhs_shape = vec![5, 3, 4, 3];
    let rhs_shape = vec![3, 7, 4, 3];
    assert_case_launches(
        tensor_f64(lhs_shape.clone(), ramp(&lhs_shape, 0.1)),
        tensor_f64(rhs_shape.clone(), ramp(&rhs_shape, -0.05)),
        DotGeneralConfig {
            lhs_contracting_dims: [1].as_slice().into(),
            rhs_contracting_dims: [0].as_slice().into(),
            lhs_batch_dims: [3].as_slice().into(),
            rhs_batch_dims: [3].as_slice().into(),
        },
        1e-12,
        GemmPath::Cutensor,
    );
}

/// A conjugated complex operand keeps cuTENSOR: this path only maps a stored-axis
/// transposition to `OP_T` and does not plumb cuBLAS' `OP_C` conjugate
/// transpose yet.
#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_conjugated_complex_batched_matmul_stays_on_the_cutensor_path() {
    let lhs_shape = vec![6, 4, 3];
    let rhs_shape = vec![4, 5, 3];
    let lhs_host = tensor_c64(lhs_shape.clone(), as_c64(&ramp(&lhs_shape, 0.1)));
    let rhs_host = tensor_c64(rhs_shape.clone(), as_c64(&ramp(&rhs_shape, -0.05)));
    let config = batched_matmul_config();

    let mut cpu = cpu_backend();
    let expected = cpu
        .with_backend_session(|__s| {
            __s.dot_general_with_conj(&lhs_host, &rhs_host, &config, true, false)
        })
        .unwrap()
        .unwrap();

    let mut gpu = gpu_backend();
    let lhs = upload(&gpu, &lhs_host);
    let rhs = upload(&gpu, &rhs_host);
    reset_batched_gemm_launches_for_test();
    let actual = gpu
        .with_backend_session(|__s| __s.dot_general_with_conj(&lhs, &rhs, &config, true, false))
        .unwrap()
        .unwrap();
    assert_eq!(
        batched_gemm_launches_for_test(),
        0,
        "a conjugated complex batched matmul must keep the cuTENSOR path"
    );
    assert_tensor_close(&download(&gpu, &actual), &expected, 1e-12);
}
