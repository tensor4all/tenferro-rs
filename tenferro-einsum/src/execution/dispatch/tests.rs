use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use super::*;
use crate::execution::util::{tensor_get, unflatten_index};
use crate::planning::plan::{GemmPlan, ReducePlan, StepPlan};
use crate::tests::semiring_backend::SemiringOnlyCpuBackend;

fn tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn zeros(dims: &[usize]) -> Tensor<f64> {
    Tensor::zeros(
        dims,
        tenferro_device::LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
}

fn assert_tensor_close(lhs: &Tensor<f64>, rhs: &Tensor<f64>) {
    assert_eq!(lhs.dims(), rhs.dims());
    let numel: usize = lhs.dims().iter().product();
    for flat in 0..numel {
        let idx = unflatten_index(flat, lhs.dims());
        let l = tensor_get(lhs, &idx).unwrap();
        let r = tensor_get(rhs, &idx).unwrap();
        assert!(
            (l - r).abs() < 1e-10,
            "mismatch at {:?}: left={} right={}",
            idx,
            l,
            r
        );
    }
}

#[test]
fn execute_pairwise_dispatches_dynamic_ewmul_after_reduce_b() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[2.0, 3.0], &[2]);
    let b = tensor(&[5.0, 7.0, 11.0, 13.0], &[2, 2]);
    let mut output = zeros(&[2]);

    let plan = StepPlan {
        diag_a: None,
        diag_b: None,
        gemm: GemmPlan {
            reduce_a: None,
            reduce_b: Some(ReducePlan {
                original_subs: vec![0, 1],
                kept_subs: vec![0],
                out_shape: vec![2],
            }),
            subs_a: vec![0],
            subs_b: vec![0],
            lo_modes: vec![],
            ro_modes: vec![],
            sum_modes: vec![],
            batch_sizes: vec![2],
            m: 1,
            n: 1,
            k: 1,
            target_a: vec![0],
            target_b: vec![0],
            c_gemm_shape: vec![1, 1, 2],
            expanded_shape: vec![2],
            canonical_modes: vec![0],
            needs_final_permute: false,
            a_gemm_shape: vec![1, 1, 2],
            b_gemm_shape: vec![1, 1, 2],
        },
    };

    execute_pairwise_with_plan::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &plan,
        None,
        None,
        &[0],
        &[0, 1],
        &[0],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[2.0 * (5.0 + 11.0), 3.0 * (7.0 + 13.0)], &[2]);
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_pairwise_accepts_semiring_only_backend() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[2.0, 3.0], &[2]);
    let b = tensor(&[5.0, 7.0, 11.0, 13.0], &[2, 2]);
    let mut output = zeros(&[2]);

    let plan = StepPlan {
        diag_a: None,
        diag_b: None,
        gemm: GemmPlan {
            reduce_a: None,
            reduce_b: Some(ReducePlan {
                original_subs: vec![0, 1],
                kept_subs: vec![0],
                out_shape: vec![2],
            }),
            subs_a: vec![0],
            subs_b: vec![0],
            lo_modes: vec![],
            ro_modes: vec![],
            sum_modes: vec![],
            batch_sizes: vec![2],
            m: 1,
            n: 1,
            k: 1,
            target_a: vec![0],
            target_b: vec![0],
            c_gemm_shape: vec![1, 1, 2],
            expanded_shape: vec![2],
            canonical_modes: vec![0],
            needs_final_permute: false,
            a_gemm_shape: vec![1, 1, 2],
            b_gemm_shape: vec![1, 1, 2],
        },
    };

    execute_pairwise_with_plan::<Standard<f64>, SemiringOnlyCpuBackend>(
        &mut ctx,
        &plan,
        None,
        None,
        &[0],
        &[0, 1],
        &[0],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[2.0 * (5.0 + 11.0), 3.0 * (7.0 + 13.0)], &[2]);
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_gemm_after_reduce_builds_direct_batched_gemm_plan() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let mut output = zeros(&[2, 2]);

    let plan = GemmPlan {
        reduce_a: None,
        reduce_b: None,
        subs_a: vec![0, 1],
        subs_b: vec![1, 2],
        lo_modes: vec![0],
        ro_modes: vec![2],
        sum_modes: vec![1],
        batch_sizes: vec![],
        m: 2,
        n: 2,
        k: 2,
        target_a: vec![0, 1],
        target_b: vec![1, 2],
        c_gemm_shape: vec![2, 2],
        expanded_shape: vec![2, 2],
        canonical_modes: vec![0, 2],
        needs_final_permute: false,
        a_gemm_shape: vec![2, 2],
        b_gemm_shape: vec![2, 2],
    };

    execute_gemm_after_reduce::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &plan,
        None,
        &[0, 2],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[23.0, 34.0, 31.0, 46.0], &[2, 2]);
    assert_tensor_close(&output, &expected);
}

#[test]
fn execute_gemm_after_reduce_builds_temp_gemm_and_physical_permute() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let mut output = zeros(&[2, 2]);

    let plan = GemmPlan {
        reduce_a: None,
        reduce_b: None,
        subs_a: vec![0, 1],
        subs_b: vec![1, 2],
        lo_modes: vec![0],
        ro_modes: vec![2],
        sum_modes: vec![1],
        batch_sizes: vec![],
        m: 2,
        n: 2,
        k: 3,
        target_a: vec![0, 1],
        target_b: vec![1, 2],
        c_gemm_shape: vec![2, 2],
        expanded_shape: vec![2, 2],
        canonical_modes: vec![0, 2],
        needs_final_permute: true,
        a_gemm_shape: vec![2, 3],
        b_gemm_shape: vec![3, 2],
    };

    execute_gemm_after_reduce::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &plan,
        None,
        &[2, 0],
        &a,
        &b,
        1.0,
        0.0,
        &mut output,
        &mut pool,
        false,
    )
    .unwrap();

    let expected = tensor(&[22.0, 49.0, 28.0, 64.0], &[2, 2]);
    assert_tensor_close(&output, &expected);
    let reused = pool.take(6);
    assert_eq!(reused.len(), 6);
}
