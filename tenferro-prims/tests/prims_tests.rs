//! Tests for tenferro-prims: CPU backend plan/execute, resolve_conj,
//! BackendRegistry, GPU stubs.

use strided_traits::ScalarBase;
use strided_view::{StridedArray, StridedView, StridedViewMut};
use tenferro_algebra::Standard;
use tenferro_prims::{
    BackendRegistry, CpuBackend, CpuContext, CpuPlan, Extension, PrimDescriptor, ReduceOp,
    TensorPrims, UnaryOp,
};

// Helper functions to disambiguate the algebra parameter S for the CPU backend.
fn cpu_plan<T: ScalarBase>(
    ctx: &mut CpuContext,
    desc: &PrimDescriptor,
    shapes: &[&[usize]],
) -> tenferro_device::Result<CpuPlan<T>>
where
    T: tenferro_algebra::Scalar,
{
    <CpuBackend as TensorPrims<Standard<T>>>::plan::<T>(ctx, desc, shapes)
}

fn cpu_execute<T: ScalarBase>(
    ctx: &mut CpuContext,
    plan: &CpuPlan<T>,
    alpha: T,
    inputs: &[&StridedView<T>],
    beta: T,
    output: &mut StridedViewMut<T>,
) -> tenferro_device::Result<()>
where
    T: tenferro_algebra::Scalar,
{
    <CpuBackend as TensorPrims<Standard<T>>>::execute(ctx, plan, alpha, inputs, beta, output)
}

fn cpu_has_ext<T: ScalarBase>(ext: Extension) -> bool
where
    T: tenferro_algebra::Scalar,
{
    <CpuBackend as TensorPrims<Standard<T>>>::has_extension_for::<T>(ext)
}

// ============================================================================
// CpuContext
// ============================================================================

#[test]
fn cpu_context_creation() {
    let ctx = CpuContext::new(2);
    assert_eq!(ctx.num_threads(), 2);
}

#[test]
fn cpu_context_thread_pool() {
    let ctx = CpuContext::new(2);
    let pool = ctx.thread_pool();
    assert_eq!(pool.current_num_threads(), 2);
}

#[test]
fn cpu_context_plan_cache() {
    let mut ctx = CpuContext::new(1);
    let _cache = ctx.plan_cache_mut();
    // Verify we get a mutable reference to PlanCache (type-level check).
}

// ============================================================================
// has_extension_for
// ============================================================================

#[test]
fn cpu_has_extension_contract() {
    assert!(cpu_has_ext::<f64>(Extension::Contract));
}

#[test]
fn cpu_has_extension_elementwise_mul() {
    assert!(cpu_has_ext::<f64>(Extension::ElementwiseMul));
}

// ============================================================================
// Permute
// ============================================================================

#[test]
fn permute_transpose_2x3() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3], |idx| (idx[0] + 1 + idx[1] * 2) as f64);
    let mut b = StridedArray::<f64>::col_major(&[3, 2]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut b.view_mut()).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(b.view().get(&[j, i]), a.view().get(&[i, j]));
        }
    }
}

#[test]
fn permute_with_alpha_beta() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3], |idx| (idx[0] + idx[1] * 2 + 1) as f64);
    let mut b = StridedArray::from_fn_col_major(&[3, 2], |_| 1.0_f64);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    // B = 2 * A^T + 3 * B
    cpu_execute(&mut ctx, &plan, 2.0, &[&a.view()], 3.0, &mut b.view_mut()).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            let expected = 2.0 * a.view().get(&[i, j]) + 3.0;
            assert_eq!(b.view().get(&[j, i]), expected);
        }
    }
}

#[test]
fn permute_3d() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3, 4], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
    });
    let mut b = StridedArray::<f64>::col_major(&[4, 2, 3]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1, 2],
        modes_b: vec![2, 0, 1],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3, 4], &[4, 2, 3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut b.view_mut()).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(b.view().get(&[k, i, j]), a.view().get(&[i, j, k]));
            }
        }
    }
}

// ============================================================================
// MakeContiguous
// ============================================================================

#[test]
fn make_contiguous() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3, 4], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut b = StridedArray::<f64>::col_major(&[3, 4]);

    let desc = PrimDescriptor::MakeContiguous;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut b.view_mut()).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(b.view().get(&[i, j]), a.view().get(&[i, j]));
        }
    }
}

// ============================================================================
// BatchedGemm
// ============================================================================

#[test]
fn batched_gemm_2x3_times_3x2() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = StridedArray::from_fn_col_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[2, 2]);

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 3,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2], &[2, 2]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        1.0,
        &[&a.view(), &b.view()],
        0.0,
        &mut c.view_mut(),
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..2 {
            let mut expected = 0.0;
            for k in 0..3 {
                expected += a.view().get(&[i, k]) * b.view().get(&[k, j]);
            }
            assert!(
                (c.view().get(&[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                c.view().get(&[i, j])
            );
        }
    }
}

#[test]
fn batched_gemm_with_batch() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 2, 3], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2] + 1) as f64
    });
    let b = StridedArray::from_fn_col_major(&[2, 3, 2], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2] + 1) as f64
    });
    let mut c = StridedArray::<f64>::col_major(&[2, 2, 2]);

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![2],
        m: 2,
        n: 2,
        k: 3,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 2, 3], &[2, 3, 2], &[2, 2, 2]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        1.0,
        &[&a.view(), &b.view()],
        0.0,
        &mut c.view_mut(),
    )
    .unwrap();

    for batch in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let mut expected = 0.0;
                for k in 0..3 {
                    expected += a.view().get(&[batch, i, k]) * b.view().get(&[batch, k, j]);
                }
                assert!(
                    (c.view().get(&[batch, i, j]) - expected).abs() < 1e-10,
                    "C[{batch},{i},{j}] = {}, expected {expected}",
                    c.view().get(&[batch, i, j])
                );
            }
        }
    }
}

// ============================================================================
// Reduce (Sum)
// ============================================================================

#[test]
fn reduce_sum_axis1() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3, 4], |idx| (idx[0] * 10 + idx[1] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[3]);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

    for i in 0..3 {
        let mut expected = 0.0;
        for j in 0..4 {
            expected += a.view().get(&[i, j]);
        }
        assert!(
            (c.view().get(&[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            c.view().get(&[i])
        );
    }
}

#[test]
fn reduce_sum_axis0() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3, 4], |idx| (idx[0] * 10 + idx[1] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[4]);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

    for j in 0..4 {
        let mut expected = 0.0;
        for i in 0..3 {
            expected += a.view().get(&[i, j]);
        }
        assert!(
            (c.view().get(&[j]) - expected).abs() < 1e-10,
            "C[{j}] = {}, expected {expected}",
            c.view().get(&[j])
        );
    }
}

#[test]
fn reduce_sum_full() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3, 4], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[]);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

    let mut expected = 0.0;
    for i in 0..3 {
        for j in 0..4 {
            expected += (i + j + 1) as f64;
        }
    }
    assert!((c.view().get(&[]) - expected).abs() < 1e-10);
}

#[test]
fn reduce_max_returns_error() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Max,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3]]).unwrap();
    let a = StridedArray::<f64>::col_major(&[3, 4]);
    let mut c = StridedArray::<f64>::col_major(&[3]);
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut());
    match result {
        Err(tenferro_device::Error::InvalidArgument(msg)) => {
            assert!(msg.contains("Max"), "error should mention Max, got: {msg}");
        }
        other => panic!("expected InvalidArgument about Max, got: {other:?}"),
    }
}

// ============================================================================
// Trace
// ============================================================================

#[test]
fn trace_2d_matrix() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3, 3], |idx| {
        if idx[0] == idx[1] {
            (idx[0] + 1) as f64
        } else {
            0.0
        }
    });
    let mut c = StridedArray::<f64>::col_major(&[]);

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 3], &[]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

    // tr(diag(1,2,3)) = 6
    assert!((c.view().get(&[]) - 6.0).abs() < 1e-10);
}

#[test]
fn trace_with_free_axis() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3, 3], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
    });
    let mut c = StridedArray::<f64>::col_major(&[2]);

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1, 2],
        modes_c: vec![0],
        paired: vec![(1, 2)],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3, 3], &[2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

    for i in 0..2 {
        let mut expected = 0.0;
        for d in 0..3 {
            expected += a.view().get(&[i, d, d]);
        }
        assert!(
            (c.view().get(&[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            c.view().get(&[i])
        );
    }
}

// ============================================================================
// ElementwiseMul
// ============================================================================

#[test]
fn elementwise_mul_2d() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3, 4], |idx| (idx[0] + 1) as f64);
    let b = StridedArray::from_fn_col_major(&[3, 4], |idx| (idx[1] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[3, 4]);

    let desc = PrimDescriptor::ElementwiseMul;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4], &[3, 4]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        1.0,
        &[&a.view(), &b.view()],
        0.0,
        &mut c.view_mut(),
    )
    .unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let expected = ((i + 1) * (j + 1)) as f64;
            assert!(
                (c.view().get(&[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                c.view().get(&[i, j])
            );
        }
    }
}

// ============================================================================
// Contract
// ============================================================================

#[test]
fn contract_matrix_multiply() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = StridedArray::from_fn_col_major(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[2, 2]);

    let desc = PrimDescriptor::Contract {
        modes_a: vec![0, 1],
        modes_b: vec![1, 2],
        modes_c: vec![0, 2],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2], &[2, 2]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        1.0,
        &[&a.view(), &b.view()],
        0.0,
        &mut c.view_mut(),
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..2 {
            let mut expected = 0.0;
            for k in 0..3 {
                expected += a.view().get(&[i, k]) * b.view().get(&[k, j]);
            }
            assert!(
                (c.view().get(&[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                c.view().get(&[i, j])
            );
        }
    }
}

#[test]
fn contract_outer_product() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3], |idx| (idx[0] + 1) as f64);
    let b = StridedArray::from_fn_col_major(&[4], |idx| (idx[0] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[3, 4]);

    let desc = PrimDescriptor::Contract {
        modes_a: vec![0],
        modes_b: vec![1],
        modes_c: vec![0, 1],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[4], &[3, 4]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        1.0,
        &[&a.view(), &b.view()],
        0.0,
        &mut c.view_mut(),
    )
    .unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let expected = ((i + 1) * (j + 1)) as f64;
            assert!(
                (c.view().get(&[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                c.view().get(&[i, j])
            );
        }
    }
}

// ============================================================================
// ElementwiseUnary — Conj is identity for real types
// ============================================================================

#[test]
fn elementwise_unary_conj_identity() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[3, 4], |idx| (idx[0] * 10 + idx[1] + 1) as f64);
    let mut c = StridedArray::<f64>::col_major(&[3, 4]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(c.view().get(&[i, j]), a.view().get(&[i, j]));
        }
    }
}

#[test]
fn elementwise_unary_negate_returns_error() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Negate,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    let a = StridedArray::<f64>::col_major(&[3]);
    let mut c = StridedArray::<f64>::col_major(&[3]);
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut());
    match result {
        Err(tenferro_device::Error::InvalidArgument(msg)) => {
            assert!(
                msg.contains("Negate"),
                "error should mention Negate, got: {msg}"
            );
        }
        other => panic!("expected InvalidArgument about Negate, got: {other:?}"),
    }
}

// ============================================================================
// resolve_conj
// ============================================================================

#[test]
fn resolve_conj_non_conjugated() {
    use tenferro_tensor::{MemoryOrder, Tensor};

    let mut ctx = CpuContext::new(1);
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
        .unwrap();
    assert!(!t.is_conjugated());

    let resolved = CpuBackend::resolve_conj(&mut ctx, &t);
    assert!(!resolved.is_conjugated());
    assert_eq!(resolved.dims(), t.dims());
}

// ============================================================================
// BackendRegistry
// ============================================================================

#[test]
fn backend_registry_creation() {
    let registry = BackendRegistry::new();
    let _cpu = registry.cpu();
    assert!(registry.cuda().is_none());
    assert!(registry.rocm().is_none());
}

#[test]
fn backend_registry_default() {
    let registry = BackendRegistry::default();
    assert!(registry.cuda().is_none());
    assert!(registry.rocm().is_none());
}

#[test]
fn load_cutensor_returns_error() {
    let mut registry = BackendRegistry::new();
    let result = registry.load_cutensor("/nonexistent/path");
    match result {
        Err(tenferro_device::Error::DeviceError(msg)) => {
            assert!(
                msg.to_lowercase().contains("cutensor"),
                "error should mention cutensor, got: {msg}"
            );
        }
        other => panic!("expected DeviceError about cutensor, got: {other:?}"),
    }
}

#[test]
fn load_hiptensor_returns_error() {
    let mut registry = BackendRegistry::new();
    let result = registry.load_hiptensor("/nonexistent/path");
    match result {
        Err(tenferro_device::Error::DeviceError(msg)) => {
            assert!(
                msg.to_lowercase().contains("hiptensor"),
                "error should mention hiptensor, got: {msg}"
            );
        }
        other => panic!("expected DeviceError about hiptensor, got: {other:?}"),
    }
}

// ============================================================================
// AntiTrace
// ============================================================================

#[test]
fn anti_trace_scalar_to_diagonal() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[], |_| 5.0_f64);
    let mut c = StridedArray::<f64>::col_major(&[3, 3]);

    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[], &[3, 3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut()).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert!(
                    (c.view().get(&[i, j]) - 5.0).abs() < 1e-10,
                    "C[{i},{j}] = {}, expected 5.0",
                    c.view().get(&[i, j])
                );
            } else {
                assert!(
                    c.view().get(&[i, j]).abs() < 1e-10,
                    "C[{i},{j}] = {}, expected 0.0",
                    c.view().get(&[i, j])
                );
            }
        }
    }
}

// ============================================================================
// Reduce with alpha/beta
// ============================================================================

#[test]
fn reduce_sum_with_alpha_beta() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3], |_| 1.0_f64);
    let mut c = StridedArray::from_fn_col_major(&[2], |_| 10.0_f64);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[2]]).unwrap();
    // C = 2 * sum(A, axis=1) + 3 * C
    // sum over 3 ones = 3, so C = 2 * 3 + 3 * 10 = 36
    cpu_execute(&mut ctx, &plan, 2.0, &[&a.view()], 3.0, &mut c.view_mut()).unwrap();

    for i in 0..2 {
        assert!(
            (c.view().get(&[i]) - 36.0).abs() < 1e-10,
            "C[{i}] = {}, expected 36.0",
            c.view().get(&[i])
        );
    }
}

// ============================================================================
// f32 type
// ============================================================================

#[test]
fn permute_complex64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3], |idx| {
        Complex64::new((idx[0] * 3 + idx[1] + 1) as f64, 0.0)
    });
    let mut b = StridedArray::<Complex64>::col_major(&[3, 2]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a.view()],
        Complex64::new(0.0, 0.0),
        &mut b.view_mut(),
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(b.view().get(&[j, i]), a.view().get(&[i, j]));
        }
    }
}

// ============================================================================
// resolve_conj for Complex64
// ============================================================================

#[test]
fn resolve_conj_complex64_non_conjugated() {
    use num_complex::Complex64;
    use tenferro_tensor::{MemoryOrder, Tensor};

    let mut ctx = CpuContext::new(1);
    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    let t = Tensor::<Complex64>::from_slice(&data, &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    assert!(!t.is_conjugated());

    let resolved = CpuBackend::resolve_conj(&mut ctx, &t);
    assert!(!resolved.is_conjugated());
    assert_eq!(resolved.dims(), t.dims());

    // Data should be unchanged (no conjugation applied)
    let resolved_data = resolved
        .buffer()
        .as_slice()
        .expect("CPU tensor must have CPU-accessible data");
    for (orig, res) in data.iter().zip(resolved_data.iter()) {
        assert_eq!(orig, res);
    }
}

#[test]
fn resolve_conj_complex64_conjugated() {
    use num_complex::Complex64;
    use tenferro_tensor::{MemoryOrder, Tensor};

    let mut ctx = CpuContext::new(1);
    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];
    let t = Tensor::<Complex64>::from_slice(&data, &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let t_conj = t.into_conj();
    assert!(t_conj.is_conjugated());

    let resolved = CpuBackend::resolve_conj(&mut ctx, &t_conj);
    assert!(!resolved.is_conjugated());
    assert_eq!(resolved.dims(), &[2, 2]);

    // Data should have imaginary parts negated
    let resolved_data = resolved
        .buffer()
        .as_slice()
        .expect("CPU tensor must have CPU-accessible data");
    let expected = vec![
        Complex64::new(1.0, -2.0),
        Complex64::new(3.0, -4.0),
        Complex64::new(5.0, -6.0),
        Complex64::new(7.0, -8.0),
    ];
    for (exp, res) in expected.iter().zip(resolved_data.iter()) {
        assert_eq!(exp, res, "expected {exp}, got {res}");
    }
}

#[test]
fn resolve_conj_f64_conjugated_is_identity() {
    use tenferro_tensor::{MemoryOrder, Tensor};

    let mut ctx = CpuContext::new(1);
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let t_conj = t.into_conj();
    assert!(t_conj.is_conjugated());

    let resolved = CpuBackend::resolve_conj(&mut ctx, &t_conj);
    assert!(!resolved.is_conjugated());

    // For real types, conjugation is identity — data should be unchanged
    let resolved_data = resolved
        .buffer()
        .as_slice()
        .expect("CPU tensor must have CPU-accessible data");
    for (orig, res) in data.iter().zip(resolved_data.iter()) {
        assert!((orig - res).abs() < 1e-15, "expected {orig}, got {res}");
    }
}

// ============================================================================
// f32 type
// ============================================================================

#[test]
fn permute_f32() {
    let mut ctx = CpuContext::new(1);
    let a = StridedArray::from_fn_col_major(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f32);
    let mut b = StridedArray::<f32>::col_major(&[3, 2]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f32>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        1.0_f32,
        &[&a.view()],
        0.0_f32,
        &mut b.view_mut(),
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(b.view().get(&[j, i]), a.view().get(&[i, j]));
        }
    }
}

// ============================================================================
// Validation: plan() shape count errors
// ============================================================================

#[test]
fn plan_batched_gemm_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 3,
        k: 4,
    };
    // Only 2 shapes instead of the required 3
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 4], &[4, 3]]);
    assert!(result.is_err(), "expected error for wrong shape count");
    let err = result.unwrap_err();
    assert!(
        matches!(err, tenferro_device::Error::InvalidArgument(_)),
        "expected InvalidArgument, got: {err:?}"
    );
}

#[test]
fn plan_batched_gemm_too_many_shapes() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 3,
        k: 4,
    };
    // 4 shapes instead of 3
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 4], &[4, 3], &[2, 3], &[1]]);
    assert!(result.is_err(), "expected error for too many shapes");
}

#[test]
fn plan_batched_gemm_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 3,
        k: 4,
    };
    // A shape [2, 4] is correct, B shape [5, 3] is wrong (should be [4, 3])
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 4], &[5, 3], &[2, 3]]);
    assert!(result.is_err(), "expected error for mismatched shapes");
    let err = result.unwrap_err();
    assert!(
        matches!(err, tenferro_device::Error::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {err:?}"
    );
}

#[test]
fn plan_reduce_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    // Only 1 shape instead of 2
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_reduce_wrong_rank() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    // Input A has rank 3 but modes_a has 2 entries
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4, 5], &[3]]);
    assert!(result.is_err(), "expected error for rank mismatch");
    let err = result.unwrap_err();
    assert!(
        matches!(
            err,
            tenferro_device::Error::RankMismatch {
                expected: 2,
                got: 3
            }
        ),
        "expected RankMismatch, got: {err:?}"
    );
}

#[test]
fn plan_permute_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    // 3 shapes instead of 2
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4, 3], &[1]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_trace_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    // Only 1 shape instead of 2
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 3]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_trace_mismatched_paired_dims() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    // Paired axes have dimensions 3 and 4 (must be equal for trace)
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[]]);
    assert!(result.is_err(), "expected error for mismatched paired dims");
    let err = result.unwrap_err();
    assert!(
        matches!(err, tenferro_device::Error::InvalidArgument(_)),
        "expected InvalidArgument, got: {err:?}"
    );
}

#[test]
fn plan_contract_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Contract {
        modes_a: vec![0, 1],
        modes_b: vec![1, 2],
        modes_c: vec![0, 2],
    };
    // Only 2 shapes instead of 3
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 4]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_contract_mismatched_contracted_dims() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Contract {
        modes_a: vec![0, 1],
        modes_b: vec![1, 2],
        modes_c: vec![0, 2],
    };
    // Mode 1 has dim 3 in A but dim 5 in B
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[5, 4], &[2, 4]]);
    assert!(
        result.is_err(),
        "expected error for mismatched contracted dims"
    );
    let err = result.unwrap_err();
    assert!(
        matches!(err, tenferro_device::Error::InvalidArgument(_)),
        "expected InvalidArgument, got: {err:?}"
    );
}

#[test]
fn plan_elementwise_mul_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseMul;
    // Only 1 shape instead of 3
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_elementwise_unary_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    // 3 shapes instead of 2
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3], &[3]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_make_contiguous_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::MakeContiguous;
    // 0 shapes instead of 2
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_anti_trace_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    // Only 1 shape instead of 2
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 3]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

#[test]
fn plan_anti_diag_wrong_shape_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::AntiDiag {
        modes_a: vec![0],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    // 3 shapes instead of 2
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3, 3], &[1]]);
    assert!(result.is_err(), "expected error for wrong shape count");
}

// ============================================================================
// Validation: execute() input count errors
// ============================================================================

#[test]
fn execute_permute_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    let a = StridedArray::<f64>::col_major(&[2, 3]);
    let b = StridedArray::<f64>::col_major(&[2, 3]);
    let mut c = StridedArray::<f64>::col_major(&[3, 2]);
    // Provide 2 inputs instead of 1
    let result = cpu_execute(
        &mut ctx,
        &plan,
        1.0,
        &[&a.view(), &b.view()],
        0.0,
        &mut c.view_mut(),
    );
    assert!(result.is_err(), "expected error for wrong input count");
    let err = result.unwrap_err();
    assert!(
        matches!(err, tenferro_device::Error::InvalidArgument(_)),
        "expected InvalidArgument, got: {err:?}"
    );
}

#[test]
fn execute_permute_zero_inputs() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    let mut c = StridedArray::<f64>::col_major(&[3, 2]);
    // Provide 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c.view_mut());
    assert!(result.is_err(), "expected error for zero inputs");
}

#[test]
fn execute_batched_gemm_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 3,
        k: 4,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 4], &[4, 3], &[2, 3]]).unwrap();
    let a = StridedArray::<f64>::col_major(&[2, 4]);
    let mut c = StridedArray::<f64>::col_major(&[2, 3]);
    // Only 1 input instead of 2
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut());
    assert!(result.is_err(), "expected error for wrong input count");
}

#[test]
fn execute_contract_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Contract {
        modes_a: vec![0, 1],
        modes_b: vec![1, 2],
        modes_c: vec![0, 2],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 4], &[2, 4]]).unwrap();
    let a = StridedArray::<f64>::col_major(&[2, 3]);
    let mut c = StridedArray::<f64>::col_major(&[2, 4]);
    // Only 1 input instead of 2
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut());
    assert!(result.is_err(), "expected error for wrong input count");
}

#[test]
fn execute_elementwise_mul_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseMul;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3], &[3]]).unwrap();
    let a = StridedArray::<f64>::col_major(&[3]);
    let mut c = StridedArray::<f64>::col_major(&[3]);
    // Only 1 input instead of 2
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a.view()], 0.0, &mut c.view_mut());
    assert!(result.is_err(), "expected error for wrong input count");
}

#[test]
fn execute_reduce_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3]]).unwrap();
    let a = StridedArray::<f64>::col_major(&[3, 4]);
    let b = StridedArray::<f64>::col_major(&[3, 4]);
    let mut c = StridedArray::<f64>::col_major(&[3]);
    // 2 inputs instead of 1
    let result = cpu_execute(
        &mut ctx,
        &plan,
        1.0,
        &[&a.view(), &b.view()],
        0.0,
        &mut c.view_mut(),
    );
    assert!(result.is_err(), "expected error for wrong input count");
}

#[test]
fn execute_trace_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 3], &[]]).unwrap();
    let mut c = StridedArray::<f64>::col_major(&[]);
    // 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c.view_mut());
    assert!(result.is_err(), "expected error for zero inputs");
}

#[test]
fn execute_make_contiguous_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::MakeContiguous;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
    let mut c = StridedArray::<f64>::col_major(&[3, 4]);
    // 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c.view_mut());
    assert!(result.is_err(), "expected error for zero inputs");
}

#[test]
fn execute_elementwise_unary_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    let mut c = StridedArray::<f64>::col_major(&[3]);
    // 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c.view_mut());
    assert!(result.is_err(), "expected error for zero inputs");
}
