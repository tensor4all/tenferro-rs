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
    let ctx = CpuContext::new(1);
    let _pool = ctx.thread_pool();
}

#[test]
fn cpu_context_plan_cache() {
    let mut ctx = CpuContext::new(1);
    let _cache = ctx.plan_cache_mut();
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
    assert!(result.is_err());
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
    assert!(result.is_err());
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
    assert!(result.is_err());
}

#[test]
fn load_hiptensor_returns_error() {
    let mut registry = BackendRegistry::new();
    let result = registry.load_hiptensor("/nonexistent/path");
    assert!(result.is_err());
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
