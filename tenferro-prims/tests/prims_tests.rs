//! Tests for tenferro-prims: CPU backend plan/execute, resolve_conj,
//! BackendRegistry, GPU stubs.
//!
//! Core numeric tests are parameterized across f32, f64, and Complex64 via the
//! `typed_prims_tests!` macro at the bottom of this file.

use tenferro_algebra::{Scalar, Standard};
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::{
    BackendRegistry, CpuBackend, CpuContext, CpuPlan, Extension, PrimDescriptor, ReduceOp,
    TensorPrims, UnaryOp,
};
use tenferro_tensor::{MemoryOrder, Tensor};

// Helper functions to disambiguate the algebra parameter S for the CPU backend.
fn cpu_plan<T: Scalar>(
    ctx: &mut CpuContext,
    desc: &PrimDescriptor,
    shapes: &[&[usize]],
) -> tenferro_device::Result<CpuPlan<T>> {
    <CpuBackend as TensorPrims<Standard<T>>>::plan::<T>(ctx, desc, shapes)
}

fn cpu_execute<T: Scalar>(
    ctx: &mut CpuContext,
    plan: &CpuPlan<T>,
    alpha: T,
    inputs: &[&Tensor<T>],
    beta: T,
    output: &mut Tensor<T>,
) -> tenferro_device::Result<()> {
    <CpuBackend as TensorPrims<Standard<T>>>::execute(ctx, plan, alpha, inputs, beta, output)
}

fn cpu_has_ext<T: Scalar>(ext: Extension) -> bool {
    <CpuBackend as TensorPrims<Standard<T>>>::has_extension_for::<T>(ext)
}

// ---------------------------------------------------------------------------
// Test helpers: Tensor construction and element access
// ---------------------------------------------------------------------------

/// Create a Tensor from a closure, column-major order.
/// Equivalent to the old `StridedArray::from_fn_col_major`.
fn tensor_from_fn<T: Scalar>(dims: &[usize], f: impl Fn(&[usize]) -> T) -> Tensor<T> {
    let ndim = dims.len();
    let n_elements: usize = dims.iter().product();
    let mut data = vec![T::zero(); n_elements];
    let strides = col_major_strides(dims);
    let mut idx = vec![0usize; ndim];
    for _ in 0..n_elements {
        let linear: usize = idx.iter().zip(strides.iter()).map(|(&i, &s)| i * s).sum();
        data[linear] = f(&idx);
        // increment index in column-major (first axis fastest)
        for d in 0..ndim {
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Tensor::from_slice(&data, dims, MemoryOrder::ColumnMajor).unwrap()
}

/// Zero-initialized Tensor (column-major).
fn tensor_zeros<T: Scalar>(dims: &[usize]) -> Tensor<T> {
    Tensor::<T>::zeros(
        dims,
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
}

/// Column-major strides in element counts (not isize).
fn col_major_strides(dims: &[usize]) -> Vec<usize> {
    let ndim = dims.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![0usize; ndim];
    strides[0] = 1;
    for i in 1..ndim {
        strides[i] = strides[i - 1] * dims[i - 1];
    }
    strides
}

/// Read a single element from a Tensor by multi-dimensional index.
fn tensor_get<T: Scalar>(t: &Tensor<T>, idx: &[usize]) -> T {
    let data = t.buffer().as_slice().expect("CPU tensor");
    let offset = t.offset()
        + idx
            .iter()
            .zip(t.strides().iter())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
    data[offset as usize]
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
// PlanCache: cache hit/miss semantics
// ============================================================================

#[test]
fn plan_cache_hit_same_signature() {
    // Repeated plan() calls with the same descriptor+shapes should hit the cache.
    let mut ctx = CpuContext::new(1);
    assert!(ctx.plan_cache_mut().is_empty());

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };

    // First call: cache miss, builds and stores the plan.
    let _plan1 = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 1);

    // Second call: cache hit, should not increase cache size.
    let _plan2 = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 1);
}

#[test]
fn plan_cache_miss_different_shapes() {
    // Different shapes should produce separate cache entries.
    let mut ctx = CpuContext::new(1);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };

    let _plan1 = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 1);

    // Different shapes: 4x5 instead of 2x3
    let _plan2 = cpu_plan::<f64>(&mut ctx, &desc, &[&[4, 5], &[5, 4]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 2);
}

#[test]
fn plan_cache_miss_different_scalar_type() {
    // Same descriptor and shapes but different scalar type should miss.
    let mut ctx = CpuContext::new(1);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };

    let _plan_f64 = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 1);

    let _plan_f32 = cpu_plan::<f32>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 2);
}

#[test]
fn plan_cache_miss_different_descriptor() {
    // Same shapes but different descriptor should miss.
    let mut ctx = CpuContext::new(1);

    let desc1 = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let desc2 = PrimDescriptor::MakeContiguous;

    let _plan1 = cpu_plan::<f64>(&mut ctx, &desc1, &[&[2, 3], &[3, 2]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 1);

    let _plan2 = cpu_plan::<f64>(&mut ctx, &desc2, &[&[2, 3], &[2, 3]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 2);
}

#[test]
fn plan_cache_clear() {
    let mut ctx = CpuContext::new(1);

    let desc = PrimDescriptor::MakeContiguous;
    let _plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 1);

    ctx.plan_cache_mut().clear();
    assert!(ctx.plan_cache_mut().is_empty());
}

#[test]
fn plan_cache_hit_produces_correct_results() {
    // Verify that a cached plan produces the same correct results as the original.
    let mut ctx = CpuContext::new(1);

    let desc = PrimDescriptor::Contract {
        modes_a: vec![0, 1],
        modes_b: vec![1, 2],
        modes_c: vec![0, 2],
    };
    let shapes: &[&[usize]] = &[&[2, 3], &[3, 4], &[2, 4]];

    // Build and cache the plan
    let _plan1 = cpu_plan::<f64>(&mut ctx, &desc, shapes).unwrap();
    assert_eq!(ctx.plan_cache_mut().len(), 1);

    // Use cached plan for actual computation
    let a = tensor_from_fn(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = tensor_from_fn(&[3, 4], |idx| (idx[0] * 4 + idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[2, 4]);

    let plan2 = cpu_plan::<f64>(&mut ctx, &desc, shapes).unwrap();
    cpu_execute(&mut ctx, &plan2, 1.0, &[&a, &b], 0.0, &mut c).unwrap();

    // Verify results match manual computation
    for i in 0..2 {
        for j in 0..4 {
            let mut expected = 0.0;
            for k in 0..3 {
                expected += tensor_get(&a, &[i, k]) * tensor_get(&b, &[k, j]);
            }
            assert!(
                (tensor_get(&c, &[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                tensor_get(&c, &[i, j])
            );
        }
    }
}

#[test]
fn plan_cache_complex64_separate_from_f64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };

    // Build f64 plan
    let _plan_f64 = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();

    // Build Complex64 plan with same shapes
    let _plan_c64 = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();

    // Should be 2 distinct cache entries
    assert_eq!(ctx.plan_cache_mut().len(), 2);

    // Execute Complex64 plan to verify it works correctly
    let a = tensor_from_fn(&[2, 3], |idx| {
        Complex64::new((idx[0] * 3 + idx[1] + 1) as f64, 0.0)
    });
    let mut b = tensor_zeros::<Complex64>(&[3, 2]);

    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a],
        Complex64::new(0.0, 0.0),
        &mut b,
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(tensor_get(&b, &[j, i]), tensor_get(&a, &[i, j]));
        }
    }
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
// Permute (original f64 tests kept for backward compatibility)
// ============================================================================

#[test]
fn permute_transpose_2x3() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3], |idx| (idx[0] + 1 + idx[1] * 2) as f64);
    let mut b = tensor_zeros::<f64>(&[3, 2]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut b).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(tensor_get(&b, &[j, i]), tensor_get(&a, &[i, j]));
        }
    }
}

#[test]
fn permute_with_alpha_beta() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3], |idx| (idx[0] + idx[1] * 2 + 1) as f64);
    let mut b = tensor_from_fn(&[3, 2], |_| 1.0_f64);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    // B = 2 * A^T + 3 * B
    cpu_execute(&mut ctx, &plan, 2.0, &[&a], 3.0, &mut b).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            let expected = 2.0 * tensor_get(&a, &[i, j]) + 3.0;
            assert_eq!(tensor_get(&b, &[j, i]), expected);
        }
    }
}

#[test]
fn permute_3d() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3, 4], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
    });
    let mut b = tensor_zeros::<f64>(&[4, 2, 3]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1, 2],
        modes_b: vec![2, 0, 1],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3, 4], &[4, 2, 3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut b).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                assert_eq!(tensor_get(&b, &[k, i, j]), tensor_get(&a, &[i, j, k]));
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
    let a = tensor_from_fn(&[3, 4], |idx| (idx[0] * 10 + idx[1]) as f64);
    let mut b = tensor_zeros::<f64>(&[3, 4]);

    let desc = PrimDescriptor::MakeContiguous;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut b).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(tensor_get(&b, &[i, j]), tensor_get(&a, &[i, j]));
        }
    }
}

// ============================================================================
// BatchedGemm
// ============================================================================

#[test]
fn batched_gemm_2x3_times_3x2() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = tensor_from_fn(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[2, 2]);

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![],
        m: 2,
        n: 2,
        k: 3,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2], &[2, 2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a, &b], 0.0, &mut c).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            let mut expected = 0.0;
            for k in 0..3 {
                expected += tensor_get(&a, &[i, k]) * tensor_get(&b, &[k, j]);
            }
            assert!(
                (tensor_get(&c, &[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                tensor_get(&c, &[i, j])
            );
        }
    }
}

#[test]
fn batched_gemm_with_batch() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 2, 3], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2] + 1) as f64
    });
    let b = tensor_from_fn(&[2, 3, 2], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2] + 1) as f64
    });
    let mut c = tensor_zeros::<f64>(&[2, 2, 2]);

    let desc = PrimDescriptor::BatchedGemm {
        batch_dims: vec![2],
        m: 2,
        n: 2,
        k: 3,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 2, 3], &[2, 3, 2], &[2, 2, 2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a, &b], 0.0, &mut c).unwrap();

    for batch in 0..2 {
        for i in 0..2 {
            for j in 0..2 {
                let mut expected = 0.0;
                for k in 0..3 {
                    expected += tensor_get(&a, &[batch, i, k]) * tensor_get(&b, &[batch, k, j]);
                }
                assert!(
                    (tensor_get(&c, &[batch, i, j]) - expected).abs() < 1e-10,
                    "C[{batch},{i},{j}] = {}, expected {expected}",
                    tensor_get(&c, &[batch, i, j])
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
    let a = tensor_from_fn(&[3, 4], |idx| (idx[0] * 10 + idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[3]);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..3 {
        let mut expected = 0.0;
        for j in 0..4 {
            expected += tensor_get(&a, &[i, j]);
        }
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn reduce_sum_axis0() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 4], |idx| (idx[0] * 10 + idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[4]);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![1],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for j in 0..4 {
        let mut expected = 0.0;
        for i in 0..3 {
            expected += tensor_get(&a, &[i, j]);
        }
        assert!(
            (tensor_get(&c, &[j]) - expected).abs() < 1e-10,
            "C[{j}] = {}, expected {expected}",
            tensor_get(&c, &[j])
        );
    }
}

#[test]
fn reduce_sum_full() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 4], |idx| (idx[0] + idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[]);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    let mut expected = 0.0;
    for i in 0..3 {
        for j in 0..4 {
            expected += (i + j + 1) as f64;
        }
    }
    assert!((tensor_get(&c, &[]) - expected).abs() < 1e-10);
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
    let a = tensor_zeros::<f64>(&[3, 4]);
    let mut c = tensor_zeros::<f64>(&[3]);
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c);
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
    let a = tensor_from_fn(&[3, 3], |idx| {
        if idx[0] == idx[1] {
            (idx[0] + 1) as f64
        } else {
            0.0
        }
    });
    let mut c = tensor_zeros::<f64>(&[]);

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1],
        modes_c: vec![],
        paired: vec![(0, 1)],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 3], &[]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    // tr(diag(1,2,3)) = 6
    assert!((tensor_get(&c, &[]) - 6.0).abs() < 1e-10);
}

#[test]
fn trace_with_free_axis() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3, 3], |idx| {
        (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
    });
    let mut c = tensor_zeros::<f64>(&[2]);

    let desc = PrimDescriptor::Trace {
        modes_a: vec![0, 1, 2],
        modes_c: vec![0],
        paired: vec![(1, 2)],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3, 3], &[2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..2 {
        let mut expected = 0.0;
        for d in 0..3 {
            expected += tensor_get(&a, &[i, d, d]);
        }
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseMul
// ============================================================================

#[test]
fn elementwise_mul_2d() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 4], |idx| (idx[0] + 1) as f64);
    let b = tensor_from_fn(&[3, 4], |idx| (idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[3, 4]);

    let desc = PrimDescriptor::ElementwiseMul;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4], &[3, 4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a, &b], 0.0, &mut c).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let expected = ((i + 1) * (j + 1)) as f64;
            assert!(
                (tensor_get(&c, &[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                tensor_get(&c, &[i, j])
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
    let a = tensor_from_fn(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f64);
    let b = tensor_from_fn(&[3, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[2, 2]);

    let desc = PrimDescriptor::Contract {
        modes_a: vec![0, 1],
        modes_b: vec![1, 2],
        modes_c: vec![0, 2],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[3, 2], &[2, 2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a, &b], 0.0, &mut c).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            let mut expected = 0.0;
            for k in 0..3 {
                expected += tensor_get(&a, &[i, k]) * tensor_get(&b, &[k, j]);
            }
            assert!(
                (tensor_get(&c, &[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                tensor_get(&c, &[i, j])
            );
        }
    }
}

#[test]
fn contract_outer_product() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| (idx[0] + 1) as f64);
    let b = tensor_from_fn(&[4], |idx| (idx[0] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[3, 4]);

    let desc = PrimDescriptor::Contract {
        modes_a: vec![0],
        modes_b: vec![1],
        modes_c: vec![0, 1],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[4], &[3, 4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a, &b], 0.0, &mut c).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            let expected = ((i + 1) * (j + 1)) as f64;
            assert!(
                (tensor_get(&c, &[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                tensor_get(&c, &[i, j])
            );
        }
    }
}

// ============================================================================
// ElementwiseUnary -- Conj is identity for real types
// ============================================================================

#[test]
fn elementwise_unary_conj_identity() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3, 4], |idx| (idx[0] * 10 + idx[1] + 1) as f64);
    let mut c = tensor_zeros::<f64>(&[3, 4]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..3 {
        for j in 0..4 {
            assert_eq!(tensor_get(&c, &[i, j]), tensor_get(&a, &[i, j]));
        }
    }
}

// ============================================================================
// ElementwiseUnary -- Conj for complex types
// ============================================================================

#[test]
fn elementwise_unary_conj_complex64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex64::new(idx[0] as f64 + 1.0, idx[0] as f64 + 2.0)
    });
    let mut c = tensor_zeros::<Complex64>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a],
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let expected = Complex64::new(i as f64 + 1.0, -(i as f64 + 2.0));
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_conj_complex32() {
    use num_complex::Complex32;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex32::new(idx[0] as f32 + 1.0, idx[0] as f32 + 2.0)
    });
    let mut c = tensor_zeros::<Complex32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    let plan = cpu_plan::<Complex32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex32::new(1.0, 0.0),
        &[&a],
        Complex32::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let expected = Complex32::new(i as f32 + 1.0, -(i as f32 + 2.0));
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-5,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- Negate
// ============================================================================

#[test]
fn elementwise_unary_negate_f64() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[4], |idx| idx[0] as f64 + 1.0);
    let mut c = tensor_zeros::<f64>(&[4]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Negate,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[4], &[4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..4 {
        let expected = -(i as f64 + 1.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- Reciprocal
// ============================================================================

#[test]
fn elementwise_unary_reciprocal_f64() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[4], |idx| idx[0] as f64 + 1.0);
    let mut c = tensor_zeros::<f64>(&[4]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Reciprocal,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[4], &[4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..4 {
        let expected = 1.0 / (i as f64 + 1.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- Abs
// ============================================================================

#[test]
fn elementwise_unary_abs_f64() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[4], |idx| -(idx[0] as f64 + 1.0));
    let mut c = tensor_zeros::<f64>(&[4]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Abs };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[4], &[4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..4 {
        let expected = (i as f64 + 1.0).abs();
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- Sqrt
// ============================================================================

#[test]
fn elementwise_unary_sqrt_f64() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[4], |idx| ((idx[0] + 1) * (idx[0] + 1)) as f64);
    let mut c = tensor_zeros::<f64>(&[4]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Sqrt };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[4], &[4]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..4 {
        let expected = i as f64 + 1.0;
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- Complex64 tests
// ============================================================================

#[test]
fn elementwise_unary_negate_complex64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex64::new(idx[0] as f64 + 1.0, idx[0] as f64 + 2.0)
    });
    let mut c = tensor_zeros::<Complex64>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Negate,
    };
    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a],
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let expected = -Complex64::new(i as f64 + 1.0, i as f64 + 2.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_reciprocal_complex64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex64::new(idx[0] as f64 + 1.0, idx[0] as f64 + 2.0)
    });
    let mut c = tensor_zeros::<Complex64>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Reciprocal,
    };
    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a],
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let z = Complex64::new(i as f64 + 1.0, i as f64 + 2.0);
        let expected = Complex64::new(1.0, 0.0) / z;
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_abs_complex64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex64::new(3.0 * (idx[0] as f64 + 1.0), 4.0 * (idx[0] as f64 + 1.0))
    });
    let mut c = tensor_zeros::<Complex64>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Abs };
    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a],
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        // |3k + 4ki| = 5k, returned as Complex64 with zero imaginary part
        let expected = Complex64::new(5.0 * (i as f64 + 1.0), 0.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_sqrt_complex64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    // Use perfect squares: sqrt(z^2) = z for z with positive real part
    let a = tensor_from_fn(&[3], |idx| {
        let z = Complex64::new(idx[0] as f64 + 1.0, 0.0);
        z * z // perfect square
    });
    let mut c = tensor_zeros::<Complex64>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Sqrt };
    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a],
        Complex64::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let expected = Complex64::new(i as f64 + 1.0, 0.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- alpha/beta support
// ============================================================================

#[test]
fn elementwise_unary_negate_with_alpha_beta() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| idx[0] as f64 + 1.0);
    let mut c = tensor_from_fn(&[3], |_| 10.0_f64);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Negate,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    // C = 2 * (-A) + 3 * C = 2 * (-(i+1)) + 3 * 10
    cpu_execute(&mut ctx, &plan, 2.0, &[&a], 3.0, &mut c).unwrap();

    for i in 0..3 {
        let expected = 2.0 * (-(i as f64 + 1.0)) + 3.0 * 10.0;
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-10,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- f32 tests
// ============================================================================

#[test]
fn elementwise_unary_negate_f32() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| idx[0] as f32 + 1.0);
    let mut c = tensor_zeros::<f32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Negate,
    };
    let plan = cpu_plan::<f32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0_f32, &[&a], 0.0_f32, &mut c).unwrap();

    for i in 0..3 {
        let expected = -(i as f32 + 1.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-5,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_reciprocal_f32() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| idx[0] as f32 + 1.0);
    let mut c = tensor_zeros::<f32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Reciprocal,
    };
    let plan = cpu_plan::<f32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0_f32, &[&a], 0.0_f32, &mut c).unwrap();

    for i in 0..3 {
        let expected = 1.0_f32 / (i as f32 + 1.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-5,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_abs_f32() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| -(idx[0] as f32 + 1.0));
    let mut c = tensor_zeros::<f32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Abs };
    let plan = cpu_plan::<f32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0_f32, &[&a], 0.0_f32, &mut c).unwrap();

    for i in 0..3 {
        let expected = i as f32 + 1.0;
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-5,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_sqrt_f32() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| ((idx[0] + 1) * (idx[0] + 1)) as f32);
    let mut c = tensor_zeros::<f32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Sqrt };
    let plan = cpu_plan::<f32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0_f32, &[&a], 0.0_f32, &mut c).unwrap();

    for i in 0..3 {
        let expected = i as f32 + 1.0;
        assert!(
            (tensor_get(&c, &[i]) - expected).abs() < 1e-5,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// ElementwiseUnary -- Complex32 tests
// ============================================================================

#[test]
fn elementwise_unary_negate_complex32() {
    use num_complex::Complex32;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex32::new(idx[0] as f32 + 1.0, idx[0] as f32 + 2.0)
    });
    let mut c = tensor_zeros::<Complex32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Negate,
    };
    let plan = cpu_plan::<Complex32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex32::new(1.0, 0.0),
        &[&a],
        Complex32::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let expected = -Complex32::new(i as f32 + 1.0, i as f32 + 2.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-5,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_reciprocal_complex32() {
    use num_complex::Complex32;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex32::new(idx[0] as f32 + 1.0, idx[0] as f32 + 2.0)
    });
    let mut c = tensor_zeros::<Complex32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary {
        op: UnaryOp::Reciprocal,
    };
    let plan = cpu_plan::<Complex32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex32::new(1.0, 0.0),
        &[&a],
        Complex32::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let z = Complex32::new(i as f32 + 1.0, i as f32 + 2.0);
        let expected = Complex32::new(1.0, 0.0) / z;
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-5,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_abs_complex32() {
    use num_complex::Complex32;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        Complex32::new(3.0 * (idx[0] as f32 + 1.0), 4.0 * (idx[0] as f32 + 1.0))
    });
    let mut c = tensor_zeros::<Complex32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Abs };
    let plan = cpu_plan::<Complex32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex32::new(1.0, 0.0),
        &[&a],
        Complex32::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let expected = Complex32::new(5.0 * (i as f32 + 1.0), 0.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-4,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

#[test]
fn elementwise_unary_sqrt_complex32() {
    use num_complex::Complex32;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[3], |idx| {
        let z = Complex32::new(idx[0] as f32 + 1.0, 0.0);
        z * z
    });
    let mut c = tensor_zeros::<Complex32>(&[3]);

    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Sqrt };
    let plan = cpu_plan::<Complex32>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex32::new(1.0, 0.0),
        &[&a],
        Complex32::new(0.0, 0.0),
        &mut c,
    )
    .unwrap();

    for i in 0..3 {
        let expected = Complex32::new(i as f32 + 1.0, 0.0);
        assert!(
            (tensor_get(&c, &[i]) - expected).norm() < 1e-4,
            "C[{i}] = {}, expected {expected}",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// resolve_conj
// ============================================================================

#[test]
fn resolve_conj_non_conjugated() {
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
    let a = tensor_from_fn(&[], |_| 5.0_f64);
    let mut c = tensor_zeros::<f64>(&[3, 3]);

    let desc = PrimDescriptor::AntiTrace {
        modes_a: vec![],
        modes_c: vec![0, 1],
        paired: vec![(0, 1)],
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[], &[3, 3]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c).unwrap();

    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert!(
                    (tensor_get(&c, &[i, j]) - 5.0).abs() < 1e-10,
                    "C[{i},{j}] = {}, expected 5.0",
                    tensor_get(&c, &[i, j])
                );
            } else {
                assert!(
                    tensor_get(&c, &[i, j]).abs() < 1e-10,
                    "C[{i},{j}] = {}, expected 0.0",
                    tensor_get(&c, &[i, j])
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
    let a = tensor_from_fn(&[2, 3], |_| 1.0_f64);
    let mut c = tensor_from_fn(&[2], |_| 10.0_f64);

    let desc = PrimDescriptor::Reduce {
        modes_a: vec![0, 1],
        modes_c: vec![0],
        op: ReduceOp::Sum,
    };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[2, 3], &[2]]).unwrap();
    // C = 2 * sum(A, axis=1) + 3 * C
    // sum over 3 ones = 3, so C = 2 * 3 + 3 * 10 = 36
    cpu_execute(&mut ctx, &plan, 2.0, &[&a], 3.0, &mut c).unwrap();

    for i in 0..2 {
        assert!(
            (tensor_get(&c, &[i]) - 36.0).abs() < 1e-10,
            "C[{i}] = {}, expected 36.0",
            tensor_get(&c, &[i])
        );
    }
}

// ============================================================================
// Complex64 permute (original test kept)
// ============================================================================

#[test]
fn permute_complex64() {
    use num_complex::Complex64;

    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3], |idx| {
        Complex64::new((idx[0] * 3 + idx[1] + 1) as f64, 0.0)
    });
    let mut b = tensor_zeros::<Complex64>(&[3, 2]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<Complex64>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    cpu_execute(
        &mut ctx,
        &plan,
        Complex64::new(1.0, 0.0),
        &[&a],
        Complex64::new(0.0, 0.0),
        &mut b,
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(tensor_get(&b, &[j, i]), tensor_get(&a, &[i, j]));
        }
    }
}

// ============================================================================
// resolve_conj for Complex64
// ============================================================================

#[test]
fn resolve_conj_complex64_non_conjugated() {
    use num_complex::Complex64;

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
    let mut ctx = CpuContext::new(1);
    let data = vec![1.0_f64, 2.0, 3.0, 4.0];
    let t = Tensor::<f64>::from_slice(&data, &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let t_conj = t.into_conj();
    assert!(t_conj.is_conjugated());

    let resolved = CpuBackend::resolve_conj(&mut ctx, &t_conj);
    assert!(!resolved.is_conjugated());

    // For real types, conjugation is identity -- data should be unchanged
    let resolved_data = resolved
        .buffer()
        .as_slice()
        .expect("CPU tensor must have CPU-accessible data");
    for (orig, res) in data.iter().zip(resolved_data.iter()) {
        assert!((orig - res).abs() < 1e-15, "expected {orig}, got {res}");
    }
}

// ============================================================================
// f32 permute (original test kept)
// ============================================================================

#[test]
fn permute_f32() {
    let mut ctx = CpuContext::new(1);
    let a = tensor_from_fn(&[2, 3], |idx| (idx[0] * 3 + idx[1] + 1) as f32);
    let mut b = tensor_zeros::<f32>(&[3, 2]);

    let desc = PrimDescriptor::Permute {
        modes_a: vec![0, 1],
        modes_b: vec![1, 0],
    };
    let plan = cpu_plan::<f32>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
    cpu_execute(&mut ctx, &plan, 1.0_f32, &[&a], 0.0_f32, &mut b).unwrap();

    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(tensor_get(&b, &[j, i]), tensor_get(&a, &[i, j]));
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
fn plan_elementwise_unary_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    // Same rank but different dimensions: input [3,4] vs output [3,5]
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 5]]);
    assert!(result.is_err(), "expected error for shape mismatch");
}

#[test]
fn plan_elementwise_unary_rank_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    // Different ranks: input [3,4] vs output [3,4,2]
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4, 2]]);
    assert!(result.is_err(), "expected error for rank mismatch");
}

#[test]
fn plan_elementwise_mul_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseMul;
    // A=[3,4], B=[3,4], C=[3,5] — C dimension mismatch
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4], &[3, 5]]);
    assert!(result.is_err(), "expected error for shape mismatch");
}

#[test]
fn plan_elementwise_mul_rank_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseMul;
    // A=[3,4], B=[3,4], C=[3] — rank mismatch
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4], &[3]]);
    assert!(result.is_err(), "expected error for rank mismatch");
}

#[test]
fn plan_elementwise_mul_b_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseMul;
    // A=[3,4], B=[3,5], C=[3,4] — B dimension mismatch
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 5], &[3, 4]]);
    assert!(result.is_err(), "expected error for B shape mismatch");
}

#[test]
fn plan_make_contiguous_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::MakeContiguous;
    // input [3,4] vs output [3,5]
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 5]]);
    assert!(result.is_err(), "expected error for shape mismatch");
}

#[test]
fn plan_make_contiguous_rank_mismatch() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::MakeContiguous;
    // input [3,4] vs output [12]
    let result = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[12]]);
    assert!(result.is_err(), "expected error for rank mismatch");
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
    let a = tensor_zeros::<f64>(&[2, 3]);
    let b = tensor_zeros::<f64>(&[2, 3]);
    let mut c = tensor_zeros::<f64>(&[3, 2]);
    // Provide 2 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a, &b], 0.0, &mut c);
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
    let mut c = tensor_zeros::<f64>(&[3, 2]);
    // Provide 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c);
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
    let a = tensor_zeros::<f64>(&[2, 4]);
    let mut c = tensor_zeros::<f64>(&[2, 3]);
    // Only 1 input instead of 2
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c);
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
    let a = tensor_zeros::<f64>(&[2, 3]);
    let mut c = tensor_zeros::<f64>(&[2, 4]);
    // Only 1 input instead of 2
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c);
    assert!(result.is_err(), "expected error for wrong input count");
}

#[test]
fn execute_elementwise_mul_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseMul;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3], &[3]]).unwrap();
    let a = tensor_zeros::<f64>(&[3]);
    let mut c = tensor_zeros::<f64>(&[3]);
    // Only 1 input instead of 2
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a], 0.0, &mut c);
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
    let a = tensor_zeros::<f64>(&[3, 4]);
    let b = tensor_zeros::<f64>(&[3, 4]);
    let mut c = tensor_zeros::<f64>(&[3]);
    // 2 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[&a, &b], 0.0, &mut c);
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
    let mut c = tensor_zeros::<f64>(&[]);
    // 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c);
    assert!(result.is_err(), "expected error for zero inputs");
}

#[test]
fn execute_make_contiguous_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::MakeContiguous;
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
    let mut c = tensor_zeros::<f64>(&[3, 4]);
    // 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c);
    assert!(result.is_err(), "expected error for zero inputs");
}

#[test]
fn execute_elementwise_unary_wrong_input_count() {
    let mut ctx = CpuContext::new(1);
    let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
    let plan = cpu_plan::<f64>(&mut ctx, &desc, &[&[3], &[3]]).unwrap();
    let mut c = tensor_zeros::<f64>(&[3]);
    // 0 inputs instead of 1
    let result = cpu_execute(&mut ctx, &plan, 1.0, &[], 0.0, &mut c);
    assert!(result.is_err(), "expected error for zero inputs");
}

// Typed test scaffolding: macro that generates test modules per scalar type
// ============================================================================

/// Trait to abstract over scalar construction and approximate comparison.
/// This enables the typed test macro to work uniformly across f32, f64, Complex64.
trait TestScalar: tenferro_algebra::Scalar + std::fmt::Debug {
    /// Convert an integer value to this scalar type (for constructing test data).
    fn from_usize(v: usize) -> Self;
    /// Convert a f64 value to this scalar type.
    fn from_f64(v: f64) -> Self;
    /// Tolerance for approximate equality checks.
    fn tol() -> f64;
    /// Check approximate equality using per-type tolerance.
    fn approx_eq(a: Self, b: Self) -> bool;
    /// Norm of the difference (for error messages).
    fn diff_norm(a: Self, b: Self) -> f64;
}

impl TestScalar for f64 {
    fn from_usize(v: usize) -> Self {
        v as f64
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn tol() -> f64 {
        1e-10
    }
    fn approx_eq(a: Self, b: Self) -> bool {
        (a - b).abs() < Self::tol()
    }
    fn diff_norm(a: Self, b: Self) -> f64 {
        (a - b).abs()
    }
}

impl TestScalar for f32 {
    fn from_usize(v: usize) -> Self {
        v as f32
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn tol() -> f64 {
        1e-4
    }
    fn approx_eq(a: Self, b: Self) -> bool {
        (a - b).abs() < Self::tol() as f32
    }
    fn diff_norm(a: Self, b: Self) -> f64 {
        (a - b).abs() as f64
    }
}

impl TestScalar for num_complex::Complex64 {
    fn from_usize(v: usize) -> Self {
        num_complex::Complex64::new(v as f64, 0.0)
    }
    fn from_f64(v: f64) -> Self {
        num_complex::Complex64::new(v, 0.0)
    }
    fn tol() -> f64 {
        1e-10
    }
    fn approx_eq(a: Self, b: Self) -> bool {
        (a - b).norm() < Self::tol()
    }
    fn diff_norm(a: Self, b: Self) -> f64 {
        (a - b).norm()
    }
}

/// Macro to generate typed test modules for prims operations.
///
/// Each invocation creates a module `typed_$mod_name` containing the core
/// correctness tests parameterized for a specific scalar type `$T`.
macro_rules! typed_prims_tests {
    ($mod_name:ident, $T:ty) => {
        mod $mod_name {
            use super::*;
            use num_complex::Complex64;

            // Suppress unused-import warning for Complex64 in f32/f64 modules.
            const _: () = {
                fn _use_complex64() {
                    let _ = std::mem::size_of::<Complex64>();
                }
            };

            #[test]
            fn permute_transpose_2x3() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] + 1 + idx[1] * 2)
                });
                let mut b = tensor_zeros::<$T>(&[3, 2]);

                let desc = PrimDescriptor::Permute {
                    modes_a: vec![0, 1],
                    modes_b: vec![1, 0],
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut b,
                )
                .unwrap();

                for i in 0..2 {
                    for j in 0..3 {
                        assert_eq!(tensor_get(&b, &[j, i]), tensor_get(&a, &[i, j]));
                    }
                }
            }

            #[test]
            fn permute_with_alpha_beta() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] + idx[1] * 2 + 1)
                });
                let mut b = tensor_from_fn(&[3, 2], |_| <$T as TestScalar>::from_f64(1.0));

                let desc = PrimDescriptor::Permute {
                    modes_a: vec![0, 1],
                    modes_b: vec![1, 0],
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[2, 3], &[3, 2]]).unwrap();
                // B = 2 * A^T + 3 * B
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(2.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(3.0),
                    &mut b,
                )
                .unwrap();

                for i in 0..2 {
                    for j in 0..3 {
                        let expected = <$T as TestScalar>::from_f64(2.0) * tensor_get(&a, &[i, j])
                            + <$T as TestScalar>::from_f64(3.0);
                        assert!(
                            <$T as TestScalar>::approx_eq(tensor_get(&b, &[j, i]), expected),
                            "B[{},{}] = {:?}, expected {:?}, diff = {}",
                            j,
                            i,
                            tensor_get(&b, &[j, i]),
                            expected,
                            <$T as TestScalar>::diff_norm(tensor_get(&b, &[j, i]), expected)
                        );
                    }
                }
            }

            #[test]
            fn permute_3d() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[2, 3, 4], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 100 + idx[1] * 10 + idx[2])
                });
                let mut b = tensor_zeros::<$T>(&[4, 2, 3]);

                let desc = PrimDescriptor::Permute {
                    modes_a: vec![0, 1, 2],
                    modes_b: vec![2, 0, 1],
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[2, 3, 4], &[4, 2, 3]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut b,
                )
                .unwrap();

                for i in 0..2 {
                    for j in 0..3 {
                        for k in 0..4 {
                            assert_eq!(tensor_get(&b, &[k, i, j]), tensor_get(&a, &[i, j, k]));
                        }
                    }
                }
            }

            #[test]
            fn make_contiguous() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[3, 4], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 10 + idx[1])
                });
                let mut b = tensor_zeros::<$T>(&[3, 4]);

                let desc = PrimDescriptor::MakeContiguous;
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut b,
                )
                .unwrap();

                for i in 0..3 {
                    for j in 0..4 {
                        assert_eq!(tensor_get(&b, &[i, j]), tensor_get(&a, &[i, j]));
                    }
                }
            }

            #[test]
            fn contract_matrix_multiply() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 3 + idx[1] + 1)
                });
                let b = tensor_from_fn(&[3, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 1)
                });
                let mut c = tensor_zeros::<$T>(&[2, 2]);

                let desc = PrimDescriptor::Contract {
                    modes_a: vec![0, 1],
                    modes_b: vec![1, 2],
                    modes_c: vec![0, 2],
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[2, 3], &[3, 2], &[2, 2]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a, &b],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut c,
                )
                .unwrap();

                for i in 0..2 {
                    for j in 0..2 {
                        let mut expected = <$T as TestScalar>::from_f64(0.0);
                        for k in 0..3 {
                            expected = expected + tensor_get(&a, &[i, k]) * tensor_get(&b, &[k, j]);
                        }
                        assert!(
                            <$T as TestScalar>::approx_eq(tensor_get(&c, &[i, j]), expected),
                            "C[{i},{j}] = {:?}, expected {:?}, diff = {}",
                            tensor_get(&c, &[i, j]),
                            expected,
                            <$T as TestScalar>::diff_norm(tensor_get(&c, &[i, j]), expected)
                        );
                    }
                }
            }

            #[test]
            fn contract_outer_product() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[3], |idx| <$T as TestScalar>::from_usize(idx[0] + 1));
                let b = tensor_from_fn(&[4], |idx| <$T as TestScalar>::from_usize(idx[0] + 1));
                let mut c = tensor_zeros::<$T>(&[3, 4]);

                let desc = PrimDescriptor::Contract {
                    modes_a: vec![0],
                    modes_b: vec![1],
                    modes_c: vec![0, 1],
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3], &[4], &[3, 4]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a, &b],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut c,
                )
                .unwrap();

                for i in 0..3 {
                    for j in 0..4 {
                        let expected = <$T as TestScalar>::from_usize((i + 1) * (j + 1));
                        assert!(
                            <$T as TestScalar>::approx_eq(tensor_get(&c, &[i, j]), expected),
                            "C[{i},{j}] = {:?}, expected {:?}, diff = {}",
                            tensor_get(&c, &[i, j]),
                            expected,
                            <$T as TestScalar>::diff_norm(tensor_get(&c, &[i, j]), expected)
                        );
                    }
                }
            }

            #[test]
            fn reduce_sum_axis1() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[3, 4], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 10 + idx[1] + 1)
                });
                let mut c = tensor_zeros::<$T>(&[3]);

                let desc = PrimDescriptor::Reduce {
                    modes_a: vec![0, 1],
                    modes_c: vec![0],
                    op: ReduceOp::Sum,
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3, 4], &[3]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut c,
                )
                .unwrap();

                for i in 0..3 {
                    let mut expected = <$T as TestScalar>::from_f64(0.0);
                    for j in 0..4 {
                        expected = expected + tensor_get(&a, &[i, j]);
                    }
                    assert!(
                        <$T as TestScalar>::approx_eq(tensor_get(&c, &[i]), expected),
                        "C[{i}] = {:?}, expected {:?}, diff = {}",
                        tensor_get(&c, &[i]),
                        expected,
                        <$T as TestScalar>::diff_norm(tensor_get(&c, &[i]), expected)
                    );
                }
            }

            #[test]
            fn reduce_sum_full() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[3, 4], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] + idx[1] + 1)
                });
                let mut c = tensor_zeros::<$T>(&[]);

                let desc = PrimDescriptor::Reduce {
                    modes_a: vec![0, 1],
                    modes_c: vec![],
                    op: ReduceOp::Sum,
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3, 4], &[]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut c,
                )
                .unwrap();

                let mut expected = <$T as TestScalar>::from_f64(0.0);
                for i in 0..3 {
                    for j in 0..4 {
                        expected = expected + <$T as TestScalar>::from_usize(i + j + 1);
                    }
                }
                assert!(
                    <$T as TestScalar>::approx_eq(tensor_get(&c, &[]), expected),
                    "scalar = {:?}, expected {:?}, diff = {}",
                    tensor_get(&c, &[]),
                    expected,
                    <$T as TestScalar>::diff_norm(tensor_get(&c, &[]), expected)
                );
            }

            #[test]
            fn trace_2d_matrix() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[3, 3], |idx| {
                    if idx[0] == idx[1] {
                        <$T as TestScalar>::from_usize(idx[0] + 1)
                    } else {
                        <$T as TestScalar>::from_f64(0.0)
                    }
                });
                let mut c = tensor_zeros::<$T>(&[]);

                let desc = PrimDescriptor::Trace {
                    modes_a: vec![0, 1],
                    modes_c: vec![],
                    paired: vec![(0, 1)],
                };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3, 3], &[]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut c,
                )
                .unwrap();

                // tr(diag(1,2,3)) = 6
                let expected = <$T as TestScalar>::from_f64(6.0);
                assert!(
                    <$T as TestScalar>::approx_eq(tensor_get(&c, &[]), expected),
                    "trace = {:?}, expected {:?}, diff = {}",
                    tensor_get(&c, &[]),
                    expected,
                    <$T as TestScalar>::diff_norm(tensor_get(&c, &[]), expected)
                );
            }

            #[test]
            fn elementwise_mul_2d() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[3, 4], |idx| <$T as TestScalar>::from_usize(idx[0] + 1));
                let b = tensor_from_fn(&[3, 4], |idx| <$T as TestScalar>::from_usize(idx[1] + 1));
                let mut c = tensor_zeros::<$T>(&[3, 4]);

                let desc = PrimDescriptor::ElementwiseMul;
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3, 4], &[3, 4], &[3, 4]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a, &b],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut c,
                )
                .unwrap();

                for i in 0..3 {
                    for j in 0..4 {
                        let expected = <$T as TestScalar>::from_usize((i + 1) * (j + 1));
                        assert!(
                            <$T as TestScalar>::approx_eq(tensor_get(&c, &[i, j]), expected),
                            "C[{i},{j}] = {:?}, expected {:?}, diff = {}",
                            tensor_get(&c, &[i, j]),
                            expected,
                            <$T as TestScalar>::diff_norm(tensor_get(&c, &[i, j]), expected)
                        );
                    }
                }
            }

            #[test]
            fn elementwise_conj() {
                let mut ctx = CpuContext::new(1);
                let a = tensor_from_fn(&[3, 4], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 10 + idx[1] + 1)
                });
                let mut c = tensor_zeros::<$T>(&[3, 4]);

                let desc = PrimDescriptor::ElementwiseUnary { op: UnaryOp::Conj };
                let plan = cpu_plan::<$T>(&mut ctx, &desc, &[&[3, 4], &[3, 4]]).unwrap();
                cpu_execute(
                    &mut ctx,
                    &plan,
                    <$T as TestScalar>::from_f64(1.0),
                    &[&a],
                    <$T as TestScalar>::from_f64(0.0),
                    &mut c,
                )
                .unwrap();

                // For real types, conj is identity. For complex, values here
                // are purely real so conj is still identity.
                for i in 0..3 {
                    for j in 0..4 {
                        assert_eq!(tensor_get(&c, &[i, j]), tensor_get(&a, &[i, j]));
                    }
                }
            }
        }
    };
}

typed_prims_tests!(typed_f64, f64);
typed_prims_tests!(typed_f32, f32);
typed_prims_tests!(typed_complex64, num_complex::Complex64);
