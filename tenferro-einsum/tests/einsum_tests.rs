//! Tests for tenferro-einsum: subscript parsing, contraction tree,
//! einsum execution (single-tensor, pairwise, N-ary), AD rules.
//!
//! Core numeric tests are parameterized across f32, f64, and Complex64 via the
//! `typed_einsum_tests!` macro at the bottom of this file.

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::{
    einsum, einsum_frule, einsum_into, einsum_owned, einsum_rrule, einsum_with_plan,
    einsum_with_subscripts, tracked_einsum, ContractionTree, Subscripts,
};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;
const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

type S = Standard<f64>;

/// Helper: read a scalar (0-d) tensor value.
fn scalar_val(t: &Tensor<f64>) -> f64 {
    assert!(t.dims().is_empty(), "expected scalar tensor");
    t.buffer().as_slice().unwrap()[t.offset() as usize]
}

/// Helper: read element at multi-index from a tensor.
fn get(t: &Tensor<f64>, idx: &[usize]) -> f64 {
    let data = t.buffer().as_slice().unwrap();
    let pos = t.offset()
        + idx
            .iter()
            .zip(t.strides())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
    data[pos as usize]
}

// ============================================================================
// Subscripts parsing
// ============================================================================

#[test]
fn parse_matmul() {
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    assert_eq!(subs.inputs.len(), 2);
    assert_eq!(subs.inputs[0], vec![8, 9]); // i=8, j=9
    assert_eq!(subs.inputs[1], vec![9, 10]); // j=9, k=10
    assert_eq!(subs.output, vec![8, 10]); // i=8, k=10
}

#[test]
fn parse_trace() {
    let subs = Subscripts::parse("ii->").unwrap();
    assert_eq!(subs.inputs.len(), 1);
    assert_eq!(subs.inputs[0], vec![8, 8]); // i=8 repeated
    assert!(subs.output.is_empty());
}

#[test]
fn parse_with_parentheses() {
    let subs = Subscripts::parse("ij,(jk,kl)->il").unwrap();
    assert_eq!(subs.inputs.len(), 3);
    // Parentheses stripped, labels parsed correctly
    assert_eq!(subs.inputs[0], vec![8, 9]); // ij
    assert_eq!(subs.inputs[1], vec![9, 10]); // jk
    assert_eq!(subs.inputs[2], vec![10, 11]); // kl
    assert_eq!(subs.output, vec![8, 11]); // il
}

#[test]
fn parse_uppercase() {
    let subs = Subscripts::parse("AB,BC->AC").unwrap();
    assert_eq!(subs.inputs[0], vec![26, 27]); // A=26, B=27
    assert_eq!(subs.inputs[1], vec![27, 28]); // B=27, C=28
    assert_eq!(subs.output, vec![26, 28]); // A=26, C=28
}

#[test]
fn parse_invalid_no_arrow() {
    assert!(
        matches!(
            Subscripts::parse("ij,jk"),
            Err(tenferro_device::Error::InvalidArgument(ref msg)) if msg.contains("->")
        ),
        "expected InvalidArgument mentioning '->' for missing arrow"
    );
}

#[test]
fn parse_invalid_char() {
    assert!(
        matches!(
            Subscripts::parse("i1,1j->ij"),
            Err(tenferro_device::Error::InvalidArgument(ref msg)) if msg.contains("invalid")
        ),
        "expected InvalidArgument for invalid character"
    );
}

#[test]
fn subscripts_new() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    assert_eq!(subs.inputs.len(), 2);
    assert_eq!(subs.output, vec![0, 2]);
}

// ============================================================================
// ContractionTree
// ============================================================================

#[test]
fn contraction_tree_single_tensor() {
    let subs = Subscripts::new(&[&[0, 1]], &[1, 0]);
    let tree = ContractionTree::optimize(&subs, &[&[3, 4]]).unwrap();
    // Single tensor -> no steps
    let a = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let mut ctx = CpuContext::new(1);
    let result = einsum_with_plan::<f64, S, CpuBackend>(&mut ctx, &tree, &[&a], None).unwrap();
    assert_eq!(result.dims(), &[4, 3]);
}

#[test]
fn contraction_tree_two_tensors() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let tree = ContractionTree::optimize(&subs, &[&[2, 3], &[3, 4]]).unwrap();
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let mut ctx = CpuContext::new(1);
    let c = einsum_with_plan::<f64, S, CpuBackend>(&mut ctx, &tree, &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
}

#[test]
fn contraction_tree_from_pairs() {
    // A[ij] B[jk] C[kl] -> D[il]
    // Contract B*C first (pair 1,2 -> index 3), then A*T (pair 0,3)
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4], &[4, 5]], &[(1, 2), (0, 3)])
        .unwrap();
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let c_tensor = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ],
        &[4, 5],
        COL,
    )
    .unwrap();
    let mut ctx = CpuContext::new(1);
    let d = einsum_with_plan::<f64, S, CpuBackend>(&mut ctx, &tree, &[&a, &b, &c_tensor], None)
        .unwrap();
    assert_eq!(d.dims(), &[2, 5]);
}

// ============================================================================
// einsum: single-tensor operations
// ============================================================================

#[test]
fn einsum_identity() {
    let mut ctx = CpuContext::new(1);
    // a[ij] -> a[ij] (identity copy)
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = einsum::<f64, S, CpuBackend>(&mut ctx, "ij->ij", &[&a], None).unwrap();
    assert_eq!(b.dims(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(get(&b, &[i, j]), get(&a, &[i, j]));
        }
    }
}

#[test]
fn einsum_transpose() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = einsum::<f64, S, CpuBackend>(&mut ctx, "ij->ji", &[&a], None).unwrap();
    assert_eq!(b.dims(), &[3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(get(&b, &[j, i]), get(&a, &[i, j]));
        }
    }
}

#[test]
fn einsum_sum_reduce() {
    let mut ctx = CpuContext::new(1);
    // Sum over j: result_i = sum_j a_{ij}
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = einsum::<f64, S, CpuBackend>(&mut ctx, "ij->i", &[&a], None).unwrap();
    assert_eq!(b.dims(), &[2]);
    // a is column-major: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4, a[0,2]=5, a[1,2]=6
    // b[0] = a[0,0] + a[0,1] + a[0,2] = 1+3+5 = 9
    // b[1] = a[1,0] + a[1,1] + a[1,2] = 2+4+6 = 12
    assert!((get(&b, &[0]) - 9.0).abs() < 1e-10);
    assert!((get(&b, &[1]) - 12.0).abs() < 1e-10);
}

#[test]
fn einsum_full_contraction() {
    let mut ctx = CpuContext::new(1);
    // Sum all elements: scalar = sum_{ij} a_{ij}
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = einsum::<f64, S, CpuBackend>(&mut ctx, "ij->", &[&a], None).unwrap();
    assert!(b.dims().is_empty());
    assert!((scalar_val(&b) - 21.0).abs() < 1e-10);
}

#[test]
fn einsum_trace() {
    let mut ctx = CpuContext::new(1);
    // tr(A) = sum_i a_{ii}
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    // column-major: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4
    let tr = einsum::<f64, S, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
    assert!(tr.dims().is_empty());
    assert!((scalar_val(&tr) - 5.0).abs() < 1e-10); // 1 + 4 = 5
}

#[test]
fn einsum_diagonal_extraction() {
    let mut ctx = CpuContext::new(1);
    // diag(A) = a_{ii}
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3], COL)
        .unwrap();
    // column-major: a[0,0]=1, a[1,0]=2, a[2,0]=3, a[0,1]=4, a[1,1]=5, a[2,1]=6,
    //               a[0,2]=7, a[1,2]=8, a[2,2]=9
    let d = einsum::<f64, S, CpuBackend>(&mut ctx, "ii->i", &[&a], None).unwrap();
    assert_eq!(d.dims(), &[3]);
    assert!((get(&d, &[0]) - 1.0).abs() < 1e-10);
    assert!((get(&d, &[1]) - 5.0).abs() < 1e-10);
    assert!((get(&d, &[2]) - 9.0).abs() < 1e-10);
}

#[test]
fn einsum_diagonal_embedding() {
    let mut ctx = CpuContext::new(1);
    // Diagonal embedding: v_i -> diag(v)_{ii}
    let v = Tensor::<f64>::from_slice(&[2.0, 3.0, 5.0], &[3], COL).unwrap();
    let d = einsum::<f64, S, CpuBackend>(&mut ctx, "i->ii", &[&v], None).unwrap();
    assert_eq!(d.dims(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { get(&v, &[i]) } else { 0.0 };
            assert!(
                (get(&d, &[i, j]) - expected).abs() < 1e-10,
                "d[{i},{j}] = {}, expected {expected}",
                get(&d, &[i, j])
            );
        }
    }
}

// ============================================================================
// einsum: two-tensor operations
// ============================================================================

#[test]
fn einsum_matmul() {
    let mut ctx = CpuContext::new(1);
    // C = A @ B
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let c = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2, 4]);

    // Verify against manual computation
    for i in 0..2 {
        for k in 0..4 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += get(&a, &[i, j]) * get(&b, &[j, k]);
            }
            assert!(
                (get(&c, &[i, k]) - expected).abs() < 1e-10,
                "C[{i},{k}] = {}, expected {expected}",
                get(&c, &[i, k])
            );
        }
    }
}

#[test]
fn einsum_outer_product() {
    let mut ctx = CpuContext::new(1);
    let u = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], COL).unwrap();
    let v = Tensor::<f64>::from_slice(&[3.0, 4.0, 5.0], &[3], COL).unwrap();
    let m = einsum::<f64, S, CpuBackend>(&mut ctx, "i,j->ij", &[&u, &v], None).unwrap();
    assert_eq!(m.dims(), &[2, 3]);

    for i in 0..2 {
        for j in 0..3 {
            let expected = get(&u, &[i]) * get(&v, &[j]);
            assert!(
                (get(&m, &[i, j]) - expected).abs() < 1e-10,
                "M[{i},{j}] = {}, expected {expected}",
                get(&m, &[i, j])
            );
        }
    }
}

#[test]
fn einsum_dot_product() {
    let mut ctx = CpuContext::new(1);
    let u = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], COL).unwrap();
    let v = Tensor::<f64>::from_slice(&[4.0, 5.0, 6.0], &[3], COL).unwrap();
    let d = einsum::<f64, S, CpuBackend>(&mut ctx, "i,i->", &[&u, &v], None).unwrap();
    assert!(d.dims().is_empty());
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((scalar_val(&d) - 32.0).abs() < 1e-10);
}

#[test]
fn einsum_matvec() {
    let mut ctx = CpuContext::new(1);
    // y = A @ x
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let x = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], COL).unwrap();
    let y = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,j->i", &[&a, &x], None).unwrap();
    assert_eq!(y.dims(), &[2]);

    for i in 0..2 {
        let mut expected = 0.0;
        for j in 0..3 {
            expected += get(&a, &[i, j]) * get(&x, &[j]);
        }
        assert!(
            (get(&y, &[i]) - expected).abs() < 1e-10,
            "y[{i}] = {}, expected {expected}",
            get(&y, &[i])
        );
    }
}

#[test]
fn einsum_elementwise_mul() {
    let mut ctx = CpuContext::new(1);
    // Hadamard product: C_{ij} = A_{ij} * B_{ij}
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,ij->ij", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2, 2]);

    for i in 0..2 {
        for j in 0..2 {
            let expected = get(&a, &[i, j]) * get(&b, &[i, j]);
            assert!(
                (get(&c, &[i, j]) - expected).abs() < 1e-10,
                "C[{i},{j}] = {}, expected {expected}",
                get(&c, &[i, j])
            );
        }
    }
}

// ============================================================================
// einsum: N-ary contraction
// ============================================================================

#[test]
fn einsum_three_matrices() {
    let mut ctx = CpuContext::new(1);
    // D = A @ B @ C (auto-optimized order)
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[9.0, 10.0, 11.0, 12.0], &[2, 2], COL).unwrap();
    let d = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    assert_eq!(d.dims(), &[2, 2]);

    // Verify: D = A @ B @ C
    // First compute AB
    let ab = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    // Then ABC
    let abc = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&ab, &c], None).unwrap();

    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (get(&d, &[i, j]) - get(&abc, &[i, j])).abs() < 1e-10,
                "D[{i},{j}] = {}, expected {}",
                get(&d, &[i, j]),
                get(&abc, &[i, j])
            );
        }
    }
}

// ============================================================================
// einsum variants
// ============================================================================

#[test]
fn einsum_with_subscripts_matmul() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let c = einsum_with_subscripts::<f64, S, CpuBackend>(&mut ctx, &subs, &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
}

#[test]
fn einsum_owned_matmul() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let c = einsum_owned::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", vec![a, b], None).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
}

#[test]
fn einsum_into_overwrite() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let mut c = Tensor::<f64>::zeros(&[2, 2], MEM, COL);

    // C = 1.0 * (A @ B) + 0.0 * C
    einsum_into::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 1.0, 0.0, &mut c, None)
        .unwrap();

    for i in 0..2 {
        for k in 0..2 {
            let mut expected = 0.0;
            for j in 0..2 {
                expected += get(&a, &[i, j]) * get(&b, &[j, k]);
            }
            assert!(
                (get(&c, &[i, k]) - expected).abs() < 1e-10,
                "C[{i},{k}] = {}, expected {expected}",
                get(&c, &[i, k])
            );
        }
    }
}

#[test]
fn einsum_into_accumulate() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let mut c = Tensor::<f64>::ones(&[2, 2], MEM, COL);

    // C = 2.0 * (A @ B) + 3.0 * C
    einsum_into::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 2.0, 3.0, &mut c, None)
        .unwrap();

    for i in 0..2 {
        for k in 0..2 {
            let mut matmul = 0.0;
            for j in 0..2 {
                matmul += get(&a, &[i, j]) * get(&b, &[j, k]);
            }
            let expected = 2.0 * matmul + 3.0; // old C was ones
            assert!(
                (get(&c, &[i, k]) - expected).abs() < 1e-10,
                "C[{i},{k}] = {}, expected {expected}",
                get(&c, &[i, k])
            );
        }
    }
}

// ============================================================================
// einsum: plan cache integration
// ============================================================================

#[test]
fn einsum_reuses_cached_plans() {
    // Repeated einsum calls with the same contraction populate the plan cache
    // and subsequent calls reuse cached plans.
    let mut ctx = CpuContext::new(1);
    assert!(ctx.plan_cache_mut().is_empty());

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();

    // First call: populates cache
    let c1 = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    let cache_size_after_first = ctx.plan_cache_mut().len();
    assert!(
        cache_size_after_first > 0,
        "cache should be populated after first einsum call"
    );

    // Second call with same shapes: cache should not grow
    let c2 = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    let cache_size_after_second = ctx.plan_cache_mut().len();
    assert_eq!(
        cache_size_after_first, cache_size_after_second,
        "cache should not grow when same contraction is repeated"
    );

    // Results should be identical
    for i in 0..2 {
        for k in 0..4 {
            assert!(
                (get(&c1, &[i, k]) - get(&c2, &[i, k])).abs() < 1e-10,
                "results should be identical across calls"
            );
        }
    }
}

#[test]
fn einsum_different_shapes_miss_cache() {
    // Calling einsum with different-shaped tensors should produce cache misses.
    let mut ctx = CpuContext::new(1);

    // 2x3 @ 3x4
    let a1 = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b1 = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let _c1 = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a1, &b1], None).unwrap();
    let cache_size_1 = ctx.plan_cache_mut().len();

    // 3x2 @ 2x5 (different shapes)
    let a2 = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], COL).unwrap();
    let b2 = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 5],
        COL,
    )
    .unwrap();
    let _c2 = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a2, &b2], None).unwrap();
    let cache_size_2 = ctx.plan_cache_mut().len();

    assert!(
        cache_size_2 > cache_size_1,
        "different shapes should produce additional cache entries"
    );
}

// ============================================================================
// einsum: AD rules (standalone) -- kept on f64 only
// ============================================================================

#[test]
fn einsum_rrule_matmul() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let grad_c = Tensor::<f64>::ones(&[2, 4], MEM, COL);

    let grads =
        einsum_rrule::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c).unwrap();
    assert_eq!(grads.len(), 2);

    // grad_A = grad_C @ B^T: shape [2,4] x [4,3] = [2,3]
    assert_eq!(grads[0].dims(), &[2, 3]);
    // grad_B = A^T @ grad_C: shape [3,2] x [2,4] = [3,4]
    assert_eq!(grads[1].dims(), &[3, 4]);

    // Verify grad_A = einsum("ik,jk->ij", [grad_c, b]) = grad_c @ b^T
    let expected_ga =
        einsum::<f64, S, CpuBackend>(&mut ctx, "ik,jk->ij", &[&grad_c, &b], None).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (get(&grads[0], &[i, j]) - get(&expected_ga, &[i, j])).abs() < 1e-10,
                "grad_A[{i},{j}] mismatch"
            );
        }
    }

    // Verify grad_B = einsum("ij,ik->jk", [a, grad_c]) = a^T @ grad_c
    let expected_gb =
        einsum::<f64, S, CpuBackend>(&mut ctx, "ij,ik->jk", &[&a, &grad_c], None).unwrap();
    for i in 0..3 {
        for j in 0..4 {
            assert!(
                (get(&grads[1], &[i, j]) - get(&expected_gb, &[i, j])).abs() < 1e-10,
                "grad_B[{i},{j}] mismatch"
            );
        }
    }
}

#[test]
fn einsum_frule_matmul() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();
    let da = Tensor::<f64>::ones(&[2, 3], MEM, COL);

    // dC = einsum("ij,jk->ik", [dA, B]) since only A has a tangent
    let dc =
        einsum_frule::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), None])
            .unwrap();
    assert_eq!(dc.dims(), &[2, 4]);

    // Verify: dC = dA @ B
    let expected = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&da, &b], None).unwrap();
    for i in 0..2 {
        for k in 0..4 {
            assert!(
                (get(&dc, &[i, k]) - get(&expected, &[i, k])).abs() < 1e-10,
                "dC[{i},{k}] mismatch"
            );
        }
    }
}

#[test]
fn einsum_frule_both_tangents() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    let db = Tensor::<f64>::ones(&[2, 2], MEM, COL);

    // dC = dA @ B + A @ dB
    let dc = einsum_frule::<f64, S, CpuBackend>(
        &mut ctx,
        "ij,jk->ik",
        &[&a, &b],
        &[Some(&da), Some(&db)],
    )
    .unwrap();
    assert_eq!(dc.dims(), &[2, 2]);

    let term1 = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&da, &b], None).unwrap();
    let term2 = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &db], None).unwrap();

    for i in 0..2 {
        for k in 0..2 {
            let expected = get(&term1, &[i, k]) + get(&term2, &[i, k]);
            assert!(
                (get(&dc, &[i, k]) - expected).abs() < 1e-10,
                "dC[{i},{k}] = {}, expected {expected}",
                get(&dc, &[i, k])
            );
        }
    }
}

// ============================================================================
// Shape mismatch errors
// ============================================================================

#[test]
fn einsum_shape_mismatch() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    // j=3 in A but j=2 in B -> shape mismatch
    let result = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None);
    assert!(
        matches!(result, Err(tenferro_device::Error::ShapeMismatch { .. })),
        "expected ShapeMismatch, got: {:?}",
        result.as_ref().err()
    );
}

#[test]
fn einsum_wrong_operand_count() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    // Subscripts say 2 inputs but only 1 provided
    let result = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a], None);
    assert!(
        matches!(result, Err(tenferro_device::Error::InvalidArgument(_))),
        "expected InvalidArgument for wrong operand count, got: {:?}",
        result.as_ref().err()
    );
}

// ============================================================================
// Tracked einsum: end-to-end AD via tape recording
// ============================================================================

#[test]
fn tracked_einsum_matmul_pullback() {
    use chainrules::Tape;

    let mut ctx = CpuContext::new(1);

    // Create tape and leaf tensors
    let tape = Tape::<Tensor<f64>>::new();
    let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b_data = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();

    let a = tape.leaf(a_data.clone());
    let b = tape.leaf(b_data.clone());
    let a_id = a.node_id().expect("leaf should have node_id");
    let b_id = b.node_id().expect("leaf should have node_id");

    // C = A @ B  (tracked)
    let c = tracked_einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b])
        .expect("tracked_einsum should succeed");

    // loss = sum_{ij} C_{ij}^2  via "ij,ij->"
    let loss = tracked_einsum::<f64, S, CpuBackend>(&mut ctx, "ij,ij->", &[&c, &c])
        .expect("tracked_einsum for loss should succeed");

    // Pullback
    let grads = tape.pullback(&loss).expect("pullback should succeed");

    // Verify gradients exist for both leaves
    let ga = grads.get(a_id).expect("gradient for A should exist");
    let gb = grads.get(b_id).expect("gradient for B should exist");

    // Gradient shapes must match the input shapes
    assert_eq!(ga.dims(), &[2, 3], "grad_A shape mismatch");
    assert_eq!(gb.dims(), &[3, 4], "grad_B shape mismatch");

    // Verify grad values are non-zero
    // loss = sum_{ik} C_{ik}^2 where C = A @ B
    // d(loss)/d(C_{ik}) = 2 * C_{ik}
    // d(loss)/d(A_{ij}) = sum_k 2*C_{ik} * B_{jk}
    // At least some gradient entries should be non-zero since inputs are non-zero
    let ga_data = ga.buffer().as_slice().expect("should get ga slice");
    let gb_data = gb.buffer().as_slice().expect("should get gb slice");
    let ga_norm: f64 = ga_data.iter().map(|x| x * x).sum();
    let gb_norm: f64 = gb_data.iter().map(|x| x * x).sum();
    assert!(
        ga_norm > 0.0,
        "gradient for A should be non-zero, got norm^2 = {ga_norm}"
    );
    assert!(
        gb_norm > 0.0,
        "gradient for B should be non-zero, got norm^2 = {gb_norm}"
    );

    // Numerical verification: d(loss)/dA = 2 * C @ B^T
    // C = A @ B
    let c_val = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a_data, &b_data], None)
        .expect("einsum for C");
    // grad_C = 2 * C (since loss = sum C_{ik}^2)
    let two_c = einsum::<f64, S, CpuBackend>(
        &mut ctx,
        "ij,->ij",
        &[
            &c_val,
            &Tensor::<f64>::from_slice(&[2.0], &[], COL).unwrap(),
        ],
        None,
    )
    .expect("scale by 2");
    // grad_A = grad_C @ B^T = einsum("ik,jk->ij", [2*C, B])
    let expected_ga = einsum::<f64, S, CpuBackend>(&mut ctx, "ik,jk->ij", &[&two_c, &b_data], None)
        .expect("expected grad_A");
    // grad_B = A^T @ grad_C = einsum("ij,ik->jk", [A, 2*C])
    let expected_gb = einsum::<f64, S, CpuBackend>(&mut ctx, "ij,ik->jk", &[&a_data, &two_c], None)
        .expect("expected grad_B");

    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (get(ga, &[i, j]) - get(&expected_ga, &[i, j])).abs() < 1e-10,
                "grad_A[{i},{j}] = {}, expected {}",
                get(ga, &[i, j]),
                get(&expected_ga, &[i, j])
            );
        }
    }
    for i in 0..3 {
        for j in 0..4 {
            assert!(
                (get(gb, &[i, j]) - get(&expected_gb, &[i, j])).abs() < 1e-10,
                "grad_B[{i},{j}] = {}, expected {}",
                get(gb, &[i, j]),
                get(&expected_gb, &[i, j])
            );
        }
    }
}

// ============================================================================
// Typed test scaffolding: trait and macro for multi-scalar einsum tests
// ============================================================================

/// Trait to abstract over scalar construction and approximate comparison.
trait TestScalar: tenferro_algebra::Scalar + strided_traits::ScalarBase + std::fmt::Debug {
    fn from_usize(v: usize) -> Self;
    fn from_f64(v: f64) -> Self;
    fn tol() -> f64;
    fn approx_eq(a: Self, b: Self) -> bool;
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

/// Generic helper: read element at multi-index from a tensor of type T.
fn get_t<T: tenferro_algebra::Scalar>(t: &Tensor<T>, idx: &[usize]) -> T {
    let data = t.buffer().as_slice().unwrap();
    let pos = t.offset()
        + idx
            .iter()
            .zip(t.strides())
            .map(|(&i, &s)| i as isize * s)
            .sum::<isize>();
    data[pos as usize]
}

/// Generic helper: read scalar (0-d) tensor value.
fn scalar_val_t<T: tenferro_algebra::Scalar>(t: &Tensor<T>) -> T {
    assert!(t.dims().is_empty(), "expected scalar tensor");
    t.buffer().as_slice().unwrap()[t.offset() as usize]
}

/// Macro to generate typed test modules for einsum operations.
macro_rules! typed_einsum_tests {
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

            type TS = Standard<$T>;

            /// Build a Tensor<$T> from a closure over multi-indices.
            /// Data is produced in column-major flat order (leftmost index varies
            /// fastest), matching `Tensor::from_slice(..., COL)`.
            fn make_tensor(dims: &[usize], f: impl Fn(&[usize]) -> $T) -> Tensor<$T> {
                let n: usize = dims.iter().product();
                let mut data = Vec::with_capacity(n);
                let mut idx = vec![0usize; dims.len()];
                for _ in 0..n {
                    data.push(f(&idx));
                    // Increment multi-index (column-major: leftmost varies fastest)
                    for d in 0..dims.len() {
                        idx[d] += 1;
                        if idx[d] < dims[d] {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
                Tensor::<$T>::from_slice(&data, dims, COL).unwrap()
            }

            #[test]
            fn einsum_identity() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 3 + idx[1] + 1)
                });
                let b = einsum::<$T, TS, CpuBackend>(&mut ctx, "ij->ij", &[&a], None).unwrap();
                assert_eq!(b.dims(), &[2, 3]);
                for i in 0..2 {
                    for j in 0..3 {
                        assert_eq!(get_t(&b, &[i, j]), get_t(&a, &[i, j]));
                    }
                }
            }

            #[test]
            fn einsum_transpose() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 3 + idx[1] + 1)
                });
                let b = einsum::<$T, TS, CpuBackend>(&mut ctx, "ij->ji", &[&a], None).unwrap();
                assert_eq!(b.dims(), &[3, 2]);
                for i in 0..2 {
                    for j in 0..3 {
                        assert_eq!(get_t(&b, &[j, i]), get_t(&a, &[i, j]));
                    }
                }
            }

            #[test]
            fn einsum_sum_reduce() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 3 + idx[1] + 1)
                });
                let b = einsum::<$T, TS, CpuBackend>(&mut ctx, "ij->i", &[&a], None).unwrap();
                assert_eq!(b.dims(), &[2]);
                for i in 0..2 {
                    let mut expected = <$T as TestScalar>::from_f64(0.0);
                    for j in 0..3 {
                        expected = expected + get_t(&a, &[i, j]);
                    }
                    assert!(
                        <$T as TestScalar>::approx_eq(get_t(&b, &[i]), expected),
                        "b[{i}] = {:?}, expected {:?}, diff = {}",
                        get_t(&b, &[i]),
                        expected,
                        <$T as TestScalar>::diff_norm(get_t(&b, &[i]), expected)
                    );
                }
            }

            #[test]
            fn einsum_full_contraction() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 3 + idx[1] + 1)
                });
                let b = einsum::<$T, TS, CpuBackend>(&mut ctx, "ij->", &[&a], None).unwrap();
                assert!(b.dims().is_empty());
                // sum of 1..6 = 21
                let expected = <$T as TestScalar>::from_f64(21.0);
                assert!(
                    <$T as TestScalar>::approx_eq(scalar_val_t(&b), expected),
                    "scalar = {:?}, expected {:?}",
                    scalar_val_t(&b),
                    expected
                );
            }

            #[test]
            fn einsum_trace() {
                let mut ctx = CpuContext::new(1);
                // 2x2 matrix with known diagonal values
                let a = make_tensor(&[2, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 1)
                });
                // a[0,0]=1, a[0,1]=2, a[1,0]=3, a[1,1]=4
                // trace = a[0,0] + a[1,1] = 1 + 4 = 5
                let tr = einsum::<$T, TS, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
                assert!(tr.dims().is_empty());
                let expected = get_t(&a, &[0, 0]) + get_t(&a, &[1, 1]);
                assert!(
                    <$T as TestScalar>::approx_eq(scalar_val_t(&tr), expected),
                    "trace = {:?}, expected {:?}",
                    scalar_val_t(&tr),
                    expected
                );
            }

            #[test]
            fn einsum_matmul() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 3], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 3 + idx[1] + 1)
                });
                let b = make_tensor(&[3, 4], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 4 + idx[1] + 1)
                });
                let c =
                    einsum::<$T, TS, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
                assert_eq!(c.dims(), &[2, 4]);

                for i in 0..2 {
                    for k in 0..4 {
                        let mut expected = <$T as TestScalar>::from_f64(0.0);
                        for j in 0..3 {
                            expected = expected + get_t(&a, &[i, j]) * get_t(&b, &[j, k]);
                        }
                        assert!(
                            <$T as TestScalar>::approx_eq(get_t(&c, &[i, k]), expected),
                            "C[{i},{k}] = {:?}, expected {:?}, diff = {}",
                            get_t(&c, &[i, k]),
                            expected,
                            <$T as TestScalar>::diff_norm(get_t(&c, &[i, k]), expected)
                        );
                    }
                }
            }

            #[test]
            fn einsum_outer_product() {
                let mut ctx = CpuContext::new(1);
                let u = make_tensor(&[2], |idx| <$T as TestScalar>::from_usize(idx[0] + 1));
                let v = make_tensor(&[3], |idx| <$T as TestScalar>::from_usize(idx[0] + 3));
                let m = einsum::<$T, TS, CpuBackend>(&mut ctx, "i,j->ij", &[&u, &v], None).unwrap();
                assert_eq!(m.dims(), &[2, 3]);

                for i in 0..2 {
                    for j in 0..3 {
                        let expected = get_t(&u, &[i]) * get_t(&v, &[j]);
                        assert!(
                            <$T as TestScalar>::approx_eq(get_t(&m, &[i, j]), expected),
                            "M[{i},{j}] = {:?}, expected {:?}",
                            get_t(&m, &[i, j]),
                            expected
                        );
                    }
                }
            }

            #[test]
            fn einsum_dot_product() {
                let mut ctx = CpuContext::new(1);
                let u = make_tensor(&[3], |idx| <$T as TestScalar>::from_usize(idx[0] + 1));
                let v = make_tensor(&[3], |idx| <$T as TestScalar>::from_usize(idx[0] + 4));
                let d = einsum::<$T, TS, CpuBackend>(&mut ctx, "i,i->", &[&u, &v], None).unwrap();
                assert!(d.dims().is_empty());
                // u = [1,2,3], v = [4,5,6], dot = 4+10+18 = 32
                let expected = <$T as TestScalar>::from_f64(32.0);
                assert!(
                    <$T as TestScalar>::approx_eq(scalar_val_t(&d), expected),
                    "dot = {:?}, expected {:?}",
                    scalar_val_t(&d),
                    expected
                );
            }

            #[test]
            fn einsum_elementwise_mul() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 1)
                });
                let b = make_tensor(&[2, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 5)
                });
                let c =
                    einsum::<$T, TS, CpuBackend>(&mut ctx, "ij,ij->ij", &[&a, &b], None).unwrap();
                assert_eq!(c.dims(), &[2, 2]);

                for i in 0..2 {
                    for j in 0..2 {
                        let expected = get_t(&a, &[i, j]) * get_t(&b, &[i, j]);
                        assert!(
                            <$T as TestScalar>::approx_eq(get_t(&c, &[i, j]), expected),
                            "C[{i},{j}] = {:?}, expected {:?}",
                            get_t(&c, &[i, j]),
                            expected
                        );
                    }
                }
            }

            #[test]
            fn einsum_three_matrices() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 1)
                });
                let b = make_tensor(&[2, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 5)
                });
                let c = make_tensor(&[2, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 9)
                });
                let d = einsum::<$T, TS, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None)
                    .unwrap();
                assert_eq!(d.dims(), &[2, 2]);

                // Verify: D = A @ B @ C
                let ab =
                    einsum::<$T, TS, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
                let abc =
                    einsum::<$T, TS, CpuBackend>(&mut ctx, "ij,jk->ik", &[&ab, &c], None).unwrap();

                for i in 0..2 {
                    for j in 0..2 {
                        assert!(
                            <$T as TestScalar>::approx_eq(get_t(&d, &[i, j]), get_t(&abc, &[i, j])),
                            "D[{i},{j}] = {:?}, expected {:?}",
                            get_t(&d, &[i, j]),
                            get_t(&abc, &[i, j])
                        );
                    }
                }
            }
        }
    };
}

typed_einsum_tests!(typed_f64, f64);
typed_einsum_tests!(typed_f32, f32);
typed_einsum_tests!(typed_complex64, num_complex::Complex64);
