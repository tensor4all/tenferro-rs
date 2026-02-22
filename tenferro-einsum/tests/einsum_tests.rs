//! Tests for tenferro-einsum: subscript parsing, contraction tree,
//! einsum execution (single-tensor, pairwise, N-ary), AD rules.

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::{
    einsum, einsum_frule, einsum_into, einsum_owned, einsum_rrule, einsum_with_plan,
    einsum_with_subscripts, ContractionTree, Subscripts,
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
    // Single tensor → no steps
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
    // Contract B*C first (pair 1,2 → index 3), then A*T (pair 0,3)
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
// einsum: AD rules (standalone)
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

    // grad_A = ḡ_C @ B^T: shape [2,4] x [4,3] = [2,3]
    assert_eq!(grads[0].dims(), &[2, 3]);
    // grad_B = A^T @ ḡ_C: shape [3,2] x [2,4] = [3,4]
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
    // j=3 in A but j=2 in B → shape mismatch
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
