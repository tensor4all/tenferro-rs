//! Tests for tenferro-einsum: subscript parsing, contraction tree,
//! einsum execution (single-tensor, pairwise, N-ary), AD rules.
//!
//! Core numeric tests are parameterized across f32, f64, and Complex64 via the
//! `typed_einsum_tests!` macro at the bottom of this file.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_einsum::{
    dual_einsum, einsum, einsum_frule, einsum_into, einsum_owned, einsum_rrule, einsum_with_path,
    einsum_with_path_into, einsum_with_plan, einsum_with_plan_owned, einsum_with_subscripts,
    einsum_with_subscripts_into, einsum_with_subscripts_owned, tracked_einsum, ContractionTree,
    Subscripts,
};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;
const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;

type S = Standard<f64>;

fn poison_mutex<T>(mutex: &Arc<Mutex<T>>) {
    let mutex = Arc::clone(mutex);
    let _ = catch_unwind(AssertUnwindSafe(move || {
        let _guard = mutex.lock().unwrap();
        panic!("poison backend mutex");
    }));
}

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

/// Helper: read all elements in col-major (Fortran) logical order.
/// Works correctly regardless of the tensor's physical memory layout.
fn to_col_major_vec(t: &Tensor<f64>) -> Vec<f64> {
    let dims = t.dims();
    let ndim = dims.len();
    let total: usize = dims.iter().product();
    let mut result = Vec::with_capacity(total);
    let mut index = vec![0usize; ndim];
    for _ in 0..total {
        result.push(get(t, &index));
        // Increment index in col-major order (leftmost axis varies fastest)
        for d in 0..ndim {
            index[d] += 1;
            if index[d] < dims[d] {
                break;
            }
            index[d] = 0;
        }
    }
    result
}

/// Helper: assert two tensors are element-wise equal (col-major logical order).
fn assert_tensors_close(a: &Tensor<f64>, b: &Tensor<f64>, label: &str) {
    assert_eq!(a.dims(), b.dims(), "{label}: shape mismatch");
    let va = to_col_major_vec(a);
    let vb = to_col_major_vec(b);
    for i in 0..va.len() {
        assert!(
            (va[i] - vb[i]).abs() < 1e-10,
            "{label}[{i}]: a={}, b={}",
            va[i],
            vb[i]
        );
    }
}

// ============================================================================
// Subscripts parsing
// ============================================================================

#[test]
fn parse_matmul() {
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    assert_eq!(subs.inputs.len(), 2);
    assert_eq!(subs.inputs[0], vec!['i' as u32, 'j' as u32]);
    assert_eq!(subs.inputs[1], vec!['j' as u32, 'k' as u32]);
    assert_eq!(subs.output, vec!['i' as u32, 'k' as u32]);
}

#[test]
fn parse_trace() {
    let subs = Subscripts::parse("ii->").unwrap();
    assert_eq!(subs.inputs.len(), 1);
    assert_eq!(subs.inputs[0], vec!['i' as u32, 'i' as u32]);
    assert!(subs.output.is_empty());
}

#[test]
fn parse_with_parentheses() {
    let subs = Subscripts::parse("ij,(jk,kl)->il").unwrap();
    assert_eq!(subs.inputs.len(), 3);
    // Parentheses stripped, labels parsed correctly
    assert_eq!(subs.inputs[0], vec!['i' as u32, 'j' as u32]);
    assert_eq!(subs.inputs[1], vec!['j' as u32, 'k' as u32]);
    assert_eq!(subs.inputs[2], vec!['k' as u32, 'l' as u32]);
    assert_eq!(subs.output, vec!['i' as u32, 'l' as u32]);
}

#[test]
fn parse_uppercase() {
    let subs = Subscripts::parse("AB,BC->AC").unwrap();
    assert_eq!(subs.inputs[0], vec!['A' as u32, 'B' as u32]);
    assert_eq!(subs.inputs[1], vec!['B' as u32, 'C' as u32]);
    assert_eq!(subs.output, vec!['A' as u32, 'C' as u32]);
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
    // Control characters (e.g. null) must be rejected.
    assert!(
        matches!(
            Subscripts::parse("i\u{0},\u{0}j->ij"),
            Err(tenferro_device::Error::InvalidArgument(ref msg)) if msg.contains("control") || msg.contains("invalid")
        ),
        "expected InvalidArgument for control character"
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
    let result = einsum_with_plan::<S, CpuBackend>(&mut ctx, &tree, &[&a], None).unwrap();
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
    let c = einsum_with_plan::<S, CpuBackend>(&mut ctx, &tree, &[&a, &b], None).unwrap();
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
    let d = einsum_with_plan::<S, CpuBackend>(&mut ctx, &tree, &[&a, &b, &c_tensor], None).unwrap();
    assert_eq!(d.dims(), &[2, 5]);
}

#[test]
fn contraction_tree_from_pairs_rejects_wrong_step_count() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let result = ContractionTree::from_pairs(&subs, &[&[2, 2], &[2, 2], &[2, 2]], &[(1, 2)]);
    assert!(result.is_err(), "wrong number of path steps must error");
}

#[test]
fn contraction_tree_from_pairs_rejects_self_pair() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let result =
        ContractionTree::from_pairs(&subs, &[&[2, 2], &[2, 2], &[2, 2]], &[(0, 0), (1, 3)]);
    assert!(result.is_err(), "self-pair contraction must error");
}

#[test]
fn contraction_tree_from_pairs_rejects_reused_consumed_operand() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let result =
        ContractionTree::from_pairs(&subs, &[&[2, 2], &[2, 2], &[2, 2]], &[(0, 1), (0, 2)]);
    assert!(result.is_err(), "reusing a consumed operand must error");
}

// ============================================================================
// einsum: single-tensor operations
// ============================================================================

#[test]
fn einsum_identity() {
    let mut ctx = CpuContext::new(1);
    // a[ij] -> a[ij] (identity copy)
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = einsum::<S, CpuBackend>(&mut ctx, "ij->ij", &[&a], None).unwrap();
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
    let b = einsum::<S, CpuBackend>(&mut ctx, "ij->ji", &[&a], None).unwrap();
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
    let b = einsum::<S, CpuBackend>(&mut ctx, "ij->i", &[&a], None).unwrap();
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
    let b = einsum::<S, CpuBackend>(&mut ctx, "ij->", &[&a], None).unwrap();
    assert!(b.dims().is_empty());
    assert!((scalar_val(&b) - 21.0).abs() < 1e-10);
}

#[test]
fn einsum_trace() {
    let mut ctx = CpuContext::new(1);
    // tr(A) = sum_i a_{ii}
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    // column-major: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4
    let tr = einsum::<S, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
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
    let d = einsum::<S, CpuBackend>(&mut ctx, "ii->i", &[&a], None).unwrap();
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
    let d = einsum::<S, CpuBackend>(&mut ctx, "i->ii", &[&v], None).unwrap();
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
    let c = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
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
    let m = einsum::<S, CpuBackend>(&mut ctx, "i,j->ij", &[&u, &v], None).unwrap();
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
    let d = einsum::<S, CpuBackend>(&mut ctx, "i,i->", &[&u, &v], None).unwrap();
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
    let y = einsum::<S, CpuBackend>(&mut ctx, "ij,j->i", &[&a, &x], None).unwrap();
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
    let c = einsum::<S, CpuBackend>(&mut ctx, "ij,ij->ij", &[&a, &b], None).unwrap();
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
    let d = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    assert_eq!(d.dims(), &[2, 2]);

    // Verify: D = A @ B @ C
    // First compute AB
    let ab = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    // Then ABC
    let abc = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&ab, &c], None).unwrap();

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
    let c = einsum_with_subscripts::<S, CpuBackend>(&mut ctx, &subs, &[&a, &b], None).unwrap();
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
    let c = einsum_owned::<S, CpuBackend>(&mut ctx, "ij,jk->ik", vec![a, b], None).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
}

#[test]
fn einsum_with_subscripts_owned_matches_borrowed() {
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

    let expected =
        einsum_with_subscripts::<S, CpuBackend>(&mut ctx, &subs, &[&a, &b], None).unwrap();
    let got =
        einsum_with_subscripts_owned::<S, CpuBackend>(&mut ctx, &subs, vec![a, b], None).unwrap();
    assert_tensors_close(&got, &expected, "with_subscripts_owned");
}

#[test]
fn einsum_with_plan_owned_matches_borrowed() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let shapes: &[&[usize]] = &[&[2, 3], &[3, 4]];
    let tree = ContractionTree::optimize(&subs, shapes).unwrap();

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        COL,
    )
    .unwrap();

    let expected = einsum_with_plan::<S, CpuBackend>(&mut ctx, &tree, &[&a, &b], None).unwrap();
    let got = einsum_with_plan_owned::<S, CpuBackend>(&mut ctx, &tree, vec![a, b], None).unwrap();
    assert_tensors_close(&got, &expected, "with_plan_owned");
}

#[test]
fn einsum_with_path_matches_flat_nary() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let pairs = vec![(1, 2), (0, 3)];

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[9.0, 10.0, 11.0, 12.0], &[2, 2], COL).unwrap();

    let via_path =
        einsum_with_path::<S, CpuBackend>(&mut ctx, &subs, &pairs, &[&a, &b, &c], None).unwrap();
    let flat = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    assert_tensors_close(&via_path, &flat, "with_path");
}

#[test]
fn einsum_with_path_invalid_pairs_errors() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let bad_pairs = vec![(0, 99), (0, 3)];

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[9.0, 10.0, 11.0, 12.0], &[2, 2], COL).unwrap();

    let result =
        einsum_with_path::<S, CpuBackend>(&mut ctx, &subs, &bad_pairs, &[&a, &b, &c], None);
    assert!(result.is_err(), "invalid contraction path must error");
}

#[test]
fn einsum_with_path_rejects_structurally_invalid_paths() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[9.0, 10.0, 11.0, 12.0], &[2, 2], COL).unwrap();
    let invalid_paths = [
        (vec![(1, 2)], "wrong step count"),
        (vec![(0, 0), (1, 3)], "self pair"),
        (vec![(0, 1), (0, 2)], "reused consumed operand"),
    ];

    for (pairs, desc) in invalid_paths {
        let with_path =
            einsum_with_path::<S, CpuBackend>(&mut ctx, &subs, &pairs, &[&a, &b, &c], None);
        assert!(
            with_path.is_err(),
            "{desc} must be rejected by einsum_with_path"
        );

        let mut out = Tensor::<f64>::zeros(&[2, 2], MEM, COL);
        let with_path_into = einsum_with_path_into::<S, CpuBackend>(
            &mut ctx,
            &subs,
            &pairs,
            &[&a, &b, &c],
            1.0,
            0.0,
            &mut out,
            None,
        );
        assert!(
            with_path_into.is_err(),
            "{desc} must be rejected by einsum_with_path_into"
        );
    }
}

#[test]
fn einsum_with_subscripts_into_accumulate() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let expected_mm =
        einsum_with_subscripts::<S, CpuBackend>(&mut ctx, &subs, &[&a, &b], None).unwrap();

    let mut out = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    einsum_with_subscripts_into::<S, CpuBackend>(
        &mut ctx,
        &subs,
        &[&a, &b],
        2.0,
        3.0,
        &mut out,
        None,
    )
    .unwrap();

    let got = out.buffer().as_slice().unwrap();
    let mm = expected_mm.buffer().as_slice().unwrap();
    for i in 0..got.len() {
        let expected = 2.0 * mm[i] + 3.0;
        assert!(
            (got[i] - expected).abs() < 1e-10,
            "with_subscripts_into[{i}] got={}, expected={expected}",
            got[i]
        );
    }
}

#[test]
fn einsum_with_path_into_accumulate() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let pairs = vec![(1, 2), (0, 3)];

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[9.0, 10.0, 11.0, 12.0], &[2, 2], COL).unwrap();
    let base =
        einsum_with_path::<S, CpuBackend>(&mut ctx, &subs, &pairs, &[&a, &b, &c], None).unwrap();

    let mut out = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    einsum_with_path_into::<S, CpuBackend>(
        &mut ctx,
        &subs,
        &pairs,
        &[&a, &b, &c],
        2.0,
        3.0,
        &mut out,
        None,
    )
    .unwrap();

    let got = out.buffer().as_slice().unwrap();
    let base_data = base.buffer().as_slice().unwrap();
    for i in 0..got.len() {
        let expected = 2.0 * base_data[i] + 3.0;
        assert!(
            (got[i] - expected).abs() < 1e-10,
            "with_path_into[{i}] got={}, expected={expected}",
            got[i]
        );
    }
}

#[test]
fn einsum_into_overwrite() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let mut c = Tensor::<f64>::zeros(&[2, 2], MEM, COL);

    // C = 1.0 * (A @ B) + 0.0 * C
    einsum_into::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 1.0, 0.0, &mut c, None).unwrap();

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
    einsum_into::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], 2.0, 3.0, &mut c, None).unwrap();

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
    let c1 = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    let cache_size_after_first = ctx.plan_cache_mut().len();
    assert!(
        cache_size_after_first > 0,
        "cache should be populated after first einsum call"
    );

    // Second call with same shapes: cache should not grow
    let c2 = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
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
    let _c1 = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a1, &b1], None).unwrap();
    let cache_size_1 = ctx.plan_cache_mut().len();

    // 3x2 @ 2x5 (different shapes)
    let a2 = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], COL).unwrap();
    let b2 = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[2, 5],
        COL,
    )
    .unwrap();
    let _c2 = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a2, &b2], None).unwrap();
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

    let grads = einsum_rrule::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &grad_c).unwrap();
    assert_eq!(grads.len(), 2);

    // grad_A = grad_C @ B^T: shape [2,4] x [4,3] = [2,3]
    assert_eq!(grads[0].dims(), &[2, 3]);
    // grad_B = A^T @ grad_C: shape [3,2] x [2,4] = [3,4]
    assert_eq!(grads[1].dims(), &[3, 4]);

    // Verify grad_A = einsum("ik,jk->ij", [grad_c, b]) = grad_c @ b^T
    let expected_ga = einsum::<S, CpuBackend>(&mut ctx, "ik,jk->ij", &[&grad_c, &b], None).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            assert!(
                (get(&grads[0], &[i, j]) - get(&expected_ga, &[i, j])).abs() < 1e-10,
                "grad_A[{i},{j}] mismatch"
            );
        }
    }

    // Verify grad_B = einsum("ij,ik->jk", [a, grad_c]) = a^T @ grad_c
    let expected_gb = einsum::<S, CpuBackend>(&mut ctx, "ij,ik->jk", &[&a, &grad_c], None).unwrap();
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
    let dc = einsum_frule::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), None])
        .unwrap();
    assert_eq!(dc.dims(), &[2, 4]);

    // Verify: dC = dA @ B
    let expected = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&da, &b], None).unwrap();
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
    let dc =
        einsum_frule::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), Some(&db)])
            .unwrap();
    assert_eq!(dc.dims(), &[2, 2]);

    let term1 = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&da, &b], None).unwrap();
    let term2 = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &db], None).unwrap();

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
    let result = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None);
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
    let result = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a], None);
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
    use std::sync::{Arc, Mutex};

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));

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
    let c = tracked_einsum::<S, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b])
        .expect("tracked_einsum should succeed");

    // loss = sum_{ij} C_{ij}^2  via "ij,ij->"
    let loss = tracked_einsum::<S, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c])
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
    let c_val = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,jk->ik",
        &[&a_data, &b_data],
        None,
    )
    .expect("einsum for C");
    // grad_C = 2 * C (since loss = sum C_{ik}^2)
    let two_c = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,->ij",
        &[
            &c_val,
            &Tensor::<f64>::from_slice(&[2.0], &[], COL).unwrap(),
        ],
        None,
    )
    .expect("scale by 2");
    // grad_A = grad_C @ B^T = einsum("ik,jk->ij", [2*C, B])
    let expected_ga = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ik,jk->ij",
        &[&two_c, &b_data],
        None,
    )
    .expect("expected grad_A");
    // grad_B = A^T @ grad_C = einsum("ij,ik->jk", [A, 2*C])
    let expected_gb = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,ik->jk",
        &[&a_data, &two_c],
        None,
    )
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

#[test]
fn tracked_einsum_rejects_mixed_tapes() {
    use chainrules::Tape;
    use std::sync::{Arc, Mutex};

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));

    let tape1 = Tape::<Tensor<f64>>::new();
    let tape2 = Tape::<Tensor<f64>>::new();

    let a_data = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b_data = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    let a = tape1.leaf(a_data);
    let b = tape2.leaf(b_data);

    let result = tracked_einsum::<S, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]);
    assert!(result.is_err(), "expected error for mixed-tape operands");
}

#[test]
fn tracked_einsum_without_grad_returns_plain_tracked_tensor() {
    use chainrules::TrackedValue;

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let a =
        TrackedValue::new(Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap());
    let b =
        TrackedValue::new(Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap());

    let result = tracked_einsum::<S, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();

    assert!(result.node_id().is_none());
    assert!(!result.requires_grad());
    assert_eq!(result.value().dims(), &[2, 2]);
}

#[test]
fn tracked_einsum_rejects_invalid_subscripts() {
    use chainrules::TrackedValue;

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let a = TrackedValue::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let b = TrackedValue::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));

    let err = tracked_einsum::<S, CpuBackend>(ctx, "ij,jk", &[&a, &b])
        .err()
        .unwrap();
    assert!(matches!(err, chainrules::AutodiffError::InvalidArgument(msg) if msg.contains("->")));
}

#[test]
fn tracked_einsum_rejects_poisoned_backend_context_on_entry() {
    use chainrules::TrackedValue;

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let a = TrackedValue::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let b = TrackedValue::new(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    poison_mutex(&ctx);

    let err = tracked_einsum::<S, CpuBackend>(ctx, "ij,jk->ik", &[&a, &b])
        .err()
        .unwrap();
    assert!(
        matches!(err, chainrules::AutodiffError::InvalidArgument(msg) if msg.contains("poisoned"))
    );
}

#[test]
fn tracked_einsum_pullback_rejects_poisoned_backend_context() {
    use chainrules::Tape;

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let tape = Tape::<Tensor<f64>>::new();
    let a = tape.leaf(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let b = tape.leaf(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let out = tracked_einsum::<S, CpuBackend>(Arc::clone(&ctx), "ij,jk->ik", &[&a, &b]).unwrap();
    let loss = tracked_einsum::<S, CpuBackend>(Arc::clone(&ctx), "ij,ij->", &[&out, &out]).unwrap();

    poison_mutex(&ctx);

    let err = tape.pullback(&loss).err().unwrap();
    assert!(
        matches!(err, chainrules::AutodiffError::InvalidArgument(msg) if msg.contains("poisoned"))
    );
}

#[test]
fn tracked_einsum_hvp_rejects_poisoned_backend_context() {
    use chainrules::Tape;

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let tape = Tape::<Tensor<f64>>::new();
    let mut a_data = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    a_data.set_fw_grad(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let a = tape.leaf(a_data);
    let b = tape.leaf(Tensor::<f64>::ones(&[2, 2], MEM, COL));
    let out = tracked_einsum::<S, CpuBackend>(Arc::clone(&ctx), "ij,jk->ik", &[&a, &b]).unwrap();
    let loss = tracked_einsum::<S, CpuBackend>(Arc::clone(&ctx), "ij,ij->", &[&out, &out]).unwrap();

    poison_mutex(&ctx);

    let err = tape.hvp(&loss).err().unwrap();
    assert!(
        matches!(err, chainrules::AutodiffError::InvalidArgument(msg) if msg.contains("poisoned"))
    );
}

// ============================================================================
// Typed test scaffolding: trait and macro for multi-scalar einsum tests
// ============================================================================

/// Trait to abstract over scalar construction and approximate comparison.
trait TestScalar: tenferro_algebra::Scalar + std::fmt::Debug {
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
                let b = einsum::<TS, CpuBackend>(&mut ctx, "ij->ij", &[&a], None).unwrap();
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
                let b = einsum::<TS, CpuBackend>(&mut ctx, "ij->ji", &[&a], None).unwrap();
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
                let b = einsum::<TS, CpuBackend>(&mut ctx, "ij->i", &[&a], None).unwrap();
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
                let b = einsum::<TS, CpuBackend>(&mut ctx, "ij->", &[&a], None).unwrap();
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
                let tr = einsum::<TS, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
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
            fn einsum_binary_multi_repeat_contract() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 2, 2, 3], |idx| {
                    <$T as TestScalar>::from_usize(
                        idx[0] + 10 * idx[1] + 100 * idx[2] + 1000 * idx[3] + 1,
                    )
                });
                let b = make_tensor(&[3, 2], |idx| {
                    <$T as TestScalar>::from_usize(idx[0] * 2 + idx[1] + 1)
                });

                let c = einsum::<TS, CpuBackend>(&mut ctx, "iiij,jk->ik", &[&a, &b], None).unwrap();
                assert_eq!(c.dims(), &[2, 2]);

                for i in 0..2 {
                    for k in 0..2 {
                        let mut expected = <$T as TestScalar>::from_f64(0.0);
                        for j in 0..3 {
                            expected = expected + get_t(&a, &[i, i, i, j]) * get_t(&b, &[j, k]);
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
            fn einsum_multi_repeat_diagonal_extract() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 2, 2, 3], |idx| {
                    <$T as TestScalar>::from_usize(
                        idx[0] + 10 * idx[1] + 100 * idx[2] + 1000 * idx[3] + 1,
                    )
                });

                let y = einsum::<TS, CpuBackend>(&mut ctx, "iiij->ij", &[&a], None).unwrap();
                assert_eq!(y.dims(), &[2, 3]);

                for i in 0..2 {
                    for j in 0..3 {
                        let expected = get_t(&a, &[i, i, i, j]);
                        assert!(
                            <$T as TestScalar>::approx_eq(get_t(&y, &[i, j]), expected),
                            "Y[{i},{j}] = {:?}, expected {:?}, diff = {}",
                            get_t(&y, &[i, j]),
                            expected,
                            <$T as TestScalar>::diff_norm(get_t(&y, &[i, j]), expected)
                        );
                    }
                }
            }

            #[test]
            fn einsum_binary_four_way_repeat_outer_product() {
                let mut ctx = CpuContext::new(1);
                let a = make_tensor(&[2, 2, 2, 2], |idx| {
                    <$T as TestScalar>::from_usize(
                        idx[0] + 10 * idx[1] + 100 * idx[2] + 1000 * idx[3] + 1,
                    )
                });
                let v = make_tensor(&[3], |idx| <$T as TestScalar>::from_usize(idx[0] + 2));

                let y = einsum::<TS, CpuBackend>(&mut ctx, "iiii,j->ij", &[&a, &v], None).unwrap();
                assert_eq!(y.dims(), &[2, 3]);

                for i in 0..2 {
                    for j in 0..3 {
                        let expected = get_t(&a, &[i, i, i, i]) * get_t(&v, &[j]);
                        assert!(
                            <$T as TestScalar>::approx_eq(get_t(&y, &[i, j]), expected),
                            "Y[{i},{j}] = {:?}, expected {:?}, diff = {}",
                            get_t(&y, &[i, j]),
                            expected,
                            <$T as TestScalar>::diff_norm(get_t(&y, &[i, j]), expected)
                        );
                    }
                }
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
                let c = einsum::<TS, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
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
                let m = einsum::<TS, CpuBackend>(&mut ctx, "i,j->ij", &[&u, &v], None).unwrap();
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
                let d = einsum::<TS, CpuBackend>(&mut ctx, "i,i->", &[&u, &v], None).unwrap();
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
                let c = einsum::<TS, CpuBackend>(&mut ctx, "ij,ij->ij", &[&a, &b], None).unwrap();
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
                let d = einsum::<TS, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None)
                    .unwrap();
                assert_eq!(d.dims(), &[2, 2]);

                // Verify: D = A @ B @ C
                let ab = einsum::<TS, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
                let abc =
                    einsum::<TS, CpuBackend>(&mut ctx, "ij,jk->ik", &[&ab, &c], None).unwrap();

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

// ============================================================================
// Opteinsum parity tests (#139)
//
// Parity checklist against strided-opteinsum integration test categories:
//
// [x] Repeated-label edge cases
//     - Self-contraction (single tensor, sum over non-trace axes):
//       einsum_self_contraction_trace
//     - Partial trace with free index ("iij->j"):
//       einsum_partial_trace_with_free_index
//     - Trace of rank-3 tensor ("iji->j"):
//       einsum_trace_rank3_to_vector
//     - Three-way repeated label (iii->):
//       einsum_three_way_repeated_label_trace
//     - Batched dot product ("bi,bi->b"):
//       einsum_batched_dot_product
//     - Additional diagonal-extract variants ("ijj->i", "ijj->ij", "jii->j"):
//       einsum_diag_extract_reduce_ijj_to_i
//       einsum_diag_extract_no_reduce_ijj_to_ij
//       einsum_diag_extract_permuted_jii_to_j
//
// [x] Size-dict dependent cases
//     - Explicit size hints via Subscripts::new + ContractionTree::optimize:
//       einsum_size_dict_explicit_shapes
//     - Size-dict for output-only label (diagonal embedding with computed
//       shapes): einsum_size_dict_output_only_label
//
// [x] Additional binary/N-ary numeric patterns from strided-opteinsum
//     - Batched matmul ("bij,bjk->bik"): einsum_batched_matmul
//     - Transposed-RHS contraction ("ij,kj->ik"): einsum_transposed_rhs_contraction
//     - Reduction over first axis ("ij->j"): einsum_reduce_first_axis
//     - Scalar/vector product family (",k->k", "i,->i", ",->"):
//       einsum_scalar_vector_products
//
// [x] Unicode label parsing
//     - Parser accepts Unicode labels: parse_unicode_labels_accepted
//     - Parser rejects separators/symbols: parse_various_invalid_chars
//
// [x] Complex-diagonal cases
//     - Diagonal extraction with Complex64 data:
//       einsum_complex_diagonal_extraction
//     - Diagonal embedding with Complex64 data:
//       einsum_complex_diagonal_embedding
//     - Trace of Complex64 matrix: einsum_complex_trace
//
// [x] Error path tests
//     - Rank mismatch (subscript vs tensor ndim):
//       einsum_error_rank_mismatch
//     - Empty inputs array: einsum_error_empty_inputs
//     - Output label not in any input:
//       einsum_error_output_label_not_in_input
//     - Non-square trace (repeated label, different sizes):
//       einsum_error_non_square_trace
//
// [ ] Known limitations (not testable with current implementation)
//     - Implicit output: parser requires explicit "->" separator
//     - Scalar generative repeated-output embeddings from size_dict
//       ("->ii", "->iii"): tracked as ignored parity tests
//     - Multi-pair trace parity ("iijj->"): tracked as ignored parity test
//     - Multi-trace with independent pairs of different dimensions
//       ("ijij->" where dim(i) != dim(j)): trace backend uses a single
//       diagonal loop, so all paired axes must share the same dimension
//     - Trace + full reduction ("iij->"): the trace backend does not sum
//       over non-paired, non-output axes (j is silently ignored)
//     - Repeated labels in a single operand during pairwise contraction
//       ("ii,j->j"): RESOLVED — unified GEMM dispatcher uses Tensor::diagonal
//       for diagonal extraction before GEMM decomposition
// ============================================================================

// ============================================================================
// Repeated-label edge cases
// ============================================================================

#[test]
fn einsum_self_contraction_trace() {
    // Self-contraction: trace of a rank-3 tensor over two indices
    // T_{ijk} with trace over i,k -> v_j = sum_i T_{iji}
    let mut ctx = CpuContext::new(1);
    // 2x3x2 tensor
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let t = Tensor::<f64>::from_slice(&data, &[2, 3, 2], COL).unwrap();
    let v = einsum::<S, CpuBackend>(&mut ctx, "ijk->j", &[&t], None).unwrap();
    assert_eq!(v.dims(), &[3]);

    // Manual verification: sum over i and k
    for j in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            for k in 0..2 {
                expected += get(&t, &[i, j, k]);
            }
        }
        assert!(
            (get(&v, &[j]) - expected).abs() < 1e-10,
            "v[{j}] = {}, expected {expected}",
            get(&v, &[j])
        );
    }
}

#[test]
fn einsum_partial_trace_with_free_index() {
    // Partial trace: T_{iij} -> v_j = sum_i T[i,i,j]
    // This is a single-pair trace with a free output index.
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let t = Tensor::<f64>::from_slice(&data, &[2, 2, 3], COL).unwrap();
    let v = einsum::<S, CpuBackend>(&mut ctx, "iij->j", &[&t], None).unwrap();
    assert_eq!(v.dims(), &[3]);

    // Manual: v[j] = sum_i T[i,i,j]
    for j in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            expected += get(&t, &[i, i, j]);
        }
        assert!(
            (get(&v, &[j]) - expected).abs() < 1e-10,
            "v[{j}] = {}, expected {expected}",
            get(&v, &[j])
        );
    }
}

#[test]
fn einsum_trace_rank3_to_vector() {
    // Trace of rank-3 tensor: T_{iji} -> v_j = sum_i T[i,j,i]
    // The repeated label 'i' at positions 0 and 2 is traced, 'j' is free.
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (1..=18).map(|x| x as f64).collect();
    let t = Tensor::<f64>::from_slice(&data, &[3, 2, 3], COL).unwrap();
    let v = einsum::<S, CpuBackend>(&mut ctx, "iji->j", &[&t], None).unwrap();
    assert_eq!(v.dims(), &[2]);

    // Manual: v[j] = sum_i T[i,j,i]
    for j in 0..2 {
        let mut expected = 0.0;
        for i in 0..3 {
            expected += get(&t, &[i, j, i]);
        }
        assert!(
            (get(&v, &[j]) - expected).abs() < 1e-10,
            "v[{j}] = {}, expected {expected}",
            get(&v, &[j])
        );
    }
}

#[test]
fn einsum_three_way_repeated_label_trace() {
    // Three-way repeated label: T_{iii} -> scalar
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (1..=27).map(|x| x as f64).collect();
    let t = Tensor::<f64>::from_slice(&data, &[3, 3, 3], COL).unwrap();
    let s = einsum::<S, CpuBackend>(&mut ctx, "iii->", &[&t], None).unwrap();
    assert!(s.dims().is_empty());

    // Manual: sum_i T[i,i,i]
    let mut expected = 0.0;
    for i in 0..3 {
        expected += get(&t, &[i, i, i]);
    }
    assert!(
        (scalar_val(&s) - expected).abs() < 1e-10,
        "3-way trace = {}, expected {expected}",
        scalar_val(&s)
    );
}

#[test]
fn einsum_batched_dot_product() {
    // Batched dot product: "bi,bi->b"
    // Two 3x2 matrices, contract over index i for each batch element b
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], COL).unwrap();
    let c = einsum::<S, CpuBackend>(&mut ctx, "bi,bi->b", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[3]);

    // Manual: c[b] = sum_i A[b,i] * B[b,i]
    for batch in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            expected += get(&a, &[batch, i]) * get(&b, &[batch, i]);
        }
        assert!(
            (get(&c, &[batch]) - expected).abs() < 1e-10,
            "c[{batch}] = {}, expected {expected}",
            get(&c, &[batch])
        );
    }
}

#[test]
fn einsum_batched_matmul() {
    // Ported pattern from strided-opteinsum: "bij,bjk->bik"
    let mut ctx = CpuContext::new(1);

    // batch=2, each is 2x2
    let a = Tensor::<f64>::from_slice(
        &[
            1.0, 0.0, 0.0, 1.0, // batch 0: I
            2.0, 0.0, 0.0, 2.0, // batch 1: 2I
        ],
        &[2, 2, 2],
        COL,
    )
    .unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ],
        &[2, 2, 2],
        COL,
    )
    .unwrap();

    let c = einsum::<S, CpuBackend>(&mut ctx, "bij,bjk->bik", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2]);

    for batch in 0..2 {
        for i in 0..2 {
            for k in 0..2 {
                let mut expected = 0.0;
                for j in 0..2 {
                    expected += get(&a, &[batch, i, j]) * get(&b, &[batch, j, k]);
                }
                assert!(
                    (get(&c, &[batch, i, k]) - expected).abs() < 1e-10,
                    "c[{batch},{i},{k}] = {}, expected {expected}",
                    get(&c, &[batch, i, k])
                );
            }
        }
    }
}

#[test]
fn einsum_transposed_rhs_contraction() {
    // Ported pattern from strided-opteinsum differential set: "ij,kj->ik"
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[
            7.0, 8.0, 9.0, // row k=0 over j
            10.0, 11.0, 12.0, // row k=1 over j
        ],
        &[2, 3],
        COL,
    )
    .unwrap();

    let c = einsum::<S, CpuBackend>(&mut ctx, "ij,kj->ik", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2, 2]);

    for i in 0..2 {
        for k in 0..2 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += get(&a, &[i, j]) * get(&b, &[k, j]);
            }
            assert!(
                (get(&c, &[i, k]) - expected).abs() < 1e-10,
                "c[{i},{k}] = {}, expected {expected}",
                get(&c, &[i, k])
            );
        }
    }
}

#[test]
fn einsum_reduce_first_axis() {
    // Ported pattern from strided-opteinsum: "ij->j"
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(
        &[
            0.0, 1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, 7.0, // row 1
            8.0, 9.0, 10.0, 11.0, // row 2
        ],
        &[3, 4],
        COL,
    )
    .unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "ij->j", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[4]);

    for j in 0..4 {
        let expected = get(&a, &[0, j]) + get(&a, &[1, j]) + get(&a, &[2, j]);
        assert!(
            (get(&y, &[j]) - expected).abs() < 1e-10,
            "y[{j}] = {}, expected {expected}",
            get(&y, &[j])
        );
    }
}

#[test]
fn einsum_multi_pair_trace_iijj() {
    // Ported pattern from strided-opteinsum: "iijj->"
    let mut ctx = CpuContext::new(1);
    let n = 3;
    let total = n * n * n * n;
    let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[n, n, n, n], COL).unwrap();

    let s = einsum::<S, CpuBackend>(&mut ctx, "iijj->", &[&a], None).unwrap();
    assert!(s.dims().is_empty());

    let mut expected = 0.0;
    for i in 0..n {
        for j in 0..n {
            expected += get(&a, &[i, i, j, j]);
        }
    }
    assert!(
        (scalar_val(&s) - expected).abs() < 1e-10,
        "trace = {}, expected {expected}",
        scalar_val(&s)
    );
}

#[test]
fn einsum_diag_extract_reduce_ijj_to_i() {
    // Ported pattern from strided-opteinsum: "ijj->i"
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[2, 3, 3], COL).unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "ijj->i", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[2]);

    for i in 0..2 {
        let mut expected = 0.0;
        for j in 0..3 {
            expected += get(&a, &[i, j, j]);
        }
        assert!(
            (get(&y, &[i]) - expected).abs() < 1e-10,
            "y[{i}] = {}, expected {expected}",
            get(&y, &[i])
        );
    }
}

#[test]
fn einsum_diag_extract_no_reduce_ijj_to_ij() {
    // Ported pattern from strided-opteinsum: "ijj->ij"
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[2, 3, 3], COL).unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "ijj->ij", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[2, 3]);

    for i in 0..2 {
        for j in 0..3 {
            let expected = get(&a, &[i, j, j]);
            assert!(
                (get(&y, &[i, j]) - expected).abs() < 1e-10,
                "y[{i},{j}] = {}, expected {expected}",
                get(&y, &[i, j])
            );
        }
    }
}

#[test]
fn einsum_diag_extract_permuted_jii_to_j() {
    // Ported pattern from strided-opteinsum: "jii->j"
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[3, 2, 2], COL).unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "jii->j", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[3]);

    for j in 0..3 {
        let expected = get(&a, &[j, 0, 0]) + get(&a, &[j, 1, 1]);
        assert!(
            (get(&y, &[j]) - expected).abs() < 1e-10,
            "y[{j}] = {}, expected {expected}",
            get(&y, &[j])
        );
    }
}

#[test]
fn einsum_scalar_vector_products() {
    // Ported patterns from strided-opteinsum: ",k->k", "i,->i", ",->"
    let mut ctx = CpuContext::new(1);
    let three = Tensor::<f64>::from_slice(&[3.0], &[], COL).unwrap();
    let two = Tensor::<f64>::from_slice(&[2.0], &[], COL).unwrap();
    let v4 = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4], COL).unwrap();
    let v3 = Tensor::<f64>::from_slice(&[10.0, 20.0, 30.0], &[3], COL).unwrap();
    let seven = Tensor::<f64>::from_slice(&[7.0], &[], COL).unwrap();

    let a = einsum::<S, CpuBackend>(&mut ctx, ",k->k", &[&three, &v4], None).unwrap();
    assert_eq!(a.dims(), &[4]);
    assert!((get(&a, &[0]) - 3.0).abs() < 1e-10);
    assert!((get(&a, &[1]) - 6.0).abs() < 1e-10);
    assert!((get(&a, &[2]) - 9.0).abs() < 1e-10);
    assert!((get(&a, &[3]) - 12.0).abs() < 1e-10);

    let b = einsum::<S, CpuBackend>(&mut ctx, "i,->i", &[&v3, &two], None).unwrap();
    assert_eq!(b.dims(), &[3]);
    assert!((get(&b, &[0]) - 20.0).abs() < 1e-10);
    assert!((get(&b, &[1]) - 40.0).abs() < 1e-10);
    assert!((get(&b, &[2]) - 60.0).abs() < 1e-10);

    let c = einsum::<S, CpuBackend>(&mut ctx, ",->", &[&three, &seven], None).unwrap();
    assert!(c.dims().is_empty());
    assert!((scalar_val(&c) - 21.0).abs() < 1e-10);
}

#[test]
fn einsum_unit_extent_contraction_is_not_misrouted_to_elementwise() {
    // Even when fused sizes are m=n=k=1, this is a true contraction:
    // A[abcdef] * B[ace] -> C[bdf]
    // The summed labels (a,c,e) and free labels (b,d,f) are distinct.
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(&[2.0], &[1, 1, 1, 1, 1, 1], COL).unwrap();
    let b = Tensor::from_slice(&[3.0], &[1, 1, 1], COL).unwrap();

    let out = einsum::<S, CpuBackend>(&mut ctx, "abcdef,ace->bdf", &[&a, &b], None).unwrap();
    assert_eq!(out.dims(), &[1, 1, 1]);
    assert_eq!(to_col_major_vec(&out), vec![6.0]);
}

// ============================================================================
// Size-dict dependent cases
// ============================================================================

#[test]
fn einsum_size_dict_explicit_shapes() {
    // Use Subscripts::new + ContractionTree::optimize to verify size-dict
    // construction from explicit shapes (not string parsing).
    // This exercises the path where sizes come from shapes, not from
    // string labels.
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let tree = ContractionTree::optimize(&subs, &[&[2, 3], &[3, 5], &[5, 4]]).unwrap();

    let mut ctx = CpuContext::new(1);
    let a_data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
    let b_data: Vec<f64> = (1..=15).map(|x| x as f64).collect();
    let c_data: Vec<f64> = (1..=20).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&b_data, &[3, 5], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&c_data, &[5, 4], COL).unwrap();

    let d = einsum_with_plan::<S, CpuBackend>(&mut ctx, &tree, &[&a, &b, &c], None).unwrap();
    assert_eq!(d.dims(), &[2, 4]);

    // Verify via sequential pairwise
    let ab = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    let abc = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&ab, &c], None).unwrap();

    for i in 0..2 {
        for j in 0..4 {
            assert!(
                (get(&d, &[i, j]) - get(&abc, &[i, j])).abs() < 1e-8,
                "D[{i},{j}] = {}, expected {}",
                get(&d, &[i, j]),
                get(&abc, &[i, j])
            );
        }
    }
}

#[test]
fn einsum_size_dict_output_only_label() {
    // Diagonal embedding: "i->ii" -- the output label 'i' appears twice.
    // The size of 'i' is inferred from the input tensor.
    // This verifies the size-dict handles output-only repeated labels correctly.
    let mut ctx = CpuContext::new(1);
    let v = Tensor::<f64>::from_slice(&[7.0, 8.0, 9.0, 10.0], &[4], COL).unwrap();
    let d = einsum::<S, CpuBackend>(&mut ctx, "i->ii", &[&v], None).unwrap();
    assert_eq!(d.dims(), &[4, 4]);

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { get(&v, &[i]) } else { 0.0 };
            assert!(
                (get(&d, &[i, j]) - expected).abs() < 1e-10,
                "d[{i},{j}] = {}, expected {expected}",
                get(&d, &[i, j])
            );
        }
    }
}

#[test]
fn einsum_size_dict_scalar_to_diagonal_and_superdiagonal() {
    // Ported patterns from strided-opteinsum: "->ii", "->iii"
    let mut ctx = CpuContext::new(1);
    let scalar1 = Tensor::<f64>::from_slice(&[1.0], &[], COL).unwrap();
    let scalar2 = Tensor::<f64>::from_slice(&[2.0], &[], COL).unwrap();

    // "->ii" with i=4
    let subs_ii = Subscripts::new(&[&[]], &[0, 0]);
    let sd_ii = std::collections::HashMap::from([(0_u32, 4_usize)]);
    let d2 = einsum_with_subscripts::<S, CpuBackend>(&mut ctx, &subs_ii, &[&scalar1], Some(&sd_ii))
        .unwrap();
    assert_eq!(d2.dims(), &[4, 4]);
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (get(&d2, &[i, j]) - expected).abs() < 1e-10,
                "d2[{i},{j}] = {}, expected {expected}",
                get(&d2, &[i, j])
            );
        }
    }

    // "->iii" with i=3
    let subs_iii = Subscripts::new(&[&[]], &[0, 0, 0]);
    let sd_iii = std::collections::HashMap::from([(0_u32, 3_usize)]);
    let d3 =
        einsum_with_subscripts::<S, CpuBackend>(&mut ctx, &subs_iii, &[&scalar2], Some(&sd_iii))
            .unwrap();
    assert_eq!(d3.dims(), &[3, 3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let expected = if i == j && j == k { 2.0 } else { 0.0 };
                assert!(
                    (get(&d3, &[i, j, k]) - expected).abs() < 1e-10,
                    "d3[{i},{j},{k}] = {}, expected {expected}",
                    get(&d3, &[i, j, k])
                );
            }
        }
    }
}

// ============================================================================
// Unicode label parsing
// ============================================================================

#[test]
fn parse_unicode_labels_accepted() {
    // Greek labels
    let greek = Subscripts::parse("\u{03B1}\u{03B2},\u{03B2}\u{03B3}->\u{03B1}\u{03B3}");
    assert!(
        greek.is_ok(),
        "unicode Greek labels should parse: {greek:?}"
    );

    // Benchmark-like labels: digit + accented + Icelandic eth
    let benchmark_like = Subscripts::parse("0Á,Áð,ðÂ->0Â");
    assert!(
        benchmark_like.is_ok(),
        "benchmark-like Unicode labels should parse: {benchmark_like:?}"
    );
}

#[test]
fn parse_various_invalid_chars() {
    // Digits and Unicode letters are accepted.
    let accepted_cases = [
        ("0i,ij->0j", "digit"),
        ("i\u{00E9},\u{00E9}j->ij", "accented char"),
        ("i\u{4E2D},\u{4E2D}j->ij", "CJK char"),
    ];
    for (notation, desc) in &accepted_cases {
        let result = Subscripts::parse(notation);
        assert!(
            result.is_ok(),
            "should accept {desc} in notation '{notation}', got: {:?}",
            result
        );
    }

    // Control characters must still be rejected.
    let invalid_cases = [
        ("i\u{0},\u{0}j->ij", "null byte"),
        ("i\u{000A}j->ij", "newline"),
        ("i!,!j->ij", "punctuation"),
        ("i j,jk->ik", "space"),
    ];
    for (notation, desc) in &invalid_cases {
        let result = Subscripts::parse(notation);
        assert!(
            result.is_err(),
            "should reject {desc} in notation '{notation}', got: {:?}",
            result
        );
    }
}

#[test]
fn nested_einsum_parse_rejects_punctuation_and_whitespace_labels() {
    let invalid_cases = [
        ("(i!,jk),kl->il", "punctuation"),
        ("(ij,j k),kl->il", "space"),
        ("(ij,jk),kl->i l", "output space"),
    ];
    for (notation, desc) in &invalid_cases {
        let result = tenferro_einsum::NestedEinsum::parse(notation);
        assert!(
            result.is_err(),
            "should reject {desc} in nested notation '{notation}', got: {:?}",
            result
        );
    }
}

// ============================================================================
// Complex-diagonal cases
// ============================================================================

#[test]
fn einsum_complex_diagonal_extraction() {
    use num_complex::Complex64;
    type CS = Standard<Complex64>;

    let mut ctx = CpuContext::new(1);
    let data: Vec<Complex64> = (1..=9)
        .map(|x| Complex64::new(x as f64, -(x as f64)))
        .collect();
    let a = Tensor::<Complex64>::from_slice(&data, &[3, 3], COL).unwrap();
    let d = einsum::<CS, CpuBackend>(&mut ctx, "ii->i", &[&a], None).unwrap();
    assert_eq!(d.dims(), &[3]);

    // Column-major 3x3: a[i,j] at offset i + 3*j
    // diagonal: a[0,0]=1-i, a[1,1]=5-5i, a[2,2]=9-9i
    let diag_expected = [
        Complex64::new(1.0, -1.0),
        Complex64::new(5.0, -5.0),
        Complex64::new(9.0, -9.0),
    ];
    for i in 0..3 {
        let got = get_t(&d, &[i]);
        assert!(
            (got - diag_expected[i]).norm() < 1e-10,
            "diag[{i}] = {got:?}, expected {:?}",
            diag_expected[i]
        );
    }
}

#[test]
fn einsum_complex_diagonal_embedding() {
    use num_complex::Complex64;
    type CS = Standard<Complex64>;

    let mut ctx = CpuContext::new(1);
    let data = vec![
        Complex64::new(2.0, 1.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(5.0, 0.5),
    ];
    let v = Tensor::<Complex64>::from_slice(&data, &[3], COL).unwrap();
    let d = einsum::<CS, CpuBackend>(&mut ctx, "i->ii", &[&v], None).unwrap();
    assert_eq!(d.dims(), &[3, 3]);

    for i in 0..3 {
        for j in 0..3 {
            let got = get_t::<Complex64>(&d, &[i, j]);
            let expected = if i == j {
                data[i]
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!(
                (got - expected).norm() < 1e-10,
                "d[{i},{j}] = {got:?}, expected {expected:?}"
            );
        }
    }
}

#[test]
fn einsum_complex_trace() {
    use num_complex::Complex64;
    type CS = Standard<Complex64>;

    let mut ctx = CpuContext::new(1);
    let data: Vec<Complex64> = (1..=4)
        .map(|x| Complex64::new(x as f64, 0.5 * x as f64))
        .collect();
    let a = Tensor::<Complex64>::from_slice(&data, &[2, 2], COL).unwrap();
    let tr = einsum::<CS, CpuBackend>(&mut ctx, "ii->", &[&a], None).unwrap();
    assert!(tr.dims().is_empty());

    // Column-major 2x2: a[0,0] + a[1,1]
    // a[0,0] = 1+0.5i, a[1,0] = 2+i, a[0,1] = 3+1.5i, a[1,1] = 4+2i
    // trace = (1+0.5i) + (4+2i) = 5+2.5i
    let expected = Complex64::new(5.0, 2.5);
    let got = scalar_val_t(&tr);
    assert!(
        (got - expected).norm() < 1e-10,
        "trace = {got:?}, expected {expected:?}"
    );
}

// ============================================================================
// Error path tests
// ============================================================================

#[test]
fn einsum_error_rank_mismatch() {
    // Subscripts say "ij" (rank 2) but tensor is rank 1
    let mut ctx = CpuContext::new(1);
    let v = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], COL).unwrap();
    let result = einsum::<S, CpuBackend>(&mut ctx, "ij->i", &[&v], None);
    assert!(result.is_err(), "should fail for rank mismatch");
}

#[test]
fn einsum_error_empty_inputs() {
    // No input tensors
    let mut ctx = CpuContext::new(1);
    let empty: &[&Tensor<f64>] = &[];
    let result = einsum::<S, CpuBackend>(&mut ctx, "->", empty, None);
    assert!(result.is_err(), "should fail for empty inputs");
}

#[test]
fn einsum_error_output_label_not_in_input() {
    // Output references label 'k' which is not in any input
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let result = einsum::<S, CpuBackend>(&mut ctx, "ij->ik", &[&a], None);
    assert!(
        result.is_err(),
        "should fail when output label not in input"
    );
}

#[test]
fn einsum_error_non_square_trace() {
    // Trace "ii->" but matrix is 2x3 (non-square), so label 'i' has conflicting sizes
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let result = einsum::<S, CpuBackend>(&mut ctx, "ii->", &[&a], None);
    assert!(
        matches!(result, Err(tenferro_device::Error::ShapeMismatch { .. })),
        "expected ShapeMismatch for non-square trace, got: {:?}",
        result.as_ref().err()
    );
}

// ============================================================================
// Forward-mode tangent auto-propagation and HVP
// ============================================================================

#[test]
fn einsum_auto_propagates_fw_grad() {
    let mut ctx = CpuContext::new(1);

    // A = [[1, 3], [2, 4]], B = [[5, 7], [6, 8]] (col-major)
    let mut a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    // Set tangent on a only: dA = ones
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    a.set_fw_grad(da.clone());

    // C = einsum("ij,jk->ik", A, B)
    let c = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();

    // C should have fw_grad = einsum_frule result
    assert!(c.has_fw_grad(), "output should carry fw_grad");

    // Compare with explicit frule
    let expected =
        einsum_frule::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], &[Some(&da), None])
            .unwrap();

    let cg = c.fw_grad().unwrap();
    let cg_data = cg.buffer().as_slice().unwrap();
    let exp_data = expected.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert!(
            (cg_data[i] - exp_data[i]).abs() < 1e-10,
            "fw_grad[{i}] = {}, expected {}",
            cg_data[i],
            exp_data[i]
        );
    }
}

#[test]
fn einsum_no_fw_grad_unchanged() {
    let mut ctx = CpuContext::new(1);

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    let c = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();
    assert!(
        !c.has_fw_grad(),
        "output should NOT carry fw_grad when inputs have none"
    );
}

/// HVP via jvp(grad(f)) composition.
///
/// f(A) = sum(A @ B) = einsum("ij,jk->ik", A, B) then einsum("ij->", C)
/// grad_A = ones_{ik} @ B^T_{kj} = einsum("ik,jk->ij", ones, B)
/// HVP in direction dA: d(grad_A)/dt = einsum("ik,jk->ij", zeros, B) = 0
///   (loss cotangent ones has no tangent, B has no tangent)
///
/// But the forward pass C = A@B, dC = dA@B, and loss = sum(C), dloss = sum(dC).
/// The pullback sees cotangent = ones (no fw_grad) and primals [A, B] where A has fw_grad.
/// Reverse einsum for grad_A: einsum("ik,jk->ij", cot, B). Neither cot nor B has fw_grad.
/// So grad_A should NOT have fw_grad. This is correct: the HVP of a linear function is 0.
///
/// For a more interesting case, use loss = sum(C^2) where C = A @ B:
/// grad_A = 2*C @ B^T, and d(grad_A)/dt = 2*(dA@B)@B^T
#[test]
fn hvp_via_fw_grad_composition() {
    use chainrules::Tape;
    use std::sync::{Arc, Mutex};

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));

    // A = [[1, 3], [2, 4]] (col-major), B = [[5, 7], [6, 8]] (col-major)
    let mut a_data = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b_data = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    // Tangent direction: dA = ones
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    a_data.set_fw_grad(da);

    let tape = Tape::<Tensor<f64>>::new();
    let a = tape.leaf(a_data.clone());
    let b = tape.leaf(b_data.clone());
    let a_id = a.node_id().unwrap();

    // loss = sum_{ij} C_{ij}^2 where C = A @ B
    let c = tracked_einsum::<S, CpuBackend>(ctx.clone(), "ij,jk->ik", &[&a, &b]).unwrap();
    let loss = tracked_einsum::<S, CpuBackend>(ctx.clone(), "ij,ij->", &[&c, &c]).unwrap();

    let grads = tape.pullback(&loss).unwrap();
    let ga = grads.get(a_id).unwrap();

    // grad_A should match 2*C @ B^T
    let c_val = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,jk->ik",
        &[&a_data, &b_data],
        None,
    )
    .unwrap();
    let two = Tensor::<f64>::from_slice(&[2.0], &[], COL).unwrap();
    let two_c = einsum::<S, CpuBackend>(&mut ctx.lock().unwrap(), "ij,->ij", &[&c_val, &two], None)
        .unwrap();
    let expected_ga = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ik,jk->ij",
        &[&two_c, &b_data],
        None,
    )
    .unwrap();

    // Verify grad_A primal
    let ga_data = ga.buffer().as_slice().unwrap();
    let exp_data = expected_ga.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert!(
            (ga_data[i] - exp_data[i]).abs() < 1e-10,
            "grad_A[{i}] = {}, expected {}",
            ga_data[i],
            exp_data[i]
        );
    }

    // HVP: d(grad_A)/dt = 2*(dA@B)@B^T where dA = ones
    // dA@B = ones @ B = einsum("ij,jk->ik", ones, B)
    let ones = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    let da_b = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ij,jk->ik",
        &[&ones, &b_data],
        None,
    )
    .unwrap();
    // 2*(dA@B)@B^T = einsum("ik,jk->ij", 2*dA_B, B)
    let two_da_b =
        einsum::<S, CpuBackend>(&mut ctx.lock().unwrap(), "ij,->ij", &[&da_b, &two], None).unwrap();
    let expected_hvp = einsum::<S, CpuBackend>(
        &mut ctx.lock().unwrap(),
        "ik,jk->ij",
        &[&two_da_b, &b_data],
        None,
    )
    .unwrap();

    // grad_A should carry fw_grad = HVP
    assert!(ga.has_fw_grad(), "gradient should carry fw_grad for HVP");
    let hvp = ga.fw_grad().unwrap();
    let hvp_data = hvp.buffer().as_slice().unwrap();
    let exp_hvp_data = expected_hvp.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert!(
            (hvp_data[i] - exp_hvp_data[i]).abs() < 1e-10,
            "hvp[{i}] = {}, expected {}",
            hvp_data[i],
            exp_hvp_data[i]
        );
    }
}

#[test]
fn hvp_via_leaf_with_tangent_tracks_einsum_direction() {
    use chainrules::Tape;
    use std::sync::{Arc, Mutex};

    let ctx = Arc::new(Mutex::new(CpuContext::new(1)));
    let tape = Tape::<Tensor<f64>>::new();
    let x = tape
        .leaf_with_tangent(
            Tensor::<f64>::ones(&[3], MEM, COL),
            Tensor::<f64>::ones(&[3], MEM, COL),
        )
        .unwrap();
    let x_id = x.node_id().unwrap();

    let loss = tracked_einsum::<S, CpuBackend>(ctx, "i,i->", &[&x, &x]).unwrap();
    let hvp = tape.hvp(&loss).unwrap();

    assert_tensors_close(
        hvp.gradients.get(x_id).unwrap(),
        &Tensor::<f64>::from_slice(&[2.0, 2.0, 2.0], &[3], COL).unwrap(),
        "grad",
    );
    assert_tensors_close(
        hvp.hvp.get(x_id).unwrap(),
        &Tensor::<f64>::from_slice(&[2.0, 2.0, 2.0], &[3], COL).unwrap(),
        "hvp",
    );
}

#[test]
fn einsum_hvp_matches_manual_matmul_rule() {
    use chainrules::Differentiable;
    use tenferro_einsum::einsum_hvp;

    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[2.0, 1.0, 0.0, 3.0], &[2, 2], COL).unwrap();
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    let grad_c = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    let dgrad_c = Tensor::<f64>::from_slice(&[0.5, 1.0, 1.5, 2.0], &[2, 2], COL).unwrap();

    let hvps = einsum_hvp::<S, CpuBackend>(
        &mut ctx,
        "ij,jk->ik",
        &[&a, &b],
        &[Some(&da), None],
        &grad_c,
        &dgrad_c,
    )
    .unwrap();

    assert_eq!(hvps.len(), 2);

    let expected_grad_a =
        einsum::<S, CpuBackend>(&mut ctx, "ik,jk->ij", &[&grad_c, &b], None).unwrap();
    let expected_hvp_a = {
        let term_from_cot =
            einsum::<S, CpuBackend>(&mut ctx, "ik,jk->ij", &[&dgrad_c, &b], None).unwrap();
        let db_term = Tensor::<f64>::zeros(&[2, 2], MEM, COL);
        let _ = db_term;
        term_from_cot
    };
    let expected_grad_b =
        einsum::<S, CpuBackend>(&mut ctx, "ij,ik->jk", &[&a, &grad_c], None).unwrap();
    let expected_hvp_b = {
        let term_from_cot =
            einsum::<S, CpuBackend>(&mut ctx, "ij,ik->jk", &[&a, &dgrad_c], None).unwrap();
        let term_from_a =
            einsum::<S, CpuBackend>(&mut ctx, "ij,ik->jk", &[&da, &grad_c], None).unwrap();
        Tensor::<f64>::accumulate_tangent(term_from_cot, &term_from_a)
    };

    assert_tensors_close(&hvps[0].0, &expected_grad_a, "hvp grad_a");
    assert_tensors_close(&hvps[0].1, &expected_hvp_a, "hvp hvp_a");
    assert_tensors_close(&hvps[1].0, &expected_grad_b, "hvp grad_b");
    assert_tensors_close(&hvps[1].1, &expected_hvp_b, "hvp hvp_b");
}

// ============================================================================
// einsum: input+output repeated labels (pipeline decomposition)
// ============================================================================

#[test]
fn einsum_input_output_repeated_iij_to_jj() {
    // iij->jj : trace over i, then embed j diagonally
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[3, 3, 2], COL).unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "iij->jj", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[2, 2]);

    for j1 in 0..2 {
        for j2 in 0..2 {
            let expected = if j1 == j2 {
                let mut s = 0.0;
                for i in 0..3 {
                    s += get(&a, &[i, i, j1]);
                }
                s
            } else {
                0.0
            };
            assert!(
                (get(&y, &[j1, j2]) - expected).abs() < 1e-10,
                "y[{j1},{j2}] = {}, expected {expected}",
                get(&y, &[j1, j2])
            );
        }
    }
}

#[test]
fn einsum_input_output_repeated_ii_to_ii() {
    // ii->ii : extract diagonal, then embed back
    let mut ctx = CpuContext::new(1);
    let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&data, &[3, 3], COL).unwrap();

    let y = einsum::<S, CpuBackend>(&mut ctx, "ii->ii", &[&a], None).unwrap();
    assert_eq!(y.dims(), &[3, 3]);

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { get(&a, &[i, i]) } else { 0.0 };
            assert!(
                (get(&y, &[i, j]) - expected).abs() < 1e-10,
                "y[{i},{j}] = {}, expected {expected}",
                get(&y, &[i, j])
            );
        }
    }
}

// ============================================================================
// NestedEinsum parsing
// ============================================================================

#[test]
fn nested_parse_flat_no_parens() {
    // Without parentheses, produces a single root node with all leaves
    let nested = tenferro_einsum::NestedEinsum::parse("ij,jk->ik").unwrap();
    match &nested {
        tenferro_einsum::NestedEinsum::Node {
            subscripts,
            children,
        } => {
            assert_eq!(children.len(), 2);
            assert_eq!(
                subscripts.output,
                tenferro_einsum::Subscripts::parse("ij,jk->ik")
                    .unwrap()
                    .output
            );
            // Children are leaves
            assert!(matches!(
                children[0],
                tenferro_einsum::NestedEinsum::Leaf(0)
            ));
            assert!(matches!(
                children[1],
                tenferro_einsum::NestedEinsum::Leaf(1)
            ));
        }
        _ => panic!("expected Node"),
    }
}

#[test]
fn nested_parse_simple_group() {
    // (ij,jk),kl->il
    // Root: two children, first is a Node (group), second is Leaf(2)
    let nested = tenferro_einsum::NestedEinsum::parse("(ij,jk),kl->il").unwrap();
    match &nested {
        tenferro_einsum::NestedEinsum::Node {
            subscripts,
            children,
        } => {
            assert_eq!(children.len(), 2);
            // Root output is "il"
            let i = 'i' as u32;
            let l = 'l' as u32;
            assert_eq!(subscripts.output, vec![i, l]);
            // First child is a Node (the group)
            match &children[0] {
                tenferro_einsum::NestedEinsum::Node {
                    subscripts: inner_subs,
                    children: inner_children,
                } => {
                    assert_eq!(inner_children.len(), 2);
                    assert!(matches!(
                        inner_children[0],
                        tenferro_einsum::NestedEinsum::Leaf(0)
                    ));
                    assert!(matches!(
                        inner_children[1],
                        tenferro_einsum::NestedEinsum::Leaf(1)
                    ));
                    // Inner output should contain labels needed outside: i and k
                    // i appears in final output, k appears in sibling kl
                    let k = 'k' as u32;
                    assert!(inner_subs.output.contains(&i));
                    assert!(inner_subs.output.contains(&k));
                }
                _ => panic!("expected inner Node"),
            }
            // Second child is Leaf(2)
            assert!(matches!(
                children[1],
                tenferro_einsum::NestedEinsum::Leaf(2)
            ));
        }
        _ => panic!("expected Node"),
    }
}

#[test]
fn nested_parse_deeply_nested() {
    // ((ij,jk),kl),lm->im
    let nested = tenferro_einsum::NestedEinsum::parse("((ij,jk),kl),lm->im").unwrap();
    // Should have depth 3: root -> group -> group -> leaves
    match &nested {
        tenferro_einsum::NestedEinsum::Node { children, .. } => {
            assert_eq!(children.len(), 2); // outer group + lm
            match &children[0] {
                tenferro_einsum::NestedEinsum::Node { children: mid, .. } => {
                    assert_eq!(mid.len(), 2); // inner group + kl
                    match &mid[0] {
                        tenferro_einsum::NestedEinsum::Node {
                            children: inner, ..
                        } => {
                            assert_eq!(inner.len(), 2); // ij + jk
                            assert!(matches!(inner[0], tenferro_einsum::NestedEinsum::Leaf(0)));
                            assert!(matches!(inner[1], tenferro_einsum::NestedEinsum::Leaf(1)));
                        }
                        _ => panic!("expected inner Node"),
                    }
                    assert!(matches!(mid[1], tenferro_einsum::NestedEinsum::Leaf(2)));
                }
                _ => panic!("expected mid Node"),
            }
            assert!(matches!(
                children[1],
                tenferro_einsum::NestedEinsum::Leaf(3)
            ));
        }
        _ => panic!("expected Node"),
    }
}

#[test]
fn nested_parse_error_mismatched_parens() {
    assert!(tenferro_einsum::NestedEinsum::parse("(ij,jk->ik").is_err());
    assert!(tenferro_einsum::NestedEinsum::parse("ij),jk->ik").is_err());
}

// ============================================================================
// NestedEinsum execution
// ============================================================================

#[test]
fn nested_einsum_simple_group() {
    // (ij,jk),kl->il should produce same result as ij,jk,kl->il
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], COL).unwrap();
    let b = Tensor::<f64>::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        &[3, 4],
        COL,
    )
    .unwrap();
    let c =
        Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], &[4, 2], COL).unwrap();

    let flat = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    let nested = einsum::<S, CpuBackend>(&mut ctx, "(ij,jk),kl->il", &[&a, &b, &c], None).unwrap();

    assert_eq!(flat.dims(), nested.dims());
    let flat_data = flat.buffer().as_slice().unwrap();
    let nested_data = nested.buffer().as_slice().unwrap();
    for (f, n) in flat_data.iter().zip(nested_data.iter()) {
        assert!((f - n).abs() < 1e-10, "flat={f}, nested={n}");
    }
}

#[test]
fn nested_einsum_deeply_nested() {
    // ((ij,jk),kl),lm->im
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[2.0, 0.0, 0.0, 2.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[3.0, 0.0, 0.0, 3.0], &[2, 2], COL).unwrap();
    let d = Tensor::<f64>::from_slice(&[4.0, 0.0, 0.0, 4.0], &[2, 2], COL).unwrap();

    let flat =
        einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl,lm->im", &[&a, &b, &c, &d], None).unwrap();
    let nested =
        einsum::<S, CpuBackend>(&mut ctx, "((ij,jk),kl),lm->im", &[&a, &b, &c, &d], None).unwrap();

    assert_eq!(flat.dims(), nested.dims());
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (get(&flat, &[i, j]) - get(&nested, &[i, j])).abs() < 1e-10,
                "mismatch at [{i},{j}]"
            );
        }
    }
}

#[test]
fn nested_einsum_nary_group() {
    // (ij,jk,kl)->il — three operands in one group
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();

    let flat = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    let nested = einsum::<S, CpuBackend>(&mut ctx, "(ij,jk,kl)->il", &[&a, &b, &c], None).unwrap();

    assert_eq!(flat.dims(), nested.dims());
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (get(&flat, &[i, j]) - get(&nested, &[i, j])).abs() < 1e-10,
                "mismatch at [{i},{j}]"
            );
        }
    }
}

#[test]
fn nested_einsum_single_operand_group() {
    // (ij)->ij — trivial single-operand group is identity
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();

    let result = einsum::<S, CpuBackend>(&mut ctx, "(ij)->ij", &[&a], None).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            assert!((get(&result, &[i, j]) - get(&a, &[i, j])).abs() < 1e-10);
        }
    }
}

// ============================================================================
// Bug-fix regression tests
// ============================================================================

#[test]
fn nested_einsum_extra_operands_error() {
    // Fix: "(ij)->ij" with 2 operands must error, not silently ignore the second
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    let result = einsum::<S, CpuBackend>(&mut ctx, "(ij)->ij", &[&a, &b], None);
    assert!(result.is_err(), "should error on extra operands");
}

#[test]
fn nested_einsum_fewer_operands_error() {
    // "(ij,jk),kl->il" with only 2 operands must error
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();

    let result = einsum::<S, CpuBackend>(&mut ctx, "(ij,jk),kl->il", &[&a, &b], None);
    assert!(
        result.is_err(),
        "should error on fewer operands than leaves"
    );
}

#[test]
fn subscripts_parse_unmatched_close_paren() {
    // "ij),jk->ik" must error in Subscripts::parse
    let result = Subscripts::parse("ij),jk->ik");
    assert!(result.is_err(), "unmatched ')' should be rejected");
}

#[test]
fn subscripts_parse_unmatched_open_paren() {
    let result = Subscripts::parse("(ij,jk->ik");
    assert!(result.is_err(), "unmatched '(' should be rejected");
}

#[test]
fn nested_einsum_propagates_fw_grad() {
    // Parenthesized path must propagate fw_grad just like the flat path
    let mut ctx = CpuContext::new(1);

    let mut a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();

    // Set tangent on a only
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    a.set_fw_grad(da.clone());

    // Parenthesized: (ij,jk),kl->il
    let result = einsum::<S, CpuBackend>(&mut ctx, "(ij,jk),kl->il", &[&a, &b, &c], None).unwrap();
    assert!(
        result.has_fw_grad(),
        "parenthesized einsum should propagate fw_grad"
    );

    // Compare with flat path
    let flat = einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a, &b, &c], None).unwrap();
    assert!(flat.has_fw_grad());

    let nested_grad = result.fw_grad().unwrap();
    let flat_grad = flat.fw_grad().unwrap();
    assert_tensors_close(&nested_grad, &flat_grad, "fw_grad");
}

#[test]
fn einsum_frule_parenthesized_operand_count_mismatch() {
    // Must return Err, not panic, when operand count doesn't match leaf count
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);

    // "(ij,jk),kl->il" expects 3 operands, but only 2 given
    let result =
        einsum_frule::<S, CpuBackend>(&mut ctx, "(ij,jk),kl->il", &[&a, &b], &[Some(&da), None]);
    assert!(result.is_err(), "should error on operand count mismatch");
}

#[test]
fn einsum_frule_parenthesized() {
    // einsum_frule with parenthesized subscripts must produce same result as flat
    let mut ctx = CpuContext::new(1);

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();

    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);

    let nested_tangent = einsum_frule::<S, CpuBackend>(
        &mut ctx,
        "(ij,jk),kl->il",
        &[&a, &b, &c],
        &[Some(&da), None, None],
    )
    .unwrap();

    let flat_tangent = einsum_frule::<S, CpuBackend>(
        &mut ctx,
        "ij,jk,kl->il",
        &[&a, &b, &c],
        &[Some(&da), None, None],
    )
    .unwrap();

    assert_tensors_close(&nested_tangent, &flat_tangent, "frule tangent");
}

#[test]
fn einsum_into_parenthesized() {
    // einsum_into with parenthesized subscripts must produce same result as flat
    let mut ctx = CpuContext::new(1);

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();

    // Test overwrite: output = 1.0 * einsum + 0.0 * output
    let mut nested_out = Tensor::<f64>::zeros(&[2, 2], MEM, COL);
    einsum_into::<S, CpuBackend>(
        &mut ctx,
        "(ij,jk),kl->il",
        &[&a, &b, &c],
        1.0,
        0.0,
        &mut nested_out,
        None,
    )
    .unwrap();

    let mut flat_out = Tensor::<f64>::zeros(&[2, 2], MEM, COL);
    einsum_into::<S, CpuBackend>(
        &mut ctx,
        "ij,jk,kl->il",
        &[&a, &b, &c],
        1.0,
        0.0,
        &mut flat_out,
        None,
    )
    .unwrap();

    let ns = nested_out.buffer().as_slice().unwrap();
    let fs = flat_out.buffer().as_slice().unwrap();
    for i in 0..ns.len() {
        assert!(
            (ns[i] - fs[i]).abs() < 1e-10,
            "einsum_into[{i}]: nested={}, flat={}",
            ns[i],
            fs[i]
        );
    }

    // Test accumulate: output = 2.0 * einsum + 1.0 * output
    let mut accum_out = Tensor::<f64>::ones(&[2, 2], MEM, COL);
    einsum_into::<S, CpuBackend>(
        &mut ctx,
        "(ij,jk),kl->il",
        &[&a, &b, &c],
        2.0,
        1.0,
        &mut accum_out,
        None,
    )
    .unwrap();

    let as_ = accum_out.buffer().as_slice().unwrap();
    // Expected: 2 * nested_result + 1 (all ones)
    for i in 0..as_.len() {
        let expected = 2.0 * ns[i] + 1.0;
        assert!(
            (as_[i] - expected).abs() < 1e-10,
            "einsum_into accumulate[{i}]: got={}, expected={}",
            as_[i],
            expected
        );
    }
}

#[test]
fn dual_einsum_parenthesized() {
    use chainrules::DualValue;
    // dual_einsum with parenthesized subscripts must produce same tangent as flat
    let mut ctx = CpuContext::new(1);

    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let c = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], COL).unwrap();
    let da = Tensor::<f64>::ones(&[2, 2], MEM, COL);

    let a_dual = DualValue::with_tangent(a.clone(), da.clone()).unwrap();
    let b_dual = DualValue::new(b.clone());
    let c_dual = DualValue::new(c.clone());

    let nested_result =
        dual_einsum::<S, CpuBackend>(&mut ctx, "(ij,jk),kl->il", &[&a_dual, &b_dual, &c_dual])
            .unwrap();

    let a_dual2 = DualValue::with_tangent(a, da).unwrap();
    let b_dual2 = DualValue::new(b);
    let c_dual2 = DualValue::new(c);

    let flat_result =
        dual_einsum::<S, CpuBackend>(&mut ctx, "ij,jk,kl->il", &[&a_dual2, &b_dual2, &c_dual2])
            .unwrap();

    // Primals must match
    assert_tensors_close(nested_result.primal(), flat_result.primal(), "dual primal");

    // Tangents must match
    assert_tensors_close(
        nested_result.tangent().unwrap(),
        flat_result.tangent().unwrap(),
        "dual tangent",
    );
}

#[test]
fn dual_einsum_without_tangents_returns_primal_only() {
    use chainrules::DualValue;

    let mut ctx = CpuContext::new(1);
    let a = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let b = Tensor::<f64>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], COL).unwrap();
    let a_dual = DualValue::new(a.clone());
    let b_dual = DualValue::new(b.clone());

    let result = dual_einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a_dual, &b_dual]).unwrap();
    let expected = einsum::<S, CpuBackend>(&mut ctx, "ij,jk->ik", &[&a, &b], None).unwrap();

    assert_tensors_close(result.primal(), &expected, "dual no tangent primal");
    assert!(result.tangent().is_none());
}

// ============================================================================
// Binary trace-like patterns (repeated labels in a single operand)
// ============================================================================

#[test]
fn einsum_binary_diag_ii_jk_to_ijk() {
    // "ii,jk->ijk": diagonal extraction on A, outer product with B
    let mut ctx = CpuContext::new(1);
    // A is 3x3
    let a = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0], &[3, 3], COL)
        .unwrap();
    // B is 2x2
    let b = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let c = einsum::<S, CpuBackend>(&mut ctx, "ii,jk->ijk", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[3, 2, 2]);

    // diag(A) = [1, 2, 3], result[i,j,k] = diag(A)[i] * B[j,k]
    let data = c.buffer().as_slice().unwrap();
    // Column-major: fastest index is i
    // (i=0,j=0,k=0) = 1*1=1, (i=1,j=0,k=0) = 2*1=2, (i=2,j=0,k=0) = 3*1=3
    assert_eq!(data[0], 1.0); // [0,0,0]
    assert_eq!(data[1], 2.0); // [1,0,0]
    assert_eq!(data[2], 3.0); // [2,0,0]
                              // (i=0,j=1,k=0) = 1*2=2, (i=1,j=1,k=0) = 2*2=4, (i=2,j=1,k=0) = 3*2=6
    assert_eq!(data[3], 2.0); // [0,1,0]
    assert_eq!(data[4], 4.0); // [1,1,0]
    assert_eq!(data[5], 6.0); // [2,1,0]
}

#[test]
fn einsum_binary_diag_iij_jk_to_ik() {
    // "iij,jk->ik": diagonal extraction on A (over i), then contract j
    let mut ctx = CpuContext::new(1);
    // A is 2x2x3 with subs [i,i,j], column-major strides [1,2,4]
    // Diagonal: A[0,0,j] = {1,2,3}, A[1,1,j] = {4,5,6}, off-diagonal = 0
    #[rustfmt::skip]
    let a_data: Vec<f64> = vec![
        // j=0: A[0,0,0]=1, A[1,0,0]=0, A[0,1,0]=0, A[1,1,0]=4
        1.0, 0.0, 0.0, 4.0,
        // j=1: A[0,0,1]=2, A[1,0,1]=0, A[0,1,1]=0, A[1,1,1]=5
        2.0, 0.0, 0.0, 5.0,
        // j=2: A[0,0,2]=3, A[1,0,2]=0, A[0,1,2]=0, A[1,1,2]=6
        3.0, 0.0, 0.0, 6.0,
    ];
    let a = Tensor::<f64>::from_slice(&a_data, &[2, 2, 3], COL).unwrap();
    // B is 3x2 with subs [j,k], column-major: B = [[1,0],[0,1],[0,0]]
    let b = Tensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[3, 2], COL).unwrap();
    let c = einsum::<S, CpuBackend>(&mut ctx, "iij,jk->ik", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2, 2]);

    // diag(A) over i gives shape [3,2] with subs [j,i]
    // A_diag[j=0,i=0]=1, A_diag[j=1,i=0]=2, A_diag[j=2,i=0]=3
    // A_diag[j=0,i=1]=4, A_diag[j=1,i=1]=5, A_diag[j=2,i=1]=6
    // C[i,k] = sum_j A_diag[j,i] * B[j,k]
    // C[0,0] = 1*1 + 2*0 + 3*0 = 1
    // C[1,0] = 4*1 + 5*0 + 6*0 = 4
    // C[0,1] = 1*0 + 2*1 + 3*0 = 2
    // C[1,1] = 4*0 + 5*1 + 6*0 = 5
    let data = c.buffer().as_slice().unwrap();
    assert!((data[0] - 1.0).abs() < 1e-10); // [0,0]
    assert!((data[1] - 4.0).abs() < 1e-10); // [1,0]
    assert!((data[2] - 2.0).abs() < 1e-10); // [0,1]
    assert!((data[3] - 5.0).abs() < 1e-10); // [1,1]
}

#[test]
fn einsum_binary_diag_ii_jj_to_ij() {
    // "ii,jj->ij": diagonal extraction on both operands
    let mut ctx = CpuContext::new(1);
    // A is 3x3
    let mut a_data = vec![0.0; 9];
    a_data[0] = 1.0; // [0,0]
    a_data[4] = 2.0; // [1,1]
    a_data[8] = 3.0; // [2,2]
    let a = Tensor::<f64>::from_slice(&a_data, &[3, 3], COL).unwrap();
    // B is 2x2
    let mut b_data = vec![0.0; 4];
    b_data[0] = 10.0; // [0,0]
    b_data[3] = 20.0; // [1,1]
    let b = Tensor::<f64>::from_slice(&b_data, &[2, 2], COL).unwrap();
    let c = einsum::<S, CpuBackend>(&mut ctx, "ii,jj->ij", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[3, 2]);

    // diag(A) = [1,2,3], diag(B) = [10,20]
    // C[i,j] = diag(A)[i] * diag(B)[j]
    let data = c.buffer().as_slice().unwrap();
    assert_eq!(data[0], 10.0); // [0,0] = 1*10
    assert_eq!(data[1], 20.0); // [1,0] = 2*10
    assert_eq!(data[2], 30.0); // [2,0] = 3*10
    assert_eq!(data[3], 20.0); // [0,1] = 1*20
    assert_eq!(data[4], 40.0); // [1,1] = 2*20
    assert_eq!(data[5], 60.0); // [2,1] = 3*20
}

#[test]
fn einsum_binary_diag_ii_j_to_j() {
    // "ii,j->j": trace of A (scalar), elementwise mul with B
    let mut ctx = CpuContext::new(1);
    // A is 3x3 with trace = 1+5+9 = 15
    let a_data: Vec<f64> = (1..=9).map(|x| x as f64).collect();
    let a = Tensor::<f64>::from_slice(&a_data, &[3, 3], COL).unwrap();
    // B is [2, 3]
    let b = Tensor::<f64>::from_slice(&[2.0, 3.0], &[2], COL).unwrap();
    let c = einsum::<S, CpuBackend>(&mut ctx, "ii,j->j", &[&a, &b], None).unwrap();
    assert_eq!(c.dims(), &[2]);

    // trace(A) = 1 + 5 + 9 = 15
    // C[j] = trace(A) * B[j] = 15 * [2, 3] = [30, 45]
    let data = c.buffer().as_slice().unwrap();
    assert_eq!(data[0], 30.0);
    assert_eq!(data[1], 45.0);
}
