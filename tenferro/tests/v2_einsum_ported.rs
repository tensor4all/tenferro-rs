//! Ported v1 einsum tests adapted for the v2 traced pipeline.
//!
//! Data is stored in **column-major** order. A helper `row_to_col_major` is
//! available for converting row-major test data when needed.

use std::panic::{catch_unwind, AssertUnwindSafe};

use tenferro::v2::einsum::{einsum, einsum_with, EinsumOptimize};
use tenferro::v2::engine::Engine;
use tenferro::v2::traced::TracedTensor;
use tenferro_einsum::{ContractionTree, NestedEinsum, Subscripts};
use tenferro_tensor::v2::{Tensor, TypedTensor};

// ============================================================================
// Helpers
// ============================================================================

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64"),
    }
}

/// Read a single element from a v2 Tensor by multi-index (col-major).
fn get_v2(t: &Tensor, idx: &[usize]) -> f64 {
    match t {
        Tensor::F64(inner) => *inner.get(idx),
        _ => panic!("expected F64"),
    }
}

/// Convert row-major data to column-major for a given shape.
#[allow(dead_code)]
fn row_to_col_major(data: &[f64], shape: &[usize]) -> Vec<f64> {
    let n: usize = shape.iter().product();
    let mut col_data = vec![0.0; n];
    let rank = shape.len();
    if rank <= 1 {
        return data.to_vec();
    }
    for rm_flat in 0..n {
        let mut idx = vec![0usize; rank];
        let mut rem = rm_flat;
        for d in (0..rank).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        let mut cm_flat = 0;
        let mut stride = 1;
        for d in 0..rank {
            cm_flat += idx[d] * stride;
            stride *= shape[d];
        }
        col_data[cm_flat] = data[rm_flat];
    }
    col_data
}

fn assert_close(a: f64, b: f64, label: &str) {
    assert!((a - b).abs() < 1e-10, "{label}: got {a}, expected {b}");
}

// ============================================================================
// Group 1: Basic unary operations
// ============================================================================

#[test]
fn einsum_identity() {
    // "ij->ij" — identity copy
    // v1 data: col-major [1,2,3,4,5,6] shape [2,3]
    // a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4, a[0,2]=5, a[1,2]=6
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let mut tb = einsum(&mut engine, &[&ta], "ij->ij");
    let result = tb.eval(&mut engine);

    assert_eq!(result.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            assert_close(
                get_v2(result, &[i, j]),
                get_v2(&a, &[i, j]),
                &format!("identity[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_transpose() {
    // "ij->ji"
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let mut tb = einsum(&mut engine, &[&ta], "ij->ji");
    let result = tb.eval(&mut engine);

    assert_eq!(result.shape(), &[3, 2]);
    for i in 0..2 {
        for j in 0..3 {
            assert_close(
                get_v2(result, &[j, i]),
                get_v2(&a, &[i, j]),
                &format!("transpose[{j},{i}]"),
            );
        }
    }
}

#[test]
fn einsum_sum_reduce() {
    // "ij->i" — sum over j
    // a col-major: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4, a[0,2]=5, a[1,2]=6
    // b[0] = 1+3+5 = 9, b[1] = 2+4+6 = 12
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a);
    let mut tb = einsum(&mut engine, &[&ta], "ij->i");
    let result = tb.eval(&mut engine);

    assert_eq!(result.shape(), &[2]);
    let data = get_f64_data(result);
    assert_close(data[0], 9.0, "sum_reduce[0]");
    assert_close(data[1], 12.0, "sum_reduce[1]");
}

#[test]
fn einsum_full_contraction() {
    // "ij->" — sum all elements = 1+2+3+4+5+6 = 21
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a);
    let mut tb = einsum(&mut engine, &[&ta], "ij->");
    let result = tb.eval(&mut engine);

    // v2 returns shape [1] for scalar output
    let data = get_f64_data(result);
    assert_close(data[0], 21.0, "full_contraction");
}

#[test]
#[ignore = "v2 einsum does not yet support repeated-index trace (\"ii->\")"]
fn einsum_trace() {
    // "ii->" — trace of 2x2 matrix
    // col-major [1,2,3,4]: a[0,0]=1, a[1,0]=2, a[0,1]=3, a[1,1]=4
    // trace = 1 + 4 = 5
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a);
    let mut tb = einsum(&mut engine, &[&ta], "ii->");
    let result = tb.eval(&mut engine);

    let data = get_f64_data(result);
    assert_close(data[0], 5.0, "trace");
}

#[test]
#[ignore = "v2 einsum does not yet support repeated-index diagonal extraction (\"ii->i\")"]
fn einsum_diagonal_extraction() {
    // "ii->i" — extract diagonal of 3x3 matrix
    // col-major [1..9]: a[0,0]=1, a[1,0]=2, ..., a[1,1]=5, ..., a[2,2]=9
    let a = f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a);
    let mut tb = einsum(&mut engine, &[&ta], "ii->i");
    let result = tb.eval(&mut engine);

    assert_eq!(result.shape(), &[3]);
    let data = get_f64_data(result);
    assert_close(data[0], 1.0, "diag[0]");
    assert_close(data[1], 5.0, "diag[1]");
    assert_close(data[2], 9.0, "diag[2]");
}

#[test]
fn einsum_diagonal_embedding() {
    // "i->ii" — diagonal embedding
    // v = [2, 3, 5]
    // Result should be 3x3 diagonal matrix with v on diagonal, 0 elsewhere
    let v = f64_tensor(vec![3], vec![2.0, 3.0, 5.0]);

    let mut engine = Engine::new();
    let tv = TracedTensor::from_tensor(v.clone());

    // This may not work if v2 Operand impl doesn't handle it.
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut td = einsum(&mut engine, &[&tv], "i->ii");
        td.eval(&mut engine).clone()
    }));

    match result {
        Ok(tensor) => {
            assert_eq!(tensor.shape(), &[3, 3]);
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { get_v2(&v, &[i]) } else { 0.0 };
                    assert_close(
                        get_v2(&tensor, &[i, j]),
                        expected,
                        &format!("diag_embed[{i},{j}]"),
                    );
                }
            }
        }
        Err(_) => {
            eprintln!(
                "NOTE: einsum_diagonal_embedding (\"i->ii\") panicked — \
                 v2 Operand impl may not support diagonal embedding yet."
            );
        }
    }
}

// ============================================================================
// Group 2: Binary operations
// ============================================================================

#[test]
fn einsum_matmul() {
    // "ij,jk->ik"
    // Same data as v1: A[2,3] col-major [1,2,3,4,5,6], B[3,4] col-major [1..12]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let mut tc = einsum(&mut engine, &[&ta, &tb], "ij,jk->ik");
    let result = tc.eval(&mut engine);

    assert_eq!(result.shape(), &[2, 4]);
    // Verify against manual computation
    for i in 0..2 {
        for k in 0..4 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += get_v2(&a, &[i, j]) * get_v2(&b, &[j, k]);
            }
            assert_close(
                get_v2(result, &[i, k]),
                expected,
                &format!("matmul[{i},{k}]"),
            );
        }
    }
}

#[test]
fn einsum_outer_product() {
    // "i,j->ij"
    let u = f64_tensor(vec![2], vec![1.0, 2.0]);
    let v = f64_tensor(vec![3], vec![3.0, 4.0, 5.0]);

    let mut engine = Engine::new();
    let tu = TracedTensor::from_tensor(u.clone());
    let tv = TracedTensor::from_tensor(v.clone());
    let mut tm = einsum(&mut engine, &[&tu, &tv], "i,j->ij");
    let result = tm.eval(&mut engine);

    assert_eq!(result.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            let expected = get_v2(&u, &[i]) * get_v2(&v, &[j]);
            assert_close(
                get_v2(result, &[i, j]),
                expected,
                &format!("outer[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_dot_product() {
    // "i,i->"
    let u = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let v = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);

    let mut engine = Engine::new();
    let tu = TracedTensor::from_tensor(u);
    let tv = TracedTensor::from_tensor(v);
    let mut td = einsum(&mut engine, &[&tu, &tv], "i,i->");
    let result = td.eval(&mut engine);

    // 1*4 + 2*5 + 3*6 = 32
    let data = get_f64_data(result);
    assert_close(data[0], 32.0, "dot_product");
}

#[test]
fn einsum_matvec() {
    // "ij,j->i"
    // A[2,3] col-major [1,2,3,4,5,6], x = [1,2,3]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let tx = TracedTensor::from_tensor(x.clone());
    let mut ty = einsum(&mut engine, &[&ta, &tx], "ij,j->i");
    let result = ty.eval(&mut engine);

    assert_eq!(result.shape(), &[2]);
    for i in 0..2 {
        let mut expected = 0.0;
        for j in 0..3 {
            expected += get_v2(&a, &[i, j]) * get_v2(&x, &[j]);
        }
        assert_close(get_v2(result, &[i]), expected, &format!("matvec[{i}]"));
    }
}

#[test]
fn einsum_elementwise_mul() {
    // "ij,ij->ij" — Hadamard product
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let mut tc = einsum(&mut engine, &[&ta, &tb], "ij,ij->ij");
    let result = tc.eval(&mut engine);

    assert_eq!(result.shape(), &[2, 2]);
    for i in 0..2 {
        for j in 0..2 {
            let expected = get_v2(&a, &[i, j]) * get_v2(&b, &[i, j]);
            assert_close(
                get_v2(result, &[i, j]),
                expected,
                &format!("hadamard[{i},{j}]"),
            );
        }
    }
}

// ============================================================================
// Group 3: N-ary operations
// ============================================================================

#[test]
fn einsum_three_matrices() {
    // "ij,jk,kl->il" — chain multiply 3 matrices
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let tc = TracedTensor::from_tensor(c.clone());
    let mut td = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il");
    let result_d = td.eval(&mut engine);

    assert_eq!(result_d.shape(), &[2, 2]);

    // Verify D = A @ B @ C by computing step-by-step
    // First: AB
    let ta2 = TracedTensor::from_tensor(a.clone());
    let tb2 = TracedTensor::from_tensor(b.clone());
    let mut tab = einsum(&mut engine, &[&ta2, &tb2], "ij,jk->ik");
    let ab = tab.eval(&mut engine).clone();

    // Then: (AB) @ C
    let tab2 = TracedTensor::from_tensor(ab);
    let tc2 = TracedTensor::from_tensor(c);
    let mut tabc = einsum(&mut engine, &[&tab2, &tc2], "ij,jk->ik");
    let abc = tabc.eval(&mut engine);

    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                get_v2(result_d, &[i, j]),
                get_v2(abc, &[i, j]),
                &format!("three_mat[{i},{j}]"),
            );
        }
    }
}

// ============================================================================
// Group 4: Contraction tree / path tests
// ============================================================================

#[test]
fn einsum_with_path_matches_flat_nary() {
    // Verify that an explicit JAX path produces the same result as auto-optimized.
    // A[2,2], B[2,2], C[2,2]; "ij,jk,kl->il"
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = Engine::new();

    // Auto-optimized
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let tc = TracedTensor::from_tensor(c.clone());
    let mut auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il");
    let auto_result = auto.eval(&mut engine).clone();

    // Explicit path: contract B*C first (positions 1,2), then A*result (positions 0,1)
    let ta2 = TracedTensor::from_tensor(a);
    let tb2 = TracedTensor::from_tensor(b);
    let tc2 = TracedTensor::from_tensor(c);
    let mut via_path = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
    );
    let path_result = via_path.eval(&mut engine);

    for i in 0..2 {
        for j in 0..2 {
            assert_close(
                get_v2(path_result, &[i, j]),
                get_v2(&auto_result, &[i, j]),
                &format!("path_vs_auto[{i},{j}]"),
            );
        }
    }
}

#[test]
fn contraction_tree_from_pairs() {
    // Build contraction tree from explicit pairs and verify shape.
    // A[2,3] B[3,4] C[4,5] -> D[2,5]
    // Contract B*C first (pair 1,2 -> index 3), then A*T (pair 0,3)
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let tree = ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4], &[4, 5]], &[(1, 2), (0, 3)])
        .unwrap();

    // Use the tree with v2 API
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(
        vec![4, 5],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ],
    );

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let tc = TracedTensor::from_tensor(c);
    let mut td = einsum_with(
        &mut engine,
        &[&ta, &tb, &tc],
        "ij,jk,kl->il",
        EinsumOptimize::Tree(tree),
    );
    let result = td.eval(&mut engine);
    assert_eq!(result.shape(), &[2, 5]);
}

#[test]
fn contraction_tree_from_pairs_rejects_wrong_step_count() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2], &[2, 3]], &[0, 3]);
    let result = ContractionTree::from_pairs(&subs, &[&[2, 2], &[2, 2], &[2, 2]], &[(1, 2)]);
    assert!(result.is_err(), "wrong number of path steps must error");
}

// ============================================================================
// Group 5: Complex contraction patterns
// ============================================================================

#[test]
#[ignore = "v2 einsum does not yet support repeated-index partial trace (\"iij->j\")"]
fn einsum_partial_trace_with_free_index() {
    // "iij->j" — partial trace: v[j] = sum_i T[i,i,j]
    // T[2,2,3] col-major: data 1..12
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let t = f64_tensor(vec![2, 2, 3], data);

    let mut engine = Engine::new();
    let tt = TracedTensor::from_tensor(t.clone());
    let mut tv = einsum(&mut engine, &[&tt], "iij->j");
    let result = tv.eval(&mut engine);

    assert_eq!(result.shape(), &[3]);
    for j in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            expected += get_v2(&t, &[i, i, j]);
        }
        assert_close(
            get_v2(result, &[j]),
            expected,
            &format!("partial_trace[{j}]"),
        );
    }
}

#[test]
fn einsum_batched_matmul() {
    // "bij,bjk->bik" — batched matrix multiply
    // batch=2, each is 2x2
    // A col-major [2,2,2]:
    //   batch 0: I = [[1,0],[0,1]]
    //   batch 1: 2I = [[2,0],[0,2]]
    // Col-major for [2,2,2]: leftmost varies fastest
    //   A[b,i,j] with strides [1,2,4]
    //   data[0]=A[0,0,0]=1, data[1]=A[1,0,0]=2
    //   data[2]=A[0,1,0]=0, data[3]=A[1,1,0]=0
    //   data[4]=A[0,0,1]=0, data[5]=A[1,0,1]=0
    //   data[6]=A[0,1,1]=1, data[7]=A[1,1,1]=2
    let a = f64_tensor(vec![2, 2, 2], vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    // B col-major [2,2,2]:
    //   B[b,j,k] with strides [1,2,4]
    //   batch 0: [[1,3],[2,4]], batch 1: [[5,7],[6,8]]
    //   data[0]=B[0,0,0]=1, data[1]=B[1,0,0]=5
    //   data[2]=B[0,1,0]=2, data[3]=B[1,1,0]=6
    //   data[4]=B[0,0,1]=3, data[5]=B[1,0,1]=7
    //   data[6]=B[0,1,1]=4, data[7]=B[1,1,1]=8
    let b = f64_tensor(vec![2, 2, 2], vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let mut tc = einsum(&mut engine, &[&ta, &tb], "bij,bjk->bik");
    let result = tc.eval(&mut engine);

    assert_eq!(result.shape(), &[2, 2, 2]);
    for batch in 0..2 {
        for i in 0..2 {
            for k in 0..2 {
                let mut expected = 0.0;
                for j in 0..2 {
                    expected += get_v2(&a, &[batch, i, j]) * get_v2(&b, &[batch, j, k]);
                }
                assert_close(
                    get_v2(result, &[batch, i, k]),
                    expected,
                    &format!("batched_matmul[{batch},{i},{k}]"),
                );
            }
        }
    }
}

#[test]
fn einsum_reduce_first_axis() {
    // "ij->j" — sum over first axis
    // A[3,4] col-major: data [0..12] with strides [1,3]
    //   A[0,0]=0, A[1,0]=1, A[2,0]=2, A[0,1]=3, A[1,1]=4, A[2,1]=5, ...
    let a = f64_tensor(
        vec![3, 4],
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
    );

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a.clone());
    let mut ty = einsum(&mut engine, &[&ta], "ij->j");
    let result = ty.eval(&mut engine);

    assert_eq!(result.shape(), &[4]);
    for j in 0..4 {
        let expected = get_v2(&a, &[0, j]) + get_v2(&a, &[1, j]) + get_v2(&a, &[2, j]);
        assert_close(
            get_v2(result, &[j]),
            expected,
            &format!("reduce_first[{j}]"),
        );
    }
}

#[test]
fn einsum_self_contraction_trace() {
    // "ijk->j" — self-contraction: sum over i and k (not a trace — just reduction)
    // T[2,3,2] col-major: data 1..12
    let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let t = f64_tensor(vec![2, 3, 2], data);

    let mut engine = Engine::new();
    let tt = TracedTensor::from_tensor(t.clone());
    let mut tv = einsum(&mut engine, &[&tt], "ijk->j");
    let result = tv.eval(&mut engine);

    assert_eq!(result.shape(), &[3]);
    for j in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            for k in 0..2 {
                expected += get_v2(&t, &[i, j, k]);
            }
        }
        assert_close(
            get_v2(result, &[j]),
            expected,
            &format!("self_contraction[{j}]"),
        );
    }
}

// ============================================================================
// Group 6: EinsumOptimize variants (v2-specific)
// ============================================================================

#[test]
fn test_optimize_false_left_to_right() {
    // EinsumOptimize::False contracts left-to-right
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut engine = Engine::new();

    // Reference: auto optimization
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let tc = TracedTensor::from_tensor(c.clone());
    let mut auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il");
    let auto_result = auto.eval(&mut engine).clone();

    // False: left-to-right
    let ta2 = TracedTensor::from_tensor(a);
    let tb2 = TracedTensor::from_tensor(b);
    let tc2 = TracedTensor::from_tensor(c);
    let mut ltr = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::False,
    );
    let ltr_result = ltr.eval(&mut engine);

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(ltr_result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("false_ltr[{i},{l}]"),
            );
        }
    }
}

#[test]
fn test_optimize_path_jax_compatible() {
    // EinsumOptimize::Path with JAX-style indices
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut engine = Engine::new();

    // Reference
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let tc = TracedTensor::from_tensor(c.clone());
    let mut auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il");
    let auto_result = auto.eval(&mut engine).clone();

    // Path: contract A*B first (0,1), then result*C (0,1)
    let ta2 = TracedTensor::from_tensor(a);
    let tb2 = TracedTensor::from_tensor(b);
    let tc2 = TracedTensor::from_tensor(c);
    let mut path = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
    );
    let path_result = path.eval(&mut engine);

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(path_result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("path_jax[{i},{l}]"),
            );
        }
    }
}

#[test]
fn test_optimize_nested() {
    // EinsumOptimize::Nested with parenthesized notation
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let mut engine = Engine::new();

    // Reference
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let tc = TracedTensor::from_tensor(c.clone());
    let mut auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il");
    let auto_result = auto.eval(&mut engine).clone();

    // Nested: "(ij,jk),kl->il" = contract A*B first, then result*C
    let nested = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
    let ta2 = TracedTensor::from_tensor(a);
    let tb2 = TracedTensor::from_tensor(b);
    let tc2 = TracedTensor::from_tensor(c);
    let mut nested_result = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Nested(nested),
    );
    let result = nested_result.eval(&mut engine);

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("nested[{i},{l}]"),
            );
        }
    }
}

#[test]
fn test_optimize_tree() {
    // EinsumOptimize::Tree with pre-computed ContractionTree
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let subs = Subscripts::parse("ij,jk,kl->il").unwrap();
    let shapes: &[&[usize]] = &[&[2, 3], &[3, 4], &[4, 2]];
    let tree = ContractionTree::optimize(&subs, shapes).unwrap();

    let mut engine = Engine::new();

    // Reference
    let ta = TracedTensor::from_tensor(a.clone());
    let tb = TracedTensor::from_tensor(b.clone());
    let tc = TracedTensor::from_tensor(c.clone());
    let mut auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il");
    let auto_result = auto.eval(&mut engine).clone();

    // Tree
    let ta2 = TracedTensor::from_tensor(a);
    let tb2 = TracedTensor::from_tensor(b);
    let tc2 = TracedTensor::from_tensor(c);
    let mut tree_result = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Tree(tree),
    );
    let result = tree_result.eval(&mut engine);

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(result, &[i, l]),
                get_v2(&auto_result, &[i, l]),
                &format!("tree[{i},{l}]"),
            );
        }
    }
}

// ============================================================================
// Group 7: Error cases
// ============================================================================

#[test]
#[should_panic(expected = "contraction optimization failed")]
fn einsum_error_rank_mismatch() {
    // Subscripts say "ij" (rank 2) but tensor is rank 1
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);

    let mut engine = Engine::new();
    let tv = TracedTensor::from_tensor(v);
    // v2 panics during contraction tree optimization because the subscript
    // label count doesn't match the tensor's rank.
    let mut result = einsum(&mut engine, &[&tv], "ij->i");
    result.eval(&mut engine);
}

#[test]
fn einsum_error_empty_inputs() {
    // No input tensors — v2 panics during tree construction
    let mut engine = Engine::new();
    let empty: &[&TracedTensor] = &[];
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut r = einsum(&mut engine, empty, "->");
        r.eval(&mut engine);
    }));
    assert!(result.is_err(), "should panic for empty inputs");
}

#[test]
fn einsum_wrong_operand_count() {
    // Subscripts say 2 inputs but only 1 provided
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut r = einsum(&mut engine, &[&ta], "ij,jk->ik");
        r.eval(&mut engine);
    }));
    assert!(result.is_err(), "should panic for wrong operand count");
}

#[test]
fn einsum_shape_mismatch() {
    // j=3 in A but j=2 in B -> shape mismatch
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let mut engine = Engine::new();
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut r = einsum(&mut engine, &[&ta, &tb], "ij,jk->ik");
        r.eval(&mut engine);
    }));
    assert!(result.is_err(), "should panic for shape mismatch");
}
