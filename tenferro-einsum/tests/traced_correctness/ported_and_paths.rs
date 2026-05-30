use super::*;

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

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference: auto optimization
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // False: left-to-right
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let ltr = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::False,
    )
    .unwrap();
    let ltr_result = ltr.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&ltr_result, &[i, l]),
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

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // Path: contract A*B first (0,1), then result*C (0,1)
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let path = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(0, 1), (0, 1)]),
    )
    .unwrap();
    let path_result = path.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&path_result, &[i, l]),
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

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // Nested: "(ij,jk),kl->il" = contract A*B first, then result*C
    let nested = NestedEinsum::parse("(ij,jk),kl->il").unwrap();
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let nested_result = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Nested(nested),
    )
    .unwrap();
    let result = nested_result.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&result, &[i, l]),
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

    let mut engine = GraphExecutor::new(CpuBackend::new());

    // Reference
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let auto = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,kl->il").unwrap();
    let auto_result = auto.run_with(&mut engine).unwrap().clone();

    // Tree
    let ta2 = TracedTensor::from_tensor_concrete_shape(a);
    let tb2 = TracedTensor::from_tensor_concrete_shape(b);
    let tc2 = TracedTensor::from_tensor_concrete_shape(c);
    let tree_result = einsum_with(
        &mut engine,
        &[&ta2, &tb2, &tc2],
        "ij,jk,kl->il",
        EinsumOptimize::Tree(tree),
    )
    .unwrap();
    let result = tree_result.run_with(&mut engine).unwrap();

    for i in 0..2 {
        for l in 0..2 {
            assert_close(
                get_v2(&result, &[i, l]),
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
fn einsum_error_rank_mismatch() {
    // Subscripts say "ij" (rank 2) but tensor is rank 1
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tv = TracedTensor::from_tensor_concrete_shape(v);
    // v2 returns Err during contraction tree optimization because the subscript
    // label count doesn't match the tensor's rank.
    let result = einsum(&mut engine, &[&tv], "ij->i");
    assert!(result.is_err());
}

#[test]
fn einsum_error_empty_inputs() {
    // No input tensors -- v2 returns Err during tree construction
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let empty: &[&TracedTensor] = &[];
    let result = einsum(&mut engine, empty, "->");
    assert!(result.is_err());
}

#[test]
fn einsum_wrong_operand_count() {
    // Subscripts say 2 inputs but only 1 provided
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let result = einsum(&mut engine, &[&ta], "ij,jk->ik");
    assert!(result.is_err());
}

#[test]
fn einsum_shape_mismatch() {
    // j=3 in A but j=2 in B -> shape mismatch
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let result = einsum(&mut engine, &[&ta, &tb], "ij,jk->ik");
    assert!(result.is_err());
}

// ============================================================================
// Group 8: Ported repeated-index and diagonal-extract patterns
// ============================================================================

#[test]
fn einsum_trace_rank3_to_vector() {
    // "iji->j" — trace of rank-3 tensor: v[j] = sum_i T[i,j,i]
    // T[3,2,3] col-major: data 1..18
    let data: Vec<f64> = (1..=18).map(|x| x as f64).collect();
    let t = f64_tensor(vec![3, 2, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tt = TracedTensor::from_tensor_concrete_shape(t.clone());
    let tv = einsum(&mut engine, &[&tt], "iji->j").unwrap();
    let result = tv.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    for j in 0..2 {
        let mut expected = 0.0;
        for i in 0..3 {
            expected += get_v2(&t, &[i, j, i]);
        }
        assert_close(
            get_v2(&result, &[j]),
            expected,
            &format!("trace_rank3[{j}]"),
        );
    }
}

#[test]
fn einsum_multi_pair_trace_iijj() {
    // "iijj->" — trace over two pairs of repeated indices
    let n = 3;
    let total = n * n * n * n;
    let data: Vec<f64> = (0..total).map(|x| x as f64).collect();
    let a = f64_tensor(vec![n, n, n, n], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ts = einsum(&mut engine, &[&ta], "iijj->").unwrap();
    let result = ts.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let mut expected = 0.0;
    for i in 0..n {
        for j in 0..n {
            expected += get_v2(&a, &[i, i, j, j]);
        }
    }
    assert_close(get_f64_data(&result)[0], expected, "multi_pair_trace_iijj");
}

#[test]
fn einsum_diag_extract_reduce_ijj_to_i() {
    // "ijj->i" — extract diagonal along axes 1,2, then free axis 0
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = f64_tensor(vec![2, 3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "ijj->i").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    for i in 0..2 {
        let mut expected = 0.0;
        for j in 0..3 {
            expected += get_v2(&a, &[i, j, j]);
        }
        assert_close(get_v2(&result, &[i]), expected, &format!("ijj_to_i[{i}]"));
    }
}

#[test]
fn einsum_diag_extract_no_reduce_ijj_to_ij() {
    // "ijj->ij" — extract diagonal along axes 1,2, keep both i and j
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = f64_tensor(vec![2, 3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "ijj->ij").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    for i in 0..2 {
        for j in 0..3 {
            let expected = get_v2(&a, &[i, j, j]);
            assert_close(
                get_v2(&result, &[i, j]),
                expected,
                &format!("ijj_to_ij[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_diag_extract_permuted_jii_to_j() {
    // "jii->j" — extract diagonal along axes 1,2, reduce over i
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let a = f64_tensor(vec![3, 2, 2], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "jii->j").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for j in 0..3 {
        let expected = get_v2(&a, &[j, 0, 0]) + get_v2(&a, &[j, 1, 1]);
        assert_close(get_v2(&result, &[j]), expected, &format!("jii_to_j[{j}]"));
    }
}

#[test]
fn einsum_scalar_vector_products() {
    // v2 scalars use shape [] (rank 0, 1 element)
    let three = f64_tensor(vec![], vec![3.0]);
    let two = f64_tensor(vec![], vec![2.0]);
    let v4 = f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let v3 = f64_tensor(vec![3], vec![10.0, 20.0, 30.0]);
    let seven = f64_tensor(vec![], vec![7.0]);

    // ",k->k": scalar * vector
    {
        let mut engine = GraphExecutor::new(CpuBackend::new());
        let ts = TracedTensor::from_tensor_concrete_shape(three.clone());
        let tv = TracedTensor::from_tensor_concrete_shape(v4.clone());
        let result = einsum(&mut engine, &[&ts, &tv], ",k->k").unwrap();
        let r = result.run_with(&mut engine).unwrap();
        assert_eq!(r.shape(), &[4]);
        assert_close(get_v2(&r, &[0]), 3.0, "scalar*vec[0]");
        assert_close(get_v2(&r, &[1]), 6.0, "scalar*vec[1]");
        assert_close(get_v2(&r, &[2]), 9.0, "scalar*vec[2]");
        assert_close(get_v2(&r, &[3]), 12.0, "scalar*vec[3]");
    }

    // "i,->i": vector * scalar
    {
        let mut engine = GraphExecutor::new(CpuBackend::new());
        let tv = TracedTensor::from_tensor_concrete_shape(v3.clone());
        let ts = TracedTensor::from_tensor_concrete_shape(two.clone());
        let result = einsum(&mut engine, &[&tv, &ts], "i,->i").unwrap();
        let r = result.run_with(&mut engine).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_close(get_v2(&r, &[0]), 20.0, "vec*scalar[0]");
        assert_close(get_v2(&r, &[1]), 40.0, "vec*scalar[1]");
        assert_close(get_v2(&r, &[2]), 60.0, "vec*scalar[2]");
    }

    // ",->": scalar * scalar
    {
        let mut engine = GraphExecutor::new(CpuBackend::new());
        let ts = TracedTensor::from_tensor_concrete_shape(three.clone());
        let t7 = TracedTensor::from_tensor_concrete_shape(seven.clone());
        let result = einsum(&mut engine, &[&ts, &t7], ",->").unwrap();
        let r = result.run_with(&mut engine).unwrap();
        assert!(r.shape().is_empty());
        assert_close(get_f64_data(&r)[0], 21.0, "scalar*scalar");
    }
}

// ============================================================================
// Group 9: Error cases (ported from v1)
// ============================================================================

#[test]
fn einsum_error_output_label_not_in_input() {
    // Output references label 'k' which is not in any input
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let err = match einsum(&mut engine, &[&ta], "ij->ik") {
        Ok(_) => panic!("expected missing output label error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("einsum output label"));
}

#[test]
fn einsum_error_non_square_trace() {
    // Trace "ii->" but matrix is 2x3 (non-square)
    // v2 catches this at contraction tree optimization time
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let result = einsum(&mut engine, &[&ta], "ii->");
    assert!(result.is_err());
}

// ============================================================================
// Group 10: Path tests (ported from v1)
// ============================================================================

#[test]
#[should_panic(expected = "references operand positions")]
fn einsum_with_path_invalid_pairs_errors() {
    // Invalid JAX path with out-of-bounds indices should panic during tree build
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = TracedTensor::from_tensor_concrete_shape(c);
    let r = einsum_with(
        &mut engine,
        &[&ta, &tb, &tc],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(0, 99), (0, 1)]),
    )
    .unwrap();
    r.run_with(&mut engine).unwrap();
}

#[test]
fn einsum_with_path_rejects_structurally_invalid_paths() {
    // Wrong step count: 3 operands needs 2 steps, giving 1 is structurally wrong
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let c = f64_tensor(vec![2, 2], vec![9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = TracedTensor::from_tensor_concrete_shape(c);
    let result = einsum_with(
        &mut engine,
        &[&ta, &tb, &tc],
        "ij,jk,kl->il",
        EinsumOptimize::Path(vec![(1, 2)]),
    );
    assert!(result.is_err());
}

// ============================================================================
// Group 11: Ported v1 tests — batched, transposed, binary diagonal patterns
// ============================================================================

fn c64_tensor(shape: Vec<usize>, data: Vec<Complex64>) -> Tensor {
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data))
}

fn get_c64(t: &Tensor, idx: &[usize]) -> Complex64 {
    match t {
        Tensor::C64(inner) => *inner.get(idx),
        _ => panic!("expected C64"),
    }
}

fn assert_close_c64(a: Complex64, b: Complex64, label: &str) {
    assert!((a - b).norm() < 1e-10, "{label}: got {a}, expected {b}");
}

#[test]
fn einsum_batched_dot_product() {
    // "bi,bi->b" — batched dot product
    let a = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "bi,bi->b").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for batch in 0..3 {
        let mut expected = 0.0;
        for i in 0..2 {
            expected += get_v2(&a, &[batch, i]) * get_v2(&b, &[batch, i]);
        }
        assert_close(
            get_v2(&result, &[batch]),
            expected,
            &format!("batched_dot[{batch}]"),
        );
    }
}

#[test]
fn einsum_transposed_rhs_contraction() {
    // "ij,kj->ik" — contraction with transposed RHS
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "ij,kj->ik").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    for i in 0..2 {
        for k in 0..2 {
            let mut expected = 0.0;
            for j in 0..3 {
                expected += get_v2(&a, &[i, j]) * get_v2(&b, &[k, j]);
            }
            assert_close(
                get_v2(&result, &[i, k]),
                expected,
                &format!("transposed_rhs[{i},{k}]"),
            );
        }
    }
}

#[test]
fn einsum_cyclic_trace() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let c = f64_tensor(vec![4, 2], vec![2.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = TracedTensor::from_tensor_concrete_shape(c.clone());
    let out = einsum(&mut engine, &[&ta, &tb, &tc], "ij,jk,ki->").unwrap();
    let result = out.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let mut expected = 0.0;
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                expected += get_v2(&a, &[i, j]) * get_v2(&b, &[j, k]) * get_v2(&c, &[k, i]);
            }
        }
    }
    assert_close(get_f64_data(&result)[0], expected, "cyclic_trace");
}

#[test]
fn einsum_three_way_repeated_label_trace() {
    // "iii->" — triple-repeated index trace
    let data: Vec<f64> = (1..=27).map(|x| x as f64).collect();
    let t = f64_tensor(vec![3, 3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tt = TracedTensor::from_tensor_concrete_shape(t.clone());
    let ts = einsum(&mut engine, &[&tt], "iii->").unwrap();
    let result = ts.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let mut expected = 0.0;
    for i in 0..3 {
        expected += get_v2(&t, &[i, i, i]);
    }
    assert_close(get_f64_data(&result)[0], expected, "iii_trace");
}

#[test]
fn einsum_binary_diag_ii_jk_to_ijk() {
    // "ii,jk->ijk" — extract diagonal from A, outer product with B
    let a = f64_tensor(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );
    let b = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b.clone());
    let tc = einsum(&mut engine, &[&ta, &tb], "ii,jk->ijk").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 2, 2]);
    let diag_a = [1.0, 2.0, 3.0];
    for (i, &diag) in diag_a.iter().enumerate() {
        for j in 0..2 {
            for k in 0..2 {
                let expected = diag * get_v2(&b, &[j, k]);
                assert_close(
                    get_v2(&result, &[i, j, k]),
                    expected,
                    &format!("ii_jk_ijk[{i},{j},{k}]"),
                );
            }
        }
    }
}

#[test]
fn einsum_binary_diag_iij_jk_to_ik() {
    // "iij,jk->ik" — diagonal extraction + contraction
    let a = {
        let mut t = TypedTensor::<f64>::zeros(vec![2, 2, 3]);
        *t.get_mut(&[0, 0, 0]) = 1.0;
        *t.get_mut(&[1, 1, 0]) = 4.0;
        *t.get_mut(&[0, 0, 1]) = 2.0;
        *t.get_mut(&[1, 1, 1]) = 5.0;
        *t.get_mut(&[0, 0, 2]) = 3.0;
        *t.get_mut(&[1, 1, 2]) = 6.0;
        Tensor::F64(t)
    };
    let b = {
        let mut t = TypedTensor::<f64>::zeros(vec![3, 2]);
        *t.get_mut(&[0, 0]) = 1.0;
        *t.get_mut(&[1, 1]) = 1.0;
        Tensor::F64(t)
    };

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "iij,jk->ik").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    assert_close(get_v2(&result, &[0, 0]), 1.0, "iij_jk_ik[0,0]");
    assert_close(get_v2(&result, &[0, 1]), 2.0, "iij_jk_ik[0,1]");
    assert_close(get_v2(&result, &[1, 0]), 4.0, "iij_jk_ik[1,0]");
    assert_close(get_v2(&result, &[1, 1]), 5.0, "iij_jk_ik[1,1]");
}

#[test]
fn einsum_binary_diag_ii_jj_to_ij() {
    // "ii,jj->ij" — outer product of two traces
    let a = {
        let mut t = TypedTensor::<f64>::zeros(vec![3, 3]);
        *t.get_mut(&[0, 0]) = 1.0;
        *t.get_mut(&[1, 1]) = 2.0;
        *t.get_mut(&[2, 2]) = 3.0;
        Tensor::F64(t)
    };
    let b = {
        let mut t = TypedTensor::<f64>::zeros(vec![2, 2]);
        *t.get_mut(&[0, 0]) = 10.0;
        *t.get_mut(&[1, 1]) = 20.0;
        Tensor::F64(t)
    };

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "ii,jj->ij").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 2]);
    let diag_a = [1.0, 2.0, 3.0];
    let diag_b = [10.0, 20.0];
    for (i, &diag_left) in diag_a.iter().enumerate() {
        for (j, &diag_right) in diag_b.iter().enumerate() {
            assert_close(
                get_v2(&result, &[i, j]),
                diag_left * diag_right,
                &format!("ii_jj_ij[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_binary_diag_ii_j_to_j() {
    // "ii,j->j" — trace times vector
    let a = f64_tensor(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let b = f64_tensor(vec![2], vec![2.0, 3.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "ii,j->j").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2]);
    let trace = get_v2(&a, &[0, 0]) + get_v2(&a, &[1, 1]) + get_v2(&a, &[2, 2]);
    assert_close(get_v2(&result, &[0]), trace * 2.0, "ii_j_to_j[0]");
    assert_close(get_v2(&result, &[1]), trace * 3.0, "ii_j_to_j[1]");
}

#[test]
fn einsum_input_output_repeated_iij_to_jj() {
    // "iij->jj" — trace over i, then embed diagonal over j
    let data: Vec<f64> = (0..18).map(|x| x as f64).collect();
    let a = f64_tensor(vec![3, 3, 2], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "iij->jj").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    for j1 in 0..2 {
        for j2 in 0..2 {
            let expected = if j1 == j2 {
                let mut s = 0.0;
                for i in 0..3 {
                    s += get_v2(&a, &[i, i, j1]);
                }
                s
            } else {
                0.0
            };
            assert_close(
                get_v2(&result, &[j1, j2]),
                expected,
                &format!("iij_to_jj[{j1},{j2}]"),
            );
        }
    }
}

#[test]
fn einsum_input_output_repeated_ii_to_ii() {
    // "ii->ii" — extract diagonal then re-embed
    let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
    let a = f64_tensor(vec![3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ty = einsum(&mut engine, &[&ta], "ii->ii").unwrap();
    let result = ty.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { get_v2(&a, &[i, i]) } else { 0.0 };
            assert_close(
                get_v2(&result, &[i, j]),
                expected,
                &format!("ii_to_ii[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_unit_extent_contraction() {
    // "abcdef,ace->bdf" — contraction with unit-extent dimensions
    let a = f64_tensor(vec![1, 1, 1, 1, 1, 1], vec![2.0]);
    let b = f64_tensor(vec![1, 1, 1], vec![3.0]);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a);
    let tb = TracedTensor::from_tensor_concrete_shape(b);
    let tc = einsum(&mut engine, &[&ta, &tb], "abcdef,ace->bdf").unwrap();
    let result = tc.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[1, 1, 1]);
    assert_close(get_v2(&result, &[0, 0, 0]), 6.0, "unit_extent");
}

// ============================================================================
// Group 12: Complex number einsum tests
// ============================================================================

#[test]
fn einsum_complex_diagonal_extraction() {
    // "ii->i" with complex numbers
    let data: Vec<Complex64> = (1..=9)
        .map(|x| Complex64::new(x as f64, -(x as f64)))
        .collect();
    let a = c64_tensor(vec![3, 3], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let tb = einsum(&mut engine, &[&ta], "ii->i").unwrap();
    let result = tb.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3]);
    for i in 0..3 {
        assert_close_c64(
            get_c64(&result, &[i]),
            get_c64(&a, &[i, i]),
            &format!("complex_diag[{i}]"),
        );
    }
}

#[test]
fn einsum_complex_diagonal_embedding() {
    // "i->ii" with complex numbers
    let v = c64_tensor(
        vec![3],
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(5.0, 0.5),
        ],
    );

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let tv = TracedTensor::from_tensor_concrete_shape(v.clone());
    let td = einsum(&mut engine, &[&tv], "i->ii").unwrap();
    let result = td.run_with(&mut engine).unwrap();

    assert_eq!(result.shape(), &[3, 3]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j {
                get_c64(&v, &[i])
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert_close_c64(
                get_c64(&result, &[i, j]),
                expected,
                &format!("complex_embed[{i},{j}]"),
            );
        }
    }
}

#[test]
fn einsum_complex_trace() {
    // "ii->" with complex numbers
    let data: Vec<Complex64> = (1..=4)
        .map(|x| Complex64::new(x as f64, 0.5 * x as f64))
        .collect();
    let a = c64_tensor(vec![2, 2], data);

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let ta = TracedTensor::from_tensor_concrete_shape(a.clone());
    let ts = einsum(&mut engine, &[&ta], "ii->").unwrap();
    let result = ts.run_with(&mut engine).unwrap();

    assert!(result.shape().is_empty());
    let expected = get_c64(&a, &[0, 0]) + get_c64(&a, &[1, 1]);
    match result {
        Tensor::C64(inner) => {
            assert_close_c64(inner.host_data()[0], expected, "complex_trace");
        }
        _ => panic!("expected C64"),
    }
}
