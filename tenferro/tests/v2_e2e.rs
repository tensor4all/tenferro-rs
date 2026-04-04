use tenferro::v2::engine::Engine;
use tenferro::v2::traced::{eval_all, TracedTensor};
use tenferro_ops::config::DotGeneralConfig;
use tenferro_tensor::v2::{Tensor, TypedTensor};

// ============================================================================
// Einsum integration tests
// ============================================================================

#[test]
fn test_einsum_matmul() {
    // A[2,3] x B[3,2] -> C[2,2]
    // Column-major: A = [[1,3,5],[2,4,6]], B = [[1,4],[2,5],[3,6]]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = TracedTensor::traced_einsum("ij,jk->ik", &[&ta, &tb]);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);
    let data = get_f64_data(result);
    // C = A*B = [[1*1+3*2+5*3, 1*4+3*5+5*6], [2*1+4*2+6*3, 2*4+4*5+6*6]]
    //         = [[22, 49], [28, 64]]
    // col-major: [22, 28, 49, 64]
    assert_eq!(data, &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_einsum_inner_product() {
    // dot(a, b) = sum_i a_i * b_i
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = TracedTensor::traced_einsum("i,i->", &[&ta, &tb]);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);
    let data = get_f64_data(result);
    // 1*4 + 2*5 + 3*6 = 32
    assert_eq!(data, &[32.0]);
}

#[test]
fn test_einsum_row_sum() {
    // sum over columns: A[2,3] -> v[2]
    // A = [[1,3,5],[2,4,6]] col-major: [1,2,3,4,5,6]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let mut tc = TracedTensor::traced_einsum("ij->i", &[&ta]);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);
    let data = get_f64_data(result);
    // row sums: [1+3+5, 2+4+6] = [9, 12]
    assert_eq!(data, &[9.0, 12.0]);
}

#[test]
fn test_einsum_hadamard() {
    // Element-wise product: A[2,2] .* B[2,2]
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = TracedTensor::traced_einsum("ij,ij->ij", &[&ta, &tb]);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);
    let data = get_f64_data(result);
    assert_eq!(data, &[5.0, 12.0, 21.0, 32.0]);
}

#[test]
fn test_einsum_outer_product() {
    // Outer product: a[3] x b[2] -> C[3,2]
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![2], vec![4.0, 5.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = TracedTensor::traced_einsum("i,j->ij", &[&ta, &tb]);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);
    let data = get_f64_data(result);
    // C[i,j] = a[i]*b[j]
    // C = [[1*4, 1*5], [2*4, 2*5], [3*4, 3*5]]
    //   = [[4, 5], [8, 10], [12, 15]]
    // col-major: [4, 8, 12, 5, 10, 15]
    assert_eq!(data, &[4.0, 8.0, 12.0, 5.0, 10.0, 15.0]);
}

#[test]
fn test_einsum_matvec() {
    // Matrix-vector product: A[2,3] x v[3] -> w[2]
    // A = [[1,3,5],[2,4,6]] col-major: [1,2,3,4,5,6]
    // v = [1, 2, 3]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);

    let ta = TracedTensor::from_tensor(a);
    let tv = TracedTensor::from_tensor(v);
    let mut tc = TracedTensor::traced_einsum("ij,j->i", &[&ta, &tv]);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);
    let data = get_f64_data(result);
    // w = A*v = [1*1+3*2+5*3, 2*1+4*2+6*3] = [22, 28]
    assert_eq!(data, &[22.0, 28.0]);
}

#[test]
fn test_einsum_transpose() {
    // Transpose: A[2,3] -> A^T[3,2] via "ij->ji"
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let mut tc = TracedTensor::traced_einsum("ij->ji", &[&ta]);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);
    let data = get_f64_data(result);
    // A^T col-major: [1,3,5, 2,4,6]
    assert_eq!(data, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn get_f64_data(t: &Tensor) -> &[f64] {
    match t {
        Tensor::F64(inner) => inner.host_data(),
        _ => panic!("expected F64"),
    }
}

#[test]
fn test_e2e_add() {
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = ta.traced_add(&tb);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);

    let data = get_f64_data(result);
    assert_eq!(data, &[5.0, 7.0, 9.0]);
}

#[test]
fn test_e2e_mul() {
    let a = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let b = f64_tensor(vec![3], vec![4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut tc = ta.traced_mul(&tb);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);

    let data = get_f64_data(result);
    assert_eq!(data, &[4.0, 10.0, 18.0]);
}

#[test]
fn test_e2e_neg() {
    let a = f64_tensor(vec![3], vec![1.0, -2.0, 3.0]);

    let ta = TracedTensor::from_tensor(a);
    let mut tb = ta.traced_neg();

    let mut engine = Engine::new();
    let result = tb.eval(&mut engine);

    let data = get_f64_data(result);
    assert_eq!(data, &[-1.0, 2.0, -3.0]);
}

#[test]
fn test_e2e_matmul() {
    // A[2,3] x B[3,2] -> C[2,2]
    // Column-major: A stored as [a00, a10, a01, a11, a02, a12]
    // A = [[1,3,5],[2,4,6]] -> col-major: [1,2, 3,4, 5,6]
    // B = [[1,4],[2,5],[3,6]] -> col-major: [1,2,3, 4,5,6]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let mut tc = ta.traced_dot_general(&tb, config);

    let mut engine = Engine::new();
    let result = tc.eval(&mut engine);

    // A = [[1,3,5],[2,4,6]], B = [[1,4],[2,5],[3,6]]
    // C = A*B = [[1*1+3*2+5*3, 1*4+3*5+5*6], [2*1+4*2+6*3, 2*4+4*5+6*6]]
    //         = [[22, 49], [28, 64]]
    // col-major: [22, 28, 49, 64]
    let data = get_f64_data(result);
    assert_eq!(data, &[22.0, 28.0, 49.0, 64.0]);
}

#[test]
fn test_e2e_broadcast_reduce() {
    // v = [1, 2, 3] (shape [3])
    // broadcast to [3, 2] with dims=[0]
    // then reduce_sum(axes=[1])
    // = [1+1, 2+2, 3+3] = [2, 4, 6]
    let v = f64_tensor(vec![3], vec![1.0, 2.0, 3.0]);
    let tv = TracedTensor::from_tensor(v);
    let tb = tv.traced_broadcast_in_dim(&[3, 2], &[0]);
    let mut tr = tb.traced_reduce_sum(&[1]);

    let mut engine = Engine::new();
    let result = tr.eval(&mut engine);
    let data = get_f64_data(result);
    assert_eq!(data, &[2.0, 4.0, 6.0]);
}

#[test]
fn test_e2e_transpose() {
    // A[2,3] -> A^T[3,2]
    // A col-major: [a00, a10, a01, a11, a02, a12] = [1,2,3,4,5,6]
    // A = [[1,3,5],[2,4,6]]
    // A^T = [[1,2],[3,4],[5,6]]
    // A^T col-major: [1,3,5, 2,4,6]
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor(a);
    let mut tb = ta.traced_transpose(&[1, 0]);

    let mut engine = Engine::new();
    let result = tb.eval(&mut engine);
    let data = get_f64_data(result);
    assert_eq!(data, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
}

#[test]
fn test_e2e_reshape() {
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let ta = TracedTensor::from_tensor(a);
    let mut tb = ta.traced_reshape(&[6]);

    let mut engine = Engine::new();
    let result = tb.eval(&mut engine);
    let data = get_f64_data(result);
    assert_eq!(data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_e2e_eval_all() {
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);
    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let mut sum = ta.traced_add(&tb);
    let mut prod = ta.traced_mul(&tb);

    let mut engine = Engine::new();
    let results = eval_all(&mut engine, &mut [&mut sum, &mut prod]);

    assert_eq!(get_f64_data(&results[0]), &[4.0, 6.0]);
    assert_eq!(get_f64_data(&results[1]), &[3.0, 8.0]);
}

#[test]
fn test_e2e_chained_ops() {
    // (a + b) * c
    let a = f64_tensor(vec![2], vec![1.0, 2.0]);
    let b = f64_tensor(vec![2], vec![3.0, 4.0]);
    let c = f64_tensor(vec![2], vec![10.0, 20.0]);

    let ta = TracedTensor::from_tensor(a);
    let tb = TracedTensor::from_tensor(b);
    let tc = TracedTensor::from_tensor(c);

    let sum = ta.traced_add(&tb);
    let mut result = sum.traced_mul(&tc);

    let mut engine = Engine::new();
    let out = result.eval(&mut engine);
    let data = get_f64_data(out);
    // (1+3)*10=40, (2+4)*20=120
    assert_eq!(data, &[40.0, 120.0]);
}
