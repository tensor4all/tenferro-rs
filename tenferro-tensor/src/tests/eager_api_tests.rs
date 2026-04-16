use crate::{cpu::CpuBackend, Tensor, TensorBackend, TypedTensor};

fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() <= tol);
    }
}

#[test]
fn tensor_new_and_typed_tensor_as_slice_work() {
    let tensor = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let typed = TypedTensor::<f64>::from_vec(vec![2], vec![7.0, 8.0]);

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(
        tensor.as_slice::<f64>(),
        Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].as_slice())
    );
    assert_eq!(typed.as_slice(), &[7.0, 8.0]);
}

#[test]
fn eager_tensor_elementwise_and_structural_methods_match_backend_results() {
    let mut ctx = CpuBackend::new();
    fn needs_backend(_ctx: &mut impl TensorBackend) {}
    needs_backend(&mut ctx);

    let a = Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = Tensor::new(vec![3], vec![4.0_f64, 5.0, 6.0]);
    let sum = a.add(&b, &mut ctx).unwrap();
    let product = a.mul(&b, &mut ctx).unwrap();
    let negated = a.neg(&mut ctx).unwrap();

    assert_eq!(sum.as_slice::<f64>(), Some([5.0, 7.0, 9.0].as_slice()));
    assert_eq!(
        product.as_slice::<f64>(),
        Some([4.0, 10.0, 18.0].as_slice())
    );
    assert_eq!(
        negated.as_slice::<f64>(),
        Some([-1.0, -2.0, -3.0].as_slice())
    );

    let matrix = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let transposed = matrix.transpose(&[1, 0], &mut ctx).unwrap();
    let reshaped = matrix.reshape(&[3, 2], &mut ctx).unwrap();
    let reduced = matrix.reduce_sum(&[1], &mut ctx).unwrap();
    let rhs = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let matmul = matrix.matmul(&rhs, &mut ctx).unwrap();

    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(
        transposed.as_slice::<f64>(),
        Some([1.0, 3.0, 5.0, 2.0, 4.0, 6.0].as_slice())
    );
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(
        reshaped.as_slice::<f64>(),
        Some([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].as_slice())
    );
    assert_eq!(reduced.shape(), &[2]);
    assert_eq!(reduced.as_slice::<f64>(), Some([9.0, 12.0].as_slice()));
    assert_eq!(matmul.shape(), &[2, 2]);
    assert_eq!(
        matmul.as_slice::<f64>(),
        Some([22.0, 28.0, 49.0, 64.0].as_slice())
    );
}

#[test]
fn eager_tensor_linalg_methods_return_expected_shapes_and_values() {
    let mut ctx = CpuBackend::new();

    let tall = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (u, s, vt) = tall.svd(&mut ctx).unwrap();
    let (q, r) = tall.qr(&mut ctx).unwrap();
    assert_eq!(u.shape(), &[3, 2]);
    assert_eq!(s.shape(), &[2]);
    assert_eq!(vt.shape(), &[2, 2]);
    assert_eq!(q.shape(), &[3, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    let permutation = Tensor::new(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]);
    let (p, l, u, parity) = permutation.lu(&mut ctx).unwrap();
    assert_eq!(p.shape(), &[2, 2]);
    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(parity.shape(), &[] as &[usize]);

    let spd = Tensor::new(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 5.0]);
    let chol = spd.cholesky(&mut ctx).unwrap();
    assert_eq!(chol.shape(), &[2, 2]);
    assert_eq!(
        chol.as_slice::<f64>(),
        Some([2.0, 1.0, 0.0, 2.0].as_slice())
    );

    let (eigh_values, eigh_vectors) = spd.eigh(&mut ctx).unwrap();
    assert_eq!(eigh_values.shape(), &[2]);
    assert_eq!(eigh_vectors.shape(), &[2, 2]);

    let diagonal = Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]);
    let (eig_values, eig_vectors) = diagonal.eig(&mut ctx).unwrap();
    assert_eq!(eig_values.shape(), &[2]);
    assert_eq!(eig_vectors.shape(), &[2, 2]);

    let a = Tensor::new(vec![2, 2], vec![2.0_f64, 1.0, 1.0, 2.0]);
    let b = Tensor::new(vec![2, 1], vec![1.0_f64, 0.0]);
    let x = a.solve(&b, &mut ctx).unwrap();
    assert_eq!(x.shape(), &[2, 1]);
    assert_close(
        x.as_slice::<f64>().unwrap(),
        &[2.0 / 3.0, -1.0 / 3.0],
        1.0e-10,
    );

    let triangular = Tensor::new(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0]);
    let rhs = Tensor::new(vec![2, 1], vec![2.0_f64, 7.0]);
    let triangular_x = triangular
        .triangular_solve(&rhs, true, true, false, false, &mut ctx)
        .unwrap();
    assert_eq!(triangular_x.shape(), &[2, 1]);
    assert_close(
        triangular_x.as_slice::<f64>().unwrap(),
        &[1.0, 2.0],
        1.0e-10,
    );
}
