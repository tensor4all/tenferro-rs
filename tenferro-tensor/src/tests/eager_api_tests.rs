use crate::{cpu::CpuBackend, Tensor, TensorBackend, TypedTensor};

#[test]
fn tensor_new_and_typed_tensor_as_slice_work() {
    let tensor = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let typed = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![7.0, 8.0]);

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

    let a = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]);
    let b = Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]);
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

    let matrix = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let transposed = matrix.transpose(&[1, 0], &mut ctx).unwrap();
    let reshaped = matrix.reshape(&[3, 2], &mut ctx).unwrap();
    let reduced = matrix.reduce_sum(&[1], &mut ctx).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
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
