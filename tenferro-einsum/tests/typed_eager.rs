use tenferro_einsum::typed_eager_einsum;
use tenferro_tensor::{cpu::CpuBackend, TypedTensor};

#[test]
fn typed_einsum_f64() {
    let mut ctx = CpuBackend::new();
    let lhs = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = TypedTensor::<f64>::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = typed_eager_einsum(&mut ctx, &[&lhs, &rhs], "ij,jk->ik").unwrap();

    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.as_slice(), &[22.0, 28.0, 49.0, 64.0]);
}
