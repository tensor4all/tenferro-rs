use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend};

use crate::eager_einsum;

#[test]
fn eager_einsum_executes_binary_and_ternary_contractions() {
    let mut ctx = CpuBackend::new();
    fn needs_backend(_ctx: &mut impl TensorBackend) {}
    needs_backend(&mut ctx);

    let a = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let matmul = eager_einsum(&mut ctx, &[&a, &b], "ij,jk->ik").unwrap();
    assert_eq!(matmul.shape(), &[2, 2]);
    assert_eq!(
        matmul.as_slice::<f64>(),
        Some([22.0, 28.0, 49.0, 64.0].as_slice())
    );

    let c = Tensor::new(vec![2, 1], vec![1.0_f64, 2.0]);
    let chain = eager_einsum(&mut ctx, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
    assert_eq!(chain.shape(), &[2, 1]);
    assert_eq!(chain.as_slice::<f64>(), Some([120.0, 156.0].as_slice()));
}

#[test]
fn eager_einsum_handles_outer_products_and_diagonal_patterns() {
    let mut ctx = CpuBackend::new();

    let lhs = Tensor::new(vec![2], vec![1.0_f64, 2.0]);
    let rhs = Tensor::new(vec![3], vec![3.0_f64, 4.0, 5.0]);
    let outer = eager_einsum(&mut ctx, &[&lhs, &rhs], "i,j->ij").unwrap();
    assert_eq!(outer.shape(), &[2, 3]);
    assert_eq!(
        outer.as_slice::<f64>(),
        Some([3.0, 6.0, 4.0, 8.0, 5.0, 10.0].as_slice())
    );

    let matrix = Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    let diagonal = eager_einsum(&mut ctx, &[&matrix], "ii->i").unwrap();
    let trace = eager_einsum(&mut ctx, &[&matrix], "ii->").unwrap();
    assert_eq!(diagonal.shape(), &[2]);
    assert_eq!(diagonal.as_slice::<f64>(), Some([1.0, 4.0].as_slice()));
    assert_eq!(trace.shape(), &[] as &[usize]);
    assert_eq!(trace.as_slice::<f64>(), Some([5.0].as_slice()));

    let embedded = eager_einsum(&mut ctx, &[&lhs], "i->ii").unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(
        embedded.as_slice::<f64>(),
        Some([1.0, 0.0, 0.0, 2.0].as_slice())
    );
}

#[test]
fn eager_einsum_rejects_empty_inputs_and_operand_count_mismatch() {
    let mut ctx = CpuBackend::new();
    let tensor = Tensor::new(vec![2], vec![1.0_f64, 2.0]);

    let empty = eager_einsum(&mut ctx, &[], "->").unwrap_err();
    assert!(matches!(
        empty,
        tenferro_tensor::Error::InvalidConfig {
            op: "eager_einsum",
            ..
        }
    ));

    let mismatch = eager_einsum(&mut ctx, &[&tensor], "i,j->ij").unwrap_err();
    assert!(matches!(
        mismatch,
        tenferro_tensor::Error::InvalidConfig {
            op: "eager_einsum",
            ..
        }
    ));
}
