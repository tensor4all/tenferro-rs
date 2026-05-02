use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend};

use crate::{eager_einsum, eager_einsum_owned};

fn assert_f64_tensor(tensor: &Tensor, shape: &[usize], expected: &[f64]) {
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.as_slice::<f64>(), Some(expected));
}

#[test]
fn eager_einsum_executes_binary_and_ternary_contractions() {
    let mut ctx = CpuBackend::new();
    fn needs_backend(_ctx: &mut impl TensorBackend) {}
    needs_backend(&mut ctx);

    let a = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let matmul = eager_einsum(&mut ctx, &[&a, &b], "ij,jk->ik").unwrap();
    assert_eq!(matmul.shape(), &[2, 2]);
    assert_eq!(
        matmul.as_slice::<f64>(),
        Some([22.0, 28.0, 49.0, 64.0].as_slice())
    );

    let c = Tensor::from_vec(vec![2, 1], vec![1.0_f64, 2.0]);
    let chain = eager_einsum(&mut ctx, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
    assert_eq!(chain.shape(), &[2, 1]);
    assert_eq!(chain.as_slice::<f64>(), Some([120.0, 156.0].as_slice()));
}

#[test]
fn eager_einsum_handles_outer_products_and_diagonal_patterns() {
    let mut ctx = CpuBackend::new();

    let lhs = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let rhs = Tensor::from_vec(vec![3], vec![3.0_f64, 4.0, 5.0]);
    let outer = eager_einsum(&mut ctx, &[&lhs, &rhs], "i,j->ij").unwrap();
    assert_eq!(outer.shape(), &[2, 3]);
    assert_eq!(
        outer.as_slice::<f64>(),
        Some([3.0, 6.0, 4.0, 8.0, 5.0, 10.0].as_slice())
    );

    let matrix = Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
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
fn eager_einsum_handles_higher_rank_repeated_labels() {
    let mut ctx = CpuBackend::new();
    let tensor = Tensor::from_vec(
        vec![2, 2, 3],
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );

    let diagonal = eager_einsum(&mut ctx, &[&tensor], "iij->ij").unwrap();

    assert_eq!(diagonal.shape(), &[2, 3]);
    assert_eq!(
        diagonal.as_slice::<f64>(),
        Some([1.0, 4.0, 5.0, 8.0, 9.0, 12.0].as_slice())
    );
}

#[test]
fn eager_einsum_rejects_empty_inputs_and_operand_count_mismatch() {
    let mut ctx = CpuBackend::new();
    let tensor = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);

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

#[test]
fn eager_einsum_owned_matches_borrowed_for_representative_cases() {
    let cases: Vec<(&str, Vec<Tensor>)> = vec![
        (
            "ii->i",
            vec![Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])],
        ),
        (
            "ij,jk->ik",
            vec![
                Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ],
        ),
        (
            "ij,jk,kl->il",
            vec![
                Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
                Tensor::from_vec(vec![2, 1], vec![1.0_f64, 2.0]),
            ],
        ),
        (
            "bik,bkj->bij",
            vec![
                Tensor::from_vec(
                    vec![2, 2, 2],
                    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                ),
                Tensor::from_vec(
                    vec![2, 2, 2],
                    vec![2.0_f64, 1.0, 0.5, 3.0, 1.5, 2.5, 4.0, 0.25],
                ),
            ],
        ),
        (
            "bi,bj,bk->bijk",
            vec![
                Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
                Tensor::from_vec(vec![2, 3], vec![2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0]),
                Tensor::from_vec(vec![2, 2], vec![1.5_f64, 0.5, 2.5, 1.0]),
            ],
        ),
        (
            "bi,bj,bk->ijk",
            vec![
                Tensor::from_vec(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]),
                Tensor::from_vec(vec![2, 3], vec![2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0]),
                Tensor::from_vec(vec![2, 2], vec![1.5_f64, 0.5, 2.5, 1.0]),
            ],
        ),
    ];

    for (subscripts, inputs) in cases {
        let mut borrowed_ctx = CpuBackend::new();
        let borrowed_refs: Vec<&Tensor> = inputs.iter().collect();
        let borrowed = eager_einsum(&mut borrowed_ctx, &borrowed_refs, subscripts).unwrap();

        let mut owned_ctx = CpuBackend::new();
        let owned = eager_einsum_owned(&mut owned_ctx, inputs, subscripts).unwrap();

        assert_eq!(
            owned.shape(),
            borrowed.shape(),
            "shape mismatch for {subscripts}"
        );
        assert_eq!(
            owned.as_slice::<f64>(),
            borrowed.as_slice::<f64>(),
            "values mismatch for {subscripts}"
        );
    }
}

#[test]
fn eager_einsum_owned_reclaims_consumed_input_buffers() {
    let mut ctx = CpuBackend::new();
    let a = Tensor::from_vec(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = Tensor::from_vec(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let result = eager_einsum_owned(&mut ctx, vec![a, b], "ij,jk->ik").unwrap();

    assert_f64_tensor(&result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
    assert!(
        ctx.buffer_pool_len() >= 2,
        "owned input buffers should be reclaimed after their last use"
    );
}
