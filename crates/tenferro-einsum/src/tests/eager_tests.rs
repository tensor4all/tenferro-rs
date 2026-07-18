use tenferro_cpu::CpuBackend;
use tenferro_tensor::{Tensor, TensorBackend, TensorRead, TensorView};

use crate::eager::{eager_einsum, eager_einsum_owned, eager_einsum_read_subscripts};
use crate::Subscripts;

fn assert_f64_tensor(tensor: &Tensor, shape: &[usize], expected: &[f64]) {
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), expected);
}

#[test]
fn eager_einsum_executes_binary_and_ternary_contractions() {
    let mut ctx = CpuBackend::new();
    fn needs_backend(_ctx: &mut impl TensorBackend) {}
    needs_backend(&mut ctx);

    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let matmul = eager_einsum(&mut ctx, &[&a, &b], "ij,jk->ik").unwrap();
    assert_eq!(matmul.shape(), &[2, 2]);
    assert_eq!(
        matmul.as_slice::<f64>().unwrap(),
        [22.0, 28.0, 49.0, 64.0].as_slice()
    );

    let c = Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap();
    let chain = eager_einsum(&mut ctx, &[&a, &b, &c], "ij,jk,kl->il").unwrap();
    assert_eq!(chain.shape(), &[2, 1]);
    assert_eq!(chain.as_slice::<f64>().unwrap(), [120.0, 156.0].as_slice());
}

#[test]
fn eager_einsum_handles_outer_products_and_diagonal_patterns() {
    let mut ctx = CpuBackend::new();

    let lhs = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]).unwrap();
    let outer = eager_einsum(&mut ctx, &[&lhs, &rhs], "i,j->ij").unwrap();
    assert_eq!(outer.shape(), &[2, 3]);
    assert_eq!(
        outer.as_slice::<f64>().unwrap(),
        [3.0, 6.0, 4.0, 8.0, 5.0, 10.0].as_slice()
    );

    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let diagonal = eager_einsum(&mut ctx, &[&matrix], "ii->i").unwrap();
    let trace = eager_einsum(&mut ctx, &[&matrix], "ii->").unwrap();
    assert_eq!(diagonal.shape(), &[2]);
    assert_eq!(diagonal.as_slice::<f64>().unwrap(), [1.0, 4.0].as_slice());
    assert_eq!(trace.shape(), &[] as &[usize]);
    assert_eq!(trace.as_slice::<f64>().unwrap(), [5.0].as_slice());

    let embedded = eager_einsum(&mut ctx, &[&lhs], "i->ii").unwrap();
    assert_eq!(embedded.shape(), &[2, 2]);
    assert_eq!(
        embedded.as_slice::<f64>().unwrap(),
        [1.0, 0.0, 0.0, 2.0].as_slice()
    );
}

#[test]
fn eager_einsum_handles_higher_rank_repeated_labels() {
    let mut ctx = CpuBackend::new();
    let tensor = Tensor::from_vec_col_major(
        vec![2, 2, 3],
        vec![
            1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    let diagonal = eager_einsum(&mut ctx, &[&tensor], "iij->ij").unwrap();

    assert_eq!(diagonal.shape(), &[2, 3]);
    assert_eq!(
        diagonal.as_slice::<f64>().unwrap(),
        [1.0, 4.0, 5.0, 8.0, 9.0, 12.0].as_slice()
    );
}

#[test]
fn eager_einsum_handles_three_or_more_repeated_labels() {
    let mut ctx = CpuBackend::new();
    let cube = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();

    let diagonal = eager_einsum(&mut ctx, &[&cube], "iii->i").unwrap();
    assert_f64_tensor(&diagonal, &[2], &[1.0, 8.0]);

    let hypercube = Tensor::from_vec_col_major(
        vec![2, 2, 2, 2],
        (1..=16).map(|value| value as f64).collect(),
    )
    .unwrap();
    let trace = eager_einsum(&mut ctx, &[&hypercube], "iiii->").unwrap();
    assert_f64_tensor(&trace, &[], &[17.0]);

    let rhs = Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 3.0, 5.0]).unwrap();
    let mixed = eager_einsum(&mut ctx, &[&cube, &rhs], "iii,j->ij").unwrap();
    assert_f64_tensor(&mixed, &[2, 3], &[2.0, 16.0, 3.0, 24.0, 5.0, 40.0]);
}

#[test]
fn eager_einsum_read_views_match_owned_inputs() {
    let a_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_shape = [2, 3];
    let b_shape = [3, 2];
    let a_owned = Tensor::from_vec_col_major(a_shape.to_vec(), a_data.to_vec()).unwrap();
    let b_owned = Tensor::from_vec_col_major(b_shape.to_vec(), b_data.to_vec()).unwrap();

    let mut owned_ctx = CpuBackend::new();
    let owned = eager_einsum(&mut owned_ctx, &[&a_owned, &b_owned], "ij,jk->ik").unwrap();

    let inputs = [
        TensorRead::from_view(TensorView::f64(&a_shape, &a_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&b_shape, &b_data).unwrap()),
    ];
    let subscripts = Subscripts::parse("ij,jk->ik").unwrap();
    let mut read_ctx = CpuBackend::new();
    let read = eager_einsum_read_subscripts(&mut read_ctx, &inputs, &subscripts).unwrap();

    assert_eq!(read.shape(), owned.shape());
    assert_eq!(
        read.as_slice::<f64>().unwrap(),
        owned.as_slice::<f64>().unwrap()
    );
}

#[test]
fn eager_einsum_binary_contract_reorders_fast_path_output() {
    let mut ctx = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let result = eager_einsum(&mut ctx, &[&a, &b], "ij,jk->ki").unwrap();

    assert_f64_tensor(&result, &[2, 2], &[22.0, 49.0, 28.0, 64.0]);
}

#[test]
fn eager_einsum_read_fast_path_handles_batched_contracts() {
    let lhs_data = [
        1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let rhs_data = [
        0.5_f64, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
    ];
    let lhs_shape = [2, 2, 3];
    let rhs_shape = [2, 3, 2];
    let mut expected = vec![0.0; 2 * 2 * 2];
    for b in 0..2 {
        for i in 0..2 {
            for k in 0..2 {
                let mut value = 0.0;
                for j in 0..3 {
                    value += lhs_data[b + 2 * (i + 2 * j)] * rhs_data[b + 2 * (j + 3 * k)];
                }
                expected[b + 2 * (i + 2 * k)] = value;
            }
        }
    }

    let inputs = [
        TensorRead::from_view(TensorView::f64(&lhs_shape, &lhs_data).unwrap()),
        TensorRead::from_view(TensorView::f64(&rhs_shape, &rhs_data).unwrap()),
    ];
    let subscripts = Subscripts::parse("bij,bjk->bik").unwrap();
    let mut read_ctx = CpuBackend::new();
    let read = eager_einsum_read_subscripts(&mut read_ctx, &inputs, &subscripts).unwrap();

    assert_eq!(read.shape(), &[2, 2, 2]);
    assert_eq!(read.as_slice::<f64>().unwrap(), expected.as_slice());
}

#[test]
fn eager_einsum_rejects_empty_inputs_and_operand_count_mismatch() {
    let mut ctx = CpuBackend::new();
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let empty = eager_einsum(&mut ctx, &[], "->").unwrap_err();
    assert!(matches!(
        empty,
        tenferro_tensor::Error::Validation {
            op: "eager_einsum",
            ..
        }
    ));

    let mismatch = eager_einsum(&mut ctx, &[&tensor], "i,j->ij").unwrap_err();
    assert!(matches!(
        mismatch,
        tenferro_tensor::Error::Validation {
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
            vec![Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap()],
        ),
        (
            "ij,jk->ik",
            vec![
                Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .unwrap(),
                Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .unwrap(),
            ],
        ),
        (
            "ij,jk,kl->il",
            vec![
                Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .unwrap(),
                Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .unwrap(),
                Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap(),
            ],
        ),
        (
            "bik,bkj->bij",
            vec![
                Tensor::from_vec_col_major(
                    vec![2, 2, 2],
                    vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                )
                .unwrap(),
                Tensor::from_vec_col_major(
                    vec![2, 2, 2],
                    vec![2.0_f64, 1.0, 0.5, 3.0, 1.5, 2.5, 4.0, 0.25],
                )
                .unwrap(),
            ],
        ),
        (
            "bi,bj,bk->bijk",
            vec![
                Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
                Tensor::from_vec_col_major(vec![2, 3], vec![2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0])
                    .unwrap(),
                Tensor::from_vec_col_major(vec![2, 2], vec![1.5_f64, 0.5, 2.5, 1.0]).unwrap(),
            ],
        ),
        (
            "bi,bj,bk->ijk",
            vec![
                Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
                Tensor::from_vec_col_major(vec![2, 3], vec![2.0_f64, 3.0, 4.0, 5.0, 6.0, 7.0])
                    .unwrap(),
                Tensor::from_vec_col_major(vec![2, 2], vec![1.5_f64, 0.5, 2.5, 1.0]).unwrap(),
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
            owned.as_slice::<f64>().unwrap(),
            borrowed.as_slice::<f64>().unwrap(),
            "values mismatch for {subscripts}"
        );
    }
}

#[test]
fn eager_einsum_owned_reclaims_consumed_input_buffers() {
    let mut ctx = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let result = eager_einsum_owned(&mut ctx, vec![a, b], "ij,jk->ik").unwrap();

    assert_f64_tensor(&result, &[2, 2], &[22.0, 28.0, 49.0, 64.0]);
    assert!(
        ctx.buffer_pool_len().unwrap() >= 2,
        "owned input buffers should be reclaimed after their last use"
    );
}
