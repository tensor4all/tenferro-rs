use std::sync::Arc;

use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};

const FD_H: f64 = 1.0e-6;
const TOL: f64 = 1.0e-10;
const FD_TOL: f64 = 5.0e-4;

fn test_ctx() -> Arc<EagerRuntime> {
    EagerRuntime::with_cpu_backend(CpuBackend::new())
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "idx {idx}: expected {expected}, got {actual}"
        );
    }
}

fn fixed_pivot_cross_loss(data: &[f64]) -> f64 {
    let a = |row: usize, col: usize| data[row + 3 * col];

    let p00 = a(0, 0);
    let p01 = a(0, 2);
    let p10 = a(2, 0);
    let p11 = a(2, 2);
    let det = p00 * p11 - p01 * p10;

    let mut loss = 0.0;
    for col in 0..3 {
        let r0 = a(0, col);
        let r1 = a(2, col);
        let x0 = (p11 * r0 - p01 * r1) / det;
        let x1 = (-p10 * r0 + p00 * r1) / det;

        for row in 0..3 {
            loss += a(row, 0) * x0 + a(row, 2) * x1;
        }
    }
    loss
}

fn finite_diff_grad(data: &[f64]) -> Vec<f64> {
    (0..data.len())
        .map(|idx| {
            let mut plus = data.to_vec();
            let mut minus = data.to_vec();
            plus[idx] += FD_H;
            minus[idx] -= FD_H;
            (fixed_pivot_cross_loss(&plus) - fixed_pivot_cross_loss(&minus)) / (2.0 * FD_H)
        })
        .collect()
}

#[test]
fn take_axis_rows_cols_and_block_select_static_indices() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 4],
            vec![
                1.0_f64, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0,
            ],
        ),
        test_ctx(),
    );

    let rows = x.take_rows(&[2, 0]).unwrap();
    assert_eq!(rows.data().shape(), &[2, 4]);
    assert_close_slice(
        f64_data(rows.data()),
        &[3.0, 1.0, 6.0, 4.0, 9.0, 7.0, 12.0, 10.0],
        TOL,
    );

    let cols = x.take_cols(&[3, 1]).unwrap();
    assert_eq!(cols.data().shape(), &[3, 2]);
    assert_close_slice(
        f64_data(cols.data()),
        &[10.0, 11.0, 12.0, 4.0, 5.0, 6.0],
        TOL,
    );

    let block = x.take_block(&[2, 0], &[3, 1]).unwrap();
    assert_eq!(block.data().shape(), &[2, 2]);
    assert_close_slice(f64_data(block.data()), &[12.0, 10.0, 6.0, 4.0], TOL);

    let axis = x.take_axis(1, &[0, 2]).unwrap();
    assert_eq!(axis.data().shape(), &[3, 2]);
    assert_close_slice(f64_data(axis.data()), &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0], TOL);
}

#[test]
fn take_block_backward_accumulates_to_source() {
    let ctx = test_ctx();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(
            vec![3, 4],
            vec![
                1.0_f64, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0,
            ],
        ),
        ctx.clone(),
    );
    let weights = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ctx,
    );

    let block = x.take_block(&[2, 0, 2], &[3, 1]).unwrap();
    let loss = block.mul(&weights).unwrap().reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(
        f64_data(x.grad().unwrap().as_ref()),
        &[0.0, 0.0, 0.0, 5.0, 0.0, 10.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0],
        TOL,
    );
}

#[test]
fn fixed_pivot_cross_matches_solve_formula_and_gradients() {
    // Fixed rows/columns, and the rank implied by their length, are
    // primal-only metadata. This test differentiates only the construction
    // C * P^{-1} * R for those fixed pivots.
    let a_data = vec![
        2.0_f64, 3.0, 1.0, //
        1.0, 4.0, 0.0, //
        0.0, 5.0, 3.0,
    ];
    let ctx = test_ctx();
    let a =
        EagerTensor::requires_grad_in(Tensor::from_vec_col_major(vec![3, 3], a_data.clone()), ctx);

    let c = a.take_cols(&[0, 2]).unwrap();
    let r = a.take_rows(&[0, 2]).unwrap();
    let p = a.take_block(&[0, 2], &[0, 2]).unwrap();

    let right = p.solve(&r).unwrap();
    let approx = c.matmul(&right).unwrap();
    assert_eq!(approx.data().shape(), &[3, 3]);
    assert_close_slice(
        f64_data(approx.data()),
        &[2.0, 3.0, 1.0, 1.0, 2.0 / 3.0, 0.0, 0.0, 5.0, 3.0],
        TOL,
    );

    let left = p.right_solve(&c).unwrap();
    let approx_from_left = left.matmul(&r).unwrap();
    assert_close_slice(
        f64_data(approx_from_left.data()),
        f64_data(approx.data()),
        TOL,
    );

    let loss = approx.reduce_sum(&[0, 1]).unwrap();
    let _ = loss.backward().unwrap();
    let expected = finite_diff_grad(&a_data);
    assert_close_slice(f64_data(a.grad().unwrap().as_ref()), &expected, FD_TOL);
}
