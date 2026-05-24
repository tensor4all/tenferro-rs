#![cfg(feature = "autodiff")]

use std::sync::Arc;

use tenferro::{CpuBackend, EagerRuntime, EagerTensor, Tensor};

const TOL: f64 = 1.0e-10;

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
