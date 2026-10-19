use std::sync::Arc;
use tenferro_ad::{EagerRuntime, EagerTensor};

use tenferro_runtime::Tensor;

const TOL: f64 = 1.0e-10;

fn test_ctx() -> Arc<EagerRuntime> {
    EagerRuntime::new().unwrap()
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
    let ctx = test_ctx();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 4],
            vec![
                1.0_f64, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let rows = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            session.take_rows(&x, &[2, 0])
        })
        .unwrap();
    assert_eq!(rows.shape(), &[2, 4]);
    assert_close_slice(
        f64_data(
            &rows
                .to_tensor(&mut tenferro_cpu::CpuEngine::serial())
                .unwrap(),
        ),
        &[3.0, 1.0, 6.0, 4.0, 9.0, 7.0, 12.0, 10.0],
        TOL,
    );

    let cols = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            session.take_cols(&x, &[3, 1])
        })
        .unwrap();
    assert_eq!(cols.shape(), &[3, 2]);
    assert_close_slice(
        f64_data(
            &cols
                .to_tensor(&mut tenferro_cpu::CpuEngine::serial())
                .unwrap(),
        ),
        &[10.0, 11.0, 12.0, 4.0, 5.0, 6.0],
        TOL,
    );

    let block = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            session.take_block(&x, &[2, 0], &[3, 1])
        })
        .unwrap();
    assert_eq!(block.shape(), &[2, 2]);
    assert_close_slice(
        f64_data(
            &block
                .to_tensor(&mut tenferro_cpu::CpuEngine::serial())
                .unwrap(),
        ),
        &[12.0, 10.0, 6.0, 4.0],
        TOL,
    );

    let axis = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            session.take_axis(&x, 1, &[0, 2])
        })
        .unwrap();
    assert_eq!(axis.shape(), &[3, 2]);
    assert_close_slice(
        f64_data(
            &axis
                .to_tensor(&mut tenferro_cpu::CpuEngine::serial())
                .unwrap(),
        ),
        &[1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        TOL,
    );
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
        )
        .unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let weights = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let block = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |session| {
            session.take_block(&x, &[2, 0, 2], &[3, 1])
        })
        .unwrap();
    let loss = ctx
        .with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
            let weighted = s.mul(&block, &weights)?;
            s.reduce_sum(&weighted, Some(&[0, 1]))
        })
        .unwrap();
    let _ = loss
        .backward(&mut tenferro_cpu::CpuEngine::serial())
        .unwrap();

    assert_close_slice(
        f64_data(
            &x.grad()
                .unwrap()
                .unwrap()
                .to_tensor(&mut tenferro_cpu::CpuEngine::serial())
                .unwrap(),
        ),
        &[0.0, 0.0, 0.0, 5.0, 0.0, 10.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0],
        TOL,
    );
}
