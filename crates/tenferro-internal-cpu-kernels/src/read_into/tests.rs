use super::*;

fn run(
    op: ElementwiseReadOp,
    inputs: &[TensorRead<'_>],
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    elementwise_read_into_with_context(op, inputs, out, &ExecContext::serial(), |_, _| {
        panic!("eligible host operation unexpectedly used fallback")
    })
}

#[test]
fn unary_conjugation_covers_all_owned_dtypes() {
    macro_rules! check {
        ($variant:ident, $ty:ty, $value:expr) => {{
            let input = Tensor::$variant(
                tenferro_tensor::TypedTensor::<$ty>::from_vec_col_major(vec![1], vec![$value])
                    .unwrap(),
            );
            let mut output = Tensor::$variant(
                tenferro_tensor::TypedTensor::<$ty>::from_vec_col_major(vec![1], vec![$value])
                    .unwrap(),
            );
            run(
                ElementwiseReadOp::Conj,
                &[TensorRead::from_tensor(&input)],
                TensorWrite::from_tensor(&mut output),
            )
            .unwrap();
        }};
    }

    check!(F32, f32, 1.0);
    check!(F64, f64, 2.0);
    check!(I32, i32, 3);
    check!(I64, i64, 4);
    check!(Bool, bool, true);
    check!(
        C32,
        num_complex::Complex32,
        num_complex::Complex32::new(5.0, 1.0)
    );
    check!(
        C64,
        num_complex::Complex64,
        num_complex::Complex64::new(6.0, 2.0)
    );
}

#[test]
fn unary_and_binary_replay_cover_view_and_dtype_dispatch() {
    macro_rules! check_view {
        ($variant:ident, $ty:ty, $value:expr) => {{
            let data = [$value];
            let view =
                tenferro_tensor::TypedTensorView::from_slice(vec![1], vec![1], 0, &data).unwrap();
            let input = TensorRead::from_view(TensorView::$variant(view));
            let mut output = Tensor::$variant(
                tenferro_tensor::TypedTensor::<$ty>::from_vec_col_major(vec![1], vec![$value])
                    .unwrap(),
            );
            run(
                ElementwiseReadOp::Conj,
                &[input],
                TensorWrite::from_tensor(&mut output),
            )
            .unwrap();
        }};
    }

    check_view!(F32, f32, 1.0);
    check_view!(F64, f64, 2.0);
    check_view!(I32, i32, 3);
    check_view!(I64, i64, 4);
    check_view!(Bool, bool, true);
    check_view!(
        C32,
        num_complex::Complex32,
        num_complex::Complex32::new(5.0, 1.0)
    );
    check_view!(
        C64,
        num_complex::Complex64,
        num_complex::Complex64::new(6.0, 2.0)
    );

    macro_rules! check_zip {
        ($variant:ident, $ty:ty, $lhs:expr, $rhs:expr) => {{
            let lhs = Tensor::$variant(
                tenferro_tensor::TypedTensor::<$ty>::from_vec_col_major(vec![1], vec![$lhs])
                    .unwrap(),
            );
            let rhs = Tensor::$variant(
                tenferro_tensor::TypedTensor::<$ty>::from_vec_col_major(vec![1], vec![$rhs])
                    .unwrap(),
            );
            let mut output = Tensor::$variant(
                tenferro_tensor::TypedTensor::<$ty>::from_vec_col_major(vec![1], vec![$lhs])
                    .unwrap(),
            );
            for op in [
                ElementwiseReadOp::Add,
                ElementwiseReadOp::Subtract,
                ElementwiseReadOp::Multiply,
                ElementwiseReadOp::Divide,
            ] {
                run(
                    op,
                    &[TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)],
                    TensorWrite::from_tensor(&mut output),
                )
                .unwrap();
            }
        }};
    }

    check_zip!(F32, f32, 6.0, 2.0);
    check_zip!(F64, f64, 6.0, 2.0);
    check_zip!(I32, i32, 6, 2);
    check_zip!(I64, i64, 6, 2);
    check_zip!(
        C32,
        num_complex::Complex32,
        num_complex::Complex32::new(6.0, 1.0),
        num_complex::Complex32::new(2.0, 1.0)
    );
    check_zip!(
        C64,
        num_complex::Complex64,
        num_complex::Complex64::new(6.0, 1.0),
        num_complex::Complex64::new(2.0, 1.0)
    );
}

#[test]
fn replay_supports_mutable_views_and_preserves_fallback_errors() {
    let input_data = [2.0_f32, 3.0];
    let input_view =
        tenferro_tensor::TypedTensorView::from_slice(vec![2], vec![1], 0, &input_data).unwrap();
    let mut output_data = [0.0_f32, 0.0];
    let output_view =
        tenferro_tensor::TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut output_data)
            .unwrap();
    run(
        ElementwiseReadOp::Negate,
        &[TensorRead::from_view(TensorView::F32(input_view))],
        TensorWrite::from_view(TensorViewMut::F32(output_view)),
    )
    .unwrap();
    assert_eq!(output_data, [-2.0, -3.0]);

    let input =
        Tensor::F32(tenferro_tensor::TypedTensor::from_vec_col_major(vec![1], vec![1.0]).unwrap());
    let mut output = Tensor::F32(
        tenferro_tensor::TypedTensor::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap(),
    );
    let mut fallback_called = false;
    elementwise_read_into_with_context(
        ElementwiseReadOp::Conj,
        &[TensorRead::from_tensor(&input)],
        TensorWrite::from_tensor(&mut output),
        &ExecContext::serial(),
        |_, _| {
            fallback_called = true;
            Err(Error::unsupported("test", "fallback"))
        },
    )
    .unwrap_err();
    assert!(fallback_called);

    let mut output =
        Tensor::F64(tenferro_tensor::TypedTensor::from_vec_col_major(vec![1], vec![0.0]).unwrap());
    let error = elementwise_read_into_with_context(
        ElementwiseReadOp::Conj,
        &[],
        TensorWrite::from_tensor(&mut output),
        &ExecContext::serial(),
        |_, _| Ok(()),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Validation { .. }));
}
