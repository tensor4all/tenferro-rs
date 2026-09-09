use super::*;
use tenferro_tensor::StridedSliceSpec;

#[test]
fn static_replay_preserves_scalar_side_and_negative_strides() {
    let mut pool = BufferPool::new();
    let scalar = Tensor::from_vec_col_major([], vec![12.0_f64]).unwrap();
    let input = TypedTensor::<f64>::from_vec_col_major([3usize], vec![2.0, 3.0, 4.0]).unwrap();
    let reverse = input
        .as_view()
        .try_slice(&[StridedSliceSpec::new(0, Some(3), -1)])
        .unwrap();
    for (left_scalar, expected_sub, expected_div) in [
        (true, vec![8.0, 9.0, 10.0], vec![3.0, 4.0, 6.0]),
        (
            false,
            vec![-8.0, -9.0, -10.0],
            vec![4.0 / 12.0, 0.25, 2.0 / 12.0],
        ),
    ] {
        let operands = || {
            let scalar = TensorRead::from_tensor(&scalar);
            let view = TensorRead::from_view(TensorView::F64(reverse.clone()));
            if left_scalar {
                (scalar, view)
            } else {
                (view, scalar)
            }
        };
        let (lhs, rhs) = operands();
        let sub = sub_read_with_pool(&mut pool, lhs, rhs).unwrap();
        assert_eq!(sub.as_slice::<f64>().unwrap(), expected_sub);
        let (lhs, rhs) = operands();
        let div = div_read_with_pool(&mut pool, lhs, rhs).unwrap();
        assert_eq!(div.as_slice::<f64>().unwrap(), expected_div);
    }
}

#[test]
fn static_replay_keeps_complex_abs_real_and_conjugation_exact() {
    let mut pool = BufferPool::new();
    let input = TypedTensor::<_>::from_vec_col_major(
        [2usize],
        vec![Complex::new(3.0_f64, 4.0), Complex::new(-5.0, 12.0)],
    )
    .unwrap();
    let read = || TensorRead::from_view(TensorView::C64(input.as_view()));
    let abs = abs_read_with_pool(&mut pool, read()).unwrap();
    assert_eq!(abs.dtype(), DType::F64);
    assert_eq!(abs.as_slice::<f64>().unwrap(), &[5.0, 13.0]);
    let conj = conj_read_with_pool(&mut pool, read()).unwrap();
    assert_eq!(
        conj.as_slice::<Complex<f64>>().unwrap(),
        &[Complex::new(3.0, -4.0), Complex::new(-5.0, -12.0)]
    );
}

#[test]
fn static_replay_preserves_wrapping_neg_and_abs() {
    let mut pool = BufferPool::new();
    macro_rules! check {
        ($ty:ty, $variant:ident) => {{
            let input =
                TypedTensor::<$ty>::from_vec_col_major([2usize], vec![<$ty>::MIN, -7]).unwrap();
            let read = || TensorRead::from_view(TensorView::$variant(input.as_view()));
            let neg = neg_read_with_pool(&mut pool, read()).unwrap();
            let abs = abs_read_with_pool(&mut pool, read()).unwrap();
            assert_eq!(neg.as_slice::<$ty>().unwrap(), &[<$ty>::MIN, 7]);
            assert_eq!(abs.as_slice::<$ty>().unwrap(), &[<$ty>::MIN, 7]);
        }};
    }
    check!(i32, I32);
    check!(i64, I64);
}

#[test]
fn static_ternary_replay_preserves_selection_and_nan_policy() {
    let mut pool = BufferPool::new();
    let input =
        TypedTensor::<_>::from_vec_col_major([3usize], vec![-2.0_f64, f64::NAN, 3.0]).unwrap();
    let lower = TypedTensor::<_>::from_vec_col_major([3usize], vec![-1.0_f64; 3]).unwrap();
    let upper = TypedTensor::<_>::from_vec_col_major([3usize], vec![1.0_f64; 3]).unwrap();
    let pred = TypedTensor::<_>::from_vec_col_major([3usize], vec![true, false, true]).unwrap();
    let clamp = typed_clamp_view_with_pool(
        &mut pool,
        &input.as_view(),
        &lower.as_view(),
        &upper.as_view(),
    )
    .unwrap();
    let values = clamp.host_data().unwrap();
    assert_eq!(values[0], -1.0);
    assert!(values[1].is_nan());
    assert_eq!(values[2usize], 1.0);
    let select = typed_select_view_with_pool(
        &mut pool,
        &pred.as_view(),
        &input.as_view(),
        &upper.as_view(),
    )
    .unwrap();
    assert_eq!(select.host_data().unwrap(), &[-2.0, 1.0, 3.0]);
    let wrong = TypedTensor::<_>::from_vec_col_major([1usize], vec![false]).unwrap();
    assert!(typed_select_view_with_pool(
        &mut pool,
        &wrong.as_view(),
        &input.as_view(),
        &upper.as_view()
    )
    .is_err());
}
