use super::*;

#[test]
fn tensor_view_covers_all_dtype_variants() {
    macro_rules! assert_read_variant {
        ($variant:ident, $ty:ty, $dtype:expr, [$a:expr, $b:expr, $c:expr, $d:expr]) => {{
            let data: [$ty; 4] = [$a, $b, $c, $d];
            let typed = TypedTensorView::from_slice([2, 2], [2, 1], 0, &data).unwrap();
            let view = TensorView::$variant(typed);
            assert_eq!(view.dtype(), $dtype);
            assert_eq!(view.shape(), &[2, 2]);
            assert_eq!(
                view.to_tensor().unwrap().as_slice::<$ty>().unwrap(),
                &[$a, $c, $b, $d]
            );
        }};
    }

    assert_read_variant!(F32, f32, DType::F32, [1.0_f32, 2.0, 3.0, 4.0]);
    assert_read_variant!(F64, f64, DType::F64, [1.0_f64, 2.0, 3.0, 4.0]);
    assert_read_variant!(I32, i32, DType::I32, [1_i32, 2, 3, 4]);
    assert_read_variant!(I64, i64, DType::I64, [1_i64, 2, 3, 4]);
    assert_read_variant!(Bool, bool, DType::Bool, [false, true, true, false]);
    assert_read_variant!(
        C32,
        Complex32,
        DType::C32,
        [
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 2.0),
            Complex32::new(3.0, 3.0),
            Complex32::new(4.0, 4.0)
        ]
    );
    assert_read_variant!(
        C64,
        Complex64,
        DType::C64,
        [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0)
        ]
    );
}

#[test]
fn typed_tensor_view_mut_covers_all_dtype_variants() {
    macro_rules! assert_mut_variant {
        (
            $variant:ident,
            $ty:ty,
            $dtype:expr,
            [$a:expr, $b:expr, $c:expr, $d:expr],
            $replacement:expr
        ) => {{
            let mut data: [$ty; 4] = [$a, $b, $c, $d];
            let mut view = TypedTensorViewMut::from_slice([2, 2], [2, 1], 0, &mut data).unwrap();
            assert_eq!(view.shape(), &[2, 2]);
            assert_eq!(view.strides(), &[2, 1]);
            assert_eq!(view.offset(), 0);
            *view.get_mut(&[1, 1]).unwrap() = $replacement;
            assert_eq!(
                materialize_typed_view_col_major(&view.as_read_only(), "test")
                    .unwrap()
                    .as_slice()
                    .unwrap(),
                &[$a, $c, $b, $replacement]
            );
            let read = TensorView::$variant(view.as_read_only());
            assert_eq!(read.dtype(), $dtype);
        }};
    }

    assert_mut_variant!(F32, f32, DType::F32, [1.0_f32, 2.0, 3.0, 4.0], 40.0);
    assert_mut_variant!(F64, f64, DType::F64, [1.0_f64, 2.0, 3.0, 4.0], 40.0);
    assert_mut_variant!(I32, i32, DType::I32, [1_i32, 2, 3, 4], 40);
    assert_mut_variant!(I64, i64, DType::I64, [1_i64, 2, 3, 4], 40);
    assert_mut_variant!(Bool, bool, DType::Bool, [false, true, true, false], true);
    assert_mut_variant!(
        C32,
        Complex32,
        DType::C32,
        [
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 2.0),
            Complex32::new(3.0, 3.0),
            Complex32::new(4.0, 4.0)
        ],
        Complex32::new(40.0, -1.0)
    );
    assert_mut_variant!(
        C64,
        Complex64,
        DType::C64,
        [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0)
        ],
        Complex64::new(40.0, -1.0)
    );
}

#[test]
fn typed_tensor_view_mut_multi_slice_covers_all_dtype_variants() {
    macro_rules! assert_dynamic_multi_slice {
        (
            $ty:ty,
            [$a:expr, $b:expr, $c:expr, $d:expr],
            $left:expr,
            $right:expr
        ) => {{
            let mut data: [$ty; 4] = [$a, $b, $c, $d];
            let mut view = TypedTensorViewMut::from_slice([4], [1], 0, &mut data).unwrap();
            {
                let (mut lhs, mut rhs) = view
                    .try_multi_slice_mut(
                        &[StridedSliceSpec::new(0, Some(2), 1)],
                        &[StridedSliceSpec::new(2, Some(4), 1)],
                    )
                    .unwrap();
                *lhs.get_mut(&[1]).unwrap() = $left;
                *rhs.get_mut(&[0]).unwrap() = $right;
            }
            assert_eq!(
                materialize_typed_view_col_major(&view.as_read_only(), "test")
                    .unwrap()
                    .as_slice()
                    .unwrap(),
                &[$a, $left, $right, $d]
            );
        }};
    }

    assert_dynamic_multi_slice!(f32, [1.0_f32, 2.0, 3.0, 4.0], 20.0, 30.0);
    assert_dynamic_multi_slice!(f64, [1.0_f64, 2.0, 3.0, 4.0], 20.0, 30.0);
    assert_dynamic_multi_slice!(i32, [1_i32, 2, 3, 4], 20, 30);
    assert_dynamic_multi_slice!(i64, [1_i64, 2, 3, 4], 20, 30);
    assert_dynamic_multi_slice!(bool, [false, false, false, true], true, true);
    assert_dynamic_multi_slice!(
        Complex32,
        [
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, 2.0),
            Complex32::new(3.0, 3.0),
            Complex32::new(4.0, 4.0)
        ],
        Complex32::new(20.0, -1.0),
        Complex32::new(30.0, -1.0)
    );
    assert_dynamic_multi_slice!(
        Complex64,
        [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0)
        ],
        Complex64::new(20.0, -1.0),
        Complex64::new(30.0, -1.0)
    );
}
