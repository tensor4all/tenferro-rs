use super::*;

#[test]
fn dynamic_strided_tensor_view_covers_all_dtype_variants() {
    macro_rules! assert_read_variant {
        ($ctor:ident, $ty:ty, $dtype:expr, [$a:expr, $b:expr, $c:expr, $d:expr]) => {{
            let data: [$ty; 4] = [$a, $b, $c, $d];
            let view = StridedTensorView::$ctor(&[2, 2], &[2, 1], 0, &data).unwrap();
            assert_eq!(view.dtype(), $dtype);
            assert_eq!(view.shape(), &[2, 2]);
            assert_eq!(view.strides(), &[2, 1]);
            assert_eq!(view.offset(), 0);
            assert_eq!(
                view.to_tensor().unwrap().as_slice::<$ty>().unwrap(),
                &[$a, $c, $b, $d]
            );
        }};
    }

    assert_read_variant!(f32, f32, DType::F32, [1.0_f32, 2.0, 3.0, 4.0]);
    assert_read_variant!(f64, f64, DType::F64, [1.0_f64, 2.0, 3.0, 4.0]);
    assert_read_variant!(i32, i32, DType::I32, [1_i32, 2, 3, 4]);
    assert_read_variant!(i64, i64, DType::I64, [1_i64, 2, 3, 4]);
    assert_read_variant!(bool, bool, DType::Bool, [false, true, true, false]);
    assert_read_variant!(
        c32,
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
        c64,
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
fn dynamic_strided_tensor_view_mut_covers_all_dtype_variants() {
    macro_rules! assert_mut_variant {
        (
            $ctor:ident,
            $variant:ident,
            $ty:ty,
            $dtype:expr,
            [$a:expr, $b:expr, $c:expr, $d:expr],
            $replacement:expr
        ) => {{
            let mut data: [$ty; 4] = [$a, $b, $c, $d];
            let mut view = StridedTensorViewMut::$ctor(&[2, 2], &[2, 1], 0, &mut data).unwrap();
            assert_eq!(view.dtype(), $dtype);
            assert_eq!(view.shape(), &[2, 2]);
            assert_eq!(view.strides(), &[2, 1]);
            assert_eq!(view.offset(), 0);
            match &mut view {
                StridedTensorViewMut::$variant(typed) => {
                    *typed.try_get_mut(&[1, 1]).unwrap() = $replacement;
                }
                _ => unreachable!(),
            }
            assert_eq!(
                view.as_read_only()
                    .to_tensor()
                    .unwrap()
                    .as_slice::<$ty>()
                    .unwrap(),
                &[$a, $c, $b, $replacement]
            );
            assert_eq!(
                view.to_tensor().unwrap().as_slice::<$ty>().unwrap(),
                &[$a, $c, $b, $replacement]
            );
        }};
    }

    assert_mut_variant!(f32, F32, f32, DType::F32, [1.0_f32, 2.0, 3.0, 4.0], 40.0);
    assert_mut_variant!(f64, F64, f64, DType::F64, [1.0_f64, 2.0, 3.0, 4.0], 40.0);
    assert_mut_variant!(i32, I32, i32, DType::I32, [1_i32, 2, 3, 4], 40);
    assert_mut_variant!(i64, I64, i64, DType::I64, [1_i64, 2, 3, 4], 40);
    assert_mut_variant!(
        bool,
        Bool,
        bool,
        DType::Bool,
        [false, true, true, false],
        true
    );
    assert_mut_variant!(
        c32,
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
        c64,
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
fn dynamic_strided_tensor_view_mut_multi_slice_covers_all_dtype_variants() {
    macro_rules! assert_dynamic_multi_slice {
        (
            $ctor:ident,
            $variant:ident,
            $ty:ty,
            [$a:expr, $b:expr, $c:expr, $d:expr],
            $left:expr,
            $right:expr
        ) => {{
            let mut data: [$ty; 4] = [$a, $b, $c, $d];
            let mut view = StridedTensorViewMut::$ctor(&[4], &[1], 0, &mut data).unwrap();
            {
                let (mut lhs, mut rhs) = view
                    .try_multi_slice_mut(
                        &[StridedSliceSpec::new(0, Some(2), 1)],
                        &[StridedSliceSpec::new(2, Some(4), 1)],
                    )
                    .unwrap();
                match &mut lhs {
                    StridedTensorViewMut::$variant(typed) => {
                        *typed.get_mut(&[1]).unwrap() = $left;
                    }
                    _ => unreachable!(),
                }
                match &mut rhs {
                    StridedTensorViewMut::$variant(typed) => {
                        *typed.get_mut(&[0]).unwrap() = $right;
                    }
                    _ => unreachable!(),
                }
            }
            assert_eq!(
                view.to_tensor().unwrap().as_slice::<$ty>().unwrap(),
                &[$a, $left, $right, $d]
            );
        }};
    }

    assert_dynamic_multi_slice!(f32, F32, f32, [1.0_f32, 2.0, 3.0, 4.0], 20.0, 30.0);
    assert_dynamic_multi_slice!(f64, F64, f64, [1.0_f64, 2.0, 3.0, 4.0], 20.0, 30.0);
    assert_dynamic_multi_slice!(i32, I32, i32, [1_i32, 2, 3, 4], 20, 30);
    assert_dynamic_multi_slice!(i64, I64, i64, [1_i64, 2, 3, 4], 20, 30);
    assert_dynamic_multi_slice!(bool, Bool, bool, [false, false, false, true], true, true);
    assert_dynamic_multi_slice!(
        c32,
        C32,
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
        c64,
        C64,
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
