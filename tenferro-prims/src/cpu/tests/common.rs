use num_complex::Complex64;
use strided_view::{StridedView, StridedViewMut};
use tenferro_device::Error;

use super::super::common::{
    execute_binary_map, execute_ternary_map, execute_unary_map, is_supported_ordered_real_type,
    is_supported_scalar_type, plan_reduction, unflatten_index_into, validate_pointwise_shapes,
};

fn view<'a, T>(data: &'a [T], dims: &[usize]) -> StridedView<'a, T> {
    let strides: Vec<isize> = dims
        .iter()
        .scan(1isize, |state, &dim| {
            let stride = *state;
            *state *= dim as isize;
            Some(stride)
        })
        .collect();
    StridedView::new(data, dims, &strides, 0).unwrap()
}

fn view_mut<'a, T>(data: &'a mut [T], dims: &[usize]) -> StridedViewMut<'a, T> {
    let strides: Vec<isize> = dims
        .iter()
        .scan(1isize, |state, &dim| {
            let stride = *state;
            *state *= dim as isize;
            Some(stride)
        })
        .collect();
    StridedViewMut::new(data, dims, &strides, 0).unwrap()
}

#[test]
fn validate_pointwise_shapes_checks_arity_and_output_shape() {
    validate_pointwise_shapes(&[&[2, 3], &[2, 3], &[2, 3]], 2, "binary").unwrap();

    let err = validate_pointwise_shapes(&[&[2, 3], &[2, 3]], 2, "binary").unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(message) if message.contains("expects 3 shapes")));

    let err = validate_pointwise_shapes(&[&[2, 3], &[2, 3], &[3, 2]], 2, "binary").unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch { expected, got } if expected == vec![3, 2] && got == vec![2, 3]
    ));
}

#[test]
fn plan_reduction_tracks_reduced_axes_and_detects_invalid_outputs() {
    let plan = plan_reduction(&[0, 1, 2], &[1], &[&[2, 3, 4], &[3]], "sum").unwrap();
    assert_eq!(plan.reduced_axes, vec![0, 2]);
    assert_eq!(plan.reduced_total, 8);

    let err = plan_reduction(&[0, 1, 2], &[5], &[&[2, 3, 4], &[3]], "sum").unwrap_err();
    assert!(err.to_string().contains("output mode 5 not found"));

    let err = plan_reduction(&[0, 1, 2], &[1], &[&[2, 3, 4], &[4]], "sum").unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch { expected, got } if expected == vec![3] && got == vec![4]
    ));
}

#[test]
fn execute_unary_map_covers_fast_path_and_accumulation_path() {
    let input_data = [1.0_f64, 2.0];
    let input = view(&input_data, &[2]);

    let mut output_fast = [10.0_f64, 20.0];
    {
        let mut output = view_mut(&mut output_fast, &[2]);
        execute_unary_map(2.0, &input, 0.0, &mut output, |x| x + 1.0).unwrap();
    }
    assert_eq!(output_fast, [4.0, 6.0]);

    let mut output_accum = [10.0_f64, 20.0];
    {
        let mut output = view_mut(&mut output_accum, &[2]);
        execute_unary_map(2.0, &input, 3.0, &mut output, |x| x + 1.0).unwrap();
    }
    assert_eq!(output_accum, [34.0, 66.0]);
}

#[test]
fn execute_binary_map_covers_fast_path_and_accumulation_path() {
    let lhs_data = [1.0_f64, 2.0];
    let rhs_data = [10.0_f64, 20.0];
    let lhs = view(&lhs_data, &[2]);
    let rhs = view(&rhs_data, &[2]);

    let mut output_fast = [0.0_f64, 0.0];
    {
        let mut output = view_mut(&mut output_fast, &[2]);
        execute_binary_map(1.0, &lhs, &rhs, 0.0, &mut output, |x, y| x + y).unwrap();
    }
    assert_eq!(output_fast, [11.0, 22.0]);

    let mut output_accum = [1.0_f64, 2.0];
    {
        let mut output = view_mut(&mut output_accum, &[2]);
        execute_binary_map(2.0, &lhs, &rhs, 3.0, &mut output, |x, y| x - y).unwrap();
    }
    assert_eq!(output_accum, [-15.0, -30.0]);
}

#[test]
fn execute_ternary_map_covers_fast_path_and_accumulation_path() {
    let cond_data = [1.0_f64, 0.0];
    let on_true_data = [10.0_f64, 20.0];
    let on_false_data = [100.0_f64, 200.0];
    let cond = view(&cond_data, &[2]);
    let on_true = view(&on_true_data, &[2]);
    let on_false = view(&on_false_data, &[2]);

    let mut output_fast = [0.0_f64, 0.0];
    {
        let mut output = view_mut(&mut output_fast, &[2]);
        execute_ternary_map(
            1.0,
            &cond,
            &on_true,
            &on_false,
            0.0,
            &mut output,
            |c, t, f| if c != 0.0 { t } else { f },
        )
        .unwrap();
    }
    assert_eq!(output_fast, [10.0, 200.0]);

    let mut output_accum = [1.0_f64, 2.0];
    {
        let mut output = view_mut(&mut output_accum, &[2]);
        execute_ternary_map(
            2.0,
            &cond,
            &on_true,
            &on_false,
            3.0,
            &mut output,
            |c, t, f| if c != 0.0 { t } else { f },
        )
        .unwrap();
    }
    assert_eq!(output_accum, [23.0, 406.0]);
}

#[test]
fn unflatten_index_and_supported_scalar_helpers_cover_both_paths() {
    let mut out = [usize::MAX; 2];
    unflatten_index_into(4, &[2, 3], &mut out);
    assert_eq!(out, [0, 2]);

    assert!(is_supported_scalar_type::<f64>());
    assert!(is_supported_scalar_type::<Complex64>());
    assert!(!is_supported_scalar_type::<i32>());

    assert!(is_supported_ordered_real_type::<f64>());
    assert!(!is_supported_ordered_real_type::<Complex64>());
    assert!(!is_supported_ordered_real_type::<i32>());
}
