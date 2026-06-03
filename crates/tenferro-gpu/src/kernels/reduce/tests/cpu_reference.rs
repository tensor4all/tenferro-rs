#[cfg(feature = "cpu-reference")]
use crate::kernels::reduce::cpu_reference::reduce_sum_i64_keepdims;

#[test]
#[cfg(feature = "cpu-reference")]
fn cpu_reference_uses_column_major_axis_order() {
    let input = vec![1, 2, 3, 4, 5, 6];

    assert_eq!(reduce_sum_i64_keepdims(&input, &[2, 3], 0), vec![3, 7, 11]);
    assert_eq!(reduce_sum_i64_keepdims(&input, &[2, 3], 1), vec![9, 12]);
}

#[test]
#[cfg(feature = "cpu-reference")]
fn cpu_reference_reduces_rank3_middle_axis_column_major() {
    let input: Vec<i64> = (1..=12).collect();

    assert_eq!(
        reduce_sum_i64_keepdims(&input, &[2, 3, 2], 1),
        vec![9, 12, 27, 30]
    );
}

#[test]
#[cfg(feature = "cpu-reference")]
#[should_panic(expected = "axis 2 is out of bounds for rank 2")]
fn cpu_reference_panics_when_axis_exceeds_rank() {
    let input = vec![1, 2, 3, 4, 5, 6];

    let _ = reduce_sum_i64_keepdims(&input, &[2, 3], 2);
}

#[test]
#[cfg(feature = "cpu-reference")]
#[should_panic(expected = "axis 0 is out of bounds for rank 0")]
fn cpu_reference_panics_when_rank_is_zero() {
    let input = vec![1];

    let _ = reduce_sum_i64_keepdims(&input, &[], 0);
}

#[test]
#[cfg(feature = "cpu-reference")]
#[should_panic(expected = "input length does not match input shape")]
fn cpu_reference_panics_when_input_length_mismatches_shape() {
    let input = vec![1, 2, 3, 4, 5];

    let _ = reduce_sum_i64_keepdims(&input, &[2, 3], 0);
}
