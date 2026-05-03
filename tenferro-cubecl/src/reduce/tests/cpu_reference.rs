#[cfg(feature = "cpu-reference")]
use crate::reduce::cpu_reference::reduce_sum_i64_keepdims;

#[test]
#[cfg(feature = "cpu-reference")]
fn cpu_reference_uses_column_major_axis_order() {
    let input = vec![1, 2, 3, 4, 5, 6];

    assert_eq!(reduce_sum_i64_keepdims(&input, &[2, 3], 0), vec![3, 7, 11]);
    assert_eq!(reduce_sum_i64_keepdims(&input, &[2, 3], 1), vec![9, 12]);
}
