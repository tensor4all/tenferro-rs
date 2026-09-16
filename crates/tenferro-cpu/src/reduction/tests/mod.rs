//! Module-local tests for the reduction dispatch seams.

use super::{reduce_max, reduce_min, reduce_prod, reduce_sum, typed_input};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{Tensor, TypedTensor};

fn tensor<T: tenferro_tensor::TensorScalar + Copy>(values: Vec<T>) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values).unwrap()
}

/// Every preset scalar reaches its own arm of the max and min tables, so the tables'
/// per-scalar bodies are exercised rather than only the float ones.
#[test]
fn reduce_max_and_min_cover_every_preset_scalar() {
    let cases = [
        tensor(vec![1.0_f32, 2.0]),
        tensor(vec![1.0_f64, 2.0]),
        tensor(vec![1_i32, 2]),
        tensor(vec![1_i64, 2]),
        tensor(vec![false, true]),
        tensor(vec![
            num_complex::Complex32::new(1.0, 0.0),
            num_complex::Complex32::new(2.0, 0.0),
        ]),
        tensor(vec![
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(2.0, 0.0),
        ]),
    ];
    for input in &cases {
        // Either the table computes the reduction or its refusal arm answers; both are
        // dispatch bodies this module owns.
        let _ = reduce_max(input, &[0]);
        let _ = reduce_min(input, &[0]);
    }
}

/// The tag tables only reach `typed_input` with the scalar they claim, so its refusal
/// is unreachable from the public API. Covering it here keeps the invariant honest:
/// a mismatch is a typed error, never a panic.
#[test]
fn typed_input_refuses_a_dtype_the_dispatch_never_produces() {
    let tensor = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    assert!(typed_input::<f32>("reduce_sum", &tensor).is_err());
}

/// The sum and product tables carry complex and integer arms that the float-only paths
/// never reach; calling them for every preset scalar exercises each dispatch body.
#[test]
fn reduce_sum_and_prod_cover_every_preset_scalar() {
    let cases = [
        tensor(vec![1.0_f32, 2.0]),
        tensor(vec![1.0_f64, 2.0]),
        tensor(vec![1_i32, 2]),
        tensor(vec![1_i64, 2]),
        tensor(vec![false, true]),
        tensor(vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, 2.0)]),
        tensor(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)]),
    ];
    let context = strided_kernel::ExecContext::serial();
    for input in &cases {
        let _ = reduce_sum(input, &[0], &context);
        let _ = reduce_prod(input, &[0], &context);
    }
}
