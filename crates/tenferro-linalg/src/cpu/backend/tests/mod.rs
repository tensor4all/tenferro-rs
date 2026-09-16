//! Module-local tests for the linalg dispatch seams.

use super::{ensure_host_tensor, typed_host, zeros_like_tensor};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{Tensor, TypedTensor};

fn cases() -> Vec<Tensor> {
    vec![
        Tensor::F32(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0]).unwrap(),
        ),
        Tensor::F64(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0f64, 0.0, 0.0, 1.0]).unwrap(),
        ),
        Tensor::I32(TypedTensor::from_vec_col_major(vec![2, 2], vec![1i32, 0, 0, 1]).unwrap()),
        Tensor::I64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1i64, 0, 0, 1]).unwrap()),
        Tensor::Bool(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![true, false, false, true]).unwrap(),
        ),
        Tensor::C32(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex32::new(1.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(1.0, 0.0),
                ],
            )
            .unwrap(),
        ),
        Tensor::C64(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            )
            .unwrap(),
        ),
    ]
}

/// Every preset scalar reaches its own arm of the host check and of the zero-like
/// builder, so no tag arm is left unreached by the crate's own suite.
#[test]
fn host_check_and_zero_like_cover_every_preset_scalar() {
    for input in cases() {
        assert!(ensure_host_tensor("lu_factor", &input).is_ok());
        let zero = zeros_like_tensor(&input).unwrap();
        assert_eq!(zero.dtype(), input.dtype());
    }
}

/// The tag tables only reach `typed_host` with the scalar they claim, so its refusal is
/// unreachable from the public API. Covering it here keeps the invariant honest: a
/// mismatch is the same typed refusal the wildcard arms produce, never a panic.
#[test]
fn typed_host_refuses_a_dtype_the_dispatch_never_produces() {
    let tensor = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    assert!(typed_host::<f32>(&tensor, "lu_factor").is_err());
}
