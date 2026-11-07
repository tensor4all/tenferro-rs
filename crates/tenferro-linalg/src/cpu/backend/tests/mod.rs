//! Module-local tests for the linalg dispatch seams.

use super::{host_operand, zeros_like_tensor, HostOperandExt};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{Tensor, TypedTensor};

fn cases() -> Vec<Tensor> {
    vec![
        Tensor::from_typed::<f32>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0f32, 0.0, 0.0, 1.0]).unwrap(),
        ),
        Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0f64, 0.0, 0.0, 1.0]).unwrap(),
        ),
        Tensor::from_typed::<i32>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1i32, 0, 0, 1]).unwrap(),
        ),
        Tensor::from_typed::<i64>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![1i64, 0, 0, 1]).unwrap(),
        ),
        Tensor::from_typed::<bool>(
            TypedTensor::from_vec_col_major(vec![2, 2], vec![true, false, false, true]).unwrap(),
        ),
        Tensor::from_typed::<tenferro_tensor::Complex32>(
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
        Tensor::from_typed::<tenferro_tensor::Complex64>(
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

/// Every preset scalar reaches its own arm of the host narrowing and of the zero-like
/// builder, so no tag arm is left unreached by the crate's own suite.
#[test]
fn host_narrowing_and_zero_like_cover_every_preset_scalar() {
    for input in cases() {
        let host = host_operand("lu_factor", &input).unwrap();
        assert_eq!(host.dtype(), input.dtype());
        let zero = zeros_like_tensor(input.dtype(), input.shape()).unwrap();
        assert_eq!(zero.dtype(), input.dtype());
    }
}

/// The tag tables only ask a host operand for the scalar they claim, so a mismatched
/// `as_typed` is unreachable from the public API. Covering it here keeps the invariant
/// honest: a mismatch is `None` (the tables turn it into their typed refusal), never a panic.
#[test]
fn host_operand_as_typed_refuses_a_dtype_the_dispatch_never_produces() {
    let tensor =
        Tensor::from_typed::<i32>(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    let host = host_operand("lu_factor", &tensor).unwrap();
    assert!(host.as_typed::<f32>().is_none());
    assert!(host.as_typed::<i32>().is_some());
}

/// A caller-owned payload has no preset-scalar host view, so the narrowing refuses it
/// with the operation's name, as the per-entry host check did.
#[test]
fn host_operand_refuses_an_external_payload_by_operation() {
    use tenferro_tensor::{DynRank, ErasedHostTensor, Host};
    let payload = ErasedHostTensor::new(
        TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![1], vec![1.0]).unwrap(),
    );
    let tensor = Tensor::external(payload);
    let err = host_operand("cholesky", &tensor).unwrap_err();
    assert!(err.to_string().contains("cholesky"), "{err}");
}
