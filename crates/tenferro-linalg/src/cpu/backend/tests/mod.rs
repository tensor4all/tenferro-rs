//! Module-local tests for the linalg dispatch seams.

use super::{
    ensure_host_tensor, tensor_write_view, typed_host, write_view_operand, zeros_like_tensor,
};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{DType, Tensor, TensorWrite, TypedTensor};

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
    let tensor =
        Tensor::from_typed::<i32>(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    assert!(typed_host::<f32>(&tensor, "lu_factor").is_err());
}

/// The accessor reports the module's refusal when the tag and the runtime dtype disagree.
#[test]
fn typed_host_refuses_a_dtype_the_table_never_produces() {
    let tensor =
        Tensor::from_typed::<i32>(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    assert!(typed_host::<f32>(&tensor, "zeros_like_tensor").is_err());
}

/// The write-view accessor's unwind site is the module's documented invariant, so a test states it rather
/// than leaving it assumed: a dtype with no mutable view cannot be adapted.
#[test]
#[should_panic(expected = "linalg validates its input dtypes first")]
fn write_view_operand_unwinds_for_a_dtype_with_no_view() {
    let mut tensor =
        Tensor::from_typed::<i32>(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    let _ = write_view_operand::<f32>(&mut tensor);
}

/// `tensor_write_view` carries one arm per preset scalar. The linalg entry points only ever hand it the
/// floating and complex dtypes they accept, so this drives every arm the `cases` table can build, which is
/// also the table's completeness check: a dtype with no arm would fail to compile rather than unwind.
#[test]
fn tensor_write_view_covers_every_preset_scalar() {
    let expected = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];

    for (mut tensor, expected_dtype) in cases().into_iter().zip(expected) {
        assert_eq!(tensor.dtype(), expected_dtype, "cases table order");
        let view = tensor_write_view(TensorWrite::from_tensor(&mut tensor));
        assert_eq!(view.dtype(), expected_dtype);
    }
}
