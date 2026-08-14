use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TensorSessionOpsExt, TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::{BackendSessionHost, Error as TensorError, ShapeMismatch, ValidationError};

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

#[test]
fn typed_tensor_reduction_and_structural_wrappers_preserve_values() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        .unwrap();

    let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let [row_sums, total, reshaped, transposed, broadcast] = backend
        .with_backend_session(
            |session| -> tenferro_tensor::Result<[TypedTensor<f64>; 5]> {
                Ok([
                    x.reduce_sum(&[1], session)?,
                    x.reduce_sum(&[0, 1], session)?,
                    x.reshape(&[3, 2], session)?,
                    x.transpose(&[1, 0], session)?,
                    row.broadcast_in_dim(&[2, 3], &[1], session)?,
                ])
            },
        )
        .unwrap();
    assert_eq!(row_sums.shape(), &[2]);
    assert_close(row_sums.host_data().unwrap(), &[6.0, 15.0]);
    assert!(total.shape().is_empty());
    assert_close(total.host_data().unwrap(), &[21.0]);
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_close(
        reshaped.host_data().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_close(
        transposed.host_data().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
    assert_eq!(broadcast.shape(), &[2, 3]);
    assert_close(
        broadcast.host_data().unwrap(),
        &[10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
    );
}

#[test]
fn typed_tensor_matmul_rejects_non_matrix_inputs_without_rank_underflow() {
    let mut backend = CpuBackend::new();
    let scalar = TypedTensor::<f64>::from_vec_col_major(vec![], vec![1.0]).unwrap();
    let vector = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();

    let err = backend
        .with_backend_session(|session| scalar.matmul(&vector, session))
        .unwrap_err();

    assert!(matches!(
        err,
        TensorError::Validation {
            op: "matmul",
            source: ValidationError::RankMismatch {
                expected: 2,
                actual: 0,
            },
        }
    ));
}

#[test]
fn direct_tensor_broadcast_uses_the_shared_shape_payload() {
    let mut backend = CpuBackend::new();
    let lhs = tenferro_tensor::Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let rhs = tenferro_tensor::Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let error = backend
        .with_backend_session(|session| lhs.add(&rhs, session))
        .unwrap_err();

    assert!(matches!(
        error,
        TensorError::Validation {
            source: ValidationError::ShapeMismatch(shape),
            ..
        } if matches!(shape.as_ref(), ShapeMismatch::IncompatibleShapes { lhs, rhs }
            if lhs.as_slice() == [2] && rhs.as_slice() == [3])
    ));
}
