use tenferro_cpu::CpuBackend;
use tenferro_runtime::{TypedTensor, TypedTensorOpsExt};
use tenferro_tensor::{Error as TensorError, ValidationError};

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

    let row_sums = x.reduce_sum(&[1], &mut backend).unwrap();
    assert_eq!(row_sums.shape(), &[2]);
    assert_close(row_sums.host_data().unwrap(), &[6.0, 15.0]);

    let total = x.reduce_sum(&[0, 1], &mut backend).unwrap();
    assert_eq!(total.shape(), &[]);
    assert_close(total.host_data().unwrap(), &[21.0]);

    let reshaped = x.reshape(&[3, 2], &mut backend).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_close(
        reshaped.host_data().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );

    let transposed = x.transpose(&[1, 0], &mut backend).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_close(
        transposed.host_data().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );

    let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let broadcast = row.broadcast_in_dim(&[2, 3], &[1], &mut backend).unwrap();
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

    let err = scalar.matmul(&vector, &mut backend).unwrap_err();

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
