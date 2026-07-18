use std::sync::Arc;

use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Error, ErrorPhase, TracedTensor};
use tenferro_tensor::{Error as TensorError, ShapeMismatch, ValidationError};

fn eager_tensor(ctx: Arc<EagerRuntime>, shape: Vec<usize>) -> EagerTensor {
    let size = shape.iter().product();
    EagerTensor::from_tensor_in(
        tenferro_tensor::Tensor::from_vec_col_major(shape, vec![1.0_f64; size]).unwrap(),
        ctx,
    )
    .unwrap()
}

fn assert_expected_actual(source: &ValidationError, expected: &[usize], actual: &[usize]) {
    assert!(matches!(
        source,
        ValidationError::ShapeMismatch(shape)
            if matches!(shape.as_ref(), ShapeMismatch::ExpectedActual { expected: found_expected, actual: found_actual }
                if found_expected.as_slice() == expected && found_actual.as_slice() == actual)
    ));
}

fn assert_rank_mismatch(source: &ValidationError, expected: usize, actual: usize) {
    assert!(matches!(
        source,
        ValidationError::RankMismatch {
            expected: found_expected,
            actual: found_actual,
        } if *found_expected == expected && *found_actual == actual
    ));
}

#[test]
fn eager_and_traced_broadcast_errors_share_payloads_across_discovery_phases() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let cases = [
        "incompatible_binary",
        "incompatible_input",
        "rank_too_large",
    ];

    for case in cases {
        let eager_error = match case {
            "incompatible_binary" => {
                let lhs = eager_tensor(ctx.clone(), vec![2]);
                let rhs = eager_tensor(ctx.clone(), vec![3]);
                lhs.add(&rhs).unwrap_err()
            }
            "incompatible_input" => {
                let input = eager_tensor(ctx.clone(), vec![2, 3]);
                input.broadcast_in_dim(&[2, 4], &[0, 1]).unwrap_err()
            }
            "rank_too_large" => {
                let input = eager_tensor(ctx.clone(), vec![2, 3]);
                input.broadcast_in_dim(&[3], &[0, 0]).unwrap_err()
            }
            _ => unreachable!(),
        };

        let traced_error = match case {
            "incompatible_binary" => {
                let lhs = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
                let rhs = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
                lhs.add(&rhs).unwrap_err()
            }
            "incompatible_input" => {
                let input = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
                input.broadcast_in_dim(&[2, 4], &[0, 1]).unwrap_err()
            }
            "rank_too_large" => {
                let input = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
                input.broadcast_in_dim(&[3], &[0, 0]).unwrap_err()
            }
            _ => unreachable!(),
        };

        assert_eq!(eager_error.kind(), traced_error.kind(), "{case}");
        match case {
            "incompatible_binary" => {
                match eager_error {
                    Error::TensorRuntime(TensorError::Validation { source, .. }) => {
                        assert!(matches!(
                            source,
                            ValidationError::ShapeMismatch(shape)
                                if matches!(shape.as_ref(), ShapeMismatch::IncompatibleShapes { lhs, rhs }
                                    if lhs.as_slice() == [2] && rhs.as_slice() == [3])
                        ));
                    }
                    other => panic!("unexpected eager error: {other:?}"),
                }
                assert!(matches!(
                    traced_error,
                    Error::Validation {
                        phase: ErrorPhase::GraphBuild,
                        source: ValidationError::ShapeMismatch(shape),
                        ..
                    } if matches!(shape.as_ref(), ShapeMismatch::IncompatibleShapes { lhs, rhs }
                        if lhs.as_slice() == [2] && rhs.as_slice() == [3])
                ));
            }
            "incompatible_input" => {
                match eager_error {
                    Error::TensorRuntime(TensorError::Validation { source, .. }) => {
                        assert_expected_actual(&source, &[2, 4], &[2, 3]);
                    }
                    other => panic!("unexpected eager error: {other:?}"),
                }
                assert!(matches!(
                    traced_error,
                    Error::Validation {
                        phase: ErrorPhase::GraphBuild,
                        source,
                        ..
                    } if { assert_expected_actual(&source, &[2, 4], &[2, 3]); true }
                ));
            }
            "rank_too_large" => {
                match eager_error {
                    Error::TensorRuntime(TensorError::Validation { source, .. }) => {
                        assert_rank_mismatch(&source, 1, 2);
                    }
                    other => panic!("unexpected eager error: {other:?}"),
                }
                assert!(matches!(
                    traced_error,
                    Error::Validation {
                        phase: ErrorPhase::GraphBuild,
                        source,
                        ..
                    } if { assert_rank_mismatch(&source, 1, 2); true }
                ));
            }
            _ => unreachable!(),
        }
    }
}
