mod support;
use std::panic::{catch_unwind, AssertUnwindSafe};
use support::RunTraced;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::traced::TracedTensor;
use tenferro_tensor::{DotGeneralConfig, Tensor, TypedTensor};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn assert_panic_contains(config: DotGeneralConfig, expected_substring: &str) {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = a.dot_general(&b, config);
    }));
    match result {
        Err(panic_payload) => {
            if let Some(msg) = panic_payload.downcast_ref::<String>() {
                assert!(
                    msg.contains(expected_substring),
                    "expected message containing '{expected_substring}', got: {msg}"
                );
            } else {
                panic!("panic payload is not a String");
            }
        }
        Ok(_) => panic!("expected panic but dot_general succeeded"),
    }
}

// Tests for stale lhs_rank/rhs_rank in config were removed after Task 4 of the
// rank-removal plan: `TracedTensor::dot_general` no longer validates any stale
// rank fields against the actual tensor rank. The traced path now derives
// ranks directly from the tensor shapes, and issue #664 removed the redundant
// `lhs_rank`/`rhs_rank` fields on `DotGeneralConfig` entirely (Task 10).

#[test]
fn traced_dot_general_rejects_out_of_bounds_contracting_dim() {
    assert_panic_contains(
        DotGeneralConfig {
            lhs_contracting_dims: vec![5],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
        "out of bounds",
    );
}

#[test]
fn traced_dot_general_rejects_contracting_batch_overlap() {
    assert_panic_contains(
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![1],
            rhs_batch_dims: vec![],
        },
        "both contracting and batch",
    );
}

#[test]
fn traced_dot_general_accepts_valid_config() {
    let a =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
    let b =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let c = a.dot_general(&b, config);
    let mut engine = tenferro_runtime::GraphExecutor::new(CpuBackend::new());
    let result = c.run_with(&mut engine).unwrap();
    let data = result.as_slice::<f64>().unwrap();
    assert_eq!(data.len(), 4);
}

#[test]
fn dot_general_config_validate_dims_ok() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![1],
    };
    assert!(config.validate_dims_with_ranks(2, 2).is_ok());
}

#[test]
fn dot_general_config_validate_dims_out_of_bounds() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![3],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let err = config.validate_dims_with_ranks(2, 2).unwrap_err();
    assert!(err.contains("out of bounds"));
}

#[test]
fn dot_general_config_validate_dims_contracting_count_mismatch() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![0, 1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let err = config.validate_dims_with_ranks(2, 2).unwrap_err();
    assert!(err.contains("contracting dim counts differ"));
}

#[test]
fn dot_general_config_validate_dims_batch_count_mismatch() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![0],
        rhs_batch_dims: vec![],
    };
    let err = config.validate_dims_with_ranks(2, 2).unwrap_err();
    assert!(err.contains("batch dim counts differ"));
}

#[test]
fn traced_dot_general_rejects_rhs_out_of_bounds_contracting_dim() {
    assert_panic_contains(
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![5],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
        "out of bounds",
    );
}

#[test]
fn traced_dot_general_rejects_lhs_batch_out_of_bounds() {
    assert_panic_contains(
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![5],
            rhs_batch_dims: vec![],
        },
        "out of bounds",
    );
}

#[test]
fn traced_dot_general_rejects_rhs_batch_out_of_bounds() {
    assert_panic_contains(
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![5],
        },
        "out of bounds",
    );
}

#[test]
fn traced_dot_general_rejects_rhs_contracting_batch_overlap() {
    assert_panic_contains(
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![0],
        },
        "both contracting and batch",
    );
}

#[test]
fn traced_dot_general_rejects_duplicate_contracting_dims() {
    assert_panic_contains(
        DotGeneralConfig {
            lhs_contracting_dims: vec![0, 0],
            rhs_contracting_dims: vec![0, 1],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        },
        "duplicate dim",
    );
}

#[test]
fn dot_general_config_validate_dims_rhs_out_of_bounds() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![5],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let err = config.validate_dims_with_ranks(2, 2).unwrap_err();
    assert!(err.contains("out of bounds"));
}

#[test]
fn dot_general_config_validate_dims_duplicate_batch_dims() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![2],
        lhs_batch_dims: vec![0, 0],
        rhs_batch_dims: vec![1, 1],
    };
    let err = config.validate_dims_with_ranks(3, 3).unwrap_err();
    assert!(err.contains("duplicate dim"));
}

#[test]
fn traced_dot_general_accepts_batched_valid_config() {
    let a = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ));
    let b = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 2, 2],
        vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
    ));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![0, 2],
        rhs_batch_dims: vec![1, 2],
    };
    let c = a.dot_general(&b, config);
    let mut engine = tenferro_runtime::GraphExecutor::new(CpuBackend::new());
    let result = c.run_with(&mut engine).unwrap();
    let data = result.as_slice::<f64>().unwrap();
    assert_eq!(data.len(), 4);
}
