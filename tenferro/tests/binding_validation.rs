//! Error-path tests for `eval_with_inputs`.
//!
//! One test per Error variant introduced by the placeholder binding API.

use tenferro::error::Error;
use tenferro::{CpuBackend, Engine, Tensor, TracedTensor};
use tenferro_tensor::DType;

#[test]
fn unexpected_binding_for_data_carrying_leaf() {
    let x = TracedTensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let mut y = x.clone();

    let mut engine = Engine::new(CpuBackend::new());
    let extra = Tensor::from_vec(vec![2], vec![9.0_f64, 9.0]);
    let err = y
        .eval_with_inputs(&mut engine, &[(&x, &extra)])
        .expect_err("binding a non-placeholder must fail");

    assert!(
        matches!(err, Error::UnexpectedBinding { binding_index: 0 }),
        "got {err:?}"
    );
}

#[test]
fn unbound_placeholder() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let mut y = x.clone();

    let mut engine = Engine::new(CpuBackend::new());
    let err = y
        .eval_with_inputs(&mut engine, &[])
        .expect_err("unbound placeholder must fail");

    assert!(
        matches!(err, Error::UnboundPlaceholder { .. }),
        "got {err:?}"
    );
}

#[test]
fn duplicate_binding() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let mut y = x.clone();

    let bound = Tensor::from_vec(vec![2], vec![1.0_f64, 2.0]);
    let mut engine = Engine::new(CpuBackend::new());
    let err = y
        .eval_with_inputs(&mut engine, &[(&x, &bound), (&x, &bound)])
        .expect_err("duplicate binding must fail");

    assert!(matches!(err, Error::DuplicateBinding { .. }), "got {err:?}");
}

#[test]
fn placeholder_dtype_mismatch() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let mut y = x.clone();

    let wrong_dtype = Tensor::from_vec(vec![2], vec![1.0_f32, 2.0]);
    let mut engine = Engine::new(CpuBackend::new());
    let err = y
        .eval_with_inputs(&mut engine, &[(&x, &wrong_dtype)])
        .expect_err("dtype mismatch must fail");

    assert!(
        matches!(
            err,
            Error::PlaceholderDtypeMismatch {
                expected: DType::F64,
                actual: DType::F32
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn placeholder_shape_mismatch_for_concrete_shape_placeholder() {
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]);
    let mut y = x.clone();

    let wrong_shape = Tensor::from_vec(vec![3, 2], vec![1.0_f64; 6]);
    let mut engine = Engine::new(CpuBackend::new());
    let err = y
        .eval_with_inputs(&mut engine, &[(&x, &wrong_shape)])
        .expect_err("shape mismatch must fail");

    match err {
        Error::PlaceholderShapeMismatch { expected, actual } => {
            assert_eq!(expected, vec![2, 3]);
            assert_eq!(actual, vec![3, 2]);
        }
        other => panic!("expected PlaceholderShapeMismatch, got {other:?}"),
    }
}

#[test]
fn placeholder_rank_mismatch_for_symbolic_shape_placeholder() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2);
    let mut y = x.clone();

    let wrong_rank = Tensor::from_vec(vec![4], vec![1.0_f64; 4]);
    let mut engine = Engine::new(CpuBackend::new());
    let err = y
        .eval_with_inputs(&mut engine, &[(&x, &wrong_rank)])
        .expect_err("rank mismatch must fail");

    assert!(
        matches!(
            err,
            Error::PlaceholderRankMismatch {
                expected: 2,
                actual: 1
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn symbolic_shape_placeholder_accepts_any_shape_of_matching_rank() {
    // Sanity check that with the right rank + dtype, binding succeeds.
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let mut y = x.clone();

    let mut engine = Engine::new(CpuBackend::new());
    let bound = Tensor::from_vec(vec![7], vec![1.0_f64; 7]);
    let out = y
        .eval_with_inputs(&mut engine, &[(&x, &bound)])
        .expect("rank-only placeholder accepts arbitrary shape of that rank");
    assert_eq!(out.shape(), &[7]);
}
