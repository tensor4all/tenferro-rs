//! Error-path tests for graph input binding.
//!
//! One test per Error variant introduced by the placeholder binding API.

mod support;
use support::RunTraced;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::error::Error;
use tenferro_runtime::{GraphCompiler, GraphExecutor, Tensor, TracedTensor};
use tenferro_tensor::DType;

#[test]
fn unexpected_binding_for_data_carrying_leaf() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let y = x.clone();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let extra = Tensor::from_vec_col_major(vec![2], vec![9.0_f64, 9.0]).unwrap();
    let err = y
        .run_with_inputs_auto(&mut engine, &[(&x, &extra)])
        .expect_err("binding a non-placeholder must fail");

    assert!(
        matches!(err, Error::UnexpectedBinding { binding_index: 0 }),
        "got {err:?}"
    );
}

#[test]
fn unbound_placeholder() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = x.clone();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let err = y
        .run_with_inputs_auto(&mut engine, &[])
        .expect_err("unbound placeholder must fail");

    assert!(
        matches!(err, Error::UnboundPlaceholder { .. }),
        "got {err:?}"
    );
}

#[test]
fn duplicate_binding() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = x.clone();

    let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let err = y
        .run_with_inputs_auto(&mut engine, &[(&x, &bound), (&x, &bound)])
        .expect_err("duplicate binding must fail");

    assert!(matches!(err, Error::DuplicateBinding { .. }), "got {err:?}");
}

#[test]
fn placeholder_dtype_mismatch() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = x.clone();

    let wrong_dtype = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let err = y
        .run_with_inputs_auto(&mut engine, &[(&x, &wrong_dtype)])
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
    let y = x.clone();

    let wrong_shape = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let err = y
        .run_with_inputs_auto(&mut engine, &[(&x, &wrong_shape)])
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
    let y = x.clone();

    let wrong_rank = Tensor::from_vec_col_major(vec![4], vec![1.0_f64; 4]).unwrap();
    let mut engine = GraphExecutor::new(CpuBackend::new());
    let err = y
        .run_with_inputs_auto(&mut engine, &[(&x, &wrong_rank)])
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
    let y = x.clone();

    let mut engine = GraphExecutor::new(CpuBackend::new());
    let bound = Tensor::from_vec_col_major(vec![7], vec![1.0_f64; 7]).unwrap();
    let out = y
        .run_with_inputs_auto(&mut engine, &[(&x, &bound)])
        .expect("rank-only placeholder accepts arbitrary shape of that rank");
    assert_eq!(out.shape(), &[7]);
}

#[test]
fn executor_run_with_inputs_validates_and_uses_bound_tensors() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = (&x + &x).unwrap();
    let bound = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, &[2])])
        .expect("symbolic placeholder binding should compile");
    assert_eq!(program.output_count(), 1);

    let mut executor = GraphExecutor::new(CpuBackend::new());
    let out = executor.run_with_inputs(&program, &[(&x, &bound)]).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.as_slice::<f64>().unwrap(), &[2.0, 4.0]);

    let err = executor.run_with_inputs(&program, &[]).unwrap_err();
    assert!(matches!(err, Error::UnboundPlaceholder { .. }));

    let err = executor
        .run_with_inputs(&program, &[(&x, &bound), (&x, &bound)])
        .unwrap_err();
    assert!(matches!(err, Error::DuplicateBinding { .. }));

    let concrete = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = executor
        .run_with_inputs(&program, &[(&concrete, &bound)])
        .unwrap_err();
    assert!(matches!(err, Error::UnexpectedBinding { binding_index: 0 }));
}

#[test]
fn compile_with_input_specs_validates_specs_without_tensor_values() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1);
    let y = (&x + &x).unwrap();
    let shape = [2usize];

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, shape.as_slice())])
        .expect("symbolic placeholder spec should compile");
    assert_eq!(program.output_count(), 1);

    let err = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F32, shape.as_slice())])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::PlaceholderDtypeMismatch {
            expected: DType::F64,
            actual: DType::F32
        }
    ));

    let wrong_rank = [2usize, 1usize];
    let err = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F64, wrong_rank.as_slice())])
        .unwrap_err();
    assert!(matches!(
        err,
        Error::PlaceholderRankMismatch {
            expected: 1,
            actual: 2
        }
    ));

    let concrete = TracedTensor::input_concrete_shape(DType::F64, &[2]);
    let concrete_y = concrete.clone();
    let wrong_shape = [3usize];
    let err = compiler
        .compile_with_input_specs(
            &concrete_y,
            &[(&concrete, DType::F64, wrong_shape.as_slice())],
        )
        .unwrap_err();
    assert!(matches!(err, Error::PlaceholderShapeMismatch { .. }));

    let data_leaf = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let err = compiler
        .compile_with_input_specs(&data_leaf, &[(&data_leaf, DType::F64, shape.as_slice())])
        .unwrap_err();
    assert!(matches!(err, Error::UnexpectedBinding { binding_index: 0 }));

    let err = compiler
        .compile_with_input_specs(
            &y,
            &[
                (&x, DType::F64, shape.as_slice()),
                (&x, DType::F64, shape.as_slice()),
            ],
        )
        .unwrap_err();
    assert!(matches!(err, Error::DuplicateBinding { .. }));

    let err = compiler.compile_with_input_specs(&y, &[]).unwrap_err();
    assert!(matches!(err, Error::UnboundPlaceholder { .. }));
}
