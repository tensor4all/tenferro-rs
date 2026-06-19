use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{
    DType, Error, GraphCompiler, GraphExecutor, Tensor, TracedTensor, TypedTensor,
};
use tenferro_tensor::Error as TensorError;

fn traced_with_dtype(dtype: DType, shape: Vec<usize>) -> TracedTensor {
    let n_elements = shape.iter().product();
    let tensor = match dtype {
        DType::F32 => {
            Tensor::F32(TypedTensor::from_vec_col_major(shape, vec![1.0_f32; n_elements]).unwrap())
        }
        DType::F64 => {
            Tensor::F64(TypedTensor::from_vec_col_major(shape, vec![1.0_f64; n_elements]).unwrap())
        }
        DType::I32 => {
            Tensor::I32(TypedTensor::from_vec_col_major(shape, vec![1_i32; n_elements]).unwrap())
        }
        DType::I64 => {
            Tensor::I64(TypedTensor::from_vec_col_major(shape, vec![1_i64; n_elements]).unwrap())
        }
        DType::Bool => {
            Tensor::Bool(TypedTensor::from_vec_col_major(shape, vec![true; n_elements]).unwrap())
        }
        DType::C32 => Tensor::C32(
            TypedTensor::from_vec_col_major(shape, vec![Complex32::new(1.0, 0.5); n_elements])
                .unwrap(),
        ),
        DType::C64 => Tensor::C64(
            TypedTensor::from_vec_col_major(shape, vec![Complex64::new(1.0, 0.5); n_elements])
                .unwrap(),
        ),
    };
    TracedTensor::from_tensor_concrete_shape(tensor).unwrap()
}

#[test]
fn svd_executes_after_runtime_registration() {
    let a = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 2.0]).unwrap(),
    )
    .unwrap();
    let (u, s, vt) = a.svd().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&u, &s, &vt]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs.len(), 3);
    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
}

#[test]
fn complex_svd_runtime_singular_values_match_traced_real_dtype() {
    let a = TracedTensor::from_tensor_concrete_shape(Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(3.0, 0.5),
                Complex64::new(0.2, -0.4),
                Complex64::new(-0.1, 0.3),
                Complex64::new(2.0, -0.2),
            ],
        )
        .unwrap(),
    ))
    .unwrap();
    let (_u, s, _vt) = a.svd().unwrap();
    assert_eq!(s.dtype, DType::F64);

    let weights = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2], vec![0.5_f64, 2.0]).unwrap(),
    )
    .unwrap();
    let weighted = (&s * &weights).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&s, &weighted]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs[0].dtype(), DType::F64);
    assert_eq!(outputs[1].dtype(), DType::F64);
    assert_eq!(outputs[0].shape(), &[2]);
    assert_eq!(outputs[1].shape(), &[2]);
}

#[test]
fn missing_runtime_reports_linalg_family() {
    let a = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 1.0]).unwrap(),
    )
    .unwrap();
    let y = a.cholesky().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile(&y).unwrap();
    let err = GraphExecutor::new(CpuBackend::new())
        .run(&program)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains(tenferro_linalg::LINALG_EXTENSION_FAMILY_ID));
}

#[test]
fn full_piv_lu_multi_output_slots_are_preserved() {
    let a = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0]).unwrap(),
    )
    .unwrap();
    let (p, l, u, q, parity) = a.full_piv_lu().unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler.compile_many(&[&p, &l, &u, &q, &parity]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    executor
        .register_extension(tenferro_linalg::register_runtime)
        .unwrap();
    let outputs = executor.run_many(&program).unwrap();

    assert_eq!(outputs[0].shape(), &[2, 2]);
    assert_eq!(outputs[1].shape(), &[2, 2]);
    assert_eq!(outputs[2].shape(), &[2, 2]);
    assert_eq!(outputs[3].shape(), &[2, 2]);
    assert_eq!(outputs[4].shape(), &[] as &[usize]);
}

#[test]
fn traced_metadata_matches_linalg_extension_shapes_and_dtypes() {
    let rectangular = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![3, 4, 2], vec![1.0_f64; 24]).unwrap(),
    )
    .unwrap();
    let square = TracedTensor::from_tensor_concrete_shape(
        Tensor::from_vec_col_major(vec![3, 3, 2], vec![1.0_f64; 18]).unwrap(),
    )
    .unwrap();
    let complex_square = TracedTensor::from_tensor_concrete_shape(Tensor::C64(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![Complex64::new(1.0, 0.0); 4]).unwrap(),
    ))
    .unwrap();
    let ints = TracedTensor::from_tensor_concrete_shape(Tensor::I64(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1, 0, 0, 2]).unwrap(),
    ))
    .unwrap();

    let (u, s, vt) = rectangular.svd().unwrap();
    assert_eq!(u.concrete_shape().unwrap(), vec![3, 3, 2]);
    assert_eq!(s.concrete_shape().unwrap(), vec![3, 2]);
    assert_eq!(vt.concrete_shape().unwrap(), vec![3, 4, 2]);

    let (q, r) = rectangular.qr().unwrap();
    assert_eq!(q.concrete_shape().unwrap(), vec![3, 3, 2]);
    assert_eq!(r.concrete_shape().unwrap(), vec![3, 4, 2]);

    let (values, vectors) = ints.eig().unwrap();
    assert_eq!(values.dtype, DType::C64);
    assert_eq!(vectors.dtype, DType::C64);

    let (eigh_values, eigh_vectors) = square.eigh().unwrap();
    assert_eq!(eigh_values.concrete_shape().unwrap(), vec![3, 2]);
    assert_eq!(eigh_vectors.concrete_shape().unwrap(), vec![3, 3, 2]);

    let (complex_eigh_values, complex_eigh_vectors) = complex_square.eigh().unwrap();
    assert_eq!(complex_eigh_values.dtype, DType::F64);
    assert_eq!(complex_eigh_vectors.dtype, DType::C64);

    let (p, l, u, q, parity) = complex_square.full_piv_lu().unwrap();
    assert_eq!(p.dtype, DType::C64);
    assert_eq!(l.dtype, DType::C64);
    assert_eq!(u.dtype, DType::C64);
    assert_eq!(q.dtype, DType::C64);
    assert_eq!(parity.dtype, DType::F64);
    assert_eq!(parity.rank, 0);
}

#[test]
fn traced_metadata_promotes_linalg_dtypes_broadly() {
    let promotion_cases = [
        (DType::Bool, DType::I32, DType::I32),
        (DType::I32, DType::I64, DType::I64),
        (DType::I32, DType::F32, DType::F64),
        (DType::I64, DType::C32, DType::C64),
        (DType::F32, DType::F64, DType::F64),
        (DType::F32, DType::C32, DType::C32),
        (DType::F32, DType::C64, DType::C64),
        (DType::F64, DType::C32, DType::C64),
        (DType::C32, DType::C64, DType::C64),
    ];

    for (a_dtype, b_dtype, expected_dtype) in promotion_cases {
        let a = traced_with_dtype(a_dtype, vec![2, 2]);
        let b = traced_with_dtype(b_dtype, vec![2, 1]);
        let solved = a.solve(&b).unwrap();
        assert_eq!(solved.dtype, expected_dtype);
        assert_eq!(solved.concrete_shape().unwrap(), vec![2, 1]);
    }

    let triangular = traced_with_dtype(DType::Bool, vec![2, 2])
        .triangular_solve(
            &traced_with_dtype(DType::F32, vec![2, 1]),
            true,
            false,
            true,
            true,
        )
        .unwrap();
    assert_eq!(triangular.dtype, DType::F32);
    assert_eq!(triangular.concrete_shape().unwrap(), vec![2, 1]);
}

#[test]
fn traced_metadata_covers_eig_output_dtype_rules() {
    for (input_dtype, expected_dtype) in [
        (DType::F64, DType::C64),
        (DType::C64, DType::C64),
        (DType::F32, DType::C32),
        (DType::C32, DType::C32),
        (DType::I32, DType::C64),
        (DType::I64, DType::C64),
        (DType::Bool, DType::C64),
    ] {
        let a = traced_with_dtype(input_dtype, vec![2, 2]);
        let (values, vectors) = a.eig().unwrap();
        assert_eq!(values.dtype, expected_dtype);
        assert_eq!(vectors.dtype, expected_dtype);
        assert_eq!(values.concrete_shape().unwrap(), vec![2]);
        assert_eq!(vectors.concrete_shape().unwrap(), vec![2, 2]);
    }
}

#[test]
fn traced_norm_rejects_integer_and_bool_dtypes_before_scalar_rounding() {
    for dtype in [DType::I32, DType::I64, DType::Bool] {
        let tensor = traced_with_dtype(dtype, vec![3]);
        let err = match tensor.norm(Some(2.0), Some(&[0]), false) {
            Ok(_) => panic!("expected unsupported dtype error for {dtype:?}"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::BackendFailure {
                    op: "norm",
                    ref message,
                }) if message.contains("unsupported dtype")
            ),
            "expected unsupported dtype error for {dtype:?}, got {err:?}"
        );

        let err = match tensor.pinv_with_rtol(1.0e-12) {
            Ok(_) => panic!("expected unsupported dtype error for {dtype:?}"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::BackendFailure {
                    op: "pinv_with_rtol",
                    ref message,
                }) if message.contains("unsupported dtype")
            ),
            "expected unsupported dtype error for {dtype:?}, got {err:?}"
        );
    }
}

#[test]
fn traced_inv_rejects_rank_less_than_two_without_panicking() {
    let scalar = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).unwrap();

    let err = scalar.inv().unwrap_err();

    assert!(matches!(
        err,
        Error::TensorRuntime(TensorError::RankMismatch {
            op: "inv",
            expected: 2,
            actual: 0,
        })
    ));
}

#[test]
fn traced_linalg_helpers_reject_symbolic_shapes_without_panicking() {
    let matrix = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();

    for (op, result) in [
        ("inv", matrix.inv()),
        ("pinv", matrix.pinv()),
        ("pinv_with_rtol", matrix.pinv_with_rtol(1.0e-12)),
        ("norm", matrix.norm(Some(2.0), Some(&[0, 1]), true)),
    ] {
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::BackendFailure {
                    op: actual_op,
                    ref message,
                }) if actual_op == op && message.contains("symbolic shape")
            ),
            "expected symbolic-shape error for {op}, got {err:?}"
        );
    }
}

#[test]
fn traced_norm_rejects_out_of_range_axis_without_panicking() {
    let tensor = TracedTensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();

    let err = tensor.norm(Some(2.0), Some(&[5]), false).unwrap_err();

    assert!(matches!(
        err,
        Error::TensorRuntime(TensorError::AxisOutOfBounds {
            op: "norm",
            axis: 5,
            rank: 1,
        })
    ));
}

#[test]
fn traced_pinv_rejects_integer_and_bool_dtypes_before_scalar_rounding() {
    for dtype in [DType::I32, DType::I64, DType::Bool] {
        let tensor = traced_with_dtype(dtype, vec![2, 2]);
        let err = match tensor.pinv() {
            Ok(_) => panic!("expected unsupported dtype error for {dtype:?}"),
            Err(err) => err,
        };
        assert!(
            matches!(
                err,
                Error::TensorRuntime(TensorError::BackendFailure {
                    op: "pinv",
                    ref message,
                }) if message.contains("unsupported dtype")
            ),
            "expected unsupported dtype error for {dtype:?}, got {err:?}"
        );
    }
}
