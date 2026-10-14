//! Default-method coverage for the cached dot-general traits, the allocating
//! elementwise fallback, and the conservative structural defaults.

use super::*;
use crate::backend::{dot_general_output_shape, elementwise_read_into_via_allocating_ops};
use crate::{Error, SessionCachedDot, ShapeMismatch, StridedSliceSpec, ValidationError};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape, data).unwrap()
}

fn backend_with_dot(value: f64) -> DefaultReadBackend {
    DefaultReadBackend {
        dot_result: Some(f64_tensor(vec![], vec![value])),
        ..Default::default()
    }
}

fn config(
    lhs_contracting: &[usize],
    rhs_contracting: &[usize],
    lhs_batch: &[usize],
    rhs_batch: &[usize],
) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: lhs_contracting.into(),
        rhs_contracting_dims: rhs_contracting.into(),
        lhs_batch_dims: lhs_batch.into(),
        rhs_batch_dims: rhs_batch.into(),
    }
}

fn scalar_value(tensor: &Tensor) -> f64 {
    tensor.as_slice::<f64>().unwrap()[0]
}

#[test]
fn dot_general_output_shape_orders_free_then_batch_axes() {
    // lhs [2, 3, 4] contracts axis 1 with rhs axis 0 and batches axis 2 with
    // rhs axis 2; free lhs (2), free rhs (5), then batch (4).
    let shape = dot_general_output_shape(
        &[2, 3, 4],
        &[3, 5, 4],
        &config(&[1], &[0], &[2], &[2]),
        "test_dot",
    )
    .unwrap();
    assert_eq!(shape.as_slice(), &[2, 5, 4]);
}

#[test]
fn dot_general_output_shape_rejects_each_malformed_config() {
    let arg = |error: Error| match error {
        Error::Validation {
            op: "test_dot",
            source: ValidationError::InvalidArgument { argument, .. },
        } => argument,
        other => panic!("expected invalid argument, got {other:?}"),
    };
    let err = dot_general_output_shape(&[2], &[2], &config(&[0], &[], &[], &[]), "test_dot");
    assert_eq!(arg(err.unwrap_err()), "contracting_dims");
    let err = dot_general_output_shape(&[2], &[2], &config(&[], &[], &[0], &[]), "test_dot");
    assert_eq!(arg(err.unwrap_err()), "batch_dims");

    let source = |error: Error| match error {
        Error::Validation {
            op: "test_dot",
            source,
        } => source,
        other => panic!("expected validation error, got {other:?}"),
    };

    let err = dot_general_output_shape(&[2], &[2], &config(&[1], &[0], &[], &[]), "test_dot");
    assert_eq!(
        source(err.unwrap_err()),
        ValidationError::AxisOutOfBounds { axis: 1, rank: 1 }
    );
    let err = dot_general_output_shape(&[2], &[2], &config(&[0], &[3], &[], &[]), "test_dot");
    assert_eq!(
        source(err.unwrap_err()),
        ValidationError::AxisOutOfBounds { axis: 3, rank: 1 }
    );
    let err = dot_general_output_shape(
        &[2, 2],
        &[2, 2],
        &config(&[0, 0], &[0, 1], &[], &[]),
        "test_dot",
    );
    assert_eq!(
        source(err.unwrap_err()),
        ValidationError::DuplicateAxis {
            axis: 0,
            role: "lhs_contracting"
        }
    );
    let err = dot_general_output_shape(&[2], &[2], &config(&[], &[], &[0], &[1]), "test_dot");
    assert_eq!(
        source(err.unwrap_err()),
        ValidationError::AxisOutOfBounds { axis: 1, rank: 1 }
    );
    let err = dot_general_output_shape(&[2], &[2], &config(&[0], &[0], &[0], &[0]), "test_dot");
    assert_eq!(
        source(err.unwrap_err()),
        ValidationError::AxisRoleConflict {
            axis: 0,
            first_role: "lhs_contracting",
            second_role: "lhs_batch",
        }
    );
    let err = dot_general_output_shape(
        &[2, 3],
        &[2, 3],
        &config(&[0], &[1], &[1], &[1]),
        "test_dot",
    );
    assert_eq!(
        source(err.unwrap_err()),
        ValidationError::AxisRoleConflict {
            axis: 1,
            first_role: "rhs_contracting",
            second_role: "rhs_batch",
        }
    );

    let err = dot_general_output_shape(&[2], &[3], &config(&[0], &[0], &[], &[]), "test_dot");
    assert_eq!(
        source(err.unwrap_err()),
        ShapeMismatch::ContractedDimensions {
            lhs_axis: 0,
            lhs_size: 2,
            rhs_axis: 0,
            rhs_size: 3,
        }
        .into()
    );
    let err = dot_general_output_shape(&[4], &[5], &config(&[], &[], &[0], &[0]), "test_dot");
    assert_eq!(
        source(err.unwrap_err()),
        ShapeMismatch::ContractedDimensions {
            lhs_axis: 0,
            lhs_size: 4,
            rhs_axis: 0,
            rhs_size: 5,
        }
        .into()
    );
}

#[test]
fn validate_dot_general_read_into_rejects_operand_dtype_and_output_shape() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs_f32 = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let rhs = f64_tensor(vec![2], vec![3.0, 4.0]);
    let mut out = f64_tensor(vec![], vec![0.0]);
    let config = vector_contract_config();

    let err = validate_dot_general_read_into(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs_f32),
        &config,
        &TensorWrite::from_tensor(&mut out),
        "test_dot",
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::Validation {
            source: ValidationError::DTypeMismatch {
                expected: DType::F64,
                actual: DType::F32,
            },
            ..
        }
    ));

    let mut wrong_shape = f64_tensor(vec![1], vec![0.0]);
    let err = validate_dot_general_read_into(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &config,
        &TensorWrite::from_tensor(&mut wrong_shape),
        "test_dot",
    )
    .unwrap_err();
    match err {
        Error::Validation {
            source: ValidationError::ShapeMismatch(mismatch),
            ..
        } => match *mismatch {
            ShapeMismatch::ExpectedActual { expected, actual } => {
                assert!(expected.is_empty());
                assert_eq!(actual.as_slice(), &[1]);
            }
            other => panic!("expected expected/actual mismatch, got {other:?}"),
        },
        other => panic!("expected output shape mismatch, got {other:?}"),
    }

    let validated = validate_dot_general_read_into(
        &TensorRead::from_tensor(&lhs),
        &TensorRead::from_tensor(&rhs),
        &config,
        &TensorWrite::from_tensor(&mut out),
        "test_dot",
    )
    .unwrap();
    assert!(validated.is_empty());
}

#[test]
fn dot_general_read_into_default_overwrites_output_with_the_contraction() {
    let lhs = f64_tensor(vec![1], vec![1.0]);
    let rhs = f64_tensor(vec![1], vec![2.0]);
    let mut out = f64_tensor(vec![], vec![-5.0]);
    let mut backend = backend_with_dot(6.5);

    backend
        .dot_general_read_into(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&rhs),
            &vector_contract_config(),
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[6.5]);
    assert_eq!(backend.calls, vec!["dot_general"]);
}

#[test]
fn backend_cached_dot_defaults_delegate_to_the_uncached_contraction() {
    let lhs = f64_tensor(vec![1], vec![1.0]);
    let rhs = f64_tensor(vec![1], vec![2.0]);
    let lhs_typed = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    let config = vector_contract_config();
    let mut cache = ();

    let mut backend = backend_with_dot(3.0);
    let result = BackendCachedDot::dot_general_cached(
        &mut backend,
        &mut cache,
        Some(1),
        &lhs,
        &rhs,
        &config,
    )
    .unwrap();
    assert_eq!(scalar_value(&result), 3.0);
    assert_eq!(backend.calls, vec!["dot_general"]);

    // A borrowed view is materialized through to_contiguous_read before the
    // cached owned-tensor path runs.
    let mut backend = backend_with_dot(4.0);
    let result = BackendCachedDot::dot_general_read_cached(
        &mut backend,
        &mut cache,
        Some(2),
        TensorRead::from_view(TensorView::F64(lhs_typed.as_view())),
        TensorRead::from_tensor(&rhs),
        &config,
    )
    .unwrap();
    assert_eq!(scalar_value(&result), 4.0);
    assert_eq!(backend.calls, vec!["dot_general"]);

    let mut backend = backend_with_dot(5.0);
    let result = BackendCachedDot::dot_general_with_conj_cached(
        &mut backend,
        &mut cache,
        None,
        &lhs,
        &rhs,
        &config,
        true,
        false,
    )
    .unwrap();
    assert_eq!(scalar_value(&result), 5.0);
    assert_eq!(backend.calls, vec!["conj", "dot_general"]);

    // Without conjugation the read form takes the cached read path directly.
    let mut backend = backend_with_dot(6.0);
    let result = BackendCachedDot::dot_general_with_conj_read_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        false,
        false,
    )
    .unwrap();
    assert_eq!(scalar_value(&result), 6.0);
    assert_eq!(backend.calls, vec!["dot_general"]);

    // With conjugation, both view operands are materialized before conj.
    let rhs_typed = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0]).unwrap();
    let mut backend = backend_with_dot(7.0);
    let result = BackendCachedDot::dot_general_with_conj_read_cached(
        &mut backend,
        &mut cache,
        None,
        TensorRead::from_view(TensorView::F64(lhs_typed.as_view())),
        TensorRead::from_view(TensorView::F64(rhs_typed.as_view())),
        &config,
        false,
        true,
    )
    .unwrap();
    assert_eq!(scalar_value(&result), 7.0);
    assert_eq!(backend.calls, vec!["conj", "dot_general"]);
}

#[test]
fn backend_cached_grouped_gemm_default_runs_the_sequential_fallback() {
    let lhs = f64_tensor(vec![1], vec![1.0]);
    let rhs = f64_tensor(vec![1], vec![2.0]);
    let mut out = f64_tensor(vec![1], vec![9.0]);
    let jobs = [GroupedGemmJob::new(0, 0, 0, 1, 1, 1)];
    let config = scalar_grouped_config(&jobs, DType::F64);
    let mut backend = DefaultReadBackend {
        dot_result: Some(f64_tensor(vec![1, 1], vec![8.0])),
        ..Default::default()
    };

    BackendCachedDot::grouped_gemm_cached(
        &mut backend,
        &mut (),
        Some(0),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[8.0]);
    assert_eq!(backend.calls, vec!["dot_general"]);
}

#[test]
fn session_cached_dot_defaults_delegate_to_the_uncached_contraction() {
    let lhs = f64_tensor(vec![1], vec![1.0]);
    let rhs = f64_tensor(vec![1], vec![2.0]);
    let rhs_typed = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0]).unwrap();
    let config = vector_contract_config();

    let mut backend = backend_with_dot(11.0);
    let result = SessionCachedDot::dot_general_with_conj_read_cached(
        &mut backend,
        Some(4),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_view(TensorView::F64(rhs_typed.as_view())),
        &config,
        true,
        false,
    )
    .unwrap();
    assert_eq!(scalar_value(&result), 11.0);
    assert_eq!(backend.calls, vec!["conj", "dot_general"]);

    let mut backend = backend_with_dot(12.0);
    let result = SessionCachedDot::dot_general_with_conj_read_cached(
        &mut backend,
        Some(4),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        false,
        false,
    )
    .unwrap();
    assert_eq!(scalar_value(&result), 12.0);
    assert_eq!(backend.calls, vec!["dot_general"]);

    // beta = 1 reads the existing output: 2 * 13 + 1 * 10 = 36.
    let mut backend = backend_with_dot(13.0);
    let mut out = f64_tensor(vec![], vec![10.0]);
    SessionCachedDot::dot_general_read_into_accum_cached(
        &mut backend,
        Some(5),
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&rhs),
        &config,
        DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: ContractionScalar::F64(2.0),
            beta: ContractionScalar::F64(1.0),
        },
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap();
    assert_eq!(out.as_slice::<f64>().unwrap(), &[36.0]);
    assert_eq!(backend.calls, vec!["dot_general"]);
}

#[test]
fn fusion_defaults_decline_without_executing() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs = f64_tensor(vec![2], vec![3.0, 4.0]);
    let mut backend = DefaultReadBackend::default();

    let tensor = backend
        .execute_broadcast_multiply(
            TensorRead::from_tensor(&lhs),
            &[2],
            &[0],
            TensorRead::from_tensor(&rhs),
            &[2],
            &[0],
        )
        .unwrap();
    assert!(tensor.is_none());
    let value = backend
        .execute_broadcast_multiply_value(
            TensorRead::from_tensor(&lhs),
            &[2],
            &[0],
            TensorRead::from_tensor(&rhs),
            &[2],
            &[0],
        )
        .unwrap();
    assert!(value.is_none());
    assert!(backend.calls.is_empty());
}

#[test]
fn allocating_elementwise_fallback_copies_each_op_result_into_the_output() {
    let lhs = f64_tensor(vec![1], vec![1.0]);
    let rhs = f64_tensor(vec![1], vec![2.0]);
    let cases = [
        (ElementwiseReadOp::Add, "add"),
        (ElementwiseReadOp::Subtract, "sub"),
        (ElementwiseReadOp::Multiply, "mul"),
        (ElementwiseReadOp::Negate, "neg"),
        (ElementwiseReadOp::Conj, "conj"),
        (ElementwiseReadOp::Divide, "div"),
    ];
    for (op, call) in cases {
        let mut backend = DefaultReadBackend::default();
        let mut out = f64_tensor(vec![1], vec![0.0]);
        let inputs = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
        let inputs = &inputs[..op.arity()];
        elementwise_read_into_via_allocating_ops(
            &mut backend,
            op,
            inputs,
            TensorWrite::from_tensor(&mut out),
        )
        .unwrap();
        // The test backend's allocating ops return the 42.0 marker tensor.
        assert_eq!(out.as_slice::<f64>().unwrap(), &[42.0], "{call}");
        assert_eq!(backend.calls, vec![call]);
    }
}

#[test]
fn allocating_elementwise_fallback_rejects_wrong_arity_before_running() {
    let lhs = f64_tensor(vec![1], vec![1.0]);
    let mut out = f64_tensor(vec![1], vec![-1.0]);
    let mut backend = DefaultReadBackend::default();

    let err = elementwise_read_into_via_allocating_ops(
        &mut backend,
        ElementwiseReadOp::Add,
        &[TensorRead::from_tensor(&lhs)],
        TensorWrite::from_tensor(&mut out),
    )
    .unwrap_err();

    match err {
        Error::Validation {
            source: ValidationError::InvalidArgument { argument, message },
            ..
        } => {
            assert_eq!(argument, "inputs");
            assert_eq!(message, "expected 2 inputs, got 1");
        }
        other => panic!("expected arity error, got {other:?}"),
    }
    assert!(backend.calls.is_empty());
    assert_eq!(out.as_slice::<f64>().unwrap(), &[-1.0]);
}

#[test]
fn div_into_default_overwrites_the_output_through_read_into() {
    let lhs = f64_tensor(vec![2], vec![3.0, 8.0]);
    let rhs = f64_tensor(vec![2], vec![2.0, 4.0]);
    let mut out = f64_tensor(vec![2], vec![0.0, 0.0]);
    let mut backend = DefaultReadBackend::default();

    backend
        .div_into(&lhs, &rhs, TensorWrite::from_tensor(&mut out))
        .unwrap();

    assert_eq!(out.as_slice::<f64>().unwrap(), &[1.5, 2.0]);
}

#[test]
fn rem_defaults_are_explicitly_unsupported() {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![7_i32]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2_i32]).unwrap();
    let mut backend = DefaultReadBackend::default();

    match backend.rem(&lhs, &rhs).unwrap_err() {
        Error::Unsupported { op, message } => {
            assert_eq!(op, "rem");
            assert!(message.contains("I32"), "{message}");
        }
        other => panic!("expected unsupported rem, got {other:?}"),
    }
    match backend
        .rem_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        .unwrap_err()
    {
        Error::Unsupported { op, message } => {
            assert_eq!(op, "rem");
            assert!(message.contains("I32"), "{message}");
        }
        other => panic!("expected unsupported rem, got {other:?}"),
    }

    let typed = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![7]).unwrap();
    match backend
        .rem_read(
            TensorRead::from_view(TensorView::I32(typed.as_view())),
            TensorRead::from_tensor(&rhs),
        )
        .unwrap_err()
    {
        Error::Unsupported { op, message } => {
            assert_eq!(op, "rem");
            assert!(message.contains("borrowed tensor views"), "{message}");
        }
        other => panic!("expected read-boundary error, got {other:?}"),
    }
}

/// A structural backend that relies on every provided default.
struct StructuralDefaults;

impl TensorStructural for StructuralDefaults {
    fn transpose_read(&mut self, _: TensorRead<'_>, _: &[usize]) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
    fn reshape_read(&mut self, _: TensorRead<'_>, _: &[usize]) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
    fn broadcast_in_dim_read(
        &mut self,
        _: TensorRead<'_>,
        _: &[usize],
        _: &[usize],
    ) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
    fn cast(&mut self, _: &Tensor, _: DType) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
    fn extract_diagonal(&mut self, _: &Tensor, _: usize, _: usize) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
    fn embed_diagonal(&mut self, _: &Tensor, _: usize, _: usize) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
    fn tril(&mut self, _: &Tensor, _: i64) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
    fn triu(&mut self, _: &Tensor, _: i64) -> crate::Result<Tensor> {
        unimplemented!("not used by the default-method tests")
    }
}

fn device_backend_tensor() -> Tensor {
    let storage = StorageBuffer::Backend(Box::new(AliasedBackendStorage {
        domain: AllocationDomainId::fresh(),
        allocation: AllocationId::from_backend_id(1944),
        len: 2,
    }));
    let placement = Placement {
        memory_kind: crate::MemoryKind::Device,
        device: None,
        cpu_affinity: None,
    };
    Tensor::from_typed::<f64>(
        TypedTensor::from_buffer_col_major(vec![2], storage, placement).unwrap(),
    )
}

#[test]
fn structural_default_materialization_copies_host_tensors_and_views() {
    let mut backend = StructuralDefaults;

    let owned = Tensor::from_vec_col_major(vec![3], vec![1_i32, 2, 3]).unwrap();
    let copy = backend
        .to_contiguous_read(TensorRead::from_tensor(&owned))
        .unwrap();
    assert_eq!(copy.shape(), &[3]);
    assert_eq!(copy.as_slice::<i32>().unwrap(), &[1, 2, 3]);

    // A compact host view at a nonzero offset copies only its logical window.
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let view = TypedTensorView::from_slice(vec![2], vec![1], 1, data.as_slice()).unwrap();
    let copy = backend
        .to_contiguous_read(TensorRead::from_view(TensorView::F64(view)))
        .unwrap();
    assert_eq!(copy.shape(), &[2]);
    assert_eq!(copy.as_slice::<f64>().unwrap(), &[2.0, 3.0]);
}

#[test]
fn structural_default_materialization_gathers_strided_views() {
    let mut backend = StructuralDefaults;

    // A transposed, non-compact view is gathered into a compact column-major copy.
    let tensor =
        TypedTensor::<i32>::from_vec_col_major(vec![2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
    let transposed = tensor.as_view().transpose_view([1, 0]).unwrap();
    assert_eq!(transposed.strides(), &[2, 1]);
    let copy = backend
        .to_contiguous_read(TensorRead::from_view(TensorView::I32(transposed)))
        .unwrap();
    assert_eq!(copy.shape(), &[3, 2]);
    assert_eq!(copy.as_slice::<i32>().unwrap(), &[1, 3, 5, 2, 4, 6]);

    // A reversed (negative-stride) view gathers in logical order.
    let data = [1_i32, 2, 3, 4];
    let view = TypedTensorView::from_slice(vec![4], vec![1], 0, data.as_slice()).unwrap();
    let reversed = view.slice_view(&[StridedSliceSpec::reverse()]).unwrap();
    assert_eq!(reversed.strides(), &[-1]);
    let copy = backend
        .to_contiguous_read(TensorRead::from_view(TensorView::I32(reversed)))
        .unwrap();
    assert_eq!(copy.shape(), &[4]);
    assert_eq!(copy.as_slice::<i32>().unwrap(), &[4, 3, 2, 1]);
}

#[test]
fn structural_default_materialization_rejects_backend_storage() {
    let mut backend = StructuralDefaults;
    let device = device_backend_tensor();

    let err = backend
        .to_contiguous_read(TensorRead::from_tensor(&device))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "to_contiguous_read",
            ..
        }
    ));

    let typed = device.as_typed::<f64>().unwrap();
    let err = backend
        .to_contiguous_read(TensorRead::from_view(TensorView::F64(typed.as_view())))
        .unwrap_err();
    assert!(matches!(
        err,
        Error::RuntimeState {
            op: "to_contiguous_read",
            ..
        }
    ));
}

#[test]
fn structural_default_copy_is_unsupported_and_leaves_destination_intact() {
    let mut backend = StructuralDefaults;
    let src = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let mut dst = Tensor::from_vec_col_major(vec![2], vec![0_i32, 0]).unwrap();

    let err = backend
        .copy_read_into(
            TensorRead::from_tensor(&src),
            TensorWrite::from_tensor(&mut dst),
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::Unsupported {
            op: "copy_read_into",
            ..
        }
    ));
    assert_eq!(dst.as_slice::<i32>().unwrap(), &[0, 0]);
}

#[test]
fn axpby_validation_rejects_coefficient_dtype_mismatches() {
    let x = f64_tensor(vec![2], vec![1.0, 2.0]);
    let mut y = f64_tensor(vec![2], vec![3.0, 4.0]);

    for (alpha, beta, actual) in [
        (
            ContractionScalar::F32(1.0),
            ContractionScalar::F64(1.0),
            DType::F32,
        ),
        (
            ContractionScalar::F64(1.0),
            ContractionScalar::C64(Complex64::new(1.0, 0.0)),
            DType::C64,
        ),
    ] {
        let err = validate_axpby_read_into_accum(
            alpha,
            &TensorRead::from_tensor(&x),
            beta,
            &TensorWrite::from_tensor(&mut y),
        )
        .unwrap_err();
        match err {
            Error::Validation {
                source:
                    ValidationError::DTypeMismatch {
                        expected,
                        actual: reported,
                    },
                ..
            } => {
                assert_eq!(expected, DType::F64);
                assert_eq!(reported, actual);
            }
            other => panic!("expected coefficient dtype mismatch, got {other:?}"),
        }
    }
    assert_eq!(y.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
}
