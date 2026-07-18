use num_complex::Complex64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use crate::{
    DType, DotGeneralConfig, Error, ErrorPhase, GraphCompiler, GraphExecutor, Tensor, TracedTensor,
};
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{
    Error as TensorError, ErrorKind, ShapeMismatch, ValidationError, ValidationKind,
};

#[test]
fn traced_binary_reuses_input_map_when_rhs_is_already_present() {
    let lhs = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();

    let sum = lhs.add(&rhs).unwrap();
    let reused = sum.add(&rhs).unwrap();

    assert!(Arc::ptr_eq(&sum.inputs_map, &reused.inputs_map));
}

#[test]
fn traced_graph_construction_uses_shared_input_map_merge_helpers() {
    let traced_source = include_str!("../traced.rs");
    let metadata_source = include_str!("../metadata.rs");
    assert!(metadata_source.contains("pub(crate) struct MetadataScopeChain"));
    assert!(traced_source.contains("MetadataScopeChain::with_new"));
    assert!(!traced_source.contains("metadata_scopes_with_new("));
    assert!(!traced_source.contains("metadata_scopes_for_scope("));
    assert!(traced_source.contains("fn merge_traced_inputs_map"));
    assert!(traced_source.contains("input_map_matches_ordered_merge"));
    assert!(!traced_source.contains("let mut merged = (*lhs.inputs_map).clone()"));
    assert!(!traced_source.contains("let mut merged = (*first.inputs_map).clone()"));
    assert!(!traced_source.contains("let mut merged = (*input.inputs_map).clone()"));
    assert!(!traced_source.contains("merged.extend(rhs.inputs_map"));
    assert!(!traced_source.contains("merged.extend(third.inputs_map"));

    let extension_source = include_str!("../extension.rs");
    assert!(extension_source.contains("merge_traced_inputs_map(inputs.iter().copied())"));
    assert!(extension_source.contains("MetadataScopeChain::with_scope"));
    assert!(!extension_source.contains("merged_map.extend(input.inputs_map"));
    assert!(!extension_source.contains("for scope in &input.metadata_scopes"));

    let shape_packing_source = include_str!("../shape_packing.rs");
    assert!(shape_packing_source.contains("merge_traced_inputs_map(tensors.iter())"));
    assert!(shape_packing_source.contains("MetadataScopeChain::with_new"));
    assert!(!shape_packing_source.contains("metadata_scopes.as_slice()"));

    let checkpoint_source = include_str!("../checkpoint.rs");
    assert!(checkpoint_source.contains("pub old_inputs: Arc<HashMap"));
    assert!(!checkpoint_source.contains("old_inputs: node.old_inputs.clone()"));
}

#[test]
fn dot_general_returns_error_for_invalid_config() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![2],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let Err(err) = lhs.dot_general(&rhs, config) else {
        panic!("invalid dim config should be a typed runtime error");
    };

    assert!(matches!(
        &err,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
            ..
        }
    ));
}

#[test]
fn integer_scale_real_rejects_non_finite_factors() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();

    for factor in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = x.scale_real(factor).unwrap_err();
        assert!(
            err.to_string().contains("finite"),
            "expected finite-value error, got {err:?}"
        );
    }
}

#[test]
fn dot_general_keeps_existing_success_metadata() {
    let lhs = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![3, 4], vec![1.0_f64; 12]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let out = lhs
        .dot_general(&rhs, config)
        .expect("valid dot_general config should build a traced tensor");

    assert_eq!(out.rank, 2);
    assert_eq!(out.try_concrete_shape().unwrap(), &[2, 4]);
}

#[test]
fn reductions_reject_invalid_axes_instead_of_saturating_rank() {
    let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();

    let out_of_bounds = x.reduce_max(&[2]).unwrap_err();
    assert!(
        out_of_bounds
            .to_string()
            .contains("axis 2 out of bounds for rank 2"),
        "{out_of_bounds}"
    );

    let duplicate = x.reduce_min(&[0, 0]).unwrap_err();
    assert!(
        duplicate.to_string().contains("duplicate reduction axis 0"),
        "{duplicate}"
    );

    let y = x.reduce_sum(&[1]).unwrap();
    assert_eq!(y.rank, 1);
    assert_eq!(y.try_concrete_shape().unwrap(), &[2]);
}

#[test]
fn symbolic_axis_accessors_reject_invalid_axes() {
    let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();

    let sym_size_err = x.sym_size(2).unwrap_err();
    assert!(
        sym_size_err
            .to_string()
            .contains("axis 2 out of bounds for rank 2"),
        "{sym_size_err}"
    );

    let axis_err = x.axis_sym_dim(2).unwrap_err();
    assert!(
        axis_err
            .to_string()
            .contains("axis 2 out of bounds for rank 2"),
        "{axis_err}"
    );

    assert_eq!(x.axis_sym_dim(0).unwrap().constant_value(), Some(2));
}

#[test]
fn broadcast_in_dim_sym_rejects_missing_shape_reference() {
    let lhs = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let target_shape = [lhs.axis_sym_dim(0).unwrap(), rhs.axis_sym_dim(0).unwrap()];

    let err = lhs
        .broadcast_in_dim_sym(&target_shape, &[0], &[])
        .unwrap_err();

    assert!(
        matches!(
            &err,
            Error::SymbolicShapeConversion {
                op: "broadcast_in_dim_sym",
                phase: ErrorPhase::GraphBuild,
                source: tenferro_ops::SymDimConversionError { .. },
            }
        ),
        "{err}"
    );
    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert_eq!(err.phase(), Some(ErrorPhase::GraphBuild));
}

#[test]
fn broadcast_in_dim_rejects_invalid_dimension_mappings() {
    let x = TracedTensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap();

    let rank_mismatch = x.broadcast_in_dim(&[2, 3, 4], &[0]).unwrap_err();
    assert!(matches!(
        &rank_mismatch,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::RankMismatch {
                expected: 2,
                actual: 1,
            },
            ..
        }
    ));

    let out_of_bounds = x.broadcast_in_dim(&[2, 3, 4], &[0, 3]).unwrap_err();
    assert!(matches!(
        &out_of_bounds,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::AxisOutOfBounds { axis: 3, rank: 3 },
            ..
        }
    ));

    let duplicate = x.broadcast_in_dim(&[2, 3, 4], &[1, 1]).unwrap_err();
    assert!(matches!(
        &duplicate,
        Error::Validation {
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::DuplicateAxis {
                axis: 1,
                role: "broadcast",
            },
            ..
        }
    ));

    let valid = x.broadcast_in_dim(&[2, 3, 4], &[0, 1]).unwrap();
    assert_eq!(valid.rank, 3);
    assert_eq!(valid.try_concrete_shape().unwrap(), &[2, 3, 4]);
}

#[test]
fn broadcast_in_dim_rejects_known_incompatible_extent_at_graph_build() {
    let x = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();

    let err = x.broadcast_in_dim(&[3], &[0]).unwrap_err();

    assert!(matches!(
        &err,
        Error::Validation {
            op: "TracedTensor::broadcast_in_dim",
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::ShapeMismatch(source),
        } if matches!(
            source.as_ref(),
            ShapeMismatch::ExpectedActual { expected, actual }
                if expected.as_slice() == [3] && actual.as_slice() == [2]
        )
    ));
}

#[test]
fn broadcast_in_dim_sym_defers_cross_tensor_symbolic_extent_validation() {
    let input = TracedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let shape_ref = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let target_shape = [shape_ref.axis_sym_dim(0).unwrap()];

    let output = input
        .broadcast_in_dim_sym(&target_shape, &[0], &[&shape_ref])
        .expect("a symbolic extent from another tensor must remain deferred");
    assert_eq!(output.try_concrete_shape(), None);

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&output, &[(&shape_ref, DType::F64, &[2])])
        .unwrap();
    let matching_shape_ref = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();
    let mut executor = GraphExecutor::new(CpuBackend::new());
    let result = executor
        .run_with_inputs(&program, &[(&shape_ref, &matching_shape_ref)])
        .unwrap();
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let mismatch_program = compiler
        .compile_with_input_specs(&output, &[(&shape_ref, DType::F64, &[3])])
        .unwrap();
    let mismatched_shape_ref = Tensor::from_vec_col_major(vec![3], vec![0.0_f64; 3]).unwrap();
    let err = executor
        .run_with_inputs(&mismatch_program, &[(&shape_ref, &mismatched_shape_ref)])
        .unwrap_err();
    assert!(matches!(
        &err,
        Error::TensorRuntime(TensorError::Validation {
            source: ValidationError::ShapeMismatch(source),
            ..
        }) if matches!(
            source.as_ref(),
            ShapeMismatch::IncompatibleShapes { lhs, rhs }
                if lhs.as_slice() == [2] && rhs.as_slice() == [3]
        )
    ));
}

#[test]
fn reshape_rejects_concrete_element_count_mismatch_at_graph_build() {
    let x = TracedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();

    let err = x.reshape(&[3]).unwrap_err();

    assert!(matches!(
        &err,
        Error::Validation {
            op: "TracedTensor::reshape",
            phase: ErrorPhase::GraphBuild,
            source: ValidationError::ShapeMismatch(source),
        }
        if matches!(
            source.as_ref(),
            ShapeMismatch::ReshapeElementCount { from: 4, to: 3 }
        )
    ));
    assert!(err.to_string().contains("element-count mismatch"), "{err}");
}

#[test]
fn reshape_allows_symbolic_input_when_element_count_cannot_be_proven() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();

    let y = x.reshape(&[3]).unwrap();

    assert_eq!(y.rank, 1);
    assert_eq!(y.try_concrete_shape(), Some(vec![3]));
}

#[test]
fn ordered_binary_ops_reject_complex_dtype_without_panicking() {
    let lhs = TracedTensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)]).unwrap();

    let err = lhs.maximum(&rhs).unwrap_err();

    assert!(
        err.to_string()
            .contains("complex numbers have no total order"),
        "{err}"
    );
}

#[test]
fn remainder_rejects_complex_dtype_at_graph_build_without_panicking() {
    let lhs = TracedTensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)]).unwrap();
    let rhs = TracedTensor::from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)]).unwrap();

    let result = catch_unwind(AssertUnwindSafe(|| lhs.rem(&rhs)));
    let result = result.expect("complex remainder validation must not panic");
    let err = result.expect_err("complex remainder must return a typed error");

    assert!(matches!(
        &err,
        Error::Unsupported {
            op: "Rem",
            phase: ErrorPhase::GraphBuild,
            ..
        }
    ));
    assert!(
        err.to_string()
            .contains("complex numbers have no total order"),
        "{err}"
    );
}

#[test]
fn ordered_reductions_reject_complex_dtype_without_panicking() {
    let x = TracedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
    )
    .unwrap();

    let err = x.reduce_max(&[0]).unwrap_err();

    assert!(
        err.to_string()
            .contains("complex numbers have no total order"),
        "{err}"
    );
}
