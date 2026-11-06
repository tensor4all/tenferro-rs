use num_complex::Complex64;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::compiler::semantic_staging::stage_semantic_program;
use crate::exec::ExecOp;
use crate::{DType, DotGeneralConfig, Error, ErrorPhase, GraphCompiler, Tensor, TracedTensor};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::{
    Error as TensorError, ErrorKind, ShapeMismatch, ValidationError, ValidationKind,
};

fn run_compiled(program: &crate::CompiledGraph, inputs: Vec<Tensor>) -> crate::Result<Vec<Tensor>> {
    let staged = stage_semantic_program(program.program(), program.compiler_options()).unwrap();
    let mut engine = CpuEngine::serial();
    crate::exec::run_exec_program_for_test(&mut engine, &staged, inputs)
}

fn run_compiled_one(program: &crate::CompiledGraph, inputs: Vec<Tensor>) -> Tensor {
    let mut outputs = run_compiled(program, inputs).unwrap();
    assert_eq!(outputs.len(), 1);
    outputs.pop().unwrap()
}

#[test]
fn traced_binary_rejects_values_from_another_context() {
    let trace = crate::TraceContext::new();
    let other = crate::TraceContext::new();
    let lhs = trace.from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = other.from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();

    let error = lhs
        .add(&rhs)
        .expect_err("different contexts cannot be combined");
    assert!(
        matches!(error, Error::Validation { .. }),
        "expected a typed validation error, got {error:?}"
    );
    let same_context = trace.from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    // A same-context operand must both combine and evaluate to the right value:
    // `is_ok()` alone would pass if the operand were dropped or mis-bound.
    let sum = lhs
        .add(&same_context)
        .expect("same-context operands must combine");
    let program = GraphCompiler::new().compile(&sum).unwrap();
    let lhs_key = lhs.input_key().unwrap();
    let rhs_key = same_context.input_key().unwrap();
    let mut ordered: [Option<Tensor>; 2] = [None, None];
    ordered[program.input_key_index(&lhs_key).unwrap()] =
        Some(Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap());
    ordered[program.input_key_index(&rhs_key).unwrap()] =
        Some(Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap());
    let inputs = ordered.into_iter().map(Option::unwrap).collect();
    assert_eq!(
        run_compiled_one(&program, inputs)
            .as_slice::<f64>()
            .unwrap(),
        &[4.0]
    );
}

#[test]
fn dot_general_returns_error_for_invalid_config() {
    let trace = crate::TraceContext::new();
    let lhs = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();
    let rhs = trace
        .from_vec_col_major(vec![3, 2], vec![1.0_f64; 6])
        .unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [2].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
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
    let trace = crate::TraceContext::new();
    let x = trace.from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();

    for factor in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = x.scale_real(factor).unwrap_err();
        assert!(
            err.to_string().contains("finite"),
            "expected finite-value error, got {err:?}"
        );
    }
}

#[test]
fn i64_scale_real_rejects_exactly_two_power_63_instead_of_saturating() {
    let trace = crate::TraceContext::new();
    // `i64::MAX as f64` is 2^63, so the old `rounded > i64::MAX as f64`
    // guard let exactly 2^63 through and `as i64` silently saturated to
    // i64::MAX. The inclusive-upper fix must reject it (issue #1685).
    let x = trace.from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();

    let err = x.scale_real(9_223_372_036_854_775_808.0).unwrap_err();
    assert!(
        matches!(
            &err,
            Error::Validation {
                phase: ErrorPhase::GraphBuild,
                source: ValidationError::InvalidArgument { .. },
                ..
            }
        ),
        "2^63 should be rejected as out of i64 range, got {err:?}"
    );
    assert!(
        err.to_string().contains("out of i64 range"),
        "expected i64 range error, got {err:?}"
    );

    // Largest representable f64 below 2^63 (spacing at 2^63 is 1024) and
    // -2^63 == i64::MIN must remain valid.
    x.scale_real(9_223_372_036_854_774_784.0)
        .expect("2^63 - 1024 is the largest valid f64 below 2^63");
    x.scale_real(-9_223_372_036_854_775_808.0)
        .expect("-2^63 == i64::MIN must remain valid");
}

#[test]
fn dot_general_keeps_existing_success_metadata() {
    let trace = crate::TraceContext::new();
    let lhs = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();
    let rhs = trace
        .from_vec_col_major(vec![3, 4], vec![1.0_f64; 12])
        .unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };

    let out = lhs
        .dot_general(&rhs, config)
        .expect("valid dot_general config should build a traced tensor");

    assert_eq!(out.rank, 2);
    assert_eq!(out.try_concrete_shape().unwrap(), &[2, 4]);
}

#[test]
fn reductions_reject_invalid_axes_instead_of_saturating_rank() {
    let trace = crate::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();

    let out_of_bounds = x.reduce_max(Some(&[2])).unwrap_err();
    assert!(
        out_of_bounds
            .to_string()
            .contains("axis 2 out of bounds for rank 2"),
        "{out_of_bounds}"
    );

    let duplicate = x.reduce_min(Some(&[0, 0])).unwrap_err();
    assert!(
        duplicate.to_string().contains("duplicate reduction axis 0"),
        "{duplicate}"
    );

    let y = x.reduce_sum(Some(&[1])).unwrap();
    assert_eq!(y.rank, 1);
    assert_eq!(y.try_concrete_shape().unwrap(), &[2]);
}

#[test]
fn reductions_distinguish_all_axes_from_empty_axes() {
    let trace = crate::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();

    assert_eq!(x.reduce_sum(None).unwrap().rank, 0);
    assert_eq!(x.reduce_prod(Some(&[])).unwrap().rank, 2);
    assert_eq!(x.reduce_max(Some(&[0])).unwrap().rank, 1);
    assert_eq!(x.reduce_min(None).unwrap().rank, 0);

    let scalar = trace.from_vec_col_major(vec![], vec![3.0_f64]).unwrap();
    assert_eq!(scalar.reduce_sum(None).unwrap().rank, 0);
}

#[test]
fn symbolic_axis_accessors_reject_invalid_axes() {
    let trace = crate::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();

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
    let trace = crate::TraceContext::new();
    let lhs = trace.input_symbolic_shape(DType::F64, 1).unwrap();
    let rhs = trace.input_symbolic_shape(DType::F64, 1).unwrap();
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
    let trace = crate::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();

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
    let trace = crate::TraceContext::new();
    let x = trace.from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();

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
    let trace = crate::TraceContext::new();
    let input = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap();
    let shape_ref = trace.input_symbolic_shape(DType::F64, 1).unwrap();
    let target_shape = [shape_ref.axis_sym_dim(0).unwrap()];

    let output = input
        .broadcast_in_dim_sym(&target_shape, &[0], &[&shape_ref])
        .expect("a symbolic extent from another tensor must remain deferred");
    assert_eq!(output.try_concrete_shape(), None);

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&output, &[(&shape_ref, DType::F64, &[2])])
        .unwrap();
    let input_value = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let matching_shape_ref = Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 0.0]).unwrap();
    // The declared spec order is the compiled program's input order.
    let result = run_compiled_one(
        &program,
        vec![matching_shape_ref, input_value.duplicate().unwrap()],
    );
    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[1.0, 2.0]);

    let mismatch_program = compiler
        .compile_with_input_specs(&output, &[(&shape_ref, DType::F64, &[3])])
        .unwrap();
    let mismatched_shape_ref = Tensor::from_vec_col_major(vec![3], vec![0.0_f64; 3]).unwrap();
    let err = run_compiled(&mismatch_program, vec![mismatched_shape_ref, input_value]).unwrap_err();
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
    let trace = crate::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])
        .unwrap();

    let err = x.reshape([3]).unwrap_err();

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
    let trace = crate::TraceContext::new();
    let x = trace.input_symbolic_shape(DType::F64, 2).unwrap();

    let y = x.reshape([3]).unwrap();

    assert_eq!(y.rank, 1);
    assert_eq!(y.try_concrete_shape(), Some(vec![3]));
}

#[test]
fn ordered_binary_ops_reject_complex_dtype_without_panicking() {
    let trace = crate::TraceContext::new();
    let lhs = trace
        .from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])
        .unwrap();
    let rhs = trace
        .from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)])
        .unwrap();

    let err = lhs.maximum(&rhs).unwrap_err();

    assert!(
        err.to_string()
            .contains("complex numbers have no total order"),
        "{err}"
    );
}

#[test]
fn remainder_rejects_complex_dtype_at_graph_build_without_panicking() {
    let trace = crate::TraceContext::new();
    let lhs = trace
        .from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])
        .unwrap();
    let rhs = trace
        .from_vec_col_major(vec![1], vec![Complex64::new(3.0, 4.0)])
        .unwrap();

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
    let trace = crate::TraceContext::new();
    let x = trace
        .from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
        )
        .unwrap();

    let err = x.reduce_max(Some(&[0])).unwrap_err();

    assert!(
        err.to_string()
            .contains("complex numbers have no total order"),
        "{err}"
    );
}

#[test]
fn concatenate_of_distinct_inputs_compiles_without_input_specs() {
    let trace = crate::TraceContext::new();
    let a = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();
    let b = trace
        .from_vec_col_major(vec![2, 4], vec![2.0_f64; 8])
        .unwrap();

    let joined = TracedTensor::concatenate(&[&a, &b], 1).unwrap();
    let program = GraphCompiler::new().compile(&joined).unwrap();

    assert_eq!(program.input_count(), 2);
    let a_key = a.input_key().unwrap();
    let b_key = b.input_key().unwrap();
    let mut ordered: [Option<Tensor>; 2] = [None, None];
    ordered[program.input_key_index(&a_key).unwrap()] =
        Some(Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap());
    ordered[program.input_key_index(&b_key).unwrap()] =
        Some(Tensor::from_vec_col_major(vec![2, 4], vec![2.0_f64; 8]).unwrap());
    let inputs = ordered.into_iter().map(Option::unwrap).collect();

    let result = run_compiled_one(&program, inputs);
    assert_eq!(result.shape(), &[2, 7]);
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn dynamic_slice_rejects_an_invalid_window_when_the_graph_is_built() {
    let trace = crate::TraceContext::new();
    // #2160: the traced surface documents a graph-build rejection for a window
    // that does not fit its operand. It rejected nothing, built a graph with the
    // wrong output metadata, and left the backend to fail at execution.
    let x = trace.input_concrete_shape(DType::F64, &[3]).unwrap();
    let starts = trace.input_concrete_shape(DType::I64, &[1]).unwrap();

    let err = x.dynamic_slice(&starts, &[2, 2]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Validation {
                phase: ErrorPhase::GraphBuild,
                source: ValidationError::RankMismatch {
                    expected: 1,
                    actual: 2
                },
                ..
            }
        ),
        "unexpected error: {err}"
    );

    let err = x.dynamic_slice(&starts, &[4]).unwrap_err();
    assert!(
        matches!(
            err,
            Error::Validation {
                phase: ErrorPhase::GraphBuild,
                source: ValidationError::ShapeMismatch(_),
                ..
            }
        ),
        "unexpected error: {err}"
    );

    // A window that fits still builds, and the result keeps the operand's rank.
    let sliced = x.dynamic_slice(&starts, &[2]).unwrap();
    assert_eq!(sliced.rank, 1);
}

#[test]
fn slice_rejects_a_limit_beyond_a_concrete_operand_extent_when_the_graph_is_built() {
    let trace = crate::TraceContext::new();
    // #2042: the raw `TracedTensor::slice` config API was the only slice surface
    // that accepted a limit beyond its operand, registered a phantom output
    // shape, and failed only at execution. The `slice_axis` builder, CPU and CUDA
    // all reject it.
    let x = trace.input_concrete_shape(DType::F64, &[4]).unwrap();
    let err = x
        .slice(tenferro_tensor::SliceConfig {
            starts: vec![0],
            limits: vec![100],
            strides: vec![1],
        })
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Validation {
                phase: ErrorPhase::GraphBuild,
                source: ValidationError::InvalidSliceBounds {
                    start: 0,
                    end: 100,
                    axis_len: 4
                },
                ..
            }
        ),
        "unexpected error: {err}"
    );

    // A limit that fits still builds, with the extent it selects.
    let sliced = x
        .slice(tenferro_tensor::SliceConfig {
            starts: vec![1],
            limits: vec![4],
            strides: vec![1],
        })
        .unwrap();
    assert_eq!(sliced.rank, 1);
}

#[test]
fn symbolic_reshape_of_a_computed_value_resolves_its_own_axes() {
    let trace = crate::TraceContext::new();
    let x = trace.input_symbolic_shape(DType::F64, 2).unwrap();
    let y = x.neg().unwrap();
    let target = [y.sym_size(0).unwrap(), y.sym_size(1).unwrap()];
    let reshaped = y
        .reshape_sym(&target)
        .expect("a computed value's own axes must convert to its reshape coordinates");

    let mut compiler = GraphCompiler::new();
    for shape in [vec![2usize, 3], vec![4, 5]] {
        let program = compiler
            .compile_with_input_specs(&reshaped, &[(&x, DType::F64, &shape)])
            .unwrap();
        let data: Vec<f64> = (0..shape[0] * shape[1]).map(|i| i as f64).collect();
        let input = Tensor::from_vec_col_major(shape.clone(), data.clone()).unwrap();
        let result = run_compiled_one(&program, vec![input]);
        assert_eq!(result.shape(), shape.as_slice());
        assert_eq!(
            result.as_slice::<f64>().unwrap(),
            data.iter()
                .map(|value| -value)
                .collect::<Vec<_>>()
                .as_slice()
        );
    }
}

#[test]
fn broadcast_in_dim_sym_keeps_computed_shape_references_distinct() {
    let trace = crate::TraceContext::new();
    // Both operands are computed, so neither is identified by an input index.
    let a = trace
        .from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0])
        .unwrap()
        .neg()
        .unwrap();
    let b = trace
        .from_vec_col_major(vec![1, 3], vec![1.0_f64, 2.0, 3.0])
        .unwrap()
        .neg()
        .unwrap();
    let target = [
        a.sym_size(0).unwrap(),
        a.sym_size(1).unwrap(),
        b.sym_size(1).unwrap(),
    ];
    let output = a.broadcast_in_dim_sym(&target, &[0, 1], &[&b]).unwrap();

    let program = GraphCompiler::new().compile(&output).unwrap();
    let result = run_compiled_one(
        &program,
        vec![
            Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 2.0]).unwrap(),
            Tensor::from_vec_col_major(vec![1, 3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ],
    );
    assert_eq!(result.shape(), &[2, 1, 3]);
    assert_eq!(
        result.as_slice::<f64>().unwrap(),
        &[-1.0, -2.0, -1.0, -2.0, -1.0, -2.0]
    );
}

#[test]
fn symbolic_axis_identity_is_unique_per_traced_value() {
    let trace = crate::TraceContext::new();
    let other = crate::TraceContext::new();
    let first = trace
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap()
        .neg()
        .unwrap();
    let second = trace
        .from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0])
        .unwrap()
        .neg()
        .unwrap();
    let foreign = other
        .from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .unwrap()
        .neg()
        .unwrap();

    let ids: Vec<u64> = [&first, &second, &foreign]
        .into_iter()
        .map(|tensor| tensor.symbolic_id())
        .collect();
    // Distinct tensors must not share an axis identity, or a shape built from one
    // silently resolves against another.
    let mut unique = ids.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), ids.len(), "symbolic ids collided: {ids:?}");

    // A shape naming a foreign context's axis must be rejected, not resolved.
    let target = [foreign.sym_size(0).unwrap()];
    let error = first.reshape_sym(&target).unwrap_err();
    assert!(
        matches!(error, Error::SymbolicShapeConversion { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn fixed_shape_outer_product_folds_its_broadcasts_into_views() {
    let trace = crate::TraceContext::new();
    let column = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();
    let row = trace
        .from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    let outer = column
        .dot_general(
            &row,
            DotGeneralConfig {
                lhs_contracting_dims: Default::default(),
                rhs_contracting_dims: Default::default(),
                lhs_batch_dims: Default::default(),
                rhs_batch_dims: Default::default(),
            },
        )
        .unwrap();

    let program = GraphCompiler::new().compile(&outer).unwrap();
    let mut staged = stage_semantic_program(program.program(), program.compiler_options()).unwrap();

    // A literal target must stay foldable. This calls the pass
    // `PreparedProgramRoot::new` runs with the placement predicate always true, so it
    // pins the program shape that decides buffering, not the placement check itself:
    // a multi-input broadcast is refused by that pass and would keep its own buffer,
    // and a single-input constant-target broadcast folds into a view.
    for instruction in &staged.instructions {
        if matches!(instruction.op, ExecOp::BroadcastInDim { .. }) {
            assert_eq!(
                instruction.input_slots.len(),
                1,
                "a fixed-shape outer product must not need a shape reference"
            );
        }
    }
    crate::compiler::fold_metadata_views(&mut staged, |_, _| true);
    assert!(
        !staged
            .instructions
            .iter()
            .any(|instruction| matches!(instruction.op, ExecOp::BroadcastInDim { .. })),
        "a fixed-shape outer product must fold its broadcasts into metadata views"
    );
    let multiply = staged
        .instructions
        .iter()
        .find(|instruction| matches!(instruction.op, ExecOp::Multiply))
        .expect("the outer product becomes an elementwise multiply");
    assert_eq!(
        multiply.input_views.len(),
        2,
        "both operands arrive as views"
    );

    let result = run_compiled_one(
        &program,
        vec![
            Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap(),
        ],
    );
    assert_eq!(result.shape(), &[2, 3, 5]);
    assert_eq!(
        result.as_slice::<f64>().unwrap()[..5],
        [1.0, 2.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn a_bound_only_extent_is_never_reported_as_a_concrete_shape() {
    let trace = crate::TraceContext::new();
    let x = trace
        .from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap();
    let keep = trace.from_vec_col_major(vec![], vec![2.0_f64]).unwrap();
    let truncated = x.dynamic_truncate(&keep, 0).unwrap();
    // The truncation declares `[4]` but holds two elements, so the extent is only an
    // upper bound and must not surface as a concrete shape.
    assert_eq!(truncated.try_concrete_shape(), None);

    // A caller-supplied hint must not restore the declared length.
    let reshaped = truncated.reshape([4]).unwrap();
    assert_eq!(
        reshaped.try_concrete_shape(),
        None,
        "the reshape hint must not turn a bounded extent into a concrete shape"
    );
    // The value has no exact extents, so it carries no symbolic cache at all; callers
    // build one axis at a time with `axis_sym_dim`.
    assert!(reshaped.sym_shape().is_none());
}

/// Program metadata names inputs by position; a handle's cached extents must
/// name the handle itself, never that position read as a tensor id (#2443).
#[test]
fn program_input_positions_never_surface_as_handle_axes() {
    let trace = crate::TraceContext::new();
    let a = trace.input_symbolic_shape(DType::F64, 1).unwrap(); // program input #0
    let b = trace.input_symbolic_shape(DType::F64, 1).unwrap(); // program input #1
    let sum = a.add(&b).unwrap();
    let derived = trace.derive_tensor(sum.trace_value()).unwrap();
    let bound = trace
        .from_vec_col_major(vec![2, 3], vec![1.0_f64; 6])
        .unwrap();

    for handle in [&a, &b, &sum, &derived] {
        let shape = handle.sym_shape().expect("exact symbolic extents");
        assert_eq!(shape.len(), 1);
        assert_eq!(
            shape[0].referenced_tensor_ids(),
            vec![handle.symbolic_id()],
            "a handle's extent must name that handle"
        );
    }
    assert_ne!(derived.symbolic_id(), sum.symbolic_id());
    let constant: Vec<_> = bound
        .sym_shape()
        .unwrap()
        .iter()
        .map(crate::SymDim::constant_value)
        .collect();
    assert_eq!(constant, vec![Some(2), Some(3)]);
}

/// A conversion given another handle's axis rejects it instead of reading the
/// axis of the handle whose id happens to match (#2443).
#[test]
fn reshape_sym_rejects_an_axis_of_another_handle() {
    let trace = crate::TraceContext::new();
    let a = trace.input_symbolic_shape(DType::F64, 1).unwrap();
    let b = trace.input_symbolic_shape(DType::F64, 1).unwrap();

    assert!(a.reshape_sym(&[a.sym_size(0).unwrap()]).is_ok());
    let err = a.reshape_sym(&[b.sym_size(0).unwrap()]).unwrap_err();
    assert!(
        matches!(
            &err,
            Error::SymbolicShapeConversion {
                op: "TracedTensor::reshape_sym",
                source: tenferro_ops::SymDimConversionError { tensor_id },
                ..
            } if *tensor_id == b.symbolic_id()
        ),
        "{err}"
    );
}
