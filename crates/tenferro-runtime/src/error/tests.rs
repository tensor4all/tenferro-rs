use std::error::Error as StdError;

use tenferro_ops::dim_expr::{DimExpr, DimExprEvalError};
use tenferro_tensor::{DType, ErrorKind, ShapeMismatch, ShapeVec, ValidationError, ValidationKind};

use super::{ContextId, Error, ErrorPhase, ShapeConstraintEvalError};

#[test]
fn dimension_evaluation_errors_keep_the_runtime_vocabulary() {
    let cases = [
        (
            DimExpr::InputDim {
                input_idx: 2,
                axis: 0,
            }
            .eval(&[&[1usize]])
            .unwrap_err(),
            ShapeConstraintEvalError::MissingInput {
                input_idx: 2,
                input_count: 1,
            },
        ),
        (
            DimExpr::InputDim {
                input_idx: 0,
                axis: 2,
            }
            .eval(&[&[1usize]])
            .unwrap_err(),
            ShapeConstraintEvalError::AxisOutOfBounds {
                input_idx: 0,
                axis: 2,
                rank: 1,
            },
        ),
        (
            DimExpr::Add(
                Box::new(DimExpr::Const(usize::MAX)),
                Box::new(DimExpr::Const(1)),
            )
            .eval(&[])
            .unwrap_err(),
            ShapeConstraintEvalError::Overflow,
        ),
        (
            DimExpr::Mul(
                Box::new(DimExpr::Const(usize::MAX)),
                Box::new(DimExpr::Const(2)),
            )
            .eval(&[])
            .unwrap_err(),
            ShapeConstraintEvalError::Overflow,
        ),
        (
            DimExpr::Sub(Box::new(DimExpr::Const(0)), Box::new(DimExpr::Const(1)))
                .eval(&[])
                .unwrap_err(),
            ShapeConstraintEvalError::Underflow,
        ),
        (
            DimExpr::FloorDiv(Box::new(DimExpr::Const(1)), Box::new(DimExpr::Const(0)))
                .eval(&[])
                .unwrap_err(),
            ShapeConstraintEvalError::DivisionByZero,
        ),
    ];

    for (actual, expected) in cases {
        assert_eq!(ShapeConstraintEvalError::from(actual), expected);
    }

    assert_eq!(
        ShapeConstraintEvalError::from(DimExprEvalError::AddOverflow { lhs: 1, rhs: 2 }),
        ShapeConstraintEvalError::Overflow
    );
}

#[test]
fn constructors_preserve_classification_and_typed_sources() {
    let shape = Error::validation(
        "reshape",
        ErrorPhase::GraphBuild,
        ShapeMismatch::ExpectedActual {
            expected: ShapeVec::from_vec(vec![2, 3]),
            actual: ShapeVec::from_vec(vec![6]),
        }
        .into(),
    );
    assert_eq!(
        shape.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert_eq!(shape.phase(), Some(ErrorPhase::GraphBuild));

    let invalid = Error::invalid_argument("slice", ErrorPhase::Compile, "step", "must be non-zero");
    assert!(matches!(
        invalid,
        Error::Validation {
            source: ValidationError::InvalidArgument {
                argument: "step",
                ..
            },
            ..
        }
    ));

    for dtype in [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ] {
        let error = Error::dtype_mismatch("cast", ErrorPhase::GraphBuild, dtype, dtype);
        assert!(matches!(
            error,
            Error::Validation {
                source: ValidationError::DTypeMismatch { .. },
                ..
            }
        ));
    }

    let extension = Error::extension(
        "extension",
        ErrorPhase::Compile,
        "example.v1",
        ErrorKind::Io,
        std::io::Error::other("manifest read failed"),
    );
    assert_eq!(extension.kind(), ErrorKind::Io);
    assert!(StdError::source(&extension).is_some());

    let state = Error::runtime_state(
        "executor",
        ErrorPhase::Execution,
        "executor is not initialized",
    );
    assert_eq!(state.kind(), ErrorKind::RuntimeState);
    let state_source = Error::runtime_state_source(
        "registry",
        ErrorPhase::Compile,
        std::io::Error::other("registry lock poisoned"),
    );
    assert_eq!(state_source.kind(), ErrorKind::RuntimeState);
    assert!(StdError::source(&state_source).is_some());
    let unsupported = Error::unsupported(
        "compare",
        ErrorPhase::Compile,
        "complex values have no total order",
    );
    assert_eq!(unsupported.kind(), ErrorKind::Unsupported);
}

#[test]
fn kind_classifies_every_runtime_variant_without_string_inspection() {
    let errors = [
        (
            Error::validation(
                "shape",
                ErrorPhase::GraphBuild,
                ValidationError::RankMismatch {
                    expected: 2,
                    actual: 1,
                },
            ),
            ErrorKind::Validation(ValidationKind::RankMismatch),
        ),
        (Error::MissingInput("x".into()), ErrorKind::RuntimeState),
        (
            Error::NonScalarGrad { shape: vec![2] },
            ErrorKind::Validation(ValidationKind::InvalidArgument),
        ),
        (
            Error::unsupported("op", ErrorPhase::Compile, "missing rule"),
            ErrorKind::Unsupported,
        ),
        (
            Error::TensorRuntime(tenferro_tensor::Error::unsupported("op", "not available")),
            ErrorKind::Unsupported,
        ),
        (
            Error::extension(
                "op",
                ErrorPhase::Execution,
                "family.v1",
                ErrorKind::NumericalFailure,
                std::io::Error::other("numerical source"),
            ),
            ErrorKind::NumericalFailure,
        ),
        (
            Error::runtime_state("op", ErrorPhase::Execution, "state"),
            ErrorKind::RuntimeState,
        ),
        (
            Error::runtime_state_source(
                "op",
                ErrorPhase::Execution,
                std::io::Error::other("state"),
            ),
            ErrorKind::RuntimeState,
        ),
        (
            Error::UnexpectedBinding { binding_index: 0 },
            ErrorKind::RuntimeState,
        ),
        (
            Error::UnboundPlaceholder {
                input_key: "x".into(),
            },
            ErrorKind::RuntimeState,
        ),
        (
            Error::DuplicateBinding {
                input_key: "x".into(),
            },
            ErrorKind::RuntimeState,
        ),
        (
            Error::PlaceholderDtypeMismatch {
                expected: DType::F32,
                actual: DType::F64,
            },
            ErrorKind::Validation(ValidationKind::DTypeMismatch),
        ),
        (
            Error::PlaceholderShapeMismatch {
                expected: vec![2],
                actual: vec![3],
            },
            ErrorKind::Validation(ValidationKind::ShapeMismatch),
        ),
        (
            Error::PlaceholderRankMismatch {
                expected: 2,
                actual: 1,
            },
            ErrorKind::Validation(ValidationKind::RankMismatch),
        ),
        (
            Error::ContextMismatch {
                lhs: ContextId::fresh(),
                rhs: ContextId::fresh(),
            },
            ErrorKind::RuntimeState,
        ),
        (
            Error::UnsupportedAdRule {
                transform: "vjp",
                op: "example".into(),
            },
            ErrorKind::Unsupported,
        ),
        (
            Error::ShapeConstraintViolation {
                family: "example.v1",
                instruction_index: Some(3),
                relation: tenferro_ops::ShapeRelation::Equal,
                lhs_expr: "m".into(),
                rhs_expr: "n".into(),
                lhs_value: 2,
                rhs_value: 3,
            },
            ErrorKind::Validation(ValidationKind::ShapeMismatch),
        ),
        (
            Error::ShapeConstraintEvaluation {
                family: "example.v1",
                instruction_index: None,
                relation: tenferro_ops::ShapeRelation::Equal,
                expression: "m+n".into(),
                cause: ShapeConstraintEvalError::Overflow,
            },
            ErrorKind::Validation(ValidationKind::InvalidArgument),
        ),
        (
            Error::SymbolicShapeConversion {
                op: "broadcast",
                phase: ErrorPhase::GraphBuild,
                source: tenferro_ops::SymDimConversionError { tensor_id: 7 },
            },
            ErrorKind::Validation(ValidationKind::InvalidArgument),
        ),
        (
            Error::ShapeExpressionEvaluation {
                expression: "m/0".into(),
                cause: ShapeConstraintEvalError::DivisionByZero,
            },
            ErrorKind::Validation(ValidationKind::InvalidArgument),
        ),
        (Error::Internal("invariant".into()), ErrorKind::Internal),
    ];

    for (error, expected) in errors {
        assert_eq!(error.kind(), expected, "classified {error:?}");
    }
}

#[test]
fn phase_reports_discovery_axis_separately_from_kind() {
    let with_phase = [
        Error::validation(
            "op",
            ErrorPhase::GraphBuild,
            ValidationError::InvalidArgument {
                argument: "x",
                message: "bad".into(),
            },
        ),
        Error::TensorRuntime(tenferro_tensor::Error::invalid_argument("op", "x", "bad")),
        Error::unsupported("op", ErrorPhase::Compile, "unsupported"),
        Error::extension(
            "op",
            ErrorPhase::GraphBuild,
            "family.v1",
            ErrorKind::Internal,
            std::io::Error::other("extension"),
        ),
        Error::runtime_state("op", ErrorPhase::Execution, "state"),
        Error::runtime_state_source("op", ErrorPhase::Compile, std::io::Error::other("state")),
        Error::PlaceholderDtypeMismatch {
            expected: DType::F32,
            actual: DType::F64,
        },
        Error::PlaceholderShapeMismatch {
            expected: vec![2],
            actual: vec![3],
        },
        Error::PlaceholderRankMismatch {
            expected: 2,
            actual: 1,
        },
        Error::UnexpectedBinding { binding_index: 0 },
        Error::UnboundPlaceholder {
            input_key: "x".into(),
        },
        Error::DuplicateBinding {
            input_key: "x".into(),
        },
        Error::SymbolicShapeConversion {
            op: "op",
            phase: ErrorPhase::Compile,
            source: tenferro_ops::SymDimConversionError { tensor_id: 1 },
        },
        Error::ShapeExpressionEvaluation {
            expression: "m".into(),
            cause: ShapeConstraintEvalError::Overflow,
        },
    ];
    let expected = [
        Some(ErrorPhase::GraphBuild),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Compile),
        Some(ErrorPhase::GraphBuild),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Compile),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Execution),
        Some(ErrorPhase::Compile),
        Some(ErrorPhase::Execution),
    ];
    for (error, expected) in with_phase.into_iter().zip(expected) {
        assert_eq!(error.phase(), expected);
    }

    let without_phase = [
        Error::MissingInput("x".into()),
        Error::NonScalarGrad { shape: vec![2] },
        Error::ContextMismatch {
            lhs: ContextId::fresh(),
            rhs: ContextId::fresh(),
        },
        Error::UnsupportedAdRule {
            transform: "jvp",
            op: "example".into(),
        },
        Error::ShapeConstraintViolation {
            family: "example.v1",
            instruction_index: None,
            relation: tenferro_ops::ShapeRelation::Equal,
            lhs_expr: "m".into(),
            rhs_expr: "n".into(),
            lhs_value: 1,
            rhs_value: 2,
        },
        Error::ShapeConstraintEvaluation {
            family: "example.v1",
            instruction_index: None,
            relation: tenferro_ops::ShapeRelation::Equal,
            expression: "m".into(),
            cause: ShapeConstraintEvalError::Overflow,
        },
        Error::Internal("invariant".into()),
    ];
    for error in without_phase {
        assert_eq!(error.phase(), None);
    }
}

#[test]
fn context_ids_are_opaque_but_displayable() {
    let first = ContextId::fresh();
    let second = ContextId::fresh();

    assert_ne!(first, second);
    assert!(first.to_string().starts_with("ctx@"));
}

#[test]
fn suppressed_error_aggregate_delegates_primary_semantics() {
    let primary = Error::extension(
        "backend_mutation",
        ErrorPhase::Compile,
        "test.primary.v1",
        ErrorKind::Io,
        std::io::Error::other("primary source"),
    );
    let suppressed = Error::runtime_state_source(
        "runtime_reconciliation",
        ErrorPhase::Execution,
        std::io::Error::other("suppressed source"),
    );
    let primary_kind = primary.kind();
    let primary_phase = primary.phase();
    let primary_display = primary.to_string();

    let aggregate = Error::with_suppressed(primary, suppressed);

    assert_eq!(aggregate.kind(), primary_kind);
    assert_eq!(aggregate.phase(), primary_phase);
    let source = StdError::source(&aggregate).expect("primary error source");
    assert_eq!(source.to_string(), primary_display);
    assert!(StdError::source(source).is_some());

    let primary = aggregate.primary().expect("aggregate primary error");
    assert_eq!(primary.kind(), primary_kind);
    assert_eq!(primary.phase(), primary_phase);
    assert!(StdError::source(primary).is_some());

    let suppressed = aggregate
        .suppressed()
        .expect("typed suppressed error metadata");
    assert_eq!(suppressed.kind(), ErrorKind::RuntimeState);
    assert_eq!(suppressed.phase(), Some(ErrorPhase::Execution));
    assert!(matches!(suppressed, Error::RuntimeStateSource { .. }));
    assert!(StdError::source(suppressed).is_some());
}
