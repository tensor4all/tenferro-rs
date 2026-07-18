use tenferro_tensor::{ErrorKind, ShapeMismatch, ValidationError, ValidationKind};

use crate::broadcast::{broadcast_error_to_validation, BroadcastError};

#[test]
fn broadcast_error_mapping_preserves_each_structured_category() {
    let cases = [
        (
            BroadcastError::IncompatibleBinary {
                lhs: vec![2, 3],
                rhs: vec![2, 4],
            },
            ValidationKind::ShapeMismatch,
        ),
        (
            BroadcastError::IncompatibleInput {
                input: vec![2, 3],
                output: vec![2, 4],
            },
            ValidationKind::ShapeMismatch,
        ),
        (
            BroadcastError::RankTooLarge {
                input: vec![2, 3],
                output: vec![3],
            },
            ValidationKind::RankMismatch,
        ),
    ];

    for (error, expected_kind) in cases {
        let validation = broadcast_error_to_validation(error);
        assert_eq!(validation.kind(), expected_kind);
        assert_eq!(
            ErrorKind::Validation(validation.kind()),
            ErrorKind::Validation(expected_kind)
        );
    }

    let incompatible_binary = broadcast_error_to_validation(BroadcastError::IncompatibleBinary {
        lhs: vec![2, 3],
        rhs: vec![2, 4],
    });
    assert!(matches!(
        incompatible_binary,
        ValidationError::ShapeMismatch(source)
            if matches!(source.as_ref(), ShapeMismatch::IncompatibleShapes { lhs, rhs }
                if lhs.as_slice() == [2, 3] && rhs.as_slice() == [2, 4])
    ));

    let incompatible_input = broadcast_error_to_validation(BroadcastError::IncompatibleInput {
        input: vec![2, 3],
        output: vec![2, 4],
    });
    assert!(matches!(
        incompatible_input,
        ValidationError::ShapeMismatch(source)
            if matches!(source.as_ref(), ShapeMismatch::ExpectedActual { expected, actual }
                if expected.as_slice() == [2, 4] && actual.as_slice() == [2, 3])
    ));

    assert!(matches!(
        broadcast_error_to_validation(BroadcastError::RankTooLarge {
            input: vec![2, 3],
            output: vec![3],
        }),
        ValidationError::RankMismatch {
            expected: 1,
            actual: 2,
        }
    ));
}
