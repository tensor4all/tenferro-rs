use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use tenferro_ops::{dim_expr::DimExpr, ShapeRelation};

use super::solver::union_representatives_are_flat;
use super::{discharge, ConstraintSource, LocalShapeConstraint, ShapeGuard};
use crate::error::{ContextId, Error, ShapeConstraintEvalError};

fn source(family_id: &'static str, instruction_index: Option<usize>) -> ConstraintSource {
    ConstraintSource {
        family_id,
        instruction_index,
    }
}

fn symbol(input_idx: usize, axis: usize) -> DimExpr {
    DimExpr::InputDim { input_idx, axis }
}

fn equal(
    source: ConstraintSource,
    lhs: impl Into<DimExpr>,
    rhs: impl Into<DimExpr>,
) -> LocalShapeConstraint {
    LocalShapeConstraint {
        source,
        relation: ShapeRelation::Equal,
        lhs: lhs.into(),
        rhs: rhs.into(),
    }
}

fn only_guard(constraints: Vec<LocalShapeConstraint>) -> ShapeGuard {
    let mut guards = discharge(constraints).unwrap();
    assert_eq!(guards.len(), 1);
    guards.remove(0)
}

#[test]
fn constant_equal_is_proven_and_constant_unequal_is_typed() {
    let src = source("test.constants", Some(2));
    assert!(discharge(vec![equal(src.clone(), 4, 4)])
        .unwrap()
        .is_empty());

    assert!(matches!(
        discharge(vec![equal(src, 3, 4)]),
        Err(Error::ShapeConstraintViolation {
            family: "test.constants",
            instruction_index: Some(2),
            relation: ShapeRelation::Equal,
            lhs_value: 3,
            rhs_value: 4,
            ..
        })
    ));
}

#[test]
fn transitive_symbols_and_constant_binding_are_order_independent() {
    let a = symbol(2, 1);
    let b = symbol(0, 3);
    let c = symbol(1, 0);
    let constraints = vec![
        equal(source("test.transitive", Some(2)), b.clone(), c.clone()),
        equal(source("test.transitive", Some(0)), a.clone(), b.clone()),
        equal(source("test.transitive", Some(1)), c, 7),
    ];
    let mut reversed = constraints.clone();
    reversed.reverse();

    let guards = discharge(constraints).unwrap();
    let reversed_guards = discharge(reversed).unwrap();
    assert_eq!(guards, reversed_guards);
    assert_eq!(guards.len(), 3);
    for guard in &guards {
        guard.evaluate(&[&[7, 7, 7, 7], &[7], &[1, 7]]).unwrap();
    }
    assert!(guards
        .iter()
        .any(|guard| guard.evaluate(&[&[7, 7, 7, 8], &[7], &[1, 7]]).is_err()));
}

#[test]
fn representative_guard_order_and_source_choice_are_deterministic() {
    let a = symbol(3, 0);
    let smaller = symbol(0, 2);
    let z = symbol(4, 0);
    let duplicated_lhs = DimExpr::add(a.clone(), DimExpr::Const(1));
    let constraints = vec![
        equal(
            source("z-family", Some(9)),
            duplicated_lhs.clone(),
            z.clone(),
        ),
        equal(source("test.union", Some(4)), a, smaller.clone()),
        equal(
            source("a-family", Some(8)),
            DimExpr::add(smaller.clone(), DimExpr::Const(1)),
            z.clone(),
        ),
        equal(
            source("a-family", Some(3)),
            DimExpr::mul(smaller.clone(), DimExpr::Const(2)),
            z,
        ),
    ];
    let mut reversed = constraints.clone();
    reversed.reverse();

    let guards = discharge(constraints).unwrap();
    assert_eq!(guards, discharge(reversed).unwrap());
    assert_eq!(guards.len(), 3);
    assert!(guards.iter().any(|guard| {
        guard.source == source("a-family", Some(8))
            && guard.rhs == DimExpr::add(DimExpr::Const(1), smaller.clone())
    }));
    assert!(guards.iter().any(|guard| {
        guard.source == source("a-family", Some(3))
            && guard.rhs == DimExpr::mul(DimExpr::Const(2), smaller.clone())
    }));
    assert!(guards
        .iter()
        .any(|guard| guard.source == source("test.union", Some(4))));
}

#[test]
fn contradictory_symbol_bindings_report_concrete_values() {
    let a = symbol(0, 0);
    assert!(matches!(
        discharge(vec![
            equal(source("test.binding", Some(1)), a.clone(), 4),
            equal(source("test.binding", Some(0)), a, 3),
        ]),
        Err(Error::ShapeConstraintViolation {
            lhs_value: 3,
            rhs_value: 4,
            ..
        })
    ));
}

#[test]
fn scaled_equality_is_retained_and_checked_at_runtime() {
    let guard = only_guard(vec![equal(
        source("test.scaled", Some(5)),
        symbol(0, 0),
        DimExpr::mul(DimExpr::Const(2), symbol(1, 0)),
    )]);

    guard.evaluate(&[&[6], &[3]]).unwrap();
    assert!(matches!(
        guard.evaluate(&[&[7], &[3]]),
        Err(Error::ShapeConstraintViolation {
            family: "test.scaled",
            instruction_index: Some(5),
            lhs_value: 7,
            rhs_value: 6,
            ..
        })
    ));
}

#[test]
fn safe_identities_normalize_without_general_algebra() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let guards = discharge(vec![
        equal(
            source("test.identity", Some(2)),
            DimExpr::add(a.clone(), DimExpr::Const(0)),
            b.clone(),
        ),
        equal(
            source("test.identity", Some(1)),
            DimExpr::sub(a.clone(), DimExpr::Const(0)),
            b.clone(),
        ),
        equal(
            source("test.identity", Some(0)),
            DimExpr::mul(a.clone(), DimExpr::Const(1)),
            b.clone(),
        ),
    ])
    .unwrap();

    assert_eq!(guards.len(), 1);
    assert_eq!(guards[0].lhs, a);
    assert_eq!(guards[0].rhs, b);
    assert_eq!(guards[0].source, source("test.identity", Some(0)));
}

#[test]
fn multiplication_by_zero_does_not_hide_subtree_evaluation_failure() {
    let failing = DimExpr::floor_div(symbol(0, 0), DimExpr::Const(0));
    let guard = only_guard(vec![equal(
        source("test.zero", Some(7)),
        DimExpr::mul(failing, DimExpr::Const(0)),
        0,
    )]);

    assert!(matches!(
        guard.evaluate(&[&[9]]),
        Err(Error::ShapeConstraintEvaluation {
            cause: ShapeConstraintEvalError::DivisionByZero,
            ..
        })
    ));
}

#[test]
fn semantic_duplicate_guards_keep_deterministic_first_source() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let constraints = vec![
        equal(
            source("z-family", Some(9)),
            DimExpr::add(a.clone(), DimExpr::Const(2)),
            b.clone(),
        ),
        equal(
            source("a-family", Some(4)),
            DimExpr::add(DimExpr::Const(2), a),
            b,
        ),
    ];
    let mut reversed = constraints.clone();
    reversed.reverse();

    let guards = discharge(constraints).unwrap();
    assert_eq!(guards, discharge(reversed).unwrap());
    assert_eq!(guards.len(), 1);
    assert_eq!(guards[0].source, source("a-family", Some(4)));
}

#[test]
fn guard_evaluation_preserves_all_typed_causes_and_provenance() {
    let cases = [
        (
            DimExpr::min(symbol(2, 0), DimExpr::Const(usize::MAX)),
            Vec::<Vec<usize>>::new(),
            ShapeConstraintEvalError::MissingInput {
                input_idx: 2,
                input_count: 0,
            },
            format!("min({}, input[2].shape[0])", usize::MAX),
        ),
        (
            DimExpr::min(symbol(0, 2), DimExpr::Const(usize::MAX)),
            vec![vec![3]],
            ShapeConstraintEvalError::AxisOutOfBounds {
                input_idx: 0,
                axis: 2,
                rank: 1,
            },
            format!("min({}, input[0].shape[2])", usize::MAX),
        ),
        (
            DimExpr::add(symbol(0, 0), DimExpr::Const(usize::MAX)),
            vec![vec![1]],
            ShapeConstraintEvalError::Overflow,
            format!("({} + input[0].shape[0])", usize::MAX),
        ),
        (
            DimExpr::sub(symbol(0, 0), DimExpr::Const(2)),
            vec![vec![1]],
            ShapeConstraintEvalError::Underflow,
            "(input[0].shape[0] - 2)".into(),
        ),
        (
            DimExpr::floor_div(symbol(0, 0), DimExpr::Const(0)),
            vec![vec![1]],
            ShapeConstraintEvalError::DivisionByZero,
            "(input[0].shape[0] / 0)".into(),
        ),
    ];

    for (lhs, owned_inputs, expected_cause, expected_expr) in cases {
        let guard = only_guard(vec![equal(
            source("test.eval", Some(11)),
            lhs,
            DimExpr::Const(17),
        )]);
        let inputs: Vec<&[usize]> = owned_inputs.iter().map(Vec::as_slice).collect();
        match guard.evaluate(&inputs) {
            Err(Error::ShapeConstraintEvaluation {
                family,
                instruction_index,
                relation,
                expression,
                cause,
            }) => {
                assert_eq!(family, "test.eval");
                assert_eq!(instruction_index, Some(11));
                assert_eq!(relation, ShapeRelation::Equal);
                assert_eq!(expression, expected_expr);
                assert_eq!(cause, expected_cause);
            }
            result => panic!("unexpected evaluation result: {result:?}"),
        }
    }
}

#[test]
fn non_bare_expressions_remain_guards_without_inverse_solving() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let guards = discharge(vec![
        equal(
            source("test.no-algebra", Some(0)),
            DimExpr::add(a.clone(), DimExpr::Const(1)),
            4,
        ),
        equal(
            source("test.no-algebra", Some(1)),
            DimExpr::min(a, DimExpr::Const(8)),
            b,
        ),
    ])
    .unwrap();

    assert_eq!(guards.len(), 2);
    guards[0].evaluate(&[&[3], &[3]]).unwrap();
    guards[1].evaluate(&[&[3], &[3]]).unwrap();
}

#[test]
fn checked_constant_folding_handles_every_expression_kind() {
    let src = source("test.fold", Some(6));
    let proven = [
        (DimExpr::add(2.into(), 3.into()), 5),
        (DimExpr::sub(7.into(), 2.into()), 5),
        (DimExpr::mul(3.into(), 4.into()), 12),
        (DimExpr::floor_div(9.into(), 2.into()), 4),
        (DimExpr::min(3.into(), 5.into()), 3),
        (DimExpr::max(3.into(), 5.into()), 5),
    ];
    for (expression, expected) in proven {
        assert!(discharge(vec![equal(src.clone(), expression, expected)])
            .unwrap()
            .is_empty());
    }

    let failures = [
        (
            DimExpr::add(usize::MAX.into(), 1.into()),
            ShapeConstraintEvalError::Overflow,
        ),
        (
            DimExpr::sub(1.into(), 2.into()),
            ShapeConstraintEvalError::Underflow,
        ),
        (
            DimExpr::mul(usize::MAX.into(), 2.into()),
            ShapeConstraintEvalError::Overflow,
        ),
        (
            DimExpr::floor_div(1.into(), 0.into()),
            ShapeConstraintEvalError::DivisionByZero,
        ),
    ];
    for (expression, expected) in failures {
        assert!(matches!(
            discharge(vec![equal(src.clone(), expression, 0)]),
            Err(Error::ShapeConstraintEvaluation {
                family: "test.fold",
                instruction_index: Some(6),
                cause,
                ..
            }) if cause == expected
        ));
    }
}

#[test]
fn duplicate_union_and_reversed_duplicate_binding_are_stable() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let guards = discharge(vec![
        equal(source("test.duplicate", Some(0)), a.clone(), b.clone()),
        equal(source("test.duplicate", Some(1)), b, a.clone()),
        equal(source("test.duplicate", Some(2)), 3, a.clone()),
        equal(source("test.duplicate", Some(3)), a, 3),
    ])
    .unwrap();
    assert_eq!(guards.len(), 2);
    for guard in &guards {
        guard.evaluate(&[&[3], &[3]]).unwrap();
    }
    assert!(guards
        .iter()
        .any(|guard| guard.evaluate(&[&[4], &[4]]).is_err()));
    assert!(guards
        .iter()
        .any(|guard| guard.evaluate(&[&[3], &[4]]).is_err()));
}

#[test]
fn context_id_diagnostic_is_nonempty_and_unique() {
    let first = ContextId::fresh();
    let second = ContextId::fresh();
    assert_ne!(first, second);
    assert!(first.to_string().starts_with("ctx@"));
}

fn guard_hash(guards: &[ShapeGuard]) -> u64 {
    let mut hasher = DefaultHasher::new();
    guards.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn min_and_max_are_canonical_but_retain_validation_guards() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let guards = discharge(vec![
        equal(
            source("test.commutative", Some(0)),
            DimExpr::min(a.clone(), b.clone()),
            DimExpr::min(b.clone(), a.clone()),
        ),
        equal(
            source("test.commutative", Some(1)),
            DimExpr::max(a.clone(), b.clone()),
            DimExpr::max(b, a),
        ),
    ])
    .unwrap();
    assert_eq!(guards.len(), 2);
    for guard in guards {
        guard.evaluate(&[&[3], &[5]]).unwrap();
    }
}

#[test]
fn swapped_min_max_guards_dedup_with_deterministic_source_and_hash() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let c = symbol(2, 0);
    let constraints = vec![
        equal(
            source("z-family", Some(9)),
            DimExpr::min(a.clone(), b.clone()),
            c.clone(),
        ),
        equal(
            source("a-family", Some(7)),
            DimExpr::min(b.clone(), a.clone()),
            c.clone(),
        ),
        equal(
            source("z-family", Some(8)),
            DimExpr::max(a.clone(), b.clone()),
            c.clone(),
        ),
        equal(source("a-family", Some(6)), DimExpr::max(b, a), c),
    ];
    let mut reversed = constraints.clone();
    reversed.reverse();

    let guards = discharge(constraints).unwrap();
    let reversed_guards = discharge(reversed).unwrap();
    assert_eq!(guards, reversed_guards);
    assert_eq!(guard_hash(&guards), guard_hash(&reversed_guards));
    assert_eq!(guards.len(), 2);
    assert_eq!(guards[0].source, source("a-family", Some(7)));
    assert_eq!(guards[1].source, source("a-family", Some(6)));
}

#[test]
fn min_max_canonicalization_does_not_hide_operand_evaluation_errors() {
    let right_fails = only_guard(vec![equal(
        source("test.commutative-errors", Some(0)),
        DimExpr::min(
            symbol(0, 0),
            DimExpr::floor_div(symbol(1, 0), DimExpr::Const(0)),
        ),
        99,
    )]);
    let left_fails = only_guard(vec![equal(
        source("test.commutative-errors", Some(1)),
        DimExpr::max(
            DimExpr::floor_div(symbol(0, 0), DimExpr::Const(0)),
            symbol(1, 0),
        ),
        99,
    )]);

    for guard in [right_fails, left_fails] {
        assert!(matches!(
            guard.evaluate(&[&[4], &[7]]),
            Err(Error::ShapeConstraintEvaluation {
                cause: ShapeConstraintEvalError::DivisionByZero,
                ..
            })
        ));
    }
}

#[test]
fn bare_axis_equality_is_enforced_at_runtime() {
    let guard = only_guard(vec![equal(
        source("test.bare", Some(4)),
        symbol(0, 0),
        symbol(1, 0),
    )]);

    guard.evaluate(&[&[6], &[6]]).unwrap();
    assert!(matches!(
        guard.evaluate(&[&[6], &[7]]),
        Err(Error::ShapeConstraintViolation {
            family: "test.bare",
            instruction_index: Some(4),
            lhs_value: 6,
            rhs_value: 7,
            ..
        })
    ));
}

#[test]
fn binding_closure_does_not_drop_establishing_runtime_obligation() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let guards = discharge(vec![
        equal(source("test.binding-basis", Some(0)), a.clone(), 3),
        equal(
            source("test.binding-basis", Some(1)),
            DimExpr::add(a, DimExpr::Const(1)),
            b,
        ),
    ])
    .unwrap();

    assert_eq!(guards.len(), 2);
    for guard in &guards {
        guard.evaluate(&[&[3], &[4]]).unwrap();
    }
    assert!(guards
        .iter()
        .any(|guard| guard.evaluate(&[&[999], &[4]]).is_err()));
}

#[test]
fn unsafe_structural_identity_remains_a_validation_guard() {
    let cases = [
        (
            DimExpr::floor_div(symbol(0, 0), symbol(1, 0)),
            vec![vec![8], vec![0]],
            ShapeConstraintEvalError::DivisionByZero,
        ),
        (
            DimExpr::add(symbol(0, 0), DimExpr::Const(usize::MAX)),
            vec![vec![1]],
            ShapeConstraintEvalError::Overflow,
        ),
        (
            DimExpr::min(symbol(2, 0), DimExpr::Const(9)),
            Vec::<Vec<usize>>::new(),
            ShapeConstraintEvalError::MissingInput {
                input_idx: 2,
                input_count: 0,
            },
        ),
        (
            DimExpr::max(symbol(0, 2), DimExpr::Const(9)),
            vec![vec![1]],
            ShapeConstraintEvalError::AxisOutOfBounds {
                input_idx: 0,
                axis: 2,
                rank: 1,
            },
        ),
    ];

    for (expression, owned_inputs, expected) in cases {
        let guard = only_guard(vec![equal(
            source("test.structural-validation", Some(5)),
            expression.clone(),
            expression,
        )]);
        let inputs: Vec<&[usize]> = owned_inputs.iter().map(Vec::as_slice).collect();
        assert!(matches!(
            guard.evaluate(&inputs),
            Err(Error::ShapeConstraintEvaluation { cause, .. }) if cause == expected
        ));
    }
}

#[test]
fn independent_constant_violations_choose_same_error_for_every_order() {
    let first = equal(source("z-family", Some(9)), 3, 4);
    let second = equal(source("a-family", Some(1)), 1, 2);
    let forward = discharge(vec![first.clone(), second.clone()]).unwrap_err();
    let reversed = discharge(vec![second, first]).unwrap_err();

    assert_eq!(format!("{forward:?}"), format!("{reversed:?}"));
    assert!(matches!(
        forward,
        Error::ShapeConstraintViolation {
            family: "a-family",
            instruction_index: Some(1),
            lhs_value: 1,
            rhs_value: 2,
            ..
        }
    ));
}

#[test]
fn independent_constant_fold_failures_choose_same_error_for_every_order() {
    let overflow = equal(
        source("z-family", Some(9)),
        DimExpr::add(usize::MAX.into(), 1.into()),
        0,
    );
    let division = equal(
        source("a-family", Some(1)),
        DimExpr::floor_div(1.into(), 0.into()),
        0,
    );
    let forward = discharge(vec![overflow.clone(), division.clone()]).unwrap_err();
    let reversed = discharge(vec![division, overflow]).unwrap_err();

    assert_eq!(format!("{forward:?}"), format!("{reversed:?}"));
    assert!(matches!(
        forward,
        Error::ShapeConstraintEvaluation {
            family: "a-family",
            instruction_index: Some(1),
            cause: ShapeConstraintEvalError::DivisionByZero,
            ..
        }
    ));
}

#[test]
fn shape_evaluation_error_exposes_standard_source_chain() {
    let expression = DimExpr::floor_div(symbol(0, 0), DimExpr::Const(0));
    let guard = only_guard(vec![equal(source("test.source", Some(2)), expression, 1)]);
    let error = guard.evaluate(&[&[4]]).unwrap_err();
    let cause = std::error::Error::source(&error).expect("typed evaluation cause");
    assert_eq!(cause.to_string(), "shape expression divided by zero");
}

#[test]
fn swapped_mul_dedups_but_sub_and_floor_div_remain_ordered() {
    let a = symbol(0, 0);
    let b = symbol(1, 0);
    let c = symbol(2, 0);
    let mul_guards = discharge(vec![
        equal(
            source("z-family", Some(3)),
            DimExpr::mul(a.clone(), b.clone()),
            c.clone(),
        ),
        equal(
            source("a-family", Some(2)),
            DimExpr::mul(b.clone(), a.clone()),
            c,
        ),
    ])
    .unwrap();
    assert_eq!(mul_guards.len(), 1);
    assert_eq!(mul_guards[0].source, source("a-family", Some(2)));

    let sub_guard = only_guard(vec![equal(
        source("test.ordered", Some(0)),
        DimExpr::sub(a.clone(), b.clone()),
        DimExpr::sub(b.clone(), a.clone()),
    )]);
    assert_ne!(sub_guard.lhs, sub_guard.rhs);

    let div_guard = only_guard(vec![equal(
        source("test.ordered", Some(1)),
        DimExpr::floor_div(a, b.clone()),
        DimExpr::floor_div(b, symbol(0, 0)),
    )]);
    assert_ne!(div_guard.lhs, div_guard.rhs);
    assert!(matches!(
        div_guard.evaluate(&[&[8], &[2]]),
        Err(Error::ShapeConstraintViolation {
            lhs_value: 4,
            rhs_value: 0,
            ..
        })
    ));
}

#[test]
fn long_union_chains_flatten_to_the_minimum_representative() {
    let count = 4096;
    let descending: Vec<_> = (1..count)
        .rev()
        .map(|index| {
            equal(
                source("test.flatten", Some(index)),
                symbol(index, 0),
                symbol(index - 1, 0),
            )
        })
        .collect();
    let mut reversed = descending.clone();
    reversed.reverse();

    assert!(union_representatives_are_flat(&descending));
    assert!(union_representatives_are_flat(&reversed));
}
