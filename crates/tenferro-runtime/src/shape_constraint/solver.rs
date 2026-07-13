// INVARIANT: Task 3 intentionally lands the solver before later compiler and
// executor tasks connect its crate-private entry points to production flow.
#![allow(dead_code)]

use std::{cmp::Ordering, collections::BTreeMap};

use tenferro_ops::{dim_expr::DimExpr, dim_expr::DimExprEvalError, ShapeRelation};

use super::{ConstraintSource, LocalShapeConstraint};
use crate::error::{Error, Result, ShapeConstraintEvalError};

type Symbol = (usize, usize);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ShapeGuard {
    pub(crate) source: ConstraintSource,
    pub(crate) relation: ShapeRelation,
    pub(crate) lhs: DimExpr,
    pub(crate) rhs: DimExpr,
}

impl ShapeGuard {
    pub(crate) fn evaluate(&self, inputs: &[&[usize]]) -> Result<()> {
        let lhs_value = self.evaluate_expression(&self.lhs, inputs)?;
        let rhs_value = self.evaluate_expression(&self.rhs, inputs)?;
        if lhs_value == rhs_value {
            return Ok(());
        }

        Err(violation(
            &self.source,
            self.relation,
            &self.lhs,
            &self.rhs,
            lhs_value,
            rhs_value,
        ))
    }

    fn evaluate_expression(&self, expression: &DimExpr, inputs: &[&[usize]]) -> Result<usize> {
        expression
            .eval(inputs)
            .map_err(|cause| Error::ShapeConstraintEvaluation {
                family: self.source.family_id,
                instruction_index: self.source.instruction_index,
                relation: self.relation,
                expression: format_expression(expression),
                cause: map_eval_error(cause),
            })
    }
}

#[derive(Default)]
struct SymbolSets {
    parents: BTreeMap<Symbol, Symbol>,
}

impl SymbolSets {
    fn ensure(&mut self, symbol: Symbol) {
        self.parents.entry(symbol).or_insert(symbol);
    }

    fn representative(&self, symbol: Symbol) -> Symbol {
        let mut current = symbol;
        loop {
            let parent = self.parents.get(&current).copied().unwrap_or(current);
            if parent == current {
                return current;
            }
            current = parent;
        }
    }

    fn union(&mut self, lhs: Symbol, rhs: Symbol) {
        self.ensure(lhs);
        self.ensure(rhs);
        let lhs_root = self.representative(lhs);
        let rhs_root = self.representative(rhs);
        if lhs_root == rhs_root {
            return;
        }
        let (representative, other) = if lhs_root < rhs_root {
            (lhs_root, rhs_root)
        } else {
            (rhs_root, lhs_root)
        };
        self.parents.insert(other, representative);
    }
}

pub(crate) fn discharge(constraints: Vec<LocalShapeConstraint>) -> Result<Vec<ShapeGuard>> {
    let mut sets = SymbolSets::default();
    for constraint in &constraints {
        if let (Some(lhs), Some(rhs)) = (as_symbol(&constraint.lhs), as_symbol(&constraint.rhs)) {
            sets.union(lhs, rhs);
        }
    }

    let mut pending_bindings = Vec::new();
    for constraint in &constraints {
        let binding = match (as_symbol(&constraint.lhs), as_const(&constraint.rhs)) {
            (Some(symbol), Some(value)) => Some((symbol, value)),
            _ => match (as_symbol(&constraint.rhs), as_const(&constraint.lhs)) {
                (Some(symbol), Some(value)) => Some((symbol, value)),
                _ => None,
            },
        };
        if let Some((symbol, value)) = binding {
            sets.ensure(symbol);
            pending_bindings.push((
                sets.representative(symbol),
                value,
                constraint.source.clone(),
            ));
        }
    }
    pending_bindings.sort_by(|lhs, rhs| {
        lhs.0
            .cmp(&rhs.0)
            .then_with(|| lhs.1.cmp(&rhs.1))
            .then_with(|| compare_source(&lhs.2, &rhs.2))
    });

    let mut bindings: BTreeMap<Symbol, usize> = BTreeMap::new();
    for (symbol, value, source) in pending_bindings {
        if let Some(bound) = bindings.get(&symbol).copied() {
            if bound != value {
                return Err(violation(
                    &source,
                    ShapeRelation::Equal,
                    &symbol_expr(symbol),
                    &DimExpr::Const(value),
                    bound,
                    value,
                ));
            }
        } else {
            bindings.insert(symbol, value);
        }
    }

    let mut guards = Vec::new();
    for constraint in constraints {
        let mut lhs = normalize_for_constraint(
            &constraint.lhs,
            &sets,
            &bindings,
            &constraint.source,
            constraint.relation,
        )?;
        let mut rhs = normalize_for_constraint(
            &constraint.rhs,
            &sets,
            &bindings,
            &constraint.source,
            constraint.relation,
        )?;
        if compare_expression(&lhs, &rhs).is_gt() {
            std::mem::swap(&mut lhs, &mut rhs);
        }

        if lhs == rhs {
            continue;
        }
        if let (DimExpr::Const(lhs_value), DimExpr::Const(rhs_value)) = (&lhs, &rhs) {
            return Err(violation(
                &constraint.source,
                constraint.relation,
                &lhs,
                &rhs,
                *lhs_value,
                *rhs_value,
            ));
        }
        guards.push(ShapeGuard {
            source: constraint.source,
            relation: constraint.relation,
            lhs,
            rhs,
        });
    }

    guards.sort_by(compare_guard);
    guards.dedup_by(|lhs, rhs| {
        lhs.relation == rhs.relation && lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs
    });
    Ok(guards)
}

fn normalize_for_constraint(
    expression: &DimExpr,
    sets: &SymbolSets,
    bindings: &BTreeMap<Symbol, usize>,
    source: &ConstraintSource,
    relation: ShapeRelation,
) -> Result<DimExpr> {
    normalize(expression, sets, bindings).map_err(|cause| Error::ShapeConstraintEvaluation {
        family: source.family_id,
        instruction_index: source.instruction_index,
        relation,
        expression: format_expression(expression),
        cause,
    })
}

fn normalize(
    expression: &DimExpr,
    sets: &SymbolSets,
    bindings: &BTreeMap<Symbol, usize>,
) -> std::result::Result<DimExpr, ShapeConstraintEvalError> {
    match expression {
        DimExpr::Const(value) => Ok(DimExpr::Const(*value)),
        DimExpr::InputDim { input_idx, axis } => {
            let representative = sets.representative((*input_idx, *axis));
            Ok(bindings
                .get(&representative)
                .copied()
                .map(DimExpr::Const)
                .unwrap_or_else(|| symbol_expr(representative)))
        }
        DimExpr::Add(lhs, rhs) => {
            let (lhs, rhs) = normalize_commutative(lhs, rhs, sets, bindings)?;
            match (&lhs, &rhs) {
                (DimExpr::Const(lhs), DimExpr::Const(rhs)) => lhs
                    .checked_add(*rhs)
                    .map(DimExpr::Const)
                    .ok_or(ShapeConstraintEvalError::Overflow),
                (DimExpr::Const(0), _) => Ok(rhs),
                (_, DimExpr::Const(0)) => Ok(lhs),
                _ => Ok(DimExpr::add(lhs, rhs)),
            }
        }
        DimExpr::Sub(lhs, rhs) => {
            let lhs = normalize(lhs, sets, bindings)?;
            let rhs = normalize(rhs, sets, bindings)?;
            match (&lhs, &rhs) {
                (DimExpr::Const(lhs), DimExpr::Const(rhs)) => lhs
                    .checked_sub(*rhs)
                    .map(DimExpr::Const)
                    .ok_or(ShapeConstraintEvalError::Underflow),
                (_, DimExpr::Const(0)) => Ok(lhs),
                _ => Ok(DimExpr::sub(lhs, rhs)),
            }
        }
        DimExpr::Mul(lhs, rhs) => {
            let (lhs, rhs) = normalize_commutative(lhs, rhs, sets, bindings)?;
            match (&lhs, &rhs) {
                (DimExpr::Const(lhs), DimExpr::Const(rhs)) => lhs
                    .checked_mul(*rhs)
                    .map(DimExpr::Const)
                    .ok_or(ShapeConstraintEvalError::Overflow),
                (DimExpr::Const(1), _) => Ok(rhs),
                (_, DimExpr::Const(1)) => Ok(lhs),
                _ => Ok(DimExpr::mul(lhs, rhs)),
            }
        }
        DimExpr::FloorDiv(lhs, rhs) => {
            let lhs = normalize(lhs, sets, bindings)?;
            let rhs = normalize(rhs, sets, bindings)?;
            match (&lhs, &rhs) {
                (DimExpr::Const(_), DimExpr::Const(0)) => {
                    Err(ShapeConstraintEvalError::DivisionByZero)
                }
                (DimExpr::Const(lhs), DimExpr::Const(rhs)) => Ok(DimExpr::Const(lhs / rhs)),
                _ => Ok(DimExpr::floor_div(lhs, rhs)),
            }
        }
        DimExpr::Min(lhs, rhs) => normalize_noncommutative(
            lhs,
            rhs,
            sets,
            bindings,
            |a, b| DimExpr::Const(a.min(b)),
            DimExpr::min,
        ),
        DimExpr::Max(lhs, rhs) => normalize_noncommutative(
            lhs,
            rhs,
            sets,
            bindings,
            |a, b| DimExpr::Const(a.max(b)),
            DimExpr::max,
        ),
    }
}

fn normalize_commutative(
    lhs: &DimExpr,
    rhs: &DimExpr,
    sets: &SymbolSets,
    bindings: &BTreeMap<Symbol, usize>,
) -> std::result::Result<(DimExpr, DimExpr), ShapeConstraintEvalError> {
    let mut lhs = normalize(lhs, sets, bindings)?;
    let mut rhs = normalize(rhs, sets, bindings)?;
    if compare_expression(&lhs, &rhs).is_gt() {
        std::mem::swap(&mut lhs, &mut rhs);
    }
    Ok((lhs, rhs))
}

fn normalize_noncommutative(
    lhs: &DimExpr,
    rhs: &DimExpr,
    sets: &SymbolSets,
    bindings: &BTreeMap<Symbol, usize>,
    fold: impl FnOnce(usize, usize) -> DimExpr,
    construct: impl FnOnce(DimExpr, DimExpr) -> DimExpr,
) -> std::result::Result<DimExpr, ShapeConstraintEvalError> {
    let lhs = normalize(lhs, sets, bindings)?;
    let rhs = normalize(rhs, sets, bindings)?;
    match (&lhs, &rhs) {
        (DimExpr::Const(lhs), DimExpr::Const(rhs)) => Ok(fold(*lhs, *rhs)),
        _ => Ok(construct(lhs, rhs)),
    }
}

fn compare_guard(lhs: &ShapeGuard, rhs: &ShapeGuard) -> Ordering {
    relation_rank(lhs.relation)
        .cmp(&relation_rank(rhs.relation))
        .then_with(|| compare_expression(&lhs.lhs, &rhs.lhs))
        .then_with(|| compare_expression(&lhs.rhs, &rhs.rhs))
        .then_with(|| compare_source(&lhs.source, &rhs.source))
}

fn compare_source(lhs: &ConstraintSource, rhs: &ConstraintSource) -> Ordering {
    lhs.family_id
        .cmp(rhs.family_id)
        .then_with(|| lhs.instruction_index.cmp(&rhs.instruction_index))
}

fn relation_rank(relation: ShapeRelation) -> u8 {
    match relation {
        ShapeRelation::Equal => 0,
    }
}

fn compare_expression(lhs: &DimExpr, rhs: &DimExpr) -> Ordering {
    expression_rank(lhs)
        .cmp(&expression_rank(rhs))
        .then_with(|| match (lhs, rhs) {
            (DimExpr::Const(lhs), DimExpr::Const(rhs)) => lhs.cmp(rhs),
            (
                DimExpr::InputDim {
                    input_idx: lhs_input,
                    axis: lhs_axis,
                },
                DimExpr::InputDim {
                    input_idx: rhs_input,
                    axis: rhs_axis,
                },
            ) => lhs_input
                .cmp(rhs_input)
                .then_with(|| lhs_axis.cmp(rhs_axis)),
            (DimExpr::Add(lhs_a, lhs_b), DimExpr::Add(rhs_a, rhs_b))
            | (DimExpr::Sub(lhs_a, lhs_b), DimExpr::Sub(rhs_a, rhs_b))
            | (DimExpr::Mul(lhs_a, lhs_b), DimExpr::Mul(rhs_a, rhs_b))
            | (DimExpr::FloorDiv(lhs_a, lhs_b), DimExpr::FloorDiv(rhs_a, rhs_b))
            | (DimExpr::Min(lhs_a, lhs_b), DimExpr::Min(rhs_a, rhs_b))
            | (DimExpr::Max(lhs_a, lhs_b), DimExpr::Max(rhs_a, rhs_b)) => {
                compare_expression(lhs_a, rhs_a).then_with(|| compare_expression(lhs_b, rhs_b))
            }
            _ => Ordering::Equal,
        })
}

fn expression_rank(expression: &DimExpr) -> u8 {
    match expression {
        DimExpr::Const(_) => 0,
        DimExpr::InputDim { .. } => 1,
        DimExpr::Add(_, _) => 2,
        DimExpr::Sub(_, _) => 3,
        DimExpr::Mul(_, _) => 4,
        DimExpr::FloorDiv(_, _) => 5,
        DimExpr::Min(_, _) => 6,
        DimExpr::Max(_, _) => 7,
    }
}

fn as_symbol(expression: &DimExpr) -> Option<Symbol> {
    match expression {
        DimExpr::InputDim { input_idx, axis } => Some((*input_idx, *axis)),
        _ => None,
    }
}

fn as_const(expression: &DimExpr) -> Option<usize> {
    match expression {
        DimExpr::Const(value) => Some(*value),
        _ => None,
    }
}

fn symbol_expr((input_idx, axis): Symbol) -> DimExpr {
    DimExpr::InputDim { input_idx, axis }
}

fn map_eval_error(error: DimExprEvalError) -> ShapeConstraintEvalError {
    match error {
        DimExprEvalError::InputOutOfBounds {
            input_idx,
            input_count,
        } => ShapeConstraintEvalError::MissingInput {
            input_idx,
            input_count,
        },
        DimExprEvalError::AxisOutOfBounds {
            input_idx,
            axis,
            rank,
        } => ShapeConstraintEvalError::AxisOutOfBounds {
            input_idx,
            axis,
            rank,
        },
        DimExprEvalError::AddOverflow { .. } | DimExprEvalError::MulOverflow { .. } => {
            ShapeConstraintEvalError::Overflow
        }
        DimExprEvalError::SubUnderflow { .. } => ShapeConstraintEvalError::Underflow,
        DimExprEvalError::FloorDivByZero { .. } => ShapeConstraintEvalError::DivisionByZero,
    }
}

fn format_expression(expression: &DimExpr) -> String {
    match expression {
        DimExpr::Const(value) => value.to_string(),
        DimExpr::InputDim { input_idx, axis } => format!("input[{input_idx}].shape[{axis}]"),
        DimExpr::Add(lhs, rhs) => format_binary(lhs, "+", rhs),
        DimExpr::Sub(lhs, rhs) => format_binary(lhs, "-", rhs),
        DimExpr::Mul(lhs, rhs) => format_binary(lhs, "*", rhs),
        DimExpr::FloorDiv(lhs, rhs) => format_binary(lhs, "/", rhs),
        DimExpr::Min(lhs, rhs) => format!(
            "min({}, {})",
            format_expression(lhs),
            format_expression(rhs)
        ),
        DimExpr::Max(lhs, rhs) => format!(
            "max({}, {})",
            format_expression(lhs),
            format_expression(rhs)
        ),
    }
}

fn format_binary(lhs: &DimExpr, operator: &str, rhs: &DimExpr) -> String {
    format!(
        "({} {operator} {})",
        format_expression(lhs),
        format_expression(rhs)
    )
}

fn violation(
    source: &ConstraintSource,
    relation: ShapeRelation,
    lhs: &DimExpr,
    rhs: &DimExpr,
    lhs_value: usize,
    rhs_value: usize,
) -> Error {
    Error::ShapeConstraintViolation {
        family: source.family_id,
        instruction_index: source.instruction_index,
        relation,
        lhs_expr: format_expression(lhs),
        rhs_expr: format_expression(rhs),
        lhs_value,
        rhs_value,
    }
}
