// INVARIANT: Task 3 intentionally lands the solver before later compiler and
// executor tasks connect its crate-private entry points to production flow.
#![allow(dead_code)]

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashSet},
};

use tenferro_ops::{dim_expr::DimExpr, dim_expr::DimExprEvalError, ShapeRelation};

use super::{ConstraintSource, LocalShapeConstraint};
use crate::error::{Error, Result, ShapeConstraintEvalError};

type Symbol = (usize, usize);

/// A normalized symbolic shape obligation retained by a compiled program.
///
/// Guards are produced by the compiler. Runtime evaluation is intentionally
/// owned by the execution pipeline rather than exposed as a user API.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ShapeGuard {
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

    fn flatten(&mut self) {
        let symbols: Vec<_> = self.parents.keys().copied().collect();
        for symbol in symbols {
            let representative = self.representative(symbol);
            self.parents.insert(symbol, representative);
        }
    }
}

/// One original bare-symbol declaration eligible for the enforcement basis.
#[derive(Clone)]
struct SymbolEdge {
    lhs: Symbol,
    rhs: Symbol,
    source: ConstraintSource,
}

/// A discharge failure plus its deterministic semantic-selection key.
struct ErrorCandidate {
    primary_expression: String,
    secondary_expression: String,
    source: ConstraintSource,
    kind: u8,
    error: Error,
}

pub(crate) fn discharge(constraints: Vec<LocalShapeConstraint>) -> Result<Vec<ShapeGuard>> {
    let mut sets = SymbolSets::default();
    let mut edges = Vec::new();
    for constraint in &constraints {
        if let (Some(lhs), Some(rhs)) = (as_symbol(&constraint.lhs), as_symbol(&constraint.rhs)) {
            sets.union(lhs, rhs);
            if lhs != rhs {
                let (lhs, rhs) = canonical_symbol_pair(lhs, rhs);
                edges.push(SymbolEdge {
                    lhs,
                    rhs,
                    source: constraint.source.clone(),
                });
            }
        }
    }
    sets.flatten();

    let mut basis_guards = spanning_tree_guards(edges);
    let mut pending_bindings = Vec::new();
    for constraint in &constraints {
        if let Some((symbol, value)) = symbol_binding(constraint) {
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

    let mut errors = Vec::new();
    let mut bindings = BTreeMap::new();
    let mut binding_index = 0;
    while binding_index < pending_bindings.len() {
        let symbol = pending_bindings[binding_index].0;
        let mut values: BTreeMap<usize, ConstraintSource> = BTreeMap::new();
        while binding_index < pending_bindings.len() && pending_bindings[binding_index].0 == symbol
        {
            let (_, value, source) = &pending_bindings[binding_index];
            values
                .entry(*value)
                .and_modify(|selected| {
                    if compare_source(source, selected).is_lt() {
                        *selected = source.clone();
                    }
                })
                .or_insert_with(|| source.clone());
            binding_index += 1;
        }

        let mut distinct_values = values.into_iter();
        let Some((first_value, first_source)) = distinct_values.next() else {
            continue;
        };
        bindings.insert(symbol, first_value);
        if let Some((second_value, second_source)) = distinct_values.next() {
            let source = if compare_source(&first_source, &second_source).is_le() {
                first_source
            } else {
                second_source
            };
            errors.push(violation_candidate(
                &source,
                ShapeRelation::Equal,
                &symbol_expr(symbol),
                &DimExpr::Const(second_value),
                first_value,
                second_value,
            ));
        } else {
            basis_guards.push(canonical_guard(
                first_source,
                ShapeRelation::Equal,
                symbol_expr(symbol),
                DimExpr::Const(first_value),
            ));
        }
    }

    let mut guards = Vec::new();
    for constraint in constraints {
        if is_basis_declaration(&constraint) {
            continue;
        }
        let lhs = normalize(&constraint.lhs, &sets, &bindings);
        let rhs = normalize(&constraint.rhs, &sets, &bindings);
        let lhs = match lhs {
            Ok(lhs) => Some(lhs),
            Err(cause) => {
                errors.push(evaluation_candidate(
                    &constraint.source,
                    constraint.relation,
                    &constraint.lhs,
                    cause,
                ));
                None
            }
        };
        let rhs = match rhs {
            Ok(rhs) => Some(rhs),
            Err(cause) => {
                errors.push(evaluation_candidate(
                    &constraint.source,
                    constraint.relation,
                    &constraint.rhs,
                    cause,
                ));
                None
            }
        };
        let (Some(mut lhs), Some(mut rhs)) = (lhs, rhs) else {
            continue;
        };
        if lhs == rhs {
            if is_statically_total(&lhs) {
                continue;
            }
            guards.push(ShapeGuard {
                source: constraint.source,
                relation: constraint.relation,
                lhs,
                rhs,
            });
            continue;
        }
        if let (DimExpr::Const(lhs_value), DimExpr::Const(rhs_value)) = (&lhs, &rhs) {
            errors.push(violation_candidate(
                &constraint.source,
                constraint.relation,
                &lhs,
                &rhs,
                *lhs_value,
                *rhs_value,
            ));
            continue;
        }
        if compare_expression(&lhs, &rhs).is_gt() {
            std::mem::swap(&mut lhs, &mut rhs);
        }
        guards.push(ShapeGuard {
            source: constraint.source,
            relation: constraint.relation,
            lhs,
            rhs,
        });
    }

    if !errors.is_empty() {
        errors.sort_by(compare_error_candidate);
        return Err(errors.remove(0).error);
    }

    sort_and_dedup_guards(&mut basis_guards);
    sort_and_dedup_guards(&mut guards);
    let basis_keys: HashSet<_> = basis_guards
        .iter()
        .map(|guard| (guard.relation, guard.lhs.clone(), guard.rhs.clone()))
        .collect();
    guards.retain(|guard| {
        !basis_keys.contains(&(guard.relation, guard.lhs.clone(), guard.rhs.clone()))
    });
    basis_guards.extend(guards);
    basis_guards.sort_by(compare_guard);
    Ok(basis_guards)
}

fn spanning_tree_guards(mut edges: Vec<SymbolEdge>) -> Vec<ShapeGuard> {
    // Kruskal over structurally ordered declarations retains exactly one
    // original-source edge per component merge, independent of caller order.
    edges.sort_by(|lhs, rhs| {
        lhs.lhs
            .cmp(&rhs.lhs)
            .then_with(|| lhs.rhs.cmp(&rhs.rhs))
            .then_with(|| compare_source(&lhs.source, &rhs.source))
    });
    let mut basis_sets = SymbolSets::default();
    let mut guards = Vec::new();
    for edge in edges {
        if basis_sets.representative(edge.lhs) == basis_sets.representative(edge.rhs) {
            continue;
        }
        basis_sets.union(edge.lhs, edge.rhs);
        guards.push(canonical_guard(
            edge.source,
            ShapeRelation::Equal,
            symbol_expr(edge.lhs),
            symbol_expr(edge.rhs),
        ));
    }
    guards
}

fn canonical_guard(
    source: ConstraintSource,
    relation: ShapeRelation,
    mut lhs: DimExpr,
    mut rhs: DimExpr,
) -> ShapeGuard {
    if compare_expression(&lhs, &rhs).is_gt() {
        std::mem::swap(&mut lhs, &mut rhs);
    }
    ShapeGuard {
        source,
        relation,
        lhs,
        rhs,
    }
}

fn symbol_binding(constraint: &LocalShapeConstraint) -> Option<(Symbol, usize)> {
    match (as_symbol(&constraint.lhs), as_const(&constraint.rhs)) {
        (Some(symbol), Some(value)) => Some((symbol, value)),
        _ => match (as_symbol(&constraint.rhs), as_const(&constraint.lhs)) {
            (Some(symbol), Some(value)) => Some((symbol, value)),
            _ => None,
        },
    }
}

fn is_basis_declaration(constraint: &LocalShapeConstraint) -> bool {
    match (as_symbol(&constraint.lhs), as_symbol(&constraint.rhs)) {
        (Some(lhs), Some(rhs)) if lhs != rhs => true,
        _ => symbol_binding(constraint).is_some(),
    }
}

fn canonical_symbol_pair(lhs: Symbol, rhs: Symbol) -> (Symbol, Symbol) {
    if lhs <= rhs {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    }
}

fn is_statically_total(expression: &DimExpr) -> bool {
    // InputDim can fail bounds validation, and every compound expression can
    // contain one. Normalization folds fully constant trees to Const first.
    matches!(expression, DimExpr::Const(_))
}

fn sort_and_dedup_guards(guards: &mut Vec<ShapeGuard>) {
    guards.sort_by(compare_guard);
    guards.dedup_by(|lhs, rhs| {
        lhs.relation == rhs.relation && lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs
    });
}

fn evaluation_candidate(
    source: &ConstraintSource,
    relation: ShapeRelation,
    expression: &DimExpr,
    cause: ShapeConstraintEvalError,
) -> ErrorCandidate {
    let formatted = format_expression(expression);
    ErrorCandidate {
        primary_expression: formatted.clone(),
        secondary_expression: String::new(),
        source: source.clone(),
        kind: 0,
        error: Error::ShapeConstraintEvaluation {
            family: source.family_id,
            instruction_index: source.instruction_index,
            relation,
            expression: formatted,
            cause,
        },
    }
}

fn violation_candidate(
    source: &ConstraintSource,
    relation: ShapeRelation,
    lhs: &DimExpr,
    rhs: &DimExpr,
    lhs_value: usize,
    rhs_value: usize,
) -> ErrorCandidate {
    let lhs_expression = format_expression(lhs);
    let rhs_expression = format_expression(rhs);
    let (primary_expression, secondary_expression) = if lhs_expression <= rhs_expression {
        (lhs_expression.clone(), rhs_expression.clone())
    } else {
        (rhs_expression.clone(), lhs_expression.clone())
    };
    ErrorCandidate {
        primary_expression,
        secondary_expression,
        source: source.clone(),
        kind: 1,
        error: Error::ShapeConstraintViolation {
            family: source.family_id,
            instruction_index: source.instruction_index,
            relation,
            lhs_expr: lhs_expression,
            rhs_expr: rhs_expression,
            lhs_value,
            rhs_value,
        },
    }
}

fn compare_error_candidate(lhs: &ErrorCandidate, rhs: &ErrorCandidate) -> Ordering {
    lhs.primary_expression
        .cmp(&rhs.primary_expression)
        .then_with(|| lhs.secondary_expression.cmp(&rhs.secondary_expression))
        .then_with(|| compare_source(&lhs.source, &rhs.source))
        .then_with(|| lhs.kind.cmp(&rhs.kind))
}

#[cfg(test)]
pub(super) fn union_representatives_are_flat(constraints: &[LocalShapeConstraint]) -> bool {
    let mut sets = SymbolSets::default();
    for constraint in constraints {
        if let (Some(lhs), Some(rhs)) = (as_symbol(&constraint.lhs), as_symbol(&constraint.rhs)) {
            sets.union(lhs, rhs);
        }
    }
    sets.flatten();
    sets.parents
        .iter()
        .all(|(symbol, parent)| *parent == sets.representative(*symbol))
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
        DimExpr::Min(lhs, rhs) => normalize_commutative_expression(
            lhs,
            rhs,
            sets,
            bindings,
            |a, b| DimExpr::Const(a.min(b)),
            DimExpr::min,
        ),
        DimExpr::Max(lhs, rhs) => normalize_commutative_expression(
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

fn normalize_commutative_expression(
    lhs: &DimExpr,
    rhs: &DimExpr,
    sets: &SymbolSets,
    bindings: &BTreeMap<Symbol, usize>,
    fold: impl FnOnce(usize, usize) -> DimExpr,
    construct: impl FnOnce(DimExpr, DimExpr) -> DimExpr,
) -> std::result::Result<DimExpr, ShapeConstraintEvalError> {
    let (lhs, rhs) = normalize_commutative(lhs, rhs, sets, bindings)?;
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
