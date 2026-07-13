use tenferro_ops::{dim_expr::DimExpr, ShapeRelation};

mod solver;

// INVARIANT: Task 3 defines the solver boundary before the compiler and
// executor integrations in later tasks consume these crate-private items.
#[allow(unused_imports)]
pub(crate) use solver::{discharge, ShapeGuard};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ConstraintSource {
    pub(crate) family_id: &'static str,
    pub(crate) instruction_index: Option<usize>,
}

impl ConstraintSource {
    // INVARIANT: graph recording assigns instruction provenance after local
    // inference; the transition remains unused until that pipeline stage.
    #[allow(dead_code)]
    pub(crate) fn with_instruction(mut self, instruction_index: usize) -> Self {
        self.instruction_index = Some(instruction_index);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LocalShapeConstraint {
    pub(crate) source: ConstraintSource,
    pub(crate) relation: ShapeRelation,
    pub(crate) lhs: DimExpr,
    pub(crate) rhs: DimExpr,
}

#[cfg(test)]
mod tests;
