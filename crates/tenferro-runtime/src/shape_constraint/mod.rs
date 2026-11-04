use tenferro_ops::{dim_expr::DimExpr, ShapeRelation};

mod solver;

pub(crate) use solver::discharge;
pub use solver::ShapeGuard;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ConstraintSource {
    pub(crate) family_id: &'static str,
    pub(crate) instruction_index: Option<usize>,
}

impl ConstraintSource {
    #[cfg(test)]
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

#[derive(Clone, Debug)]
pub(crate) struct SlotScopedShapeConstraint {
    pub(crate) origin_slots: Vec<usize>,
    pub(crate) input_slots: Vec<usize>,
    pub(crate) local: LocalShapeConstraint,
}

#[cfg(test)]
mod tests;
