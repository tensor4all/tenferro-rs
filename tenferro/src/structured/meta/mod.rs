mod planning;
mod types;

pub use planning::plan_axis_classes_for_subscripts;
pub use types::{AxisClassMergePlan, AxisClassPlanError, OperandAxisClassPlan, OperandAxisClasses};

#[cfg(test)]
mod tests;
