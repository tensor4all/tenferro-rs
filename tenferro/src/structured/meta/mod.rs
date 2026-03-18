mod planning;
mod types;

pub use planning::plan_axis_classes_for_subscripts;
#[cfg(test)]
pub use types::AxisClassPlanError;
pub use types::OperandAxisClasses;

#[cfg(test)]
mod tests;
