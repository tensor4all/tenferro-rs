use chainrules_core::{AdResult, Differentiable};
use tidu::Schema;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointHint {
    CheapReplay,
    ExpensiveReplay,
}

pub trait LinearizableOp<V: Differentiable>: Send + Sync {
    type Linearized: LinearizedOp<V>;

    fn primal(&self, inputs: &[&V]) -> AdResult<Vec<V>>;
    fn input_schema(&self, inputs: &[&V]) -> AdResult<Schema>;
    fn output_schema(&self, inputs: &[&V], outputs: &[V]) -> AdResult<Schema>;
    fn linearize(&self, inputs: &[&V], outputs: &[V]) -> AdResult<Self::Linearized>;
    fn checkpoint_hint(&self) -> CheckpointHint;
}

pub trait LinearizedOp<V: Differentiable>: Send + Sync {
    fn jvp(&self, input_tangents: &[Option<V>]) -> AdResult<Vec<Option<V>>>;
    fn vjp(
        &self,
        output_cotangents: &[Option<V>],
        input_grad_mask: &[bool],
    ) -> AdResult<Vec<Option<V>>>;
}
