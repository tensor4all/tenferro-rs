//! Tests for chainrules: Tape, TrackedTensor, DualTensor, Gradients,
//! PullbackPlan, and pullback with dummy operations.

use chainrules::{
    AdResult, AutodiffError, Differentiable, DualTensor, Gradients, NodeId, PullbackPlan,
    ReverseRule, Tape, TrackedTensor,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct ScalarBox(f64);

impl Differentiable for ScalarBox {
    type Tangent = Self;

    fn zero_tangent(&self) -> Self::Tangent {
        Self(0.0)
    }

    fn accumulate_tangent(a: Self::Tangent, b: &Self::Tangent) -> Self::Tangent {
        Self(a.0 + b.0)
    }

    fn num_elements(&self) -> usize {
        1
    }

    fn seed_cotangent(&self) -> Self::Tangent {
        Self(1.0)
    }
}

// ============================================================================
// Tape creation
// ============================================================================

#[test]
fn tape_new() {
    let tape = Tape::<f64>::new();
    // A fresh tape should be able to create leaves
    let x = tape.leaf(1.0);
    assert!(x.requires_grad());
    assert_eq!(*x.value(), 1.0);
}

#[test]
fn tape_default() {
    let tape = Tape::<f64>::default();
    // Default tape behaves the same as Tape::new
    let x = tape.leaf(2.0);
    assert!(x.requires_grad());
    assert_eq!(*x.value(), 2.0);
}

#[test]
fn tape_clone_shares_state() {
    let tape1 = Tape::<f64>::new();
    let tape2 = tape1.clone();
    // Both tapes share state: leaf on one is visible via pullback on the other
    let x = tape1.leaf(2.0);
    let grads = tape2.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}

// ============================================================================
// Tape::leaf
// ============================================================================

#[test]
fn leaf_requires_grad() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.14);
    assert!(x.requires_grad());
}

#[test]
fn leaf_has_node_id() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.14);
    assert!(x.node_id().is_some());
    assert_eq!(x.node_id().unwrap().index(), 0);
}

#[test]
fn leaf_sequential_node_ids() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y = tape.leaf(2.0);
    assert_eq!(x.node_id().unwrap().index(), 0);
    assert_eq!(y.node_id().unwrap().index(), 1);
}

#[test]
fn leaf_no_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    assert!(!x.has_tangent());
    assert!(x.tangent().is_none());
}

// ============================================================================
// Tape::leaf_with_tangent
// ============================================================================

#[test]
fn leaf_with_tangent_has_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(3.14, 1.0).unwrap();
    assert!(x.requires_grad());
    assert!(x.has_tangent());
    assert_eq!(*x.tangent().unwrap(), 1.0);
}

#[test]
fn leaf_with_tangent_has_node_id() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(3.14, 1.0).unwrap();
    assert!(x.node_id().is_some());
}

// ============================================================================
// TrackedTensor::new
// ============================================================================

#[test]
fn tracked_new_no_grad() {
    let x = TrackedTensor::new(42.0_f64);
    assert!(!x.requires_grad());
    assert!(x.node_id().is_none());
    assert!(!x.has_tangent());
}

#[test]
fn tracked_value() {
    let x = TrackedTensor::new(42.0_f64);
    assert_eq!(*x.value(), 42.0);
}

#[test]
fn tracked_into_value() {
    let x = TrackedTensor::new(42.0_f64);
    assert_eq!(x.into_value(), 42.0);
}

// ============================================================================
// TrackedTensor::detach
// ============================================================================

#[test]
fn detach_removes_grad() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.14);
    assert!(x.requires_grad());
    let d = x.detach();
    assert!(!d.requires_grad());
    assert!(d.node_id().is_none());
    assert!(!d.has_tangent());
    assert_eq!(*d.value(), 3.14);
}

#[test]
fn detach_removes_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(3.14, 1.0).unwrap();
    assert!(x.has_tangent());
    let d = x.detach();
    assert!(!d.has_tangent());
}

// ============================================================================
// DualTensor
// ============================================================================

#[test]
fn dual_new_no_tangent() {
    let x = DualTensor::new(3.14_f64);
    assert!(!x.has_tangent());
    assert_eq!(*x.primal(), 3.14);
}

#[test]
fn dual_with_tangent() {
    let x = DualTensor::with_tangent(3.14_f64, 1.0).unwrap();
    assert!(x.has_tangent());
    assert_eq!(*x.tangent().unwrap(), 1.0);
    assert_eq!(*x.primal(), 3.14);
}

#[test]
fn dual_into_parts() {
    let x = DualTensor::with_tangent(3.14_f64, 1.0).unwrap();
    let (p, t) = x.into_parts();
    assert_eq!(p, 3.14);
    assert_eq!(t, Some(1.0));
}

#[test]
fn dual_into_parts_no_tangent() {
    let x = DualTensor::new(3.14_f64);
    let (p, t) = x.into_parts();
    assert_eq!(p, 3.14);
    assert_eq!(t, None);
}

#[test]
fn dual_detach_tangent() {
    let x = DualTensor::with_tangent(3.14_f64, 1.0).unwrap();
    let c = x.detach_tangent();
    assert!(!c.has_tangent());
    assert_eq!(*c.primal(), 3.14);
}

// ============================================================================
// Gradients
// ============================================================================

#[test]
fn gradients_new_empty() {
    let grads = Gradients::<f64>::new();
    assert!(grads.entries().is_empty());
}

#[test]
fn gradients_default() {
    let grads = Gradients::<f64>::default();
    assert!(grads.entries().is_empty());
}

#[test]
fn gradients_get_missing() {
    let grads = Gradients::<f64>::new();
    assert!(grads.get(NodeId::new(0)).is_none());
}

#[test]
fn gradients_accumulate_insert() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 3.0).unwrap();
    assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 3.0);
}

#[test]
fn gradients_accumulate_adds() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 2.0).unwrap();
    grads.accumulate(NodeId::new(0), 3.0).unwrap();
    assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 5.0);
}

#[test]
fn gradients_accumulate_multiple_nodes() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 1.0).unwrap();
    grads.accumulate(NodeId::new(1), 2.0).unwrap();
    assert_eq!(*grads.get(NodeId::new(0)).unwrap(), 1.0);
    assert_eq!(*grads.get(NodeId::new(1)).unwrap(), 2.0);
}

#[test]
fn gradients_entries() {
    let mut grads = Gradients::<f64>::new();
    grads.accumulate(NodeId::new(0), 1.0).unwrap();
    grads.accumulate(NodeId::new(1), 2.0).unwrap();
    assert_eq!(grads.entries().len(), 2);
}

// ============================================================================
// Differentiable: num_elements and seed_cotangent for f64
// ============================================================================

#[test]
fn f64_num_elements() {
    assert_eq!(42.0_f64.num_elements(), 1);
}

#[test]
fn f64_seed_cotangent() {
    assert_eq!(42.0_f64.seed_cotangent(), 1.0_f64);
}

#[test]
fn f32_num_elements() {
    assert_eq!(42.0_f32.num_elements(), 1);
}

#[test]
fn f32_seed_cotangent() {
    assert_eq!(42.0_f32.seed_cotangent(), 1.0_f32);
}

// ============================================================================
// Pullback: leaf only (d(x)/d(x) = 1)
// ============================================================================

#[test]
fn pullback_leaf_identity() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let grads = tape.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}

#[test]
fn pullback_leaf_identity_custom_type() {
    let tape = Tape::<ScalarBox>::new();
    let x = tape.leaf(ScalarBox(2.0));
    let grads = tape.pullback(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), ScalarBox(1.0));
}

#[test]
fn pullback_missing_node_error() {
    let tape = Tape::<f64>::new();
    let x = TrackedTensor::new(2.0);
    let result = tape.pullback(&x);
    assert!(result.is_err());
    match result {
        Err(AutodiffError::MissingNode) => {}
        Err(other) => panic!("expected MissingNode, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

// ============================================================================
// Pullback with dummy operations
// ============================================================================

/// Rule: y = 2*x, so dy/dx = 2
struct MultiplyBy2Rule {
    input: NodeId,
}

impl ReverseRule<f64> for MultiplyBy2Rule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(self.input, cotangent * 2.0)])
    }
    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }
}

#[test]
fn pullback_single_op() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    // y = 2*x = 6, dy/dx = 2
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let grads = tape.pullback(&y).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 2.0);
}

#[test]
fn pullback_chain_of_ops() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    // y = 2*x
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    // z = 2*y = 4*x
    let z = tape.record_op(
        12.0,
        Box::new(MultiplyBy2Rule {
            input: y.node_id().unwrap(),
        }),
        None,
    );
    let grads = tape.pullback(&z).unwrap();
    // dz/dx = dz/dy * dy/dx = 2 * 2 = 4
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 4.0);
}

/// Rule: z = x + y, so dz/dx = 1, dz/dy = 1
struct AddRule {
    inputs: Vec<NodeId>,
}

impl ReverseRule<f64> for AddRule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(self.inputs.iter().map(|&id| (id, *cotangent)).collect())
    }
    fn inputs(&self) -> Vec<NodeId> {
        self.inputs.clone()
    }
}

#[test]
fn pullback_multi_input() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let y = tape.leaf(3.0);
    // z = x + y = 5
    let z = tape.record_op(
        5.0,
        Box::new(AddRule {
            inputs: vec![x.node_id().unwrap(), y.node_id().unwrap()],
        }),
        None,
    );
    let grads = tape.pullback(&z).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
    assert_eq!(*grads.get(y.node_id().unwrap()).unwrap(), 1.0);
}

/// Rule: y = x * x (same input used twice), so dy/dx = 2*x
/// Pullback: returns two entries for the same input, each cotangent * x
struct SquareRule {
    input: NodeId,
    saved_x: f64,
}

impl ReverseRule<f64> for SquareRule {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        // d(x*x)/dx = 2x, but expressed as two contributions of x each
        Ok(vec![
            (self.input, cotangent * self.saved_x),
            (self.input, cotangent * self.saved_x),
        ])
    }
    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }
}

#[test]
fn pullback_repeated_input() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    // y = x^2 = 9, dy/dx = 2*x = 6
    let y = tape.record_op(
        9.0,
        Box::new(SquareRule {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
        }),
        None,
    );
    let grads = tape.pullback(&y).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 6.0);
}

#[test]
fn pullback_diamond_graph() {
    // x -> y1 = 2*x, x -> y2 = 2*x, z = y1 + y2
    // dz/dx = dz/dy1 * dy1/dx + dz/dy2 * dy2/dx = 1*2 + 1*2 = 4
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y1 = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let y2 = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let z = tape.record_op(
        4.0,
        Box::new(AddRule {
            inputs: vec![y1.node_id().unwrap(), y2.node_id().unwrap()],
        }),
        None,
    );
    let grads = tape.pullback(&z).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 4.0);
}

#[test]
fn pullback_only_returns_leaf_grads() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let grads = tape.pullback(&y).unwrap();
    // Only leaf node (x) should have a gradient, not intermediate (y)
    assert_eq!(grads.entries().len(), 1);
    assert_eq!(grads.entries()[0].0, x.node_id().unwrap());
}

// ============================================================================
// PullbackPlan
// ============================================================================

#[test]
fn pullback_plan_build() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let plan = PullbackPlan::build(&x).unwrap();
    assert_eq!(plan.loss_node().index(), 0);
}

#[test]
fn pullback_plan_build_missing_node() {
    let x = TrackedTensor::new(2.0_f64);
    let result = PullbackPlan::build(&x);
    match result {
        Err(AutodiffError::MissingNode) => {}
        Err(other) => panic!("expected MissingNode, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn pullback_plan_execute() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(2.0);
    let plan = PullbackPlan::build(&x).unwrap();
    let grads = plan.execute(&x).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 1.0);
}

#[test]
fn pullback_plan_execute_with_op() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(3.0);
    let y = tape.record_op(
        6.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    let plan = PullbackPlan::build(&y).unwrap();
    let grads = plan.execute(&y).unwrap();
    assert_eq!(*grads.get(x.node_id().unwrap()).unwrap(), 2.0);
}

// ============================================================================
// HVP with dummy operation
// ============================================================================

/// Rule: y = x^2
/// pullback: dy = 2*x * dL
/// pullback_with_tangents:
///   cotangent of input = 2*x * dL (same as pullback)
///   cotangent tangent of input = 2*dx * dL + 2*x * dL_tangent
///   where dx is the tangent of x, dL is the cotangent, dL_tangent is cotangent tangent
struct SquareRuleHvp {
    input: NodeId,
    saved_x: f64,
    saved_dx: f64,
}

impl ReverseRule<f64> for SquareRuleHvp {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(self.input, 2.0 * self.saved_x * cotangent)])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![self.input]
    }

    fn pullback_with_tangents(
        &self,
        cotangent: &f64,
        cotangent_tangent: &f64,
    ) -> AdResult<Vec<(NodeId, f64, f64)>> {
        // grad = 2*x * cotangent
        let grad = 2.0 * self.saved_x * cotangent;
        // grad_tangent = 2*dx * cotangent + 2*x * cotangent_tangent
        let grad_tangent = 2.0 * self.saved_dx * cotangent + 2.0 * self.saved_x * cotangent_tangent;
        Ok(vec![(self.input, grad, grad_tangent)])
    }
}

#[test]
fn hvp_square_function() {
    // f(x) = x^2, ∇f = 2x, H = 2, Hv = 2v
    // x = 3.0, v = 1.0 (tangent direction)
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(3.0, 1.0).unwrap();
    // y = x^2 = 9
    let y = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
            saved_dx: 1.0, // tangent of x
        }),
        None, // output tangent (dy = 2*x*dx = 6) not needed for hvp traversal
    );
    let result = tape.hvp(&y).unwrap();
    // Gradient: d(x^2)/dx = 2*3 = 6
    assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 6.0);
    // HVP: H*v = 2*1 = 2
    assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 2.0);
}

/// Rule: z = a + b (HVP-aware addition)
/// pullback: dz/da = cotangent, dz/db = cotangent
/// pullback_with_tangents: pass cotangent and cotangent_tangent through unchanged
struct AddRuleHvp {
    inputs: Vec<NodeId>,
}

impl ReverseRule<f64> for AddRuleHvp {
    fn pullback(&self, cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(self.inputs.iter().map(|&id| (id, *cotangent)).collect())
    }

    fn inputs(&self) -> Vec<NodeId> {
        self.inputs.clone()
    }

    fn pullback_with_tangents(
        &self,
        cotangent: &f64,
        cotangent_tangent: &f64,
    ) -> AdResult<Vec<(NodeId, f64, f64)>> {
        Ok(self
            .inputs
            .iter()
            .map(|&id| (id, *cotangent, *cotangent_tangent))
            .collect())
    }
}

#[test]
fn hvp_dag_merge_point() {
    // f(x) = x^2 + x^2 = 2x^2
    // ∇f = 4x, H = 4, Hv = 4v
    // At x = 3.0, v = 1.0: gradient = 12, HVP = 4
    //
    // DAG:  x ──> y1 = x^2 ──> z = y1 + y2
    //       └──> y2 = x^2 ──┘
    //
    // During reverse traversal, x receives cotangent contributions from
    // both y1 and y2, hitting the Some(existing) accumulation branches
    // on lines 448 and 458 of lib.rs.
    let tape = Tape::<f64>::new();
    let x = tape.leaf_with_tangent(3.0, 1.0).unwrap();

    // y1 = x^2 = 9, tangent dy1 = 2*x*dx = 6
    let y1 = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
            saved_dx: 1.0,
        }),
        None,
    );
    // y2 = x^2 = 9, tangent dy2 = 2*x*dx = 6
    let y2 = tape.record_op(
        9.0,
        Box::new(SquareRuleHvp {
            input: x.node_id().unwrap(),
            saved_x: 3.0,
            saved_dx: 1.0,
        }),
        None,
    );
    // z = y1 + y2 = 18
    let z = tape.record_op(
        18.0,
        Box::new(AddRuleHvp {
            inputs: vec![y1.node_id().unwrap(), y2.node_id().unwrap()],
        }),
        None,
    );

    let result = tape.hvp(&z).unwrap();

    // Gradient: d(2x^2)/dx = 4*3 = 12
    assert_eq!(*result.gradients.get(x.node_id().unwrap()).unwrap(), 12.0);
    // HVP: H*v = 4*1 = 4
    assert_eq!(*result.hvp.get(x.node_id().unwrap()).unwrap(), 4.0);
}

#[test]
fn hvp_missing_node_error() {
    let tape = Tape::<f64>::new();
    let x = TrackedTensor::new(2.0_f64);
    let result = tape.hvp(&x);
    assert!(result.is_err());
    match result {
        Err(AutodiffError::MissingNode) => {}
        Err(other) => panic!("expected MissingNode, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

// ============================================================================
// record_op basics
// ============================================================================

#[test]
fn record_op_creates_tracked() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        None,
    );
    assert!(y.requires_grad());
    assert!(y.node_id().is_some());
    assert_eq!(y.node_id().unwrap().index(), 1);
    assert_eq!(*y.value(), 2.0);
}

#[test]
fn record_op_with_tangent() {
    let tape = Tape::<f64>::new();
    let x = tape.leaf(1.0);
    let y = tape.record_op(
        2.0,
        Box::new(MultiplyBy2Rule {
            input: x.node_id().unwrap(),
        }),
        Some(2.0), // output tangent
    );
    assert!(y.has_tangent());
    assert_eq!(*y.tangent().unwrap(), 2.0);
}
