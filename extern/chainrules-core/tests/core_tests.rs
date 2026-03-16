//! Tests for chainrules-core: Differentiable impls, NodeId, SavePolicy,
//! AutodiffError construction and display, ReverseRule default HVP.

use std::hint::black_box;

use chainrules_core::{
    AdResult, AutodiffError, Differentiable, ForwardRule, NodeId, ReverseRule, SavePolicy,
};

#[inline(never)]
fn call_f32_zero_tangent(x: &f32) -> f32 {
    <f32 as Differentiable>::zero_tangent(x)
}

#[inline(never)]
fn call_f32_accumulate_tangent(a: f32, b: &f32) -> f32 {
    <f32 as Differentiable>::accumulate_tangent(a, b)
}

#[inline(never)]
fn call_f32_num_elements(x: &f32) -> usize {
    <f32 as Differentiable>::num_elements(x)
}

#[inline(never)]
fn call_f32_seed_cotangent(x: &f32) -> f32 {
    <f32 as Differentiable>::seed_cotangent(x)
}

// ============================================================================
// Differentiable for f64
// ============================================================================

#[test]
fn f64_zero_tangent() {
    assert_eq!(42.0_f64.zero_tangent(), 0.0_f64);
}

#[test]
fn f64_accumulate_tangent() {
    assert_eq!(f64::accumulate_tangent(1.5, &2.5), 4.0);
}

#[test]
fn f64_accumulate_with_zero() {
    let z = 5.0_f64.zero_tangent();
    assert_eq!(f64::accumulate_tangent(z, &3.0), 3.0);
}

// ============================================================================
// Differentiable for f64 (num_elements, seed_cotangent)
// ============================================================================

#[test]
fn f64_num_elements() {
    assert_eq!(42.0_f64.num_elements(), 1);
}

#[test]
fn f64_seed_cotangent() {
    assert_eq!(42.0_f64.seed_cotangent(), 1.0_f64);
}

// ============================================================================
// Differentiable for f32
// ============================================================================

#[test]
fn f32_zero_tangent() {
    let x = black_box(42.0_f32);
    assert_eq!(call_f32_zero_tangent(&x), 0.0_f32);
}

#[test]
fn f32_accumulate_tangent() {
    let lhs = black_box(1.5_f32);
    let rhs = black_box(2.5_f32);
    assert_eq!(call_f32_accumulate_tangent(lhs, &rhs), 4.0);
}

#[test]
fn f32_num_elements() {
    let x = black_box(42.0_f32);
    assert_eq!(call_f32_num_elements(&x), 1);
}

#[test]
fn f32_seed_cotangent() {
    let x = black_box(42.0_f32);
    assert_eq!(call_f32_seed_cotangent(&x), 1.0_f32);
}

// ============================================================================
// NodeId
// ============================================================================

#[test]
fn node_id_new_and_index() {
    let id = NodeId::new(7);
    assert_eq!(id.index(), 7);
}

#[test]
fn node_id_zero() {
    let id = NodeId::new(0);
    assert_eq!(id.index(), 0);
}

#[test]
fn node_id_equality() {
    assert_eq!(NodeId::new(3), NodeId::new(3));
}

#[test]
fn node_id_inequality() {
    assert_ne!(NodeId::new(1), NodeId::new(2));
}

#[test]
fn node_id_debug() {
    let id = NodeId::new(5);
    assert!(format!("{id:?}").contains('5'));
}

#[test]
fn node_id_clone() {
    let id = NodeId::new(10);
    let id2 = id;
    assert_eq!(id, id2);
}

#[test]
fn node_id_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(NodeId::new(1));
    set.insert(NodeId::new(2));
    set.insert(NodeId::new(1)); // duplicate
    assert_eq!(set.len(), 2);
}

// ============================================================================
// SavePolicy
// ============================================================================

#[test]
fn save_policy_equality() {
    assert_eq!(SavePolicy::SaveForPullback, SavePolicy::SaveForPullback);
    assert_eq!(
        SavePolicy::RecomputeOnPullback,
        SavePolicy::RecomputeOnPullback
    );
}

#[test]
fn save_policy_inequality() {
    assert_ne!(SavePolicy::SaveForPullback, SavePolicy::RecomputeOnPullback);
}

#[test]
fn save_policy_debug() {
    assert!(format!("{:?}", SavePolicy::SaveForPullback).contains("SaveForPullback"));
    assert!(format!("{:?}", SavePolicy::RecomputeOnPullback).contains("RecomputeOnPullback"));
}

// ============================================================================
// AutodiffError construction and display
// ============================================================================

#[test]
fn error_non_scalar_loss_display() {
    let err = AutodiffError::NonScalarLoss { num_elements: 8 };
    let msg = err.to_string();
    assert!(msg.contains("scalar"));
    assert!(msg.contains("8"));
}

#[test]
fn error_missing_node_display() {
    let err = AutodiffError::MissingNode;
    assert!(err.to_string().contains("not connected"));
}

#[test]
fn error_tangent_shape_mismatch_display() {
    let err = AutodiffError::TangentShapeMismatch {
        expected: "[2, 3]".into(),
        got: "[4, 5]".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("[2, 3]"));
    assert!(msg.contains("[4, 5]"));
}

#[test]
fn error_hvp_not_supported_display() {
    let err = AutodiffError::HvpNotSupported;
    assert!(err.to_string().contains("HVP"));
}

#[test]
fn error_mode_not_supported_display() {
    let err = AutodiffError::ModeNotSupported {
        mode: "frule".into(),
        reason: "tropical einsum supports rrule only".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("frule"));
    assert!(msg.contains("tropical"));
}

#[test]
fn error_mode_not_supported_create_graph_tangent_display() {
    let err = AutodiffError::ModeNotSupported {
        mode: "create_graph_tangent".into(),
        reason: "grad_tangent does not support create_graph".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("create_graph_tangent"));
    assert!(msg.contains("grad_tangent"));
}

#[test]
fn error_invalid_argument_display() {
    let err = AutodiffError::InvalidArgument("bad index".into());
    assert!(err.to_string().contains("bad index"));
}

#[test]
fn error_graph_freed_display() {
    let err = AutodiffError::GraphFreed;
    let msg = err.to_string();
    assert!(msg.contains("freed"));
}

// ============================================================================
// AdResult alias
// ============================================================================

#[test]
fn ad_result_ok() {
    let result: AdResult<i32> = Ok(42);
    assert!(matches!(result, Ok(42)));
}

#[test]
fn ad_result_err() {
    let result: AdResult<i32> = Err(AutodiffError::MissingNode);
    match result {
        Err(AutodiffError::MissingNode) => {}
        other => panic!("expected Err(MissingNode), got {other:?}"),
    }
}

// ============================================================================
// ReverseRule: default pullback_with_tangents returns HvpNotSupported
// ============================================================================

/// Minimal ReverseRule impl to test the default HVP method.
struct DummyRule;

impl ReverseRule<f64> for DummyRule {
    fn pullback(&self, _cotangent: &f64) -> AdResult<Vec<(NodeId, f64)>> {
        Ok(vec![(NodeId::new(0), 1.0)])
    }

    fn inputs(&self) -> Vec<NodeId> {
        vec![NodeId::new(0)]
    }
}

#[test]
fn reverse_rule_pullback_works() {
    let rule = DummyRule;
    let result = rule.pullback(&1.0).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, NodeId::new(0));
    assert_eq!(result[0].1, 1.0);
}

#[test]
fn reverse_rule_inputs() {
    let rule = DummyRule;
    assert_eq!(rule.inputs(), vec![NodeId::new(0)]);
}

#[test]
fn reverse_rule_default_hvp_returns_error() {
    let rule = DummyRule;
    let result = rule.pullback_with_tangents(&1.0, &1.0);
    assert!(result.is_err());
    match result.unwrap_err() {
        AutodiffError::HvpNotSupported => {}
        other => panic!("expected HvpNotSupported, got {other:?}"),
    }
}

// ============================================================================
// ForwardRule
// ============================================================================

/// Minimal ForwardRule impl for testing.
struct DummyFrule;

impl ForwardRule<f64> for DummyFrule {
    fn pushforward(&self, tangents: &[Option<&f64>]) -> AdResult<f64> {
        // Sum all provided tangents
        let mut result = 0.0;
        for t in tangents.iter().flatten() {
            result += **t;
        }
        Ok(result)
    }
}

#[test]
fn forward_rule_pushforward() {
    let rule = DummyFrule;
    let t1 = 1.5_f64;
    let t2 = 2.5_f64;
    let result = rule.pushforward(&[Some(&t1), Some(&t2)]).unwrap();
    assert_eq!(result, 4.0);
}

#[test]
fn forward_rule_pushforward_with_none() {
    let rule = DummyFrule;
    let t1 = 3.0_f64;
    let result = rule.pushforward(&[Some(&t1), None]).unwrap();
    assert_eq!(result, 3.0);
}
