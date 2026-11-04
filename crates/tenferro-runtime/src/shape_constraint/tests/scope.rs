use std::sync::Arc;

use computegraph::types::ValueKey;
use tenferro_ops::{dim_expr::DimExpr, ShapeRelation};

use super::super::{
    ConstraintScopeChain, ConstraintSource, LocalShapeConstraint, ScopedShapeConstraint,
    ShapeConstraintScope,
};

fn scope(_input: u64) -> Arc<ShapeConstraintScope> {
    let origin = crate::traced::next_input_key();
    let input = crate::traced::next_input_key();
    Arc::new(ShapeConstraintScope::new(vec![ScopedShapeConstraint {
        origins: vec![ValueKey::Input(origin)],
        inputs: vec![ValueKey::Input(input)],
        local: LocalShapeConstraint {
            source: ConstraintSource {
                family_id: "test.scope",
                instruction_index: None,
            },
            relation: ShapeRelation::Equal,
            lhs: DimExpr::InputDim {
                input_idx: 0,
                axis: 0,
            },
            rhs: DimExpr::Const(3),
        },
    }]))
}

#[test]
fn constraint_scope_shared_parent_is_pointer_deduplicated() {
    let shared = scope(10);
    let left = ConstraintScopeChain::with_scope(Arc::clone(&shared), []);
    let right = ConstraintScopeChain::with_scope(Arc::clone(&shared), []);
    let merged = ConstraintScopeChain::with_scope(scope(20), [&left, &right]);

    let materialized = merged.materialize();

    assert_eq!(materialized.len(), 2);
    assert!(Arc::ptr_eq(&materialized[1], &shared));
}

#[test]
fn constraint_scope_materialized_round_trip_preserves_pointer_identity_and_order() {
    let first = scope(30);
    let second = scope(40);
    let chain = ConstraintScopeChain::from_materialized(vec![
        Arc::clone(&first),
        Arc::clone(&second),
        Arc::clone(&first),
    ]);

    let materialized = chain.materialize();

    assert_eq!(materialized.len(), 2);
    assert!(Arc::ptr_eq(&materialized[0], &first));
    assert!(Arc::ptr_eq(&materialized[1], &second));
}

#[test]
fn constraint_scope_deep_shared_dag_visits_each_chain_node_once() {
    const DEPTH: usize = 12;
    let shared = scope(50);
    let mut chain = ConstraintScopeChain::with_scope(Arc::clone(&shared), []);
    for _ in 0..DEPTH {
        chain = ConstraintScopeChain::merge([&chain, &chain]);
    }

    let (materialized, visited_nodes) = chain.materialize_with_visit_count();

    assert_eq!(materialized.len(), 1);
    assert!(Arc::ptr_eq(&materialized[0], &shared));
    assert_eq!(visited_nodes, DEPTH + 1);
}
