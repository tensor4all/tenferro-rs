use computegraph::types::ValueKey;

use super::*;
use crate::traced::next_input_key;

fn scope() -> Arc<GlobalMetadataScope> {
    Arc::new(
        register_scoped_value_metadata(
            ValueKey::Input(next_input_key()),
            TensorMeta::exact(DType::F64, vec![]),
        )
        .unwrap(),
    )
}

#[test]
fn metadata_scope_merge_reuses_single_parent_chain() {
    let parent = MetadataScopeChain::with_scope(scope(), []);
    let merged = MetadataScopeChain::merge([&parent]);

    assert!(merged.shares_root(&parent));
}

#[test]
fn metadata_scope_materialization_deduplicates_shared_nodes_and_scopes() {
    let shared = MetadataScopeChain::with_scope(scope(), []);
    let repeated_scope = scope();
    let left = MetadataScopeChain::with_scope(Arc::clone(&repeated_scope), [&shared]);
    let right = MetadataScopeChain::with_scope(repeated_scope, [&shared]);
    let merged = MetadataScopeChain::merge([&left, &right]);

    let (scopes, visited_nodes) = merged.materialize_with_visit_count();
    assert_eq!(scopes.len(), 2);
    assert_eq!(visited_nodes, 4);
}

#[test]
fn metadata_scope_shared_history_materializes_in_linear_node_visits() {
    let depth = 512;
    let mut chain = MetadataScopeChain::with_scope(scope(), []);
    for _ in 0..depth {
        chain = MetadataScopeChain::merge([&chain, &chain]);
    }

    let (scopes, visited_nodes) = chain.materialize_with_visit_count();
    assert_eq!(scopes.len(), 1);
    assert_eq!(visited_nodes, depth + 1);
}
