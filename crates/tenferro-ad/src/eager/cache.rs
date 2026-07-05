use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem::{size_of, size_of_val};
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::CacheStats;
use tidu::eager::RecordedGraph;
use tidu::LinearizedGraph;

const DEFAULT_EAGER_AD_TRANSFORM_CACHE_CAPACITY: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EagerAdTransformCacheKey {
    recorded_graph_fingerprint: u64,
    output_slots: Vec<usize>,
}

impl EagerAdTransformCacheKey {
    pub(crate) fn new(graph: &RecordedGraph<StdTensorOp>, output_slots: &[usize]) -> Self {
        Self {
            recorded_graph_fingerprint: eager_recorded_graph_fingerprint(graph),
            output_slots: output_slots.to_vec(),
        }
    }
}

fn eager_recorded_graph_fingerprint(graph: &RecordedGraph<StdTensorOp>) -> u64 {
    let mut hasher = DefaultHasher::new();
    graph.input_keys().hash(&mut hasher);
    graph.output_keys().hash(&mut hasher);
    graph.as_graph().inputs().hash(&mut hasher);
    graph.as_graph().outputs().hash(&mut hasher);
    for value in graph.as_graph().values() {
        value.key.hash(&mut hasher);
        value.producer.hash(&mut hasher);
    }
    for op in graph.as_graph().operations() {
        op.operation.hash(&mut hasher);
        op.inputs.hash(&mut hasher);
        op.outputs.hash(&mut hasher);
        op.role.hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Debug)]
pub(crate) struct EagerAdTransformCache {
    entries: LruCache<EagerAdTransformCacheKey, Arc<LinearizedGraph<StdTensorOp>>>,
}

impl EagerAdTransformCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: LruCache::new(
                NonZeroUsize::new(DEFAULT_EAGER_AD_TRANSFORM_CACHE_CAPACITY)
                    .unwrap_or(NonZeroUsize::MIN),
            ),
        }
    }

    pub(crate) fn get(
        &mut self,
        key: &EagerAdTransformCacheKey,
    ) -> Option<Arc<LinearizedGraph<StdTensorOp>>> {
        self.entries.get(key).cloned()
    }

    pub(crate) fn put(
        &mut self,
        key: EagerAdTransformCacheKey,
        value: Arc<LinearizedGraph<StdTensorOp>>,
    ) {
        self.entries.put(key, value);
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    pub(crate) fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            retained_bytes: self
                .entries
                .iter()
                .map(|(key, linear)| eager_ad_transform_cache_entry_retained_bytes(key, linear))
                .sum(),
        }
    }
}

fn eager_ad_transform_cache_entry_retained_bytes(
    key: &EagerAdTransformCacheKey,
    linear: &Arc<LinearizedGraph<StdTensorOp>>,
) -> usize {
    size_of::<EagerAdTransformCacheKey>()
        + key.output_slots.capacity() * size_of::<usize>()
        + size_of::<Arc<LinearizedGraph<StdTensorOp>>>()
        + size_of::<LinearizedGraph<StdTensorOp>>()
        + size_of_val(linear.as_graph().values())
        + size_of_val(linear.as_graph().operations())
        + size_of_val(linear.tangent_inputs())
        + size_of_val(linear.tangent_outputs())
}
