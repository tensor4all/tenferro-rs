use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::hash::Hash;

use crate::{
    AnalyticPrimsDescriptor, ScalarPrimsDescriptor, SemiringCoreDescriptor,
    SemiringFastPathDescriptor,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum PlanCacheDescriptor {
    SemiringCore(SemiringCoreDescriptor),
    SemiringFastPath(SemiringFastPathDescriptor),
    Scalar(ScalarPrimsDescriptor),
    Analytic(AnalyticPrimsDescriptor),
}

pub(crate) trait CacheDescriptor: Clone + Eq + Hash + 'static {
    fn into_cache_descriptor(self) -> PlanCacheDescriptor;
}

impl CacheDescriptor for SemiringCoreDescriptor {
    fn into_cache_descriptor(self) -> PlanCacheDescriptor {
        PlanCacheDescriptor::SemiringCore(self)
    }
}

impl CacheDescriptor for SemiringFastPathDescriptor {
    fn into_cache_descriptor(self) -> PlanCacheDescriptor {
        PlanCacheDescriptor::SemiringFastPath(self)
    }
}

impl CacheDescriptor for ScalarPrimsDescriptor {
    fn into_cache_descriptor(self) -> PlanCacheDescriptor {
        PlanCacheDescriptor::Scalar(self)
    }
}

impl CacheDescriptor for AnalyticPrimsDescriptor {
    fn into_cache_descriptor(self) -> PlanCacheDescriptor {
        PlanCacheDescriptor::Analytic(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PlanCacheKey {
    plan_type_id: TypeId,
    descriptor_type_id: TypeId,
    descriptor: PlanCacheDescriptor,
    shapes: Vec<Vec<usize>>,
}

impl PlanCacheKey {
    fn new<P: 'static, D: CacheDescriptor>(desc: &D, shapes: &[&[usize]]) -> Self {
        Self {
            plan_type_id: TypeId::of::<P>(),
            descriptor_type_id: TypeId::of::<D>(),
            descriptor: desc.clone().into_cache_descriptor(),
            shapes: shapes.iter().map(|shape| shape.to_vec()).collect(),
        }
    }
}

/// Cache for pre-computed execution plans.
///
/// The cache is keyed by plan type, descriptor family, descriptor value, and
/// concrete tensor shapes. It is intentionally family-aware so the public
/// primitive protocol can stay split into focused traits without reintroducing
/// a monolithic descriptor surface.
///
/// # Examples
///
/// ```
/// use tenferro_prims::PlanCache;
///
/// let cache = PlanCache::new();
/// assert!(cache.is_empty());
/// ```
pub struct PlanCache {
    entries: HashMap<PlanCacheKey, Box<dyn Any>>,
}

impl PlanCache {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn get<P, D>(&self, desc: &D, shapes: &[&[usize]]) -> Option<P>
    where
        P: Clone + 'static,
        D: CacheDescriptor,
    {
        let key = PlanCacheKey::new::<P, D>(desc, shapes);
        self.entries
            .get(&key)
            .and_then(|boxed| boxed.downcast_ref::<P>())
            .cloned()
    }

    pub(crate) fn insert<P, D>(&mut self, desc: &D, shapes: &[&[usize]], plan: P)
    where
        P: Clone + 'static,
        D: CacheDescriptor,
    {
        let key = PlanCacheKey::new::<P, D>(desc, shapes);
        self.entries.insert(key, Box::new(plan));
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}
