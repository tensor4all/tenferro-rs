use std::mem::size_of;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock};

use lru::LruCache;
use tenferro_tensor::DType;

use crate::{ConcreteEinsumPlan, EinsumAxis, EinsumNotation, EinsumSubscripts};

/// Stable family identifier for the standard tenferro einsum extension.
pub const EINSUM_EXTENSION_FAMILY_ID: &str = "tenferro.einsum.v1";

/// Compiler-side subscript parse cache name.
pub(crate) const EINSUM_PARSE_CACHE: &str = "parse";
/// Executor-side runtime contraction-plan cache name.
pub(crate) const EINSUM_RUNTIME_PLANS_CACHE: &str = "runtime_plans";
/// EagerTensor expanded standard-op program cache name.
#[cfg(feature = "autodiff")]
pub(crate) const EINSUM_EAGER_EXPANDED_PROGRAMS_CACHE: &str = "eager_expanded_programs";

/// Process-wide bound on retained prepared plans.
///
/// A prepared plan is small (a contraction tree plus one spec per operand), so
/// this bound is about keeping the cache from growing without limit across
/// unrelated call sites, not about bytes.
const MAX_CACHED_PLANS: usize = 64;

/// Identity of a prepared concrete einsum plan.
///
/// A plan depends on the operand subscript columns, each operand's dtype, rank,
/// layout and the output specification. The layout columns are recorded because
/// the key contract requires them; the planner itself is stride-independent, so
/// a layout difference costs a cache miss but never a wrong plan.
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct PlanCacheKey {
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    operands: Vec<(DType, Vec<usize>, Vec<isize>)>,
}

impl PlanCacheKey {
    pub(crate) fn new(
        subscripts: &EinsumSubscripts,
        operands: Vec<(DType, Vec<usize>, Vec<isize>)>,
    ) -> Self {
        Self {
            inputs: subscripts.inputs.clone(),
            output: subscripts.output.clone(),
            operands,
        }
    }
}

fn plan_cache() -> &'static Mutex<LruCache<PlanCacheKey, Arc<ConcreteEinsumPlan>>> {
    static CACHE: OnceLock<Mutex<LruCache<PlanCacheKey, Arc<ConcreteEinsumPlan>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(MAX_CACHED_PLANS).unwrap_or(NonZeroUsize::MIN),
        ))
    })
}

/// Return the prepared plan identified by `key`, building and retaining it on a
/// miss.
///
/// Callers reach this through `&self`-shaped sessions, so the cache is a
/// process-wide mutex-guarded LRU rather than engine state; the lock is never
/// held across `build`, so planning does not serialize unrelated callers.
pub(crate) fn cached_plan(
    key: PlanCacheKey,
    build: impl FnOnce() -> crate::Result<ConcreteEinsumPlan>,
) -> crate::Result<Arc<ConcreteEinsumPlan>> {
    if let Some(plan) = plan_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
    {
        return Ok(Arc::clone(plan));
    }
    let plan = Arc::new(build()?);
    plan_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .put(key, Arc::clone(&plan));
    Ok(plan)
}

/// Parsed einsum notation retained by parse caches.
pub(crate) struct ParsedEinsum {
    /// Parsed rank-unresolved notation.
    pub(crate) notation: EinsumNotation,
}

/// Return the retained-byte estimate for canonical subscripts.
#[must_use]
pub(crate) fn einsum_subscripts_retained_bytes(subscripts: &EinsumSubscripts) -> usize {
    saturating_sum([
        vec_of_vec_retained_bytes(&subscripts.inputs),
        vec_retained_bytes(&subscripts.output),
    ])
}

pub(crate) fn einsum_notation_retained_bytes(notation: &EinsumNotation) -> usize {
    saturating_sum([
        vec_of_vec_retained_bytes(&notation.inputs),
        vec_retained_bytes(&notation.output),
        (notation.inputs.iter().map(Vec::capacity).sum::<usize>() + notation.output.capacity())
            .saturating_mul(std::mem::size_of::<EinsumAxis>()),
    ])
}

pub(crate) fn vec_retained_bytes<T>(values: &Vec<T>) -> usize {
    values.capacity().saturating_mul(size_of::<T>())
}

pub(crate) fn vec_of_vec_retained_bytes<T>(values: &[Vec<T>]) -> usize {
    saturating_sum(values.iter().map(vec_retained_bytes))
}

pub(crate) fn saturating_sum(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().fold(0usize, usize::saturating_add)
}

#[cfg(test)]
mod tests {
    use super::saturating_sum;

    #[test]
    fn retained_byte_sums_saturate() {
        assert_eq!(saturating_sum([usize::MAX, 1]), usize::MAX);
        assert_eq!(saturating_sum([usize::MAX - 4, 2, 8]), usize::MAX);
    }
}

#[cfg(test)]
mod plan_cache_tests {
    use super::{cached_plan, PlanCacheKey};
    use crate::{ConcreteEinsumPlan, EinsumSubscripts};
    use tenferro_cpu::CpuBackend;
    use tenferro_tensor::{BackendSessionHost, DType, Tensor, TensorRead};

    fn key(subscripts: &EinsumSubscripts, inputs: &[&Tensor]) -> PlanCacheKey {
        PlanCacheKey::new(
            subscripts,
            inputs
                .iter()
                .map(|t| {
                    (
                        t.dtype(),
                        t.shape().to_vec(),
                        TensorRead::from_tensor(t).strides().unwrap(),
                    )
                })
                .collect(),
        )
    }

    fn fixture() -> (Tensor, Tensor) {
        let a = Tensor::from_vec_col_major(vec![2, 2], vec![0.125_f64, -0.375, 0.5, 1.75]).unwrap();
        let b = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.25, -1.5, 0.625]).unwrap();
        (a, b)
    }

    /// A cache hit must return bit-identical values to a freshly prepared plan.
    #[test]
    fn cached_plan_reuse_is_bit_identical() {
        let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let (a, b) = fixture();
        let inputs = [&a, &b];

        let fresh = ConcreteEinsumPlan::prepare_read_subscripts(
            &[TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)],
            &subscripts,
        )
        .unwrap();

        let k = key(&subscripts, &inputs);
        let built = cached_plan(k, || {
            ConcreteEinsumPlan::prepare_read_subscripts(
                &[TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)],
                &subscripts,
            )
        })
        .unwrap();
        // Second lookup must hit the cache and hand back the same plan.
        let hit = cached_plan(key(&subscripts, &inputs), || {
            panic!("cache hit must not rebuild the plan")
        })
        .unwrap();
        assert!(std::sync::Arc::ptr_eq(&built, &hit));

        let mut backend = CpuBackend::with_threads(1).unwrap();
        let (fresh_out, cached_out) = backend
            .with_backend_session(|session| {
                let reads = [TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)];
                let fresh_out = fresh.execute_read(&reads, session).unwrap();
                let cached_out = hit.execute_read(&reads, session).unwrap();
                Ok::<_, tenferro_tensor::Error>((fresh_out, cached_out))
            })
            .unwrap()
            .unwrap();

        let fresh_bits: Vec<u64> = fresh_out
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let cached_bits: Vec<u64> = cached_out
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(fresh_bits, cached_bits);
        assert_eq!(fresh_out.shape(), cached_out.shape());
        assert_eq!(fresh_out.dtype(), DType::F64);
    }

    /// Different layouts must not share a plan entry.
    #[test]
    fn plan_cache_key_separates_layout() {
        let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let (a, b) = fixture();
        let plain = key(&subscripts, &[&a, &b]);
        let relabelled = PlanCacheKey::new(
            &EinsumSubscripts::new(&[&[1, 2], &[0, 1]], &[0, 2]),
            vec![
                (a.dtype(), a.shape().to_vec(), vec![1, 2]),
                (b.dtype(), b.shape().to_vec(), vec![1, 2]),
            ],
        );
        assert_ne!(plain, relabelled);
    }
}
