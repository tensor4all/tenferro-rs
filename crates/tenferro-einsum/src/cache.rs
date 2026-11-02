use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
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

/// Layout class of one operand as it enters the plan cache.
///
/// An owned tensor is always compact, so its strides are implied by its shape
/// and recording them would only cost an allocation. A borrowed view carries
/// arbitrary strides and must contribute them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutRef<'a> {
    /// Owned compact tensor.
    Compact,
    /// Borrowed view with explicit strides.
    Strided(&'a [isize]),
}

impl LayoutRef<'_> {
    fn hash_into(self, hasher: &mut DefaultHasher) {
        match self {
            Self::Compact => 0u8.hash(hasher),
            Self::Strided(strides) => {
                1u8.hash(hasher);
                strides.hash(hasher);
            }
        }
    }
}

/// Borrowed identity of one operand, used on the lookup path.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OperandRef<'a> {
    pub(crate) dtype: DType,
    pub(crate) shape: &'a [usize],
    pub(crate) layout: LayoutRef<'a>,
}

/// Owned identity of one operand, retained only for a cached entry.
#[derive(Debug, PartialEq, Eq)]
struct OperandIdentity {
    dtype: DType,
    shape: Vec<usize>,
    layout: LayoutRefOwned,
}

#[derive(Debug, PartialEq, Eq)]
enum LayoutRefOwned {
    Compact,
    Strided(Vec<isize>),
}

/// Full identity of a prepared concrete einsum plan, retained for collision
/// verification against a hash-only lookup.
///
/// A plan depends on the operand subscript columns, each operand's dtype, rank,
/// layout and the output specification. The layout column is recorded because
/// the key contract requires it; the planner itself is stride-independent, so a
/// layout difference costs a cache miss but never a wrong plan.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PlanIdentity {
    inputs: Vec<Vec<u32>>,
    output: Vec<u32>,
    operands: Vec<OperandIdentity>,
}

impl PlanIdentity {
    fn new<'a>(
        subscripts: &EinsumSubscripts,
        operands: impl Iterator<Item = OperandRef<'a>>,
    ) -> Self {
        Self {
            inputs: subscripts.inputs.clone(),
            output: subscripts.output.clone(),
            operands: operands
                .map(|operand| OperandIdentity {
                    dtype: operand.dtype,
                    shape: operand.shape.to_vec(),
                    layout: match operand.layout {
                        LayoutRef::Compact => LayoutRefOwned::Compact,
                        LayoutRef::Strided(strides) => LayoutRefOwned::Strided(strides.to_vec()),
                    },
                })
                .collect(),
        }
    }

    fn matches<'a>(
        &self,
        subscripts: &EinsumSubscripts,
        operands: impl Iterator<Item = OperandRef<'a>>,
    ) -> bool {
        if self.inputs != subscripts.inputs || self.output != subscripts.output {
            return false;
        }
        let mut incoming = operands;
        for stored in &self.operands {
            let Some(operand) = incoming.next() else {
                return false;
            };
            if stored.dtype != operand.dtype || stored.shape.as_slice() != operand.shape {
                return false;
            }
            let layout_matches = match (&stored.layout, operand.layout) {
                (LayoutRefOwned::Compact, LayoutRef::Compact) => true,
                (LayoutRefOwned::Strided(stored), LayoutRef::Strided(incoming)) => {
                    stored.as_slice() == incoming
                }
                _ => false,
            };
            if !layout_matches {
                return false;
            }
        }
        incoming.next().is_none()
    }
}

/// Allocation-free lookup hash over the same fields [`PlanIdentity`] verifies.
fn plan_hash<'a>(
    subscripts: &EinsumSubscripts,
    operands: impl Iterator<Item = OperandRef<'a>>,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    subscripts.inputs.hash(&mut hasher);
    subscripts.output.hash(&mut hasher);
    for operand in operands {
        operand.dtype.hash(&mut hasher);
        operand.shape.hash(&mut hasher);
        operand.layout.hash_into(&mut hasher);
    }
    hasher.finish()
}

struct CachedPlan {
    identity: PlanIdentity,
    plan: Arc<ConcreteEinsumPlan>,
}

fn plan_cache() -> &'static Mutex<LruCache<u64, CachedPlan>> {
    static CACHE: OnceLock<Mutex<LruCache<u64, CachedPlan>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        Mutex::new(LruCache::new(
            NonZeroUsize::new(MAX_CACHED_PLANS).unwrap_or(NonZeroUsize::MIN),
        ))
    })
}

/// Return the prepared plan matching `subscripts`/`operands`, building and
/// retaining it on a miss.
///
/// The lookup allocates nothing: the LRU is keyed by an allocation-free hash and
/// the stored identity is compared against borrowed operand metadata, so a hash
/// collision becomes a miss rather than a wrong plan. The lock is never held
/// across `build`, so a miss does not serialize unrelated callers.
pub(crate) fn cached_plan<'a>(
    subscripts: &EinsumSubscripts,
    operands: impl Iterator<Item = OperandRef<'a>> + Clone,
    build: impl FnOnce() -> crate::Result<ConcreteEinsumPlan>,
) -> crate::Result<Arc<ConcreteEinsumPlan>> {
    let hash = plan_hash(subscripts, operands.clone());
    if let Some(entry) = plan_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&hash)
    {
        if entry.identity.matches(subscripts, operands.clone()) {
            return Ok(Arc::clone(&entry.plan));
        }
    }
    let plan = Arc::new(build()?);
    let identity = PlanIdentity::new(subscripts, operands);
    plan_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .put(
            hash,
            CachedPlan {
                identity,
                plan: Arc::clone(&plan),
            },
        );
    Ok(plan)
}

/// Parsed einsum notation retained by parse caches./// Parsed einsum notation retained by parse caches.
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
    use super::{cached_plan, LayoutRef, OperandRef};
    use crate::{ConcreteEinsumPlan, EinsumSubscripts};
    use tenferro_cpu::CpuEngine;
    use tenferro_tensor::{DType, Tensor, TensorRead};

    fn operands<'a>(inputs: &'a [&'a Tensor]) -> impl Iterator<Item = OperandRef<'a>> + Clone + 'a {
        inputs.iter().map(|t| OperandRef {
            dtype: t.dtype(),
            shape: t.shape(),
            layout: LayoutRef::Compact,
        })
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
        let reads = [TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)];

        let fresh = ConcreteEinsumPlan::prepare_read_subscripts(&reads, &subscripts).unwrap();

        let built = cached_plan(&subscripts, operands(&inputs), || {
            ConcreteEinsumPlan::prepare_read_subscripts(&reads, &subscripts)
        })
        .unwrap();
        // A second lookup must hit the cache and hand back the same plan.
        let hit = cached_plan(&subscripts, operands(&inputs), || {
            panic!("cache hit must not rebuild the plan")
        })
        .unwrap();
        assert!(std::sync::Arc::ptr_eq(&built, &hit));

        let mut engine = CpuEngine::serial();
        let fresh_out = fresh.execute_read(&reads, &mut engine).unwrap();
        let cached_out = hit.execute_read(&reads, &mut engine).unwrap();

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

    /// Different subscript columns or layouts must not share an entry.
    #[test]
    fn plan_cache_key_separates_subscripts_and_layout() {
        let subscripts = EinsumSubscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
        let swapped = EinsumSubscripts::new(&[&[1, 2], &[0, 1]], &[0, 2]);
        let (a, b) = fixture();
        let inputs = [&a, &b];

        let mut first = cached_plan(&subscripts, operands(&inputs), || {
            ConcreteEinsumPlan::prepare_read_subscripts(
                &[TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)],
                &subscripts,
            )
        })
        .unwrap();
        // Same operands, different subscript columns: must rebuild.
        let second = cached_plan(&swapped, operands(&inputs), || {
            ConcreteEinsumPlan::prepare_read_subscripts(
                &[TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)],
                &swapped,
            )
        })
        .unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &second));

        // A strided view of the same shape must not reuse the compact entry.
        let data = [1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0];
        let view =
            tenferro_tensor::TypedTensorView::from_slice(vec![2, 2], vec![2, 1], 0, &data).unwrap();
        let strided = cached_plan(
            &subscripts,
            std::iter::once(OperandRef {
                dtype: DType::F64,
                shape: &[2, 2],
                layout: LayoutRef::Strided(&[2, 1]),
            })
            .chain(std::iter::once(OperandRef {
                dtype: DType::F64,
                shape: &[2, 2],
                layout: LayoutRef::Compact,
            })),
            || {
                ConcreteEinsumPlan::prepare_read_subscripts(
                    &[
                        TensorRead::from_view(tenferro_tensor::TensorView::F64(view)),
                        TensorRead::from_tensor(&b),
                    ],
                    &subscripts,
                )
            },
        )
        .unwrap();
        first = cached_plan(&subscripts, operands(&inputs), || {
            panic!("compact entry must still be present")
        })
        .unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &strided));
    }
}
