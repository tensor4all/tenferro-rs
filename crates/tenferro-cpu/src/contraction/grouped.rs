//! Initialized grouped outputs; offsets are relative to compact logical buffers.

use super::{
    BufferPool, ContractionScalar, Exec, PlanConfig, Scalar, TensorRead, TensorView, TensorViewMut,
    TensorWrite, TypedTensorView, TypedTensorViewMut,
};
use crate::contraction::LowerContractionError;
use cpueinsum::tprims_contract::api::Op;
#[cfg(feature = "native")]
use cpueinsum::GroupedPlan;
#[cfg(feature = "blas")]
use cpueinsum_blas::GroupedPlan;
use std::any::Any;
use std::fmt;
use tenferro_tensor::backend::{GroupedGemmConfig, GroupedGemmJob};
const OP: &str = "grouped_gemm";

struct Entry {
    dtype: cpueinsum::tprims_contract::api::DType,
    jobs: Vec<GroupedGemmJob>,
    ops: [Op; 3],
    plan: Box<dyn Any + Send + Sync>,
}
#[derive(Default)]
pub(crate) struct GroupCache {
    slots: Vec<Option<Entry>>,
    hits: u64,
    misses: u64,
    evictions: u64,
    clears: u64,
}
impl fmt::Debug for GroupCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GroupCache")
            .field("slots", &self.slots.len())
            .finish()
    }
}
impl GroupCache {
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.slots.shrink_to_fit();
        self.clears = self.clears.saturating_add(1);
    }
    pub(crate) fn stats(&self) -> tenferro_tensor::CacheStats {
        let mut stats = tenferro_tensor::CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            clears: self.clears,
            retained_bytes: self.slots.capacity() * std::mem::size_of::<Option<Entry>>(),
            ..Default::default()
        };
        for entry in self.slots.iter().flatten() {
            stats.entries += 1;
            // Host job descriptors and lower plan headers; opaque lower heaps
            // require a footprint report from their owning library.
            stats.retained_bytes = stats
                .retained_bytes
                .saturating_add(entry.jobs.capacity() * std::mem::size_of::<GroupedGemmJob>())
                .saturating_add(std::mem::size_of_val(entry.plan.as_ref()));
        }
        stats
    }
    pub(crate) fn set_capacity(&mut self, capacity: usize) {
        self.evictions = self.evictions.saturating_add(
            self.slots
                .iter()
                .skip(capacity)
                .filter(|slot| slot.is_some())
                .count() as u64,
        );
        self.slots.truncate(capacity);
        self.slots.shrink_to(capacity);
    }
    fn with_plan<T: Scalar, R>(
        &mut self,
        slot: Option<usize>,
        capacity: usize,
        config: &GroupedGemmConfig<'_>,
        run: impl FnOnce(&GroupedPlan<T>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let accumulation = config.accumulation();
        let ops = [
            if accumulation.lhs_conj {
                Op::Conjugate
            } else {
                Op::Identity
            },
            if accumulation.rhs_conj {
                Op::Conjugate
            } else {
                Op::Identity
            },
            Op::Identity,
        ];
        let dtype = <T as cpueinsum::tprims_contract::api::Scalar>::STORAGE;
        let prepare = || {
            // Constructors intentionally differ: host out/lhs/rhs, lower lhs/rhs/out.
            let jobs: Vec<_> = config
                .jobs()
                .iter()
                .map(|job| {
                    cpueinsum::GroupedGemmJob::new(
                        job.lhs_offset(),
                        job.rhs_offset(),
                        job.out_offset(),
                        job.rows(),
                        job.contracted(),
                        job.cols(),
                    )
                })
                .collect();
            GroupedPlan::<T>::new_with_ops(&jobs, ops, &PlanConfig::default())
                .map_err(|error| error.into_crate_error(OP))
        };
        let Some(slot) = slot.filter(|&slot| slot < capacity) else {
            return run(&prepare()?);
        };
        let matches = self
            .slots
            .get(slot)
            .and_then(Option::as_ref)
            .is_some_and(|entry| {
                entry.dtype == dtype && entry.jobs == config.jobs() && entry.ops == ops
            });
        if matches {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
            let plan = prepare()?;
            if self.slots.len() <= slot {
                self.slots
                    .try_reserve(slot + 1 - self.slots.len())
                    .map_err(|error| crate::Error::backend_source(OP, error))?;
                self.slots.resize_with(slot + 1, || None);
            }
            self.slots[slot] = Some(Entry {
                dtype,
                jobs: config.jobs().to_vec(),
                ops,
                plan: Box::new(plan),
            });
        }
        let plan = self.slots[slot]
            .as_ref()
            .and_then(|entry| entry.plan.downcast_ref::<GroupedPlan<T>>())
            .ok_or_else(|| {
                crate::Error::runtime_state(OP, "cached grouped plan has incompatible scalar type")
            })?;
        run(plan)
    }
}

fn range(shape: &[usize], offset: isize) -> crate::Result<std::ops::Range<usize>> {
    let len = tenferro_tensor::validate::checked_shape_product(OP, "buffer", shape)?;
    let start = usize::try_from(offset).map_err(|error| crate::Error::backend_source(OP, error))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| crate::Error::runtime_state(OP, "compact buffer span overflow"))?;
    Ok(start..end)
}
#[allow(clippy::too_many_arguments)]
fn typed<T: Scalar>(
    exec: &Exec<'_>,
    buffers: &mut BufferPool,
    cache: &mut crate::gemm::GemmAnalysisCache,
    slot: Option<usize>,
    lhs: TypedTensorView<'_, T>,
    rhs: TypedTensorView<'_, T>,
    mut out: TypedTensorViewMut<'_, T>,
    config: &GroupedGemmConfig<'_>,
    alpha: T,
    beta: T,
) -> crate::Result<()> {
    let ar = range(lhs.shape(), lhs.offset())?;
    let br = range(rhs.shape(), rhs.offset())?;
    let dr = range(out.shape(), out.offset())?;
    let a = lhs
        .host_storage()?
        .get(ar)
        .ok_or_else(|| crate::Error::runtime_state(OP, "lhs compact span outside backing"))?;
    let b = rhs
        .host_storage()?
        .get(br)
        .ok_or_else(|| crate::Error::runtime_state(OP, "rhs compact span outside backing"))?;
    let d = out
        .host_storage_mut()?
        .get_mut(dr)
        .ok_or_else(|| crate::Error::runtime_state(OP, "output compact span outside backing"))?;
    let capacity = cache.capacity();
    cache
        .grouped
        .with_plan::<T, _>(slot, capacity, config, |plan| {
            #[cfg(feature = "blas")]
            let mut work = T::pool_acquire_zeroed(buffers, plan.work_len());
            #[cfg(feature = "native")]
            let _ = buffers;
            let result = plan
                .execute_into_accum(
                    exec,
                    alpha,
                    a,
                    b,
                    beta,
                    d,
                    #[cfg(feature = "blas")]
                    &mut work,
                )
                .map(|_| ())
                .map_err(|error| error.into_crate_error(OP));
            #[cfg(feature = "blas")]
            T::pool_release(buffers, work);
            result
        })
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute(
    exec: &Exec<'_>,
    buffers: &mut BufferPool,
    cache: &mut crate::gemm::GemmAnalysisCache,
    slot: Option<usize>,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &GroupedGemmConfig<'_>,
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    tenferro_tensor::backend::validate_grouped_gemm(&lhs, &rhs, &out, config, OP)?;
    crate::contraction::validate_host_operands(OP, &lhs, &rhs)?;
    crate::contraction::validate_host_output(OP, &out)?;
    if !lhs.is_col_major_contiguous()?
        || !rhs.is_col_major_contiguous()?
        || !out.is_col_major_contiguous()?
    {
        return Err(crate::Error::unsupported(OP, "grouped GEMM requires compact column-major operands; use binary dot_general for strided operands"));
    }
    let dtype = lhs.dtype();
    let accumulation = config.accumulation();
    let out = super::writable_view(out, OP)?;
    macro_rules! dispatch {
        ($($variant:ident),*) => { match (lhs.tensor_view(), rhs.tensor_view(), out, accumulation.alpha, accumulation.beta) {
            $((TensorView::$variant(a), TensorView::$variant(b), TensorViewMut::$variant(d), ContractionScalar::$variant(alpha), ContractionScalar::$variant(beta)) =>
                typed(exec, buffers, cache, slot, a, b, d, config, alpha, beta),)*
            _ => Err(crate::Error::unsupported_dtype(OP, dtype, crate::cpu_contraction_unsupported_dtype_message(dtype))),
        } };
    }
    dispatch!(F32, F64, C32, C64)
}
