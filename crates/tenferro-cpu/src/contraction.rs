//! Tensor-boundary adapters; numerical plans and scheduling belong to cpueinsum.

pub(crate) mod grouped;
mod workspaces;
pub use workspaces::ContractionWorkspaces;
pub(crate) use workspaces::WorkspaceLease;

use crate::buffer_pool::{BufferPool, PoolScalar};
use cpueinsum::tprims_contract::{
    api::{DotGeneral, LayoutSpec, OperandSpec, Problem},
    PlanConfig,
};
use cpueinsum::{ArenaProvider, Exec, Pool, SliceAccumulationSource};
use smallvec::SmallVec;
use std::any::Any;
use std::fmt;
use std::sync::Arc;
use tenferro_cpu_basic::PooledUninitOutput;
use tenferro_tensor::{
    ContractionScalar, DotGeneralAccumulation, DotGeneralConfig, Tensor, TensorRead, TensorView,
    TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};

#[cfg(feature = "native")]
use cpueinsum::BinaryPlan;
#[cfg(feature = "blas")]
use cpueinsum_blas::BinaryPlan;

#[cfg(feature = "native")]
trait Scalar: PoolScalar + cpueinsum::Scalar {}
#[cfg(feature = "native")]
impl<T: PoolScalar + cpueinsum::Scalar> Scalar for T {}
#[cfg(feature = "blas")]
trait Scalar: PoolScalar + cpueinsum_blas::BlasScalar {}
#[cfg(feature = "blas")]
impl<T: PoolScalar + cpueinsum_blas::BlasScalar> Scalar for T {}

/// Translate a lower numerical failure, keeping a typed route decline distinct.
///
/// A lower `Unsupported`/declined step is the lower library refusing a route it
/// owns, not a tenferro backend failure, so it maps to `Unsupported` while the
/// lower diagnostic stays in the message.
pub(crate) trait LowerContractionError {
    fn into_crate_error(self, op: &'static str) -> crate::Error;
}

/// A bare tprims API failure already carries the same decline classification.
impl LowerContractionError for cpueinsum::tprims_contract::Error {
    fn into_crate_error(self, op: &'static str) -> crate::Error {
        if self.is_unsupported() {
            return crate::Error::unsupported(
                op,
                format!("lower contraction declined this route: {self}"),
            );
        }
        crate::Error::backend_source(op, self)
    }
}

impl LowerContractionError for cpueinsum::Error {
    fn into_crate_error(self, op: &'static str) -> crate::Error {
        if let cpueinsum::Error::Contract { source, .. } = &self {
            if source.is_unsupported() {
                return crate::Error::unsupported(
                    op,
                    format!("lower contraction declined this route: {source}"),
                );
            }
        }
        crate::Error::backend_source(op, self)
    }
}

#[cfg(feature = "blas")]
impl LowerContractionError for cpueinsum_blas::BlasError {
    fn into_crate_error(self, op: &'static str) -> crate::Error {
        let inner = match &self {
            cpueinsum_blas::BlasError::Prepare { source }
            | cpueinsum_blas::BlasError::Native { source } => Some(source),
            _ => None,
        };
        if let Some(cpueinsum::Error::Contract { source, .. }) = inner {
            if source.is_unsupported() {
                return crate::Error::unsupported(
                    op,
                    format!("lower contraction declined this route: {source}"),
                );
            }
        }
        crate::Error::backend_source(op, self)
    }
}

// Exactly one retained wrapper per raw pool, shared by CpuContext clones.
pub(crate) struct ExecutionResources {
    pool: Option<Pool<'static>>,
    serial: ArenaProvider,
    pub(crate) nary: ContractionWorkspaces,
}
impl fmt::Debug for ExecutionResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContractionResources")
            .field("threads", &self.pool.as_ref().map_or(1, Pool::size))
            .field("workspace", &self.workspace().stats())
            .finish()
    }
}
impl ExecutionResources {
    pub(crate) fn new(pool: Option<Arc<rayon::ThreadPool>>) -> Self {
        Self {
            pool: pool.map(Pool::shared),
            serial: ArenaProvider::new(),
            nary: ContractionWorkspaces::default(),
        }
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        self.workspace().retained_bytes()
    }
    pub(crate) fn trim(&self) {
        use cpueinsum::WorkspaceProvider;
        self.workspace().trim();
    }
    pub(crate) fn retention_guard(&self, max_bytes: usize) -> WorkspaceRetention<'_> {
        WorkspaceRetention {
            arena: self.workspace(),
            max_bytes,
        }
    }
    fn workspace(&self) -> &ArenaProvider {
        self.pool.as_ref().map_or(&self.serial, Pool::workspace)
    }
    pub(crate) fn exec(&self, budget: usize) -> crate::Result<Exec<'_>> {
        let exec = match &self.pool {
            Some(pool) if budget > 1 => Exec::rayon(pool),
            _ => Exec::serial_with_workspace(self.workspace()),
        };
        exec.with_budget(budget)
            .map_err(|error| crate::Error::backend_source("cpu_contraction", error))
    }
}

pub(crate) struct WorkspaceRetention<'a> {
    arena: &'a ArenaProvider,
    max_bytes: usize,
}
impl Drop for WorkspaceRetention<'_> {
    fn drop(&mut self) {
        use cpueinsum::WorkspaceProvider;
        if self.max_bytes != usize::MAX && self.arena.retained_bytes() > self.max_bytes {
            self.arena.trim();
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    dims: SmallVec<[usize; 8]>,
    strides: SmallVec<[isize; 8]>,
    offset: isize,
}
impl Layout {
    fn new(dims: &[usize], strides: &[isize], offset: isize) -> Self {
        Self {
            dims: dims.into(),
            strides: strides.into(),
            offset,
        }
    }
    fn spec(&self) -> crate::Result<OperandSpec> {
        LayoutSpec::new(&self.dims, &self.strides, self.offset)
            .map(OperandSpec::new)
            .map_err(|error| error.into_crate_error("dot_general"))
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct Key {
    dtype: cpueinsum::tprims_contract::api::DType,
    lhs: Layout,
    rhs: Layout,
    out: Layout,
    lhs_conj: bool,
    rhs_conj: bool,
}
struct CachedBinary {
    key: Key,
    config: DotGeneralConfig,
    plan: Box<dyn Any + Send + Sync>,
}
#[derive(Default)]
pub(crate) struct BinaryCache {
    slots: Vec<Option<CachedBinary>>,
    hits: u64,
    misses: u64,
    evictions: u64,
    clears: u64,
}
impl fmt::Debug for BinaryCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinaryCache")
            .field("slots", &self.slots.len())
            .finish()
    }
}
impl BinaryCache {
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
            retained_bytes: self.slots.capacity() * std::mem::size_of::<Option<CachedBinary>>(),
            ..Default::default()
        };
        for entry in self.slots.iter().flatten() {
            stats.entries += 1;
            // Known owned descriptor payload and opaque lower plan headers.
            // Lower-private metadata heaps need a lower-library footprint report.
            stats.retained_bytes = stats
                .retained_bytes
                .saturating_add(std::mem::size_of_val(entry.plan.as_ref()));
            for layout in [&entry.key.lhs, &entry.key.rhs, &entry.key.out] {
                if layout.dims.spilled() {
                    stats.retained_bytes = stats
                        .retained_bytes
                        .saturating_add(layout.dims.capacity() * std::mem::size_of::<usize>());
                }
                if layout.strides.spilled() {
                    stats.retained_bytes = stats
                        .retained_bytes
                        .saturating_add(layout.strides.capacity() * std::mem::size_of::<isize>());
                }
            }
            for axes in [
                &entry.config.lhs_contracting_dims,
                &entry.config.rhs_contracting_dims,
                &entry.config.lhs_batch_dims,
                &entry.config.rhs_batch_dims,
            ] {
                if axes.spilled() {
                    stats.retained_bytes = stats
                        .retained_bytes
                        .saturating_add(axes.capacity() * std::mem::size_of::<usize>());
                }
            }
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
        key: Key,
        config: &DotGeneralConfig,
        run: impl FnOnce(&BinaryPlan<T>) -> crate::Result<R>,
    ) -> crate::Result<R> {
        let prepare = || {
            let lhs = if key.lhs_conj {
                key.lhs.spec()?.conj()
            } else {
                key.lhs.spec()?
            };
            let rhs = if key.rhs_conj {
                key.rhs.spec()?.conj()
            } else {
                key.rhs.spec()?
            };
            let dot = DotGeneral::new(
                &config.lhs_contracting_dims,
                &config.rhs_contracting_dims,
                &config.lhs_batch_dims,
                &config.rhs_batch_dims,
            );
            let problem = Problem::from_dot_general(key.dtype, lhs, rhs, key.out.spec()?, &dot)
                .map_err(|error| error.into_crate_error("dot_general"))?;
            BinaryPlan::<T>::new(&problem, &PlanConfig::default())
                .map_err(|error| error.into_crate_error("dot_general"))
        };
        let Some(slot) = slot.filter(|&slot| slot < capacity) else {
            return run(&prepare()?);
        };
        let matches = self
            .slots
            .get(slot)
            .and_then(Option::as_ref)
            .is_some_and(|entry| entry.key == key && entry.config == *config);
        if matches {
            self.hits = self.hits.saturating_add(1);
        } else {
            self.misses = self.misses.saturating_add(1);
            let plan = prepare()?;
            if self.slots.len() <= slot {
                self.slots
                    .try_reserve(slot + 1 - self.slots.len())
                    .map_err(|error| crate::Error::backend_source("dot_general", error))?;
                self.slots.resize_with(slot + 1, || None);
            }
            self.slots[slot] = Some(CachedBinary {
                key,
                config: config.clone(),
                plan: Box::new(plan),
            });
        }
        let plan = self.slots[slot]
            .as_ref()
            .and_then(|entry| entry.plan.downcast_ref::<BinaryPlan<T>>())
            .ok_or_else(|| {
                crate::Error::runtime_state(
                    "dot_general",
                    "cached numerical plan has incompatible scalar type",
                )
            })?;
        run(plan)
    }
}

/// Reject non-host operand placements before any numerical work.
pub(crate) fn validate_host_operands(
    op: &'static str,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
) -> crate::Result<()> {
    crate::validate_cpu_host_read(op, "lhs", lhs)?;
    crate::validate_cpu_host_read(op, "rhs", rhs)
}

/// Reject a non-host destination placement before any numerical work.
pub(crate) fn validate_host_output(op: &'static str, out: &TensorWrite<'_>) -> crate::Result<()> {
    crate::validate_cpu_host_write(op, "output", out)
}

fn key<T: Scalar>(
    lhs: &TypedTensorView<'_, T>,
    rhs: &TypedTensorView<'_, T>,
    out: Layout,
    accumulation: DotGeneralAccumulation,
) -> Key {
    Key {
        dtype: <T as cpueinsum::tprims_contract::api::Scalar>::STORAGE,
        lhs: Layout::new(lhs.shape(), lhs.strides(), lhs.offset()),
        rhs: Layout::new(rhs.shape(), rhs.strides(), rhs.offset()),
        out,
        lhs_conj: accumulation.lhs_conj,
        rhs_conj: accumulation.rhs_conj,
    }
}

#[allow(clippy::too_many_arguments)]
fn initialized<T: Scalar>(
    exec: &Exec<'_>,
    buffers: &mut BufferPool,
    cache: &mut crate::gemm::GemmAnalysisCache,
    slot: Option<usize>,
    lhs: TypedTensorView<'_, T>,
    rhs: TypedTensorView<'_, T>,
    mut out: TypedTensorViewMut<'_, T>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    alpha: T,
    beta: T,
) -> crate::Result<()> {
    let key = key(
        &lhs,
        &rhs,
        Layout::new(out.shape(), out.strides(), out.offset()),
        accumulation,
    );
    let output_offset = out.offset();
    let a = lhs.host_storage()?;
    let b = rhs.host_storage()?;
    let d = out.host_storage_mut()?;
    let capacity = cache.capacity();
    cache
        .binary
        .with_plan::<T, _>(slot, capacity, key, config, |plan| {
            #[cfg(feature = "blas")]
            let mut work = T::pool_acquire_zeroed(buffers, plan.work_len());
            #[cfg(feature = "native")]
            let _ = buffers;
            let result = plan
                .execute_slices_accum(
                    exec,
                    alpha,
                    (a, lhs.offset()),
                    (b, rhs.offset()),
                    beta,
                    SliceAccumulationSource::Output,
                    (d, output_offset),
                    #[cfg(feature = "blas")]
                    &mut work,
                )
                .map(|_| ())
                .map_err(|error| error.into_crate_error("dot_general"));
            #[cfg(feature = "blas")]
            T::pool_release(buffers, work);
            result
        })
}

#[allow(clippy::too_many_arguments)]
fn fresh<T: Scalar>(
    exec: &Exec<'_>,
    buffers: &mut BufferPool,
    cache: &mut crate::gemm::GemmAnalysisCache,
    slot: Option<usize>,
    lhs: TypedTensorView<'_, T>,
    rhs: TypedTensorView<'_, T>,
    shape: Vec<usize>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    alpha: T,
    beta: T,
) -> crate::Result<Tensor> {
    let strides = tenferro_tensor::col_major_strides(&shape)?;
    let key = key(&lhs, &rhs, Layout::new(&shape, &strides, 0), accumulation);
    let a = lhs.host_storage()?;
    let b = rhs.host_storage()?;
    let capacity = cache.capacity();
    cache
        .binary
        .with_plan::<T, _>(slot, capacity, key, config, |plan| {
            #[cfg(feature = "blas")]
            let mut work = T::pool_acquire_zeroed(buffers, plan.work_len());
            let mut output = PooledUninitOutput::<T>::new(buffers, shape)?;
            let result = plan
                .execute_uninit_slices(
                    exec,
                    alpha,
                    (a, lhs.offset()),
                    (b, rhs.offset()),
                    beta,
                    SliceAccumulationSource::Absent,
                    (output.as_uninit_slice_mut(), 0),
                    #[cfg(feature = "blas")]
                    &mut work,
                )
                .map(|_| ())
                .map_err(|error| error.into_crate_error("dot_general"));
            #[cfg(feature = "blas")]
            T::pool_release(buffers, work);
            result?;
            // SAFETY: BinaryPlan succeeded on MaybeUninit storage and proves complete
            // physical coverage. No initialized output reference existed before it.
            unsafe { output.assume_init() }.map(Tensor::from_typed::<T>)
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_fresh(
    exec: &Exec<'_>,
    buffers: &mut BufferPool,
    cache: &mut crate::gemm::GemmAnalysisCache,
    slot: Option<usize>,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    shape: Vec<usize>,
) -> crate::Result<Tensor> {
    crate::dot_runtime::validate_dot_operands(&lhs, &rhs, config, accumulation)?;
    macro_rules! dispatch {
        ($($variant:ident),*) => { match (lhs.tensor_view(), rhs.tensor_view(), accumulation.alpha, accumulation.beta) {
            $((TensorView::$variant(a), TensorView::$variant(b), ContractionScalar::$variant(alpha), ContractionScalar::$variant(beta)) =>
                fresh(exec, buffers, cache, slot, a, b, shape, config, accumulation, alpha, beta),)*
            _ => Err(crate::Error::runtime_state("dot_general", "operands and accumulation scalars must have matching supported dtypes")),
        } };
    }
    dispatch!(F32, F64, C32, C64)
}

fn writable_view<'a>(out: TensorWrite<'a>, op: &'static str) -> crate::Result<TensorViewMut<'a>> {
    Ok(match out {
        TensorWrite::View(view) => view,
        TensorWrite::Tensor(tensor) => {
            macro_rules! view { ($($variant:ident => $ty:ty),*) => { match tensor.dtype() {
                $(tenferro_tensor::DType::$variant => TensorViewMut::$variant(tensor.as_typed_mut::<$ty>()
                    .ok_or_else(|| crate::Error::runtime_state(op, "incompatible typed output"))?.as_view_mut()),)*
                dtype => return Err(crate::Error::unsupported_dtype(op, dtype, crate::cpu_contraction_unsupported_dtype_message(dtype))),
            } }; }
            view!(F32 => f32, F64 => f64, C32 => num_complex::Complex32, C64 => num_complex::Complex64)
        }
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dot_into(
    exec: &Exec<'_>,
    buffers: &mut BufferPool,
    cache: &mut crate::gemm::GemmAnalysisCache,
    slot: Option<usize>,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    out: TensorWrite<'_>,
) -> crate::Result<()> {
    crate::dot_runtime::validate_dot_general(&lhs, &rhs, &out, config, accumulation)?;
    if lhs.dtype() != rhs.dtype() {
        return Err(crate::Error::dtype_mismatch(
            "dot_general",
            lhs.dtype(),
            rhs.dtype(),
        ));
    }
    if lhs.dtype() != out.dtype() {
        return Err(crate::Error::dtype_mismatch(
            "dot_general",
            lhs.dtype(),
            out.dtype(),
        ));
    }
    let shape = tenferro_tensor::backend::dot_general_output_shape(
        lhs.shape(),
        rhs.shape(),
        config,
        "dot_general",
    )?;
    if shape != out.shape() {
        return Err(crate::Error::shape_mismatch(
            "dot_general",
            shape,
            out.shape().to_vec(),
        ));
    }
    for scalar in [accumulation.alpha, accumulation.beta] {
        if scalar.dtype() != lhs.dtype() {
            return Err(crate::Error::dtype_mismatch(
                "dot_general",
                lhs.dtype(),
                scalar.dtype(),
            ));
        }
    }
    let out = writable_view(out, "dot_general")?;
    macro_rules! dispatch {
        ($($variant:ident),*) => { match (lhs.tensor_view(), rhs.tensor_view(), out, accumulation.alpha, accumulation.beta) {
            $((TensorView::$variant(a), TensorView::$variant(b), TensorViewMut::$variant(d), ContractionScalar::$variant(alpha), ContractionScalar::$variant(beta)) =>
                initialized(exec, buffers, cache, slot, a, b, d, config, accumulation, alpha, beta),)*
            _ => Err(crate::Error::runtime_state("dot_general", "operands and accumulation scalars must have matching supported dtypes")),
        } };
    }
    dispatch!(F32, F64, C32, C64)
}

#[cfg(test)]
mod retention_tests {
    use super::*;
    use cpueinsum::WorkspaceProvider;
    #[test]
    fn retained_lower_workspace_is_trimmed_on_normal_and_unwind_exit() {
        let resources = ExecutionResources::new(None);
        {
            let _guard = resources.retention_guard(0);
            let mut team = resources.workspace().take_team(&Default::default(), 1, 1);
            team.panel(512);
            drop(team);
            assert!(resources.workspace().retained_bytes() > 0);
        }
        assert_eq!(resources.workspace().retained_bytes(), 0);
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = resources.retention_guard(0);
            let mut team = resources.workspace().take_team(&Default::default(), 1, 1);
            team.panel(512);
            drop(team);
            panic!("exercise workspace cleanup");
        }));
        assert!(unwind.is_err());
        assert_eq!(resources.workspace().retained_bytes(), 0);
    }
}
