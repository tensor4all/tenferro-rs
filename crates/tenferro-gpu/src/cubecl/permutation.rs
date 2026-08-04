use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use cubecl::prelude::{CubeElement, CubePrimitive};
use num_complex::{Complex32, Complex64};
use num_traits::One;
use tenferro_tensor::{CacheStats, DType, TensorRank, TypedTensorView, TypedTensorViewMut};

use super::dispatch::{
    ensure_resident_on_runtime, ensure_view_mut_resident_on_runtime,
    ensure_view_resident_on_runtime, prepared_tensor_access, prepared_view_access,
    prepared_view_mut_access, CubeclPreparedAccess,
};
use super::ffi::cutensor::{
    CudaDataType, CutensorComputeDescriptor, CutensorCudaStream, CutensorHandle, CutensorOperator,
    OperationDescriptor, Plan, PlanPreference, TensorDescriptor,
};
use super::interop::cuda_device_ptr_from_addr;
use super::{CudaBackend, CudaRuntime};
use crate::{col_major_strides, Error, TypedTensor};

const OP_TRANSPOSE: &str = "transpose";
const CUDA_ALLOCATION_ALIGNMENT: u32 = 256;
const DEFAULT_CUTENSOR_PERMUTATION_PLAN_CACHE_MAX_ENTRIES: usize = 64;

type CutensorPermutationPlanCacheState = Arc<Mutex<CutensorPermutationPlanCache>>;

pub(super) trait CutensorPermutationScalar:
    CubeElement + CubePrimitive + Clone + One + Send + Sync + 'static
{
    const DATA_TYPE: CudaDataType;
    const DTYPE: DType;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor;
}

impl CutensorPermutationScalar for f32 {
    const DATA_TYPE: CudaDataType = CudaDataType::R32F;
    const DTYPE: DType = DType::F32;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_32f()
    }
}

impl CutensorPermutationScalar for f64 {
    const DATA_TYPE: CudaDataType = CudaDataType::R64F;
    const DTYPE: DType = DType::F64;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_64f()
    }
}

impl CutensorPermutationScalar for Complex32 {
    const DATA_TYPE: CudaDataType = CudaDataType::C32F;
    const DTYPE: DType = DType::C32;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_32f()
    }
}

impl CutensorPermutationScalar for Complex64 {
    const DATA_TYPE: CudaDataType = CudaDataType::C64F;
    const DTYPE: DType = DType::C64;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_64f()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CutensorPermutationLayoutKey {
    extents: Vec<i64>,
    strides: Vec<i64>,
    modes: Vec<i32>,
}

impl CutensorPermutationLayoutKey {
    fn new(extents: &[i64], strides: &[i64], modes: &[i32]) -> Self {
        Self {
            extents: extents.to_vec(),
            strides: strides.to_vec(),
            modes: modes.to_vec(),
        }
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.extents.capacity() * std::mem::size_of::<i64>())
            .saturating_add(self.strides.capacity() * std::mem::size_of::<i64>())
            .saturating_add(self.modes.capacity() * std::mem::size_of::<i32>())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CutensorPermutationKey {
    dtype: DType,
    input: CutensorPermutationLayoutKey,
    output: CutensorPermutationLayoutKey,
    input_alignment_requirement: u32,
    output_alignment_requirement: u32,
    input_op: CutensorOperator,
}

impl CutensorPermutationKey {
    fn from_spec<T: CutensorPermutationScalar>(spec: &CutensorPermutationSpec<'_>) -> Self {
        Self {
            dtype: T::DTYPE,
            input: CutensorPermutationLayoutKey::new(
                spec.input_extents,
                spec.input_strides,
                spec.input_modes,
            ),
            output: CutensorPermutationLayoutKey::new(
                spec.output_extents,
                spec.output_strides,
                spec.output_modes,
            ),
            input_alignment_requirement: spec.input_alignment_requirement,
            output_alignment_requirement: spec.output_alignment_requirement,
            input_op: spec.input_op,
        }
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.input.retained_bytes())
            .saturating_add(self.output.retained_bytes())
    }
}

struct CutensorPermutationSpec<'a> {
    input_extents: &'a [i64],
    input_strides: &'a [i64],
    input_modes: &'a [i32],
    output_extents: &'a [i64],
    output_strides: &'a [i64],
    output_modes: &'a [i32],
    input_alignment_requirement: u32,
    output_alignment_requirement: u32,
    input_op: CutensorOperator,
}

struct CachedCutensorPermutation {
    // Drop the cuTENSOR plan before the descriptor objects it was built from.
    plan: Plan,
    _plan_preference: PlanPreference,
    _operation_descriptor: OperationDescriptor,
    _output_descriptor: TensorDescriptor,
    _input_descriptor: TensorDescriptor,
}

// SAFETY: cached cuTENSOR state is tied to one `CudaBackend` and is only used
// while holding the enclosing plan-cache mutex. The opaque cuTENSOR handles are
// created and destroyed through the same loaded cuTENSOR library.
unsafe impl Send for CachedCutensorPermutation {}

impl CachedCutensorPermutation {
    fn new<T>(
        cutensor: &CutensorHandle,
        spec: &CutensorPermutationSpec<'_>,
        op: &'static str,
    ) -> crate::Result<Self>
    where
        T: CutensorPermutationScalar,
    {
        let desc_input = TensorDescriptor::new(
            cutensor,
            spec.input_extents,
            spec.input_strides,
            T::DATA_TYPE,
            spec.input_alignment_requirement,
            op,
        )?;
        let desc_output = TensorDescriptor::new(
            cutensor,
            spec.output_extents,
            spec.output_strides,
            T::DATA_TYPE,
            spec.output_alignment_requirement,
            op,
        )?;
        let op_desc = OperationDescriptor::new_permutation(
            cutensor,
            &desc_input,
            spec.input_modes,
            spec.input_op,
            &desc_output,
            spec.output_modes,
            T::compute_descriptor(cutensor),
            op,
        )?;
        let pref = PlanPreference::new_default(cutensor, op)?;
        let plan = Plan::new(cutensor, &op_desc, &pref, 0, op)?;
        Ok(Self {
            plan,
            _plan_preference: pref,
            _operation_descriptor: op_desc,
            _output_descriptor: desc_output,
            _input_descriptor: desc_input,
        })
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

struct CutensorPermutationPlanCache {
    max_entries: NonZeroUsize,
    entries: HashMap<CutensorPermutationKey, CachedCutensorPermutation>,
    order: VecDeque<CutensorPermutationKey>,
    stats: CacheStats,
}

impl CutensorPermutationPlanCache {
    fn new(max_entries: NonZeroUsize) -> Self {
        Self {
            max_entries,
            entries: HashMap::new(),
            order: VecDeque::new(),
            stats: CacheStats::empty(),
        }
    }

    fn ensure<T>(
        &mut self,
        cutensor: &CutensorHandle,
        key: &CutensorPermutationKey,
        spec: &CutensorPermutationSpec<'_>,
        op: &'static str,
    ) -> crate::Result<()>
    where
        T: CutensorPermutationScalar,
    {
        if self.entries.contains_key(key) {
            self.stats.hits = self.stats.hits.saturating_add(1);
            self.touch(key);
            return Ok(());
        }
        self.stats.misses = self.stats.misses.saturating_add(1);
        let cached = CachedCutensorPermutation::new::<T>(cutensor, spec, op)?;
        self.entries.insert(key.clone(), cached);
        self.touch(key);
        self.evict_to_limit();
        Ok(())
    }

    fn get(&self, key: &CutensorPermutationKey) -> crate::Result<&CachedCutensorPermutation> {
        self.entries.get(key).ok_or_else(|| {
            Error::runtime_state(
                "cutensor_permutation_plan_cache",
                "cached cuTENSOR permutation was evicted before use",
            )
        })
    }

    fn touch(&mut self, key: &CutensorPermutationKey) {
        self.order.retain(|existing| existing != key);
        self.order.push_back(key.clone());
    }

    fn evict_to_limit(&mut self) {
        while self.entries.len() > self.max_entries.get() {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if self.entries.remove(&oldest).is_some() {
                self.stats.evictions = self.stats.evictions.saturating_add(1);
            }
        }
    }

    fn max_entries(&self) -> NonZeroUsize {
        self.max_entries
    }

    fn set_max_entries(&mut self, max_entries: NonZeroUsize) {
        self.max_entries = max_entries;
        self.evict_to_limit();
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            retained_bytes: self.retained_bytes(),
            ..self.stats
        }
    }

    fn retained_bytes(&self) -> usize {
        let entries_bytes =
            self.entries
                .iter()
                .fold(std::mem::size_of::<Self>(), |total, (key, cached)| {
                    total
                        .saturating_add(key.retained_bytes())
                        .saturating_add(cached.retained_bytes())
                });
        entries_bytes
            .saturating_add(self.order.capacity() * std::mem::size_of::<CutensorPermutationKey>())
    }
}

struct ResolvedPermutationOperand {
    ptr: *mut c_void,
    alignment: u32,
}

pub(super) fn transpose<T>(
    backend: &CudaBackend,
    input: &TypedTensor<T>,
    perm: &[usize],
) -> crate::Result<TypedTensor<T>>
where
    T: CutensorPermutationScalar,
{
    super::validate_permutation(OP_TRANSPOSE, perm, input.shape().len())?;
    backend.runtime().set_current_cuda_context(OP_TRANSPOSE)?;
    let output_shape: Vec<usize> = perm.iter().map(|&axis| input.shape()[axis]).collect();
    let output = super::dispatch::alloc_output::<T>(backend.runtime(), &output_shape)?;
    let input_strides = compact_strides_i64(OP_TRANSPOSE, input.shape())?;
    let output_strides = compact_strides_i64(OP_TRANSPOSE, output.shape())?;
    let input_modes = identity_modes(OP_TRANSPOSE, input.shape().len())?;
    let output_modes = modes_from_perm(OP_TRANSPOSE, perm)?;
    let input_extents = dims_to_i64(OP_TRANSPOSE, input.shape())?;
    let output_extents = dims_to_i64(OP_TRANSPOSE, output.shape())?;
    let input_res = resolve_owned_operand(backend.runtime(), input, OP_TRANSPOSE)?;
    let output_res = resolve_owned_operand(backend.runtime(), &output, OP_TRANSPOSE)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }

    execute_permutation::<T>(
        backend,
        input_res,
        output_res,
        CutensorPermutationSpec {
            input_extents: &input_extents,
            input_strides: &input_strides,
            input_modes: &input_modes,
            output_extents: &output_extents,
            output_strides: &output_strides,
            output_modes: &output_modes,
            input_alignment_requirement: CUDA_ALLOCATION_ALIGNMENT,
            output_alignment_requirement: CUDA_ALLOCATION_ALIGNMENT,
            input_op: CutensorOperator::Identity,
        },
        OP_TRANSPOSE,
    )?;
    Ok(output)
}

pub(super) fn to_contiguous_view<T, R>(
    backend: &CudaBackend,
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R>>
where
    T: CutensorPermutationScalar,
    R: TensorRank,
{
    backend.runtime().set_current_cuda_context(op)?;
    let output = backend.alloc_ranked_output::<T, R>(view.shape(), op)?;
    let input_strides = view_strides_i64(view.strides(), op)?;
    let output_strides = compact_strides_i64(op, output.shape())?;
    let modes = identity_modes(op, view.shape().len())?;
    let input_extents = dims_to_i64(op, view.shape())?;
    let output_extents = dims_to_i64(op, output.shape())?;
    let input_res = resolve_view_operand(backend.runtime(), view, op)?;
    let output_res = resolve_owned_operand(backend.runtime(), &output, op)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }

    execute_permutation::<T>(
        backend,
        input_res,
        output_res,
        CutensorPermutationSpec {
            input_extents: &input_extents,
            input_strides: &input_strides,
            input_modes: &modes,
            output_extents: &output_extents,
            output_strides: &output_strides,
            output_modes: &modes,
            input_alignment_requirement: view_descriptor_alignment_requirement::<T>(),
            output_alignment_requirement: CUDA_ALLOCATION_ALIGNMENT,
            input_op: CutensorOperator::Identity,
        },
        op,
    )?;
    Ok(output)
}

/// Copy a compact source view into a caller-owned CUDA destination view using
/// the same cached cuTENSOR permutation plan as allocating materialization.
///
/// The caller selects this path only for layouts supported by cuTENSOR. This
/// function deliberately does not fall back when cuTENSOR loading or plan
/// creation fails: supported CUDA permutation paths require the NVIDIA
/// library stack.
pub(super) fn copy_view_into<T, R>(
    backend: &CudaBackend,
    src: &TypedTensorView<'_, T, R>,
    dst: &mut TypedTensorViewMut<'_, T, R>,
    op: &'static str,
) -> crate::Result<()>
where
    T: CutensorPermutationScalar,
    R: TensorRank,
{
    ensure_view_resident_on_runtime(backend.runtime(), src, op)?;
    ensure_view_mut_resident_on_runtime(backend.runtime(), dst, op)?;
    if src.shape() != dst.shape() {
        return Err(crate::Error::shape_mismatch(
            op,
            src.shape().to_vec(),
            dst.shape().to_vec(),
        ));
    }
    let source_storage = src
        .backend_buffer()
        .ok_or_else(|| Error::runtime_state(op, "expected a CUDA source view"))?;
    let destination_storage = dst
        .backend_buffer()
        .ok_or_else(|| Error::runtime_state(op, "expected a CUDA destination view"))?;
    if std::ptr::eq(source_storage, destination_storage) {
        return Err(crate::Error::invalid_argument(
            op,
            "source/destination",
            "CUDA copy_into source and destination allocations must not alias",
        ));
    }
    if !src.is_col_major_contiguous()? || src.offset() != 0 {
        return Err(crate::Error::invalid_argument(
            op,
            "source",
            "CUDA copy_into requires a compact source view covering its full allocation; arbitrary-stride source views are unsupported without explicit canonicalization",
        ));
    }
    if dst.n_elements() == 0 {
        return Ok(());
    }

    backend.runtime().set_current_cuda_context(op)?;
    let input_extents = dims_to_i64(op, src.shape())?;
    let input_strides = compact_strides_i64(op, src.shape())?;
    let modes = identity_modes(op, src.shape().len())?;
    // Describe the destination in physical stride order. This is equivalent
    // to the logical-order view descriptor, but lets cuTENSOR see the same
    // permutation layout as the direct destination-reuse control. In
    // particular, a transposed 2D view becomes extents/strides `[n, m]` /
    // `[1, n]` with modes `[1, 0]`, instead of an identity-mode descriptor
    // with swapped strides that may select a slower plan.
    let (output_extents, output_strides, output_modes) =
        physical_output_descriptor(op, dst.shape(), dst.strides())?;
    let input_res = resolve_prepared_device_region::<T>(
        backend.runtime(),
        prepared_view_access(src, op)?,
        src.offset(),
        op,
    )?;
    let output_res = resolve_prepared_device_region::<T>(
        backend.runtime(),
        prepared_view_mut_access(dst, op)?,
        dst.offset(),
        op,
    )?;
    let input_alignment_requirement = input_res.alignment;
    let output_alignment_requirement = output_res.alignment;
    execute_permutation::<T>(
        backend,
        input_res,
        output_res,
        CutensorPermutationSpec {
            input_extents: &input_extents,
            input_strides: &input_strides,
            input_modes: &modes,
            output_extents: &output_extents,
            output_strides: &output_strides,
            output_modes: &output_modes,
            // Use the resolved pointer alignment rather than the scalar size.
            // The destination-reuse benchmark passes whole allocations with
            // 256-byte alignment, and cuTENSOR may choose a materially
            // different kernel when the descriptor advertises only 8-byte
            // alignment for f64. The resolved values are also the alignment
            // that the actual pointers can satisfy.
            input_alignment_requirement,
            output_alignment_requirement,
            input_op: CutensorOperator::Identity,
        },
        op,
    )
}

fn physical_output_descriptor(
    op: &'static str,
    shape: &[usize],
    strides: &[isize],
) -> crate::Result<(Vec<i64>, Vec<i64>, Vec<i32>)> {
    if shape.len() != strides.len() {
        return Err(Error::invalid_argument(
            op,
            "layout",
            "destination shape and stride ranks must match",
        ));
    }
    if strides.iter().any(|&stride| stride < 0) {
        return Err(Error::invalid_argument(
            op,
            "layout",
            "cuTENSOR destination descriptors do not support negative strides",
        ));
    }
    let mut axes: Vec<usize> = (0..shape.len()).collect();
    axes.sort_by_key(|&axis| (strides[axis], axis));
    let extents = axes
        .iter()
        .map(|&axis| i64::try_from(shape[axis]))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| Error::invalid_argument(op, "shape", "dimension exceeds i64 extent limit"))?;
    let sorted_strides = axes
        .iter()
        .map(|&axis| i64::try_from(strides[axis]))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| Error::invalid_argument(op, "layout", "stride exceeds i64 limit"))?;
    let modes = axes
        .iter()
        .map(|&axis| {
            i32::try_from(axis)
                .map_err(|_| Error::invalid_argument(op, "rank", "rank exceeds i32 mode limit"))
        })
        .collect::<crate::Result<Vec<_>>>()?;
    Ok((extents, sorted_strides, modes))
}

fn execute_permutation<T>(
    backend: &CudaBackend,
    input_res: ResolvedPermutationOperand,
    output_res: ResolvedPermutationOperand,
    spec: CutensorPermutationSpec<'_>,
    op: &'static str,
) -> crate::Result<()>
where
    T: CutensorPermutationScalar,
{
    validate_descriptor_alignment(
        input_res.alignment,
        spec.input_alignment_requirement,
        "input",
        op,
    )?;
    validate_descriptor_alignment(
        output_res.alignment,
        spec.output_alignment_requirement,
        "output",
        op,
    )?;
    let alpha = T::one();
    let stream = raw_stream(backend.runtime())?;
    cached_cutensor_permutation::<T, _>(backend, &spec, op, |cutensor, plan| unsafe {
        cutensor.permute(
            plan,
            &alpha as *const T as *const c_void,
            input_res.ptr as *const c_void,
            output_res.ptr,
            stream,
            op,
        )
    })
}

fn default_cutensor_permutation_plan_cache_max_entries() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_CUTENSOR_PERMUTATION_PLAN_CACHE_MAX_ENTRIES)
        .unwrap_or(NonZeroUsize::MIN)
}

fn new_cutensor_permutation_plan_cache_state(
    max_entries: NonZeroUsize,
) -> CutensorPermutationPlanCacheState {
    Arc::new(Mutex::new(CutensorPermutationPlanCache::new(max_entries)))
}

fn get_or_init_cutensor_permutation_plan_cache(
    backend: &CudaBackend,
) -> crate::Result<CutensorPermutationPlanCacheState> {
    let guard = backend
        .cuda_extension_cache()
        .get_or_try_init::<CutensorPermutationPlanCacheState>(|| {
            Ok(new_cutensor_permutation_plan_cache_state(
                default_cutensor_permutation_plan_cache_max_entries(),
            ))
        })?;
    Ok(Arc::clone(&guard))
}

fn lock_cutensor_permutation_plan_cache(
    cache: &CutensorPermutationPlanCacheState,
) -> crate::Result<std::sync::MutexGuard<'_, CutensorPermutationPlanCache>> {
    cache.lock().map_err(|_| {
        Error::runtime_state(
            "cutensor_permutation_plan_cache",
            "plan cache lock poisoned",
        )
    })
}

pub(super) fn cutensor_permutation_plan_cache_stats(
    backend: &CudaBackend,
) -> crate::Result<CacheStats> {
    let Some(plan_cache) = backend
        .cuda_extension_cache()
        .get_cloned::<CutensorPermutationPlanCacheState>()?
    else {
        return Ok(CacheStats::empty());
    };
    let plan_cache = lock_cutensor_permutation_plan_cache(&plan_cache)?;
    Ok(plan_cache.stats())
}

pub(super) fn cutensor_permutation_plan_cache_max_entries(
    backend: &CudaBackend,
) -> crate::Result<NonZeroUsize> {
    let Some(plan_cache) = backend
        .cuda_extension_cache()
        .get_cloned::<CutensorPermutationPlanCacheState>()?
    else {
        return Ok(default_cutensor_permutation_plan_cache_max_entries());
    };
    let plan_cache = lock_cutensor_permutation_plan_cache(&plan_cache)?;
    Ok(plan_cache.max_entries())
}

pub(super) fn set_cutensor_permutation_plan_cache_max_entries(
    backend: &CudaBackend,
    max_entries: NonZeroUsize,
) -> crate::Result<()> {
    let plan_cache = get_or_init_cutensor_permutation_plan_cache(backend)?;
    let mut plan_cache = lock_cutensor_permutation_plan_cache(&plan_cache)?;
    plan_cache.set_max_entries(max_entries);
    let retained_bytes = plan_cache.retained_bytes();
    backend
        .cuda_extension_cache()
        .update_retained_bytes::<CutensorPermutationPlanCacheState>(retained_bytes)
}

fn cached_cutensor_permutation<T, R>(
    backend: &CudaBackend,
    spec: &CutensorPermutationSpec<'_>,
    op: &'static str,
    execute: impl FnOnce(&CutensorHandle, &Plan) -> crate::Result<R>,
) -> crate::Result<R>
where
    T: CutensorPermutationScalar,
{
    let cutensor = backend.cutensor_handle()?;
    let key = CutensorPermutationKey::from_spec::<T>(spec);
    let plan_cache = get_or_init_cutensor_permutation_plan_cache(backend)?;
    let mut plan_cache = lock_cutensor_permutation_plan_cache(&plan_cache)?;
    plan_cache.ensure::<T>(cutensor, &key, spec, op)?;
    let retained_bytes = plan_cache.retained_bytes();
    backend
        .cuda_extension_cache()
        .update_retained_bytes::<CutensorPermutationPlanCacheState>(retained_bytes)?;
    let cached = plan_cache.get(&key)?;
    execute(cutensor, &cached.plan)
}

fn resolve_owned_operand<T, R>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, R>,
    op: &'static str,
) -> crate::Result<ResolvedPermutationOperand>
where
    T: 'static,
    R: TensorRank,
{
    ensure_resident_on_runtime(rt, tensor, op)?;
    Ok(ResolvedPermutationOperand {
        ptr: typed_device_ptr(rt, tensor, op)?,
        alignment: CUDA_ALLOCATION_ALIGNMENT,
    })
}

fn resolve_view_operand<T, R>(
    rt: &CudaRuntime,
    view: &TypedTensorView<'_, T, R>,
    op: &'static str,
) -> crate::Result<ResolvedPermutationOperand>
where
    T: 'static,
    R: TensorRank,
{
    ensure_view_resident_on_runtime(rt, view, op)?;
    resolve_prepared_device_region::<T>(rt, prepared_view_access(view, op)?, view.offset(), op)
}

fn typed_device_ptr<T, R>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T, R>,
    op: &'static str,
) -> crate::Result<*mut c_void>
where
    T: 'static,
    R: TensorRank,
{
    let prepared = prepared_tensor_access(tensor, op)?;
    let resource = rt
        .client()
        .get_resource(prepared.into_handle())
        .map_err(|err| Error::backend_source(op, err))?;
    cuda_device_ptr_from_addr(resource.resource().ptr, op)
}

fn resolve_prepared_device_region<T: 'static>(
    rt: &CudaRuntime,
    prepared: CubeclPreparedAccess,
    offset: isize,
    op: &'static str,
) -> crate::Result<ResolvedPermutationOperand> {
    let offset = usize::try_from(offset).map_err(|_| {
        Error::invalid_argument(
            op,
            "layout",
            format!("view offset {offset} must be nonnegative for cuTENSOR permutation"),
        )
    })?;
    let resource = rt
        .client()
        .get_resource(prepared.into_handle())
        .map_err(|err| Error::backend_source(op, err))?;
    let offset_bytes = (offset as u64)
        .checked_mul(std::mem::size_of::<T>() as u64)
        .ok_or_else(|| Error::invalid_argument(op, "layout", "view byte offset overflows u64"))?;
    let addr = resource
        .resource()
        .ptr
        .checked_add(offset_bytes)
        .ok_or_else(|| {
            Error::invalid_argument(op, "layout", "view device address overflows u64")
        })?;
    Ok(ResolvedPermutationOperand {
        ptr: cuda_device_ptr_from_addr(addr, op)?,
        alignment: CUDA_ALLOCATION_ALIGNMENT,
    })
}

fn view_descriptor_alignment_requirement<T>() -> u32 {
    u32::try_from(std::mem::size_of::<T>()).unwrap_or(CUDA_ALLOCATION_ALIGNMENT)
}

fn validate_descriptor_alignment(
    actual_alignment: u32,
    alignment_requirement: u32,
    slot: &'static str,
    op: &'static str,
) -> crate::Result<()> {
    if actual_alignment >= alignment_requirement {
        return Ok(());
    }
    Err(Error::invalid_argument(
        op,
        "alignment",
        format!(
            "{slot} device pointer alignment {actual_alignment} is smaller than the cuTENSOR \
             descriptor requirement {alignment_requirement}"
        ),
    ))
}

fn raw_stream(rt: &CudaRuntime) -> crate::Result<CutensorCudaStream> {
    Ok(rt.raw_cuda_stream()? as usize as CutensorCudaStream)
}

fn dims_to_i64(op: &'static str, dims: &[usize]) -> crate::Result<Vec<i64>> {
    dims.iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| {
                Error::invalid_argument(
                    op,
                    "shape",
                    format!("dimension {dim} exceeds cuTENSOR i64 extent limit"),
                )
            })
        })
        .collect()
}

fn compact_strides_i64(op: &'static str, shape: &[usize]) -> crate::Result<Vec<i64>> {
    let strides = col_major_strides(shape)?;
    view_strides_i64(&strides, op)
}

fn view_strides_i64(strides: &[isize], op: &'static str) -> crate::Result<Vec<i64>> {
    strides
        .iter()
        .map(|&stride| {
            i64::try_from(stride).map_err(|_| {
                Error::invalid_argument(
                    op,
                    "layout",
                    format!("view stride {stride} exceeds cuTENSOR i64 stride limit"),
                )
            })
        })
        .collect()
}

fn identity_modes(op: &'static str, rank: usize) -> crate::Result<Vec<i32>> {
    (0..rank)
        .map(|axis| {
            i32::try_from(axis).map_err(|_| {
                Error::invalid_argument(
                    op,
                    "rank",
                    format!("rank {rank} exceeds cuTENSOR i32 mode limit"),
                )
            })
        })
        .collect()
}

fn modes_from_perm(op: &'static str, perm: &[usize]) -> crate::Result<Vec<i32>> {
    perm.iter()
        .map(|&axis| {
            i32::try_from(axis).map_err(|_| {
                Error::invalid_argument(
                    op,
                    "rank",
                    format!("axis {axis} exceeds cuTENSOR i32 mode limit"),
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::physical_output_descriptor;

    #[test]
    fn physical_output_descriptor_matches_transposed_destination() {
        let (extents, strides, modes) =
            physical_output_descriptor("test", &[4, 3], &[3, 1]).unwrap();
        assert_eq!(extents, [3, 4]);
        assert_eq!(strides, [1, 3]);
        assert_eq!(modes, [1, 0]);
    }

    #[test]
    fn physical_output_descriptor_preserves_identity_layout() {
        let (extents, strides, modes) =
            physical_output_descriptor("test", &[4, 3], &[1, 4]).unwrap();
        assert_eq!(extents, [4, 3]);
        assert_eq!(strides, [1, 4]);
        assert_eq!(modes, [0, 1]);
    }

    #[test]
    fn physical_output_descriptor_rejects_negative_strides() {
        let err = physical_output_descriptor("test", &[4], &[-1]).unwrap_err();
        assert!(err.to_string().contains("negative strides"));
    }
}
