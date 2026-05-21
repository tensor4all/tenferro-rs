use num_traits::{One, Zero};
use smallvec::{Array, SmallVec};
use std::mem::size_of;

use crate::buffer_pool::{BufferPool, PoolScalar};
use crate::cache::{CacheStats, RuntimeCacheControl};
use crate::config::DotGeneralConfig;
#[cfg(feature = "cpu-blas")]
use crate::cpu::elementwise::typed_conj_with_pool;
use crate::cpu::structural::typed_transpose_with_pool;
#[cfg(feature = "cpu-blas")]
use crate::types::ConjElem;
use crate::types::{
    col_major_strides, default_placement, Buffer, TensorRead, TensorView, TypedTensor,
    TypedTensorView,
};
use crate::Error;

#[cfg(feature = "cpu-blas")]
mod blas_gemm;
#[cfg(feature = "cpu-faer")]
mod faer_gemm;

#[cfg(feature = "cpu-blas")]
use blas_gemm::BlasGemm;
#[cfg(feature = "cpu-faer")]
use faer_gemm::FaerGemm;

#[derive(Clone)]
struct GemmDims {
    m: usize,
    n: usize,
    k: usize,
    batch_total: usize,
    a_rs: isize,
    a_cs: isize,
    a_bs: isize,
    b_rs: isize,
    b_cs: isize,
    b_bs: isize,
    c_rs: isize,
    c_cs: isize,
    c_bs: isize,
    out_shape: SmallVec<[usize; 8]>,
}

#[derive(Clone)]
struct GemmAnalysisPlan {
    lhs_shape: SmallVec<[usize; 8]>,
    rhs_shape: SmallVec<[usize; 8]>,
    config: GemmConfigKey,
    dims: Option<GemmDims>,
}

#[derive(Clone)]
struct GemmConfigKey {
    lhs_contracting_dims: SmallVec<[usize; 4]>,
    rhs_contracting_dims: SmallVec<[usize; 4]>,
    lhs_batch_dims: SmallVec<[usize; 4]>,
    rhs_batch_dims: SmallVec<[usize; 4]>,
}

pub(crate) const DEFAULT_GEMM_ANALYSIS_CACHE_CAPACITY: usize = 1024;

#[derive(Default)]
struct GemmAnalysisCacheSlot {
    direct: Option<GemmAnalysisPlan>,
    canonical: Option<GemmAnalysisPlan>,
}

#[derive(Clone, Copy)]
enum GemmAnalysisCacheKind {
    Direct,
    Canonical,
}

impl GemmAnalysisPlan {
    fn matches<L, R, T>(&self, lhs: &L, rhs: &R, config: &DotGeneralConfig) -> bool
    where
        L: TypedTensorRead<T>,
        R: TypedTensorRead<T>,
    {
        self.lhs_shape.as_slice() == lhs.shape()
            && self.rhs_shape.as_slice() == rhs.shape()
            && self.config.matches(config)
    }
}

impl GemmConfigKey {
    fn from_config(config: &DotGeneralConfig) -> Self {
        Self {
            lhs_contracting_dims: config.lhs_contracting_dims.iter().copied().collect(),
            rhs_contracting_dims: config.rhs_contracting_dims.iter().copied().collect(),
            lhs_batch_dims: config.lhs_batch_dims.iter().copied().collect(),
            rhs_batch_dims: config.rhs_batch_dims.iter().copied().collect(),
        }
    }

    fn matches(&self, config: &DotGeneralConfig) -> bool {
        self.lhs_contracting_dims.as_slice() == config.lhs_contracting_dims.as_slice()
            && self.rhs_contracting_dims.as_slice() == config.rhs_contracting_dims.as_slice()
            && self.lhs_batch_dims.as_slice() == config.lhs_batch_dims.as_slice()
            && self.rhs_batch_dims.as_slice() == config.rhs_batch_dims.as_slice()
    }
}

trait TypedTensorRead<T> {
    fn shape(&self) -> &[usize];
    fn host_data_opt(&self) -> Option<&[T]>;
}

impl<T> TypedTensorRead<T> for TypedTensor<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn host_data_opt(&self) -> Option<&[T]> {
        match &self.buffer {
            Buffer::Host(v) => Some(v),
            Buffer::Backend(_) => None,
            #[cfg(feature = "cubecl")]
            Buffer::Cubecl(_) => None,
        }
    }
}

impl<T> TypedTensorRead<T> for TypedTensorView<'_, T> {
    fn shape(&self) -> &[usize] {
        self.shape
    }

    fn host_data_opt(&self) -> Option<&[T]> {
        Some(self.data)
    }
}

impl<T: Clone> TypedTensorRead<T> for std::borrow::Cow<'_, TypedTensor<T>> {
    fn shape(&self) -> &[usize] {
        self.as_ref().shape()
    }

    fn host_data_opt(&self) -> Option<&[T]> {
        self.as_ref().host_data_opt()
    }
}

impl GemmAnalysisCacheSlot {
    fn get(&self, kind: GemmAnalysisCacheKind) -> Option<&GemmAnalysisPlan> {
        match kind {
            GemmAnalysisCacheKind::Direct => self.direct.as_ref(),
            GemmAnalysisCacheKind::Canonical => self.canonical.as_ref(),
        }
    }

    fn set(&mut self, kind: GemmAnalysisCacheKind, plan: GemmAnalysisPlan) {
        match kind {
            GemmAnalysisCacheKind::Direct => self.direct = Some(plan),
            GemmAnalysisCacheKind::Canonical => self.canonical = Some(plan),
        }
    }
}

#[doc(hidden)]
pub struct GemmAnalysisCache {
    slots: Vec<GemmAnalysisCacheSlot>,
    max_slots: usize,
}

impl GemmAnalysisCache {
    pub(crate) fn with_capacity(max_slots: usize) -> Self {
        Self {
            slots: Vec::new(),
            max_slots,
        }
    }

    #[doc(hidden)]
    pub fn capacity(&self) -> usize {
        self.max_slots
    }

    #[doc(hidden)]
    pub fn set_capacity(&mut self, max_slots: usize) {
        self.max_slots = max_slots;
        self.slots.truncate(max_slots);
    }

    fn cached_dims<L, R, T>(
        &self,
        slot: usize,
        kind: GemmAnalysisCacheKind,
        lhs: &L,
        rhs: &R,
        config: &DotGeneralConfig,
    ) -> Option<Option<GemmDims>>
    where
        L: TypedTensorRead<T>,
        R: TypedTensorRead<T>,
    {
        self.slots
            .get(slot)
            .and_then(|entry| entry.get(kind))
            .filter(|plan| plan.matches(lhs, rhs, config))
            .map(|plan| plan.dims.clone())
    }

    fn store(
        &mut self,
        slot: usize,
        kind: GemmAnalysisCacheKind,
        lhs_shape: SmallVec<[usize; 8]>,
        rhs_shape: SmallVec<[usize; 8]>,
        config: GemmConfigKey,
        dims: Option<GemmDims>,
    ) {
        if slot >= self.max_slots {
            return;
        }
        if self.slots.len() <= slot {
            self.slots.resize_with(slot + 1, Default::default);
        }
        self.slots[slot].set(
            kind,
            GemmAnalysisPlan {
                lhs_shape,
                rhs_shape,
                config,
                dims,
            },
        );
    }
}

impl Default for GemmAnalysisCache {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_GEMM_ANALYSIS_CACHE_CAPACITY)
    }
}

impl RuntimeCacheControl for GemmAnalysisCache {
    fn clear(&mut self) {
        self.slots.clear();
    }

    fn stats(&self) -> CacheStats {
        let mut entries = 0usize;
        let mut retained_bytes = self.slots.capacity() * size_of::<GemmAnalysisCacheSlot>();
        for slot in &self.slots {
            if let Some(plan) = &slot.direct {
                entries += 1;
                retained_bytes += gemm_analysis_plan_retained_bytes(plan);
            }
            if let Some(plan) = &slot.canonical {
                entries += 1;
                retained_bytes += gemm_analysis_plan_retained_bytes(plan);
            }
        }
        CacheStats {
            entries,
            retained_bytes,
        }
    }
}

fn smallvec_retained_bytes<A: Array>(values: &SmallVec<A>) -> usize {
    if values.spilled() {
        values.capacity() * size_of::<A::Item>()
    } else {
        0
    }
}

fn gemm_dims_retained_bytes(dims: &GemmDims) -> usize {
    smallvec_retained_bytes(&dims.out_shape)
}

fn gemm_config_key_retained_bytes(config: &GemmConfigKey) -> usize {
    smallvec_retained_bytes(&config.lhs_contracting_dims)
        + smallvec_retained_bytes(&config.rhs_contracting_dims)
        + smallvec_retained_bytes(&config.lhs_batch_dims)
        + smallvec_retained_bytes(&config.rhs_batch_dims)
}

fn gemm_analysis_plan_retained_bytes(plan: &GemmAnalysisPlan) -> usize {
    size_of::<GemmAnalysisPlan>()
        + smallvec_retained_bytes(&plan.lhs_shape)
        + smallvec_retained_bytes(&plan.rhs_shape)
        + gemm_config_key_retained_bytes(&plan.config)
        + plan.dims.as_ref().map_or(0, gemm_dims_retained_bytes)
}

fn validate_axis_list(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen: SmallVec<[bool; 8]> = smallvec::smallvec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(Error::AxisOutOfBounds { op, axis, rank });
        }
        if seen[axis] {
            return Err(Error::DuplicateAxis { op, axis, role });
        }
        seen[axis] = true;
    }
    Ok(())
}

fn validate_role_disjoint(
    op: &'static str,
    first_role: &'static str,
    first_axes: &[usize],
    second_role: &'static str,
    second_axes: &[usize],
) -> crate::Result<()> {
    for &axis in first_axes {
        if second_axes.contains(&axis) {
            return Err(Error::AxisRoleConflict {
                op,
                axis,
                first_role,
                second_role,
            });
        }
    }
    Ok(())
}

fn validate_dot_general<L, R, T>(lhs: &L, rhs: &R, config: &DotGeneralConfig) -> crate::Result<()>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    const OP: &str = "dot_general";
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs contracting dim counts differ".into(),
        });
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::InvalidConfig {
            op: OP,
            message: "lhs/rhs batch dim counts differ".into(),
        });
    }

    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();
    validate_axis_list(
        OP,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        lhs_rank,
    )?;
    validate_axis_list(
        OP,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        rhs_rank,
    )?;
    validate_axis_list(OP, "lhs_batch", &config.lhs_batch_dims, lhs_rank)?;
    validate_axis_list(OP, "rhs_batch", &config.rhs_batch_dims, rhs_rank)?;
    validate_role_disjoint(
        OP,
        "lhs_contracting",
        &config.lhs_contracting_dims,
        "lhs_batch",
        &config.lhs_batch_dims,
    )?;
    validate_role_disjoint(
        OP,
        "rhs_contracting",
        &config.rhs_contracting_dims,
        "rhs_batch",
        &config.rhs_batch_dims,
    )?;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: format!(
                    "contracting dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs_shape[lhs_axis], rhs_shape[rhs_axis]
                ),
            });
        }
    }
    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(Error::InvalidConfig {
                op: OP,
                message: format!(
                    "batch dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs_shape[lhs_axis], rhs_shape[rhs_axis]
                ),
            });
        }
    }

    Ok(())
}

fn checked_product(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn try_fuse_dims(shapes: &[usize], strides: &[isize]) -> Option<(usize, isize)> {
    if shapes.is_empty() {
        return Some((1, 0));
    }
    if shapes.len() == 1 {
        isize::try_from(shapes[0]).ok()?;
        return Some((shapes[0], strides[0]));
    }
    let mut dims: SmallVec<[(usize, isize); 8]> = shapes
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .collect();
    dims.sort_by_key(|&(_, stride)| stride.unsigned_abs());
    let base_stride = dims[0].1;
    let mut expected = base_stride;
    for (shape, stride) in dims {
        if stride != expected {
            return None;
        }
        let shape = isize::try_from(shape).ok()?;
        expected = stride.checked_mul(shape)?;
    }
    Some((checked_product(shapes)?, base_stride))
}

fn checked_batch_offset(batch: usize, stride: isize) -> Option<isize> {
    let batch = isize::try_from(batch).ok()?;
    batch.checked_mul(stride)
}

fn stride_sort_order(strides: &[isize]) -> SmallVec<[usize; 8]> {
    let mut order: SmallVec<[usize; 8]> = (0..strides.len()).collect();
    order.sort_by_key(|&idx| strides[idx].unsigned_abs());
    order
}

fn is_identity_order(order: &[usize]) -> bool {
    order.iter().enumerate().all(|(idx, &value)| idx == value)
}

/// Compute permutations that reorder lhs/rhs into canonical GEMM layout.
///
/// Canonical col-major layouts (batch trailing):
/// - lhs: `[free..., contract..., batch...]`
/// - rhs: `[contract..., free..., batch...]`
fn canonical_gemm_layout(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
) -> (SmallVec<[usize; 8]>, SmallVec<[usize; 8]>, DotGeneralConfig) {
    let lhs_free: SmallVec<[usize; 8]> = (0..lhs_rank)
        .filter(|d| !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d))
        .collect();
    let rhs_free: SmallVec<[usize; 8]> = (0..rhs_rank)
        .filter(|d| !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d))
        .collect();

    let mut lhs_perm = SmallVec::<[usize; 8]>::with_capacity(lhs_rank);
    lhs_perm.extend_from_slice(&lhs_free);
    lhs_perm.extend_from_slice(&config.lhs_contracting_dims);
    lhs_perm.extend_from_slice(&config.lhs_batch_dims);

    let mut rhs_perm = SmallVec::<[usize; 8]>::with_capacity(rhs_rank);
    rhs_perm.extend_from_slice(&config.rhs_contracting_dims);
    rhs_perm.extend_from_slice(&rhs_free);
    rhs_perm.extend_from_slice(&config.rhs_batch_dims);

    let nf_lhs = lhs_free.len();
    let nc = config.lhs_contracting_dims.len();
    let nb = config.lhs_batch_dims.len();
    let nf_rhs = rhs_free.len();

    let new_config = DotGeneralConfig {
        lhs_contracting_dims: (nf_lhs..nf_lhs + nc).collect(),
        rhs_contracting_dims: (0..nc).collect(),
        lhs_batch_dims: (nf_lhs + nc..nf_lhs + nc + nb).collect(),
        rhs_batch_dims: (nc + nf_rhs..nc + nf_rhs + nb).collect(),
    };

    (lhs_perm, rhs_perm, new_config)
}

fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

fn analyse_gemm<L, R, T>(lhs: &L, rhs: &R, config: &DotGeneralConfig) -> Option<GemmDims>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();

    let lhs_free: SmallVec<[usize; 8]> = (0..lhs_rank)
        .filter(|d| !config.lhs_contracting_dims.contains(d) && !config.lhs_batch_dims.contains(d))
        .collect();
    let rhs_free: SmallVec<[usize; 8]> = (0..rhs_rank)
        .filter(|d| !config.rhs_contracting_dims.contains(d) && !config.rhs_batch_dims.contains(d))
        .collect();

    let lhs_strides: SmallVec<[isize; 8]> = col_major_strides(lhs_shape).into_iter().collect();
    let rhs_strides: SmallVec<[isize; 8]> = col_major_strides(rhs_shape).into_iter().collect();

    let batch_shapes: SmallVec<[usize; 8]> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs_shape[d])
        .collect();
    let batch_total = checked_product(&batch_shapes)?;

    let lhs_free_shapes: SmallVec<[usize; 8]> = lhs_free.iter().map(|&d| lhs_shape[d]).collect();
    let rhs_free_shapes: SmallVec<[usize; 8]> = rhs_free.iter().map(|&d| rhs_shape[d]).collect();
    let contract_shapes: SmallVec<[usize; 8]> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs_shape[d])
        .collect();

    let m = checked_product(&lhs_free_shapes)?;
    let n = checked_product(&rhs_free_shapes)?;
    let k = checked_product(&contract_shapes)?;

    let lhs_free_strides: SmallVec<[isize; 8]> = lhs_free.iter().map(|&d| lhs_strides[d]).collect();
    let rhs_free_strides: SmallVec<[isize; 8]> = rhs_free.iter().map(|&d| rhs_strides[d]).collect();
    let lhs_contract_strides: SmallVec<[isize; 8]> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs_strides[d])
        .collect();
    let rhs_contract_strides: SmallVec<[isize; 8]> = config
        .rhs_contracting_dims
        .iter()
        .map(|&d| rhs_strides[d])
        .collect();
    let lhs_batch_strides: SmallVec<[isize; 8]> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs_strides[d])
        .collect();
    let rhs_batch_strides: SmallVec<[isize; 8]> = config
        .rhs_batch_dims
        .iter()
        .map(|&d| rhs_strides[d])
        .collect();

    if !is_identity_order(&stride_sort_order(&lhs_free_strides))
        || !is_identity_order(&stride_sort_order(&rhs_free_strides))
        || !is_identity_order(&stride_sort_order(&lhs_batch_strides))
        || !is_identity_order(&stride_sort_order(&rhs_batch_strides))
        || stride_sort_order(&lhs_contract_strides) != stride_sort_order(&rhs_contract_strides)
    {
        return None;
    }

    let (_, a_rs) = try_fuse_dims(&lhs_free_shapes, &lhs_free_strides)?;
    let (_, a_cs) = try_fuse_dims(&contract_shapes, &lhs_contract_strides)?;
    let (_, b_rs) = try_fuse_dims(&contract_shapes, &rhs_contract_strides)?;
    let (_, b_cs) = try_fuse_dims(&rhs_free_shapes, &rhs_free_strides)?;
    let (_, a_bs) = try_fuse_dims(&batch_shapes, &lhs_batch_strides)?;
    let (_, b_bs) = try_fuse_dims(&batch_shapes, &rhs_batch_strides)?;

    let mut out_shape = SmallVec::<[usize; 8]>::new();
    out_shape.extend_from_slice(&lhs_free_shapes);
    out_shape.extend_from_slice(&rhs_free_shapes);
    out_shape.extend_from_slice(&batch_shapes);

    let out_strides: SmallVec<[isize; 8]> = col_major_strides(&out_shape).into_iter().collect();
    let nm = lhs_free_shapes.len();
    let nn = rhs_free_shapes.len();
    let out_m_shapes = &out_shape[..nm];
    let out_m_strides = &out_strides[..nm];
    let out_n_shapes = &out_shape[nm..nm + nn];
    let out_n_strides = &out_strides[nm..nm + nn];
    let out_b_shapes = &out_shape[nm + nn..];
    let out_b_strides = &out_strides[nm + nn..];

    let (_, c_rs) = try_fuse_dims(out_m_shapes, out_m_strides)?;
    let (_, c_cs) = try_fuse_dims(out_n_shapes, out_n_strides)?;
    let (_, c_bs) = try_fuse_dims(out_b_shapes, out_b_strides)?;

    Some(GemmDims {
        m,
        n,
        k,
        batch_total,
        a_rs,
        a_cs,
        a_bs,
        b_rs,
        b_cs,
        b_bs,
        c_rs,
        c_cs,
        c_bs,
        out_shape,
    })
}

fn analyse_gemm_cached<L, R, T>(
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    cache_kind: GemmAnalysisCacheKind,
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
) -> crate::Result<Option<GemmDims>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    let cache_slot = cache_slot.filter(|&slot| slot < cache.max_slots);
    if let Some(slot) = cache_slot {
        if let Some(cached) = cache.cached_dims(slot, cache_kind, lhs, rhs, config) {
            return Ok(cached);
        }
    }

    validate_dot_general(lhs, rhs, config)?;
    let dims = analyse_gemm(lhs, rhs, config);
    if let Some(slot) = cache_slot {
        cache.store(
            slot,
            cache_kind,
            lhs.shape().iter().copied().collect(),
            rhs.shape().iter().copied().collect(),
            GemmConfigKey::from_config(config),
            dims.clone(),
        );
    }
    Ok(dims)
}

#[cfg(feature = "cpu-faer")]
#[allow(dead_code)]
pub(crate) fn dot_general<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    ctx: &crate::cpu::CpuContext,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    dot_general_cached(buffers, cache, None, ctx, lhs, rhs, config)
}

#[cfg(feature = "cpu-faer")]
pub(crate) fn dot_general_cached<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    ctx: &crate::cpu::CpuContext,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    dot_general_with_conj_cached(
        buffers, cache, cache_slot, ctx, lhs, rhs, config, false, false,
    )
}

#[cfg(feature = "cpu-faer")]
#[allow(dead_code)]
pub(crate) fn dot_general_with_conj<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    ctx: &crate::cpu::CpuContext,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<T>>
where
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    dot_general_with_conj_cached(
        buffers, cache, None, ctx, lhs, rhs, config, lhs_conj, rhs_conj,
    )
}

#[cfg(feature = "cpu-faer")]
pub(crate) fn dot_general_with_conj_cached<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    ctx: &crate::cpu::CpuContext,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<T>>
where
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    if let Some(result) = typed_faer_gemm(
        buffers,
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Direct,
        ctx,
        lhs,
        rhs,
        config,
        lhs_conj,
        rhs_conj,
    )? {
        return Ok(result);
    }
    let (lhs_perm, rhs_perm, new_config) =
        canonical_gemm_layout(config, lhs.shape.len(), rhs.shape.len());
    let lhs_canon = if is_identity_perm(&lhs_perm) {
        std::borrow::Cow::Borrowed(lhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose_with_pool(buffers, lhs, &lhs_perm)?)
    };
    let rhs_canon = if is_identity_perm(&rhs_perm) {
        std::borrow::Cow::Borrowed(rhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose_with_pool(buffers, rhs, &rhs_perm)?)
    };
    typed_faer_gemm(
        buffers,
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Canonical,
        ctx,
        &lhs_canon,
        &rhs_canon,
        &new_config,
        lhs_conj,
        rhs_conj,
    )?
    .ok_or_else(|| Error::BackendFailure {
        op: "dot_general",
        message: "CPU GEMM requires host-backed canonical inputs".into(),
    })
}

#[cfg(feature = "cpu-faer")]
pub(crate) fn dot_general_read_cached(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    ctx: &crate::cpu::CpuContext,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
) -> crate::Result<Option<crate::Tensor>> {
    macro_rules! dispatch {
        ($owned:ident, $view:ident, $wrap:ident) => {
            match (lhs, rhs) {
                (
                    TensorRead::Tensor(crate::Tensor::$owned(a)),
                    TensorRead::Tensor(crate::Tensor::$owned(b)),
                ) => {
                    return typed_faer_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        ctx,
                        a,
                        b,
                        config,
                        false,
                        false,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                (
                    TensorRead::Tensor(crate::Tensor::$owned(a)),
                    TensorRead::View(TensorView::$view(b)),
                ) => {
                    return typed_faer_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        ctx,
                        a,
                        &b,
                        config,
                        false,
                        false,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                (
                    TensorRead::View(TensorView::$view(a)),
                    TensorRead::Tensor(crate::Tensor::$owned(b)),
                ) => {
                    return typed_faer_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        ctx,
                        &a,
                        b,
                        config,
                        false,
                        false,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                (
                    TensorRead::View(TensorView::$view(a)),
                    TensorRead::View(TensorView::$view(b)),
                ) => {
                    return typed_faer_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        ctx,
                        &a,
                        &b,
                        config,
                        false,
                        false,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                _ => {}
            }
        };
    }

    dispatch!(F32, F32, F32);
    dispatch!(F64, F64, F64);
    dispatch!(C32, C32, C32);
    dispatch!(C64, C64, C64);

    if lhs.dtype() == rhs.dtype() {
        Ok(None)
    } else {
        Err(Error::DTypeMismatch {
            op: "dot_general",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })
    }
}

#[cfg(feature = "cpu-faer")]
fn typed_faer_gemm<L, R, T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    cache_kind: GemmAnalysisCacheKind,
    ctx: &crate::cpu::CpuContext,
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<Option<TypedTensor<T>>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: FaerGemm + PoolScalar + Copy + Clone + Zero + One + PartialEq,
{
    let Some(dims) = analyse_gemm_cached(cache, cache_slot, cache_kind, lhs, rhs, config)? else {
        return Ok(None);
    };
    let out_n = checked_product(&dims.out_shape).ok_or_else(|| Error::BackendFailure {
        op: "dot_general",
        message: "output element count overflow".into(),
    })?;
    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        // SAFETY: the buffer is fully initialized immediately by fill.
        let mut data = unsafe { T::pool_acquire(buffers, out_n) };
        data.fill(T::zero());
        return Ok(Some(TypedTensor {
            buffer: Buffer::Host(data),
            shape: dims.out_shape.into_vec(),
            placement: default_placement(),
        }));
    }

    let Some(a_data) = lhs.host_data_opt().map(<[T]>::as_ptr) else {
        return Ok(None);
    };
    let Some(b_data) = rhs.host_data_opt().map(<[T]>::as_ptr) else {
        return Ok(None);
    };

    // SAFETY: this GEMM path uses beta = 0 and overwrites every output element.
    let mut out_data: Vec<T> = unsafe { T::pool_acquire(buffers, out_n) };
    let c_ptr = out_data.as_mut_ptr();

    for batch in 0..dims.batch_total {
        let a_off =
            checked_batch_offset(batch, dims.a_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "lhs batch offset overflow".into(),
            })?;
        let b_off =
            checked_batch_offset(batch, dims.b_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "rhs batch offset overflow".into(),
            })?;
        let c_off =
            checked_batch_offset(batch, dims.c_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "output batch offset overflow".into(),
            })?;
        unsafe {
            if lhs_conj || rhs_conj {
                T::strided_gemm_with_conj(
                    ctx,
                    T::one(),
                    a_data.offset(a_off),
                    dims.m,
                    dims.k,
                    dims.a_rs,
                    dims.a_cs,
                    lhs_conj,
                    b_data.offset(b_off),
                    dims.n,
                    dims.b_rs,
                    dims.b_cs,
                    rhs_conj,
                    T::zero(),
                    c_ptr.offset(c_off),
                    dims.c_rs,
                    dims.c_cs,
                );
            } else {
                T::strided_gemm(
                    ctx,
                    T::one(),
                    a_data.offset(a_off),
                    dims.m,
                    dims.k,
                    dims.a_rs,
                    dims.a_cs,
                    b_data.offset(b_off),
                    dims.n,
                    dims.b_rs,
                    dims.b_cs,
                    T::zero(),
                    c_ptr.offset(c_off),
                    dims.c_rs,
                    dims.c_cs,
                );
            }
        }
    }

    Ok(Some(TypedTensor {
        buffer: Buffer::Host(out_data),
        shape: dims.out_shape.into_vec(),
        placement: default_placement(),
    }))
}

#[cfg(feature = "cpu-blas")]
#[allow(dead_code)]
pub(crate) fn dot_general<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One,
{
    dot_general_cached(buffers, cache, None, lhs, rhs, config)
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn dot_general_cached<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One,
{
    if let Some(result) = typed_blas_gemm(
        buffers,
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Direct,
        lhs,
        rhs,
        config,
    )? {
        return Ok(result);
    }
    let (lhs_perm, rhs_perm, new_config) =
        canonical_gemm_layout(config, lhs.shape.len(), rhs.shape.len());
    let lhs_canon = if is_identity_perm(&lhs_perm) {
        std::borrow::Cow::Borrowed(lhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose_with_pool(buffers, lhs, &lhs_perm)?)
    };
    let rhs_canon = if is_identity_perm(&rhs_perm) {
        std::borrow::Cow::Borrowed(rhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose_with_pool(buffers, rhs, &rhs_perm)?)
    };
    typed_blas_gemm(
        buffers,
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Canonical,
        &lhs_canon,
        &rhs_canon,
        &new_config,
    )?
    .ok_or_else(|| Error::BackendFailure {
        op: "dot_general",
        message: "CPU GEMM requires host-backed canonical inputs".into(),
    })
}

#[cfg(feature = "cpu-blas")]
#[allow(dead_code)]
pub(crate) fn dot_general_with_conj<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<T>>
where
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One + ConjElem,
{
    dot_general_with_conj_cached(buffers, cache, None, lhs, rhs, config, lhs_conj, rhs_conj)
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn dot_general_with_conj_cached<T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<T>>
where
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One + ConjElem,
{
    if !lhs_conj && !rhs_conj {
        return dot_general_cached(buffers, cache, cache_slot, lhs, rhs, config);
    }

    if let Some(result) = typed_blas_gemm_with_conj(
        buffers,
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Direct,
        lhs,
        rhs,
        config,
        lhs_conj,
        rhs_conj,
    )? {
        return Ok(result);
    }
    let (lhs_perm, rhs_perm, new_config) =
        canonical_gemm_layout(config, lhs.shape.len(), rhs.shape.len());
    let lhs_canon = if is_identity_perm(&lhs_perm) {
        std::borrow::Cow::Borrowed(lhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose_with_pool(buffers, lhs, &lhs_perm)?)
    };
    let rhs_canon = if is_identity_perm(&rhs_perm) {
        std::borrow::Cow::Borrowed(rhs)
    } else {
        std::borrow::Cow::Owned(typed_transpose_with_pool(buffers, rhs, &rhs_perm)?)
    };
    if let Some(result) = typed_blas_gemm_with_conj(
        buffers,
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Canonical,
        &lhs_canon,
        &rhs_canon,
        &new_config,
        lhs_conj,
        rhs_conj,
    )? {
        return Ok(result);
    }

    let lhs_tmp;
    let lhs_ref = if lhs_conj {
        lhs_tmp = typed_conj_with_pool(buffers, lhs)?;
        &lhs_tmp
    } else {
        lhs
    };
    let rhs_tmp;
    let rhs_ref = if rhs_conj {
        rhs_tmp = typed_conj_with_pool(buffers, rhs)?;
        &rhs_tmp
    } else {
        rhs
    };
    dot_general_cached(buffers, cache, cache_slot, lhs_ref, rhs_ref, config)
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn dot_general_read_cached(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    lhs: TensorRead<'_>,
    rhs: TensorRead<'_>,
    config: &DotGeneralConfig,
) -> crate::Result<Option<crate::Tensor>> {
    macro_rules! dispatch {
        ($owned:ident, $view:ident, $wrap:ident) => {
            match (lhs, rhs) {
                (
                    TensorRead::Tensor(crate::Tensor::$owned(a)),
                    TensorRead::Tensor(crate::Tensor::$owned(b)),
                ) => {
                    return typed_blas_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        a,
                        b,
                        config,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                (
                    TensorRead::Tensor(crate::Tensor::$owned(a)),
                    TensorRead::View(TensorView::$view(b)),
                ) => {
                    return typed_blas_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        a,
                        &b,
                        config,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                (
                    TensorRead::View(TensorView::$view(a)),
                    TensorRead::Tensor(crate::Tensor::$owned(b)),
                ) => {
                    return typed_blas_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        &a,
                        b,
                        config,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                (
                    TensorRead::View(TensorView::$view(a)),
                    TensorRead::View(TensorView::$view(b)),
                ) => {
                    return typed_blas_gemm(
                        buffers,
                        cache,
                        cache_slot,
                        GemmAnalysisCacheKind::Direct,
                        &a,
                        &b,
                        config,
                    )
                    .map(|result| result.map(crate::Tensor::$wrap));
                }
                _ => {}
            }
        };
    }

    dispatch!(F32, F32, F32);
    dispatch!(F64, F64, F64);
    dispatch!(C32, C32, C32);
    dispatch!(C64, C64, C64);

    if lhs.dtype() == rhs.dtype() {
        Ok(None)
    } else {
        Err(Error::DTypeMismatch {
            op: "dot_general",
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        })
    }
}

#[cfg(feature = "cpu-blas")]
fn typed_blas_gemm_with_conj<L, R, T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    cache_kind: GemmAnalysisCacheKind,
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<Option<TypedTensor<T>>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One,
{
    let dims = match analyse_gemm_cached(cache, cache_slot, cache_kind, lhs, rhs, config)? {
        Some(dims) => dims,
        None => return Ok(None),
    };
    let out_n = checked_product(&dims.out_shape).ok_or_else(|| Error::BackendFailure {
        op: "dot_general",
        message: "output element count overflow".into(),
    })?;
    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        // SAFETY: the buffer is fully initialized immediately by fill.
        let mut data = unsafe { T::pool_acquire(buffers, out_n) };
        data.fill(T::zero());
        return Ok(Some(TypedTensor {
            buffer: Buffer::Host(data),
            shape: dims.out_shape.into_vec(),
            placement: default_placement(),
        }));
    }

    let a_rs = normalize_singleton_stride(dims.a_rs, dims.m, dims.k);
    let a_cs = normalize_singleton_stride(dims.a_cs, dims.k, dims.m);
    let b_rs = normalize_singleton_stride(dims.b_rs, dims.k, dims.n);
    let b_cs = normalize_singleton_stride(dims.b_cs, dims.n, dims.k);
    let c_rs = normalize_singleton_stride(dims.c_rs, dims.m, 1);
    let c_cs = normalize_singleton_stride(dims.c_cs, dims.n, dims.m);

    let a_ok = a_rs == 1 || a_cs == 1;
    let b_ok = b_rs == 1 || b_cs == 1;
    let c_ok = c_rs == 1;
    if !a_ok || !b_ok || !c_ok {
        return Ok(None);
    }
    // `BlasGemm::strided_gemm_with_conj` maps row-contiguous operands to
    // CblasNoTrans, which cannot express conjugation without materializing.
    // Detect that before acquiring the output buffer for a path that will
    // return Ok(false).
    if (lhs_conj && a_rs == 1) || (rhs_conj && b_rs == 1) {
        return Ok(None);
    }

    let Some(a_data) = lhs.host_data_opt().map(<[T]>::as_ptr) else {
        return Ok(None);
    };
    let Some(b_data) = rhs.host_data_opt().map(<[T]>::as_ptr) else {
        return Ok(None);
    };

    // SAFETY: each batch GEMM writes its full output block with beta = 0 when
    // the BLAS layout can represent the requested conjugation.
    let mut out: Vec<T> = unsafe { T::pool_acquire(buffers, out_n) };
    let c_ptr = out.as_mut_ptr();

    for batch in 0..dims.batch_total {
        let a_off =
            checked_batch_offset(batch, dims.a_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "lhs batch offset overflow".into(),
            })?;
        let b_off =
            checked_batch_offset(batch, dims.b_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "rhs batch offset overflow".into(),
            })?;
        let c_off =
            checked_batch_offset(batch, dims.c_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "output batch offset overflow".into(),
            })?;
        let executed = unsafe {
            T::strided_gemm_with_conj(
                T::one(),
                a_data.offset(a_off),
                dims.m,
                dims.k,
                a_rs,
                a_cs,
                lhs_conj,
                b_data.offset(b_off),
                dims.n,
                b_rs,
                b_cs,
                rhs_conj,
                T::zero(),
                c_ptr.offset(c_off),
                c_rs,
                c_cs,
            )?
        };
        if !executed {
            return Ok(None);
        }
    }

    Ok(Some(TypedTensor {
        buffer: Buffer::Host(out),
        shape: dims.out_shape.into_vec(),
        placement: default_placement(),
    }))
}

#[cfg(feature = "cpu-blas")]
fn typed_blas_gemm<L, R, T>(
    buffers: &mut BufferPool,
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    cache_kind: GemmAnalysisCacheKind,
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
) -> crate::Result<Option<TypedTensor<T>>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: BlasGemm + PoolScalar + Copy + Clone + Zero + One,
{
    let dims = match analyse_gemm_cached(cache, cache_slot, cache_kind, lhs, rhs, config)? {
        Some(dims) => dims,
        None => return Ok(None),
    };
    let out_n = checked_product(&dims.out_shape).ok_or_else(|| Error::BackendFailure {
        op: "dot_general",
        message: "output element count overflow".into(),
    })?;
    if dims.m == 0 || dims.n == 0 || dims.k == 0 || dims.batch_total == 0 {
        // SAFETY: the buffer is fully initialized immediately by fill.
        let mut data = unsafe { T::pool_acquire(buffers, out_n) };
        data.fill(T::zero());
        return Ok(Some(TypedTensor {
            buffer: Buffer::Host(data),
            shape: dims.out_shape.into_vec(),
            placement: default_placement(),
        }));
    }

    let a_rs = normalize_singleton_stride(dims.a_rs, dims.m, dims.k);
    let a_cs = normalize_singleton_stride(dims.a_cs, dims.k, dims.m);
    let b_rs = normalize_singleton_stride(dims.b_rs, dims.k, dims.n);
    let b_cs = normalize_singleton_stride(dims.b_cs, dims.n, dims.k);
    let c_rs = normalize_singleton_stride(dims.c_rs, dims.m, 1);
    let c_cs = normalize_singleton_stride(dims.c_cs, dims.n, dims.m);

    let a_ok = a_rs == 1 || a_cs == 1;
    let b_ok = b_rs == 1 || b_cs == 1;
    let c_ok = c_rs == 1;
    if !a_ok || !b_ok || !c_ok {
        return Ok(None);
    }

    let Some(a_data) = lhs.host_data_opt().map(<[T]>::as_ptr) else {
        return Ok(None);
    };
    let Some(b_data) = rhs.host_data_opt().map(<[T]>::as_ptr) else {
        return Ok(None);
    };

    // SAFETY: each batch GEMM writes its full output block with beta = 0.
    let mut out: Vec<T> = unsafe { T::pool_acquire(buffers, out_n) };
    let c_ptr = out.as_mut_ptr();

    for batch in 0..dims.batch_total {
        let a_off =
            checked_batch_offset(batch, dims.a_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "lhs batch offset overflow".into(),
            })?;
        let b_off =
            checked_batch_offset(batch, dims.b_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "rhs batch offset overflow".into(),
            })?;
        let c_off =
            checked_batch_offset(batch, dims.c_bs).ok_or_else(|| Error::BackendFailure {
                op: "dot_general",
                message: "output batch offset overflow".into(),
            })?;
        unsafe {
            T::strided_gemm(
                T::one(),
                a_data.offset(a_off),
                dims.m,
                dims.k,
                a_rs,
                a_cs,
                b_data.offset(b_off),
                dims.n,
                b_rs,
                b_cs,
                T::zero(),
                c_ptr.offset(c_off),
                c_rs,
                c_cs,
            )?;
        }
    }

    Ok(Some(TypedTensor {
        buffer: Buffer::Host(out),
        shape: dims.out_shape.into_vec(),
        placement: default_placement(),
    }))
}

fn normalize_singleton_stride(stride: isize, extent: usize, fallback: usize) -> isize {
    if extent == 1 {
        let fallback = fallback.max(1) as isize;
        stride.max(fallback)
    } else {
        stride
    }
}

#[cfg(test)]
mod tests;
