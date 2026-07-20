#[cfg(feature = "cpu-tblis-provider")]
use num_traits::One;
use num_traits::Zero;
use smallvec::{Array, SmallVec};
use std::fmt;
use std::mem::size_of;
use std::sync::{Arc, Weak};

use crate::dot_runtime::CpuProviderBundleInner;
#[cfg(feature = "cpu-faer")]
use crate::provider::CpuKernelParallelism;
#[cfg(any(feature = "cpu-blas", feature = "cpu-tblis-provider"))]
use crate::provider::CpuOperand;
#[cfg(feature = "cpu-tblis-provider")]
use crate::provider::{CpuContractionAxes, CpuDotGeneralRequest};
use crate::provider::{
    CpuGemmRequest, CpuGroupedGemmRequest, CpuProviderContext, CpuProviderOutcome,
    CpuProviderUnsupported,
};
use crate::{Error, Result};
use tenferro_tensor::backend::GroupedGemmConfig;
use tenferro_tensor::{
    col_major_strides, Buffer, TensorRead, TensorView, TensorViewMut, TensorWrite, TypedTensor,
    TypedTensorView, TypedTensorViewMut, ValidationError,
};
use tenferro_tensor::{CacheStats, RuntimeCacheControl};
use tenferro_tensor::{ContractionScalar, DotGeneralAccumulation, DotGeneralConfig};

#[cfg(feature = "cpu-blas")]
mod blas_gemm;
#[cfg(feature = "cpu-faer")]
mod faer_gemm;
#[cfg(feature = "cpu-tblis-provider")]
mod tblis_gemm;

#[cfg(feature = "cpu-blas")]
use blas_gemm::BlasGemm;
#[cfg(feature = "cpu-blas")]
use blas_gemm::BlasGemmBatch;
#[cfg(feature = "cpu-faer")]
use faer_gemm::FaerGemm;
#[cfg(feature = "cpu-tblis-provider")]
use tblis_gemm::TblisGemm;

const OP: &str = "dot_general";

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
    out_shape: SmallVec<[usize; 8]>,
}

#[derive(Clone)]
struct GemmAnalysisPlan {
    lhs_shape: SmallVec<[usize; 8]>,
    rhs_shape: SmallVec<[usize; 8]>,
    lhs_strides: SmallVec<[isize; 8]>,
    rhs_strides: SmallVec<[isize; 8]>,
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
    fn matches<L, R, T>(&self, lhs: &L, rhs: &R, config: &DotGeneralConfig) -> crate::Result<bool>
    where
        L: TypedTensorRead<T>,
        R: TypedTensorRead<T>,
    {
        Ok(self.lhs_shape.as_slice() == lhs.shape()
            && self.rhs_shape.as_slice() == rhs.shape()
            && self.lhs_strides.as_slice() == lhs.strides()?.as_slice()
            && self.rhs_strides.as_slice() == rhs.strides()?.as_slice()
            && self.config.matches(config))
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
    fn strides(&self) -> crate::Result<SmallVec<[isize; 8]>>;
    fn offset(&self) -> isize;
    fn host_data_opt(&self) -> crate::Result<Option<&[T]>>;
}

impl<T: Clone> TypedTensorRead<T> for TypedTensor<T> {
    fn shape(&self) -> &[usize] {
        self.layout().shape()
    }

    fn strides(&self) -> crate::Result<SmallVec<[isize; 8]>> {
        Ok(col_major_strides(self.shape())?.into_iter().collect())
    }

    fn offset(&self) -> isize {
        0
    }

    fn host_data_opt(&self) -> crate::Result<Option<&[T]>> {
        Ok(match self.buffer() {
            Buffer::Host(v) => Some(v.as_slice()),
            Buffer::Backend(_) => None,
        })
    }
}

impl<T: 'static> TypedTensorRead<T> for TypedTensorView<'_, T> {
    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn strides(&self) -> crate::Result<SmallVec<[isize; 8]>> {
        Ok(self.strides().iter().copied().collect())
    }

    fn offset(&self) -> isize {
        self.offset()
    }

    fn host_data_opt(&self) -> crate::Result<Option<&[T]>> {
        if self.backend_buffer().is_some() {
            Ok(None)
        } else {
            Ok(Some(self.host_storage()?))
        }
    }
}

impl<T: Clone> TypedTensorRead<T> for std::borrow::Cow<'_, TypedTensor<T>> {
    fn shape(&self) -> &[usize] {
        self.as_ref().shape()
    }

    fn strides(&self) -> crate::Result<SmallVec<[isize; 8]>> {
        self.as_ref().strides()
    }

    fn offset(&self) -> isize {
        self.as_ref().offset()
    }

    fn host_data_opt(&self) -> crate::Result<Option<&[T]>> {
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
    provider_bundle: Option<Weak<CpuProviderBundleInner>>,
}

impl fmt::Debug for GemmAnalysisCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GemmAnalysisCache")
            .field("slots_len", &self.slots.len())
            .field("max_slots", &self.max_slots)
            .field("provider_bundle_bound", &self.provider_bundle.is_some())
            .finish_non_exhaustive()
    }
}

impl GemmAnalysisCache {
    pub(crate) fn with_capacity(max_slots: usize) -> Self {
        Self {
            slots: Vec::new(),
            max_slots,
            provider_bundle: None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn bind_provider_bundle(&mut self, bundle: &Arc<CpuProviderBundleInner>) {
        let matches = self
            .provider_bundle
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some_and(|current| Arc::ptr_eq(&current, bundle));
        if !matches {
            self.slots.clear();
            self.provider_bundle = Some(Arc::downgrade(bundle));
        }
    }

    #[doc(hidden)]
    pub fn capacity(&self) -> usize {
        self.max_slots
    }

    #[doc(hidden)]
    pub fn set_capacity(&mut self, max_slots: usize) {
        if max_slots < self.max_slots {
            self.slots.clear();
            self.slots.shrink_to(max_slots);
        }
        self.max_slots = max_slots;
    }

    fn cached_dims<L, R, T>(
        &self,
        slot: usize,
        kind: GemmAnalysisCacheKind,
        lhs: &L,
        rhs: &R,
        config: &DotGeneralConfig,
    ) -> crate::Result<Option<Option<GemmDims>>>
    where
        L: TypedTensorRead<T>,
        R: TypedTensorRead<T>,
    {
        let Some(entry) = self.slots.get(slot).and_then(|entry| entry.get(kind)) else {
            return Ok(None);
        };
        if entry.matches(lhs, rhs, config)? {
            Ok(Some(entry.dims.clone()))
        } else {
            Ok(None)
        }
    }

    // Cache entries mirror the full analyzed GEMM key to avoid rebuilding a temporary key object.
    #[allow(clippy::too_many_arguments)]
    fn store(
        &mut self,
        slot: usize,
        kind: GemmAnalysisCacheKind,
        lhs_shape: SmallVec<[usize; 8]>,
        rhs_shape: SmallVec<[usize; 8]>,
        lhs_strides: SmallVec<[isize; 8]>,
        rhs_strides: SmallVec<[isize; 8]>,
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
                lhs_strides,
                rhs_strides,
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
        let mut retained_bytes = self.slots.capacity() * size_of::<GemmAnalysisCacheSlot>()
            + self
                .provider_bundle
                .as_ref()
                .map_or(0, |_| size_of::<Weak<CpuProviderBundleInner>>());
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
        + smallvec_retained_bytes(&plan.lhs_strides)
        + smallvec_retained_bytes(&plan.rhs_strides)
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
            return Err(Error::axis_out_of_bounds(op, axis, rank));
        }
        if seen[axis] {
            return Err(Error::duplicate_axis(op, axis, role));
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
            return Err(Error::validation(
                op,
                ValidationError::AxisRoleConflict {
                    axis,
                    first_role,
                    second_role,
                },
            ));
        }
    }
    Ok(())
}

fn validate_dot_general<L, R, T>(lhs: &L, rhs: &R, config: &DotGeneralConfig) -> crate::Result<()>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "lhs/rhs contracting dim counts differ",
        ));
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "configuration",
            "lhs/rhs batch dim counts differ",
        ));
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
            return Err(Error::invalid_argument(
                OP,
                "configuration",
                format!(
                    "contracting dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs_shape[lhs_axis], rhs_shape[rhs_axis]
                ),
            ));
        }
    }
    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(Error::invalid_argument(
                OP,
                "configuration",
                format!(
                    "batch dim size mismatch: lhs axis {lhs_axis}={} rhs axis {rhs_axis}={}",
                    lhs_shape[lhs_axis], rhs_shape[rhs_axis]
                ),
            ));
        }
    }

    Ok(())
}

fn checked_product(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

fn try_fuse_dims(shapes: &[usize], strides: &[isize]) -> Result<Option<(usize, isize)>> {
    if shapes.is_empty() {
        return Ok(Some((1, 0)));
    }
    if shapes.len() == 1 {
        dim_to_isize(shapes[0], "try_fuse_dims")?;
        return Ok(Some((shapes[0], strides[0])));
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
            return Ok(None);
        }
        let shape = dim_to_isize(shape, "try_fuse_dims")?;
        expected = stride.checked_mul(shape).ok_or_else(|| {
            Error::invalid_argument(
                OP,
                "configuration",
                format!("fused stride overflows isize: stride={stride} shape={shape}"),
            )
        })?;
    }
    let fused = checked_product(shapes).ok_or_else(|| {
        Error::invalid_argument(
            OP,
            "configuration",
            format!("fused dimension product overflows usize for shape {shapes:?}"),
        )
    })?;
    Ok(Some((fused, base_stride)))
}

fn checked_batch_offset(batch: usize, stride: isize) -> Result<isize> {
    let batch_isize = isize::try_from(batch).map_err(|_| {
        Error::invalid_argument(
            OP,
            "configuration",
            format!("batch index {batch} does not fit in isize"),
        )
    })?;
    batch_isize.checked_mul(stride).ok_or_else(|| {
        Error::invalid_argument(
            OP,
            "configuration",
            format!("batch offset overflows isize: batch={batch} stride={stride}"),
        )
    })
}

fn checked_view_batch_offset(base: isize, batch: usize, stride: isize) -> Result<isize> {
    base.checked_add(checked_batch_offset(batch, stride)?)
        .ok_or_else(|| {
            Error::invalid_argument(
                OP,
                "configuration",
                format!(
                    "view batch offset overflows isize: base={base} batch={batch} stride={stride}"
                ),
            )
        })
}

fn output_gemm_strides(
    out_shape: &[usize],
    out_strides: &[isize],
    lhs_rank: usize,
    rhs_rank: usize,
    config: &DotGeneralConfig,
) -> crate::Result<Option<(isize, isize, isize)>> {
    let nm = lhs_rank
        .checked_sub(config.lhs_contracting_dims.len() + config.lhs_batch_dims.len())
        .ok_or_else(|| {
            Error::invalid_argument(
                OP,
                "configuration",
                "lhs free rank underflow while analyzing output strides",
            )
        })?;
    let nn = rhs_rank
        .checked_sub(config.rhs_contracting_dims.len() + config.rhs_batch_dims.len())
        .ok_or_else(|| {
            Error::invalid_argument(
                OP,
                "configuration",
                "rhs free rank underflow while analyzing output strides",
            )
        })?;
    let nb = config.lhs_batch_dims.len();
    if out_shape.len() != nm + nn + nb || out_strides.len() != out_shape.len() {
        return Ok(None);
    }

    let out_m_shapes = &out_shape[..nm];
    let out_m_strides = &out_strides[..nm];
    let out_n_shapes = &out_shape[nm..nm + nn];
    let out_n_strides = &out_strides[nm..nm + nn];
    let out_b_shapes = &out_shape[nm + nn..];
    let out_b_strides = &out_strides[nm + nn..];

    let Some((_, c_rs)) = try_fuse_dims(out_m_shapes, out_m_strides)? else {
        return Ok(None);
    };
    let Some((_, c_cs)) = try_fuse_dims(out_n_shapes, out_n_strides)? else {
        return Ok(None);
    };
    let Some((_, c_bs)) = try_fuse_dims(out_b_shapes, out_b_strides)? else {
        return Ok(None);
    };
    Ok(Some((c_rs, c_cs, c_bs)))
}

fn scale_empty_contract_output<T>(out: &mut TypedTensorViewMut<'_, T>, beta: T) -> crate::Result<()>
where
    T: Copy + Zero + PartialEq + std::ops::Mul<Output = T> + 'static,
{
    let beta_is_zero = beta == T::zero();
    let shape = out.shape().to_vec();
    let element_count = checked_product(&shape).ok_or_else(|| {
        Error::invalid_argument(
            OP,
            "configuration",
            "output element count overflow while scaling empty contraction output",
        )
    })?;
    for linear in 0..element_count {
        let indices = flat_to_multi_for_shape(&shape, linear);
        let output = out.get_mut(&indices).ok_or_else(|| {
            Error::invalid_argument(
                OP,
                "configuration",
                format!("output index {indices:?} is outside accumulation target"),
            )
        })?;
        // INVARIANT: beta == 0 follows GEMM semantics and overwrites the
        // destination with zero without reading its previous value.
        *output = if beta_is_zero {
            T::zero()
        } else {
            beta * *output
        };
    }
    Ok(())
}

fn flat_to_multi_for_shape(shape: &[usize], mut linear: usize) -> SmallVec<[usize; 8]> {
    let mut indices = SmallVec::<[usize; 8]>::with_capacity(shape.len());
    for &dim in shape {
        if dim == 0 {
            indices.push(0);
        } else {
            indices.push(linear % dim);
            linear /= dim;
        }
    }
    indices
}

fn dim_to_isize(dim: usize, context: &'static str) -> Result<isize> {
    isize::try_from(dim).map_err(|_| {
        Error::invalid_argument(
            OP,
            "configuration",
            format!("{context}: dimension {dim} does not fit in isize"),
        )
    })
}

fn add_job_offset(base: isize, offset: usize, role: &'static str) -> Result<isize> {
    let offset = isize::try_from(offset).map_err(|_| {
        Error::invalid_argument(
            "grouped_gemm",
            "configuration",
            format!("{role} offset {offset} does not fit in isize"),
        )
    })?;
    base.checked_add(offset).ok_or_else(|| {
        Error::invalid_argument(
            "grouped_gemm",
            "configuration",
            format!("{role} offset overflows isize: base={base} offset={offset}"),
        )
    })
}

fn scale_grouped_empty_output<T>(
    c_ptr: *mut T,
    rows: usize,
    cols: usize,
    beta: T,
) -> crate::Result<()>
where
    T: Copy + Zero + PartialEq + std::ops::Mul<Output = T>,
{
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    let beta_is_zero = beta == T::zero();
    for col in 0..cols {
        let col_offset = col.checked_mul(rows).ok_or_else(|| {
            Error::invalid_argument(
                "grouped_gemm",
                "configuration",
                format!("output column offset overflows usize: col={col} rows={rows}"),
            )
        })?;
        for row in 0..rows {
            let offset = col_offset + row;
            unsafe {
                let dst = c_ptr.add(offset);
                *dst = if beta_is_zero { T::zero() } else { beta * *dst };
            }
        }
    }
    Ok(())
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
pub(crate) fn canonical_gemm_layout(
    config: &DotGeneralConfig,
    lhs_rank: usize,
    rhs_rank: usize,
) -> (SmallVec<[usize; 8]>, SmallVec<[usize; 8]>, DotGeneralConfig) {
    let lhs_free: SmallVec<[usize; 8]> = (0..lhs_rank)
        .filter(|&d| {
            !config.lhs_contracting_dims.contains(&d) && !config.lhs_batch_dims.contains(&d)
        })
        .collect();
    let rhs_free: SmallVec<[usize; 8]> = (0..rhs_rank)
        .filter(|&d| {
            !config.rhs_contracting_dims.contains(&d) && !config.rhs_batch_dims.contains(&d)
        })
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

fn analyse_gemm<L, R, T>(
    lhs: &L,
    rhs: &R,
    config: &DotGeneralConfig,
) -> crate::Result<Option<GemmDims>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_rank = lhs_shape.len();
    let rhs_rank = rhs_shape.len();

    let lhs_free: SmallVec<[usize; 8]> = (0..lhs_rank)
        .filter(|&d| {
            !config.lhs_contracting_dims.contains(&d) && !config.lhs_batch_dims.contains(&d)
        })
        .collect();
    let rhs_free: SmallVec<[usize; 8]> = (0..rhs_rank)
        .filter(|&d| {
            !config.rhs_contracting_dims.contains(&d) && !config.rhs_batch_dims.contains(&d)
        })
        .collect();

    let lhs_strides = lhs.strides()?;
    let rhs_strides = rhs.strides()?;

    let batch_shapes: SmallVec<[usize; 8]> = config
        .lhs_batch_dims
        .iter()
        .map(|&d| lhs_shape[d])
        .collect();
    let Some(batch_total) = checked_product(&batch_shapes) else {
        return Ok(None);
    };

    let lhs_free_shapes: SmallVec<[usize; 8]> = lhs_free.iter().map(|&d| lhs_shape[d]).collect();
    let rhs_free_shapes: SmallVec<[usize; 8]> = rhs_free.iter().map(|&d| rhs_shape[d]).collect();
    let contract_shapes: SmallVec<[usize; 8]> = config
        .lhs_contracting_dims
        .iter()
        .map(|&d| lhs_shape[d])
        .collect();

    let Some(m) = checked_product(&lhs_free_shapes) else {
        return Ok(None);
    };
    let Some(n) = checked_product(&rhs_free_shapes) else {
        return Ok(None);
    };
    let Some(k) = checked_product(&contract_shapes) else {
        return Ok(None);
    };

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
        return Ok(None);
    }

    let Some((_, a_rs)) = try_fuse_dims(&lhs_free_shapes, &lhs_free_strides)? else {
        return Ok(None);
    };
    let Some((_, a_cs)) = try_fuse_dims(&contract_shapes, &lhs_contract_strides)? else {
        return Ok(None);
    };
    let Some((_, b_rs)) = try_fuse_dims(&contract_shapes, &rhs_contract_strides)? else {
        return Ok(None);
    };
    let Some((_, b_cs)) = try_fuse_dims(&rhs_free_shapes, &rhs_free_strides)? else {
        return Ok(None);
    };
    let Some((_, a_bs)) = try_fuse_dims(&batch_shapes, &lhs_batch_strides)? else {
        return Ok(None);
    };
    let Some((_, b_bs)) = try_fuse_dims(&batch_shapes, &rhs_batch_strides)? else {
        return Ok(None);
    };

    let mut out_shape = SmallVec::<[usize; 8]>::new();
    out_shape.extend_from_slice(&lhs_free_shapes);
    out_shape.extend_from_slice(&rhs_free_shapes);
    out_shape.extend_from_slice(&batch_shapes);

    Ok(Some(GemmDims {
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
        out_shape,
    }))
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
        if let Some(cached) = cache.cached_dims(slot, cache_kind, lhs, rhs, config)? {
            return Ok(cached);
        }
    }

    validate_dot_general(lhs, rhs, config)?;
    let dims = analyse_gemm(lhs, rhs, config)?;
    if let Some(slot) = cache_slot {
        cache.store(
            slot,
            cache_kind,
            lhs.shape().iter().copied().collect(),
            rhs.shape().iter().copied().collect(),
            lhs.strides()?,
            rhs.strides()?,
            GemmConfigKey::from_config(config),
            dims.clone(),
        );
    }
    Ok(dims)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProviderGemmPlan {
    rows: usize,
    columns: usize,
    contracted: usize,
    batch_count: usize,
    lhs_layout: crate::provider::CpuBatchedMatrixLayout,
    rhs_layout: crate::provider::CpuBatchedMatrixLayout,
    output_layout: crate::provider::CpuBatchedMatrixLayout,
}

impl ProviderGemmPlan {
    pub(crate) fn batch_count(self) -> usize {
        self.batch_count
    }

    pub(crate) fn request<'request, 'input, 'output>(
        self,
        lhs: &'request TensorRead<'input>,
        rhs: &'request TensorRead<'input>,
        output: &'request mut TensorWrite<'output>,
        accumulation: DotGeneralAccumulation,
    ) -> CpuGemmRequest<'request, 'input, 'output> {
        CpuGemmRequest::new(
            lhs,
            rhs,
            output,
            self.rows,
            self.columns,
            self.contracted,
            self.batch_count,
            self.lhs_layout,
            self.rhs_layout,
            self.output_layout,
            accumulation,
        )
    }
}

fn provider_output_strides(output: &TensorWrite<'_>) -> Result<SmallVec<[isize; 8]>> {
    Ok(match output {
        TensorWrite::Tensor(output) => col_major_strides(output.shape())?.into_iter().collect(),
        TensorWrite::View(output) => output.strides().iter().copied().collect(),
    })
}

fn prepare_provider_gemm_typed<L, R, T>(
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    cache_kind: GemmAnalysisCacheKind,
    lhs: &L,
    rhs: &R,
    lhs_offset: isize,
    rhs_offset: isize,
    output: &TensorWrite<'_>,
    config: &DotGeneralConfig,
) -> Result<Option<ProviderGemmPlan>>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
{
    let Some(dims) = analyse_gemm_cached(cache, cache_slot, cache_kind, lhs, rhs, config)? else {
        return Ok(None);
    };
    let output_strides = provider_output_strides(output)?;
    let Some((output_row_stride, output_column_stride, output_batch_stride)) = output_gemm_strides(
        output.shape(),
        &output_strides,
        lhs.shape().len(),
        rhs.shape().len(),
        config,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(ProviderGemmPlan {
        rows: dims.m,
        columns: dims.n,
        contracted: dims.k,
        batch_count: dims.batch_total,
        lhs_layout: crate::provider::CpuBatchedMatrixLayout::new(
            lhs_offset, dims.a_rs, dims.a_cs, dims.a_bs,
        ),
        rhs_layout: crate::provider::CpuBatchedMatrixLayout::new(
            rhs_offset, dims.b_rs, dims.b_cs, dims.b_bs,
        ),
        output_layout: crate::provider::CpuBatchedMatrixLayout::new(
            output.offset(),
            output_row_stride,
            output_column_stride,
            output_batch_stride,
        ),
    }))
}

fn prepare_provider_gemm_kind(
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    cache_kind: GemmAnalysisCacheKind,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    output: &TensorWrite<'_>,
    config: &DotGeneralConfig,
) -> Result<Option<ProviderGemmPlan>> {
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            match (lhs, rhs) {
                (
                    TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                    TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                ) => {
                    return prepare_provider_gemm_typed(
                        cache,
                        cache_slot,
                        cache_kind,
                        lhs,
                        rhs,
                        lhs.offset(),
                        rhs.offset(),
                        output,
                        config,
                    );
                }
                (
                    TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                    TensorRead::View(TensorView::$view(rhs)),
                ) => {
                    return prepare_provider_gemm_typed(
                        cache,
                        cache_slot,
                        cache_kind,
                        lhs,
                        rhs,
                        lhs.offset(),
                        rhs.offset(),
                        output,
                        config,
                    );
                }
                (
                    TensorRead::View(TensorView::$view(lhs)),
                    TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                ) => {
                    return prepare_provider_gemm_typed(
                        cache,
                        cache_slot,
                        cache_kind,
                        lhs,
                        rhs,
                        lhs.offset(),
                        rhs.offset(),
                        output,
                        config,
                    );
                }
                (
                    TensorRead::View(TensorView::$view(lhs)),
                    TensorRead::View(TensorView::$view(rhs)),
                ) => {
                    return prepare_provider_gemm_typed(
                        cache,
                        cache_slot,
                        cache_kind,
                        lhs,
                        rhs,
                        lhs.offset(),
                        rhs.offset(),
                        output,
                        config,
                    );
                }
                _ => {}
            }
        };
    }
    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(None)
}

pub(crate) fn prepare_provider_gemm(
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    output: &TensorWrite<'_>,
    config: &DotGeneralConfig,
) -> Result<Option<ProviderGemmPlan>> {
    prepare_provider_gemm_kind(
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Direct,
        lhs,
        rhs,
        output,
        config,
    )
}

pub(crate) fn prepare_provider_gemm_canonical(
    cache: &mut GemmAnalysisCache,
    cache_slot: Option<usize>,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    output: &TensorWrite<'_>,
    config: &DotGeneralConfig,
) -> Result<Option<ProviderGemmPlan>> {
    prepare_provider_gemm_kind(
        cache,
        cache_slot,
        GemmAnalysisCacheKind::Canonical,
        lhs,
        rhs,
        output,
        config,
    )
}

#[cfg(feature = "cpu-faer")]
fn grouped_gemm_faer_with_parallelism(
    ctx: &crate::CpuContext,
    kernel_parallelism: CpuKernelParallelism,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &GroupedGemmConfig<'_>,
    out: &mut TensorWrite<'_>,
) -> crate::Result<bool> {
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            if let (ContractionScalar::$owned(alpha), ContractionScalar::$owned(beta)) =
                (config.accumulation().alpha, config.accumulation().beta)
            {
                match (lhs, rhs, &mut *out) {
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            &mut c,
                        );
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            &mut c,
                        );
                    }
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            &mut c,
                        );
                    }
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            &mut c,
                        );
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => {
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            c,
                        )
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => {
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            c,
                        )
                    }
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => {
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            c,
                        )
                    }
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => {
                        return grouped_gemm_faer_typed(
                            ctx,
                            kernel_parallelism,
                            a,
                            b,
                            config,
                            alpha,
                            beta,
                            c,
                        )
                    }
                    _ => {}
                }
            }
        };
    }

    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(false)
}

#[cfg(feature = "cpu-faer")]
fn grouped_gemm_faer_typed<L, R, T>(
    ctx: &crate::CpuContext,
    kernel_parallelism: CpuKernelParallelism,
    lhs: &L,
    rhs: &R,
    config: &GroupedGemmConfig<'_>,
    alpha: T,
    beta: T,
    out: &mut TypedTensorViewMut<'_, T>,
) -> crate::Result<bool>
where
    L: TypedTensorRead<T> + Sync,
    R: TypedTensorRead<T> + Sync,
    T: FaerGemm
        + Copy
        + Clone
        + Zero
        + PartialEq
        + std::ops::Mul<Output = T>
        + Send
        + Sync
        + 'static,
{
    let Some(a_data) = lhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Ok(false);
    };
    let Some(b_data) = rhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Ok(false);
    };
    let a_base = lhs.offset();
    let b_base = rhs.offset();
    let c_base = out.offset();
    let c_data = out.host_storage_mut()?.as_mut_ptr();
    let a_addr = a_data as usize;
    let b_addr = b_data as usize;
    let c_addr = c_data as usize;
    // Rayon closures cannot capture raw pointers as Sync. Keep only integer
    // base addresses outside the closure and reconstruct execution-local
    // pointers after validate_grouped_gemm has checked each job range.
    let run_job = |job: &tenferro_tensor::backend::GroupedGemmJob| -> crate::Result<()> {
        let a_ptr =
            (a_addr as *const T).wrapping_offset(add_job_offset(a_base, job.lhs_offset(), "lhs")?);
        let b_ptr =
            (b_addr as *const T).wrapping_offset(add_job_offset(b_base, job.rhs_offset(), "rhs")?);
        let c_ptr =
            (c_addr as *mut T).wrapping_offset(add_job_offset(c_base, job.out_offset(), "out")?);
        if job.rows() == 0 || job.cols() == 0 || job.contracted() == 0 {
            return scale_grouped_empty_output(c_ptr, job.rows(), job.cols(), beta);
        }
        let rows = dim_to_isize(job.rows(), "grouped_gemm rows")?;
        let contracted = dim_to_isize(job.contracted(), "grouped_gemm contracted")?;
        // SAFETY: validate_grouped_gemm checked input/output ranges and output
        // disjointness before this provider path. The engine selects whether
        // one job may use inner kernel parallelism.
        unsafe {
            T::strided_gemm_with_conj_par(
                ctx,
                match kernel_parallelism {
                    CpuKernelParallelism::Sequential => ctx.faer_seq(),
                    CpuKernelParallelism::Inner => ctx.faer_par(),
                },
                alpha,
                a_ptr,
                job.rows(),
                job.contracted(),
                1,
                rows,
                config.accumulation().lhs_conj,
                b_ptr,
                job.cols(),
                1,
                contracted,
                config.accumulation().rhs_conj,
                beta,
                c_ptr,
                1,
                rows,
            );
        }
        Ok(())
    };

    for job in config.jobs() {
        run_job(job)?;
    }
    Ok(true)
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn grouped_gemm_blas_cached(
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &GroupedGemmConfig<'_>,
    out: &mut TensorWrite<'_>,
) -> crate::Result<bool> {
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            if let (ContractionScalar::$owned(alpha), ContractionScalar::$owned(beta)) =
                (config.accumulation().alpha, config.accumulation().beta)
            {
                match (lhs, rhs, &mut *out) {
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_blas_typed(a, b, config, alpha, beta, &mut c);
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_blas_typed(a, b, config, alpha, beta, &mut c);
                    }
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_blas_typed(a, b, config, alpha, beta, &mut c);
                    }
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::Tensor(crate::Tensor::$owned(c)),
                    ) => {
                        let mut c = c.as_view_mut();
                        return grouped_gemm_blas_typed(a, b, config, alpha, beta, &mut c);
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => return grouped_gemm_blas_typed(a, b, config, alpha, beta, c),
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => return grouped_gemm_blas_typed(a, b, config, alpha, beta, c),
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::Tensor(crate::Tensor::$owned(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => return grouped_gemm_blas_typed(a, b, config, alpha, beta, c),
                    (
                        TensorRead::View(TensorView::$view(a)),
                        TensorRead::View(TensorView::$view(b)),
                        TensorWrite::View(TensorViewMut::$view(c)),
                    ) => return grouped_gemm_blas_typed(a, b, config, alpha, beta, c),
                    _ => {}
                }
            }
        };
    }

    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(false)
}

#[cfg(feature = "cpu-blas")]
fn grouped_gemm_blas_typed<L, R, T>(
    lhs: &L,
    rhs: &R,
    config: &GroupedGemmConfig<'_>,
    alpha: T,
    beta: T,
    out: &mut TypedTensorViewMut<'_, T>,
) -> crate::Result<bool>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: BlasGemm + Copy + Clone + Zero + PartialEq + std::ops::Mul<Output = T> + 'static,
{
    if config.accumulation().lhs_conj || config.accumulation().rhs_conj {
        return Ok(false);
    }
    let Some(a_data) = lhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Ok(false);
    };
    let Some(b_data) = rhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Ok(false);
    };
    let a_base = lhs.offset();
    let b_base = rhs.offset();
    let c_base = out.offset();
    let c_data = out.host_storage_mut()?.as_mut_ptr();

    // Raw pointers are execution-local and must not be cached. The job count is
    // runtime-dependent, so use a reserved Vec rather than a SmallVec threshold
    // for the contiguous provider descriptor slice.
    let mut batches = Vec::with_capacity(config.jobs().len());
    for job in config.jobs() {
        // SAFETY: validate_grouped_gemm checked each job's range before this
        // provider path; add_job_offset only combines the checked view base
        // with the checked element offset.
        let (a_ptr, b_ptr, c_ptr) = unsafe {
            (
                a_data.offset(add_job_offset(a_base, job.lhs_offset(), "lhs")?),
                b_data.offset(add_job_offset(b_base, job.rhs_offset(), "rhs")?),
                c_data.offset(add_job_offset(c_base, job.out_offset(), "out")?),
            )
        };
        if job.rows() == 0 || job.cols() == 0 || job.contracted() == 0 {
            scale_grouped_empty_output(c_ptr, job.rows(), job.cols(), beta)?;
            continue;
        }
        let rows = dim_to_isize(job.rows(), "grouped_gemm rows")?;
        let contracted = dim_to_isize(job.contracted(), "grouped_gemm contracted")?;
        batches.push(BlasGemmBatch {
            a_ptr,
            b_ptr,
            c_ptr,
            m: job.rows(),
            n: job.cols(),
            k: job.contracted(),
            a_rs: 1,
            a_cs: rows,
            b_rs: 1,
            b_cs: contracted,
            c_rs: 1,
            c_cs: rows,
        });
    }
    // SAFETY: every descriptor uses validated dimensions and output regions are
    // pairwise-disjoint, so the BLAS provider may run jobs in batch order or as
    // a native grouped call.
    unsafe {
        T::grouped_gemm(alpha, beta, &batches)?;
    }
    Ok(true)
}

#[cfg(feature = "cpu-tblis-provider")]
fn execute_tblis_request_typed<L, R, T>(
    lhs: &L,
    rhs: &R,
    axes: CpuContractionAxes<'_>,
    output: &mut TypedTensorViewMut<'_, T>,
    execution: tblis_gemm::TblisExecution<T>,
) -> Result<CpuProviderOutcome>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: TblisGemm + Copy + Clone + Zero + One + PartialEq + std::ops::Mul<Output = T> + 'static,
{
    if !tblis_gemm::runtime_available()? {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ));
    }
    let Some(plan) = tblis_gemm::plan_from_axes(lhs, rhs, &axes, output.shape(), output.strides())?
    else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Layout(CpuOperand::Output),
        ));
    };
    let Some(lhs_data) = lhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Err(crate::cpu_backend_buffer_error(OP));
    };
    let Some(rhs_data) = rhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Err(crate::cpu_backend_buffer_error(OP));
    };
    // SAFETY: the common validator proved non-negative, reachable offsets.
    let lhs_ptr = unsafe { lhs_data.offset(lhs.offset()) };
    // SAFETY: the common validator proved non-negative, reachable offsets.
    let rhs_ptr = unsafe { rhs_data.offset(rhs.offset()) };
    tblis_gemm::execute(plan, lhs_ptr, rhs_ptr, output, execution)?;
    Ok(CpuProviderOutcome::Executed)
}

#[cfg(feature = "cpu-tblis-provider")]
pub(crate) fn execute_tblis_general_request(
    _context: &CpuProviderContext<'_>,
    request: CpuDotGeneralRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    let (lhs, rhs, output, axes, accumulation) = request.into_parts();
    let dtype = lhs.dtype();
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            if let (ContractionScalar::$owned(alpha), ContractionScalar::$owned(beta)) =
                (accumulation.alpha, accumulation.beta)
            {
                let execution = tblis_gemm::TblisExecution::new(
                    alpha,
                    beta,
                    accumulation.lhs_conj,
                    accumulation.rhs_conj,
                );
                match (lhs, rhs, &mut *output) {
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_tblis_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_tblis_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_tblis_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_tblis_request_typed(lhs, rhs, axes, &mut output, execution);
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_tblis_request_typed(lhs, rhs, axes, output, execution),
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_tblis_request_typed(lhs, rhs, axes, output, execution),
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_tblis_request_typed(lhs, rhs, axes, output, execution),
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => return execute_tblis_request_typed(lhs, rhs, axes, output, execution),
                    _ => {}
                }
            }
        };
    }
    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::DType(dtype),
    ))
}

#[cfg(any(feature = "cpu-blas", feature = "cpu-faer"))]
fn normalize_singleton_stride(stride: isize, extent: usize, fallback: usize) -> isize {
    if extent == 1 {
        let fallback = fallback.max(1) as isize;
        stride.max(fallback)
    } else {
        stride
    }
}

#[cfg(feature = "cpu-blas")]
fn blas_lhs_layout_supported(m: usize, k: usize, row_stride: isize, col_stride: isize) -> bool {
    if row_stride == 1 {
        blas_leading_stride_supported(col_stride, m)
    } else if col_stride == 1 {
        blas_leading_stride_supported(row_stride, k)
    } else {
        false
    }
}

#[cfg(feature = "cpu-blas")]
fn blas_rhs_layout_supported(k: usize, n: usize, row_stride: isize, col_stride: isize) -> bool {
    if row_stride == 1 {
        blas_leading_stride_supported(col_stride, k)
    } else if col_stride == 1 {
        blas_leading_stride_supported(row_stride, n)
    } else {
        false
    }
}

#[cfg(feature = "cpu-blas")]
fn blas_output_layout_supported(m: usize, row_stride: isize, col_stride: isize) -> bool {
    row_stride == 1 && blas_leading_stride_supported(col_stride, m)
}

#[cfg(feature = "cpu-blas")]
fn blas_leading_stride_supported(stride: isize, minimum: usize) -> bool {
    if stride <= 0 {
        return false;
    }
    let Ok(stride_usize) = usize::try_from(stride) else {
        return false;
    };
    stride_usize >= minimum && i32::try_from(stride).is_ok()
}

#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
#[derive(Clone, Copy)]
struct ProviderGemmDescriptor {
    rows: usize,
    columns: usize,
    contracted: usize,
    batch_count: usize,
    lhs_layout: crate::provider::CpuBatchedMatrixLayout,
    rhs_layout: crate::provider::CpuBatchedMatrixLayout,
    output_layout: crate::provider::CpuBatchedMatrixLayout,
    accumulation: DotGeneralAccumulation,
}

#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
impl ProviderGemmDescriptor {
    fn from_parts(parts: &crate::provider::CpuGemmRequestParts<'_, '_, '_>) -> Self {
        Self {
            rows: parts.rows,
            columns: parts.columns,
            contracted: parts.contracted,
            batch_count: parts.batch_count,
            lhs_layout: parts.lhs_layout,
            rhs_layout: parts.rhs_layout,
            output_layout: parts.output_layout,
            accumulation: parts.accumulation,
        }
    }
}

#[cfg(feature = "cpu-faer")]
fn execute_faer_request_typed<L, R, T>(
    context: &CpuProviderContext<'_>,
    descriptor: ProviderGemmDescriptor,
    lhs: &L,
    rhs: &R,
    output: &mut TypedTensorViewMut<'_, T>,
    alpha: T,
    beta: T,
) -> Result<()>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: FaerGemm + Copy + Zero + PartialEq + std::ops::Mul<Output = T> + 'static,
{
    if descriptor.rows == 0
        || descriptor.columns == 0
        || descriptor.contracted == 0
        || descriptor.batch_count == 0
    {
        return scale_empty_contract_output(output, beta);
    }
    let Some(lhs_data) = lhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Err(crate::cpu_backend_buffer_error(OP));
    };
    let Some(rhs_data) = rhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Err(crate::cpu_backend_buffer_error(OP));
    };
    let output_data = output.host_storage_mut()?.as_mut_ptr();
    let par = match context.kernel_parallelism() {
        CpuKernelParallelism::Sequential => context.cpu_context().faer_seq(),
        CpuKernelParallelism::Inner => context.cpu_context().faer_par(),
    };
    for batch in 0..descriptor.batch_count {
        checked_view_batch_offset(
            descriptor.lhs_layout.offset(),
            batch,
            descriptor.lhs_layout.batch_stride(),
        )?;
        checked_view_batch_offset(
            descriptor.rhs_layout.offset(),
            batch,
            descriptor.rhs_layout.batch_stride(),
        )?;
        checked_view_batch_offset(
            descriptor.output_layout.offset(),
            batch,
            descriptor.output_layout.batch_stride(),
        )?;
    }
    for batch in 0..descriptor.batch_count {
        let lhs_offset = checked_view_batch_offset(
            descriptor.lhs_layout.offset(),
            batch,
            descriptor.lhs_layout.batch_stride(),
        )?;
        let rhs_offset = checked_view_batch_offset(
            descriptor.rhs_layout.offset(),
            batch,
            descriptor.rhs_layout.batch_stride(),
        )?;
        let output_offset = checked_view_batch_offset(
            descriptor.output_layout.offset(),
            batch,
            descriptor.output_layout.batch_stride(),
        )?;
        // SAFETY: the engine validates every request layout and reachable range
        // before provider entry. Batch offsets are rechecked above and output
        // batches are uniquely writable.
        unsafe {
            T::strided_gemm_with_conj_par(
                context.cpu_context(),
                par,
                alpha,
                lhs_data.offset(lhs_offset),
                descriptor.rows,
                descriptor.contracted,
                descriptor.lhs_layout.row_stride(),
                descriptor.lhs_layout.column_stride(),
                descriptor.accumulation.lhs_conj,
                rhs_data.offset(rhs_offset),
                descriptor.columns,
                descriptor.rhs_layout.row_stride(),
                descriptor.rhs_layout.column_stride(),
                descriptor.accumulation.rhs_conj,
                beta,
                output_data.offset(output_offset),
                descriptor.output_layout.row_stride(),
                descriptor.output_layout.column_stride(),
            );
        }
    }
    Ok(())
}

#[cfg(feature = "cpu-faer")]
pub(crate) fn execute_faer_gemm_request(
    context: &CpuProviderContext<'_>,
    request: CpuGemmRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    let parts = request.into_parts();
    let descriptor = ProviderGemmDescriptor::from_parts(&parts);
    let lhs = parts.lhs;
    let rhs = parts.rhs;
    let dtype = lhs.dtype();
    let output = parts.output;
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            if let (ContractionScalar::$owned(alpha), ContractionScalar::$owned(beta)) =
                (descriptor.accumulation.alpha, descriptor.accumulation.beta)
            {
                match (lhs, rhs, &mut *output) {
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        execute_faer_request_typed(
                            context,
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        execute_faer_request_typed(
                            context,
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        execute_faer_request_typed(
                            context,
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        execute_faer_request_typed(
                            context,
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        execute_faer_request_typed(
                            context, descriptor, lhs, rhs, output, alpha, beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        execute_faer_request_typed(
                            context, descriptor, lhs, rhs, output, alpha, beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        execute_faer_request_typed(
                            context, descriptor, lhs, rhs, output, alpha, beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        execute_faer_request_typed(
                            context, descriptor, lhs, rhs, output, alpha, beta,
                        )?;
                        return Ok(CpuProviderOutcome::Executed);
                    }
                    _ => {}
                }
            }
        };
    }
    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::DType(dtype),
    ))
}

#[cfg(feature = "cpu-blas")]
fn blas_descriptor_unsupported(
    descriptor: ProviderGemmDescriptor,
) -> Option<CpuProviderUnsupported> {
    if i32::try_from(descriptor.rows).is_err()
        || i32::try_from(descriptor.columns).is_err()
        || i32::try_from(descriptor.contracted).is_err()
    {
        return Some(CpuProviderUnsupported::Rank { lhs: 2, rhs: 2 });
    }
    if !blas_lhs_layout_supported(
        descriptor.rows,
        descriptor.contracted,
        descriptor.lhs_layout.row_stride(),
        descriptor.lhs_layout.column_stride(),
    ) {
        return Some(CpuProviderUnsupported::Layout(CpuOperand::Lhs));
    }
    if !blas_rhs_layout_supported(
        descriptor.contracted,
        descriptor.columns,
        descriptor.rhs_layout.row_stride(),
        descriptor.rhs_layout.column_stride(),
    ) {
        return Some(CpuProviderUnsupported::Layout(CpuOperand::Rhs));
    }
    if !blas_output_layout_supported(
        descriptor.rows,
        descriptor.output_layout.row_stride(),
        descriptor.output_layout.column_stride(),
    ) {
        return Some(CpuProviderUnsupported::Layout(CpuOperand::Output));
    }
    None
}

#[cfg(feature = "cpu-blas")]
fn execute_blas_request_typed<L, R, T>(
    descriptor: ProviderGemmDescriptor,
    lhs: &L,
    rhs: &R,
    output: &mut TypedTensorViewMut<'_, T>,
    alpha: T,
    beta: T,
) -> Result<CpuProviderOutcome>
where
    L: TypedTensorRead<T>,
    R: TypedTensorRead<T>,
    T: BlasGemm + Copy + Zero + PartialEq + std::ops::Mul<Output = T> + 'static,
{
    if let Some(reason) = blas_descriptor_unsupported(descriptor) {
        return Ok(CpuProviderOutcome::Unsupported(reason));
    }
    if (descriptor.accumulation.lhs_conj && descriptor.lhs_layout.row_stride() == 1)
        || (descriptor.accumulation.rhs_conj && descriptor.rhs_layout.row_stride() == 1)
    {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Conjugation,
        ));
    }
    if descriptor.rows == 0
        || descriptor.columns == 0
        || descriptor.contracted == 0
        || descriptor.batch_count == 0
    {
        scale_empty_contract_output(output, beta)?;
        return Ok(CpuProviderOutcome::Executed);
    }

    let Some(lhs_data) = lhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Err(crate::cpu_backend_buffer_error(OP));
    };
    let Some(rhs_data) = rhs.host_data_opt()?.map(<[T]>::as_ptr) else {
        return Err(crate::cpu_backend_buffer_error(OP));
    };
    let output_data = output.host_storage_mut()?.as_mut_ptr();
    for batch in 0..descriptor.batch_count {
        checked_view_batch_offset(
            descriptor.lhs_layout.offset(),
            batch,
            descriptor.lhs_layout.batch_stride(),
        )?;
        checked_view_batch_offset(
            descriptor.rhs_layout.offset(),
            batch,
            descriptor.rhs_layout.batch_stride(),
        )?;
        checked_view_batch_offset(
            descriptor.output_layout.offset(),
            batch,
            descriptor.output_layout.batch_stride(),
        )?;
    }
    for batch in 0..descriptor.batch_count {
        let lhs_offset = checked_view_batch_offset(
            descriptor.lhs_layout.offset(),
            batch,
            descriptor.lhs_layout.batch_stride(),
        )?;
        let rhs_offset = checked_view_batch_offset(
            descriptor.rhs_layout.offset(),
            batch,
            descriptor.rhs_layout.batch_stride(),
        )?;
        let output_offset = checked_view_batch_offset(
            descriptor.output_layout.offset(),
            batch,
            descriptor.output_layout.batch_stride(),
        )?;
        // SAFETY: the engine validates every reachable range and uniquely
        // writable output batch before constructing the provider request.
        let executed = unsafe {
            T::strided_gemm_with_conj(
                alpha,
                lhs_data.offset(lhs_offset),
                descriptor.rows,
                descriptor.contracted,
                descriptor.lhs_layout.row_stride(),
                descriptor.lhs_layout.column_stride(),
                descriptor.accumulation.lhs_conj,
                rhs_data.offset(rhs_offset),
                descriptor.columns,
                descriptor.rhs_layout.row_stride(),
                descriptor.rhs_layout.column_stride(),
                descriptor.accumulation.rhs_conj,
                beta,
                output_data.offset(output_offset),
                descriptor.output_layout.row_stride(),
                descriptor.output_layout.column_stride(),
            )?
        };
        debug_assert!(
            executed,
            "BLAS capability was checked before output mutation"
        );
    }
    Ok(CpuProviderOutcome::Executed)
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn execute_blas_gemm_request(
    _context: &CpuProviderContext<'_>,
    request: CpuGemmRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    let parts = request.into_parts();
    let descriptor = ProviderGemmDescriptor::from_parts(&parts);
    let lhs = parts.lhs;
    let rhs = parts.rhs;
    let dtype = lhs.dtype();
    let output = parts.output;
    macro_rules! dispatch {
        ($owned:ident, $view:ident) => {
            if let (ContractionScalar::$owned(alpha), ContractionScalar::$owned(beta)) =
                (descriptor.accumulation.alpha, descriptor.accumulation.beta)
            {
                match (lhs, rhs, &mut *output) {
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_blas_request_typed(
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        );
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_blas_request_typed(
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        );
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_blas_request_typed(
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        );
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::Tensor(crate::Tensor::$owned(output)),
                    ) => {
                        let mut output = output.as_view_mut();
                        return execute_blas_request_typed(
                            descriptor,
                            lhs,
                            rhs,
                            &mut output,
                            alpha,
                            beta,
                        );
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        return execute_blas_request_typed(
                            descriptor, lhs, rhs, output, alpha, beta,
                        )
                    }
                    (
                        TensorRead::Tensor(crate::Tensor::$owned(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        return execute_blas_request_typed(
                            descriptor, lhs, rhs, output, alpha, beta,
                        )
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::Tensor(crate::Tensor::$owned(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        return execute_blas_request_typed(
                            descriptor, lhs, rhs, output, alpha, beta,
                        )
                    }
                    (
                        TensorRead::View(TensorView::$view(lhs)),
                        TensorRead::View(TensorView::$view(rhs)),
                        TensorWrite::View(TensorViewMut::$view(output)),
                    ) => {
                        return execute_blas_request_typed(
                            descriptor, lhs, rhs, output, alpha, beta,
                        )
                    }
                    _ => {}
                }
            }
        };
    }
    dispatch!(F32, F32);
    dispatch!(F64, F64);
    dispatch!(C32, C32);
    dispatch!(C64, C64);
    Ok(CpuProviderOutcome::Unsupported(
        CpuProviderUnsupported::DType(dtype),
    ))
}

#[cfg(feature = "cpu-blas")]
pub(crate) fn execute_blas_grouped_request(
    _context: &CpuProviderContext<'_>,
    request: CpuGroupedGemmRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    let (lhs, rhs, output, jobs, accumulation) = request.into_parts();
    if accumulation.lhs_conj || accumulation.rhs_conj {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Conjugation,
        ));
    }
    let config = GroupedGemmConfig::new(jobs, accumulation);
    if grouped_gemm_blas_cached(lhs, rhs, &config, output)? {
        Ok(CpuProviderOutcome::Executed)
    } else {
        Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ))
    }
}

#[cfg(feature = "cpu-faer")]
pub(crate) fn execute_faer_grouped_request(
    context: &CpuProviderContext<'_>,
    request: CpuGroupedGemmRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    let (lhs, rhs, output, jobs, accumulation) = request.into_parts();
    let config = GroupedGemmConfig::new(jobs, accumulation);
    if grouped_gemm_faer_with_parallelism(
        context.cpu_context(),
        context.kernel_parallelism(),
        lhs,
        rhs,
        &config,
        output,
    )? {
        Ok(CpuProviderOutcome::Executed)
    } else {
        Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ))
    }
}

#[cfg(test)]
mod tests;
