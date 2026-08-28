use std::ffi::c_void;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use cubecl::prelude::{CubeElement, CubePrimitive};
use cubecl::stream_id::StreamId;
use cubecl_cuda::CudaRuntime as CubeclCudaRuntime;
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};

use super::dispatch::{
    alloc_output, cube_count_for_len, cube_dim_1d, cubecl_buffer, dtype_mismatch,
    ensure_resident_on_runtime, ensure_view_mut_resident_on_runtime,
    ensure_view_resident_on_runtime, launch_nullary_into, prepared_tensor_access,
    prepared_view_access, prepared_view_mut_access, CubeclPreparedAccess,
};
use super::error::{unsupported_dtype, unsupported_operation, workspace_size_overflow};
use super::ffi::cutensor::{
    CudaDataType, CutensorComputeDescriptor, CutensorCudaStream, CutensorHandle, CutensorOperator,
    CutensorWorksizePreference, OperationDescriptor, Plan, PlanPreference, TensorDescriptor,
};
use super::interop::cuda_device_ptr_from_addr;
use super::plan_cache::LruPlanCache;
use super::{CudaBackend, CudaRuntime};
use crate::config::DotGeneralConfig;
use crate::kernels::structural;
use crate::{col_major_strides, Error, Tensor, TypedTensor};
use tenferro_tensor::{
    CacheStats, ContractionScalar, DType, DotGeneralAccumulation, TensorRead, TensorScalar,
    TensorView, TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};

const OP: &str = "dot_general";
const CUDA_ALLOCATION_ALIGNMENT: u32 = 256;
const DEFAULT_CUTENSOR_PLAN_CACHE_MAX_ENTRIES: usize = 64;
type CutensorContractionPlanCache = LruPlanCache<CutensorContractionKey, CachedCutensorContraction>;
type CutensorPlanCacheState = Arc<Mutex<CutensorContractionPlanCache>>;

trait CutensorScalar: CubeElement + TensorScalar + CubePrimitive + Clone + One + Zero {
    const DATA_TYPE: CudaDataType;
    const DTYPE: DType;
    const IS_COMPLEX: bool;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor;

    /// Unwrap the matching dtype variant from dtype-erased tensors.
    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>>;

    /// Unwrap the matching dtype variant from dtype-erased borrowed views.
    fn unwrap_view<'a, 'b>(view: &'a TensorView<'b>) -> Option<&'a TypedTensorView<'b, Self>>;
    fn unwrap_view_mut<'a, 'b>(
        view: &'a mut TensorViewMut<'b>,
    ) -> Option<&'a mut TypedTensorViewMut<'b, Self>>;
    fn unwrap_tensor_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>>;

    /// Launch the in-place scale kernel for this scalar type.
    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    );
}

/// Implement the dtype-erased variant accessors for one scalar type.
macro_rules! cutensor_variant_accessors {
    ($variant:ident) => {
        fn unwrap_view<'a, 'b>(view: &'a TensorView<'b>) -> Option<&'a TypedTensorView<'b, Self>> {
            match view {
                TensorView::$variant(view) => Some(view),
                _ => None,
            }
        }

        fn unwrap_view_mut<'a, 'b>(
            view: &'a mut TensorViewMut<'b>,
        ) -> Option<&'a mut TypedTensorViewMut<'b, Self>> {
            match view {
                TensorViewMut::$variant(view) => Some(view),
                _ => None,
            }
        }

        fn unwrap_tensor_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>> {
            match tensor {
                Tensor::$variant(tensor) => Some(tensor),
                _ => None,
            }
        }
    };
}

impl CutensorScalar for f32 {
    cutensor_variant_accessors!(F32);

    const DATA_TYPE: CudaDataType = CudaDataType::R32F;
    const DTYPE: DType = DType::F32;
    const IS_COMPLEX: bool = false;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_32f()
    }
    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::F32(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_float_kernel::launch_unchecked::<f32, CubeclCudaRuntime>(
                client, count, dim, out, factor,
            );
        }
    }
}

impl CutensorScalar for f64 {
    cutensor_variant_accessors!(F64);

    const DATA_TYPE: CudaDataType = CudaDataType::R64F;
    const DTYPE: DType = DType::F64;
    const IS_COMPLEX: bool = false;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_64f()
    }
    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::F64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_float_kernel::launch_unchecked::<f64, CubeclCudaRuntime>(
                client, count, dim, out, factor,
            );
        }
    }
}

impl CutensorScalar for Complex32 {
    cutensor_variant_accessors!(C32);

    const DATA_TYPE: CudaDataType = CudaDataType::C32F;
    const DTYPE: DType = DType::C32;
    const IS_COMPLEX: bool = true;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_32f()
    }
    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::C32(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_complex_kernel::launch_unchecked::<
                Complex32,
                CubeclCudaRuntime,
            >(client, count, dim, out, factor);
        }
    }
}

impl CutensorScalar for Complex64 {
    cutensor_variant_accessors!(C64);

    const DATA_TYPE: CudaDataType = CudaDataType::C64F;
    const DTYPE: DType = DType::C64;
    const IS_COMPLEX: bool = true;

    fn compute_descriptor(handle: &CutensorHandle) -> CutensorComputeDescriptor {
        handle.compute_desc_64f()
    }
    fn unwrap_tensor(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        match tensor {
            Tensor::C64(tensor) => Some(tensor),
            _ => None,
        }
    }

    fn launch_scale_in_place(
        client: &cubecl::prelude::ComputeClient<CubeclCudaRuntime>,
        count: cubecl::prelude::CubeCount,
        dim: cubecl::prelude::CubeDim,
        out: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
        factor: cubecl::prelude::ArrayArg<CubeclCudaRuntime>,
    ) {
        // SAFETY: caller validated residency, lengths, and launch domain.
        unsafe {
            structural::scale_in_place_complex_kernel::launch_unchecked::<
                Complex64,
                CubeclCudaRuntime,
            >(client, count, dim, out, factor);
        }
    }
}

struct DotGeneralLayout {
    lhs_modes: Vec<i32>,
    rhs_modes: Vec<i32>,
    output_modes: Vec<i32>,
    output_shape: Vec<usize>,
    lhs_extents: Vec<i64>,
    rhs_extents: Vec<i64>,
    output_extents: Vec<i64>,
    lhs_strides: Vec<i64>,
    rhs_strides: Vec<i64>,
    output_strides: Vec<i64>,
    contracting_elements: usize,
}

struct Workspace {
    // INVARIANT: in pinned CubeCL rev 1c88bb6, CUDA storage records handle
    // deallocation and matches async allocations with `cuMemFreeAsync` on the
    // allocation stream (or sync allocations with `cuMemFree`). Dropping an
    // evicted handle therefore follows CubeCL's stream-owned release contract;
    // this cache does not free the device pointer directly.
    _handle: Option<cubecl_runtime::server::Handle>,
    ptr: *mut c_void,
    size: u64,
}

impl Workspace {
    fn none() -> Self {
        Self {
            _handle: None,
            ptr: std::ptr::null_mut(),
            size: 0,
        }
    }
}

// SAFETY: `Workspace` owns a CubeCL server handle that keeps the device
// allocation alive. The raw pointer is only submitted back to cuTENSOR while
// the cached contraction mutex is held.
unsafe impl Send for Workspace {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CutensorOperandLayoutKey {
    extents: Vec<i64>,
    strides: Vec<i64>,
    modes: Vec<i32>,
}

impl CutensorOperandLayoutKey {
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
struct CutensorContractionKey {
    dtype: DType,
    lhs: CutensorOperandLayoutKey,
    rhs: CutensorOperandLayoutKey,
    output: CutensorOperandLayoutKey,
    lhs_alignment_requirement: u32,
    rhs_alignment_requirement: u32,
    output_alignment_requirement: u32,
    lhs_op: CutensorOperator,
    rhs_op: CutensorOperator,
    workspace_preference: CutensorWorksizePreference,
}

impl CutensorContractionKey {
    fn from_spec<T: CutensorScalar>(spec: &CutensorContractionSpec<'_>) -> Self {
        Self {
            dtype: T::DTYPE,
            lhs: CutensorOperandLayoutKey::new(
                &spec.layout.lhs_extents,
                spec.lhs_strides,
                &spec.layout.lhs_modes,
            ),
            rhs: CutensorOperandLayoutKey::new(
                &spec.layout.rhs_extents,
                spec.rhs_strides,
                &spec.layout.rhs_modes,
            ),
            output: CutensorOperandLayoutKey::new(
                &spec.layout.output_extents,
                spec.output_strides,
                &spec.layout.output_modes,
            ),
            lhs_alignment_requirement: spec.lhs_alignment_requirement,
            rhs_alignment_requirement: spec.rhs_alignment_requirement,
            output_alignment_requirement: spec.output_alignment_requirement,
            lhs_op: cutensor_conj_op::<T>(spec.lhs_conj),
            rhs_op: cutensor_conj_op::<T>(spec.rhs_conj),
            workspace_preference: spec.workspace_preference,
        }
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.lhs.retained_bytes())
            .saturating_add(self.rhs.retained_bytes())
            .saturating_add(self.output.retained_bytes())
    }
}

struct CutensorContractionSpec<'a> {
    layout: &'a DotGeneralLayout,
    lhs_strides: &'a [i64],
    rhs_strides: &'a [i64],
    output_strides: &'a [i64],
    lhs_alignment_requirement: u32,
    rhs_alignment_requirement: u32,
    output_alignment_requirement: u32,
    lhs_conj: bool,
    rhs_conj: bool,
    workspace_preference: CutensorWorksizePreference,
}

struct CachedCutensorContraction {
    // Drop the cuTENSOR plan before the descriptor objects it was built from.
    plan: Plan,
    _plan_preference: PlanPreference,
    _operation_descriptor: OperationDescriptor,
    _output_descriptor: TensorDescriptor,
    _rhs_descriptor: TensorDescriptor,
    _lhs_descriptor: TensorDescriptor,
    workspace: Workspace,
}

// SAFETY: cached cuTENSOR state is tied to one `CudaBackend` and is only used
// while holding the enclosing plan-cache mutex. The opaque cuTENSOR handles are
// created and destroyed through the same loaded cuTENSOR library.
unsafe impl Send for CachedCutensorContraction {}

impl CachedCutensorContraction {
    fn new<T>(
        rt: &CudaRuntime,
        cutensor: &CutensorHandle,
        spec: &CutensorContractionSpec<'_>,
    ) -> crate::Result<Self>
    where
        T: CutensorScalar,
    {
        let desc_a = TensorDescriptor::new(
            cutensor,
            &spec.layout.lhs_extents,
            spec.lhs_strides,
            T::DATA_TYPE,
            spec.lhs_alignment_requirement,
            OP,
        )?;
        let desc_b = TensorDescriptor::new(
            cutensor,
            &spec.layout.rhs_extents,
            spec.rhs_strides,
            T::DATA_TYPE,
            spec.rhs_alignment_requirement,
            OP,
        )?;
        let desc_out = TensorDescriptor::new(
            cutensor,
            &spec.layout.output_extents,
            spec.output_strides,
            T::DATA_TYPE,
            spec.output_alignment_requirement,
            OP,
        )?;
        let op_desc = OperationDescriptor::new_contraction_with_ops(
            cutensor,
            &desc_a,
            &spec.layout.lhs_modes,
            cutensor_conj_op::<T>(spec.lhs_conj),
            &desc_b,
            &spec.layout.rhs_modes,
            cutensor_conj_op::<T>(spec.rhs_conj),
            &desc_out,
            &spec.layout.output_modes,
            &desc_out,
            &spec.layout.output_modes,
            T::compute_descriptor(cutensor),
            OP,
        )?;
        let pref = PlanPreference::new_default(cutensor, OP)?;
        let workspace_size =
            cutensor.estimate_workspace_size(&op_desc, &pref, spec.workspace_preference, OP)?;
        let plan = Plan::new(cutensor, &op_desc, &pref, workspace_size, OP)?;
        let workspace = alloc_workspace(rt, workspace_size)?;
        Ok(Self {
            plan,
            _plan_preference: pref,
            _operation_descriptor: op_desc,
            _output_descriptor: desc_out,
            _rhs_descriptor: desc_b,
            _lhs_descriptor: desc_a,
            workspace,
        })
    }

    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(usize::try_from(self.workspace.size).unwrap_or(usize::MAX))
    }
}

/// Hash `spec` into the plan-cache key hash without materializing an owned
/// key. Field order mirrors [`CutensorContractionKey::from_spec`]; the stored
/// key is compared with [`key_matches_spec`] on lookup, so a 64-bit collision
/// degrades to a plan rebuild instead of a wrong plan.
fn spec_hash<T: CutensorScalar>(spec: &CutensorContractionSpec<'_>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    T::DTYPE.hash(&mut hasher);
    spec.layout.lhs_extents.hash(&mut hasher);
    spec.lhs_strides.hash(&mut hasher);
    spec.layout.lhs_modes.hash(&mut hasher);
    spec.layout.rhs_extents.hash(&mut hasher);
    spec.rhs_strides.hash(&mut hasher);
    spec.layout.rhs_modes.hash(&mut hasher);
    spec.layout.output_extents.hash(&mut hasher);
    spec.output_strides.hash(&mut hasher);
    spec.layout.output_modes.hash(&mut hasher);
    spec.lhs_alignment_requirement.hash(&mut hasher);
    spec.rhs_alignment_requirement.hash(&mut hasher);
    spec.output_alignment_requirement.hash(&mut hasher);
    cutensor_conj_op::<T>(spec.lhs_conj).hash(&mut hasher);
    cutensor_conj_op::<T>(spec.rhs_conj).hash(&mut hasher);
    spec.workspace_preference.hash(&mut hasher);
    hasher.finish()
}

/// Verify a stored materialized key against a borrowed spec.
fn key_matches_spec<T: CutensorScalar>(
    key: &CutensorContractionKey,
    spec: &CutensorContractionSpec<'_>,
) -> bool {
    key.dtype == T::DTYPE
        && key.lhs.extents == spec.layout.lhs_extents
        && key.lhs.strides == spec.lhs_strides
        && key.lhs.modes == spec.layout.lhs_modes
        && key.rhs.extents == spec.layout.rhs_extents
        && key.rhs.strides == spec.rhs_strides
        && key.rhs.modes == spec.layout.rhs_modes
        && key.output.extents == spec.layout.output_extents
        && key.output.strides == spec.output_strides
        && key.output.modes == spec.layout.output_modes
        && key.lhs_alignment_requirement == spec.lhs_alignment_requirement
        && key.rhs_alignment_requirement == spec.rhs_alignment_requirement
        && key.output_alignment_requirement == spec.output_alignment_requirement
        && key.lhs_op == cutensor_conj_op::<T>(spec.lhs_conj)
        && key.rhs_op == cutensor_conj_op::<T>(spec.rhs_conj)
        && key.workspace_preference == spec.workspace_preference
}

pub(super) fn dot_general(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(lhs), Tensor::F32(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::F32)
        }
        (Tensor::F64(lhs), Tensor::F64(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::F64)
        }
        (Tensor::C32(lhs), Tensor::C32(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::C32)
        }
        (Tensor::C64(lhs), Tensor::C64(rhs)) => {
            dot_general_typed(backend, lhs, rhs, config).map(Tensor::C64)
        }
        _ => Err(dtype_mismatch(OP, lhs, rhs)),
    }
}

pub(super) fn dot_general_with_conj(
    backend: &CudaBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<Tensor> {
    match (lhs, rhs) {
        (Tensor::F32(lhs), Tensor::F32(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::F32)
        }
        (Tensor::F64(lhs), Tensor::F64(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::F64)
        }
        (Tensor::C32(lhs), Tensor::C32(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::C32)
        }
        (Tensor::C64(lhs), Tensor::C64(rhs)) => {
            dot_general_typed_with_conj(backend, lhs, rhs, config, lhs_conj, rhs_conj)
                .map(Tensor::C64)
        }
        _ => Err(dtype_mismatch(OP, lhs, rhs)),
    }
}

/// Local extraction of typed accumulation coefficients; dtype mismatches
/// between the coefficient and the operand dtype are explicit errors.
trait FromContractionScalar: Sized {
    fn from_contraction_scalar(value: ContractionScalar) -> crate::Result<Self>;
}

macro_rules! impl_from_contraction_scalar {
    ($ty:ty, $variant:ident) => {
        impl FromContractionScalar for $ty {
            fn from_contraction_scalar(value: ContractionScalar) -> crate::Result<Self> {
                match value {
                    ContractionScalar::$variant(value) => Ok(value),
                    other => Err(Error::dtype_mismatch(
                        OP,
                        <$ty as tenferro_tensor::TensorScalar>::dtype(),
                        other.dtype(),
                    )),
                }
            }
        }
    };
}

impl_from_contraction_scalar!(f32, F32);
impl_from_contraction_scalar!(f64, F64);
impl_from_contraction_scalar!(Complex32, C32);
impl_from_contraction_scalar!(Complex64, C64);

/// Read-slot operand for the accumulate path: an owned compact tensor or a
/// borrowed strided view over a device buffer.
enum ReadOperand<'a, 'b, T> {
    Owned(&'a TypedTensor<T>),
    View(&'a TypedTensorView<'b, T>),
}

impl<T: 'static> ReadOperand<'_, '_, T> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Owned(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }
}

fn read_operand_alignment_requirement<T: CutensorScalar>(operand: &ReadOperand<'_, '_, T>) -> u32 {
    match operand {
        ReadOperand::Owned(_) => CUDA_ALLOCATION_ALIGNMENT,
        ReadOperand::View(_) => view_descriptor_alignment_requirement::<T>(),
    }
}

/// Write-slot operand for the accumulate path.
enum WriteOperand<'a, 'b, T> {
    Owned(&'a mut TypedTensor<T>),
    View(&'a mut TypedTensorViewMut<'b, T>),
}

impl<T: 'static> WriteOperand<'_, '_, T> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Owned(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    fn n_elements(&self) -> usize {
        match self {
            Self::Owned(tensor) => tensor.n_elements(),
            Self::View(view) => view.n_elements(),
        }
    }
}

fn write_operand_alignment_requirement<T: CutensorScalar>(
    operand: &WriteOperand<'_, '_, T>,
) -> u32 {
    match operand {
        WriteOperand::Owned(_) => CUDA_ALLOCATION_ALIGNMENT,
        WriteOperand::View(_) => view_descriptor_alignment_requirement::<T>(),
    }
}

fn view_descriptor_alignment_requirement<T: CutensorScalar>() -> u32 {
    u32::try_from(std::mem::size_of::<T>()).unwrap_or(CUDA_ALLOCATION_ALIGNMENT)
}

/// Device pointer plus cuTENSOR descriptor metadata for one operand.
/// Compact owned operands borrow the layout's precomputed strides; strided
/// views own their converted strides.
struct ResolvedOperand<'a> {
    ptr: *mut c_void,
    strides: std::borrow::Cow<'a, [i64]>,
    alignment: u32,
}

fn read_operand<'a, 'b, T: CutensorScalar>(
    read: &'a TensorRead<'b>,
) -> Option<ReadOperand<'a, 'b, T>> {
    match read {
        TensorRead::Tensor(tensor) => T::unwrap_tensor(tensor).map(ReadOperand::Owned),
        TensorRead::View(view) => T::unwrap_view(view).map(ReadOperand::View),
    }
}

fn write_operand<'a, 'b, T: CutensorScalar>(
    write: &'a mut TensorWrite<'b>,
) -> Option<WriteOperand<'a, 'b, T>> {
    match write {
        TensorWrite::Tensor(tensor) => T::unwrap_tensor_mut(tensor).map(WriteOperand::Owned),
        TensorWrite::View(view) => T::unwrap_view_mut(view).map(WriteOperand::View),
    }
}

/// CUDA-native accumulate-form contraction:
/// `out = alpha * op(lhs) * op(rhs) + beta * out` executed by a single
/// cuTENSOR contraction with `C = D = out` (no temporary result tensor).
///
/// Stage-2 scope (tensor4all/tenferro-rs#1287): every slot accepts either a
/// compact GPU-resident owned tensor or a borrowed strided view over a
/// GPU-resident device buffer. Host-backed views, negative view strides, and
/// out-of-bounds regions are explicit errors — no hidden host transfer and no
/// silent fallback.
pub(super) fn dot_general_read_into_accum(
    backend: &CudaBackend,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    out: &mut TensorWrite<'_>,
) -> crate::Result<()> {
    match lhs.dtype() {
        DType::F32 => accum_erased::<f32>(backend, lhs, rhs, config, accumulation, out),
        DType::F64 => accum_erased::<f64>(backend, lhs, rhs, config, accumulation, out),
        DType::C32 => accum_erased::<Complex32>(backend, lhs, rhs, config, accumulation, out),
        DType::C64 => accum_erased::<Complex64>(backend, lhs, rhs, config, accumulation, out),
        dtype => Err(unsupported_dtype(OP, dtype)),
    }
}

fn accum_erased<T>(
    backend: &CudaBackend,
    lhs: &TensorRead<'_>,
    rhs: &TensorRead<'_>,
    config: &DotGeneralConfig,
    accumulation: DotGeneralAccumulation,
    out: &mut TensorWrite<'_>,
) -> crate::Result<()>
where
    T: CutensorScalar + FromContractionScalar + PartialEq + tenferro_tensor::TensorScalar,
{
    let (lhs_dtype, rhs_dtype, out_dtype) = (lhs.dtype(), rhs.dtype(), out.dtype());
    let (Some(lhs), Some(rhs), Some(out)) = (
        read_operand::<T>(lhs),
        read_operand::<T>(rhs),
        write_operand::<T>(out),
    ) else {
        let (expected, actual) = if lhs_dtype != rhs_dtype {
            (lhs_dtype, rhs_dtype)
        } else {
            (lhs_dtype, out_dtype)
        };
        return Err(Error::dtype_mismatch(OP, expected, actual));
    };
    dot_general_typed_into_accum(
        backend,
        lhs,
        rhs,
        config,
        accumulation.lhs_conj,
        accumulation.rhs_conj,
        T::from_contraction_scalar(accumulation.alpha)?,
        T::from_contraction_scalar(accumulation.beta)?,
        out,
    )
}

#[allow(clippy::too_many_arguments)]
fn dot_general_typed_into_accum<T>(
    backend: &CudaBackend,
    lhs: ReadOperand<'_, '_, T>,
    rhs: ReadOperand<'_, '_, T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
    alpha: T,
    beta: T,
    mut out: WriteOperand<'_, '_, T>,
) -> crate::Result<()>
where
    T: CutensorScalar + PartialEq + tenferro_tensor::TensorScalar,
{
    backend.runtime().set_current_cuda_context(OP)?;
    validate_dot_general(lhs.shape(), rhs.shape(), config)?;
    let layout = build_layout(lhs.shape(), rhs.shape(), config)?;
    if out.shape() != layout.output_shape.as_slice() {
        return Err(Error::shape_mismatch(
            OP,
            out.shape().to_vec(),
            layout.output_shape.clone(),
        ));
    }
    // Residency, buffer-family, stride-sign, and bounds validation for all
    // three slots happens here, before any degenerate-case early return.
    let lhs_res = resolve_read_operand(backend.runtime(), &lhs, &layout.lhs_strides)?;
    let rhs_res = resolve_read_operand(backend.runtime(), &rhs, &layout.rhs_strides)?;
    let out_res = resolve_write_operand(backend.runtime(), &mut out, &layout.output_strides)?;
    if out.n_elements() == 0 {
        return Ok(());
    }
    if layout.contracting_elements == 0 {
        // The contraction sum is empty: out = beta * out.
        return match out {
            WriteOperand::Owned(tensor) => scale_in_place(backend.runtime(), tensor, beta),
            WriteOperand::View(_) => {
                if beta == T::one() {
                    Ok(())
                } else {
                    // No strided in-place scale kernel exists yet; an explicit
                    // error is required instead of a silent wrong result.
                    Err(unsupported_operation(
                        OP,
                        "zero-sized contraction with beta != 1 is not supported for borrowed view outputs",
                    ))
                }
            }
        };
    }

    let stream = raw_stream(backend.runtime())?;
    let spec = CutensorContractionSpec {
        layout: &layout,
        lhs_strides: &lhs_res.strides,
        rhs_strides: &rhs_res.strides,
        output_strides: &out_res.strides,
        lhs_alignment_requirement: read_operand_alignment_requirement(&lhs),
        rhs_alignment_requirement: read_operand_alignment_requirement(&rhs),
        output_alignment_requirement: write_operand_alignment_requirement(&out),
        lhs_conj,
        rhs_conj,
        workspace_preference: CutensorWorksizePreference::Default,
    };
    validate_descriptor_alignment(lhs_res.alignment, spec.lhs_alignment_requirement, "lhs")?;
    validate_descriptor_alignment(rhs_res.alignment, spec.rhs_alignment_requirement, "rhs")?;
    validate_descriptor_alignment(out_res.alignment, spec.output_alignment_requirement, "out")?;
    // C = D = out: cuTENSOR reads the destination as the accumulator (skipped
    // by cuTENSOR itself when beta == 0) and writes the result in place.
    // Overlap between the out region and the lhs/rhs regions is the caller's
    // responsibility, as with BLAS-style in-place update APIs.
    cached_cutensor_contraction::<T, _>(backend, &spec, |cutensor, plan, workspace| unsafe {
        cutensor.contract(
            plan,
            &alpha as *const T as *const c_void,
            lhs_res.ptr as *const c_void,
            rhs_res.ptr as *const c_void,
            &beta as *const T as *const c_void,
            out_res.ptr as *const c_void,
            out_res.ptr,
            workspace.ptr,
            workspace.size,
            stream,
            OP,
        )
    })
}

fn resolve_read_operand<'a, T>(
    rt: &CudaRuntime,
    operand: &ReadOperand<'_, '_, T>,
    compact_strides: &'a [i64],
) -> crate::Result<ResolvedOperand<'a>>
where
    T: CutensorScalar + 'static,
{
    match operand {
        ReadOperand::Owned(tensor) => Ok(ResolvedOperand {
            ptr: typed_device_ptr(rt, tensor)?,
            strides: std::borrow::Cow::Borrowed(compact_strides),
            alignment: CUDA_ALLOCATION_ALIGNMENT,
        }),
        ReadOperand::View(view) => {
            ensure_view_resident_on_runtime(rt, view, OP)?;
            let prepared = prepared_view_access(view, OP)?;
            resolve_prepared_device_region::<T>(rt, prepared, view.strides(), view.offset())
        }
    }
}

fn resolve_write_operand<'a, T>(
    rt: &CudaRuntime,
    operand: &mut WriteOperand<'_, '_, T>,
    compact_strides: &'a [i64],
) -> crate::Result<ResolvedOperand<'a>>
where
    T: CutensorScalar + 'static,
{
    match operand {
        WriteOperand::Owned(tensor) => Ok(ResolvedOperand {
            ptr: typed_device_ptr(rt, tensor)?,
            strides: std::borrow::Cow::Borrowed(compact_strides),
            alignment: CUDA_ALLOCATION_ALIGNMENT,
        }),
        WriteOperand::View(view) => {
            ensure_view_mut_resident_on_runtime(rt, view, OP)?;
            let prepared = prepared_view_mut_access(view, OP)?;
            resolve_prepared_device_region::<T>(rt, prepared, view.strides(), view.offset())
        }
    }
}

/// Resolve a strided view region over a device buffer into an effective
/// cuTENSOR operand: `ptr = base + offset * size_of::<T>()`, the view's own
/// element strides, and the alignment actually guaranteed by the effective
/// byte address.
fn resolve_prepared_device_region<T: CutensorScalar + 'static>(
    rt: &CudaRuntime,
    prepared: CubeclPreparedAccess,
    strides: &[isize],
    offset: isize,
) -> crate::Result<ResolvedOperand<'static>> {
    let mut strides_i64 = Vec::with_capacity(strides.len());
    for &stride in strides {
        if stride < 0 {
            return Err(Error::invalid_argument(
                OP,
                "layout",
                format!(
                    "cuTENSOR dot-general accumulation requires nonnegative view strides, got {strides:?}; canonicalize the view on device first"
                ),
            ));
        }
        strides_i64.push(stride as i64);
    }
    let offset = usize::try_from(offset)
        .map_err(|_| Error::invalid_argument(OP, "layout", "view offset must be nonnegative"))?;
    let handle = prepared.into_handle();
    let resource = rt
        .client()
        .get_resource(handle)
        .map_err(|err| Error::backend_source(OP, err))?;
    let offset_bytes = offset
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::invalid_argument(OP, "layout", "view byte offset overflows"))?;
    let addr = resource
        .resource()
        .ptr
        .checked_add(offset_bytes as u64)
        .ok_or_else(|| Error::invalid_argument(OP, "layout", "view device address overflows"))?;
    // INVARIANT: CubeCL root allocations are at least 256-byte aligned, and
    // the checked element offset preserves alignment to the scalar size. A
    // strided view therefore guarantees the scalar-sized cuTENSOR requirement,
    // even when the view starts inside the root allocation.
    Ok(ResolvedOperand {
        ptr: cuda_device_ptr_from_addr(addr, OP)?,
        strides: std::borrow::Cow::Owned(strides_i64),
        alignment: view_descriptor_alignment_requirement::<T>(),
    })
}

/// Device-side `out *= beta` for the degenerate zero-contraction case. The
/// factor is materialized as an explicit one-element device constant; user
/// operand tensors are never transferred.
fn scale_in_place<T>(rt: &CudaRuntime, out: &mut TypedTensor<T>, beta: T) -> crate::Result<()>
where
    T: CutensorScalar + PartialEq + tenferro_tensor::TensorScalar,
{
    if beta == T::one() {
        return Ok(());
    }
    if beta == T::zero() {
        return launch_nullary_into(
            rt,
            out,
            OP,
            cube_count_for_len(out.n_elements())?,
            cube_dim_1d(),
            |client, count, dim, out| unsafe {
                structural::fill_zero_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client, count, dim, out,
                );
            },
        );
    }
    super::interop::scale_typed_tensor_for_op(rt, out, beta, OP, T::launch_scale_in_place)
}

fn dot_general_typed<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
) -> crate::Result<TypedTensor<T>>
where
    T: CutensorScalar,
{
    dot_general_typed_with_conj(backend, lhs, rhs, config, false, false)
}

fn dot_general_typed_with_conj<T>(
    backend: &CudaBackend,
    lhs: &TypedTensor<T>,
    rhs: &TypedTensor<T>,
    config: &DotGeneralConfig,
    lhs_conj: bool,
    rhs_conj: bool,
) -> crate::Result<TypedTensor<T>>
where
    T: CutensorScalar,
{
    backend.runtime().set_current_cuda_context(OP)?;
    validate_dot_general(lhs.shape(), rhs.shape(), config)?;
    let layout = build_layout(lhs.shape(), rhs.shape(), config)?;
    let output = alloc_output::<T>(backend.runtime(), &layout.output_shape)?;
    if output.n_elements() == 0 {
        return Ok(output);
    }
    if layout.contracting_elements == 0 {
        // The contraction sum is empty: fill the already-allocated output with
        // zeros instead of allocating a second output tensor.
        launch_nullary_into(
            backend.runtime(),
            &output,
            OP,
            cube_count_for_len(output.n_elements())?,
            cube_dim_1d(),
            |client, count, dim, out| unsafe {
                structural::fill_zero_kernel::launch_unchecked::<T, CubeclCudaRuntime>(
                    client, count, dim, out,
                );
            },
        )?;
        return Ok(output);
    }

    let lhs_ptr = typed_device_ptr(backend.runtime(), lhs)?;
    let rhs_ptr = typed_device_ptr(backend.runtime(), rhs)?;
    let output_ptr = typed_device_ptr(backend.runtime(), &output)?;

    let alpha = T::one();
    let beta = T::zero();
    let stream = raw_stream(backend.runtime())?;
    let spec = CutensorContractionSpec {
        layout: &layout,
        lhs_strides: &layout.lhs_strides,
        rhs_strides: &layout.rhs_strides,
        output_strides: &layout.output_strides,
        lhs_alignment_requirement: CUDA_ALLOCATION_ALIGNMENT,
        rhs_alignment_requirement: CUDA_ALLOCATION_ALIGNMENT,
        output_alignment_requirement: CUDA_ALLOCATION_ALIGNMENT,
        lhs_conj,
        rhs_conj,
        workspace_preference: CutensorWorksizePreference::Default,
    };
    // C = D = output: cuTENSOR never reads the accumulator slot when
    // beta == 0, so the freshly allocated output serves as both C and D and
    // no separate accumulator tensor is needed.
    cached_cutensor_contraction::<T, _>(backend, &spec, |cutensor, plan, workspace| unsafe {
        cutensor.contract(
            plan,
            &alpha as *const T as *const c_void,
            lhs_ptr as *const c_void,
            rhs_ptr as *const c_void,
            &beta as *const T as *const c_void,
            output_ptr as *const c_void,
            output_ptr,
            workspace.ptr,
            workspace.size,
            stream,
            OP,
        )
    })?;

    Ok(output)
}

fn cutensor_conj_op<T: CutensorScalar>(conj: bool) -> CutensorOperator {
    if conj && T::IS_COMPLEX {
        CutensorOperator::Conj
    } else {
        CutensorOperator::Identity
    }
}

fn default_cutensor_plan_cache_max_entries() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_CUTENSOR_PLAN_CACHE_MAX_ENTRIES).unwrap_or(NonZeroUsize::MIN)
}

fn new_cutensor_plan_cache_state(max_entries: NonZeroUsize) -> CutensorPlanCacheState {
    Arc::new(Mutex::new(CutensorContractionPlanCache::new(max_entries)))
}

fn get_or_init_cutensor_plan_cache(backend: &CudaBackend) -> crate::Result<CutensorPlanCacheState> {
    let guard = backend
        .cuda_extension_cache()
        .get_or_try_init::<CutensorPlanCacheState>(|| {
            Ok(new_cutensor_plan_cache_state(
                default_cutensor_plan_cache_max_entries(),
            ))
        })?;
    Ok(Arc::clone(&guard))
}

fn lock_cutensor_plan_cache(
    cache: &CutensorPlanCacheState,
) -> crate::Result<std::sync::MutexGuard<'_, CutensorContractionPlanCache>> {
    cache
        .lock()
        .map_err(|_| Error::runtime_state("cutensor_plan_cache", "plan cache lock poisoned"))
}

pub(super) fn cutensor_plan_cache_stats(backend: &CudaBackend) -> crate::Result<CacheStats> {
    let Some(plan_cache) = backend
        .cuda_extension_cache()
        .get_cloned::<CutensorPlanCacheState>()?
    else {
        return Ok(CacheStats::empty());
    };
    let plan_cache = lock_cutensor_plan_cache(&plan_cache)?;
    Ok(plan_cache.stats())
}

#[cfg(test)]
pub(super) fn cutensor_plan_cache_workspace_bytes(backend: &CudaBackend) -> crate::Result<u64> {
    let Some(plan_cache) = backend
        .cuda_extension_cache()
        .get_cloned::<CutensorPlanCacheState>()?
    else {
        return Ok(0);
    };
    let plan_cache = lock_cutensor_plan_cache(&plan_cache)?;
    Ok(plan_cache.values().fold(0, |total, cached| {
        total.saturating_add(cached.workspace.size)
    }))
}

pub(super) fn cutensor_plan_cache_max_entries(
    backend: &CudaBackend,
) -> crate::Result<NonZeroUsize> {
    let Some(plan_cache) = backend
        .cuda_extension_cache()
        .get_cloned::<CutensorPlanCacheState>()?
    else {
        return Ok(default_cutensor_plan_cache_max_entries());
    };
    let plan_cache = lock_cutensor_plan_cache(&plan_cache)?;
    Ok(plan_cache.max_entries())
}

pub(super) fn set_cutensor_plan_cache_max_entries(
    backend: &CudaBackend,
    max_entries: NonZeroUsize,
) -> crate::Result<()> {
    let plan_cache = get_or_init_cutensor_plan_cache(backend)?;
    let mut plan_cache = lock_cutensor_plan_cache(&plan_cache)?;
    plan_cache.set_max_entries(max_entries);
    let retained_bytes = plan_cache.retained_bytes();
    backend
        .cuda_extension_cache()
        .update_retained_bytes::<CutensorPlanCacheState>(retained_bytes)
}

fn cached_cutensor_contraction<T, R>(
    backend: &CudaBackend,
    spec: &CutensorContractionSpec<'_>,
    execute: impl FnOnce(&CutensorHandle, &Plan, &Workspace) -> crate::Result<R>,
) -> crate::Result<R>
where
    T: CutensorScalar,
{
    let cutensor = backend.cutensor_handle()?;
    let hash = spec_hash::<T>(spec);
    let plan_cache = get_or_init_cutensor_plan_cache(backend)?;
    let mut plan_cache = lock_cutensor_plan_cache(&plan_cache)?;
    let entries_changed = plan_cache.ensure(
        hash,
        |key| key_matches_spec::<T>(key, spec),
        || {
            let cached = CachedCutensorContraction::new::<T>(backend.runtime(), cutensor, spec)?;
            let key = CutensorContractionKey::from_spec::<T>(spec);
            let retained_bytes = key.retained_bytes().saturating_add(cached.retained_bytes());
            Ok((key, cached, retained_bytes))
        },
    )?;
    if entries_changed {
        // Retained bytes only move on insert/evict, so cache hits skip the
        // extension-cache accounting write entirely.
        let retained_bytes = plan_cache.retained_bytes();
        backend
            .cuda_extension_cache()
            .update_retained_bytes::<CutensorPlanCacheState>(retained_bytes)?;
    }
    let cached = plan_cache
        .get(hash, |key| key_matches_spec::<T>(key, spec))
        .ok_or_else(|| {
            Error::runtime_state(
                "cutensor_plan_cache",
                "cached cuTENSOR contraction was evicted before use",
            )
        })?;
    execute(cutensor, &cached.plan, &cached.workspace)
}

fn validate_descriptor_alignment(
    actual_alignment: u32,
    alignment_requirement: u32,
    slot: &'static str,
) -> crate::Result<()> {
    if actual_alignment >= alignment_requirement {
        return Ok(());
    }
    Err(Error::invalid_argument(
        OP,
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

fn alloc_workspace(rt: &CudaRuntime, workspace_size: u64) -> crate::Result<Workspace> {
    if workspace_size == 0 {
        return Ok(Workspace::none());
    }
    let workspace_len =
        usize::try_from(workspace_size).map_err(|_| workspace_size_overflow(OP, workspace_size))?;
    let handle = rt.client().empty(workspace_len);
    let resource = rt
        .client()
        .get_resource(handle.clone())
        .map_err(|err| crate::Error::backend_source(OP, err))?;
    Ok(Workspace {
        _handle: Some(handle),
        ptr: cuda_device_ptr_from_addr(resource.resource().ptr, OP)?,
        size: workspace_size,
    })
}

fn typed_device_ptr<T: TensorScalar + 'static>(
    rt: &CudaRuntime,
    tensor: &TypedTensor<T>,
) -> crate::Result<*mut c_void> {
    ensure_resident_on_runtime(rt, tensor, OP)?;
    let prepared = prepared_tensor_access(tensor, OP)?;
    let buffer = cubecl_buffer(tensor, OP)?;
    // Fast path: reuse the buffer's memoized device address when executing on
    // the stream that created the allocation. In that case `get_resource` is
    // only a pointer lookup — CubeCL's cross-stream alignment pass skips
    // bindings whose creation stream equals the current stream — so no
    // synchronization is lost. Any other stream takes the full `get_resource`
    // round trip below, preserving CubeCL's cross-stream alignment.
    let same_stream = StreamId::current() == buffer.handle().stream;
    if same_stream {
        if let Some(addr) = buffer.cached_device_addr() {
            // The residency check above ties this raw FFI pointer to the
            // caller's runtime/device.
            return cuda_device_ptr_from_addr(addr, OP);
        }
    }
    let handle = prepared.into_handle();
    let resource = rt
        .client()
        .get_resource(handle)
        .map_err(|err| crate::Error::backend_source(OP, err))?;
    let addr = resource.resource().ptr;
    // See `CubeclBuffer::device_addr` for the address-stability invariant.
    buffer.memoize_device_addr(addr);
    // The residency check above ties this raw FFI pointer to the caller's runtime/device.
    cuda_device_ptr_from_addr(addr, OP)
}

fn build_layout(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &DotGeneralConfig,
) -> crate::Result<DotGeneralLayout> {
    let lhs_free = free_axes(
        lhs_shape.len(),
        &config.lhs_contracting_dims,
        &config.lhs_batch_dims,
    );
    let rhs_free = free_axes(
        rhs_shape.len(),
        &config.rhs_contracting_dims,
        &config.rhs_batch_dims,
    );

    let mut lhs_modes = vec![-1i32; lhs_shape.len()];
    let mut rhs_modes = vec![-1i32; rhs_shape.len()];
    let mut output_modes =
        Vec::with_capacity(lhs_free.len() + rhs_free.len() + config.lhs_batch_dims.len());
    let mut output_shape = Vec::with_capacity(output_modes.capacity());
    let mut batch_modes = Vec::with_capacity(config.lhs_batch_dims.len());
    let mut batch_shape = Vec::with_capacity(config.lhs_batch_dims.len());
    let mut next_mode = 0i32;
    let mut contracting_elements = 1usize;

    for (&lhs_axis, &rhs_axis) in config
        .lhs_contracting_dims
        .iter()
        .zip(&config.rhs_contracting_dims)
    {
        let mode = next_mode;
        next_mode += 1;
        lhs_modes[lhs_axis] = mode;
        rhs_modes[rhs_axis] = mode;
        contracting_elements = contracting_elements
            .checked_mul(lhs_shape[lhs_axis])
            .ok_or_else(|| {
                Error::invalid_argument(
                    OP,
                    "shape",
                    format!(
                        "contracting dimension product overflows usize for lhs shape {lhs_shape:?}"
                    ),
                )
            })?;
    }

    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        let mode = next_mode;
        next_mode += 1;
        lhs_modes[lhs_axis] = mode;
        rhs_modes[rhs_axis] = mode;
        batch_modes.push(mode);
        batch_shape.push(lhs_shape[lhs_axis]);
    }

    for &lhs_axis in &lhs_free {
        let mode = next_mode;
        next_mode += 1;
        lhs_modes[lhs_axis] = mode;
        output_modes.push(mode);
        output_shape.push(lhs_shape[lhs_axis]);
    }

    for &rhs_axis in &rhs_free {
        let mode = next_mode;
        next_mode += 1;
        rhs_modes[rhs_axis] = mode;
        output_modes.push(mode);
        output_shape.push(rhs_shape[rhs_axis]);
    }

    output_modes.extend_from_slice(&batch_modes);
    output_shape.extend_from_slice(&batch_shape);

    let lhs_extents = dims_to_i64(lhs_shape)?;
    let rhs_extents = dims_to_i64(rhs_shape)?;
    let output_extents = dims_to_i64(&output_shape)?;
    let lhs_strides = strides_to_i64(&col_major_strides(lhs_shape)?)?;
    let rhs_strides = strides_to_i64(&col_major_strides(rhs_shape)?)?;
    let output_strides = strides_to_i64(&col_major_strides(&output_shape)?)?;

    Ok(DotGeneralLayout {
        lhs_modes,
        rhs_modes,
        output_modes,
        output_shape,
        lhs_extents,
        rhs_extents,
        output_extents,
        lhs_strides,
        rhs_strides,
        output_strides,
        contracting_elements,
    })
}

fn dims_to_i64(dims: &[usize]) -> crate::Result<Vec<i64>> {
    dims.iter()
        .map(|&dim| {
            i64::try_from(dim).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "shape",
                    format!("extent {dim} exceeds cuTENSOR i64 limit"),
                )
            })
        })
        .collect()
}

fn strides_to_i64(strides: &[isize]) -> crate::Result<Vec<i64>> {
    strides
        .iter()
        .map(|&stride| {
            i64::try_from(stride).map_err(|_| {
                Error::invalid_argument(
                    OP,
                    "stride",
                    format!("stride {stride} exceeds cuTENSOR i64 limit"),
                )
            })
        })
        .collect()
}

fn free_axes(rank: usize, contracting: &[usize], batch: &[usize]) -> Vec<usize> {
    (0..rank)
        .filter(|axis| !contracting.contains(axis) && !batch.contains(axis))
        .collect()
}

fn validate_axis_list(
    op: &'static str,
    role: &'static str,
    axes: &[usize],
    rank: usize,
) -> crate::Result<()> {
    let mut seen = vec![false; rank];
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
                tenferro_tensor::ValidationError::AxisRoleConflict {
                    axis,
                    first_role,
                    second_role,
                },
            ));
        }
    }
    Ok(())
}

fn validate_dot_general(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    config: &DotGeneralConfig,
) -> crate::Result<()> {
    if config.lhs_contracting_dims.len() != config.rhs_contracting_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "contracting_dims",
            "lhs/rhs contracting dim counts differ",
        ));
    }
    if config.lhs_batch_dims.len() != config.rhs_batch_dims.len() {
        return Err(Error::invalid_argument(
            OP,
            "batch_dims",
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
            return Err(Error::validation(
                OP,
                tenferro_tensor::ShapeMismatch::ContractedDimensions {
                    lhs_axis,
                    lhs_size: lhs_shape[lhs_axis],
                    rhs_axis,
                    rhs_size: rhs_shape[rhs_axis],
                }
                .into(),
            ));
        }
    }

    for (&lhs_axis, &rhs_axis) in config.lhs_batch_dims.iter().zip(&config.rhs_batch_dims) {
        if lhs_shape[lhs_axis] != rhs_shape[rhs_axis] {
            return Err(Error::shape_mismatch(
                OP,
                lhs_shape.to_vec(),
                rhs_shape.to_vec(),
            ));
        }
    }

    Ok(())
}
