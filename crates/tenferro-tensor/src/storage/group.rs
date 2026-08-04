use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;

use crate::types::{tensor_from_group, tensor_view_from_group};
use crate::{DType, DynRank, Placement, TensorLayout, TensorRank, TensorRead, TensorScalar};
use smallvec::SmallVec;

use super::prepared::{
    prepare_read, prepare_write, validate_descriptor, AccessError, AccessTarget, CheckedDescriptor,
    CheckedRead, CheckedWrite, PreparedRead, PreparedWrite, ProviderReadMapping,
    ProviderWriteMapping, WriteInjectivityProof,
};
use super::root::{BackendAllocation, OwnedStorage, ProviderKind};
use super::span::{ByteRange, RootBoundSpan};

/// A group-local append-only allocation entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AllocationSlot(u32);

impl AllocationSlot {
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }

    #[cfg(test)]
    pub(crate) const fn test_raw(raw: u32) -> Self {
        Self(raw)
    }
}

/// A group-local descriptor lookup key. It carries no ownership authority.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::DescriptorSlot;
///
/// let slot = DescriptorSlot::from_index(3).unwrap();
/// assert_eq!(slot.index(), 3);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DescriptorSlot(u32);

impl DescriptorSlot {
    /// Return the zero-based descriptor index.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DescriptorSlot;
    ///
    /// let slot = DescriptorSlot::from_index(2).unwrap();
    /// assert_eq!(slot.index(), 2);
    /// ```
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Convert a host index into a descriptor slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::DescriptorSlot;
    ///
    /// assert_eq!(DescriptorSlot::from_index(1).unwrap().index(), 1);
    /// assert!(DescriptorSlot::from_index(usize::MAX).is_none());
    /// ```
    pub fn from_index(index: usize) -> Option<Self> {
        match u32::try_from(index) {
            Ok(index) => Some(Self(index)),
            Err(_) => None,
        }
    }

    #[cfg(test)]
    pub(crate) const fn test_raw(raw: u32) -> Self {
        Self(raw)
    }
}

/// Construction input for one rank-specific descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DescriptorInput<R: TensorRank> {
    relative: ByteRange,
    shape: R::Shape,
    strides: R::Strides,
    offset: isize,
    require_injective: bool,
}

impl<R: TensorRank> DescriptorInput<R> {
    pub(crate) fn new(
        relative: ByteRange,
        shape: R::Shape,
        strides: R::Strides,
        offset: isize,
        require_injective: bool,
    ) -> Self {
        Self {
            relative,
            shape,
            strides,
            offset,
            require_injective,
        }
    }
}

/// One validated, non-owning logical descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DescriptorRecord {
    allocation: AllocationSlot,
    root: super::identity::RootResourceIdentity,
    span: RootBoundSpan,
    layout: TensorLayout<DynRank>,
    dtype: DType,
    element_size: usize,
    element_count: usize,
    provider: ProviderKind,
    placement: Placement,
    envelope: Option<ByteRange>,
    write_injective: bool,
    checked: CheckedDescriptor<DynRank>,
}

impl DescriptorRecord {
    pub(crate) const fn allocation(&self) -> AllocationSlot {
        self.allocation
    }

    pub(crate) const fn span(&self) -> RootBoundSpan {
        self.span
    }

    pub(crate) const fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) const fn element_size(&self) -> usize {
        self.element_size
    }

    pub(crate) const fn element_count(&self) -> usize {
        self.element_count
    }

    pub(crate) fn layout(&self) -> &TensorLayout<DynRank> {
        &self.layout
    }

    pub(crate) const fn provider(&self) -> ProviderKind {
        self.provider
    }

    pub(crate) fn placement(&self) -> &Placement {
        &self.placement
    }

    pub(crate) const fn envelope(&self) -> Option<ByteRange> {
        self.envelope
    }

    pub(crate) const fn write_injective(&self) -> bool {
        self.write_injective
    }
}

/// Group construction and slot errors.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::GroupError;
///
/// let error = GroupError::IndexOverflow;
/// assert!(error.to_string().contains("overflows"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GroupError {
    #[error("group index overflows u32")]
    IndexOverflow,
    #[error("descriptor slot {slot} is outside the group")]
    DescriptorSlotOutOfBounds { slot: usize },
    #[error("descriptor slot {slot} is vacant")]
    DescriptorSlotVacant { slot: usize },
    #[error("allocation slot {slot} is outside the group")]
    AllocationSlotOutOfBounds { slot: usize },
    #[error("allocation slot {slot} is vacant")]
    AllocationSlotVacant { slot: usize },
    #[error("descriptor validation failed: {message}")]
    InvalidDescriptor { message: String },
    #[error("descriptor dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("descriptor rank mismatch: expected {expected}, actual {actual}")]
    RankMismatch { expected: usize, actual: usize },
    #[error("allocation slot {allocation} has more than one live descriptor")]
    AliasedAllocation { allocation: usize },
}

/// N-way mutable split errors. Every error leaves the group unchanged.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DisjointViewError {
    #[error(transparent)]
    Group(#[from] GroupError),
    #[error("descriptor slot {slot} appears more than once")]
    DuplicateSlot { slot: usize },
    #[error("descriptor slot {slot} has a non-injective mutable layout")]
    NonInjective { slot: usize },
    #[error("requested mutable descriptor envelopes overlap")]
    PairwiseOverlap,
    #[error("requested mutable descriptors are not provably disjoint")]
    NotProvablyDisjoint,
}

/// Structural extraction errors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum ExtractError {
    #[error(transparent)]
    Group(#[from] GroupError),
    #[error("allocation slot {allocation} still has another descriptor")]
    AliasedAllocation { allocation: usize },
}

/// One group of move-only owners and append-only logical descriptors.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::AllocationGroup;
///
/// let (group, bindings) = AllocationGroup::from_tensors(Vec::new())?;
/// assert!(bindings.is_empty());
/// assert!(format!("{group:?}").contains("AllocationGroup"));
/// # Ok::<(), tenferro_tensor::GroupError>(())
/// ```
#[derive(Default)]
pub struct AllocationGroup {
    // Most public tensors contain one root and one descriptor. Keep that
    // common case inline so the ownership boundary does not add a per-result
    // metadata allocation to CPU hot paths; the vectors still grow for
    // explicit multi-descriptor groups.
    allocations: SmallVec<[Option<OwnedStorage>; 1]>,
    descriptors: SmallVec<[Option<DescriptorRecord>; 1]>,
}

/// A shared descriptor child bounded by the group's shared borrow.
pub(crate) struct GroupReadView<'a, T, R: TensorRank> {
    owner: NonNull<OwnedStorage>,
    descriptor: DescriptorRecord,
    _borrow: PhantomData<(&'a OwnedStorage, T, R)>,
}

// SAFETY: the view is a shared capability over an immutable root borrow;
// `OwnedStorage` is `Sync`, and the descriptor contains only checked metadata.
unsafe impl<'a, T: Send, R: TensorRank> Send for GroupReadView<'a, T, R> {}
unsafe impl<'a, T: Sync, R: TensorRank> Sync for GroupReadView<'a, T, R> {}

impl<'a, T, R: TensorRank> Clone for GroupReadView<'a, T, R> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner,
            descriptor: self.descriptor.clone(),
            _borrow: PhantomData,
        }
    }
}

impl<'a, T, R: TensorRank> GroupReadView<'a, T, R> {
    pub(crate) fn clone_dyn(&self) -> GroupReadView<'a, T, crate::DynRank> {
        GroupReadView {
            owner: self.owner,
            descriptor: self.descriptor.clone(),
            _borrow: PhantomData,
        }
    }
}

impl<T, R: TensorRank> std::fmt::Debug for GroupReadView<'_, T, R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroupReadView")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl<'a, T: TensorScalar, R: TensorRank> GroupReadView<'a, T, R> {
    pub(crate) fn descriptor(&self) -> &DescriptorRecord {
        &self.descriptor
    }

    pub(crate) fn provider_kind(&self) -> ProviderKind {
        self.descriptor.provider
    }

    pub(crate) fn backend_identity(
        &self,
    ) -> Option<(crate::AllocationDomainId, crate::AllocationId)> {
        if self.descriptor.provider == ProviderKind::Cpu {
            return None;
        }
        let key = unsafe { self.owner.as_ref().root_identity().extent().key() };
        Some((key.domain(), key.local()))
    }

    pub(crate) fn storage_buffer(&self) -> Option<&'a crate::StorageBuffer<T>> {
        // SAFETY: the owner pointer is bounded by the group borrow carried by
        // this view, and root storage never changes its concrete buffer.
        let buffer = unsafe {
            self.owner
                .as_ref()
                .host_buffer::<T>()
                .or_else(|| self.owner.as_ref().backend_buffer::<T>())
        }?;
        Some(unsafe {
            std::mem::transmute::<&crate::StorageBuffer<T>, &'a crate::StorageBuffer<T>>(buffer)
        })
    }

    pub(crate) fn map_read(&self) -> Result<ProviderReadMapping<'_>, AccessError> {
        // SAFETY: `owner` points into the group borrowed for `'a`; this method
        // creates only a shared owner capability and maps the retained span.
        unsafe {
            self.owner
                .as_ref()
                .as_ref()
                .map_read(self.descriptor.span, self.descriptor.dtype)
        }
    }

    pub(crate) fn backend_allocation(&self) -> Option<&'a dyn BackendAllocation> {
        let allocation = unsafe { self.owner.as_ref().backend_allocation() }?;
        Some(unsafe {
            std::mem::transmute::<&dyn BackendAllocation, &'a dyn BackendAllocation>(allocation)
        })
    }

    pub(crate) fn prepare_device_read_for_layout(
        &self,
        layout: &TensorLayout<R>,
    ) -> Result<Box<dyn crate::PreparedDeviceAccess + 'a>, AccessError> {
        let owner: crate::storage::root::StorageRef<'a> = unsafe { self.owner.as_ref().as_ref() };
        let checked: CheckedRead<'a, R> = CheckedRead::new::<T>(
            // SAFETY: `owner` is bounded by the group's shared borrow.
            owner,
            self.descriptor.span,
            R::shape_from_vec(layout.shape().iter().copied().collect()).map_err(|error| {
                AccessError::InvalidLayout {
                    message: error.to_string(),
                }
            })?,
            R::strides_from_vec(layout.strides().iter().copied().collect()).map_err(|error| {
                AccessError::InvalidLayout {
                    message: error.to_string(),
                }
            })?,
            layout.offset(),
        )?;
        prepare_read::<T, R>(checked, AccessTarget::Device)
            .map_err(|failure| failure.1)?
            .into_device_state()
    }

    pub(crate) fn prepare_host_read(&self) -> Result<PreparedRead<'_, T, DynRank>, AccessError> {
        prepare_read(
            CheckedRead::from_validated(
                // SAFETY: the pointer is bounded by the group's shared borrow.
                unsafe { self.owner.as_ref().as_ref() },
                self.descriptor.checked.clone(),
            ),
            AccessTarget::Host,
        )
        .map_err(|failure| failure.1)
    }

    pub(crate) fn host_slice(&self) -> Result<&'a [T], AccessError> {
        // SAFETY: `owner` is bounded by the group borrow carried by `'a`.
        unsafe {
            self.owner
                .as_ref()
                .as_ref()
                .host_slice(self.descriptor.span, self.descriptor.dtype)
        }
    }
}

impl<'a, T: 'static, R: TensorRank> GroupReadView<'a, T, R> {
    pub(crate) fn backend_buffer(&self) -> Option<&'a crate::StorageBuffer<T>> {
        // SAFETY: the owner pointer is bounded by the group borrow carried by
        // this view, and the root buffer cannot be resized after import.
        let buffer = unsafe { self.owner.as_ref().backend_buffer::<T>() }?;
        Some(unsafe {
            std::mem::transmute::<&crate::StorageBuffer<T>, &'a crate::StorageBuffer<T>>(buffer)
        })
    }
}

/// A non-cloneable mutable descriptor child bounded by the group's exclusive
/// borrow. The raw owner pointer is never exposed and is dereferenced only for
/// a provider mapping whose retained byte envelope was proven by the group.
pub(crate) struct GroupWriteView<'a, T, R: TensorRank> {
    owner: NonNull<OwnedStorage>,
    descriptor: DescriptorRecord,
    _borrow: PhantomData<(&'a mut [u8], T, R)>,
}

// SAFETY: this is the exclusive capability for one owner borrow; moving it
// transfers that exclusive borrow and cannot create a second access path.
unsafe impl<'a, T: Send, R: TensorRank> Send for GroupWriteView<'a, T, R> {}

impl<T, R: TensorRank> std::fmt::Debug for GroupWriteView<'_, T, R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroupWriteView")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl<'a, T: TensorScalar, R: TensorRank> GroupWriteView<'a, T, R> {
    pub(crate) fn descriptor(&self) -> &DescriptorRecord {
        &self.descriptor
    }

    pub(crate) fn map_write(&mut self) -> Result<ProviderWriteMapping<'_>, AccessError> {
        // SAFETY: the group proof established that this child owns a distinct
        // reachable byte envelope; the temporary exclusive capability does not
        // escape this mapping call except through that provider span guard.
        unsafe {
            self.owner
                .as_mut()
                .as_mut()
                .map_write(self.descriptor.span, self.descriptor.dtype)
        }
    }

    pub(crate) fn backend_buffer_mut(&mut self) -> Option<&'a mut crate::StorageBuffer<T>> {
        // SAFETY: this child carries the group's exclusive borrow, so the
        // mutable root projection cannot alias another owner capability.
        let owner = unsafe { &mut *self.owner.as_ptr() };
        let buffer = owner.backend_buffer_mut::<T>()?;
        Some(unsafe {
            std::mem::transmute::<&mut crate::StorageBuffer<T>, &'a mut crate::StorageBuffer<T>>(
                buffer,
            )
        })
    }

    pub(crate) fn prepare_device_write_for_layout(
        &mut self,
        layout: &TensorLayout<R>,
    ) -> Result<Box<dyn crate::PreparedDeviceAccess + 'a>, AccessError> {
        let owner: crate::storage::root::StorageMut<'a> =
            unsafe { (&mut *self.owner.as_ptr()).as_mut() };
        let checked: CheckedWrite<'a, R> = CheckedWrite::new::<T>(
            // SAFETY: this child carries the group's exclusive borrow.
            owner,
            self.descriptor.span,
            R::shape_from_vec(layout.shape().iter().copied().collect()).map_err(|error| {
                AccessError::InvalidLayout {
                    message: error.to_string(),
                }
            })?,
            R::strides_from_vec(layout.strides().iter().copied().collect()).map_err(|error| {
                AccessError::InvalidLayout {
                    message: error.to_string(),
                }
            })?,
            layout.offset(),
        )?;
        prepare_write::<T, R>(checked, AccessTarget::Device)
            .map_err(|failure| failure.1)?
            .into_device_state()
    }

    pub(crate) fn prepare_host_write(
        &mut self,
    ) -> Result<PreparedWrite<'_, T, DynRank>, AccessError> {
        let checked = CheckedWrite::from_validated(
            // SAFETY: the pointer is bounded by this child group's exclusive borrow.
            unsafe { self.owner.as_mut().as_mut() },
            self.descriptor.checked.clone(),
            WriteInjectivityProof,
        );
        prepare_write(checked, AccessTarget::Host).map_err(|failure| failure.1)
    }

    pub(crate) fn host_slice_mut(&mut self) -> Result<&'a mut [T], AccessError> {
        // SAFETY: the group borrow is exclusive for `'a` and the descriptor
        // retains the checked span used by the private host root.
        unsafe {
            self.owner
                .as_mut()
                .as_mut()
                .host_slice_mut(self.descriptor.span, self.descriptor.dtype)
        }
    }
}

impl<'a, T: 'static, R: TensorRank> GroupWriteView<'a, T, R> {
    pub(crate) fn backend_buffer(&self) -> Option<&crate::StorageBuffer<T>> {
        // SAFETY: the owner pointer is bounded by the group's exclusive borrow;
        // this shared inspection does not expose a mutable projection.
        unsafe { self.owner.as_ref().backend_buffer::<T>() }
    }
}

impl fmt::Debug for AllocationGroup {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AllocationGroup")
            .field("allocation_count", &self.allocations.len())
            .field("descriptor_count", &self.descriptors.len())
            .finish()
    }
}

impl AllocationGroup {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Build one move-only group for detached runtime input ownership.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::AllocationGroup;
    ///
    /// let (group, bindings) = AllocationGroup::from_tensors(Vec::new())?;
    /// assert!(bindings.is_empty());
    /// assert!(format!("{group:?}").contains("AllocationGroup"));
    /// # Ok::<(), tenferro_tensor::GroupError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`GroupError::IndexOverflow`] when the input count cannot be
    /// represented by a descriptor slot.
    pub fn from_tensors(
        tensors: Vec<crate::Tensor>,
    ) -> Result<(Self, Box<[DescriptorSlot]>), GroupError> {
        let mut group = Self::new();
        let mut bindings = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            let (source, source_slot) = tensor.into_group_parts();
            bindings.push(group.append_group(source, source_slot)?);
        }
        Ok((group, bindings.into_boxed_slice()))
    }

    /// Borrow dtype-erased read views for descriptor bindings without
    /// materializing or cloning any owner.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError::DescriptorSlotOutOfBounds`],
    /// [`GroupError::DescriptorSlotVacant`], or [`GroupError::InvalidDescriptor`]
    /// when a binding does not identify a valid descriptor.
    pub fn read_views<'a>(
        &'a self,
        bindings: &[DescriptorSlot],
    ) -> Result<Vec<TensorRead<'a>>, GroupError> {
        bindings
            .iter()
            .map(|&slot| self.tensor_read(slot))
            .collect()
    }

    /// Borrow one dtype-erased read view for a descriptor without materializing
    /// or cloning its physical owner.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError::DescriptorSlotOutOfBounds`],
    /// [`GroupError::DescriptorSlotVacant`], or [`GroupError::InvalidDescriptor`]
    /// when `slot` is not a valid descriptor.
    pub fn read_view<'a>(&'a self, slot: DescriptorSlot) -> Result<TensorRead<'a>, GroupError> {
        self.tensor_read(slot)
    }

    fn tensor_read<'a>(&'a self, slot: DescriptorSlot) -> Result<TensorRead<'a>, GroupError> {
        let dtype = self.resolve_descriptor(slot)?.1.dtype();
        let view = match dtype {
            DType::F32 => tensor_view_from_group(self.view::<f32, DynRank>(slot)?),
            DType::F64 => tensor_view_from_group(self.view::<f64, DynRank>(slot)?),
            DType::I32 => tensor_view_from_group(self.view::<i32, DynRank>(slot)?),
            DType::I64 => tensor_view_from_group(self.view::<i64, DynRank>(slot)?),
            DType::Bool => tensor_view_from_group(self.view::<bool, DynRank>(slot)?),
            DType::C32 => {
                tensor_view_from_group(self.view::<num_complex::Complex32, DynRank>(slot)?)
            }
            DType::C64 => {
                tensor_view_from_group(self.view::<num_complex::Complex64, DynRank>(slot)?)
            }
        }
        .map_err(|error| GroupError::InvalidDescriptor {
            message: error.to_string(),
        })?;
        Ok(TensorRead::from_view(view))
    }

    /// Append a tensor owner and return its new descriptor slot without copying.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError::IndexOverflow`] when allocation or descriptor
    /// indices cannot be represented, or [`GroupError::InvalidDescriptor`]
    /// when the consumed tensor descriptor is invalid.
    pub fn append_tensor(&mut self, tensor: crate::Tensor) -> Result<DescriptorSlot, GroupError> {
        let (source, source_slot) = tensor.into_group_parts();
        self.append_group(source, source_slot)
    }

    /// Append one descriptor and all of its physical owners without copying.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError::IndexOverflow`] when group indices overflow or
    /// [`GroupError::InvalidDescriptor`] when `source_slot` is invalid.
    pub fn append_group(
        &mut self,
        mut source: AllocationGroup,
        source_slot: DescriptorSlot,
    ) -> Result<DescriptorSlot, GroupError> {
        let allocation_offset =
            u32::try_from(self.allocations.len()).map_err(|_| GroupError::IndexOverflow)?;
        let descriptor_offset =
            u32::try_from(self.descriptors.len()).map_err(|_| GroupError::IndexOverflow)?;
        source.resolve_descriptor(source_slot)?;

        for owner in source.allocations.drain(..) {
            self.allocations.push(owner);
        }
        for descriptor in source.descriptors.drain(..) {
            let descriptor = match descriptor {
                Some(mut descriptor) => {
                    let allocation = descriptor
                        .allocation
                        .0
                        .checked_add(allocation_offset)
                        .ok_or(GroupError::IndexOverflow)?;
                    descriptor.allocation = AllocationSlot(allocation);
                    Some(descriptor)
                }
                None => None,
            };
            self.descriptors.push(descriptor);
        }

        let source_descriptor_index =
            u32::try_from(source_slot.index()).map_err(|_| GroupError::IndexOverflow)?;
        Ok(DescriptorSlot(
            source_descriptor_index
                .checked_add(descriptor_offset)
                .ok_or(GroupError::IndexOverflow)?,
        ))
    }

    pub(crate) fn set_descriptor_placement(
        &mut self,
        slot: DescriptorSlot,
        placement: Placement,
    ) -> Result<(), GroupError> {
        let descriptor = self
            .descriptors
            .get_mut(slot.index())
            .ok_or(GroupError::DescriptorSlotOutOfBounds { slot: slot.index() })?
            .as_mut()
            .ok_or(GroupError::DescriptorSlotVacant { slot: slot.index() })?;
        descriptor.placement = placement;
        Ok(())
    }

    pub(crate) fn from_host_vec<T: TensorScalar, R: TensorRank>(
        shape: R::Shape,
        data: Vec<T>,
    ) -> Result<(Self, DescriptorSlot), GroupError> {
        let owner =
            super::root::import_host_vec(data).map_err(|error| GroupError::InvalidDescriptor {
                message: error.to_string(),
            })?;
        let span = owner.root_span();
        let mut group = Self::new();
        let allocation = group.insert_owner(owner)?;
        let layout =
            TensorLayout::<R>::compact(shape).map_err(|error| GroupError::InvalidDescriptor {
                message: error.to_string(),
            })?;
        let input = DescriptorInput::new(
            ByteRange::new(0, span.byte_len()),
            R::shape_from_vec(layout.shape().iter().copied().collect()).map_err(|error| {
                GroupError::InvalidDescriptor {
                    message: error.to_string(),
                }
            })?,
            R::strides_from_vec(layout.strides().iter().copied().collect()).map_err(|error| {
                GroupError::InvalidDescriptor {
                    message: error.to_string(),
                }
            })?,
            layout.offset(),
            true,
        );
        let slot = group.insert_descriptor::<T, R>(allocation, input)?;
        Ok((group, slot))
    }

    /// Build one compact descriptor by consuming a scalar-independent provider root.
    #[doc(hidden)]
    pub fn from_backend_allocation<T: TensorScalar, R: TensorRank>(
        shape: R::Shape,
        allocation: Box<dyn BackendAllocation>,
    ) -> Result<(Self, DescriptorSlot), GroupError> {
        let owner = super::root::import_unique_root(allocation).map_err(|error| {
            GroupError::InvalidDescriptor {
                message: error.to_string(),
            }
        })?;
        let span = owner.root_span();
        let mut group = Self::new();
        let allocation = group.insert_owner(owner)?;
        let layout =
            TensorLayout::<R>::compact(shape).map_err(|error| GroupError::InvalidDescriptor {
                message: error.to_string(),
            })?;
        let input = DescriptorInput::new(
            ByteRange::new(0, span.byte_len()),
            R::shape_from_vec(layout.shape().iter().copied().collect()).map_err(|error| {
                GroupError::InvalidDescriptor {
                    message: error.to_string(),
                }
            })?,
            R::strides_from_vec(layout.strides().iter().copied().collect()).map_err(|error| {
                GroupError::InvalidDescriptor {
                    message: error.to_string(),
                }
            })?,
            layout.offset(),
            true,
        );
        let slot = group.insert_descriptor::<T, R>(allocation, input)?;
        Ok((group, slot))
    }

    pub(crate) fn from_backend_buffer<T: TensorScalar, R: TensorRank>(
        shape: R::Shape,
        buffer: crate::StorageBuffer<T>,
    ) -> Result<(Self, DescriptorSlot), GroupError> {
        let owner = super::root::import_backend_buffer(buffer).map_err(|error| {
            GroupError::InvalidDescriptor {
                message: error.to_string(),
            }
        })?;
        let span = owner.root_span();
        let mut group = Self::new();
        let allocation = group.insert_owner(owner)?;
        let layout =
            TensorLayout::<R>::compact(shape).map_err(|error| GroupError::InvalidDescriptor {
                message: error.to_string(),
            })?;
        let input = DescriptorInput::new(
            ByteRange::new(0, span.byte_len()),
            R::shape_from_vec(layout.shape().iter().copied().collect()).map_err(|error| {
                GroupError::InvalidDescriptor {
                    message: error.to_string(),
                }
            })?,
            R::strides_from_vec(layout.strides().iter().copied().collect()).map_err(|error| {
                GroupError::InvalidDescriptor {
                    message: error.to_string(),
                }
            })?,
            layout.offset(),
            true,
        );
        let slot = group.insert_descriptor::<T, R>(allocation, input)?;
        Ok((group, slot))
    }

    pub(crate) fn from_backend_root<T: Send + Sync + 'static>(
        buffer: crate::StorageBuffer<T>,
    ) -> Result<Self, GroupError> {
        let owner = super::root::import_backend_buffer(buffer).map_err(|error| {
            GroupError::InvalidDescriptor {
                message: error.to_string(),
            }
        })?;
        let mut group = Self::new();
        group.insert_owner(owner)?;
        Ok(group)
    }

    pub(crate) fn insert_owner(
        &mut self,
        owner: OwnedStorage,
    ) -> Result<AllocationSlot, GroupError> {
        let index = self.allocations.len();
        let slot = u32::try_from(index).map_err(|_| GroupError::IndexOverflow)?;
        self.allocations.push(Some(owner));
        Ok(AllocationSlot(slot))
    }

    pub(crate) fn insert_descriptor<T: TensorScalar, R: TensorRank>(
        &mut self,
        allocation: AllocationSlot,
        input: DescriptorInput<R>,
    ) -> Result<DescriptorSlot, GroupError> {
        let allocation_index = allocation.index();
        let owner = self
            .allocations
            .get(allocation_index)
            .ok_or(GroupError::AllocationSlotOutOfBounds {
                slot: allocation_index,
            })?
            .as_ref()
            .ok_or(GroupError::AllocationSlotVacant {
                slot: allocation_index,
            })?;

        let root = owner.as_ref().root_identity();
        let span = root.bind_relative_range(input.relative).map_err(|error| {
            GroupError::InvalidDescriptor {
                message: error.to_string(),
            }
        })?;
        let element_size = size_of::<T>();
        if element_size == 0 || !span.byte_len().is_multiple_of(element_size) {
            return Err(GroupError::InvalidDescriptor {
                message: format!(
                    "byte span {} is not divisible by element size {}",
                    span.byte_len(),
                    element_size
                ),
            });
        }
        if !span
            .guaranteed_alignment()
            .get()
            .is_multiple_of(align_of::<T>())
        {
            return Err(GroupError::InvalidDescriptor {
                message: format!(
                    "span alignment {} is insufficient for {}-byte alignment",
                    span.guaranteed_alignment().get(),
                    align_of::<T>()
                ),
            });
        }

        let shape = R::shape_into_vec(input.shape);
        let strides = R::strides_into_vec(input.strides);
        let layout = TensorLayout::<DynRank>::from_parts(
            shape,
            strides,
            input.offset,
            span.byte_len() / element_size,
        )
        .map_err(|error| GroupError::InvalidDescriptor {
            message: error.to_string(),
        })?;
        let element_count = logical_element_count(layout.shape())?;
        let write_injective = if input.require_injective {
            layout.validate_mutable_no_overlap().map_err(|error| {
                GroupError::InvalidDescriptor {
                    message: error.to_string(),
                }
            })?;
            true
        } else {
            false
        };
        let envelope = reachable_envelope(&span, &layout, element_size)?;
        let (checked, _) = validate_descriptor::<T, DynRank>(
            &root,
            span,
            layout.shape().iter().copied().collect(),
            layout.strides().iter().copied().collect(),
            layout.offset(),
            false,
        )
        .map_err(|error| GroupError::InvalidDescriptor {
            message: error.to_string(),
        })?;
        let record = DescriptorRecord {
            allocation,
            root,
            span,
            layout,
            dtype: T::dtype(),
            element_size,
            element_count,
            provider: owner.as_ref().provider_kind(),
            placement: Placement::default(),
            envelope,
            write_injective,
            checked,
        };

        let descriptor_index = self.descriptors.len();
        let slot = u32::try_from(descriptor_index).map_err(|_| GroupError::IndexOverflow)?;
        self.descriptors.push(Some(record));
        Ok(DescriptorSlot(slot))
    }

    /// Replace one uniquely-owned descriptor's logical layout while retaining
    /// its scalar representation and allocation root.
    // INVARIANT: the dynamic dtype dispatch below selects the sealed scalar
    // pair before the existing descriptor validator runs.
    #[allow(clippy::result_large_err)]
    pub(crate) fn update_descriptor_layout(
        mut self,
        slot: DescriptorSlot,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> Result<Self, (Self, GroupError)> {
        let dtype = match self.resolve_descriptor(slot) {
            Ok((_, descriptor)) => descriptor.dtype,
            Err(error) => return Err((self, error)),
        };
        if let Some(Some(descriptor)) = self.descriptors.get_mut(slot.index()) {
            // A metadata-only read view may be non-injective (for example a
            // broadcast). Mutable access revalidates injectivity when asked.
            descriptor.write_injective = false;
        }
        match dtype {
            DType::F32 => self.reinterpret_descriptor::<f32, f32>(slot, shape, strides, offset),
            DType::F64 => self.reinterpret_descriptor::<f64, f64>(slot, shape, strides, offset),
            DType::I32 => self.reinterpret_descriptor::<i32, i32>(slot, shape, strides, offset),
            DType::I64 => self.reinterpret_descriptor::<i64, i64>(slot, shape, strides, offset),
            DType::Bool => self.reinterpret_descriptor::<bool, bool>(slot, shape, strides, offset),
            DType::C32 => self
                .reinterpret_descriptor::<num_complex::Complex32, num_complex::Complex32>(
                    slot, shape, strides, offset,
                ),
            DType::C64 => self
                .reinterpret_descriptor::<num_complex::Complex64, num_complex::Complex64>(
                    slot, shape, strides, offset,
                ),
        }
    }

    /// Replace one uniquely-owned descriptor with a sealed representation
    /// reinterpretation while retaining the same allocation root.
    ///
    /// The group is returned unchanged with the typed error when validation
    /// fails, so consuming owner callers can recover the original tensor.
    // INVARIANT: returning the unchanged move-only group with validation
    // failure is required so consuming callers can recover its owner.
    #[allow(clippy::result_large_err)]
    pub(crate) fn reinterpret_descriptor<T: TensorScalar, U: TensorScalar>(
        mut self,
        slot: DescriptorSlot,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> Result<Self, (Self, GroupError)> {
        let (descriptor_index, descriptor) = match self.resolve_descriptor(slot) {
            Ok((index, descriptor)) => (index, descriptor.clone()),
            Err(error) => return Err((self, error)),
        };
        if descriptor.dtype != T::dtype() {
            return Err((
                self,
                GroupError::DTypeMismatch {
                    expected: descriptor.dtype,
                    actual: T::dtype(),
                },
            ));
        }
        let allocation = descriptor.allocation;
        let references = self
            .descriptors
            .iter()
            .flatten()
            .filter(|candidate| candidate.allocation == allocation)
            .count();
        if references != 1 {
            return Err((
                self,
                GroupError::AliasedAllocation {
                    allocation: allocation.index(),
                },
            ));
        }

        let root = descriptor.root;
        let span = descriptor.span;
        let target_element_size = std::mem::size_of::<U>();
        if target_element_size == 0 || !span.byte_len().is_multiple_of(target_element_size) {
            return Err((
                self,
                GroupError::InvalidDescriptor {
                    message: format!(
                        "byte span {} is not divisible by target element size {}",
                        span.byte_len(),
                        target_element_size
                    ),
                },
            ));
        }
        let layout = match TensorLayout::<DynRank>::from_parts(
            shape.into(),
            strides.into(),
            offset,
            span.byte_len() / target_element_size,
        ) {
            Ok(layout) => layout,
            Err(error) => {
                return Err((
                    self,
                    GroupError::InvalidDescriptor {
                        message: error.to_string(),
                    },
                ))
            }
        };
        let write_injective = descriptor.write_injective;
        let envelope = match reachable_envelope(&span, &layout, target_element_size) {
            Ok(envelope) => envelope,
            Err(error) => return Err((self, error)),
        };
        let (checked, _) = match validate_descriptor::<U, DynRank>(
            &root,
            span,
            layout.shape().iter().copied().collect(),
            layout.strides().iter().copied().collect(),
            layout.offset(),
            write_injective,
        ) {
            Ok(value) => value,
            Err(error) => {
                return Err((
                    self,
                    GroupError::InvalidDescriptor {
                        message: error.to_string(),
                    },
                ))
            }
        };
        let element_count = match logical_element_count(layout.shape()) {
            Ok(count) => count,
            Err(error) => return Err((self, error)),
        };
        self.descriptors[descriptor_index] = Some(DescriptorRecord {
            allocation,
            root,
            span,
            layout,
            dtype: U::dtype(),
            element_size: target_element_size,
            element_count,
            provider: descriptor.provider,
            placement: descriptor.placement.clone(),
            envelope,
            write_injective,
            checked,
        });
        Ok(self)
    }

    pub(crate) fn view<T: TensorScalar, R: TensorRank>(
        &self,
        slot: DescriptorSlot,
    ) -> Result<GroupReadView<'_, T, R>, GroupError> {
        let (descriptor_index, descriptor) = self.resolve_descriptor(slot)?;
        check_typed::<T, R>(descriptor)?;
        let owner = self
            .allocations
            .get(descriptor.allocation.index())
            .ok_or(GroupError::AllocationSlotOutOfBounds {
                slot: descriptor.allocation.index(),
            })?
            .as_ref()
            .ok_or(GroupError::AllocationSlotVacant {
                slot: descriptor.allocation.index(),
            })?;
        let _ = descriptor_index;
        Ok(GroupReadView {
            owner: NonNull::from(owner),
            descriptor: descriptor.clone(),
            _borrow: PhantomData,
        })
    }

    pub(crate) fn view_raw<T: 'static, R: TensorRank>(
        &self,
        slot: DescriptorSlot,
    ) -> Result<GroupReadView<'_, T, R>, GroupError> {
        let (_, descriptor) = self.resolve_descriptor(slot)?;
        let owner = self
            .allocations
            .get(descriptor.allocation.index())
            .ok_or(GroupError::AllocationSlotOutOfBounds {
                slot: descriptor.allocation.index(),
            })?
            .as_ref()
            .ok_or(GroupError::AllocationSlotVacant {
                slot: descriptor.allocation.index(),
            })?;
        Ok(GroupReadView {
            owner: NonNull::from(owner),
            descriptor: descriptor.clone(),
            _borrow: PhantomData,
        })
    }

    pub(crate) fn prepare_device_read_for_layout<T: TensorScalar, R: TensorRank>(
        &self,
        slot: DescriptorSlot,
        layout: &TensorLayout<R>,
    ) -> Result<Box<dyn crate::PreparedDeviceAccess + '_>, AccessError> {
        self.view_raw::<T, R>(slot)
            .map_err(|error| AccessError::InvalidLayout {
                message: error.to_string(),
            })?
            .prepare_device_read_for_layout(layout)
    }

    pub(crate) fn allocation_index(&self, slot: DescriptorSlot) -> Result<usize, GroupError> {
        Ok(self.resolve_descriptor(slot)?.1.allocation.index())
    }

    pub(crate) fn host_buffer_at<T: 'static>(
        &self,
        allocation_index: usize,
    ) -> Option<&crate::StorageBuffer<T>> {
        self.allocations
            .get(allocation_index)?
            .as_ref()?
            .host_buffer::<T>()
    }

    pub(crate) fn host_root_metadata<T: 'static>(
        &self,
        slot: DescriptorSlot,
    ) -> Option<(usize, usize)> {
        let (_, descriptor) = self.resolve_descriptor(slot).ok()?;
        let owner = self
            .allocations
            .get(descriptor.allocation.index())?
            .as_ref()?;
        if descriptor.span != owner.root_span() {
            return None;
        }
        let crate::StorageBuffer::Host(data) = owner.host_buffer::<T>()? else {
            return None;
        };
        let pointer = data.as_ptr() as usize;
        let byte_len = data.len().checked_mul(size_of::<T>())?;
        Some((pointer, byte_len))
    }

    pub(crate) fn backend_buffer<T: 'static>(
        &self,
        slot: DescriptorSlot,
    ) -> Option<&crate::StorageBuffer<T>> {
        let (_, descriptor) = self.resolve_descriptor(slot).ok()?;
        self.allocations
            .get(descriptor.allocation.index())?
            .as_ref()?
            .backend_buffer::<T>()
    }

    pub(crate) fn descriptor_len(&self, slot: DescriptorSlot) -> Option<usize> {
        self.resolve_descriptor(slot)
            .ok()
            .map(|(_, descriptor)| descriptor.element_count)
    }

    pub(crate) fn backend_identity(
        &self,
        slot: DescriptorSlot,
    ) -> Option<(crate::AllocationDomainId, crate::AllocationId)> {
        let (_, descriptor) = self.resolve_descriptor(slot).ok()?;
        if descriptor.provider == ProviderKind::Cpu {
            return None;
        }
        let owner = self
            .allocations
            .get(descriptor.allocation.index())?
            .as_ref()?;
        let key = owner.root_identity().extent().key();
        Some((key.domain(), key.local()))
    }

    pub(crate) fn provider_kind(&self, slot: DescriptorSlot) -> Option<ProviderKind> {
        self.resolve_descriptor(slot)
            .ok()
            .map(|(_, descriptor)| descriptor.provider)
    }

    pub(crate) fn backend_root_buffer<T: 'static>(&self) -> Option<&crate::StorageBuffer<T>> {
        self.allocations
            .first()
            .and_then(|owner| owner.as_ref())
            .and_then(|owner| owner.backend_buffer::<T>())
    }

    pub(crate) fn backend_buffer_mut<T: 'static>(
        &mut self,
        slot: DescriptorSlot,
    ) -> Option<&mut crate::StorageBuffer<T>> {
        let allocation = self.resolve_descriptor(slot).ok()?.1.allocation;
        self.allocations
            .get_mut(allocation.index())?
            .as_mut()?
            .backend_buffer_mut::<T>()
    }

    pub(crate) fn backend_root_buffer_mut<T: 'static>(
        &mut self,
    ) -> Option<&mut crate::StorageBuffer<T>> {
        self.allocations
            .first_mut()
            .and_then(|owner| owner.as_mut())
            .and_then(|owner| owner.backend_buffer_mut::<T>())
    }

    pub(crate) fn view_mut_raw<T: 'static, R: TensorRank>(
        &mut self,
        slot: DescriptorSlot,
    ) -> Result<GroupWriteView<'_, T, R>, GroupError> {
        let (_, descriptor) = self.resolve_descriptor(slot)?;
        let descriptor = descriptor.clone();
        let owner = self
            .allocations
            .get_mut(descriptor.allocation.index())
            .ok_or(GroupError::AllocationSlotOutOfBounds {
                slot: descriptor.allocation.index(),
            })?
            .as_mut()
            .ok_or(GroupError::AllocationSlotVacant {
                slot: descriptor.allocation.index(),
            })?;
        Ok(GroupWriteView {
            owner: NonNull::from(&mut *owner),
            descriptor: descriptor.clone(),
            _borrow: PhantomData,
        })
    }

    pub(crate) fn prepare_device_write_for_layout<T: TensorScalar, R: TensorRank>(
        &mut self,
        slot: DescriptorSlot,
        layout: &TensorLayout<R>,
    ) -> Result<Box<dyn crate::PreparedDeviceAccess + '_>, AccessError> {
        self.view_mut_raw::<T, R>(slot)
            .map_err(|error| AccessError::InvalidLayout {
                message: error.to_string(),
            })?
            .prepare_device_write_for_layout(layout)
    }

    pub(crate) fn view_mut<T: TensorScalar, R: TensorRank>(
        &mut self,
        slot: DescriptorSlot,
    ) -> Result<GroupWriteView<'_, T, R>, GroupError> {
        let (descriptor_index, descriptor) = self.resolve_descriptor(slot)?;
        let mut descriptor = descriptor.clone();
        check_typed::<T, R>(&descriptor)?;
        if !descriptor.write_injective {
            descriptor
                .layout
                .validate_mutable_no_overlap()
                .map_err(|error| GroupError::InvalidDescriptor {
                    message: error.to_string(),
                })?;
            if let Some(Some(retained)) = self.descriptors.get_mut(descriptor_index) {
                retained.write_injective = true;
            }
            descriptor.write_injective = true;
        }
        let owner = self
            .allocations
            .get_mut(descriptor.allocation.index())
            .ok_or(GroupError::AllocationSlotOutOfBounds {
                slot: descriptor.allocation.index(),
            })?
            .as_mut()
            .ok_or(GroupError::AllocationSlotVacant {
                slot: descriptor.allocation.index(),
            })?;
        Ok(GroupWriteView {
            owner: NonNull::from(&mut *owner),
            descriptor,
            _borrow: PhantomData,
        })
    }

    pub(crate) fn split_mut<T: TensorScalar, R: TensorRank>(
        &mut self,
        slots: &[DescriptorSlot],
    ) -> Result<Vec<GroupWriteView<'_, T, R>>, DisjointViewError> {
        let mut selected = Vec::with_capacity(slots.len());
        for &slot in slots {
            let (descriptor_index, descriptor) = self.resolve_descriptor(slot)?;
            if selected
                .iter()
                .any(|(seen, _, _): &(DescriptorSlot, usize, DescriptorRecord)| *seen == slot)
            {
                return Err(DisjointViewError::DuplicateSlot { slot: slot.index() });
            }
            check_typed::<T, R>(descriptor)?;
            if !descriptor.write_injective {
                descriptor
                    .layout
                    .validate_mutable_no_overlap()
                    .map_err(|_| DisjointViewError::NonInjective { slot: slot.index() })?;
            }
            selected.push((slot, descriptor_index, descriptor.clone()));
        }

        for left in 0..selected.len() {
            for right in (left + 1)..selected.len() {
                let first = &selected[left].2;
                let second = &selected[right].2;
                if first.root.root_resource() != second.root.root_resource() {
                    continue;
                }
                match (first.envelope, second.envelope) {
                    (None, _) | (_, None) => {}
                    (Some(first), Some(second)) => {
                        if first
                            .overlaps(second)
                            .map_err(|_| DisjointViewError::NotProvablyDisjoint)?
                        {
                            return Err(DisjointViewError::PairwiseOverlap);
                        }
                    }
                }
            }
        }

        for (_, descriptor_index, _) in &selected {
            if let Some(Some(descriptor)) = self.descriptors.get_mut(*descriptor_index) {
                descriptor.write_injective = true;
            }
        }

        let mut children = Vec::with_capacity(selected.len());
        for (_, _, mut descriptor) in selected {
            descriptor.write_injective = true;
            let owner = self
                .allocations
                .get_mut(descriptor.allocation.index())
                .ok_or(GroupError::AllocationSlotOutOfBounds {
                    slot: descriptor.allocation.index(),
                })?
                .as_mut()
                .ok_or(GroupError::AllocationSlotVacant {
                    slot: descriptor.allocation.index(),
                })?;
            children.push(GroupWriteView {
                owner: NonNull::from(&mut *owner),
                descriptor,
                _borrow: PhantomData,
            });
        }
        Ok(children)
    }

    pub(crate) fn try_extract(
        &mut self,
        slot: DescriptorSlot,
    ) -> Result<OwnedStorage, ExtractError> {
        let (descriptor_index, descriptor) = self.resolve_descriptor(slot)?;
        let allocation = descriptor.allocation;
        let references = self
            .descriptors
            .iter()
            .flatten()
            .filter(|candidate| candidate.allocation == allocation)
            .count();
        if references != 1 {
            return Err(ExtractError::AliasedAllocation {
                allocation: allocation.index(),
            });
        }
        let _ = self.descriptors[descriptor_index].take();
        self.allocations
            .get_mut(allocation.index())
            .ok_or(GroupError::AllocationSlotOutOfBounds {
                slot: allocation.index(),
            })?
            .take()
            .ok_or(GroupError::AllocationSlotVacant {
                slot: allocation.index(),
            })
            .map_err(ExtractError::from)
    }

    /// Extract one uniquely-owned descriptor while retaining all other group
    /// descriptors in place.
    ///
    /// This is structural: aliased allocations return a typed error and the
    /// group remains unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError::AliasedAllocation`] for an aliased allocation or
    /// [`GroupError::InvalidDescriptor`] for an invalid slot; every extraction
    /// failure leaves the group unchanged.
    #[allow(clippy::result_large_err)]
    pub fn take_tensor(&mut self, slot: DescriptorSlot) -> Result<crate::Tensor, GroupError> {
        let (_, descriptor) = self.resolve_descriptor(slot)?;
        let descriptor = descriptor.clone();
        let dtype = descriptor.dtype;
        let layout = descriptor.layout.clone();
        let placement = descriptor.placement.clone();
        let owner = self
            .try_extract(slot)
            .map_err(|error| GroupError::InvalidDescriptor {
                message: error.to_string(),
            })?;
        let mut extracted = Self::new();
        extracted.allocations.push(Some(owner));
        let mut descriptor = descriptor.clone();
        descriptor.allocation = AllocationSlot(0);
        extracted.descriptors.push(Some(descriptor));
        Ok(tensor_from_group(
            extracted,
            DescriptorSlot(0),
            0,
            dtype,
            layout,
            placement,
        ))
    }

    // INVARIANT: extraction failure returns the unchanged group because the
    // caller must retain ownership when a descriptor cannot be detached.
    #[allow(clippy::result_large_err)]
    pub(crate) fn into_owner(
        mut self,
        slot: DescriptorSlot,
    ) -> Result<OwnedStorage, (Self, ExtractError)> {
        let result = self.try_extract(slot);
        match result {
            Ok(owner) => Ok(owner),
            Err(error) => Err((self, error)),
        }
    }

    // INVARIANT: returning the unchanged move-only group is the extraction
    // failure carrier required by the ownership contract.
    /// Consume the group and extract one descriptor as a standalone tensor.
    ///
    /// Extraction is structural: it succeeds only when no other descriptor
    /// aliases the selected physical allocation. Failure returns the exact
    /// unchanged group and typed error without copying.
    ///
    /// # Errors
    ///
    /// Returns [`GroupError::AliasedAllocation`] when another descriptor
    /// references the allocation, or [`GroupError::InvalidDescriptor`] for an
    /// invalid slot. Each extraction failure returns the unchanged group.
    #[allow(clippy::result_large_err)]
    pub fn into_tensor(self, slot: DescriptorSlot) -> Result<crate::Tensor, (Self, GroupError)> {
        let (_, descriptor) = match self.resolve_descriptor(slot) {
            Ok(value) => value,
            Err(error) => return Err((self, error)),
        };
        let allocation = descriptor.allocation;
        let references = self
            .descriptors
            .iter()
            .flatten()
            .filter(|candidate| candidate.allocation == allocation)
            .count();
        if references != 1 {
            return Err((
                self,
                GroupError::AliasedAllocation {
                    allocation: allocation.index(),
                },
            ));
        }
        let dtype = descriptor.dtype;
        let layout = descriptor.layout.clone();
        let placement = descriptor.placement.clone();
        let (group, slot) = match self.into_single_descriptor(slot) {
            Ok(value) => value,
            Err((group, error)) => return Err((group, error)),
        };
        Ok(tensor_from_group(group, slot, 0, dtype, layout, placement))
    }

    // INVARIANT: structural extraction must return the unchanged group on
    // every validation failure so no owner is lost.
    #[allow(clippy::result_large_err)]
    fn into_single_descriptor(
        mut self,
        slot: DescriptorSlot,
    ) -> Result<(Self, DescriptorSlot), (Self, GroupError)> {
        let (descriptor_index, descriptor) = match self.resolve_descriptor(slot) {
            Ok(value) => value,
            Err(error) => return Err((self, error)),
        };
        let allocation = descriptor.allocation;
        let mut descriptor = match self.descriptors[descriptor_index].take() {
            Some(descriptor) => descriptor,
            None => {
                return Err((
                    self,
                    GroupError::DescriptorSlotVacant { slot: slot.index() },
                ))
            }
        };
        let owner = match self.allocations.get_mut(allocation.index()) {
            Some(owner) => match owner.take() {
                Some(owner) => owner,
                None => {
                    self.descriptors[descriptor_index] = Some(descriptor);
                    return Err((
                        self,
                        GroupError::AllocationSlotVacant {
                            slot: allocation.index(),
                        },
                    ));
                }
            },
            None => {
                self.descriptors[descriptor_index] = Some(descriptor);
                return Err((
                    self,
                    GroupError::AllocationSlotOutOfBounds {
                        slot: allocation.index(),
                    },
                ));
            }
        };
        let mut group = Self::new();
        descriptor.allocation = AllocationSlot(0);
        group.allocations.push(Some(owner));
        group.descriptors.push(Some(descriptor));
        Ok((group, DescriptorSlot(0)))
    }

    pub(crate) fn into_host_vec<T: TensorScalar>(
        self,
        slot: DescriptorSlot,
    ) -> Result<Vec<T>, String> {
        let owner = self
            .into_owner(slot)
            .map_err(|(_, error)| error.to_string())?;
        owner
            .into_host_vec::<T>()
            .map_err(|error| error.to_string())
    }

    fn resolve_descriptor(
        &self,
        slot: DescriptorSlot,
    ) -> Result<(usize, &DescriptorRecord), GroupError> {
        let index = slot.index();
        let descriptor = self
            .descriptors
            .get(index)
            .ok_or(GroupError::DescriptorSlotOutOfBounds { slot: index })?
            .as_ref()
            .ok_or(GroupError::DescriptorSlotVacant { slot: index })?;
        Ok((index, descriptor))
    }

    #[cfg(test)]
    pub(crate) fn test_vacate_allocation(&mut self, slot: AllocationSlot) {
        if let Some(entry) = self.allocations.get_mut(slot.index()) {
            *entry = None;
        }
    }
}

fn check_typed<T: TensorScalar, R: TensorRank>(
    descriptor: &DescriptorRecord,
) -> Result<(), GroupError> {
    if descriptor.dtype != T::dtype() {
        return Err(GroupError::DTypeMismatch {
            expected: descriptor.dtype,
            actual: T::dtype(),
        });
    }
    if let Some(expected) = R::RANK {
        let actual = descriptor.layout.shape().len();
        if expected != actual {
            return Err(GroupError::RankMismatch { expected, actual });
        }
    }
    Ok(())
}

fn logical_element_count(shape: &[usize]) -> Result<usize, GroupError> {
    shape.iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| GroupError::InvalidDescriptor {
                message: "logical element count overflows".to_owned(),
            })
    })
}

fn reachable_envelope(
    span: &RootBoundSpan,
    layout: &TensorLayout<DynRank>,
    element_size: usize,
) -> Result<Option<ByteRange>, GroupError> {
    if layout.shape().contains(&0) {
        return Ok(None);
    }
    let mut minimum = layout.offset() as i128;
    let mut maximum = minimum;
    for (&extent, &stride) in layout.shape().iter().zip(layout.strides()) {
        let steps = i128::try_from(extent - 1).map_err(|_| GroupError::InvalidDescriptor {
            message: "layout extent does not fit i128".to_owned(),
        })?;
        let contribution =
            (stride as i128)
                .checked_mul(steps)
                .ok_or_else(|| GroupError::InvalidDescriptor {
                    message: "reachable layout arithmetic overflows".to_owned(),
                })?;
        if contribution < 0 {
            minimum =
                minimum
                    .checked_add(contribution)
                    .ok_or_else(|| GroupError::InvalidDescriptor {
                        message: "reachable layout minimum overflows".to_owned(),
                    })?;
        } else {
            maximum =
                maximum
                    .checked_add(contribution)
                    .ok_or_else(|| GroupError::InvalidDescriptor {
                        message: "reachable layout maximum overflows".to_owned(),
                    })?;
        }
    }
    let minimum = usize::try_from(minimum).map_err(|_| GroupError::InvalidDescriptor {
        message: "reachable layout minimum is negative".to_owned(),
    })?;
    let maximum = usize::try_from(maximum).map_err(|_| GroupError::InvalidDescriptor {
        message: "reachable layout maximum is negative or too large".to_owned(),
    })?;
    let byte_offset = span
        .byte_offset()
        .checked_add(minimum.checked_mul(element_size).ok_or_else(|| {
            GroupError::InvalidDescriptor {
                message: "reachable byte offset overflows".to_owned(),
            }
        })?)
        .ok_or_else(|| GroupError::InvalidDescriptor {
            message: "reachable byte offset overflows".to_owned(),
        })?;
    let byte_len = maximum
        .checked_sub(minimum)
        .and_then(|length| length.checked_add(1))
        .and_then(|length| length.checked_mul(element_size))
        .ok_or_else(|| GroupError::InvalidDescriptor {
            message: "reachable byte length overflows".to_owned(),
        })?;
    let range = ByteRange::new(byte_offset, byte_len);
    range
        .checked_end()
        .map_err(|error| GroupError::InvalidDescriptor {
            message: error.to_string(),
        })?;
    Ok(Some(range))
}

#[cfg(test)]
pub(crate) fn test_logical_element_count(shape: &[usize]) -> Result<usize, GroupError> {
    logical_element_count(shape)
}

#[cfg(test)]
pub(crate) fn test_reachable_envelope(
    span: RootBoundSpan,
    layout: TensorLayout<DynRank>,
    element_size: usize,
) -> Result<Option<ByteRange>, GroupError> {
    reachable_envelope(&span, &layout, element_size)
}
