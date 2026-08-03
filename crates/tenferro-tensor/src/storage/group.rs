use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ptr::NonNull;

use crate::{DType, DynRank, TensorLayout, TensorRank, TensorScalar};

use super::prepared::{AccessError, ProviderReadMapping, ProviderWriteMapping};
use super::root::{OwnedStorage, ProviderKind};
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DescriptorSlot(u32);

impl DescriptorSlot {
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
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
    envelope: Option<ByteRange>,
    write_injective: bool,
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

    pub(crate) const fn envelope(&self) -> Option<ByteRange> {
        self.envelope
    }

    pub(crate) const fn write_injective(&self) -> bool {
        self.write_injective
    }
}

/// Group construction and slot errors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum GroupError {
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
#[derive(Default)]
pub(crate) struct AllocationGroup {
    allocations: Vec<Option<OwnedStorage>>,
    descriptors: Vec<Option<DescriptorRecord>>,
}

/// A shared descriptor child bounded by the group's shared borrow.
pub(crate) struct GroupReadView<'a, T: TensorScalar, R: TensorRank> {
    owner: NonNull<OwnedStorage>,
    descriptor: DescriptorRecord,
    _borrow: PhantomData<(&'a OwnedStorage, T, R)>,
}

impl<T: TensorScalar, R: TensorRank> std::fmt::Debug for GroupReadView<'_, T, R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroupReadView")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl<T: TensorScalar, R: TensorRank> GroupReadView<'_, T, R> {
    pub(crate) fn descriptor(&self) -> &DescriptorRecord {
        &self.descriptor
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
}

/// A non-cloneable mutable descriptor child bounded by the group's exclusive
/// borrow. The raw owner pointer is never exposed and is dereferenced only for
/// a provider mapping whose retained byte envelope was proven by the group.
pub(crate) struct GroupWriteView<'a, T: TensorScalar, R: TensorRank> {
    owner: NonNull<OwnedStorage>,
    descriptor: DescriptorRecord,
    _borrow: PhantomData<(&'a mut [u8], T, R)>,
}

impl<T: TensorScalar, R: TensorRank> std::fmt::Debug for GroupWriteView<'_, T, R> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GroupWriteView")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl<T: TensorScalar, R: TensorRank> GroupWriteView<'_, T, R> {
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
}

impl AllocationGroup {
    pub(crate) fn new() -> Self {
        Self::default()
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
        let record = DescriptorRecord {
            allocation,
            root,
            span,
            layout,
            dtype: T::dtype(),
            element_size,
            element_count,
            provider: owner.as_ref().provider_kind(),
            envelope,
            write_injective,
        };

        let descriptor_index = self.descriptors.len();
        let slot = u32::try_from(descriptor_index).map_err(|_| GroupError::IndexOverflow)?;
        self.descriptors.push(Some(record));
        Ok(DescriptorSlot(slot))
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
