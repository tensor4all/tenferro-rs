use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut, Range};
use std::ptr::NonNull;

use crate::{
    DType, DeviceAccessError, DeviceAccessRequest, PreparedDeviceAccess, TensorLayout, TensorRank,
    TensorScalar,
};

use super::identity::RootResourceIdentity;
use super::root::{StorageMut, StorageRef};
use super::span::RootBoundSpan;

/// Typed failure at the private prepared-access boundary.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum AccessError {
    #[error("invalid checked layout: {message}")]
    InvalidLayout { message: String },
    #[error("dtype mismatch: expected {expected:?}, actual {actual:?}")]
    DTypeMismatch { expected: DType, actual: DType },
    #[error("mapped byte length mismatch: expected {expected}, actual {actual}")]
    LengthMismatch { expected: usize, actual: usize },
    #[error("mapped bytes are not aligned for {required}-byte alignment")]
    Misaligned { required: usize },
    #[error("provider `{backend}` does not support the requested mapping")]
    Unsupported { backend: &'static str },
    #[error("provider mapping failed: {message}")]
    Provider { message: String },
    #[error("provider completion cannot be proven: {message}")]
    CompletionUnproven { message: String },
}

trait ReadMappingAccess {
    fn bytes(&self) -> &[u8];
}

trait WriteMappingAccess {
    fn len(&self) -> usize;
    fn bytes_mut(&mut self) -> &mut [u8];
}

struct BorrowedRead<'a>(&'a [u8]);

impl ReadMappingAccess for BorrowedRead<'_> {
    fn bytes(&self) -> &[u8] {
        self.0
    }
}

struct GuardRead<G>(G);

impl<G> ReadMappingAccess for GuardRead<G>
where
    G: Deref,
    G::Target: AsRef<[u8]>,
{
    fn bytes(&self) -> &[u8] {
        self.0.deref().as_ref()
    }
}

struct BorrowedWrite<'a>(&'a mut [u8]);

impl WriteMappingAccess for BorrowedWrite<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self.0
    }
}

struct GuardWrite<G>(G);

impl<G> WriteMappingAccess for GuardWrite<G>
where
    G: DerefMut,
    G::Target: AsMut<[u8]> + AsRef<[u8]>,
{
    fn len(&self) -> usize {
        self.0.deref().as_ref().len()
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self.0.deref_mut().as_mut()
    }
}

/// Borrowed provider mapping retained by a prepared read.
pub(crate) struct ProviderReadMapping<'a> {
    access: Box<dyn ReadMappingAccess + 'a>,
}

impl fmt::Debug for ProviderReadMapping<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderReadMapping")
            .field("byte_len", &self.bytes().len())
            .finish_non_exhaustive()
    }
}

impl<'a> ProviderReadMapping<'a> {
    pub(crate) fn from_slice(bytes: &'a [u8]) -> Self {
        Self {
            access: Box::new(BorrowedRead(bytes)),
        }
    }

    pub(crate) fn from_guard<G>(guard: G) -> Self
    where
        G: Deref + 'a,
        G::Target: AsRef<[u8]>,
    {
        Self {
            access: Box::new(GuardRead(guard)),
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        self.access.bytes()
    }
}

/// Borrowed provider mapping retained by a prepared write.
pub(crate) struct ProviderWriteMapping<'a> {
    access: Box<dyn WriteMappingAccess + 'a>,
}

impl fmt::Debug for ProviderWriteMapping<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderWriteMapping")
            .field("byte_len", &self.access.len())
            .finish_non_exhaustive()
    }
}

impl<'a> ProviderWriteMapping<'a> {
    pub(crate) fn from_slice(bytes: &'a mut [u8]) -> Self {
        Self {
            access: Box::new(BorrowedWrite(bytes)),
        }
    }

    pub(crate) fn from_guard<G>(guard: G) -> Self
    where
        G: DerefMut + 'a,
        G::Target: AsMut<[u8]> + AsRef<[u8]>,
    {
        Self {
            access: Box::new(GuardWrite(guard)),
        }
    }

    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        self.access.bytes_mut()
    }

    pub(crate) fn len(&self) -> usize {
        self.access.len()
    }
}

/// Host or device access selected by a prepared transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AccessTarget {
    Host,
    Device,
}

/// Proof that a descriptor's logical element addresses are injective.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WriteInjectivityProof;

/// Checked incremental layout retained by a prepared strided access.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckedStrided<R: TensorRank> {
    shape: R::Shape,
    strides: R::Strides,
    carry: Box<[isize]>,
    offset: isize,
    element_count: usize,
    _rank: PhantomData<R>,
}

impl<R: TensorRank> CheckedStrided<R> {
    pub(crate) fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }

    pub(crate) fn strides(&self) -> &[isize] {
        self.strides.as_ref()
    }

    pub(crate) fn carry(&self) -> &[isize] {
        &self.carry
    }

    pub(crate) const fn offset(&self) -> isize {
        self.offset
    }

    pub(crate) const fn element_count(&self) -> usize {
        self.element_count
    }
}

/// Layout state retained by a checked descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CheckedLayout<R: TensorRank> {
    Contiguous { element_range: Range<usize> },
    Strided(CheckedStrided<R>),
}

impl<R: TensorRank> CheckedLayout<R> {
    pub(crate) fn element_count(&self) -> usize {
        match self {
            Self::Contiguous { element_range } => element_range.len(),
            Self::Strided(strided) => strided.element_count(),
        }
    }
}

/// Descriptor facts checked before provider mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckedDescriptor<R: TensorRank> {
    span: RootBoundSpan,
    layout: CheckedLayout<R>,
    shape: R::Shape,
    strides: R::Strides,
    offset: isize,
    dtype: DType,
    element_size: usize,
}

impl<R: TensorRank> CheckedDescriptor<R> {
    pub(crate) const fn span(&self) -> RootBoundSpan {
        self.span
    }

    pub(crate) const fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) const fn element_size(&self) -> usize {
        self.element_size
    }

    pub(crate) fn layout(&self) -> &CheckedLayout<R> {
        &self.layout
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.shape.as_ref()
    }

    pub(crate) fn strides(&self) -> &[isize] {
        self.strides.as_ref()
    }

    pub(crate) const fn offset(&self) -> isize {
        self.offset
    }
}

/// A checked descriptor retaining the mutable injectivity proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckedInjectiveDescriptor<R: TensorRank> {
    descriptor: CheckedDescriptor<R>,
    proof: WriteInjectivityProof,
}

impl<R: TensorRank> CheckedInjectiveDescriptor<R> {
    pub(crate) fn descriptor(&self) -> &CheckedDescriptor<R> {
        &self.descriptor
    }

    pub(crate) const fn proof(&self) -> WriteInjectivityProof {
        self.proof
    }
}

/// Opaque checked shared capability paired with a descriptor.
pub(crate) struct CheckedRead<'a, R: TensorRank> {
    owner: StorageRef<'a>,
    descriptor: CheckedDescriptor<R>,
}

/// Opaque checked exclusive capability paired with an injective descriptor.
pub(crate) struct CheckedWrite<'a, R: TensorRank> {
    owner: StorageMut<'a>,
    descriptor: CheckedInjectiveDescriptor<R>,
}

impl<R: TensorRank> fmt::Debug for CheckedRead<'_, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckedRead")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl<R: TensorRank> fmt::Debug for CheckedWrite<'_, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CheckedWrite")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

fn invalid_layout(error: impl fmt::Display) -> AccessError {
    AccessError::InvalidLayout {
        message: error.to_string(),
    }
}

fn make_descriptor<T, R>(
    owner: &RootResourceIdentity,
    span: RootBoundSpan,
    shape: R::Shape,
    strides: R::Strides,
    offset: isize,
    require_injective: bool,
) -> Result<(CheckedDescriptor<R>, Option<WriteInjectivityProof>), AccessError>
where
    T: TensorScalar,
    R: TensorRank,
{
    owner.validate_bound_span(&span).map_err(invalid_layout)?;

    let element_size = size_of::<T>();
    let byte_len = span.byte_len();
    if element_size == 0 || !byte_len.is_multiple_of(element_size) {
        return Err(AccessError::LengthMismatch {
            expected: byte_len / element_size.max(1) * element_size.max(1),
            actual: byte_len,
        });
    }
    let required_alignment = align_of::<T>();
    if !span
        .guaranteed_alignment()
        .get()
        .is_multiple_of(required_alignment)
    {
        return Err(AccessError::Misaligned {
            required: required_alignment,
        });
    }

    let buffer_len = byte_len / element_size;
    let layout = TensorLayout::<R>::from_parts(shape.clone(), strides.clone(), offset, buffer_len)
        .map_err(invalid_layout)?;
    let element_count = layout.shape().iter().try_fold(1usize, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| invalid_layout("logical element count overflows"))
    })?;

    let proof = if require_injective {
        layout
            .validate_mutable_no_overlap()
            .map_err(invalid_layout)?;
        Some(WriteInjectivityProof)
    } else {
        None
    };

    let checked_layout = if layout.is_compact_col_major().map_err(invalid_layout)? {
        let start = usize::try_from(layout.offset()).map_err(invalid_layout)?;
        let end = start
            .checked_add(element_count)
            .ok_or_else(|| invalid_layout("contiguous element range overflows"))?;
        CheckedLayout::Contiguous {
            element_range: start..end,
        }
    } else {
        let carry = shape
            .as_ref()
            .iter()
            .zip(strides.as_ref().iter())
            .map(|(&extent, &stride)| {
                let extent = isize::try_from(extent)
                    .map_err(|_| invalid_layout("stride extent overflows isize"))?;
                let steps = extent
                    .checked_sub(1)
                    .ok_or_else(|| invalid_layout("stride extent underflows"))?;
                stride
                    .checked_mul(steps)
                    .and_then(isize::checked_neg)
                    .ok_or_else(|| invalid_layout("stride carry overflows"))
            })
            .collect::<Result<Box<[_]>, AccessError>>()?;
        CheckedLayout::Strided(CheckedStrided {
            shape: shape.clone(),
            strides: strides.clone(),
            carry,
            offset: layout.offset(),
            element_count,
            _rank: PhantomData,
        })
    };

    Ok((
        CheckedDescriptor {
            span,
            layout: checked_layout,
            shape: shape.clone(),
            strides: strides.clone(),
            offset,
            dtype: T::dtype(),
            element_size,
        },
        proof,
    ))
}

impl<'a, R: TensorRank> CheckedRead<'a, R> {
    pub(crate) fn from_validated(owner: StorageRef<'a>, descriptor: CheckedDescriptor<R>) -> Self {
        Self { owner, descriptor }
    }

    pub(crate) fn new<T: TensorScalar>(
        owner: StorageRef<'a>,
        span: RootBoundSpan,
        shape: R::Shape,
        strides: R::Strides,
        offset: isize,
    ) -> Result<Self, AccessError> {
        let root = owner.root_identity();
        let (descriptor, _) = make_descriptor::<T, R>(&root, span, shape, strides, offset, false)?;
        Ok(Self { owner, descriptor })
    }

    pub(crate) const fn descriptor(&self) -> &CheckedDescriptor<R> {
        &self.descriptor
    }
}

impl<'a, R: TensorRank> CheckedWrite<'a, R> {
    pub(crate) fn from_validated(
        owner: StorageMut<'a>,
        descriptor: CheckedDescriptor<R>,
        proof: WriteInjectivityProof,
    ) -> Self {
        Self {
            owner,
            descriptor: CheckedInjectiveDescriptor { descriptor, proof },
        }
    }

    pub(crate) fn new<T: TensorScalar>(
        owner: StorageMut<'a>,
        span: RootBoundSpan,
        shape: R::Shape,
        strides: R::Strides,
        offset: isize,
    ) -> Result<Self, AccessError> {
        let root = owner.root_identity();
        let (descriptor, proof) =
            make_descriptor::<T, R>(&root, span, shape, strides, offset, true)?;
        Ok(Self {
            owner,
            descriptor: CheckedInjectiveDescriptor {
                descriptor,
                proof: proof.expect("injective proof requested"),
            },
        })
    }

    pub(crate) const fn descriptor(&self) -> &CheckedInjectiveDescriptor<R> {
        &self.descriptor
    }
}

/// Build the one checked descriptor retained by an allocation-group record.
pub(crate) fn validate_descriptor<T: TensorScalar, R: TensorRank>(
    owner: &RootResourceIdentity,
    span: RootBoundSpan,
    shape: R::Shape,
    strides: R::Strides,
    offset: isize,
    require_injective: bool,
) -> Result<(CheckedDescriptor<R>, Option<WriteInjectivityProof>), AccessError> {
    make_descriptor::<T, R>(owner, span, shape, strides, offset, require_injective)
}

struct TypedReadAccess<'a, T: TensorScalar> {
    mapping: ProviderReadMapping<'a>,
    pointer: NonNull<T>,
    len: usize,
    _borrow: PhantomData<&'a T>,
}

impl<T: TensorScalar> fmt::Debug for TypedReadAccess<'_, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TypedReadAccess")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl<'a, T: TensorScalar> TypedReadAccess<'a, T> {
    fn new(mapping: ProviderReadMapping<'a>, expected_bytes: usize) -> Result<Self, AccessError> {
        let bytes = mapping.bytes();
        if bytes.len() != expected_bytes {
            return Err(AccessError::LengthMismatch {
                expected: expected_bytes,
                actual: bytes.len(),
            });
        }
        if bytes.as_ptr().align_offset(align_of::<T>()) != 0 {
            return Err(AccessError::Misaligned {
                required: align_of::<T>(),
            });
        }
        let len = bytes.len() / size_of::<T>();
        let pointer = NonNull::new(bytes.as_ptr() as *mut T).unwrap_or_else(NonNull::dangling);
        Ok(Self {
            mapping,
            pointer,
            len,
            _borrow: PhantomData,
        })
    }

    fn as_slice(&self, range: Range<usize>) -> &[T] {
        debug_assert!(range.end <= self.len);
        // SAFETY: the constructor checked byte length/alignment and the
        // checked layout proves this range is within the mapped allocation.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr().add(range.start), range.len()) }
    }
}

struct TypedWriteAccess<'a, T: TensorScalar> {
    mapping: ProviderWriteMapping<'a>,
    pointer: NonNull<T>,
    len: usize,
    _borrow: PhantomData<&'a mut T>,
}

impl<'a, T: TensorScalar> TypedWriteAccess<'a, T> {
    fn new(
        mut mapping: ProviderWriteMapping<'a>,
        expected_bytes: usize,
    ) -> Result<Self, AccessError> {
        let bytes = mapping.bytes_mut();
        if bytes.len() != expected_bytes {
            return Err(AccessError::LengthMismatch {
                expected: expected_bytes,
                actual: bytes.len(),
            });
        }
        if bytes.as_ptr().align_offset(align_of::<T>()) != 0 {
            return Err(AccessError::Misaligned {
                required: align_of::<T>(),
            });
        }
        let len = bytes.len() / size_of::<T>();
        let pointer = NonNull::new(bytes.as_mut_ptr() as *mut T).unwrap_or_else(NonNull::dangling);
        Ok(Self {
            mapping,
            pointer,
            len,
            _borrow: PhantomData,
        })
    }

    fn as_slice_mut(&mut self, range: Range<usize>) -> &mut [T] {
        debug_assert!(range.end <= self.len);
        // SAFETY: the constructor checked byte length/alignment and the
        // checked layout proves this range is within the mapped allocation.
        unsafe {
            std::slice::from_raw_parts_mut(self.pointer.as_ptr().add(range.start), range.len())
        }
    }
}

pub(crate) struct PreparedContiguousRead<'a, T: TensorScalar, R: TensorRank> {
    access: TypedReadAccess<'a, T>,
    element_range: Range<usize>,
    _rank: PhantomData<R>,
}

impl<T: TensorScalar, R: TensorRank> PreparedContiguousRead<'_, T, R> {
    pub(crate) fn as_slice(&self) -> &[T] {
        self.access.as_slice(self.element_range.clone())
    }

    pub(crate) fn iter_contiguous(&self) -> std::slice::Iter<'_, T> {
        self.as_slice().iter()
    }
}

pub(crate) struct PreparedContiguousWrite<'a, T: TensorScalar, R: TensorRank> {
    access: TypedWriteAccess<'a, T>,
    element_range: Range<usize>,
    _rank: PhantomData<R>,
}

impl<T: TensorScalar, R: TensorRank> PreparedContiguousWrite<'_, T, R> {
    pub(crate) fn as_slice_mut(&mut self) -> &mut [T] {
        self.access.as_slice_mut(self.element_range.clone())
    }

    pub(crate) fn iter_contiguous_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.as_slice_mut().iter_mut()
    }
}

pub(crate) struct PreparedStridedRead<'a, T: TensorScalar, R: TensorRank> {
    access: TypedReadAccess<'a, T>,
    plan: CheckedStrided<R>,
}

pub(crate) struct PreparedStridedWrite<'a, T: TensorScalar, R: TensorRank> {
    access: TypedWriteAccess<'a, T>,
    plan: CheckedStrided<R>,
}

pub(crate) struct PreparedStridedIter<'i, 'a, T: TensorScalar, R: TensorRank> {
    access: &'i TypedReadAccess<'a, T>,
    plan: &'i CheckedStrided<R>,
    coordinate: Box<[usize]>,
    next_offset: isize,
    remaining: usize,
}

pub(crate) struct PreparedStridedIterMut<'i, 'a, T: TensorScalar, R: TensorRank> {
    access: &'i mut TypedWriteAccess<'a, T>,
    plan: &'i CheckedStrided<R>,
    coordinate: Box<[usize]>,
    next_offset: isize,
    remaining: usize,
}

fn advance_cursor<R: TensorRank>(
    plan: &CheckedStrided<R>,
    coordinate: &mut [usize],
    next_offset: &mut isize,
    remaining: &mut usize,
) {
    *remaining -= 1;
    if *remaining == 0 {
        return;
    }
    for (axis, coordinate_value) in coordinate.iter_mut().enumerate() {
        *coordinate_value += 1;
        if *coordinate_value < plan.shape()[axis] {
            *next_offset = next_offset
                .checked_add(plan.strides()[axis])
                .expect("checked strided step");
            return;
        }
        *coordinate_value = 0;
        *next_offset = next_offset
            .checked_add(plan.carry()[axis])
            .expect("checked strided carry");
    }
}

impl<'i, 'a, T: TensorScalar, R: TensorRank> Iterator for PreparedStridedIter<'i, 'a, T, R> {
    type Item = &'i T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let offset = usize::try_from(self.next_offset).expect("checked strided offset");
        // SAFETY: CheckedStrided was constructed from TensorLayout bounds; the
        // cursor visits each proven offset exactly once.
        let value = unsafe { &*self.access.pointer.as_ptr().add(offset) };
        advance_cursor(
            self.plan,
            &mut self.coordinate,
            &mut self.next_offset,
            &mut self.remaining,
        );
        Some(value)
    }
}

impl<'i, 'a, T: TensorScalar, R: TensorRank> Iterator for PreparedStridedIterMut<'i, 'a, T, R> {
    type Item = &'i mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let offset = usize::try_from(self.next_offset).expect("checked strided offset");
        // SAFETY: CheckedInjectiveDescriptor proved unique in-bounds offsets;
        // this iterator owns the only mutable borrow of the mapped allocation.
        let value = unsafe { &mut *self.access.pointer.as_ptr().add(offset) };
        advance_cursor(
            self.plan,
            &mut self.coordinate,
            &mut self.next_offset,
            &mut self.remaining,
        );
        Some(value)
    }
}

impl<'a, T: TensorScalar, R: TensorRank> PreparedStridedRead<'a, T, R> {
    pub(crate) fn iter_strided(&self) -> PreparedStridedIter<'_, 'a, T, R> {
        PreparedStridedIter {
            access: &self.access,
            plan: &self.plan,
            coordinate: vec![0; self.plan.shape().len()].into_boxed_slice(),
            next_offset: self.plan.offset(),
            remaining: self.plan.element_count(),
        }
    }
}

impl<'a, T: TensorScalar, R: TensorRank> PreparedStridedWrite<'a, T, R> {
    pub(crate) fn iter_strided_mut<'i>(&'i mut self) -> PreparedStridedIterMut<'i, 'a, T, R> {
        PreparedStridedIterMut {
            access: &mut self.access,
            plan: &self.plan,
            coordinate: vec![0; self.plan.shape().len()].into_boxed_slice(),
            next_offset: self.plan.offset(),
            remaining: self.plan.element_count(),
        }
    }
}

pub(crate) enum PreparedHostRead<'a, T: TensorScalar, R: TensorRank> {
    Contiguous(PreparedContiguousRead<'a, T, R>),
    Strided(PreparedStridedRead<'a, T, R>),
}

pub(crate) enum PreparedHostWrite<'a, T: TensorScalar, R: TensorRank> {
    Contiguous(PreparedContiguousWrite<'a, T, R>),
    Strided(PreparedStridedWrite<'a, T, R>),
}

pub(crate) struct PreparedDeviceRead<'a, T: TensorScalar, R: TensorRank> {
    checked: CheckedRead<'a, R>,
    provider_state: Box<dyn PreparedDeviceAccess>,
    _scalar: PhantomData<T>,
}

pub(crate) struct PreparedDeviceWrite<'a, T: TensorScalar, R: TensorRank> {
    checked: CheckedWrite<'a, R>,
    provider_state: Box<dyn PreparedDeviceAccess>,
    _scalar: PhantomData<T>,
}

pub(crate) enum PreparedRead<'a, T: TensorScalar, R: TensorRank> {
    Host(PreparedHostRead<'a, T, R>),
    Device(PreparedDeviceRead<'a, T, R>),
}

pub(crate) enum PreparedWrite<'a, T: TensorScalar, R: TensorRank> {
    Host(PreparedHostWrite<'a, T, R>),
    Device(PreparedDeviceWrite<'a, T, R>),
}

impl<'a, T: TensorScalar, R: TensorRank> PreparedRead<'a, T, R> {
    pub(crate) fn into_device_state(
        self,
    ) -> Result<Box<dyn PreparedDeviceAccess + 'a>, AccessError> {
        match self {
            Self::Device(device) => Ok(device.provider_state),
            Self::Host(_) => Err(AccessError::Unsupported { backend: "host" }),
        }
    }

    pub(crate) fn as_slice(&self) -> Option<&[T]> {
        match self {
            Self::Host(PreparedHostRead::Contiguous(access)) => Some(access.as_slice()),
            _ => None,
        }
    }

    pub(crate) fn iter_strided(&self) -> Option<PreparedStridedIter<'_, 'a, T, R>> {
        match self {
            Self::Host(PreparedHostRead::Strided(access)) => Some(access.iter_strided()),
            _ => None,
        }
    }
}

impl<'a, T: TensorScalar, R: TensorRank> PreparedWrite<'a, T, R> {
    pub(crate) fn into_device_state(
        self,
    ) -> Result<Box<dyn PreparedDeviceAccess + 'a>, AccessError> {
        match self {
            Self::Device(device) => Ok(device.provider_state),
            Self::Host(_) => Err(AccessError::Unsupported { backend: "host" }),
        }
    }

    pub(crate) fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        match self {
            Self::Host(PreparedHostWrite::Contiguous(access)) => Some(access.as_slice_mut()),
            _ => None,
        }
    }

    pub(crate) fn iter_strided_mut<'i>(
        &'i mut self,
    ) -> Option<PreparedStridedIterMut<'i, 'a, T, R>> {
        match self {
            Self::Host(PreparedHostWrite::Strided(access)) => Some(access.iter_strided_mut()),
            _ => None,
        }
    }
}

fn typed_mapping_error<T: TensorScalar>(
    descriptor: &CheckedDescriptor<impl TensorRank>,
) -> Option<AccessError> {
    (descriptor.dtype() != T::dtype()).then_some(AccessError::DTypeMismatch {
        expected: descriptor.dtype(),
        actual: T::dtype(),
    })
}

type PreparedReadFailure<'a, R> = Box<(CheckedRead<'a, R>, AccessError)>;
type PreparedWriteFailure<'a, R> = Box<(CheckedWrite<'a, R>, AccessError)>;

fn device_access_error(error: DeviceAccessError) -> AccessError {
    match error {
        DeviceAccessError::Unsupported { backend } => AccessError::Unsupported { backend },
        DeviceAccessError::InvalidRequest { message }
        | DeviceAccessError::ProviderFailure { message } => AccessError::Provider { message },
    }
}

fn device_request<'a, R: TensorRank>(
    owner: RootResourceIdentity,
    descriptor: &'a CheckedDescriptor<R>,
) -> DeviceAccessRequest<'a> {
    let key = owner.extent().key();
    // The request owns no identity or access authority. Its layout metadata is
    // copied into the provider transition so the provider sees the checked
    // descriptor rather than an empty root fallback.
    DeviceAccessRequest::new(
        key.domain(),
        key.local(),
        descriptor.span.byte_len(),
        descriptor.element_size(),
        descriptor.shape(),
        descriptor.strides(),
        descriptor.offset(),
    )
}

pub(crate) fn prepare_read<'a, T: TensorScalar, R: TensorRank>(
    checked: CheckedRead<'a, R>,
    target: AccessTarget,
) -> Result<PreparedRead<'a, T, R>, PreparedReadFailure<'a, R>> {
    if let Some(error) = typed_mapping_error::<T>(&checked.descriptor) {
        return Err(Box::new((checked, error)));
    }
    if target == AccessTarget::Device {
        let CheckedRead { owner, descriptor } = checked;
        let request = device_request(owner.root_identity(), &descriptor);
        let provider_state = match owner.prepare_device_access(request) {
            Ok(state) => state,
            Err(error) => {
                return Err(Box::new((
                    CheckedRead { owner, descriptor },
                    device_access_error(error),
                )))
            }
        };
        return Ok(PreparedRead::Device(PreparedDeviceRead {
            checked: CheckedRead { owner, descriptor },
            provider_state,
            _scalar: PhantomData,
        }));
    }

    let CheckedRead { owner, descriptor } = checked;
    let span = descriptor.span;
    let mapping = match owner.map_read(span, descriptor.dtype) {
        Ok(mapping) => mapping,
        Err(error) => return Err(Box::new((CheckedRead { owner, descriptor }, error))),
    };
    let access = match TypedReadAccess::new(mapping, span.byte_len()) {
        Ok(access) => access,
        Err(error) => return Err(Box::new((CheckedRead { owner, descriptor }, error))),
    };
    let host = match descriptor.layout {
        CheckedLayout::Contiguous { element_range } => {
            PreparedHostRead::Contiguous(PreparedContiguousRead {
                access,
                element_range,
                _rank: PhantomData,
            })
        }
        CheckedLayout::Strided(plan) => {
            PreparedHostRead::Strided(PreparedStridedRead { access, plan })
        }
    };
    Ok(PreparedRead::Host(host))
}

pub(crate) fn prepare_write<'a, T: TensorScalar, R: TensorRank>(
    checked: CheckedWrite<'a, R>,
    target: AccessTarget,
) -> Result<PreparedWrite<'a, T, R>, PreparedWriteFailure<'a, R>> {
    if let Some(error) = typed_mapping_error::<T>(&checked.descriptor.descriptor) {
        return Err(Box::new((checked, error)));
    }
    if target == AccessTarget::Device {
        let CheckedWrite { owner, descriptor } = checked;
        let descriptor_value = descriptor.descriptor.clone();
        let request = device_request(owner.root_identity(), &descriptor_value);
        let provider_state = match owner.prepare_device_access(request) {
            Ok(state) => state,
            Err(error) => {
                return Err(Box::new((
                    CheckedWrite { owner, descriptor },
                    device_access_error(error),
                )))
            }
        };
        return Ok(PreparedWrite::Device(PreparedDeviceWrite {
            checked: CheckedWrite { owner, descriptor },
            provider_state,
            _scalar: PhantomData,
        }));
    }

    let CheckedWrite { owner, descriptor } = checked;
    let span = descriptor.descriptor.span;
    let descriptor_value = descriptor.descriptor.clone();
    let mapping = match owner.map_write(span, descriptor_value.dtype) {
        Ok(mapping) => mapping,
        Err(error) => return Err(Box::new((CheckedWrite { owner, descriptor }, error))),
    };
    let access = match TypedWriteAccess::new(mapping, span.byte_len()) {
        Ok(access) => access,
        Err(error) => return Err(Box::new((CheckedWrite { owner, descriptor }, error))),
    };
    let host = match descriptor_value.layout {
        CheckedLayout::Contiguous { element_range } => {
            PreparedHostWrite::Contiguous(PreparedContiguousWrite {
                access,
                element_range,
                _rank: PhantomData,
            })
        }
        CheckedLayout::Strided(plan) => {
            PreparedHostWrite::Strided(PreparedStridedWrite { access, plan })
        }
    };
    Ok(PreparedWrite::Host(host))
}
