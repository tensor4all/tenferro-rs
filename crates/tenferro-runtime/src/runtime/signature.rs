use std::mem::align_of;
use std::mem::size_of;

use tenferro_tensor::{
    AllocationDomainId, DType, Placement, ShapeVec, StrideVec, Tensor, TensorRead, TensorScalar,
    TensorView, TypedTensor, TypedTensorView,
};

use super::{InputSignatureError, LayoutClass, PrepareError};

const COMPACT_COL_MAJOR_LAYOUT: &str = "tenferro.layout.compact-col-major.v1";
const STRIDED_LAYOUT: &str = "tenferro.layout.strided.v1";

/// Value-free metadata signature for a tensor input.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
/// use tenferro_tensor::Placement;
///
/// let entry = InputSignatureEntry::new(
///     DType::F64,
///     [2_usize].into_iter().collect(),
///     Placement::default(),
///     LayoutClass::new("tenferro.layout.strided")?,
///     [1_isize].into_iter().collect(),
///     Some(3),
/// )?;
/// assert_eq!(entry.dtype(), DType::F64);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSignatureEntry {
    dtype: DType,
    shape: ShapeVec,
    placement: Placement,
    layout_class: LayoutClass,
    strides: StrideVec,
    alignment_log2: Option<u8>,
    backend_family: Option<&'static str>,
    allocation_domain: Option<AllocationDomainId>,
}

#[derive(Clone, Copy)]
struct InputPhysicalIdentity {
    backend_family: Option<&'static str>,
    allocation_domain: Option<AllocationDomainId>,
}

impl InputSignatureEntry {
    /// Build one value-free input signature entry.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let entry = InputSignatureEntry::new(
    ///     DType::I32,
    ///     [4_usize].into_iter().collect(),
    ///     Placement::default(),
    ///     LayoutClass::new("tenferro.layout.compact")?,
    ///     [1_isize].into_iter().collect(),
    ///     None,
    /// )?;
    /// assert_eq!(entry.shape(), &[4]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`InputSignatureError::ShapeStrideRankMismatch`] when shape and
    /// stride ranks differ, or [`InputSignatureError::InvalidAlignmentClass`]
    /// when `alignment_log2` is outside the finite `usize` alignment lattice.
    pub fn new(
        dtype: DType,
        shape: ShapeVec,
        placement: Placement,
        layout_class: LayoutClass,
        strides: StrideVec,
        alignment_log2: Option<u8>,
    ) -> Result<Self, InputSignatureError> {
        validate_entry(&shape, &strides, alignment_log2)?;
        Ok(Self {
            dtype,
            shape,
            placement,
            layout_class,
            strides,
            alignment_log2,
            backend_family: None,
            allocation_domain: None,
        })
    }

    fn from_validated_metadata(
        dtype: DType,
        shape: ShapeVec,
        placement: Placement,
        layout_class: LayoutClass,
        strides: StrideVec,
        alignment_log2: Option<u8>,
        physical_identity: InputPhysicalIdentity,
    ) -> Self {
        Self {
            dtype,
            shape,
            placement,
            layout_class,
            strides,
            alignment_log2,
            backend_family: physical_identity.backend_family,
            allocation_domain: physical_identity.allocation_domain,
        }
    }

    /// Return the dtype component.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let entry = InputSignatureEntry::new(
    ///     DType::Bool,
    ///     [1_usize].into_iter().collect(),
    ///     Placement::default(),
    ///     LayoutClass::new("tenferro.layout.strided")?,
    ///     [1_isize].into_iter().collect(),
    ///     None,
    /// )?;
    /// assert_eq!(entry.dtype(), DType::Bool);
    /// # Ok(())
    /// # }
    /// ```
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the shape component.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let entry = InputSignatureEntry::new(
    ///     DType::F64,
    ///     [2_usize, 3].into_iter().collect(),
    ///     Placement::default(),
    ///     LayoutClass::new("tenferro.layout.strided")?,
    ///     [1_isize, 2].into_iter().collect(),
    ///     None,
    /// )?;
    /// assert_eq!(entry.shape(), &[2, 3]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the placement metadata component.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
    /// use tenferro_tensor::{MemoryKind, Placement};
    ///
    /// let entry = InputSignatureEntry::new(
    ///     DType::F64,
    ///     [1_usize].into_iter().collect(),
    ///     Placement::default(),
    ///     LayoutClass::new("tenferro.layout.strided")?,
    ///     [1_isize].into_iter().collect(),
    ///     None,
    /// )?;
    /// assert_eq!(entry.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok(())
    /// # }
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Return the layout class component.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let layout = LayoutClass::new("tenferro.layout.strided")?;
    /// let entry = InputSignatureEntry::new(
    ///     DType::F64,
    ///     [1_usize].into_iter().collect(),
    ///     Placement::default(),
    ///     layout.clone(),
    ///     [1_isize].into_iter().collect(),
    ///     None,
    /// )?;
    /// assert_eq!(entry.layout_class(), &layout);
    /// # Ok(())
    /// # }
    /// ```
    pub fn layout_class(&self) -> &LayoutClass {
        &self.layout_class
    }

    /// Return the stride metadata component.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let entry = InputSignatureEntry::new(
    ///     DType::F64,
    ///     [2_usize].into_iter().collect(),
    ///     Placement::default(),
    ///     LayoutClass::new("tenferro.layout.strided")?,
    ///     [2_isize].into_iter().collect(),
    ///     None,
    /// )?;
    /// assert_eq!(entry.strides(), &[2]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Return the known alignment class, if available.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{DType, InputSignatureEntry, LayoutClass};
    /// use tenferro_tensor::Placement;
    ///
    /// let entry = InputSignatureEntry::new(
    ///     DType::F64,
    ///     [1_usize].into_iter().collect(),
    ///     Placement::default(),
    ///     LayoutClass::new("tenferro.layout.strided")?,
    ///     [1_isize].into_iter().collect(),
    ///     Some(3),
    /// )?;
    /// assert_eq!(entry.alignment_log2(), Some(3));
    /// # Ok(())
    /// # }
    /// ```
    pub fn alignment_log2(&self) -> Option<u8> {
        self.alignment_log2
    }

    pub(super) fn backend_family(&self) -> Option<&'static str> {
        self.backend_family
    }

    pub(super) fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.allocation_domain
    }

    pub(crate) fn logical_retained_bytes(&self) -> Option<usize> {
        checked_sum([
            spilled_bytes::<usize>(self.shape.spilled(), self.shape.len())?,
            spilled_bytes::<isize>(self.strides.spilled(), self.strides.len())?,
        ])
    }
}

/// Value-free signature of all tensor inputs for one prepare request.
///
/// # Examples
///
/// ```
/// use tenferro_runtime::InputSignature;
///
/// let signature = InputSignature::new(Vec::new());
/// assert!(signature.entries().is_empty());
/// ```
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct InputSignature {
    entries: Vec<InputSignatureEntry>,
}

impl InputSignature {
    /// Build a signature from already prepared entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::InputSignature;
    ///
    /// let signature = InputSignature::new(Vec::new());
    /// assert_eq!(signature.entries().len(), 0);
    /// ```
    pub fn new(entries: Vec<InputSignatureEntry>) -> Self {
        Self { entries }
    }

    /// Build a value-free signature from borrowed tensor reads.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use tenferro_runtime::{InputSignature, TensorRead, Tensor};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let signature = InputSignature::from_reads(&[TensorRead::from_tensor(&tensor)])?;
    /// assert_eq!(signature.entries()[0].shape(), &[2]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`PrepareError::InputSignature`] with the original typed tensor
    /// metadata error when shape, stride, or compactness metadata cannot be read.
    pub fn from_reads(reads: &[TensorRead<'_>]) -> Result<Self, PrepareError> {
        let mut entries = Vec::with_capacity(reads.len());
        for (input, read) in reads.iter().enumerate() {
            let strides = read
                .strides()
                .map_err(|source| PrepareError::InputSignature {
                    source: InputSignatureError::TensorMetadata { input, source },
                })?;
            let compact =
                read.is_col_major_contiguous()
                    .map_err(|source| PrepareError::InputSignature {
                        source: InputSignatureError::TensorMetadata { input, source },
                    })?;
            let shape = read.shape().iter().copied().collect();
            entries.push(InputSignatureEntry::from_validated_metadata(
                read.dtype(),
                shape,
                read_placement(read),
                layout_class(compact),
                strides.into_iter().collect(),
                read_alignment_log2(read),
                InputPhysicalIdentity {
                    backend_family: read.backend_family(),
                    allocation_domain: read.allocation_domain(),
                },
            ));
        }
        Ok(Self { entries })
    }

    /// Return the per-input entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_runtime::InputSignature;
    ///
    /// assert!(InputSignature::new(Vec::new()).entries().is_empty());
    /// ```
    pub fn entries(&self) -> &[InputSignatureEntry] {
        &self.entries
    }

    pub(crate) fn logical_retained_bytes(&self) -> Option<usize> {
        checked_sum([
            self.entries
                .len()
                .checked_mul(size_of::<InputSignatureEntry>())?,
            checked_sum_options(
                self.entries
                    .iter()
                    .map(InputSignatureEntry::logical_retained_bytes),
            )?,
        ])
    }
}

fn spilled_bytes<T>(spilled: bool, len: usize) -> Option<usize> {
    if spilled {
        len.checked_mul(size_of::<T>())
    } else {
        Some(0)
    }
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value))
}

fn checked_sum_options(values: impl IntoIterator<Item = Option<usize>>) -> Option<usize> {
    values
        .into_iter()
        .try_fold(0usize, |sum, value| sum.checked_add(value?))
}

fn validate_entry(
    shape: &[usize],
    strides: &[isize],
    alignment_log2: Option<u8>,
) -> Result<(), InputSignatureError> {
    if shape.len() != strides.len() {
        return Err(InputSignatureError::ShapeStrideRankMismatch {
            rank: shape.len(),
            stride_count: strides.len(),
        });
    }
    if let Some(alignment_log2) = alignment_log2 {
        if u32::from(alignment_log2) >= usize::BITS {
            return Err(InputSignatureError::InvalidAlignmentClass { alignment_log2 });
        }
    }
    Ok(())
}

pub(super) fn read_placement(read: &TensorRead<'_>) -> Placement {
    match read {
        TensorRead::Tensor(tensor) => tensor.placement().clone(),
        TensorRead::View(view) => view_placement(view),
    }
}

fn view_placement(view: &TensorView<'_>) -> Placement {
    match view {
        TensorView::F32(view) => view.placement().clone(),
        TensorView::F64(view) => view.placement().clone(),
        TensorView::I32(view) => view.placement().clone(),
        TensorView::I64(view) => view.placement().clone(),
        TensorView::Bool(view) => view.placement().clone(),
        TensorView::C32(view) => view.placement().clone(),
        TensorView::C64(view) => view.placement().clone(),
    }
}

fn layout_class(compact: bool) -> LayoutClass {
    let value = if compact {
        COMPACT_COL_MAJOR_LAYOUT
    } else {
        STRIDED_LAYOUT
    };
    LayoutClass::runtime_created(value)
}

fn read_alignment_log2(read: &TensorRead<'_>) -> Option<u8> {
    match read {
        TensorRead::Tensor(tensor) => tensor_alignment_log2(tensor),
        TensorRead::View(view) => view_alignment_log2(view),
    }
}

fn tensor_alignment_log2(tensor: &Tensor) -> Option<u8> {
    match tensor {
        Tensor::F32(tensor) => typed_tensor_alignment_log2(tensor),
        Tensor::F64(tensor) => typed_tensor_alignment_log2(tensor),
        Tensor::I32(tensor) => typed_tensor_alignment_log2(tensor),
        Tensor::I64(tensor) => typed_tensor_alignment_log2(tensor),
        Tensor::Bool(tensor) => typed_tensor_alignment_log2(tensor),
        Tensor::C32(tensor) => typed_tensor_alignment_log2(tensor),
        Tensor::C64(tensor) => typed_tensor_alignment_log2(tensor),
    }
}

fn typed_tensor_alignment_log2<T: TensorScalar>(tensor: &TypedTensor<T>) -> Option<u8> {
    if tensor.buffer().is_backend() {
        return None;
    }
    if shape_is_empty(tensor.shape()) {
        return Some(type_alignment_log2::<T>());
    }
    tensor
        .host_data()
        .ok()
        .map(|data| pointer_alignment_log2::<T>(data.as_ptr()))
}

fn view_alignment_log2(view: &TensorView<'_>) -> Option<u8> {
    match view {
        TensorView::F32(view) => typed_view_alignment_log2(view),
        TensorView::F64(view) => typed_view_alignment_log2(view),
        TensorView::I32(view) => typed_view_alignment_log2(view),
        TensorView::I64(view) => typed_view_alignment_log2(view),
        TensorView::Bool(view) => typed_view_alignment_log2(view),
        TensorView::C32(view) => typed_view_alignment_log2(view),
        TensorView::C64(view) => typed_view_alignment_log2(view),
    }
}

fn typed_view_alignment_log2<T: 'static>(view: &TypedTensorView<'_, T>) -> Option<u8> {
    if view.backend_buffer().is_some() {
        return None;
    }
    if shape_is_empty(view.shape()) {
        return Some(type_alignment_log2::<T>());
    }
    view.host_storage().ok().map(|data| {
        let pointer = data.as_ptr().wrapping_offset(view.offset());
        pointer_alignment_log2::<T>(pointer)
    })
}

fn shape_is_empty(shape: &[usize]) -> bool {
    shape.contains(&0)
}

fn type_alignment_log2<T>() -> u8 {
    align_of::<T>().trailing_zeros().min(usize::BITS - 1) as u8
}

fn pointer_alignment_log2<T>(pointer: *const T) -> u8 {
    (pointer as usize)
        .trailing_zeros()
        .min(align_of::<T>().trailing_zeros())
        .min(usize::BITS - 1) as u8
}
