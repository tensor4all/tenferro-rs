use num_complex::{Complex, Complex32, Complex64};
use num_traits::{One, Zero};
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

use crate::config::SliceConfig;
use tenferro_tensor_core::SliceSpec as CoreSliceSpec;
pub use tenferro_tensor_core::{DynRank, Rank, TensorLayout, TensorRank};

mod accessors;
mod shape_packing;
mod strided_view;

pub use strided_view::StridedSliceSpec;

/// Memory location for tensor storage.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::MemoryKind;
///
/// let kind = MemoryKind::UnpinnedHost;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Managed,
    Other(String),
}

/// Compute device family.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DeviceKind;
///
/// let kind = DeviceKind::Cpu;
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    Gpu(GpuBackendKind),
    Other(String),
}

/// GPU backend family used by placement metadata.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::GpuBackendKind;
///
/// let kind = GpuBackendKind::Cuda;
/// let webgpu = GpuBackendKind::WebGpu;
/// assert_ne!(kind, webgpu);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GpuBackendKind {
    Cuda,
    WebGpu,
    Rocm,
    Other(String),
}

/// Concrete compute device identifier.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DeviceId, DeviceKind, GpuBackendKind};
///
/// let device = DeviceId {
///     kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
///     ordinal: 0,
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId {
    pub kind: DeviceKind,
    pub ordinal: usize,
}

/// Placement metadata for a tensor buffer.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement};
///
/// let placement = Placement {
///     memory_kind: MemoryKind::Device,
///     device: Some(DeviceId {
///         kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
///         ordinal: 0,
///     }),
/// };
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    pub memory_kind: MemoryKind,
    pub device: Option<DeviceId>,
}

/// Backend-owned buffer handle.
///
/// `BufferHandle::new` creates an empty opaque handle. Use
/// [`BufferHandle::new_with_len`] when test or adapter code needs to model a
/// non-empty backend allocation.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::BufferHandle;
///
/// let handle = BufferHandle::<f64>::new(7);
/// ```
#[derive(Clone)]
pub struct BufferHandle<T> {
    id: u64,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Debug for BufferHandle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferHandle")
            .field("id", &self.id)
            .finish()
    }
}

impl<T> BufferHandle<T> {
    /// Create a new backend buffer handle.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::BufferHandle;
    ///
    /// let handle = BufferHandle::<f64>::new(1);
    /// assert_eq!(tenferro_tensor::BackendBuffer::len(&handle), 0);
    /// ```
    pub fn new(id: u64) -> Self {
        Self::new_with_len(id, 0)
    }

    /// Create a new backend buffer handle with a logical element count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{BackendBuffer, BufferHandle};
    ///
    /// let handle = BufferHandle::<f64>::new_with_len(1, 4);
    /// assert_eq!(BackendBuffer::len(&handle), 4);
    /// ```
    pub fn new_with_len(id: u64, len: usize) -> Self {
        Self {
            id,
            len,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Opaque backend-owned tensor buffer.
///
/// Tensor core never inspects backend-native allocations directly. Backend
/// crates store their own concrete handle types behind this trait and
/// downcast inside the owning backend only.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
/// use tenferro_tensor::{BackendBuffer, BufferHandle};
///
/// let buffer: Arc<dyn BackendBuffer<f64>> = Arc::new(BufferHandle::<f64>::new_with_len(7, 2));
/// assert_eq!(buffer.backend_family(), "opaque");
/// assert_eq!(buffer.len(), 2);
/// ```
pub trait BackendBuffer<T>: Debug + Send + Sync + 'static {
    /// Stable backend family identifier.
    fn backend_family(&self) -> &'static str;

    /// Number of logical elements in the backend allocation.
    fn len(&self) -> usize;

    /// Returns `true` when the backend allocation is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Type-erased access for the backend crate that owns the concrete handle.
    fn as_any(&self) -> &dyn Any;
}

impl<T: Send + Sync + 'static> BackendBuffer<T> for BufferHandle<T> {
    fn backend_family(&self) -> &'static str {
        "opaque"
    }

    fn len(&self) -> usize {
        self.len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Tensor storage.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::Buffer;
///
/// let host = Buffer::Host(vec![1.0_f64, 2.0]);
/// ```
#[derive(Clone, Debug)]
pub enum Buffer<T> {
    Host(Vec<T>),
    Backend(Arc<dyn BackendBuffer<T>>),
}

impl<T: 'static> Buffer<T> {
    /// Return the physical element count in this buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Buffer;
    ///
    /// assert_eq!(Buffer::Host(vec![1_i32, 2]).len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
        }
    }

    /// Return whether this buffer has no physical elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Buffer;
    ///
    /// assert!(Buffer::<i32>::Host(Vec::new()).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return whether the storage is backend-owned rather than host-owned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Buffer;
    ///
    /// assert!(!Buffer::Host(vec![1_i32]).is_backend());
    /// ```
    pub fn is_backend(&self) -> bool {
        matches!(self, Self::Backend(_))
    }
}

/// Runtime typed tensor storage with compile-time scalar type and rank metadata.
///
/// Owned tensors are compact column-major. Arbitrary strides and metadata-only
/// layout changes are represented by [`TypedTensorView`] and
/// [`TypedTensorViewMut`]. The buffer may be host-backed or backend-backed;
/// host-inspection methods do not download backend buffers implicitly.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Rank, Tensor, TypedTensor};
///
/// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(t.shape(), &[2, 2]);
///
/// let static_rank = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
/// assert_eq!(static_rank.rank(), 2);
///
/// let dynamic = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
/// assert_eq!(dynamic.shape(), &[2, 2]);
/// ```
///
/// The `R` parameter stores rank metadata. It defaults to dynamic rank
/// (`DynRank`); use [`Rank<N>`](Rank) for compile-time rank validation.
/// The dtype-erased [`Tensor`] enum remains dynamic-rank.
#[derive(Clone, Debug)]
pub struct TypedTensor<T, R: TensorRank = DynRank> {
    buffer: Buffer<T>,
    layout: TensorLayout<R>,
    placement: Placement,
}

/// Borrowed tensor buffer reference used by read-only typed views.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorBufferRef;
///
/// let data = [1_i32, 2];
/// let buffer = TensorBufferRef::Host(&data);
/// assert_eq!(buffer.len(), 2);
/// ```
#[derive(Debug)]
pub enum TensorBufferRef<'a, T> {
    Host(&'a [T]),
    Backend(Arc<dyn BackendBuffer<T>>),
}

impl<T> Clone for TensorBufferRef<'_, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Host(data) => Self::Host(data),
            Self::Backend(buffer) => Self::Backend(Arc::clone(buffer)),
        }
    }
}

impl<T: 'static> TensorBufferRef<'_, T> {
    /// Return the logical length of the backing allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRef;
    ///
    /// let data = [1_i32, 2, 3];
    /// assert_eq!(TensorBufferRef::Host(&data).len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
        }
    }

    /// Return whether the backing allocation is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRef;
    ///
    /// let data: [f64; 0] = [];
    /// assert!(TensorBufferRef::Host(&data).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed tensor buffer reference used by mutable typed views.
///
/// Backend buffers can be represented for residency metadata, but this crate
/// does not expose host mutation for backend-native allocations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TensorBufferRefMut;
///
/// let mut data = [1_i32, 2];
/// let buffer = TensorBufferRefMut::Host(&mut data);
/// assert_eq!(buffer.len(), 2);
/// ```
#[derive(Debug)]
pub enum TensorBufferRefMut<'a, T> {
    Host(&'a mut [T]),
    Backend(Arc<dyn BackendBuffer<T>>),
}

impl<T: 'static> TensorBufferRefMut<'_, T> {
    /// Return the logical length of the backing allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRefMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// assert_eq!(TensorBufferRefMut::Host(&mut data).len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Host(data) => data.len(),
            Self::Backend(buffer) => buffer.len(),
        }
    }

    /// Return whether the backing allocation is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TensorBufferRefMut;
    ///
    /// let mut data: [f64; 0] = [];
    /// assert!(TensorBufferRefMut::Host(&mut data).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Read-only borrowed view of typed tensor storage with arbitrary strides.
///
/// `TypedTensorView` is the typed representation for layout-only tensor
/// transformations. It borrows an existing host or backend allocation and
/// carries a logical shape, strides, and an offset. Slicing, reshaping when
/// stride-compatible, and [`transpose_view`](TypedTensorView::transpose_view)
/// update only metadata and do not copy storage.
///
/// Materialize through [`TensorStructural::to_contiguous_read`](crate::TensorStructural::to_contiguous_read)
/// on the active backend session when a compact owned [`TypedTensor`] is
/// required. Use [`TypedTensorView::as_slice`] only when the current view is
/// contiguous in the requested layout.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Rank, TypedTensorView};
///
/// let data = [1_i32, 2, 3, 4];
/// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data)?;
/// assert_eq!(view.get(&[1, 1]), Some(&4));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct TypedTensorView<'a, T, R: TensorRank = DynRank> {
    buffer: TensorBufferRef<'a, T>,
    layout: TensorLayout<R>,
    placement: Placement,
}

impl<'a, T: 'static> TypedTensorView<'a, T, DynRank> {
    /// Create a borrowed dynamic-rank view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_col_major(&[2, 2], &data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_col_major(shape: &[usize], data: &'a [T]) -> crate::Result<Self> {
        let layout = TensorLayout::<DynRank>::compact(shape.to_vec().into())
            .map_err(|err| tensor_layout_error("TypedTensorView::from_col_major", err))?;
        Self::from_buffer_ref(
            layout.shape().to_vec(),
            layout.strides().to_vec(),
            layout.offset(),
            TensorBufferRef::Host(data),
            default_placement(),
            "TypedTensorView::from_col_major",
        )
    }

    /// Create a borrowed host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![-1], 2, &data)?;
    /// assert_eq!(view.get(&[2]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_slice(
        shape: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: isize,
        data: &'a [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref(
            shape.as_ref().to_vec(),
            strides.as_ref().to_vec(),
            offset,
            TensorBufferRef::Host(data),
            default_placement(),
            "TypedTensorView::from_slice",
        )
    }
}

impl<'a, T: 'static, R: TensorRank> TypedTensorView<'a, T, R> {
    /// Create a rank-generic borrowed host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data)?;
    /// assert_eq!(view.get(&[1, 1]), Some(&4));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_slice_ranked(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref(
            shape,
            strides,
            offset,
            TensorBufferRef::Host(data),
            default_placement(),
            "TypedTensorView::from_slice_ranked",
        )
    }

    fn from_buffer_ref(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        buffer: TensorBufferRef<'a, T>,
        placement: Placement,
        op: &'static str,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            buffer,
            layout,
            placement,
        })
    }

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![-1], 1, &data)?;
    /// assert_eq!(view.strides(), &[-1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Return the physical element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![1], vec![1], 1, &data)?;
    /// assert_eq!(view.offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    /// Return the borrowed host storage backing this view.
    ///
    /// This exposes the entire backing host allocation, not just the logical
    /// slice covered by this view. Use [`TypedTensorView::as_slice`] when the
    /// caller needs the contiguous logical region instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.host_storage()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn host_storage(&self) -> crate::Result<&'a [T]> {
        match &self.buffer {
            TensorBufferRef::Host(data) => Ok(data),
            TensorBufferRef::Backend(_) => Err(crate::Error::backend_failure(
                "TypedTensorView::host_storage",
                "backend buffers cannot expose host storage; download explicitly first",
            )),
        }
    }

    /// Return the number of logical elements in this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [0_i32; 6];
    /// let view = TypedTensorView::from_slice(vec![2, 3], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.n_elements(), 6);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn n_elements(&self) -> usize {
        // Invariant: public view constructors validate logical element count.
        match checked_view_element_count(self.shape(), "TypedTensorView::n_elements") {
            Ok(n) => n,
            Err(err) => {
                unreachable!("TypedTensorView layout shape is validated at construction: {err}")
            }
        }
    }

    /// Return layout metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert!(view.layout().is_compact_col_major().unwrap());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Return placement metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, TypedTensorView};
    ///
    /// let data = [1_i32];
    /// let view = TypedTensorView::from_slice(vec![1], vec![1], 0, &data)?;
    /// assert_eq!(view.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Return the backend allocation for backend integrations.
    #[doc(hidden)]
    pub fn backend_buffer(&self) -> Option<&Arc<dyn BackendBuffer<T>>> {
        match &self.buffer {
            TensorBufferRef::Host(_) => None,
            TensorBufferRef::Backend(buffer) => Some(buffer),
        }
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![-1], 2, &data)?;
    /// assert_eq!(view.linear_offset(&[2]), Some(0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_view_offset(self.shape(), self.strides(), self.offset(), indices)
    }

    /// Compute the physical element offset for a logical index, returning a typed error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice([3], [-1], 2, &data)?;
    /// assert_eq!(view.layout_linear_offset(&[2])?, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        checked_view_offset_result(
            self.shape(),
            self.strides(),
            self.offset(),
            indices,
            "TypedTensorView::layout_linear_offset",
        )
    }

    /// Return whether this view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice([2], [1], 0, &data)?;
    /// assert!(view.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        self.layout
            .is_compact_col_major()
            .map_err(|err| tensor_layout_error("TypedTensorView::is_col_major_contiguous", err))
    }

    /// Return a compact string summary of this view's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice([2], [1], 0, &data)?;
    /// assert!(view.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    /// Assert this view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice([2], [1], 0, &data)?;
    /// view.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TypedTensorView::assert_col_major_contiguous",
        )
    }

    /// Borrow one host element by logical index.
    ///
    /// Returns `None` for out-of-bounds indices and backend buffers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 0, &data)?;
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.linear_offset(indices)?;
        match &self.buffer {
            TensorBufferRef::Host(data) => data.get(offset),
            TensorBufferRef::Backend(_) => None,
        }
    }

    /// Borrow the contiguous host slice covered by this view.
    ///
    /// Returns an explicit error for backend buffers and for non-contiguous
    /// layouts. This method never downloads or materializes backend data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![2], vec![1], 1, &data)?;
    /// assert_eq!(view.as_slice()?, &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_slice(&self) -> crate::Result<&'a [T]> {
        let data =
            match &self.buffer {
                TensorBufferRef::Host(data) => data,
                TensorBufferRef::Backend(_) => return Err(crate::Error::backend_failure(
                    "TypedTensorView::as_slice",
                    "backend buffers cannot be inspected as host slices; download explicitly first",
                )),
            };
        contiguous_layout_slice(self.layout(), data, "TypedTensorView::as_slice")
    }

    /// Return a metadata-only axis permutation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4, 5, 6];
    /// let view = TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 3], [1, 2], 0, &data)?;
    /// let transposed = view.transpose_view([1, 0])?;
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn transpose_view(&self, axes: impl AsRef<[usize]>) -> crate::Result<Self> {
        let layout = self
            .layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TypedTensorView::transpose_view", err))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            layout,
            placement: self.placement.clone(),
        })
    }

    /// Return a metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3];
    /// let view = TypedTensorView::from_slice(vec![3], vec![1], 0, &data)?;
    /// let reversed = view.try_slice(&[StridedSliceSpec::reverse()])?;
    /// assert_eq!(reversed.get(&[0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice(&self, slices: &[StridedSliceSpec]) -> crate::Result<Self> {
        let specs = core_slice_specs(slices, self.shape(), "TypedTensorView::try_slice")?;
        let layout = self
            .layout
            .slice_view(specs, self.buffer.len())
            .map_err(|err| tensor_layout_error("TypedTensorView::try_slice", err))?;
        Ok(Self {
            buffer: self.buffer.clone(),
            layout,
            placement: self.placement.clone(),
        })
    }

    /// Return a metadata-only slice along one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorView};
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_slice(vec![2, 2], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.try_slice_axis(1, StridedSliceSpec::reverse())?.get(&[0, 0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice_axis(&self, axis: usize, slice: StridedSliceSpec) -> crate::Result<Self> {
        let slices = slice_axis_specs(
            self.shape().len(),
            axis,
            slice,
            "TypedTensorView::try_slice_axis",
        )?;
        self.try_slice(&slices)
    }

    /// Return a metadata-only dynamic-rank reshape for contiguous column-major views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorView;
    ///
    /// let data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorView::from_slice(vec![2, 2], vec![1, 2], 0, &data)?;
    /// assert_eq!(view.try_reshape(&[4])?.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_reshape(&self, shape: &[usize]) -> crate::Result<TypedTensorView<'a, T, DynRank>> {
        let layout = reshape_layout_dyn(
            &self.layout,
            shape,
            self.buffer.len(),
            "TypedTensorView::try_reshape",
        )?;
        Ok(TypedTensorView {
            buffer: self.buffer.clone(),
            layout,
            placement: self.placement.clone(),
        })
    }
}

/// Mutable borrowed view of typed tensor storage with arbitrary strides.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::TypedTensorViewMut;
///
/// let mut data = [1_i32, 2, 3];
/// let mut view = TypedTensorViewMut::from_slice(vec![3], vec![-1], 2, &mut data)?;
/// *view.get_mut(&[2]).unwrap() = 10;
/// assert_eq!(view.as_read_only().get(&[2]), Some(&10));
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Debug)]
pub struct TypedTensorViewMut<'a, T, R: TensorRank = DynRank> {
    buffer: TensorBufferRefMut<'a, T>,
    layout: TensorLayout<R>,
    placement: Placement,
}

/// Pair of mutable tensor views returned by disjoint multi-slice operations.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut, TypedTensorViewMutPair};
///
/// let mut data = [1_i32, 2, 3, 4];
/// let mut view = TypedTensorViewMut::from_slice(vec![4], vec![1], 0, &mut data)?;
/// let pair: TypedTensorViewMutPair<'_, i32> = view
///     .try_multi_slice_mut(
///         &[StridedSliceSpec::new(0, Some(2), 1)],
///         &[StridedSliceSpec::new(2, Some(4), 1)],
///     )
///     ?
///     .unwrap();
/// assert_eq!(pair.0.shape(), &[2]);
/// assert_eq!(pair.1.shape(), &[2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub type TypedTensorViewMutPair<'a, T, R = DynRank> =
    (TypedTensorViewMut<'a, T, R>, TypedTensorViewMut<'a, T, R>);

impl<'a, T: 'static> TypedTensorViewMut<'a, T, DynRank> {
    /// Create a mutable dynamic-rank view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::from_col_major(&[2, 2], &mut data)?;
    /// assert_eq!(view.strides(), &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_col_major(shape: &[usize], data: &'a mut [T]) -> crate::Result<Self> {
        let layout = TensorLayout::<DynRank>::compact(shape.to_vec().into())
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::from_col_major", err))?;
        Self::from_buffer_ref_mut(
            layout.shape().to_vec(),
            layout.strides().to_vec(),
            layout.offset(),
            TensorBufferRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_col_major",
        )
    }

    /// Create a mutable host view from explicit layout metadata.
    ///
    /// Layouts where distinct logical elements can alias the same physical
    /// element are rejected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// assert!(TypedTensorViewMut::from_slice(vec![2], vec![0], 0, &mut data).is_err());
    /// ```
    pub fn from_slice(
        shape: impl AsRef<[usize]>,
        strides: impl AsRef<[isize]>,
        offset: isize,
        data: &'a mut [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref_mut(
            shape.as_ref().to_vec(),
            strides.as_ref().to_vec(),
            offset,
            TensorBufferRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_slice",
        )
    }
}

impl<'a, T: 'static, R: TensorRank> TypedTensorViewMut<'a, T, R> {
    /// Create a rank-generic mutable host view from explicit layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &mut data)?;
    /// assert_eq!(view.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_slice_ranked(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        data: &'a mut [T],
    ) -> crate::Result<Self> {
        Self::from_buffer_ref_mut(
            shape,
            strides,
            offset,
            TensorBufferRefMut::Host(data),
            default_placement(),
            "TypedTensorViewMut::from_slice_ranked",
        )
    }

    fn from_buffer_ref_mut(
        shape: impl Into<R::Shape>,
        strides: impl Into<R::Strides>,
        offset: isize,
        buffer: TensorBufferRefMut<'a, T>,
        placement: Placement,
        op: &'static str,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            buffer,
            layout,
            placement,
        })
    }

    /// Return the logical shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Return strides in element units.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![-1], 1, &mut data)?;
    /// assert_eq!(view.strides(), &[-1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    /// Return the physical element offset.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 1, &mut data)?;
    /// assert_eq!(view.offset(), 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    /// Return the borrowed host storage backing this view.
    ///
    /// This exposes the entire backing host allocation, not just the logical
    /// slice covered by this view. Use [`TypedTensorViewMut::as_read_only`]
    /// with [`TypedTensorView::as_slice`] when the caller needs the contiguous
    /// logical region instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.host_storage()?, &[1, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn host_storage(&self) -> crate::Result<&[T]> {
        match &self.buffer {
            TensorBufferRefMut::Host(data) => Ok(data),
            TensorBufferRefMut::Backend(_) => Err(crate::Error::backend_failure(
                "TypedTensorViewMut::host_storage",
                "backend buffers cannot expose host storage; download explicitly first",
            )),
        }
    }

    /// Mutably borrow the host storage backing this view.
    ///
    /// This exposes the entire backing host allocation, not just the logical
    /// slice covered by this view. Prefer scalar element accessors when mutating
    /// a logical region; tensor-sized copies belong to an active backend.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// view.host_storage_mut()?[0] = 3;
    /// assert_eq!(view.get(&[0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn host_storage_mut(&mut self) -> crate::Result<&mut [T]> {
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => Ok(data),
            TensorBufferRefMut::Backend(_) => Err(crate::Error::backend_failure(
                "TypedTensorViewMut::host_storage_mut",
                "backend buffers cannot expose mutable host storage; download explicitly first",
            )),
        }
    }

    /// Return the number of logical elements in this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [0_i32; 6];
    /// let view = TypedTensorViewMut::from_slice(vec![2, 3], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.n_elements(), 6);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn n_elements(&self) -> usize {
        // Invariant: public mutable view constructors validate logical element count.
        match checked_view_element_count(self.shape(), "TypedTensorViewMut::n_elements") {
            Ok(n) => n,
            Err(err) => {
                unreachable!("TypedTensorViewMut layout shape is validated at construction: {err}")
            }
        }
    }

    /// Return layout metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert!(view.layout().is_compact_col_major().unwrap());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Return placement metadata for this view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Return the backend allocation for backend integrations.
    #[doc(hidden)]
    pub fn backend_buffer(&self) -> Option<&Arc<dyn BackendBuffer<T>>> {
        match &self.buffer {
            TensorBufferRefMut::Host(_) => None,
            TensorBufferRefMut::Backend(buffer) => Some(buffer),
        }
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let view = TypedTensorViewMut::from_slice(vec![3], vec![-1], 2, &mut data)?;
    /// assert_eq!(view.linear_offset(&[2]), Some(0));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> Option<usize> {
        checked_view_offset(self.shape(), self.strides(), self.offset(), indices)
    }

    /// Compute the physical element offset for a logical index, returning a typed error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let view = TypedTensorViewMut::from_slice([3], [-1], 2, &mut data)?;
    /// assert_eq!(view.layout_linear_offset(&[2])?, 0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        checked_view_offset_result(
            self.shape(),
            self.strides(),
            self.offset(),
            indices,
            "TypedTensorViewMut::layout_linear_offset",
        )
    }

    /// Return whether this mutable view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?;
    /// assert!(view.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        self.layout
            .is_compact_col_major()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::is_col_major_contiguous", err))
    }

    /// Return a compact string summary of this mutable view's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?;
    /// assert!(view.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    /// Assert this mutable view is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?;
    /// view.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TypedTensorViewMut::assert_col_major_contiguous",
        )
    }

    /// Borrow one host element by logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// assert_eq!(view.get(&[1]), Some(&2));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.linear_offset(indices)?;
        match &self.buffer {
            TensorBufferRefMut::Host(data) => data.get(offset),
            TensorBufferRefMut::Backend(_) => None,
        }
    }

    /// Mutably borrow one host element by logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2], vec![1], 0, &mut data)?;
    /// *view.get_mut(&[1]).unwrap() = 20;
    /// assert_eq!(view.get(&[1]), Some(&20));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let offset = self.linear_offset(indices)?;
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => data.get_mut(offset),
            TensorBufferRefMut::Backend(_) => None,
        }
    }

    /// Borrow this mutable view as a read-only view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.as_read_only().get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_read_only(&self) -> TypedTensorView<'_, T, R> {
        let buffer = match &self.buffer {
            TensorBufferRefMut::Host(data) => TensorBufferRef::Host(data),
            TensorBufferRefMut::Backend(buffer) => TensorBufferRef::Backend(Arc::clone(buffer)),
        };
        TypedTensorView {
            buffer,
            layout: self.layout.clone(),
            placement: self.placement.clone(),
        }
    }

    /// Convert this mutable view into a read-only view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32];
    /// let view = TypedTensorViewMut::from_slice(vec![1], vec![1], 0, &mut data)?;
    /// assert_eq!(view.into_read_only().get(&[0]), Some(&1));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_read_only(self) -> TypedTensorView<'a, T, R> {
        let buffer = match self.buffer {
            TensorBufferRefMut::Host(data) => TensorBufferRef::Host(data),
            TensorBufferRefMut::Backend(buffer) => TensorBufferRef::Backend(buffer),
        };
        TypedTensorView {
            buffer,
            layout: self.layout,
            placement: self.placement,
        }
    }

    /// Consume this mutable view and return a metadata-only axis permutation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let view = TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &mut data)?;
    /// let transposed = view.transpose_view([1, 0])?;
    /// assert_eq!(transposed.strides(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn transpose_view(
        self,
        axes: impl AsRef<[usize]>,
    ) -> crate::Result<TypedTensorViewMut<'a, T, R>> {
        let Self {
            buffer,
            layout,
            placement,
        } = self;
        let layout = layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::transpose_view", err))?;
        match buffer {
            TensorBufferRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Host(data),
                layout,
                placement,
            }),
            TensorBufferRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Backend(buffer),
                layout,
                placement,
            }),
        }
    }

    /// Return a mutable metadata-only slice using one [`StridedSliceSpec`] per axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3];
    /// let mut view = TypedTensorViewMut::from_slice(vec![3], vec![1], 0, &mut data)?;
    /// *view.try_slice(&[StridedSliceSpec::reverse()])?.get_mut(&[0]).unwrap() = 30;
    /// assert_eq!(view.get(&[2]), Some(&30));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice(
        &mut self,
        slices: &[StridedSliceSpec],
    ) -> crate::Result<TypedTensorViewMut<'_, T, R>> {
        let specs = core_slice_specs(slices, self.shape(), "TypedTensorViewMut::try_slice")?;
        let layout = self
            .layout
            .slice_view(specs, self.buffer.len())
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_slice", err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_slice", err))?;
        let placement = self.placement.clone();
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Host(data),
                layout,
                placement,
            }),
            TensorBufferRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Backend(Arc::clone(buffer)),
                layout,
                placement,
            }),
        }
    }

    /// Return a mutable metadata-only slice along one axis.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.try_slice_axis(1, StridedSliceSpec::reverse())?.get(&[0, 0]), Some(&3));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_slice_axis(
        &mut self,
        axis: usize,
        slice: StridedSliceSpec,
    ) -> crate::Result<TypedTensorViewMut<'_, T, R>> {
        let slices = slice_axis_specs(
            self.shape().len(),
            axis,
            slice,
            "TypedTensorViewMut::try_slice_axis",
        )?;
        self.try_slice(&slices)
    }

    /// Return two mutable metadata-only slices when their physical ranges are disjoint.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{StridedSliceSpec, TypedTensorViewMut};
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![4], vec![1], 0, &mut data)?;
    /// let (left, right) = view
    ///     .try_multi_slice_mut(
    ///         &[StridedSliceSpec::new(0, Some(2), 1)],
    ///         &[StridedSliceSpec::new(2, Some(4), 1)],
    ///     )
    ///     ?
    ///     .unwrap();
    /// assert_eq!(left.shape(), &[2]);
    /// assert_eq!(right.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_multi_slice_mut(
        &mut self,
        first: &[StridedSliceSpec],
        second: &[StridedSliceSpec],
    ) -> crate::Result<Option<TypedTensorViewMutPair<'_, T, R>>> {
        let op = "TypedTensorViewMut::try_multi_slice_mut";
        let first_specs = core_slice_specs(first, self.shape(), op)?;
        let second_specs = core_slice_specs(second, self.shape(), op)?;
        let buffer_len = self.buffer.len();
        let first_layout = self
            .layout
            .slice_view(first_specs, buffer_len)
            .map_err(|err| tensor_layout_error(op, err))?;
        let second_layout = self
            .layout
            .slice_view(second_specs, buffer_len)
            .map_err(|err| tensor_layout_error(op, err))?;
        first_layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        second_layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;

        match (
            reachable_layout_span(
                first_layout.shape(),
                first_layout.strides(),
                first_layout.offset(),
            )?,
            reachable_layout_span(
                second_layout.shape(),
                second_layout.strides(),
                second_layout.offset(),
            )?,
        ) {
            (Some(first_span), Some(second_span)) => {
                let first_offset = adjusted_view_offset(first_layout.offset(), first_span.0)?;
                let second_offset = adjusted_view_offset(second_layout.offset(), second_span.0)?;
                let (first_data, second_data) = match &mut self.buffer {
                    TensorBufferRefMut::Host(data) => {
                        match split_two_mut_ranges(data, first_span, second_span) {
                            Some(ranges) => ranges,
                            None => return Ok(None),
                        }
                    }
                    TensorBufferRefMut::Backend(_) => return Ok(None),
                };
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    first_offset,
                    first_data,
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    second_offset,
                    second_data,
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
            (None, Some(second_span)) => {
                let second_offset = adjusted_view_offset(second_layout.offset(), second_span.0)?;
                let (_, after_start) = match &mut self.buffer {
                    TensorBufferRefMut::Host(data) => data.split_at_mut(second_span.0),
                    TensorBufferRefMut::Backend(_) => return Ok(None),
                };
                let (second_data, _) = after_start.split_at_mut(second_span.1 - second_span.0 + 1);
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    second_offset,
                    second_data,
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
            (Some(first_span), None) => {
                let first_offset = adjusted_view_offset(first_layout.offset(), first_span.0)?;
                let (_, after_start) = match &mut self.buffer {
                    TensorBufferRefMut::Host(data) => data.split_at_mut(first_span.0),
                    TensorBufferRefMut::Backend(_) => return Ok(None),
                };
                let (first_data, _) = after_start.split_at_mut(first_span.1 - first_span.0 + 1);
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    first_offset,
                    first_data,
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
            (None, None) => {
                let first_view = view_mut_from_layout_and_slice(
                    &first_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                let second_view = view_mut_from_layout_and_slice(
                    &second_layout,
                    0,
                    &mut [],
                    self.placement.clone(),
                )?;
                Ok(Some((first_view, second_view)))
            }
        }
    }

    /// Return a mutable metadata-only dynamic-rank reshape for contiguous views.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensorViewMut;
    ///
    /// let mut data = [1_i32, 2, 3, 4];
    /// let mut view = TypedTensorViewMut::from_slice(vec![2, 2], vec![1, 2], 0, &mut data)?;
    /// assert_eq!(view.try_reshape(&[4])?.shape(), &[4]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_reshape(
        &mut self,
        shape: &[usize],
    ) -> crate::Result<TypedTensorViewMut<'_, T, DynRank>> {
        let layout = reshape_layout_dyn(
            &self.layout,
            shape,
            self.buffer.len(),
            "TypedTensorViewMut::try_reshape",
        )?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_reshape", err))?;
        let placement = self.placement.clone();
        match &mut self.buffer {
            TensorBufferRefMut::Host(data) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Host(data),
                layout,
                placement,
            }),
            TensorBufferRefMut::Backend(buffer) => Ok(TypedTensorViewMut {
                buffer: TensorBufferRefMut::Backend(Arc::clone(buffer)),
                layout,
                placement,
            }),
        }
    }
}

/// Runtime scalar dtype tag.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::DType;
///
/// assert_eq!(DType::F64 as u8, DType::F64 as u8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
    C32,
    C64,
}

/// Sealed trait for scalar types that can be stored in a [`Tensor`].
///
/// This trait is implemented for `f64`, `f32`, `i32`, `i64`, `bool`,
/// [`Complex64`], and [`Complex32`].
///
/// # Examples
///
/// ```
/// use tenferro_tensor::TensorScalar;
///
/// let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0])?;
/// assert_eq!(tensor.as_slice::<f64>()?, [1.0, 2.0].as_slice());
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub trait TensorScalar: Copy + Clone + Send + Sync + 'static + private::Sealed {
    /// Real-valued counterpart of this scalar type.
    type Real: TensorScalar;

    /// The [`DType`] tag corresponding to this scalar type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar};
    ///
    /// assert_eq!(f64::dtype(), DType::F64);
    /// assert_eq!(f32::dtype(), DType::F32);
    /// ```
    fn dtype() -> DType;

    /// Wrap typed column-major data into a [`Tensor`] enum variant.
    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> crate::Result<Tensor>;

    /// Wrap a typed tensor into its dynamic [`Tensor`] enum variant.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar, TypedTensor};
    ///
    /// let typed = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0])?;
    /// let tensor = <f64 as TensorScalar>::typed_tensor_into_tensor(typed);
    /// assert!(matches!(tensor, Tensor::F64(_)));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn typed_tensor_into_tensor(tensor: TypedTensor<Self>) -> Tensor;

    /// Borrow a typed tensor as a dtype-erased [`TensorRead`] view.
    ///
    /// This keeps the typed tensor borrowed instead of copying host data into
    /// a new dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let read = f64::tensor_read(&tensor);
    /// assert_eq!(read.dtype(), DType::F64);
    /// assert_eq!(read.shape(), &[2]);
    /// ```
    fn tensor_read(tensor: &TypedTensor<Self>) -> TensorRead<'_>;

    /// Wrap a typed borrowed view as a dtype-erased [`TensorView`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensorView};
    ///
    /// let data = [1.0_f64];
    /// let view = TypedTensorView::from_col_major(&[1], &data)?;
    /// assert_eq!(f64::tensor_view(view).dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn tensor_view<'a>(view: TypedTensorView<'a, Self>) -> TensorView<'a>;

    /// Wrap a typed mutable borrowed view as a dtype-erased [`TensorViewMut`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensorViewMut};
    ///
    /// let mut data = [1.0_f64, 2.0];
    /// let view = TypedTensorViewMut::from_col_major(&[2], &mut data)?;
    /// let erased = f64::tensor_view_mut(view);
    /// assert_eq!(erased.dtype(), DType::F64);
    /// assert_eq!(erased.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn tensor_view_mut<'a>(view: TypedTensorViewMut<'a, Self>) -> TensorViewMut<'a>;

    /// Mutably borrow a typed tensor as a dtype-erased [`TensorWrite`] view.
    ///
    /// This keeps the typed output borrowed instead of wrapping it in a
    /// temporary dynamic tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorScalar, TypedTensor};
    ///
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![0.0]).unwrap();
    /// let write = f64::tensor_write(&mut tensor);
    /// assert_eq!(write.dtype(), DType::F64);
    /// ```
    fn tensor_write(tensor: &mut TypedTensor<Self>) -> TensorWrite<'_>;

    /// Borrow the host data from a [`Tensor`].
    fn as_slice(tensor: &Tensor) -> crate::Result<&[Self]>;

    /// Mutably borrow the host data from a [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// <f64 as TensorScalar>::as_slice_mut(&mut tensor)?[0] = 3.0;
    ///
    /// assert_eq!(tensor.as_slice::<f64>()?, &[3.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn as_slice_mut(tensor: &mut Tensor) -> crate::Result<&mut [Self]>;

    /// Extract a [`TypedTensor<Self>`] from a dynamic [`Tensor`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let typed = <f64 as TensorScalar>::into_typed(tensor)?;
    ///
    /// assert_eq!(typed.as_slice()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    fn into_typed(tensor: Tensor) -> crate::Result<TypedTensor<Self>>;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for bool {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}

macro_rules! impl_tensor_scalar {
    ($ty:ty, $real:ty, $dtype:ident, $variant:ident) => {
        impl TensorScalar for $ty {
            type Real = $real;

            fn dtype() -> DType {
                DType::$dtype
            }

            fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> crate::Result<Tensor> {
                TypedTensor::from_vec_col_major(shape, data).map(Tensor::$variant)
            }

            fn typed_tensor_into_tensor(tensor: TypedTensor<Self>) -> Tensor {
                Tensor::$variant(tensor)
            }

            fn tensor_read(tensor: &TypedTensor<Self>) -> TensorRead<'_> {
                TensorRead::from_view(TensorView::$variant(tensor.as_view()))
            }

            fn tensor_view<'a>(view: TypedTensorView<'a, Self>) -> TensorView<'a> {
                TensorView::$variant(view)
            }

            fn tensor_view_mut<'a>(view: TypedTensorViewMut<'a, Self>) -> TensorViewMut<'a> {
                TensorViewMut::$variant(view)
            }

            fn tensor_write(tensor: &mut TypedTensor<Self>) -> TensorWrite<'_> {
                TensorWrite::from_view(TensorViewMut::$variant(tensor.as_view_mut()))
            }

            fn as_slice(tensor: &Tensor) -> crate::Result<&[Self]> {
                let actual = tensor.dtype();
                match tensor {
                    Tensor::$variant(t) => t.host_data(),
                    _ => Err(crate::Error::DTypeMismatch {
                        op: "Tensor::as_slice",
                        lhs: Self::dtype(),
                        rhs: actual,
                    }),
                }
            }

            fn as_slice_mut(tensor: &mut Tensor) -> crate::Result<&mut [Self]> {
                let actual = tensor.dtype();
                match tensor {
                    Tensor::$variant(t) => t.host_data_mut(),
                    _ => Err(crate::Error::DTypeMismatch {
                        op: "Tensor::as_slice_mut",
                        lhs: Self::dtype(),
                        rhs: actual,
                    }),
                }
            }

            fn into_typed(tensor: Tensor) -> crate::Result<TypedTensor<Self>> {
                let actual = tensor.dtype();
                match tensor {
                    Tensor::$variant(inner) => Ok(inner),
                    _ => Err(crate::Error::DTypeMismatch {
                        op: "TensorScalar::into_typed",
                        lhs: Self::dtype(),
                        rhs: actual,
                    }),
                }
            }
        }
    };
}

impl_tensor_scalar!(f64, f64, F64, F64);
impl_tensor_scalar!(f32, f32, F32, F32);
impl_tensor_scalar!(i64, i64, I64, I64);
impl_tensor_scalar!(i32, i32, I32, I32);
impl_tensor_scalar!(bool, bool, Bool, Bool);
impl_tensor_scalar!(Complex64, f64, C64, C64);
impl_tensor_scalar!(Complex32, f32, C32, C32);

/// Dynamic tensor enum over the supported scalar types.
///
/// The enum keeps dtype dynamic and rank dynamic. Use
/// [`TypedTensor<T, R>`](TypedTensor) directly when the scalar type or rank
/// should be represented in Rust's type system.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
/// assert_eq!(t.shape(), &[2]);
///
/// let erased = Tensor::from_vec_col_major(vec![1, 2], vec![1.0_f64, 2.0]).unwrap();
/// assert_eq!(erased.shape().len(), 2);
/// ```
#[derive(Clone, Debug)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    I32(TypedTensor<i32>),
    I64(TypedTensor<i64>),
    Bool(TypedTensor<bool>),
    C32(TypedTensor<Complex<f32>>),
    C64(TypedTensor<Complex<f64>>),
}

/// Dynamic read-only borrowed tensor view.
///
/// `TensorView` keeps dtype erased while borrowing typed view metadata and
/// storage. Use [`TypedTensorView`] directly when the scalar type is statically
/// known.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, TensorView, TypedTensorView};
///
/// let data = [1_i32, 2, 3, 4];
/// let typed = TypedTensorView::from_slice([2, 2], [1, 2], 0, &data)?;
/// let view = TensorView::I32(typed);
///
/// assert_eq!(view.dtype(), DType::I32);
/// assert_eq!(view.shape(), &[2, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub enum TensorView<'a> {
    F32(TypedTensorView<'a, f32>),
    F64(TypedTensorView<'a, f64>),
    I32(TypedTensorView<'a, i32>),
    I64(TypedTensorView<'a, i64>),
    Bool(TypedTensorView<'a, bool>),
    C32(TypedTensorView<'a, Complex<f32>>),
    C64(TypedTensorView<'a, Complex<f64>>),
}

/// Dynamic mutable borrowed tensor view.
///
/// `TensorViewMut` is the mutable counterpart to [`TensorView`]. It keeps the
/// dtype erased while preserving the typed mutable view's shape, strides, and
/// offset metadata.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, TensorViewMut, TypedTensorViewMut};
///
/// let mut data = [1.0_f64, 2.0];
/// let view = TensorViewMut::F64(TypedTensorViewMut::from_slice([2], [1], 0, &mut data)?);
/// assert_eq!(view.dtype(), DType::F64);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum TensorViewMut<'a> {
    F32(TypedTensorViewMut<'a, f32>),
    F64(TypedTensorViewMut<'a, f64>),
    I32(TypedTensorViewMut<'a, i32>),
    I64(TypedTensorViewMut<'a, i64>),
    Bool(TypedTensorViewMut<'a, bool>),
    C32(TypedTensorViewMut<'a, Complex<f32>>),
    C64(TypedTensorViewMut<'a, Complex<f64>>),
}

/// Read-only tensor input accepted by synchronous eager kernels.
///
/// `TensorRead` lets kernels accept either an owned tensor reference or a
/// borrowed [`TensorView`] without forcing callers to materialize first.
/// The `View` variant preserves arbitrary strides and offsets, so kernels that
/// support strided reads can consume transposes, slices, and broadcasts directly.
///
/// `TensorRead` is intentionally borrowed. It is an input-dispatch type, not an
/// owned lazy tensor value. APIs that need to store a lazy layout result should
/// keep an owned base tensor plus layout metadata, then expose a `TensorRead`
/// only for the duration of kernel dispatch.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TensorRead};
///
/// let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let read = TensorRead::from_tensor(&tensor);
///
/// assert_eq!(read.dtype(), DType::F64);
/// assert_eq!(read.shape(), &[2]);
/// ```
// Keep borrowed views inline to avoid allocation on read-only tensor dispatch paths.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum TensorRead<'a> {
    Tensor(&'a Tensor),
    View(TensorView<'a>),
}

/// Mutable typed tensor output accepted by synchronous eager kernels.
///
/// `TypedTensorWrite` is the typed counterpart to [`TensorWrite`]. It accepts
/// either an owned compact [`TypedTensor`] or an arbitrary-strided mutable
/// [`TypedTensorViewMut`] without erasing the scalar type at the public API
/// boundary.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{TypedTensorViewMut, TypedTensorWrite};
///
/// let mut data = [0.0_f64, 1.0, 0.0, 2.0];
/// let view = TypedTensorViewMut::from_slice([2], [2], 1, &mut data)?;
/// let write = TypedTensorWrite::from_view(view).into_tensor_write();
/// assert_eq!(write.shape(), &[2]);
/// assert_eq!(write.strides()?, [2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum TypedTensorWrite<'a, T> {
    /// An owned compact typed tensor borrowed mutably for the write.
    Tensor(&'a mut TypedTensor<T>),
    /// An arbitrary-strided mutable typed tensor view.
    View(TypedTensorViewMut<'a, T>),
}

impl<'a, T> TypedTensorWrite<'a, T> {
    /// Create a writable target from an owned typed tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TypedTensor, TypedTensorWrite};
    ///
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![0.0])?;
    /// let write = TypedTensorWrite::from_tensor(&mut tensor);
    /// assert!(matches!(write, TypedTensorWrite::Tensor(_)));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_tensor(tensor: &'a mut TypedTensor<T>) -> Self {
        Self::Tensor(tensor)
    }

    /// Create a writable target from a mutable typed tensor view.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{TypedTensorViewMut, TypedTensorWrite};
    ///
    /// let mut data = [0.0_f64, 1.0];
    /// let view = TypedTensorViewMut::from_col_major(&[2], &mut data)?;
    /// let write = TypedTensorWrite::from_view(view);
    /// assert!(matches!(write, TypedTensorWrite::View(_)));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_view(view: TypedTensorViewMut<'a, T>) -> Self {
        Self::View(view)
    }
}

impl<'a, T: TensorScalar> TypedTensorWrite<'a, T> {
    /// Erase the scalar type while preserving the output layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TypedTensor, TypedTensorWrite};
    ///
    /// let mut tensor = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0; 2])?;
    /// let write = TypedTensorWrite::from_tensor(&mut tensor).into_tensor_write();
    /// assert_eq!(write.dtype(), DType::F64);
    /// assert_eq!(write.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn into_tensor_write(self) -> TensorWrite<'a> {
        match self {
            Self::Tensor(tensor) => T::tensor_write(tensor),
            Self::View(view) => TensorWrite::from_view(T::tensor_view_mut(view)),
        }
    }
}

impl<'a, T> From<&'a mut TypedTensor<T>> for TypedTensorWrite<'a, T> {
    fn from(tensor: &'a mut TypedTensor<T>) -> Self {
        Self::from_tensor(tensor)
    }
}

impl<'a, T> From<TypedTensorViewMut<'a, T>> for TypedTensorWrite<'a, T> {
    fn from(view: TypedTensorViewMut<'a, T>) -> Self {
        Self::from_view(view)
    }
}

/// Mutable tensor output accepted by synchronous eager kernels.
///
/// `TensorWrite` mirrors [`TensorRead`] for output dispatch: it can target an
/// owned compact [`Tensor`] or a borrowed mutable [`TensorViewMut`]. The target
/// is never resized.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TensorWrite};
///
/// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![0.0_f64])?;
/// let write = TensorWrite::from_tensor(&mut tensor);
/// assert_eq!(write.shape(), &[1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum TensorWrite<'a> {
    Tensor(&'a mut Tensor),
    View(TensorViewMut<'a>),
}

/// Owned lazy tensor view over a shared base tensor.
///
/// This stores only ownership of the base allocation plus logical layout
/// metadata. Borrow it as [`TensorRead`] and materialize it through an active
/// backend session when compact storage is required.
#[derive(Clone, Debug)]
pub struct TensorOwnedView {
    base: Arc<Tensor>,
    layout: TensorLayout<DynRank>,
}

/// Owned tensor value that can be compact or a lazy view.
///
/// `TensorValue` is the owned counterpart to [`TensorRead`]. It is suitable for
/// storing eager results that should remain lazy until an operation actually
/// requires compact materialized storage.
#[derive(Clone, Debug)]
pub enum TensorValue {
    Tensor(Arc<Tensor>),
    View(TensorOwnedView),
}

impl TensorOwnedView {
    /// Create an owned view preserving the base tensor's current layout.
    pub fn from_tensor(base: Arc<Tensor>) -> Self {
        let layout = tensor_layout(base.as_ref());
        Self { base, layout }
    }

    /// Create an owned view with explicit layout metadata.
    pub fn from_parts(
        base: Arc<Tensor>,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> crate::Result<Self> {
        let layout = TensorLayout::from_parts(
            shape.into(),
            strides.into(),
            offset,
            tensor_buffer_len(&base),
        )
        .map_err(|err| tensor_layout_error("TensorOwnedView::from_parts", err))?;
        Ok(Self { base, layout })
    }

    pub fn dtype(&self) -> DType {
        self.base.dtype()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    pub fn offset(&self) -> isize {
        self.layout.offset()
    }

    pub fn tensor_view(&self) -> TensorView<'_> {
        tensor_view_with_layout(self.base.as_ref(), self.layout.clone())
    }

    pub fn tensor_read(&self) -> TensorRead<'_> {
        TensorRead::from_view(self.tensor_view())
    }

    pub fn transpose_view(&self, axes: impl AsRef<[usize]>) -> crate::Result<Self> {
        let layout = self
            .layout
            .transpose_view(axes)
            .map_err(|err| tensor_layout_error("TensorOwnedView::transpose_view", err))?;
        Ok(Self {
            base: Arc::clone(&self.base),
            layout,
        })
    }

    pub fn reshape_view(&self, shape: &[usize]) -> crate::Result<Self> {
        let layout = reshape_layout_dyn(
            &self.layout,
            shape,
            tensor_buffer_len(&self.base),
            "TensorOwnedView::reshape_view",
        )?;
        Ok(Self {
            base: Arc::clone(&self.base),
            layout,
        })
    }

    pub fn slice_view(&self, config: &SliceConfig) -> crate::Result<Self> {
        let op = "TensorOwnedView::slice_view";
        if config.starts.len() != self.shape().len() {
            return Err(crate::Error::RankMismatch {
                op,
                expected: self.shape().len(),
                actual: config.starts.len(),
            });
        }
        if config.limits.len() != self.shape().len() {
            return Err(crate::Error::RankMismatch {
                op,
                expected: self.shape().len(),
                actual: config.limits.len(),
            });
        }
        if config.strides.len() != self.shape().len() {
            return Err(crate::Error::RankMismatch {
                op,
                expected: self.shape().len(),
                actual: config.strides.len(),
            });
        }

        let mut slices = Vec::with_capacity(self.shape().len());
        for ((&start, &limit), &stride) in config
            .starts
            .iter()
            .zip(config.limits.iter())
            .zip(config.strides.iter())
        {
            let start = isize::try_from(start).map_err(|_| crate::Error::InvalidConfig {
                op,
                message: format!("slice start {start} does not fit in isize"),
            })?;
            let limit = isize::try_from(limit).map_err(|_| crate::Error::InvalidConfig {
                op,
                message: format!("slice limit {limit} does not fit in isize"),
            })?;
            let stride = isize::try_from(stride).map_err(|_| crate::Error::InvalidConfig {
                op,
                message: format!("slice stride {stride} does not fit in isize"),
            })?;
            slices.push(StridedSliceSpec::new(start, Some(limit), stride));
        }

        let specs = core_slice_specs(&slices, self.shape(), op)?;
        let layout = self
            .layout
            .slice_view(&specs, tensor_buffer_len(&self.base))
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(Self {
            base: Arc::clone(&self.base),
            layout,
        })
    }

    pub fn broadcast_in_dim_view(&self, shape: &[usize], dims: &[usize]) -> crate::Result<Self> {
        let layout = self
            .layout
            .broadcast_in_dim_view::<DynRank>(
                shape.to_vec().into(),
                dims,
                tensor_buffer_len(&self.base),
            )
            .map_err(|err| tensor_layout_error("TensorOwnedView::broadcast_in_dim_view", err))?;
        Ok(Self {
            base: Arc::clone(&self.base),
            layout,
        })
    }
}

impl TensorValue {
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self::Tensor(Arc::new(tensor))
    }

    pub fn from_tensor_arc(tensor: Arc<Tensor>) -> Self {
        Self::Tensor(tensor)
    }

    pub fn as_tensor_arc(&self) -> Option<&Arc<Tensor>> {
        match self {
            Self::Tensor(tensor) => Some(tensor),
            Self::View(_) => None,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    pub fn tensor_read(&self) -> TensorRead<'_> {
        match self {
            Self::Tensor(tensor) => TensorRead::from_tensor(tensor.as_ref()),
            Self::View(view) => view.tensor_read(),
        }
    }

    pub fn transpose_view(&self, axes: impl AsRef<[usize]>) -> crate::Result<Self> {
        match self {
            Self::Tensor(tensor) => TensorOwnedView::from_tensor(Arc::clone(tensor))
                .transpose_view(axes)
                .map(Self::View),
            Self::View(view) => view.transpose_view(axes).map(Self::View),
        }
    }

    pub fn reshape_view(&self, shape: &[usize]) -> crate::Result<Self> {
        match self {
            Self::Tensor(tensor) => TensorOwnedView::from_tensor(Arc::clone(tensor))
                .reshape_view(shape)
                .map(Self::View),
            Self::View(view) => view.reshape_view(shape).map(Self::View),
        }
    }

    pub fn slice_view(&self, config: &SliceConfig) -> crate::Result<Self> {
        match self {
            Self::Tensor(tensor) => TensorOwnedView::from_tensor(Arc::clone(tensor))
                .slice_view(config)
                .map(Self::View),
            Self::View(view) => view.slice_view(config).map(Self::View),
        }
    }

    pub fn broadcast_in_dim_view(&self, shape: &[usize], dims: &[usize]) -> crate::Result<Self> {
        match self {
            Self::Tensor(tensor) => TensorOwnedView::from_tensor(Arc::clone(tensor))
                .broadcast_in_dim_view(shape, dims)
                .map(Self::View),
            Self::View(view) => view.broadcast_in_dim_view(shape, dims).map(Self::View),
        }
    }
}

fn tensor_layout(tensor: &Tensor) -> TensorLayout<DynRank> {
    match tensor {
        Tensor::F32(tensor) => tensor.layout.clone(),
        Tensor::F64(tensor) => tensor.layout.clone(),
        Tensor::I32(tensor) => tensor.layout.clone(),
        Tensor::I64(tensor) => tensor.layout.clone(),
        Tensor::Bool(tensor) => tensor.layout.clone(),
        Tensor::C32(tensor) => tensor.layout.clone(),
        Tensor::C64(tensor) => tensor.layout.clone(),
    }
}

fn tensor_buffer_len(tensor: &Tensor) -> usize {
    match tensor {
        Tensor::F32(tensor) => buffer_len(&tensor.buffer),
        Tensor::F64(tensor) => buffer_len(&tensor.buffer),
        Tensor::I32(tensor) => buffer_len(&tensor.buffer),
        Tensor::I64(tensor) => buffer_len(&tensor.buffer),
        Tensor::Bool(tensor) => buffer_len(&tensor.buffer),
        Tensor::C32(tensor) => buffer_len(&tensor.buffer),
        Tensor::C64(tensor) => buffer_len(&tensor.buffer),
    }
}

fn buffer_len<T: 'static>(buffer: &Buffer<T>) -> usize {
    match buffer {
        Buffer::Host(data) => data.len(),
        Buffer::Backend(buffer) => buffer.len(),
    }
}

fn tensor_view_with_layout(tensor: &Tensor, layout: TensorLayout<DynRank>) -> TensorView<'_> {
    match tensor {
        Tensor::F32(tensor) => TensorView::F32(typed_view_with_layout(tensor, layout)),
        Tensor::F64(tensor) => TensorView::F64(typed_view_with_layout(tensor, layout)),
        Tensor::I32(tensor) => TensorView::I32(typed_view_with_layout(tensor, layout)),
        Tensor::I64(tensor) => TensorView::I64(typed_view_with_layout(tensor, layout)),
        Tensor::Bool(tensor) => TensorView::Bool(typed_view_with_layout(tensor, layout)),
        Tensor::C32(tensor) => TensorView::C32(typed_view_with_layout(tensor, layout)),
        Tensor::C64(tensor) => TensorView::C64(typed_view_with_layout(tensor, layout)),
    }
}

fn typed_view_with_layout<T: 'static>(
    tensor: &TypedTensor<T>,
    layout: TensorLayout<DynRank>,
) -> TypedTensorView<'_, T> {
    let buffer = match &tensor.buffer {
        Buffer::Host(data) => TensorBufferRef::Host(data),
        Buffer::Backend(buffer) => TensorBufferRef::Backend(Arc::clone(buffer)),
    };
    TypedTensorView {
        buffer,
        layout,
        placement: tensor.placement.clone(),
    }
}

/// Wrap an `f64` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<f64>> for Tensor {
    fn from(t: TypedTensor<f64>) -> Self {
        Tensor::F64(t)
    }
}

/// Wrap an `f32` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<f32>> for Tensor {
    fn from(t: TypedTensor<f32>) -> Self {
        Tensor::F32(t)
    }
}

/// Wrap an `i64` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::I64);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<i64>> for Tensor {
    fn from(t: TypedTensor<i64>) -> Self {
        Tensor::I64(t)
    }
}

/// Wrap an `i32` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::I32);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<i32>> for Tensor {
    fn from(t: TypedTensor<i32>) -> Self {
        Tensor::I32(t)
    }
}

/// Wrap a `bool` [`TypedTensor`] into the corresponding [`Tensor`] variant.
///
/// # Examples
///
/// ```
/// use tenferro_tensor::{DType, Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.dtype(), DType::Bool);
/// assert_eq!(tensor.shape(), &[2]);
/// ```
impl From<TypedTensor<bool>> for Tensor {
    fn from(t: TypedTensor<bool>) -> Self {
        Tensor::Bool(t)
    }
}

/// Wrap a [`Complex64`] [`TypedTensor`] into the corresponding [`Tensor`]
/// variant.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(
///     vec![1],
///     vec![Complex64::new(1.0, 2.0)],
/// ).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[1]);
/// ```
impl From<TypedTensor<Complex<f64>>> for Tensor {
    fn from(t: TypedTensor<Complex<f64>>) -> Self {
        Tensor::C64(t)
    }
}

/// Wrap a [`Complex32`] [`TypedTensor`] into the corresponding [`Tensor`]
/// variant.
///
/// # Examples
///
/// ```
/// use num_complex::Complex32;
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let typed = TypedTensor::from_vec_col_major(
///     vec![1],
///     vec![Complex32::new(1.0, 2.0)],
/// ).unwrap();
/// let tensor: Tensor = typed.into();
/// assert_eq!(tensor.shape(), &[1]);
/// ```
impl From<TypedTensor<Complex<f32>>> for Tensor {
    fn from(t: TypedTensor<Complex<f32>>) -> Self {
        Tensor::C32(t)
    }
}

impl<'a> TensorView<'a> {
    /// Create a dynamic `f32` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1.0_f32, 2.0];
    /// let view = TensorView::f32(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::F32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn f32(shape: &'a [usize], data: &'a [f32]) -> crate::Result<Self> {
        Ok(Self::F32(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `f64` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1.0_f64, 2.0];
    /// let view = TensorView::f64(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn f64(shape: &'a [usize], data: &'a [f64]) -> crate::Result<Self> {
        Ok(Self::F64(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `i64` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1_i64, 2];
    /// let view = TensorView::i64(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::I64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn i64(shape: &'a [usize], data: &'a [i64]) -> crate::Result<Self> {
        Ok(Self::I64(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `i32` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [1_i32, 2];
    /// let view = TensorView::i32(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::I32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn i32(shape: &'a [usize], data: &'a [i32]) -> crate::Result<Self> {
        Ok(Self::I32(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `bool` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [true, false];
    /// let view = TensorView::bool(&[2], &data)?;
    /// assert_eq!(view.dtype(), DType::Bool);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn bool(shape: &'a [usize], data: &'a [bool]) -> crate::Result<Self> {
        Ok(Self::Bool(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `Complex32` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex32;
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [Complex32::new(1.0, 2.0)];
    /// let view = TensorView::c32(&[1], &data)?;
    /// assert_eq!(view.dtype(), DType::C32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn c32(shape: &'a [usize], data: &'a [Complex32]) -> crate::Result<Self> {
        Ok(Self::C32(TypedTensorView::from_col_major(shape, data)?))
    }

    /// Create a dynamic `Complex64` view over compact column-major host data.
    ///
    /// # Examples
    ///
    /// ```
    /// use num_complex::Complex64;
    /// use tenferro_tensor::{DType, TensorView};
    ///
    /// let data = [Complex64::new(1.0, 2.0)];
    /// let view = TensorView::c64(&[1], &data)?;
    /// assert_eq!(view.dtype(), DType::C64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn c64(shape: &'a [usize], data: &'a [Complex64]) -> crate::Result<Self> {
        Ok(Self::C64(TypedTensorView::from_col_major(shape, data)?))
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
            Self::C32(t) => t.shape(),
            Self::C64(t) => t.shape(),
        }
    }

    /// Return strides in element units.
    pub fn strides(&self) -> &[isize] {
        match self {
            Self::F32(t) => t.strides(),
            Self::F64(t) => t.strides(),
            Self::I32(t) => t.strides(),
            Self::I64(t) => t.strides(),
            Self::Bool(t) => t.strides(),
            Self::C32(t) => t.strides(),
            Self::C64(t) => t.strides(),
        }
    }

    /// Return the physical element offset.
    pub fn offset(&self) -> isize {
        match self {
            Self::F32(t) => t.offset(),
            Self::F64(t) => t.offset(),
            Self::I32(t) => t.offset(),
            Self::I64(t) => t.offset(),
            Self::Bool(t) => t.offset(),
            Self::C32(t) => t.offset(),
            Self::C64(t) => t.offset(),
        }
    }

    /// Compute the physical element offset for a logical index.
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::F32(t) => t.layout_linear_offset(indices),
            Self::F64(t) => t.layout_linear_offset(indices),
            Self::I32(t) => t.layout_linear_offset(indices),
            Self::I64(t) => t.layout_linear_offset(indices),
            Self::Bool(t) => t.layout_linear_offset(indices),
            Self::C32(t) => t.layout_linear_offset(indices),
            Self::C64(t) => t.layout_linear_offset(indices),
        }
    }

    /// Return whether this view is compact column-major.
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::F32(t) => t.is_col_major_contiguous(),
            Self::F64(t) => t.is_col_major_contiguous(),
            Self::I32(t) => t.is_col_major_contiguous(),
            Self::I64(t) => t.is_col_major_contiguous(),
            Self::Bool(t) => t.is_col_major_contiguous(),
            Self::C32(t) => t.is_col_major_contiguous(),
            Self::C64(t) => t.is_col_major_contiguous(),
        }
    }

    /// Return a compact string summary of this view's layout metadata.
    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    /// Assert this view is compact column-major.
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TensorView::assert_col_major_contiguous",
        )
    }
}

impl<'a> TensorViewMut<'a> {
    /// Create a dynamic `f64` mutable view over compact column-major host data.
    pub fn f64(shape: &'a [usize], data: &'a mut [f64]) -> crate::Result<Self> {
        Ok(Self::F64(TypedTensorViewMut::from_col_major(shape, data)?))
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
            Self::C32(t) => t.shape(),
            Self::C64(t) => t.shape(),
        }
    }

    pub fn strides(&self) -> &[isize] {
        match self {
            Self::F32(t) => t.strides(),
            Self::F64(t) => t.strides(),
            Self::I32(t) => t.strides(),
            Self::I64(t) => t.strides(),
            Self::Bool(t) => t.strides(),
            Self::C32(t) => t.strides(),
            Self::C64(t) => t.strides(),
        }
    }

    pub fn offset(&self) -> isize {
        match self {
            Self::F32(t) => t.offset(),
            Self::F64(t) => t.offset(),
            Self::I32(t) => t.offset(),
            Self::I64(t) => t.offset(),
            Self::Bool(t) => t.offset(),
            Self::C32(t) => t.offset(),
            Self::C64(t) => t.offset(),
        }
    }

    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::F32(t) => t.layout_linear_offset(indices),
            Self::F64(t) => t.layout_linear_offset(indices),
            Self::I32(t) => t.layout_linear_offset(indices),
            Self::I64(t) => t.layout_linear_offset(indices),
            Self::Bool(t) => t.layout_linear_offset(indices),
            Self::C32(t) => t.layout_linear_offset(indices),
            Self::C64(t) => t.layout_linear_offset(indices),
        }
    }

    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::F32(t) => t.is_col_major_contiguous(),
            Self::F64(t) => t.is_col_major_contiguous(),
            Self::I32(t) => t.is_col_major_contiguous(),
            Self::I64(t) => t.is_col_major_contiguous(),
            Self::Bool(t) => t.is_col_major_contiguous(),
            Self::C32(t) => t.is_col_major_contiguous(),
            Self::C64(t) => t.is_col_major_contiguous(),
        }
    }

    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.strides(), self.offset())
    }

    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.strides(),
            self.offset(),
            "TensorViewMut::assert_col_major_contiguous",
        )
    }

    pub fn as_read_only(&self) -> TensorView<'_> {
        match self {
            Self::F32(t) => TensorView::F32(t.as_read_only()),
            Self::F64(t) => TensorView::F64(t.as_read_only()),
            Self::I32(t) => TensorView::I32(t.as_read_only()),
            Self::I64(t) => TensorView::I64(t.as_read_only()),
            Self::Bool(t) => TensorView::Bool(t.as_read_only()),
            Self::C32(t) => TensorView::C32(t.as_read_only()),
            Self::C64(t) => TensorView::C64(t.as_read_only()),
        }
    }
}

impl<'a> TensorRead<'a> {
    pub fn from_tensor(tensor: &'a Tensor) -> Self {
        Self::Tensor(tensor)
    }

    pub fn from_view(view: TensorView<'a>) -> Self {
        Self::View(view)
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    pub fn strides(&self) -> crate::Result<Vec<isize>> {
        match self {
            Self::Tensor(tensor) => col_major_strides(tensor.shape()),
            Self::View(view) => Ok(view.strides().to_vec()),
        }
    }

    pub fn offset(&self) -> isize {
        match self {
            Self::Tensor(_) => 0,
            Self::View(view) => view.offset(),
        }
    }

    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::Tensor(tensor) => tensor.layout_linear_offset(indices),
            Self::View(view) => view.layout_linear_offset(indices),
        }
    }

    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::Tensor(tensor) => tensor.is_col_major_contiguous(),
            Self::View(view) => view.is_col_major_contiguous(),
        }
    }

    pub fn layout_summary(&self) -> String {
        let strides = match self.strides() {
            Ok(strides) => strides,
            Err(err) => return format!("layout unavailable: {err}"),
        };
        layout_summary(self.shape(), &strides, self.offset())
    }

    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        let strides = self.strides()?;
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            &strides,
            self.offset(),
            "TensorRead::assert_col_major_contiguous",
        )
    }

    pub fn as_tensor(&self) -> Option<&'a Tensor> {
        match self {
            Self::Tensor(tensor) => Some(*tensor),
            Self::View(_) => None,
        }
    }
}

impl<'a> TensorWrite<'a> {
    pub fn from_tensor(tensor: &'a mut Tensor) -> Self {
        Self::Tensor(tensor)
    }

    pub fn from_view(view: TensorViewMut<'a>) -> Self {
        Self::View(view)
    }

    /// Borrow this writable target as a read-only tensor input.
    ///
    /// This is useful for explicit read-modify-write kernels such as
    /// accumulation updates. The returned view borrows through `&self`, so it
    /// cannot outlive the current read-only borrow of the writable target.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TensorWrite};
    ///
    /// let mut tensor = Tensor::from_vec_col_major(vec![1], vec![2.0_f64])?;
    /// let write = TensorWrite::from_tensor(&mut tensor);
    /// let read = write.as_read();
    /// assert_eq!(read.dtype(), DType::F64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_read(&self) -> TensorRead<'_> {
        match self {
            Self::Tensor(tensor) => TensorRead::from_tensor(tensor),
            Self::View(view) => TensorRead::from_view(view.as_read_only()),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    pub fn strides(&self) -> crate::Result<Vec<isize>> {
        match self {
            Self::Tensor(tensor) => col_major_strides(tensor.shape()),
            Self::View(view) => Ok(view.strides().to_vec()),
        }
    }

    pub fn offset(&self) -> isize {
        match self {
            Self::Tensor(_) => 0,
            Self::View(view) => view.offset(),
        }
    }

    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Self::Tensor(tensor) => tensor.layout_linear_offset(indices),
            Self::View(view) => view.layout_linear_offset(indices),
        }
    }

    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Self::Tensor(tensor) => tensor.is_col_major_contiguous(),
            Self::View(view) => view.is_col_major_contiguous(),
        }
    }

    pub fn layout_summary(&self) -> String {
        let strides = match self.strides() {
            Ok(strides) => strides,
            Err(err) => return format!("layout unavailable: {err}"),
        };
        layout_summary(self.shape(), &strides, self.offset())
    }

    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        let strides = self.strides()?;
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            &strides,
            self.offset(),
            "TensorWrite::assert_col_major_contiguous",
        )
    }
}

/// Column-major strides derived from a shape.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::col_major_strides;
///
/// assert_eq!(col_major_strides(&[2, 3])?, vec![1, 2]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
pub fn col_major_strides(shape: &[usize]) -> crate::Result<Vec<isize>> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1isize;
    for &extent in shape {
        strides.push(stride);
        let extent = isize::try_from(extent).map_err(|_| crate::Error::InvalidConfig {
            op: "col_major_strides",
            message: format!("shape extent {extent} does not fit in isize"),
        })?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "col_major_strides",
                message: format!("column-major stride overflows for shape {shape:?}"),
            })?;
    }
    Ok(strides)
}

fn try_linear_offset_for_shape(
    shape: &[usize],
    indices: &[usize],
    op: &'static str,
) -> crate::Result<usize> {
    if indices.len() != shape.len() {
        return Err(crate::Error::RankMismatch {
            op,
            expected: shape.len(),
            actual: indices.len(),
        });
    }
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (axis, (&idx, &extent)) in indices.iter().zip(shape).enumerate() {
        if idx >= extent {
            return Err(crate::Error::InvalidConfig {
                op,
                message: format!("index {idx} out of bounds for axis {axis} extent {extent}"),
            });
        }
        offset = offset
            .checked_add(
                idx.checked_mul(stride)
                    .ok_or_else(|| crate::Error::InvalidConfig {
                        op,
                        message: "linear offset multiply overflows".to_string(),
                    })?,
            )
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "linear offset add overflows".to_string(),
            })?;
        stride = stride
            .checked_mul(extent)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "linear offset stride overflows".to_string(),
            })?;
    }
    Ok(offset)
}

fn checked_view_offset_result(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    indices: &[usize],
    op: &'static str,
) -> crate::Result<usize> {
    if indices.len() != shape.len() {
        return Err(crate::Error::RankMismatch {
            op,
            expected: shape.len(),
            actual: indices.len(),
        });
    }
    for (axis, (&index, &extent)) in indices.iter().zip(shape).enumerate() {
        if index >= extent {
            return Err(crate::Error::InvalidConfig {
                op,
                message: format!("index {index} out of bounds for axis {axis} extent {extent}"),
            });
        }
    }
    checked_view_offset(shape, strides, base_offset, indices).ok_or_else(|| {
        crate::Error::InvalidConfig {
            op,
            message: format!(
                "layout offset overflow for shape={shape:?} strides={strides:?} offset={base_offset} indices={indices:?}"
            ),
        }
    })
}

fn layout_summary(shape: &[usize], strides: &[isize], offset: isize) -> String {
    format!("shape={shape:?} strides={strides:?} offset={offset}")
}

fn assert_layout_col_major_contiguous(
    is_contiguous: bool,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
    op: &'static str,
) -> crate::Result<()> {
    if is_contiguous {
        Ok(())
    } else {
        Err(crate::Error::InvalidConfig {
            op,
            message: format!(
                "expected compact column-major layout, got {}",
                layout_summary(shape, strides, offset)
            ),
        })
    }
}

fn try_shape_product(shape: &[usize], op: &'static str) -> crate::Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("shape product overflows for shape {shape:?}"),
            })
    })
}

fn try_checked_shape_len(shape: &[usize], data_len: usize, op: &'static str) -> crate::Result<()> {
    let n = try_shape_product(shape, op)?;
    if data_len != n {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!("data length {data_len} does not match shape product {n}"),
        });
    }
    Ok(())
}

fn try_compact_layout<R: TensorRank>(
    shape: impl Into<R::Shape>,
    op: &'static str,
) -> crate::Result<TensorLayout<R>> {
    TensorLayout::compact(shape.into()).map_err(|err| tensor_layout_error(op, err))
}

fn tensor_layout_error(op: &'static str, err: tenferro_tensor_core::Error) -> crate::Error {
    match err {
        tenferro_tensor_core::Error::RankMismatch { expected, actual } => {
            crate::Error::RankMismatch {
                op,
                expected,
                actual,
            }
        }
        tenferro_tensor_core::Error::AxisOutOfBounds { axis, rank } => {
            crate::Error::AxisOutOfBounds { op, axis, rank }
        }
        tenferro_tensor_core::Error::DuplicateAxis { axis } => crate::Error::DuplicateAxis {
            op,
            axis,
            role: "permutation",
        },
        tenferro_tensor_core::Error::InvalidPermutationLength { expected, actual } => {
            crate::Error::RankMismatch {
                op,
                expected,
                actual,
            }
        }
        other => crate::Error::InvalidConfig {
            op,
            message: other.to_string(),
        },
    }
}

fn checked_view_element_count(shape: &[usize], op: &'static str) -> crate::Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("shape product overflows for shape {shape:?}"),
            })
    })
}

fn checked_view_offset(
    shape: &[usize],
    strides: &[isize],
    base_offset: isize,
    indices: &[usize],
) -> Option<usize> {
    if indices.len() != shape.len() {
        return None;
    }

    let mut offset = base_offset;
    for ((&index, &extent), &stride) in indices.iter().zip(shape).zip(strides) {
        if index >= extent {
            return None;
        }
        let index = isize::try_from(index).ok()?;
        let delta = index.checked_mul(stride)?;
        offset = offset.checked_add(delta)?;
    }

    usize::try_from(offset).ok()
}

fn reachable_layout_span(
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> crate::Result<Option<(usize, usize)>> {
    if shape.contains(&0) {
        return Ok(None);
    }

    let mut min_offset = offset;
    let mut max_offset = offset;
    for (&extent, &stride) in shape.iter().zip(strides) {
        let steps =
            isize::try_from(extent.saturating_sub(1)).map_err(|_| crate::Error::InvalidConfig {
                op: "TypedTensorViewMut::try_multi_slice_mut",
                message: "shape extent does not fit in isize".to_string(),
            })?;
        let end = stride
            .checked_mul(steps)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensorViewMut::try_multi_slice_mut",
                message: "stride span overflows".to_string(),
            })?;
        let (axis_min, axis_max) = if end < 0 { (end, 0) } else { (0, end) };
        min_offset =
            min_offset
                .checked_add(axis_min)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op: "TypedTensorViewMut::try_multi_slice_mut",
                    message: "minimum reachable offset overflows".to_string(),
                })?;
        max_offset =
            max_offset
                .checked_add(axis_max)
                .ok_or_else(|| crate::Error::InvalidConfig {
                    op: "TypedTensorViewMut::try_multi_slice_mut",
                    message: "maximum reachable offset overflows".to_string(),
                })?;
    }

    let min_offset = usize::try_from(min_offset).map_err(|_| crate::Error::InvalidConfig {
        op: "TypedTensorViewMut::try_multi_slice_mut",
        message: "minimum reachable offset is negative".to_string(),
    })?;
    let max_offset = usize::try_from(max_offset).map_err(|_| crate::Error::InvalidConfig {
        op: "TypedTensorViewMut::try_multi_slice_mut",
        message: "maximum reachable offset is negative".to_string(),
    })?;
    Ok(Some((min_offset, max_offset)))
}

fn split_two_mut_ranges<T>(
    data: &mut [T],
    first: (usize, usize),
    second: (usize, usize),
) -> Option<(&mut [T], &mut [T])> {
    if first.1 < second.0 {
        let (_, after_first_start) = data.split_at_mut(first.0);
        let (first_slice, after_first) = after_first_start.split_at_mut(first.1 - first.0 + 1);
        let (_, after_gap) = after_first.split_at_mut(second.0 - first.1 - 1);
        let (second_slice, _) = after_gap.split_at_mut(second.1 - second.0 + 1);
        Some((first_slice, second_slice))
    } else if second.1 < first.0 {
        let (_, after_second_start) = data.split_at_mut(second.0);
        let (second_slice, after_second) = after_second_start.split_at_mut(second.1 - second.0 + 1);
        let (_, after_gap) = after_second.split_at_mut(first.0 - second.1 - 1);
        let (first_slice, _) = after_gap.split_at_mut(first.1 - first.0 + 1);
        Some((first_slice, second_slice))
    } else {
        None
    }
}

fn adjusted_view_offset(offset: isize, span_start: usize) -> crate::Result<isize> {
    let span_start = isize::try_from(span_start).map_err(|_| crate::Error::InvalidConfig {
        op: "TypedTensorViewMut::try_multi_slice_mut",
        message: "view span start does not fit in isize".to_string(),
    })?;
    offset
        .checked_sub(span_start)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op: "TypedTensorViewMut::try_multi_slice_mut",
            message: "adjusted view offset overflows".to_string(),
        })
}

fn view_mut_from_layout_and_slice<'a, T: 'static, R: TensorRank>(
    layout: &TensorLayout<R>,
    offset: isize,
    data: &'a mut [T],
    placement: Placement,
) -> crate::Result<TypedTensorViewMut<'a, T, R>> {
    let shape = R::shape_from_vec(layout.shape().to_vec().into())
        .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_multi_slice_mut", err))?;
    let strides = R::strides_from_vec(layout.strides().to_vec().into())
        .map_err(|err| tensor_layout_error("TypedTensorViewMut::try_multi_slice_mut", err))?;
    TypedTensorViewMut::from_buffer_ref_mut(
        shape,
        strides,
        offset,
        TensorBufferRefMut::Host(data),
        placement,
        "TypedTensorViewMut::try_multi_slice_mut",
    )
}

fn contiguous_layout_slice<'a, T, R: TensorRank>(
    layout: &TensorLayout<R>,
    data: &'a [T],
    op: &'static str,
) -> crate::Result<&'a [T]> {
    if !layout
        .is_compact_col_major()
        .map_err(|err| tensor_layout_error(op, err))?
    {
        return Err(crate::Error::InvalidConfig {
            op,
            message: "view is not contiguous column-major".to_string(),
        });
    }
    let len = checked_view_element_count(layout.shape(), op)?;
    let start = usize::try_from(layout.offset()).map_err(|_| crate::Error::InvalidConfig {
        op,
        message: "view offset is negative".to_string(),
    })?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: "contiguous view range overflows".to_string(),
        })?;
    data.get(start..end)
        .ok_or_else(|| crate::Error::InvalidConfig {
            op,
            message: "contiguous view range is outside host buffer".to_string(),
        })
}

fn relaxed_col_major_contiguous(
    shape: &[usize],
    strides: &[isize],
    op: &'static str,
) -> crate::Result<bool> {
    if shape.contains(&0) {
        return Ok(true);
    }

    let mut expected = 1isize;
    for (&extent, &stride) in shape.iter().zip(strides) {
        if extent <= 1 {
            continue;
        }
        if stride != expected {
            return Ok(false);
        }
        let extent = isize::try_from(extent).map_err(|_| crate::Error::InvalidConfig {
            op,
            message: "shape extent does not fit in isize".to_string(),
        })?;
        expected = expected
            .checked_mul(extent)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "contiguous stride overflows".to_string(),
            })?;
    }
    Ok(true)
}

fn reshape_layout_dyn<R: TensorRank>(
    layout: &TensorLayout<R>,
    shape: &[usize],
    buffer_len: usize,
    op: &'static str,
) -> crate::Result<TensorLayout<DynRank>> {
    match layout.reshape_view_as::<DynRank>(shape.to_vec().into(), buffer_len) {
        Ok(layout) => Ok(layout),
        Err(err) => {
            if !relaxed_col_major_contiguous(layout.shape(), layout.strides(), op)? {
                return Err(tensor_layout_error(op, err));
            }
            let from = checked_view_element_count(layout.shape(), op)?;
            let to = checked_view_element_count(shape, op)?;
            if from != to {
                return Err(tensor_layout_error(
                    op,
                    tenferro_tensor_core::Error::ReshapeElementCountMismatch { from, to },
                ));
            }
            TensorLayout::<DynRank>::compact(shape.to_vec().into())
                .and_then(|compact| {
                    TensorLayout::from_parts(
                        compact.shape().to_vec().into(),
                        compact.strides().to_vec().into(),
                        layout.offset(),
                        buffer_len,
                    )
                })
                .map_err(|err| tensor_layout_error(op, err))
        }
    }
}

fn core_slice_specs(
    slices: &[StridedSliceSpec],
    shape: &[usize],
    op: &'static str,
) -> crate::Result<Vec<CoreSliceSpec>> {
    if slices.len() != shape.len() {
        return Err(crate::Error::RankMismatch {
            op,
            expected: shape.len(),
            actual: slices.len(),
        });
    }

    let mut specs = Vec::with_capacity(slices.len());
    for (slice, &axis_len) in slices.iter().zip(shape) {
        specs.push(core_slice_spec(*slice, axis_len, op)?);
    }
    Ok(specs)
}

fn core_slice_spec(
    slice: StridedSliceSpec,
    axis_len: usize,
    op: &'static str,
) -> crate::Result<CoreSliceSpec> {
    if slice.step() == 0 {
        return Err(crate::Error::InvalidConfig {
            op,
            message: "slice step must not be zero".to_string(),
        });
    }

    let start = normalize_strided_bound(slice.start(), axis_len, op, "slice start")?;
    let end = match slice.end() {
        Some(end) => normalize_strided_bound(end, axis_len, op, "slice end")?,
        None => isize::try_from(axis_len).map_err(|_| crate::Error::InvalidConfig {
            op,
            message: format!("axis length {axis_len} does not fit in isize"),
        })?,
    };

    if slice.step() > 0 {
        return Ok(CoreSliceSpec {
            start,
            end,
            step: slice.step(),
        });
    }

    if start >= end {
        return Ok(CoreSliceSpec {
            start,
            end: start,
            step: slice.step(),
        });
    }

    Ok(CoreSliceSpec {
        start: end
            .checked_sub(1)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "negative-step slice start overflows".to_string(),
            })?,
        end: start
            .checked_sub(1)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: "negative-step slice end overflows".to_string(),
            })?,
        step: slice.step(),
    })
}

fn normalize_strided_bound(
    bound: isize,
    axis_len: usize,
    op: &'static str,
    role: &'static str,
) -> crate::Result<isize> {
    let axis_len = isize::try_from(axis_len).map_err(|_| crate::Error::InvalidConfig {
        op,
        message: format!("axis length {axis_len} does not fit in isize"),
    })?;
    let bound = if bound < 0 {
        axis_len
            .checked_add(bound)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op,
                message: format!("{role} {bound} overflows"),
            })?
    } else {
        bound
    };
    if !(0..=axis_len).contains(&bound) {
        return Err(crate::Error::InvalidConfig {
            op,
            message: format!("{role} {bound} is outside 0..={axis_len}"),
        });
    }
    Ok(bound)
}

fn slice_axis_specs(
    rank: usize,
    axis: usize,
    slice: StridedSliceSpec,
    op: &'static str,
) -> crate::Result<Vec<StridedSliceSpec>> {
    if axis >= rank {
        return Err(crate::Error::AxisOutOfBounds { op, axis, rank });
    }

    let mut slices = vec![StridedSliceSpec::all(); rank];
    slices[axis] = slice;
    Ok(slices)
}

pub(crate) fn default_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        device: None,
    }
}

fn typed_tensor_from_vec_col_major<T, R: TensorRank>(
    shape: impl Into<R::Shape>,
    data: Vec<T>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_from_vec_col_major(shape, data, op)
}

fn try_typed_tensor_from_vec_col_major<T, R: TensorRank>(
    shape: impl Into<R::Shape>,
    data: Vec<T>,
    op: &'static str,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, op)?;
    try_checked_shape_len(layout.shape(), data.len(), op)?;
    Ok(TypedTensor {
        buffer: Buffer::Host(data),
        layout,
        placement: default_placement(),
    })
}

fn typed_tensor_zeros<T: Clone + Zero, R: TensorRank>(
    shape: impl Into<R::Shape>,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_zeros(shape)
}

fn try_typed_tensor_zeros<T: Clone + Zero, R: TensorRank>(
    shape: impl Into<R::Shape>,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, "zeros")?;
    let n = try_shape_product(layout.shape(), "zeros")?;
    Ok(TypedTensor {
        buffer: Buffer::Host(vec![T::zero(); n]),
        layout,
        placement: default_placement(),
    })
}

fn typed_tensor_ones<T: Clone + One + Zero, R: TensorRank>(
    shape: impl Into<R::Shape>,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_ones(shape)
}

fn try_typed_tensor_ones<T: Clone + One + Zero, R: TensorRank>(
    shape: impl Into<R::Shape>,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, "ones")?;
    let n = try_shape_product(layout.shape(), "ones")?;
    Ok(TypedTensor {
        buffer: Buffer::Host(vec![T::one(); n]),
        layout,
        placement: default_placement(),
    })
}

fn typed_tensor_from_buffer_col_major<T: 'static, R: TensorRank>(
    shape: impl Into<R::Shape>,
    buffer: Buffer<T>,
    placement: Placement,
) -> crate::Result<TypedTensor<T, R>> {
    try_typed_tensor_from_buffer_col_major(shape, buffer, placement)
}

fn try_typed_tensor_from_buffer_col_major<T: 'static, R: TensorRank>(
    shape: impl Into<R::Shape>,
    buffer: Buffer<T>,
    placement: Placement,
) -> crate::Result<TypedTensor<T, R>> {
    let layout = try_compact_layout(shape, "from_buffer_col_major")?;
    let len = buffer.len();
    try_checked_shape_len(layout.shape(), len, "from_buffer_col_major")?;
    Ok(TypedTensor {
        buffer,
        layout,
        placement,
    })
}

impl<T: Clone + Zero, R: TensorRank> TypedTensor<T, R> {
    /// Allocate a zero-filled tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]).unwrap();
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    pub fn zeros(shape: impl Into<R::Shape>) -> crate::Result<Self> {
        typed_tensor_zeros(shape)
    }
}

impl<T: Clone + One + Zero, R: TensorRank> TypedTensor<T, R> {
    /// Allocate a one-filled tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::ones(vec![2]).unwrap();
    /// assert_eq!(t.host_data().unwrap(), &[1.0, 1.0]);
    /// ```
    pub fn ones(shape: impl Into<R::Shape>) -> crate::Result<Self> {
        typed_tensor_ones(shape)
    }
}

impl<T, R: TensorRank> TypedTensor<T, R> {
    /// Create a tensor from an existing buffer and compact column-major layout.
    ///
    /// This preserves the owned tensor invariant that layout metadata is
    /// compact column-major, including for backend-owned buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Buffer, Placement, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64>::from_buffer_col_major(
    ///     vec![2],
    ///     Buffer::Host(vec![1.0, 2.0]),
    ///     Placement {
    ///         memory_kind: tenferro_tensor::MemoryKind::UnpinnedHost,
    ///         device: None,
    ///     },
    /// )
    /// .unwrap();
    /// assert_eq!(tensor.shape(), &[2]);
    /// ```
    pub fn from_buffer_col_major(
        shape: impl Into<R::Shape>,
        buffer: Buffer<T>,
        placement: Placement,
    ) -> crate::Result<Self>
    where
        T: 'static,
    {
        typed_tensor_from_buffer_col_major(shape, buffer, placement)
    }

    /// Convert this tensor into static rank metadata after validating its rank.
    ///
    /// The buffer and placement are preserved. This method changes only the
    /// compile-time rank marker on the owned compact column-major tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let ranked: TypedTensor<f64, Rank<2>> = tensor.try_into_rank::<2>()?;
    /// assert_eq!(ranked.shape(), &[2, 3]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn try_into_rank<const N: usize>(self) -> crate::Result<TypedTensor<T, Rank<N>>> {
        let op = "TypedTensor::try_into_rank";
        let shape = <Rank<N> as TensorRank>::shape_from_vec(self.shape().to_vec().into())
            .map_err(|err| tensor_layout_error(op, err))?;
        let layout =
            TensorLayout::<Rank<N>>::compact(shape).map_err(|err| tensor_layout_error(op, err))?;
        Ok(TypedTensor {
            buffer: self.buffer,
            layout,
            placement: self.placement,
        })
    }

    /// Number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    pub fn n_elements(&self) -> usize {
        // Invariant: owned tensor constructors validate compact shape length against buffer length.
        match try_shape_product(self.shape(), "TypedTensor::n_elements") {
            Ok(n) => n,
            Err(err) => {
                unreachable!("TypedTensor compact shape is validated at construction: {err}")
            }
        }
    }

    /// Tensor shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Tensor rank.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.rank(), 2);
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Tensor layout metadata.
    ///
    /// Owned typed tensors are always compact column-major layouts.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
    /// assert_eq!(t.layout().strides(), &[1, 2]);
    /// ```
    pub fn layout(&self) -> &TensorLayout<R> {
        &self.layout
    }

    /// Return the storage backing this tensor.
    ///
    /// This is an explicit storage-inspection API for backend glue and tests.
    /// Host value inspection should prefer [`TypedTensor::host_data`] when the
    /// caller requires host storage.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Buffer, TypedTensor};
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert!(matches!(t.buffer(), Buffer::Host(_)));
    /// ```
    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }

    /// Return placement metadata for this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryKind, TypedTensor};
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    /// assert_eq!(t.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// ```
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// Replace placement metadata without changing the storage buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{MemoryKind, Placement, TypedTensor};
    ///
    /// let mut t = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    /// t.set_placement(Placement {
    ///     memory_kind: MemoryKind::PinnedHost,
    ///     device: None,
    /// });
    /// assert_eq!(t.placement().memory_kind, MemoryKind::PinnedHost);
    /// ```
    pub fn set_placement(&mut self, placement: Placement) {
        self.placement = placement;
    }

    /// Borrow this tensor as a typed view preserving rank and layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Rank, TypedTensor};
    ///
    /// let tensor = TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0; 4]).unwrap();
    /// let view = tensor.as_view();
    /// assert_eq!(view.strides(), &[1, 2]);
    /// ```
    pub fn as_view(&self) -> TypedTensorView<'_, T, R>
    where
        T: 'static,
    {
        let buffer = match &self.buffer {
            Buffer::Host(data) => TensorBufferRef::Host(data),
            Buffer::Backend(buffer) => TensorBufferRef::Backend(Arc::clone(buffer)),
        };
        TypedTensorView {
            buffer,
            layout: self.layout.clone(),
            placement: self.placement.clone(),
        }
    }

    /// Mutably borrow this tensor as a typed view preserving rank and layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut tensor = TypedTensor::<i32>::from_vec_col_major(vec![1], vec![1]).unwrap();
    /// *tensor.as_view_mut().get_mut(&[0]).unwrap() = 2;
    /// assert_eq!(tensor.as_slice().unwrap(), &[2]);
    /// ```
    pub fn as_view_mut(&mut self) -> TypedTensorViewMut<'_, T, R>
    where
        T: 'static,
    {
        let layout = self.layout.clone();
        let placement = self.placement.clone();
        let buffer = match &mut self.buffer {
            Buffer::Host(data) => TensorBufferRefMut::Host(data),
            Buffer::Backend(buffer) => TensorBufferRefMut::Backend(Arc::clone(buffer)),
        };
        TypedTensorViewMut {
            buffer,
            layout,
            placement,
        }
    }

    /// Borrow a read-only strided region view over this tensor's backend
    /// (device) buffer from explicit layout metadata.
    ///
    /// This is a metadata-only view: no data is copied or transferred. The
    /// layout's reachable element span is validated against the backend
    /// buffer's physical length. Host-backed tensors are rejected with an
    /// explicit backend error; host regions are expressed with
    /// [`TypedTensorView::from_slice`] over host storage instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// // Host tensors are rejected: this constructor is for backend buffers.
    /// let host = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap();
    /// let err = host.backend_region_view(vec![2, 2], vec![1, 2], 0).unwrap_err();
    /// assert!(err.to_string().contains("backend"));
    /// ```
    pub fn backend_region_view(
        &self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> crate::Result<TypedTensorView<'_, T, DynRank>>
    where
        T: 'static,
    {
        let op = "TypedTensor::backend_region_view";
        let Buffer::Backend(buffer) = &self.buffer else {
            return Err(crate::Error::backend_failure(
                op,
                "expected a backend (device) buffer; host tensors use \
                 TypedTensorView::from_slice over host storage",
            ));
        };
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(TypedTensorView {
            buffer: TensorBufferRef::Backend(Arc::clone(buffer)),
            layout,
            placement: self.placement.clone(),
        })
    }

    /// Borrow a mutable strided region view over this tensor's backend
    /// (device) buffer from explicit layout metadata.
    ///
    /// This is the mutable counterpart of
    /// [`TypedTensor::backend_region_view`]. The layout's reachable element
    /// span is validated against the backend buffer's physical length, and
    /// layouts whose logical elements alias the same physical element are
    /// rejected. Host-backed tensors are rejected with an explicit backend
    /// error; mutable host regions must go through
    /// [`TypedTensorViewMut::try_multi_slice_mut`] or host constructors.
    ///
    /// Backend buffers are shared handles, so distinct region views over one
    /// buffer can coexist; disjointness between regions used concurrently by
    /// backend operations is the caller's contract (as with BLAS-style
    /// in-place update APIs).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// // Host tensors are rejected: this constructor is for backend buffers.
    /// let mut host = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![0.0; 4]).unwrap();
    /// let err = host.backend_region_view_mut(vec![2, 2], vec![1, 2], 0).unwrap_err();
    /// assert!(err.to_string().contains("backend"));
    /// ```
    pub fn backend_region_view_mut(
        &mut self,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: isize,
    ) -> crate::Result<TypedTensorViewMut<'_, T, DynRank>>
    where
        T: 'static,
    {
        let op = "TypedTensor::backend_region_view_mut";
        let Buffer::Backend(buffer) = &self.buffer else {
            return Err(crate::Error::backend_failure(
                op,
                "expected a backend (device) buffer; mutable host regions use \
                 TypedTensorViewMut host constructors or try_multi_slice_mut",
            ));
        };
        let layout = TensorLayout::from_parts(shape.into(), strides.into(), offset, buffer.len())
            .map_err(|err| tensor_layout_error(op, err))?;
        layout
            .validate_mutable_no_overlap()
            .map_err(|err| tensor_layout_error(op, err))?;
        Ok(TypedTensorViewMut {
            buffer: TensorBufferRefMut::Backend(Arc::clone(buffer)),
            layout,
            placement: self.placement.clone(),
        })
    }

    /// Consume this tensor and return its layout metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert!(t.into_layout().is_compact_col_major().unwrap());
    /// ```
    pub fn into_layout(self) -> TensorLayout<R> {
        self.layout
    }

    /// Consume this tensor and return its storage, layout, and placement.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Buffer, TypedTensor};
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let (buffer, layout, placement) = t.into_parts();
    /// assert!(matches!(buffer, Buffer::Host(_)));
    /// assert_eq!(layout.shape(), &[2]);
    /// assert!(placement.device.is_none());
    /// ```
    pub fn into_parts(self) -> (Buffer<T>, TensorLayout<R>, Placement) {
        (self.buffer, self.layout, self.placement)
    }
}

impl<T: Clone, R: TensorRank> TypedTensor<T, R> {
    /// Create a tensor from a column-major buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert_eq!(t.get(&[1, 0])?, &2.0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn from_vec_col_major(shape: impl Into<R::Shape>, data: Vec<T>) -> crate::Result<Self> {
        typed_tensor_from_vec_col_major(shape, data, "from_vec_col_major")
    }

    /// Consume this tensor and return its owned column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// let (shape, data) = t.into_vec_col_major().unwrap();
    /// assert_eq!(shape, vec![2]);
    /// assert_eq!(data, vec![1.0, 2.0]);
    /// ```
    pub fn into_vec_col_major(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
        let shape = self.shape().to_vec();
        match self.buffer {
            Buffer::Host(data) => Ok((shape, data)),
            Buffer::Backend(_) => Err(crate::Error::backend_failure(
                "into_vec_col_major",
                "backend buffers cannot be exported as host Vec",
            )),
        }
    }

    /// Borrow the host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert_eq!(t.host_data()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn host_data(&self) -> crate::Result<&[T]> {
        match &self.buffer {
            Buffer::Host(v) => Ok(v),
            Buffer::Backend(_) => Err(crate::Error::backend_failure(
                "TypedTensor::host_data",
                "backend buffers cannot be inspected as host slices; download explicitly first",
            )),
        }
    }

    /// View the tensor data as a flat slice.
    ///
    /// This is an alias for `host_data()` for API consistency with
    /// `Tensor::as_slice`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert_eq!(t.as_slice()?, &[1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_slice(&self) -> crate::Result<&[T]> {
        self.host_data()
    }

    /// Mutably borrow the host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::zeros(vec![2]).unwrap();
    /// t.host_data_mut()?[0] = 3.0;
    /// assert_eq!(t.host_data()?, &[3.0, 0.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn host_data_mut(&mut self) -> crate::Result<&mut [T]> {
        match &mut self.buffer {
            Buffer::Host(v) => Ok(v),
            Buffer::Backend(_) => Err(crate::Error::backend_failure(
                "TypedTensor::host_data_mut",
                "backend buffers cannot be mutated as host slices; download explicitly first",
            )),
        }
    }

    /// Compute the linear physical-buffer offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]).unwrap();
    /// assert_eq!(t.linear_offset(&[1, 2])?, 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), indices, "TypedTensor::linear_offset")
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]).unwrap();
    /// assert_eq!(t.layout_linear_offset(&[1, 2])?, 5);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        try_linear_offset_for_shape(self.shape(), indices, "TypedTensor::layout_linear_offset")
    }

    /// Return whether this owned tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2]).unwrap();
    /// assert!(t.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        self.layout
            .is_compact_col_major()
            .map_err(|err| tensor_layout_error("TypedTensor::is_col_major_contiguous", err))
    }

    /// Return a compact string summary of this tensor's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2]).unwrap();
    /// assert!(t.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        layout_summary(self.shape(), self.layout.strides(), self.layout.offset())
    }

    /// Assert this tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2]).unwrap();
    /// t.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            self.shape(),
            self.layout.strides(),
            self.layout.offset(),
            "TypedTensor::assert_col_major_contiguous",
        )
    }

    /// Borrow a single element by multi-index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    /// assert_eq!(t.get(&[1])?, &2.0);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get(&self, indices: &[usize]) -> crate::Result<&T> {
        let off = self.linear_offset(indices)?;
        self.host_data()?
            .get(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensor::get",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }

    /// Mutably borrow a single element by multi-index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::zeros(vec![1]).unwrap();
    /// *t.get_mut(&[0])? = 7.0;
    /// assert_eq!(t.host_data()?, &[7.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> crate::Result<&mut T> {
        let off = self.linear_offset(indices)?;
        self.host_data_mut()?
            .get_mut(off)
            .ok_or_else(|| crate::Error::InvalidConfig {
                op: "TypedTensor::get_mut",
                message: format!("linear offset {off} is outside host buffer"),
            })
    }
}

impl Tensor {
    /// Create a tensor from a shape and column-major flat data.
    ///
    /// This is the `Tensor`-level equivalent of
    /// `TypedTensor::<T>::from_vec_col_major`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    /// assert_eq!(t.shape(), &[2, 2]);
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn from_vec_col_major<T: TensorScalar>(
        shape: Vec<usize>,
        data: Vec<T>,
    ) -> crate::Result<Self> {
        T::into_tensor(shape, data)
    }

    /// Tensor shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(t) => t.shape(),
            Tensor::F64(t) => t.shape(),
            Tensor::I32(t) => t.shape(),
            Tensor::I64(t) => t.shape(),
            Tensor::Bool(t) => t.shape(),
            Tensor::C32(t) => t.shape(),
            Tensor::C64(t) => t.shape(),
        }
    }

    /// Tensor dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DType, Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![1.0]).unwrap());
    /// assert_eq!(t.dtype(), DType::F64);
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::F32(_) => DType::F32,
            Tensor::F64(_) => DType::F64,
            Tensor::I32(_) => DType::I32,
            Tensor::I64(_) => DType::I64,
            Tensor::Bool(_) => DType::Bool,
            Tensor::C32(_) => DType::C32,
            Tensor::C64(_) => DType::C64,
        }
    }

    /// Return placement metadata for this dtype-erased tensor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{MemoryKind, Tensor};
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// assert_eq!(t.placement().memory_kind, MemoryKind::UnpinnedHost);
    /// ```
    pub fn placement(&self) -> &Placement {
        match self {
            Tensor::F32(t) => t.placement(),
            Tensor::F64(t) => t.placement(),
            Tensor::I32(t) => t.placement(),
            Tensor::I64(t) => t.placement(),
            Tensor::Bool(t) => t.placement(),
            Tensor::C32(t) => t.placement(),
            Tensor::C64(t) => t.placement(),
        }
    }

    /// Return whether this tensor is backed by backend-native storage.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    /// assert!(!t.is_backend_buffer());
    /// ```
    pub fn is_backend_buffer(&self) -> bool {
        match self {
            Tensor::F32(t) => t.buffer().is_backend(),
            Tensor::F64(t) => t.buffer().is_backend(),
            Tensor::I32(t) => t.buffer().is_backend(),
            Tensor::I64(t) => t.buffer().is_backend(),
            Tensor::Bool(t) => t.buffer().is_backend(),
            Tensor::C32(t) => t.buffer().is_backend(),
            Tensor::C64(t) => t.buffer().is_backend(),
        }
    }

    /// Compute the physical element offset for a logical index.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert_eq!(t.layout_linear_offset(&[1])?, 1);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_linear_offset(&self, indices: &[usize]) -> crate::Result<usize> {
        match self {
            Tensor::F32(t) => t.layout_linear_offset(indices),
            Tensor::F64(t) => t.layout_linear_offset(indices),
            Tensor::I32(t) => t.layout_linear_offset(indices),
            Tensor::I64(t) => t.layout_linear_offset(indices),
            Tensor::Bool(t) => t.layout_linear_offset(indices),
            Tensor::C32(t) => t.layout_linear_offset(indices),
            Tensor::C64(t) => t.layout_linear_offset(indices),
        }
    }

    /// Return whether this tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(t.is_col_major_contiguous()?);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_col_major_contiguous(&self) -> crate::Result<bool> {
        match self {
            Tensor::F32(t) => t.is_col_major_contiguous(),
            Tensor::F64(t) => t.is_col_major_contiguous(),
            Tensor::I32(t) => t.is_col_major_contiguous(),
            Tensor::I64(t) => t.is_col_major_contiguous(),
            Tensor::Bool(t) => t.is_col_major_contiguous(),
            Tensor::C32(t) => t.is_col_major_contiguous(),
            Tensor::C64(t) => t.is_col_major_contiguous(),
        }
    }

    /// Return a compact string summary of this tensor's layout metadata.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// assert!(t.layout_summary().contains("shape=[2]"));
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn layout_summary(&self) -> String {
        let layout = tensor_layout(self);
        layout_summary(layout.shape(), layout.strides(), layout.offset())
    }

    /// Assert this tensor is compact column-major.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// t.assert_col_major_contiguous()?;
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn assert_col_major_contiguous(&self) -> crate::Result<()> {
        let layout = tensor_layout(self);
        assert_layout_col_major_contiguous(
            self.is_col_major_contiguous()?,
            layout.shape(),
            layout.strides(),
            layout.offset(),
            "Tensor::assert_col_major_contiguous",
        )
    }

    /// Try to borrow the host data as a typed slice.
    ///
    /// Returns an error if the tensor dtype does not match `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    /// assert_eq!(t.as_slice::<f64>().unwrap(), [1.0, 2.0, 3.0].as_slice());
    /// assert!(t.as_slice::<f32>().is_err());
    /// ```
    pub fn as_slice<T: TensorScalar>(&self) -> crate::Result<&[T]> {
        T::as_slice(self)
    }

    /// Consume this tensor and return its owned column-major buffer when the
    /// dtype matches.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    /// assert_eq!(t.into_vec_col_major::<f64>().unwrap().1, vec![2.0]);
    /// ```
    pub fn into_vec_col_major<T: TensorScalar>(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
        let typed = T::into_typed(self)?;
        typed.into_vec_col_major()
    }
}

// Kept for crate-local layout tests while tensor indexing helpers remain split
// across tensor and CPU crates.
#[allow(dead_code)]
pub(crate) fn flat_to_multi(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    for i in 0..shape.len() {
        if shape[i] == 0 {
            out[i] = 0;
        } else {
            out[i] = flat % shape[i];
            flat /= shape[i];
        }
    }
}
