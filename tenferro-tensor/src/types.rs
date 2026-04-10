use num_complex::Complex;
use num_traits::{One, Zero};
use std::sync::Arc;
use strided_kernel::{copy_into, Identity, StridedArray, StridedView};

/// Memory location for tensor storage.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::MemoryKind;
///
/// let kind = MemoryKind::UnpinnedHost;
/// ```
#[derive(Clone, Debug)]
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Other(String),
}

/// Concrete compute device description.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::ComputeDevice;
///
/// let device = ComputeDevice { kind: "cuda".into(), ordinal: 0 };
/// ```
#[derive(Clone, Debug)]
pub struct ComputeDevice {
    pub kind: String,
    pub ordinal: usize,
}

/// Placement metadata for a tensor buffer.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{ComputeDevice, MemoryKind, Placement};
///
/// let placement = Placement {
///     memory_kind: MemoryKind::Device,
///     resident_device: Some(ComputeDevice { kind: "cuda".into(), ordinal: 0 }),
/// };
/// ```
#[derive(Clone, Debug)]
pub struct Placement {
    pub memory_kind: MemoryKind,
    pub resident_device: Option<ComputeDevice>,
}

/// Backend-owned buffer handle.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::BufferHandle;
///
/// let handle = BufferHandle::<f64>::new(7);
/// ```
#[derive(Clone, Debug)]
pub struct BufferHandle<T> {
    pub id: u64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> BufferHandle<T> {
    /// Create a new backend buffer handle.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::BufferHandle;
    ///
    /// let handle = BufferHandle::<f64>::new(1);
    /// assert_eq!(handle.id, 1);
    /// ```
    pub fn new(id: u64) -> Self {
        Self {
            id,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Tensor storage.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use tenferro_tensor::Buffer;
///
/// let host = Buffer::Host(Arc::new(vec![1.0_f64, 2.0]));
/// ```
#[derive(Clone, Debug)]
pub enum Buffer<T> {
    Host(Arc<Vec<T>>),
    Backend(BufferHandle<T>),
}

/// Stride-aware typed tensor storage.
///
/// Cloning a host-backed tensor shares the underlying storage handle until a
/// mutable host access triggers copy-on-write.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::TypedTensor;
///
/// let t = TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
/// assert_eq!(t.shape, vec![2, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct TypedTensor<T> {
    pub buffer: Buffer<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
    pub placement: Placement,
}

/// Dense materialization order.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::LayoutOrder;
///
/// let order = LayoutOrder::ColumnMajor;
/// assert!(matches!(order, LayoutOrder::ColumnMajor));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutOrder {
    ColumnMajor,
    RowMajor,
}

/// Runtime scalar dtype tag.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::DType;
///
/// assert_eq!(DType::F64 as u8, DType::F64 as u8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    C32,
    C64,
}

/// Dynamic tensor enum over the supported scalar types.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::{Tensor, TypedTensor};
///
/// let t = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
/// assert_eq!(t.shape(), &[2]);
/// ```
#[derive(Clone, Debug)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    C32(TypedTensor<Complex<f32>>),
    C64(TypedTensor<Complex<f64>>),
}

/// Column-major strides derived from a shape.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::col_major_strides;
///
/// assert_eq!(col_major_strides(&[2, 3]), vec![1, 2]);
/// ```
pub fn col_major_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1isize; shape.len()];
    for i in 1..shape.len() {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

/// Row-major strides derived from a shape.
///
/// # Examples
///
/// ```ignore
/// use tenferro_tensor::row_major_strides;
///
/// assert_eq!(row_major_strides(&[2, 3]), vec![3, 1]);
/// ```
pub fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1isize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

pub(crate) fn default_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::UnpinnedHost,
        resident_device: None,
    }
}

impl<T: Clone + Zero> TypedTensor<T> {
    /// Allocate a zero-filled tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]);
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            buffer: Buffer::Host(Arc::new(vec![T::zero(); n])),
            strides: col_major_strides(&shape),
            offset: 0,
            shape,
            placement: default_placement(),
        }
    }
}

impl<T: Clone + One + Zero> TypedTensor<T> {
    /// Allocate a one-filled tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::ones(vec![2]);
    /// assert_eq!(t.host_data(), &[1.0, 1.0]);
    /// ```
    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self {
            buffer: Buffer::Host(Arc::new(vec![T::one(); n])),
            strides: col_major_strides(&shape),
            offset: 0,
            shape,
            placement: default_placement(),
        }
    }
}

impl<T> TypedTensor<T> {
    /// Create a tensor from a column-major buffer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(t.get(&[1, 0]), &2.0);
    /// ```
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "data length {} does not match shape product {}",
            data.len(),
            n
        );
        Self {
            buffer: Buffer::Host(Arc::new(data)),
            strides: col_major_strides(&shape),
            offset: 0,
            shape,
            placement: default_placement(),
        }
    }

    /// Borrow the logical strides.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![0.0; 6]);
    /// assert_eq!(t.strides(), &[1, 2]);
    /// ```
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Base offset into the underlying storage.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.offset(), 0);
    /// ```
    pub fn offset(&self) -> isize {
        self.offset
    }

    /// Returns true when the tensor is contiguous in column-major order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![0.0; 6]);
    /// assert!(t.is_contiguous_col_major());
    /// ```
    pub fn is_contiguous_col_major(&self) -> bool {
        self.offset == 0 && self.strides == col_major_strides(&self.shape)
    }

    /// Returns true when the tensor is contiguous in row-major order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
    /// assert!(t.is_contiguous_row_major());
    /// ```
    pub fn is_contiguous_row_major(&self) -> bool {
        self.offset == 0 && self.strides == row_major_strides(&self.shape)
    }

    /// Number of elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![0.0; 6]);
    /// assert_eq!(t.n_elements(), 6);
    /// ```
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Borrow the underlying host storage buffer.
    ///
    /// This exposes raw storage order, not logical iteration order. For
    /// non-contiguous tensors, use [`Self::get`] for indexed access or
    /// [`Self::to_contiguous`] to materialize a dense layout first.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.host_data(), &[1.0, 2.0]);
    /// ```
    pub fn host_data(&self) -> &[T] {
        match &self.buffer {
            Buffer::Host(v) => v.as_slice(),
            Buffer::Backend(_) => panic!("host_data called on backend buffer"),
        }
    }

    /// Mutably borrow the underlying host storage buffer.
    ///
    /// This triggers copy-on-write when the storage handle is shared by cloned
    /// tensors or metadata-only views.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::zeros(vec![2]);
    /// t.host_data_mut()[0] = 3.0;
    /// assert_eq!(t.host_data(), &[3.0, 0.0]);
    /// ```
    pub fn host_data_mut(&mut self) -> &mut [T]
    where
        T: Clone,
    {
        match &mut self.buffer {
            Buffer::Host(v) => Arc::make_mut(v).as_mut_slice(),
            Buffer::Backend(_) => panic!("host_data_mut called on backend buffer"),
        }
    }

    /// Compute the storage offset for a logical multi-index.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::zeros(vec![2, 3]);
    /// assert_eq!(t.linear_offset(&[1, 2]), 5);
    /// ```
    pub fn linear_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut offset = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
            offset += idx as isize * self.strides[i];
        }
        usize::try_from(offset).expect("tensor offset must be non-negative")
    }

    /// Borrow a single element by multi-index.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.get(&[1]), &2.0);
    /// ```
    pub fn get(&self, indices: &[usize]) -> &T {
        let off = self.linear_offset(indices);
        &self.host_data()[off]
    }

    /// Mutably borrow a single element by multi-index.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::TypedTensor;
    ///
    /// let mut t = TypedTensor::<f64>::zeros(vec![1]);
    /// *t.get_mut(&[0]) = 7.0;
    /// assert_eq!(t.host_data(), &[7.0]);
    /// ```
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T
    where
        T: Clone,
    {
        let off = self.linear_offset(indices);
        &mut self.host_data_mut()[off]
    }
}

impl<T: Copy + Default> TypedTensor<T> {
    /// Materialize the tensor into a dense contiguous buffer in the requested order.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{LayoutOrder, TypedTensor};
    ///
    /// let t = TypedTensor::<f64>::from_vec(vec![2, 3], vec![1.0; 6]);
    /// let row_major = t.to_contiguous(LayoutOrder::RowMajor).unwrap();
    /// assert!(row_major.is_contiguous_row_major());
    /// ```
    pub fn to_contiguous(&self, order: LayoutOrder) -> crate::Result<Self> {
        let strides = match order {
            LayoutOrder::ColumnMajor => col_major_strides(&self.shape),
            LayoutOrder::RowMajor => row_major_strides(&self.shape),
        };

        let data = match &self.buffer {
            Buffer::Host(data) => data,
            Buffer::Backend(_) => {
                return Err(crate::Error::BackendFailure {
                    op: "to_contiguous",
                    message: "backend buffer materialization is not implemented".into(),
                });
            }
        };

        let src: StridedView<'_, T, Identity> =
            StridedView::new(data, &self.shape, &self.strides, self.offset).map_err(|err| {
                crate::Error::BackendFailure {
                    op: "to_contiguous",
                    message: err.to_string(),
                }
            })?;

        let mut dst = StridedArray::<T>::from_parts(
            vec![T::default(); self.n_elements()],
            &self.shape,
            &strides,
            0,
        )
        .map_err(|err| crate::Error::BackendFailure {
            op: "to_contiguous",
            message: err.to_string(),
        })?;
        copy_into(&mut dst.view_mut(), &src).map_err(|err| crate::Error::BackendFailure {
            op: "to_contiguous",
            message: err.to_string(),
        })?;

        Ok(Self {
            buffer: Buffer::Host(Arc::new(dst.into_data())),
            shape: self.shape.clone(),
            strides,
            offset: 0,
            placement: self.placement.clone(),
        })
    }
}

/// Element-wise conjugation helper.
pub trait ConjElem {
    fn conj_elem(self) -> Self;
}

impl ConjElem for f32 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for f64 {
    fn conj_elem(self) -> Self {
        self
    }
}

impl ConjElem for Complex<f32> {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

impl ConjElem for Complex<f64> {
    fn conj_elem(self) -> Self {
        self.conj()
    }
}

macro_rules! dispatch_tensor {
    ($self:expr, $inner:ident => $body:expr) => {
        match $self {
            Tensor::F32($inner) => Tensor::F32($body),
            Tensor::F64($inner) => Tensor::F64($body),
            Tensor::C32($inner) => Tensor::C32($body),
            Tensor::C64($inner) => Tensor::C64($body),
        }
    };
}

macro_rules! dispatch_binary {
    ($lhs:expr, $rhs:expr, |$a:ident, $b:ident| $body:expr) => {
        match ($lhs, $rhs) {
            (Tensor::F32($a), Tensor::F32($b)) => Tensor::F32($body),
            (Tensor::F64($a), Tensor::F64($b)) => Tensor::F64($body),
            (Tensor::C32($a), Tensor::C32($b)) => Tensor::C32($body),
            (Tensor::C64($a), Tensor::C64($b)) => Tensor::C64($body),
            _ => panic!("dtype mismatch in binary op"),
        }
    };
}

pub(crate) use dispatch_binary;
pub(crate) use dispatch_tensor;

impl Tensor {
    /// Tensor shape.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
    /// assert_eq!(t.shape(), &[2]);
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(t) => &t.shape,
            Tensor::F64(t) => &t.shape,
            Tensor::C32(t) => &t.shape,
            Tensor::C64(t) => &t.shape,
        }
    }

    /// Tensor dtype tag.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use tenferro_tensor::{DType, Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec(vec![], vec![1.0]));
    /// assert_eq!(t.dtype(), DType::F64);
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Tensor::F32(_) => DType::F32,
            Tensor::F64(_) => DType::F64,
            Tensor::C32(_) => DType::C32,
            Tensor::C64(_) => DType::C64,
        }
    }
}

pub(crate) fn flat_to_multi(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    for i in 0..shape.len() {
        out[i] = flat % shape[i];
        flat /= shape[i];
    }
}
