use num_complex::{Complex, Complex32, Complex64};
use num_traits::{One, Zero};

use crate::{DotGeneralConfig, TensorBackend};

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
/// use tenferro_tensor::Buffer;
///
/// let host = Buffer::Host(vec![1.0_f64, 2.0]);
/// ```
#[derive(Clone, Debug)]
pub enum Buffer<T> {
    Host(Vec<T>),
    Backend(BufferHandle<T>),
    #[cfg(feature = "cubecl")]
    Cubecl(CubeclBuffer<T>),
}

/// CubeCL-managed GPU buffer.
///
/// This wraps a CubeCL server handle that owns the underlying GPU allocation.
///
/// # Examples
///
/// ```
/// let _name = core::any::type_name::<tenferro_tensor::CubeclBuffer<f64>>();
/// assert!(_name.contains("CubeclBuffer"));
/// ```
#[cfg(feature = "cubecl")]
#[derive(Clone, Debug)]
pub struct CubeclBuffer<T> {
    /// CubeCL server handle that owns the GPU allocation.
    pub handle: cubecl::server::Handle,
    /// Number of elements stored in the allocation.
    pub len: usize,
    pub(crate) _marker: std::marker::PhantomData<T>,
}

#[cfg(feature = "cubecl")]
impl<T> CubeclBuffer<T> {
    /// Create a CubeCL buffer wrapper from a handle and element count.
    ///
    /// # Examples
    ///
    /// ```
    /// let _new = tenferro_tensor::CubeclBuffer::<f64>::new;
    /// let _ = _new;
    /// ```
    pub fn new(handle: cubecl::server::Handle, len: usize) -> Self {
        Self {
            handle,
            len,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Contiguous column-major typed tensor storage.
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
    pub placement: Placement,
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

/// Sealed trait for scalar types that can be stored in a [`Tensor`].
///
/// This trait is implemented for `f64`, `f32`, [`Complex64`], and
/// [`Complex32`].
///
/// # Examples
///
/// ```
/// use tenferro_tensor::TensorScalar;
///
/// let tensor = <f64 as TensorScalar>::into_tensor(vec![2], vec![1.0, 2.0]);
/// assert_eq!(tensor.as_slice::<f64>(), Some([1.0, 2.0].as_slice()));
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

    /// Wrap typed data into a [`Tensor`] enum variant.
    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor;

    /// Try to borrow the host data from a [`Tensor`].
    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]>;

    /// Try to extract a [`TypedTensor<Self>`] from a dynamic [`Tensor`].
    ///
    /// Returns `None` if the tensor dtype does not match `Self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorScalar};
    ///
    /// let tensor = Tensor::new(vec![2], vec![1.0_f64, 2.0]);
    /// let typed = <f64 as TensorScalar>::try_into_typed(tensor).unwrap();
    ///
    /// assert_eq!(typed.as_slice(), &[1.0, 2.0]);
    /// ```
    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>>;
}

mod private {
    pub trait Sealed {}

    impl Sealed for f64 {}
    impl Sealed for f32 {}
    impl Sealed for num_complex::Complex64 {}
    impl Sealed for num_complex::Complex32 {}
}

impl TensorScalar for f64 {
    type Real = f64;

    fn dtype() -> DType {
        DType::F64
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::F64(TypedTensor::from_vec(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::F64(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::F64(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for f32 {
    type Real = f32;

    fn dtype() -> DType {
        DType::F32
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::F32(TypedTensor::from_vec(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::F32(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::F32(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for Complex64 {
    type Real = f64;

    fn dtype() -> DType {
        DType::C64
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::C64(TypedTensor::from_vec(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::C64(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::C64(inner) => Some(inner),
            _ => None,
        }
    }
}

impl TensorScalar for Complex32 {
    type Real = f32;

    fn dtype() -> DType {
        DType::C32
    }

    fn into_tensor(shape: Vec<usize>, data: Vec<Self>) -> Tensor {
        Tensor::C32(TypedTensor::from_vec(shape, data))
    }

    fn try_as_slice(tensor: &Tensor) -> Option<&[Self]> {
        match tensor {
            Tensor::C32(t) => Some(t.host_data()),
            _ => None,
        }
    }

    fn try_into_typed(tensor: Tensor) -> Option<TypedTensor<Self>> {
        match tensor {
            Tensor::C32(inner) => Some(inner),
            _ => None,
        }
    }
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
            buffer: Buffer::Host(vec![T::zero(); n]),
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
            buffer: Buffer::Host(vec![T::one(); n]),
            shape,
            placement: default_placement(),
        }
    }
}

impl<T: Clone> TypedTensor<T> {
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
            buffer: Buffer::Host(data),
            shape,
            placement: default_placement(),
        }
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

    /// Borrow the host buffer.
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
            Buffer::Host(v) => v,
            Buffer::Backend(_) => panic!("host_data called on backend buffer"),
            #[cfg(feature = "cubecl")]
            Buffer::Cubecl(_) => {
                panic!(
                    "Cannot access GPU buffer (Buffer::Cubecl) as host data. \
                       Use cubecl::download_tensor() to transfer to CPU first."
                )
            }
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
    /// let t = TypedTensor::<f64>::from_vec(vec![2], vec![1.0, 2.0]);
    /// assert_eq!(t.as_slice(), &[1.0, 2.0]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.host_data()
    }

    /// Mutably borrow the host buffer.
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
    pub fn host_data_mut(&mut self) -> &mut [T] {
        match &mut self.buffer {
            Buffer::Host(v) => v,
            Buffer::Backend(_) => panic!("host_data_mut called on backend buffer"),
            #[cfg(feature = "cubecl")]
            Buffer::Cubecl(_) => {
                panic!(
                    "Cannot access GPU buffer (Buffer::Cubecl) as host data. \
                       Use cubecl::download_tensor() to transfer to CPU first."
                )
            }
        }
    }

    /// Compute the linear column-major offset for an index.
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
        let mut offset = 0usize;
        let mut stride = 1usize;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
            offset += idx * stride;
            stride *= self.shape[i];
        }
        offset
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
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let off = self.linear_offset(indices);
        &mut self.host_data_mut()[off]
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
    /// Create a tensor from a shape and flat data.
    ///
    /// This is the `Tensor`-level equivalent of `TypedTensor::<T>::from_vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::Tensor;
    ///
    /// let t = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn new<T: TensorScalar>(shape: Vec<usize>, data: Vec<T>) -> Self {
        T::into_tensor(shape, data)
    }

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

    /// Try to borrow the host data as a typed slice.
    ///
    /// Returns `None` if the tensor dtype does not match `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TypedTensor};
    ///
    /// let t = Tensor::F64(TypedTensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]));
    /// assert_eq!(t.as_slice::<f64>(), Some([1.0, 2.0, 3.0].as_slice()));
    /// assert_eq!(t.as_slice::<f32>(), None);
    /// ```
    pub fn as_slice<T: TensorScalar>(&self) -> Option<&[T]> {
        T::try_as_slice(self)
    }

    /// Singular value decomposition: `A = U diag(S) Vt`.
    ///
    /// Returns `(U, S, Vt)` using the thin/economy SVD.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let (u, s, vt) = a.svd(&mut ctx).unwrap();
    ///
    /// assert_eq!(u.shape(), &[3, 2]);
    /// assert_eq!(s.shape(), &[2]);
    /// assert_eq!(vt.shape(), &[2, 2]);
    /// ```
    pub fn svd(&self, ctx: &mut impl TensorBackend) -> crate::Result<(Self, Self, Self)> {
        ctx.with_exec_session(|exec| unpack_three("svd", exec.svd(self)?))
    }

    /// QR decomposition: `A = Q R`.
    ///
    /// Returns `(Q, R)` using the thin/economy QR decomposition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let (q, r) = a.qr(&mut ctx).unwrap();
    ///
    /// assert_eq!(q.shape(), &[3, 2]);
    /// assert_eq!(r.shape(), &[2, 2]);
    /// ```
    pub fn qr(&self, ctx: &mut impl TensorBackend) -> crate::Result<(Self, Self)> {
        ctx.with_exec_session(|exec| unpack_two("qr", exec.qr(self)?))
    }

    /// LU decomposition with partial pivoting: `P A = L U`.
    ///
    /// Returns `(P, L, U, parity)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 2], vec![0.0_f64, 1.0, 1.0, 0.0]);
    /// let (p, l, u, parity) = a.lu(&mut ctx).unwrap();
    ///
    /// assert_eq!(p.shape(), &[2, 2]);
    /// assert_eq!(l.shape(), &[2, 2]);
    /// assert_eq!(u.shape(), &[2, 2]);
    /// assert_eq!(parity.shape(), &[] as &[usize]);
    /// ```
    pub fn lu(&self, ctx: &mut impl TensorBackend) -> crate::Result<(Self, Self, Self, Self)> {
        ctx.with_exec_session(|exec| unpack_four("lu", exec.lu(self)?))
    }

    /// Cholesky decomposition: `A = L L^T` or `A = L L^H` for complex inputs.
    ///
    /// Returns the lower-triangular factor `L`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]);
    /// let l = a.cholesky(&mut ctx).unwrap();
    ///
    /// assert_eq!(l.shape(), &[2, 2]);
    /// ```
    pub fn cholesky(&self, ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| exec.cholesky(self))
    }

    /// Symmetric or Hermitian eigendecomposition: `A = V diag(W) V^T`.
    ///
    /// Returns `(eigenvalues, eigenvectors)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]);
    /// let (w, v) = a.eigh(&mut ctx).unwrap();
    ///
    /// assert_eq!(w.shape(), &[2]);
    /// assert_eq!(v.shape(), &[2, 2]);
    /// ```
    pub fn eigh(&self, ctx: &mut impl TensorBackend) -> crate::Result<(Self, Self)> {
        ctx.with_exec_session(|exec| unpack_two("eigh", exec.eigh(self)?))
    }

    /// General eigendecomposition.
    ///
    /// Returns `(eigenvalues, eigenvectors)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 2], vec![1.0_f64, 0.0, 0.0, 3.0]);
    /// let (w, v) = a.eig(&mut ctx).unwrap();
    ///
    /// assert_eq!(w.shape(), &[2]);
    /// assert_eq!(v.shape(), &[2, 2]);
    /// ```
    pub fn eig(&self, ctx: &mut impl TensorBackend) -> crate::Result<(Self, Self)> {
        ctx.with_exec_session(|exec| unpack_two("eig", exec.eig(self)?))
    }

    /// Solve `A x = b` for `x`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 2], vec![2.0_f64, 1.0, 1.0, 2.0]);
    /// let b = Tensor::new(vec![2, 1], vec![1.0_f64, 0.0]);
    /// let x = a.solve(&b, &mut ctx).unwrap();
    ///
    /// assert_eq!(x.shape(), &[2, 1]);
    /// ```
    pub fn solve(&self, b: &Self, ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.solve(self, b)
    }

    /// Solve a triangular system.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0]);
    /// let b = Tensor::new(vec![2, 1], vec![2.0_f64, 7.0]);
    /// let x = a
    ///     .triangular_solve(&b, true, true, false, false, &mut ctx)
    ///     .unwrap();
    ///
    /// assert_eq!(x.shape(), &[2, 1]);
    /// ```
    pub fn triangular_solve(
        &self,
        b: &Self,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
        ctx: &mut impl TensorBackend,
    ) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| {
            exec.triangular_solve(self, b, left_side, lower, transpose_a, unit_diagonal)
        })
    }

    /// Elementwise addition.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]);
    /// let b = Tensor::new(vec![3], vec![4.0_f64, 5.0, 6.0]);
    /// let c = a.add(&b, &mut ctx).unwrap();
    ///
    /// assert_eq!(c.as_slice::<f64>().unwrap(), &[5.0, 7.0, 9.0]);
    /// ```
    pub fn add(&self, other: &Self, ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| exec.add(self, other))
    }

    /// Elementwise multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![3], vec![1.0_f64, 2.0, 3.0]);
    /// let b = Tensor::new(vec![3], vec![4.0_f64, 5.0, 6.0]);
    /// let c = a.mul(&b, &mut ctx).unwrap();
    ///
    /// assert_eq!(c.as_slice::<f64>().unwrap(), &[4.0, 10.0, 18.0]);
    /// ```
    pub fn mul(&self, other: &Self, ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| exec.mul(self, other))
    }

    /// Negation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![3], vec![1.0_f64, -2.0, 3.0]);
    /// let b = a.neg(&mut ctx).unwrap();
    ///
    /// assert_eq!(b.as_slice::<f64>().unwrap(), &[-1.0, 2.0, -3.0]);
    /// ```
    pub fn neg(&self, ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| exec.neg(self))
    }

    /// Transpose with an explicit permutation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]);
    /// let b = a.transpose(&[1, 0], &mut ctx).unwrap();
    ///
    /// assert_eq!(b.shape(), &[2, 2]);
    /// assert_eq!(b.as_slice::<f64>().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn transpose(&self, perm: &[usize], ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| exec.transpose(self, perm))
    }

    /// Reshape to a new shape with the same number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let b = a.reshape(&[3, 2], &mut ctx).unwrap();
    ///
    /// assert_eq!(b.shape(), &[3, 2]);
    /// assert_eq!(b.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    pub fn reshape(&self, shape: &[usize], ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| exec.reshape(self, shape))
    }

    /// Reduce sum over the specified axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let b = a.reduce_sum(&[1], &mut ctx).unwrap();
    ///
    /// assert_eq!(b.shape(), &[2]);
    /// assert_eq!(b.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
    /// ```
    pub fn reduce_sum(&self, axes: &[usize], ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        ctx.with_exec_session(|exec| exec.reduce_sum(self, axes))
    }

    /// Matrix multiplication for rank-2 tensors.
    ///
    /// This is a convenience wrapper around `dot_general`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro_tensor::{Tensor, TensorBackend, cpu::CpuBackend};
    ///
    /// let mut ctx = CpuBackend::new();
    /// let a = Tensor::new(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let b = Tensor::new(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// let c = a.matmul(&b, &mut ctx).unwrap();
    ///
    /// assert_eq!(c.shape(), &[2, 2]);
    /// assert_eq!(c.as_slice::<f64>().unwrap(), &[22.0, 28.0, 49.0, 64.0]);
    /// ```
    pub fn matmul(&self, other: &Self, ctx: &mut impl TensorBackend) -> crate::Result<Self> {
        let config = DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            lhs_rank: self.shape().len(),
            rhs_rank: other.shape().len(),
        };
        ctx.with_exec_session(|exec| exec.dot_general(self, other, &config))
    }
}

pub(crate) fn flat_to_multi(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    for i in 0..shape.len() {
        out[i] = flat % shape[i];
        flat /= shape[i];
    }
}

fn invalid_output_count(op: &'static str, expected: usize, actual: usize) -> crate::Error {
    crate::Error::BackendFailure {
        op,
        message: format!("expected {expected} output tensors, got {actual}"),
    }
}

fn unpack_two(op: &'static str, results: Vec<Tensor>) -> crate::Result<(Tensor, Tensor)> {
    let actual = results.len();
    let mut iter = results.into_iter();
    match (iter.next(), iter.next(), iter.next()) {
        (Some(a), Some(b), None) => Ok((a, b)),
        _ => Err(invalid_output_count(op, 2, actual)),
    }
}

fn unpack_three(op: &'static str, results: Vec<Tensor>) -> crate::Result<(Tensor, Tensor, Tensor)> {
    let actual = results.len();
    let mut iter = results.into_iter();
    match (iter.next(), iter.next(), iter.next(), iter.next()) {
        (Some(a), Some(b), Some(c), None) => Ok((a, b, c)),
        _ => Err(invalid_output_count(op, 3, actual)),
    }
}

fn unpack_four(
    op: &'static str,
    results: Vec<Tensor>,
) -> crate::Result<(Tensor, Tensor, Tensor, Tensor)> {
    let actual = results.len();
    let mut iter = results.into_iter();
    match (
        iter.next(),
        iter.next(),
        iter.next(),
        iter.next(),
        iter.next(),
    ) {
        (Some(a), Some(b), Some(c), Some(d), None) => Ok((a, b, c, d)),
        _ => Err(invalid_output_count(op, 4, actual)),
    }
}
