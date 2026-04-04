use num_complex::Complex;
use num_traits::{One, Zero};

#[derive(Clone, Debug)]
pub enum MemoryKind {
    Device,
    PinnedHost,
    UnpinnedHost,
    Other(String),
}

#[derive(Clone, Debug)]
pub struct ComputeDevice {
    pub kind: String,
    pub ordinal: usize,
}

#[derive(Clone, Debug)]
pub struct Placement {
    pub memory_kind: MemoryKind,
    pub resident_device: Option<ComputeDevice>,
}

#[derive(Clone, Debug)]
pub struct BufferHandle<T> {
    pub id: u64,
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Clone, Debug)]
pub enum Buffer<T> {
    Host(Vec<T>),
    Backend(BufferHandle<T>),
}

#[derive(Clone, Debug)]
pub struct TypedTensor<T> {
    pub buffer: Buffer<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub placement: Placement,
    pub preferred_compute_device: Option<ComputeDevice>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    C32,
    C64,
}

#[derive(Clone, Debug)]
pub enum Tensor {
    F32(TypedTensor<f32>),
    F64(TypedTensor<f64>),
    C32(TypedTensor<Complex<f32>>),
    C64(TypedTensor<Complex<f64>>),
}

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
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let strides = col_major_strides(&shape);
        Self {
            buffer: Buffer::Host(vec![T::zero(); n]),
            shape,
            strides,
            placement: default_placement(),
            preferred_compute_device: None,
        }
    }
}

impl<T: Clone + One + Zero> TypedTensor<T> {
    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        let strides = col_major_strides(&shape);
        Self {
            buffer: Buffer::Host(vec![T::one(); n]),
            shape,
            strides,
            placement: default_placement(),
            preferred_compute_device: None,
        }
    }
}

impl<T: Clone> TypedTensor<T> {
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            n,
            "data length {} does not match shape product {}",
            data.len(),
            n
        );
        let strides = col_major_strides(&shape);
        Self {
            buffer: Buffer::Host(data),
            shape,
            strides,
            placement: default_placement(),
            preferred_compute_device: None,
        }
    }

    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn host_data(&self) -> &[T] {
        match &self.buffer {
            Buffer::Host(v) => v,
            Buffer::Backend(_) => panic!("host_data called on backend buffer"),
        }
    }

    pub fn host_data_mut(&mut self) -> &mut [T] {
        match &mut self.buffer {
            Buffer::Host(v) => v,
            Buffer::Backend(_) => panic!("host_data_mut called on backend buffer"),
        }
    }

    pub(crate) fn linear_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len());
        let mut offset: isize = 0;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "index out of bounds");
            offset += idx as isize * self.strides[i];
        }
        offset as usize
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        let off = self.linear_offset(indices);
        &self.host_data()[off]
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let off = self.linear_offset(indices);
        &mut self.host_data_mut()[off]
    }
}

pub trait ConjElem {
    fn conj_elem(&self) -> Self;
}

impl ConjElem for f32 {
    fn conj_elem(&self) -> Self {
        *self
    }
}

impl ConjElem for f64 {
    fn conj_elem(&self) -> Self {
        *self
    }
}

impl ConjElem for Complex<f32> {
    fn conj_elem(&self) -> Self {
        self.conj()
    }
}

impl ConjElem for Complex<f64> {
    fn conj_elem(&self) -> Self {
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
    pub fn shape(&self) -> &[usize] {
        match self {
            Tensor::F32(t) => &t.shape,
            Tensor::F64(t) => &t.shape,
            Tensor::C32(t) => &t.shape,
            Tensor::C64(t) => &t.shape,
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Tensor::F32(_) => DType::F32,
            Tensor::F64(_) => DType::F64,
            Tensor::C32(_) => DType::C32,
            Tensor::C64(_) => DType::C64,
        }
    }

    pub fn neg(&self) -> Self {
        dispatch_tensor!(self, t => typed_neg(t))
    }

    pub fn transpose_perm(&self, perm: &[usize]) -> Self {
        dispatch_tensor!(self, t => typed_transpose(t, perm))
    }

    pub fn extract_diagonal(&self, axis_a: usize, axis_b: usize) -> Self {
        dispatch_tensor!(self, t => typed_extract_diagonal(t, axis_a, axis_b))
    }
}

fn typed_neg<T>(t: &TypedTensor<T>) -> TypedTensor<T>
where
    T: Clone + std::ops::Neg<Output = T> + Zero,
{
    let n = t.n_elements();
    let mut result = TypedTensor::zeros(t.shape.clone());
    let mut idx_buf = vec![0usize; t.shape.len()];
    for flat in 0..n {
        flat_to_multi(flat, &t.shape, &mut idx_buf);
        let val = t.get(&idx_buf).clone().neg();
        let off_out = result.linear_offset(&idx_buf);
        result.host_data_mut()[off_out] = val;
    }
    result
}

fn typed_transpose<T: Clone + Zero>(t: &TypedTensor<T>, perm: &[usize]) -> TypedTensor<T> {
    let rank = t.shape.len();
    assert_eq!(perm.len(), rank);
    let new_shape: Vec<usize> = perm.iter().map(|&p| t.shape[p]).collect();
    let n = t.n_elements();
    let mut result = TypedTensor::zeros(new_shape.clone());
    let mut src_idx = vec![0usize; rank];
    let mut dst_idx = vec![0usize; rank];
    for flat in 0..n {
        flat_to_multi(flat, &t.shape, &mut src_idx);
        for (d, &p) in perm.iter().enumerate() {
            dst_idx[d] = src_idx[p];
        }
        let val = t.host_data()[t.linear_offset(&src_idx)].clone();
        let off_out = result.linear_offset(&dst_idx);
        result.host_data_mut()[off_out] = val;
    }
    result
}

fn typed_extract_diagonal<T: Clone + Zero>(
    t: &TypedTensor<T>,
    axis_a: usize,
    axis_b: usize,
) -> TypedTensor<T> {
    assert!(axis_a < axis_b, "extract_diagonal requires axis_a < axis_b");
    assert_eq!(
        t.shape[axis_a], t.shape[axis_b],
        "extract_diagonal requires equal sizes on both axes"
    );
    let n = t.shape[axis_a];
    // New shape: remove axis_b, keep axis_a
    let mut new_shape = Vec::new();
    for (i, &s) in t.shape.iter().enumerate() {
        if i != axis_b {
            new_shape.push(s);
        }
    }
    let out_n: usize = new_shape.iter().product();
    let mut result = TypedTensor::zeros(new_shape.clone());
    let out_rank = new_shape.len();
    let in_rank = t.shape.len();
    let mut out_idx = vec![0usize; out_rank];
    let mut in_idx = vec![0usize; in_rank];
    for flat in 0..out_n {
        flat_to_multi(flat, &new_shape, &mut out_idx);
        // Map output index back to input index.
        // axis_b was removed. Positions before axis_b map 1:1.
        // Positions >= axis_b in output map to position+1 in input.
        // axis_a in output gives the diagonal index; axis_b in input
        // gets the same value as axis_a.
        let mut j = 0;
        for i in 0..in_rank {
            if i == axis_b {
                in_idx[i] = in_idx[axis_a]; // diagonal: same index
            } else {
                in_idx[i] = out_idx[j];
                j += 1;
            }
        }
        let _ = n; // used only in assertions above
        let val = t.get(&in_idx).clone();
        let off = result.linear_offset(&out_idx);
        result.host_data_mut()[off] = val;
    }
    result
}

pub(crate) fn flat_to_multi(mut flat: usize, shape: &[usize], out: &mut [usize]) {
    for i in 0..shape.len() {
        out[i] = flat % shape[i];
        flat /= shape[i];
    }
}
