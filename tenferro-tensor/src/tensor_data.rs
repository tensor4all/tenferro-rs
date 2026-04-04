use super::types::{col_major_strides, Buffer, TypedTensor};

pub trait TensorData {
    type Scalar;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn as_slice(&self) -> &[Self::Scalar];
    fn from_dense(shape: Vec<usize>, data: Vec<Self::Scalar>) -> Self;
}

impl<T: Clone> TensorData for TypedTensor<T> {
    type Scalar = T;

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[isize] {
        &self.strides
    }

    fn as_slice(&self) -> &[T] {
        match &self.buffer {
            Buffer::Host(v) => v,
            Buffer::Backend(_) => panic!("as_slice called on backend buffer"),
        }
    }

    fn from_dense(shape: Vec<usize>, data: Vec<T>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "data length mismatch");
        let strides = col_major_strides(&shape);
        TypedTensor {
            buffer: Buffer::Host(data),
            shape,
            strides,
            placement: super::types::default_placement(),
            preferred_compute_device: None,
        }
    }
}
