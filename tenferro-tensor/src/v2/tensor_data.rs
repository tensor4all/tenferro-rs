pub trait TensorData {
    type Scalar;
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[isize];
    fn as_slice(&self) -> &[Self::Scalar];
    fn from_dense(shape: Vec<usize>, data: Vec<Self::Scalar>) -> Self;
}
