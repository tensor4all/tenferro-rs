//! Forward-mode (inference) implementation of [`TensorNetworkOps`] for the
//! NdArray backend.

use burn::backend::NdArray;
use burn::tensor::ops::FloatTensor;

use crate::TensorNetworkOps;

impl TensorNetworkOps for NdArray<f64> {
    fn tn_einsum(_subscripts: &str, _inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        todo!()
    }
}
