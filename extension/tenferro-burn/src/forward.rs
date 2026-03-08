//! Forward-mode (inference) implementation of [`TensorNetworkOps`] for the
//! NdArray backend.
//!
//! # Current Limitations
//!
//! `TensorNetworkOps` is currently only implemented for `NdArray<f64>`.
//! Implementations for other backends (e.g., `Wgpu`, `LibTorch`) and element
//! types (e.g., `f32`) will be added in future versions.

use burn::backend::NdArray;
use burn::tensor::ops::FloatTensor;

use crate::TensorNetworkOps;

impl TensorNetworkOps for NdArray<f64> {
    fn tn_einsum(_subscripts: &str, _inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        todo!()
    }
}
