//! Forward-mode (inference) implementation of [`TensorNetworkOps`] for the
//! NdArray backend.
//!
//! # Current Limitations
//!
//! The concrete forward backend implementation is currently `NdArray<f64>`.
//! `Autodiff<NdArray<f64>>` builds on top of this in [`crate::backward`].
//! Other backends and element types remain future work.

use burn::backend::NdArray;
use burn::tensor::ops::FloatTensor;

use crate::{primitive_einsum, TensorNetworkOps};

impl TensorNetworkOps for NdArray<f64> {
    fn tn_einsum(subscripts: &str, inputs: Vec<FloatTensor<Self>>) -> FloatTensor<Self> {
        primitive_einsum::<Self>(subscripts, inputs)
    }
}
