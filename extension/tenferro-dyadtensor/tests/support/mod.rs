#![allow(dead_code)]

use num_complex::Complex64;
use tenferro_dyadtensor::{DynAdTensor, DynTape, StructuredTensor};
use tenferro_tensor::{MemoryOrder, Tensor};

pub(crate) fn vector_f64(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn vector_c64(values: &[Complex64]) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn scalar_f64(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn scalar_c64(value: Complex64) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn forward_rank0_f64(primal: f64, tangent: f64) -> DynAdTensor {
    DynAdTensor::new_forward(scalar_f64(primal), scalar_f64(tangent)).unwrap()
}

pub(crate) fn forward_rank0_c64(primal: Complex64, tangent: Complex64) -> DynAdTensor {
    DynAdTensor::new_forward(scalar_c64(primal), scalar_c64(tangent)).unwrap()
}

pub(crate) fn primal_rank0_f64(primal: f64) -> DynAdTensor {
    DynAdTensor::new_primal(scalar_f64(primal))
}

pub(crate) fn primal_rank0_c64(primal: Complex64) -> DynAdTensor {
    DynAdTensor::new_primal(scalar_c64(primal))
}

pub(crate) fn reverse_rank0_f64(primal: f64, tape: &DynTape) -> DynAdTensor {
    DynAdTensor::new_reverse_leaf(scalar_f64(primal), tape).unwrap()
}

pub(crate) fn reverse_rank0_c64(primal: Complex64, tape: &DynTape) -> DynAdTensor {
    DynAdTensor::new_reverse_leaf(scalar_c64(primal), tape).unwrap()
}

pub(crate) fn reverse_vector_f64(values: &[f64], tape: &DynTape) -> DynAdTensor {
    DynAdTensor::new_reverse_leaf(vector_f64(values), tape).unwrap()
}

pub(crate) fn reverse_vector_c64(values: &[Complex64], tape: &DynTape) -> DynAdTensor {
    DynAdTensor::new_reverse_leaf(vector_c64(values), tape).unwrap()
}

pub(crate) fn rank0_value_f64(tensor: &StructuredTensor<f64>) -> f64 {
    tensor.payload().buffer().as_slice().unwrap()[0]
}

pub(crate) fn rank0_value_c64(tensor: &StructuredTensor<Complex64>) -> Complex64 {
    tensor.payload().buffer().as_slice().unwrap()[0]
}

pub(crate) fn dyn_values_f64(tensor: &DynAdTensor) -> &[f64] {
    tensor
        .as_f64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap()
}

pub(crate) fn dyn_rank0_value_f64(tensor: &DynAdTensor) -> f64 {
    dyn_values_f64(tensor)[0]
}
