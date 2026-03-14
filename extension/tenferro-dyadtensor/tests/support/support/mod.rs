#![allow(dead_code)]

use chainrules::Tape;
use num_complex::Complex64;
use tenferro_dyadtensor::{AdTensor, DynAdTensor, StructuredTensor};
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
    AdTensor::new_forward(scalar_f64(primal), scalar_f64(tangent))
        .unwrap()
        .into()
}

pub(crate) fn forward_rank0_c64(primal: Complex64, tangent: Complex64) -> DynAdTensor {
    AdTensor::new_forward(scalar_c64(primal), scalar_c64(tangent))
        .unwrap()
        .into()
}

pub(crate) fn primal_rank0_f64(primal: f64) -> DynAdTensor {
    AdTensor::new_primal(scalar_f64(primal)).into()
}

pub(crate) fn primal_rank0_c64(primal: Complex64) -> DynAdTensor {
    AdTensor::new_primal(scalar_c64(primal)).into()
}

pub(crate) fn reverse_rank0_f64(primal: f64, tape: &Tape<tenferro_dyadtensor::DynTensor>) -> DynAdTensor {
    AdTensor::new_reverse_leaf(scalar_f64(primal), tape)
        .unwrap()
        .into()
}

pub(crate) fn reverse_rank0_c64(
    primal: Complex64,
    tape: &Tape<tenferro_dyadtensor::DynTensor>,
) -> DynAdTensor {
    AdTensor::new_reverse_leaf(scalar_c64(primal), tape)
        .unwrap()
        .into()
}

pub(crate) fn reverse_vector_f64(
    values: &[f64],
    tape: &Tape<tenferro_dyadtensor::DynTensor>,
) -> DynAdTensor {
    AdTensor::new_reverse_leaf(vector_f64(values), tape)
        .unwrap()
        .into()
}

pub(crate) fn reverse_vector_c64(
    values: &[Complex64],
    tape: &Tape<tenferro_dyadtensor::DynTensor>,
) -> DynAdTensor {
    AdTensor::new_reverse_leaf(vector_c64(values), tape)
        .unwrap()
        .into()
}

pub(crate) fn rank0_value_f64(tensor: &StructuredTensor<f64>) -> f64 {
    tensor.payload().buffer().as_slice().unwrap()[0]
}

pub(crate) fn rank0_value_c64(tensor: &StructuredTensor<Complex64>) -> Complex64 {
    tensor.payload().buffer().as_slice().unwrap()[0]
}
