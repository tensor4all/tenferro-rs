#![allow(dead_code)]

use num_complex::{Complex32, Complex64};
use tenferro::{forward_ad, grad, GradOptions, Tensor};
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

pub(crate) fn vector_f32(values: &[f32]) -> DenseTensor<f32> {
    DenseTensor::<f32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn vector_f64(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn vector_c32(values: &[Complex32]) -> DenseTensor<Complex32> {
    DenseTensor::<Complex32>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn vector_c64(values: &[Complex64]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn scalar_f32(value: f32) -> DenseTensor<f32> {
    DenseTensor::<f32>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn scalar_f64(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn scalar_c32(value: Complex32) -> DenseTensor<Complex32> {
    DenseTensor::<Complex32>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn scalar_c64(value: Complex64) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(crate) fn forward_rank0_f32(primal: f32, tangent: f32) -> Tensor {
    forward_tensor_f32(scalar_f32(primal), scalar_f32(tangent))
}

pub(crate) fn forward_rank0_f64(primal: f64, tangent: f64) -> Tensor {
    forward_tensor_f64(scalar_f64(primal), scalar_f64(tangent))
}

pub(crate) fn forward_rank0_c32(primal: Complex32, tangent: Complex32) -> Tensor {
    forward_tensor_c32(scalar_c32(primal), scalar_c32(tangent))
}

pub(crate) fn forward_rank0_c64(primal: Complex64, tangent: Complex64) -> Tensor {
    forward_tensor_c64(scalar_c64(primal), scalar_c64(tangent))
}

pub(crate) fn forward_tensor_f32(primal: DenseTensor<f32>, tangent: DenseTensor<f32>) -> Tensor {
    forward_ad::dual_level(|fw| {
        fw.make_dual(&Tensor::from_tensor(primal), &Tensor::from_tensor(tangent))
    })
    .unwrap()
}

pub(crate) fn forward_tensor_f64(primal: DenseTensor<f64>, tangent: DenseTensor<f64>) -> Tensor {
    forward_ad::dual_level(|fw| {
        fw.make_dual(&Tensor::from_tensor(primal), &Tensor::from_tensor(tangent))
    })
    .unwrap()
}

pub(crate) fn forward_tensor_c32(
    primal: DenseTensor<Complex32>,
    tangent: DenseTensor<Complex32>,
) -> Tensor {
    forward_ad::dual_level(|fw| {
        fw.make_dual(&Tensor::from_tensor(primal), &Tensor::from_tensor(tangent))
    })
    .unwrap()
}

pub(crate) fn forward_tensor_c64(
    primal: DenseTensor<Complex64>,
    tangent: DenseTensor<Complex64>,
) -> Tensor {
    forward_ad::dual_level(|fw| {
        fw.make_dual(&Tensor::from_tensor(primal), &Tensor::from_tensor(tangent))
    })
    .unwrap()
}

pub(crate) fn primal_rank0_f32(primal: f32) -> Tensor {
    Tensor::from_tensor(scalar_f32(primal))
}

pub(crate) fn primal_rank0_f64(primal: f64) -> Tensor {
    Tensor::from_tensor(scalar_f64(primal))
}

pub(crate) fn primal_rank0_c32(primal: Complex32) -> Tensor {
    Tensor::from_tensor(scalar_c32(primal))
}

pub(crate) fn primal_rank0_c64(primal: Complex64) -> Tensor {
    Tensor::from_tensor(scalar_c64(primal))
}

pub(crate) fn reverse_rank0_f32(primal: f32) -> Tensor {
    let mut tensor = Tensor::from_tensor(scalar_f32(primal));
    tensor.set_requires_grad(true).unwrap();
    tensor
}

pub(crate) fn reverse_rank0_f64(primal: f64) -> Tensor {
    let mut tensor = Tensor::from_tensor(scalar_f64(primal));
    tensor.set_requires_grad(true).unwrap();
    tensor
}

pub(crate) fn reverse_rank0_f32_like(primal: f32, anchor: &Tensor) -> Tensor {
    let _ = anchor;
    reverse_rank0_f32(primal)
}

pub(crate) fn reverse_rank0_f64_like(primal: f64, anchor: &Tensor) -> Tensor {
    let _ = anchor;
    reverse_rank0_f64(primal)
}

pub(crate) fn reverse_rank0_c32(primal: Complex32) -> Tensor {
    let mut tensor = Tensor::from_tensor(scalar_c32(primal));
    tensor.set_requires_grad(true).unwrap();
    tensor
}

pub(crate) fn reverse_rank0_c64(primal: Complex64) -> Tensor {
    let mut tensor = Tensor::from_tensor(scalar_c64(primal));
    tensor.set_requires_grad(true).unwrap();
    tensor
}

pub(crate) fn reverse_rank0_c32_like(primal: Complex32, anchor: &Tensor) -> Tensor {
    let _ = anchor;
    reverse_rank0_c32(primal)
}

pub(crate) fn reverse_rank0_c64_like(primal: Complex64, anchor: &Tensor) -> Tensor {
    let _ = anchor;
    reverse_rank0_c64(primal)
}

pub(crate) fn reverse_vector_f64(values: &[f64]) -> Tensor {
    let mut tensor = Tensor::from_tensor(vector_f64(values));
    tensor.set_requires_grad(true).unwrap();
    tensor
}

pub(crate) fn reverse_vector_f64_like(values: &[f64], anchor: &Tensor) -> Tensor {
    let _ = anchor;
    reverse_vector_f64(values)
}

pub(crate) fn reverse_vector_c64(values: &[Complex64]) -> Tensor {
    let mut tensor = Tensor::from_tensor(vector_c64(values));
    tensor.set_requires_grad(true).unwrap();
    tensor
}

pub(crate) fn reverse_vector_c64_like(values: &[Complex64], anchor: &Tensor) -> Tensor {
    let _ = anchor;
    reverse_vector_c64(values)
}

pub(crate) fn diag_f64(values: &[f64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector_f64(values))).unwrap()
}

pub(crate) fn diag_c64(values: &[Complex64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector_c64(values))).unwrap()
}

pub(crate) fn with_axis_classes_f64(
    values: &[f64],
    payload_dims: &[usize],
    axis_classes: &[usize],
) -> Tensor {
    Tensor::with_axis_classes(
        Tensor::from_tensor(
            DenseTensor::<f64>::from_slice(values, payload_dims, MemoryOrder::ColumnMajor).unwrap(),
        ),
        axis_classes,
    )
    .unwrap()
}

pub(crate) fn grad_wrt(
    output: &Tensor,
    cotangent: &Tensor,
    wrt: &[&Tensor],
) -> Vec<Option<Tensor>> {
    let grad_outputs = [cotangent.clone()];
    grad(&[output], wrt, Some(&grad_outputs), GradOptions::default()).unwrap()
}

pub(crate) fn rank0_value_f64(tensor: &Tensor) -> f64 {
    tensor
        .as_f64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap()[0]
}

pub(crate) fn rank0_value_c32(tensor: &Tensor) -> Complex32 {
    tensor
        .as_c32()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap()[0]
}

pub(crate) fn rank0_value_c64(tensor: &Tensor) -> Complex64 {
    tensor
        .as_c64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap()[0]
}

pub(crate) fn dyn_values_f64(tensor: &Tensor) -> &[f64] {
    tensor
        .as_f64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap()
}

pub(crate) fn dyn_rank0_value_f64(tensor: &Tensor) -> f64 {
    dyn_values_f64(tensor)[0]
}
