#![allow(dead_code)]

use chainrules_core::NodeId;
use num_complex::Complex64;
use tenferro_algebra::Scalar;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};
use tidu::Tape;

use crate::core::AdMode;
use crate::structured::StructuredTensor;
use crate::AdTensor;

pub(super) trait TensorLike<T: Scalar> {
    fn tensor_ref(&self) -> &DenseTensor<T>;
}

impl<T: Scalar> TensorLike<T> for DenseTensor<T> {
    fn tensor_ref(&self) -> &DenseTensor<T> {
        self
    }
}

impl<T: Scalar> TensorLike<T> for StructuredTensor<T> {
    fn tensor_ref(&self) -> &DenseTensor<T> {
        self.payload()
    }
}

pub(super) fn f64_2x2(values: [f64; 4]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

pub(super) fn c64_2x2(values: [Complex64; 4]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(&values, &[2, 2], MemoryOrder::ColumnMajor).unwrap()
}

pub(super) fn scalar_f64(value: f64) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(super) fn scalar_c64(value: Complex64) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

pub(super) fn reverse_leaf_f64(
    tensor: impl Into<StructuredTensor<f64>>,
    tape: &Tape<crate::DynTensor>,
) -> AdTensor<f64> {
    AdTensor::new_reverse_leaf(tensor, tape).unwrap()
}

pub(super) fn reverse_leaf_c64(
    tensor: impl Into<StructuredTensor<Complex64>>,
    tape: &Tape<crate::DynTensor>,
) -> AdTensor<Complex64> {
    AdTensor::new_reverse_leaf(tensor, tape).unwrap()
}

pub(super) fn assert_forward_mode<T: Scalar>(tensor: &AdTensor<T>) {
    assert_eq!(tensor.mode(), AdMode::Forward);
    assert!(tensor.tangent().is_some());
    assert!(tensor.node_id().is_none());
}

pub(super) fn assert_reverse_on_tape<T: Scalar>(
    tensor: &AdTensor<T>,
    tape: &Tape<crate::DynTensor>,
) {
    assert_eq!(tensor.mode(), AdMode::Reverse);
    assert!(tensor.node_id().is_some());
    let actual_tape = tensor
        .tape()
        .expect("reverse tensor should expose its tape");
    assert!(actual_tape.same_tape(tape));
}

pub(super) fn node_id<T: Scalar>(tensor: &AdTensor<T>) -> NodeId {
    tensor
        .node_id()
        .expect("reverse tensor should expose a node id")
}

pub(super) fn as_slice<T>(t: &T) -> &[f64]
where
    T: TensorLike<f64> + ?Sized,
{
    t.tensor_ref()
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))
}

pub(super) fn tensor_to_vec_f64<T>(t: &T) -> Vec<f64>
where
    T: TensorLike<f64> + ?Sized,
{
    let t = t.tensor_ref().contiguous(MemoryOrder::ColumnMajor);
    let off = t.offset() as usize;
    let len: usize = t.dims().iter().product();
    t.buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))[off..off + len]
        .to_vec()
}

pub(super) fn tensor_to_vec_c64<T>(t: &T) -> Vec<Complex64>
where
    T: TensorLike<Complex64> + ?Sized,
{
    let t = t.tensor_ref().contiguous(MemoryOrder::ColumnMajor);
    let off = t.offset() as usize;
    let len: usize = t.dims().iter().product();
    t.buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))[off..off + len]
        .to_vec()
}

pub(super) fn tensor_from_vec_f64(data: &[f64], dims: &[usize]) -> DenseTensor<f64> {
    DenseTensor::<f64>::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

pub(super) fn tensor_from_vec_c64(data: &[Complex64], dims: &[usize]) -> DenseTensor<Complex64> {
    DenseTensor::<Complex64>::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

pub(super) fn max_abs_diff<A, B>(a: &A, b: &B) -> f64
where
    A: TensorLike<f64> + ?Sized,
    B: TensorLike<f64> + ?Sized,
{
    assert_eq!(a.tensor_ref().dims(), b.tensor_ref().dims());
    tensor_to_vec_f64(a)
        .iter()
        .zip(tensor_to_vec_f64(b).iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

pub(super) fn complex_max_abs_diff<A, B>(a: &A, b: &B) -> f64
where
    A: TensorLike<Complex64> + ?Sized,
    B: TensorLike<Complex64> + ?Sized,
{
    assert_eq!(a.tensor_ref().dims(), b.tensor_ref().dims());
    let a = a.tensor_ref().contiguous(MemoryOrder::ColumnMajor);
    let b = b.tensor_ref().contiguous(MemoryOrder::ColumnMajor);
    let a_off = a.offset() as usize;
    let b_off = b.offset() as usize;
    let len: usize = a.dims().iter().product();
    let a_data = &a
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))[a_off..a_off + len];
    let b_data = &b
        .buffer()
        .as_slice()
        .unwrap_or_else(|| panic!("expected CPU-backed contiguous tensor"))[b_off..b_off + len];
    a_data
        .iter()
        .zip(b_data.iter())
        .map(|(x, y)| (*x - *y).norm())
        .fold(0.0_f64, f64::max)
}

pub(super) fn add_scaled_f64(
    base: &DenseTensor<f64>,
    direction: &DenseTensor<f64>,
    alpha: f64,
) -> DenseTensor<f64> {
    assert_eq!(base.dims(), direction.dims());
    let data = tensor_to_vec_f64(base);
    let dir = tensor_to_vec_f64(direction);
    let out: Vec<f64> = data
        .iter()
        .zip(dir.iter())
        .map(|(x, d)| *x + alpha * *d)
        .collect();
    tensor_from_vec_f64(&out, base.dims())
}

pub(super) fn add_scaled_c64(
    base: &DenseTensor<Complex64>,
    direction: &DenseTensor<Complex64>,
    alpha: f64,
) -> DenseTensor<Complex64> {
    assert_eq!(base.dims(), direction.dims());
    let data = tensor_to_vec_c64(base);
    let dir = tensor_to_vec_c64(direction);
    let out: Vec<Complex64> = data
        .iter()
        .zip(dir.iter())
        .map(|(x, d)| *x + *d * alpha)
        .collect();
    tensor_from_vec_c64(&out, base.dims())
}

pub(super) fn scale_f64(t: &DenseTensor<f64>, alpha: f64) -> DenseTensor<f64> {
    let data: Vec<f64> = tensor_to_vec_f64(t)
        .into_iter()
        .map(|x| x * alpha)
        .collect();
    tensor_from_vec_f64(&data, t.dims())
}

pub(super) fn scale_c64(t: &DenseTensor<Complex64>, alpha: Complex64) -> DenseTensor<Complex64> {
    let data: Vec<Complex64> = tensor_to_vec_c64(t)
        .into_iter()
        .map(|x| x * alpha)
        .collect();
    tensor_from_vec_c64(&data, t.dims())
}

pub(super) fn central_diff_f64(
    plus: &DenseTensor<f64>,
    minus: &DenseTensor<f64>,
    eps: f64,
) -> DenseTensor<f64> {
    assert_eq!(plus.dims(), minus.dims());
    let dims = plus.dims().to_vec();
    let plus_data = tensor_to_vec_f64(plus);
    let minus_data = tensor_to_vec_f64(minus);
    let out: Vec<f64> = plus_data
        .iter()
        .zip(minus_data.iter())
        .map(|(p, m)| (*p - *m) / (2.0 * eps))
        .collect();
    tensor_from_vec_f64(&out, &dims)
}

pub(super) fn central_diff_c64(
    plus: &DenseTensor<Complex64>,
    minus: &DenseTensor<Complex64>,
    eps: f64,
) -> DenseTensor<Complex64> {
    assert_eq!(plus.dims(), minus.dims());
    let dims = plus.dims().to_vec();
    let plus_data = tensor_to_vec_c64(plus);
    let minus_data = tensor_to_vec_c64(minus);
    let out: Vec<Complex64> = plus_data
        .iter()
        .zip(minus_data.iter())
        .map(|(p, m)| (*p - *m) / (2.0 * eps))
        .collect();
    tensor_from_vec_c64(&out, &dims)
}

pub(super) fn sum_mul_f64<A, B>(a: &A, b: &B) -> f64
where
    A: TensorLike<f64> + ?Sized,
    B: TensorLike<f64> + ?Sized,
{
    assert_eq!(a.tensor_ref().dims(), b.tensor_ref().dims());
    tensor_to_vec_f64(a)
        .iter()
        .zip(tensor_to_vec_f64(b).iter())
        .map(|(x, y)| x * y)
        .sum()
}

pub(super) fn sum_mul_c64<A, B>(a: &A, b: &B) -> Complex64
where
    A: TensorLike<Complex64> + ?Sized,
    B: TensorLike<Complex64> + ?Sized,
{
    assert_eq!(a.tensor_ref().dims(), b.tensor_ref().dims());
    tensor_to_vec_c64(a)
        .iter()
        .zip(tensor_to_vec_c64(b).iter())
        .map(|(x, y)| *x * *y)
        .sum()
}

pub(super) fn sum_conj_mul_real_c64<A, B>(a: &A, b: &B) -> f64
where
    A: TensorLike<Complex64> + ?Sized,
    B: TensorLike<Complex64> + ?Sized,
{
    assert_eq!(a.tensor_ref().dims(), b.tensor_ref().dims());
    tensor_to_vec_c64(a)
        .iter()
        .zip(tensor_to_vec_c64(b).iter())
        .map(|(x, y)| (x.conj() * *y).re)
        .sum()
}
