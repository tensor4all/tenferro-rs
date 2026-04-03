mod organization;

use super::common::col_major_flat_index;
use super::*;
use crate::{MaxPlus, MaxPlusAlgebra};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

#[test]
fn test_promote_extract_roundtrip() {
    assert_promote_extract_roundtrip::<MaxPlus<f64>>(
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap(),
    );
}

#[test]
fn test_promote_extract_roundtrip_row_major_and_empty() {
    assert_promote_extract_roundtrip::<MaxPlus<f64>>(
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::RowMajor).unwrap(),
    );
    assert_promote_extract_roundtrip::<MaxPlus<f64>>(
        Tensor::<f64>::from_slice(&[], &[0, 2], COL).unwrap(),
    );
}

fn assert_promote_extract_roundtrip<T>(tensor: Tensor<T::Inner>)
where
    T: TropicalScalar,
    T::Inner: PartialEq + std::fmt::Debug,
{
    let tropical = promote_to_tropical::<T>(&tensor).unwrap();
    let back = extract_inner::<T>(&tropical).unwrap();
    let expected = tensor.contiguous(COL);
    assert_eq!(back.dims(), expected.dims());
    assert_eq!(back.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn test_col_major_flat_index() {
    // 2x3 matrix
    assert_eq!(col_major_flat_index(&[2, 3], &[0, 0]), 0);
    assert_eq!(col_major_flat_index(&[2, 3], &[1, 0]), 1);
    assert_eq!(col_major_flat_index(&[2, 3], &[0, 1]), 2);
    assert_eq!(col_major_flat_index(&[2, 3], &[1, 1]), 3);
    assert_eq!(col_major_flat_index(&[2, 3], &[0, 2]), 4);
    assert_eq!(col_major_flat_index(&[2, 3], &[1, 2]), 5);
}

#[test]
fn tropical_einsum_frule_routes_unary_tangent_to_winner() {
    let mut ctx = CpuContext::new(1);
    let primal = Tensor::<MaxPlus<f64>>::from_slice(
        &[MaxPlus(1.0), MaxPlus(5.0), MaxPlus(4.0), MaxPlus(2.0)],
        &[2, 2],
        COL,
    )
    .unwrap();
    let tangent = Tensor::<f64>::from_slice(&[10.0, 20.0, 30.0, 40.0], &[2, 2], COL).unwrap();

    let output = tropical_einsum_frule::<MaxPlus<f64>, MaxPlusAlgebra<f64>, CpuBackend>(
        &mut ctx,
        "ij->j",
        &[&primal],
        &[Some(&tangent)],
    )
    .unwrap();

    assert_eq!(output.dims(), &[2]);
    assert_eq!(output.buffer().as_slice().unwrap(), &[20.0, 30.0]);
}

#[test]
fn tropical_einsum_frule_accumulates_binary_tangent_contributions() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::<MaxPlus<f64>>::from_slice(
        &[1.0, 2.0, 3.0, 4.0]
            .iter()
            .copied()
            .map(MaxPlus)
            .collect::<Vec<_>>()
            .as_slice(),
        &[2, 2],
        COL,
    )
    .unwrap();
    let b = Tensor::<MaxPlus<f64>>::from_slice(
        &[5.0, 6.0, 7.0, 8.0]
            .iter()
            .copied()
            .map(MaxPlus)
            .collect::<Vec<_>>()
            .as_slice(),
        &[2, 2],
        COL,
    )
    .unwrap();
    let da = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let db = Tensor::<f64>::from_slice(&[10.0, 20.0, 30.0, 40.0], &[2, 2], COL).unwrap();

    let output = tropical_einsum_frule::<MaxPlus<f64>, MaxPlusAlgebra<f64>, CpuBackend>(
        &mut ctx,
        "ij,jk->ik",
        &[&a, &b],
        &[Some(&da), Some(&db)],
    )
    .unwrap();

    assert_eq!(output.dims(), &[2, 2]);
    assert_eq!(
        output.buffer().as_slice().unwrap(),
        &[23.0, 24.0, 43.0, 44.0]
    );
}
