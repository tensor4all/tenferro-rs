mod organization;

use super::common::col_major_flat_index;
use super::*;
use crate::{MaxPlus, MaxPlusAlgebra};
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

const COL: MemoryOrder = MemoryOrder::ColumnMajor;

#[test]
fn test_promote_extract_roundtrip() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], COL).unwrap();
    let tropical = promote_to_tropical::<MaxPlus<f64>>(&t).unwrap();
    let back = extract_inner::<MaxPlus<f64>>(&tropical).unwrap();
    let orig_data = t.buffer().as_slice().unwrap();
    let back_data = back.buffer().as_slice().unwrap();
    for i in 0..4 {
        assert_eq!(orig_data[i], back_data[i]);
    }
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
