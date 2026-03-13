use std::collections::HashMap;

use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};

use super::*;
use crate::api::einsum_with_subscripts;

fn tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn manual_einsum_matches_matmul() {
    let mut ctx = CpuContext::new(1);
    let subs = Subscripts::parse("ij,jk->ik").unwrap();
    let a = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let size_dict = HashMap::from([
        ('i' as u32, 2usize),
        ('j' as u32, 2usize),
        ('k' as u32, 2usize),
    ]);

    let manual = manual_einsum(&subs, &[a.clone(), b.clone()], &size_dict).unwrap();
    let backend =
        einsum_with_subscripts::<Standard<f64>, CpuBackend>(&mut ctx, &subs, &[&a, &b], None)
            .unwrap();

    assert_eq!(manual.dims(), &[2, 2]);
    assert_eq!(manual.buffer().as_slice(), backend.buffer().as_slice());
}

#[test]
fn manual_einsum_trace_returns_scalar() {
    let subs = Subscripts::parse("ii->").unwrap();
    let a = tensor(&[1.0, 0.0, 0.0, 3.0], &[2, 2]);
    let size_dict = HashMap::from([('i' as u32, 2usize)]);

    let result = manual_einsum(&subs, &[a], &size_dict).unwrap();
    let data = result.buffer().as_slice().unwrap();

    assert!(result.dims().is_empty());
    assert!((data[result.offset() as usize] - 4.0).abs() < 1e-10);
}
