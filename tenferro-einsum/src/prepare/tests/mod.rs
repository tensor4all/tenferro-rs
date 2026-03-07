use tenferro_algebra::Standard;
use tenferro_device::LogicalMemorySpace;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::MemoryOrder;

use super::*;
use crate::util::{tensor_get, unflatten_index};

fn tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

fn assert_tensor_close(lhs: &Tensor<f64>, rhs: &Tensor<f64>) {
    assert_eq!(lhs.dims(), rhs.dims());
    let numel: usize = lhs.dims().iter().product();
    for flat in 0..numel {
        let idx = unflatten_index(flat, lhs.dims());
        let l = tensor_get(lhs, &idx);
        let r = tensor_get(rhs, &idx);
        assert!(
            (l - r).abs() < 1e-10,
            "mismatch at {:?}: left={} right={}",
            idx,
            l,
            r
        );
    }
}

#[test]
fn prepare_one_operand_zero_copy_fuses_groups() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let input = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let prepared = prepare_one_operand::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &input,
        &[0, 1],
        &[0, 1],
        0,
        1,
        1,
        &[2, 2],
        &mut pool,
    )
    .unwrap();

    assert_eq!(prepared.dims(), &[2, 2]);
    assert_eq!(prepared.buffer().as_ptr(), input.buffer().as_ptr());
}

#[test]
fn prepare_one_operand_partial_fallback_when_group2_nonfusable() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let mut ref_pool = BufferPool::new();
    let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let input = tensor(&data, &[2, 3, 4]);
    let fallback_shape = [3, 8];

    let prepared = prepare_one_operand::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &input,
        &[0, 1, 2],
        &[1, 2, 0],
        0,
        1,
        2,
        &fallback_shape,
        &mut pool,
    )
    .unwrap();
    let expected = permute_or_copy::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &input,
        &[0, 1, 2],
        &[1, 2, 0],
        &mut ref_pool,
    )
    .unwrap()
    .reshape(&fallback_shape)
    .unwrap();

    assert_eq!(prepared.dims(), &fallback_shape);
    assert!(prepared.is_contiguous());
    assert_tensor_close(&prepared, &expected);
}

#[test]
fn prepare_one_operand_partial_fallback_when_group1_nonfusable() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let mut ref_pool = BufferPool::new();
    let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
    let input = tensor(&data, &[2, 3, 4]);
    let fallback_shape = [8, 3];

    let prepared = prepare_one_operand::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &input,
        &[0, 1, 2],
        &[2, 0, 1],
        0,
        2,
        1,
        &fallback_shape,
        &mut pool,
    )
    .unwrap();
    let expected = permute_or_copy::<Standard<f64>, CpuBackend>(
        &mut ctx,
        &input,
        &[0, 1, 2],
        &[2, 0, 1],
        &mut ref_pool,
    )
    .unwrap()
    .reshape(&fallback_shape)
    .unwrap();

    assert_eq!(prepared.dims(), &fallback_shape);
    assert!(prepared.is_contiguous());
    assert_tensor_close(&prepared, &expected);
}

#[test]
fn permute_or_copy_transpose_materializes_contiguous_copy() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let input = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let prepared =
        permute_or_copy::<Standard<f64>, CpuBackend>(&mut ctx, &input, &[0, 1], &[1, 0], &mut pool)
            .unwrap();

    assert_eq!(prepared.dims(), &[3, 2]);
    assert!(prepared.is_contiguous());
    assert!((tensor_get(&prepared, &[0, 0]) - 1.0).abs() < 1e-10);
    assert!((tensor_get(&prepared, &[1, 0]) - 3.0).abs() < 1e-10);
    assert!((tensor_get(&prepared, &[2, 0]) - 5.0).abs() < 1e-10);
    assert!((tensor_get(&prepared, &[0, 1]) - 2.0).abs() < 1e-10);
}

#[test]
fn permute_or_copy_returns_contiguous_view_for_unit_extent_permute() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let input = tensor(&[1.0, 2.0], &[2, 1]);

    let prepared =
        permute_or_copy::<Standard<f64>, CpuBackend>(&mut ctx, &input, &[0, 1], &[1, 0], &mut pool)
            .unwrap();

    assert_eq!(prepared.dims(), &[1, 2]);
    assert!(prepared.is_contiguous());
    assert_eq!(prepared.buffer().as_ptr(), input.buffer().as_ptr());
}

#[test]
fn make_contiguous_if_needed_copies_only_when_required() {
    let mut ctx = CpuContext::new(1);
    let mut pool = BufferPool::new();
    let contiguous = tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let shared =
        make_contiguous_if_needed::<Standard<f64>, CpuBackend>(&mut ctx, &contiguous, &mut pool)
            .unwrap();
    assert_eq!(shared.buffer().as_ptr(), contiguous.buffer().as_ptr());

    let transposed = contiguous.permute(&[1, 0]).unwrap();
    let copied =
        make_contiguous_if_needed::<Standard<f64>, CpuBackend>(&mut ctx, &transposed, &mut pool)
            .unwrap();

    assert!(copied.is_contiguous());
    assert_eq!(copied.dims(), &[2, 2]);
    assert_eq!(
        copied.logical_memory_space(),
        LogicalMemorySpace::MainMemory
    );
    assert!((tensor_get(&copied, &[0, 0]) - 1.0).abs() < 1e-10);
    assert!((tensor_get(&copied, &[1, 0]) - 3.0).abs() < 1e-10);
    assert!((tensor_get(&copied, &[0, 1]) - 2.0).abs() < 1e-10);
}
