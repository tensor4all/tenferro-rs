//! Tests for non-trivial logic in `exec_op_on_tensors`.
//!
//! Trivial forwarding branches (Add, Mul, Exp, etc.) are exercised
//! via EagerTensor AD tests. These tests target branches with real logic:
//! edge-case handling, byte parsing, multi-output unpacking, etc.

use tenferro::eager_exec::exec_op_on_tensors;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_tensor::cpu::CpuBackend;
use tenferro_tensor::{DType, Tensor, TypedTensor};

fn f64t(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec(shape, data))
}

fn scalar(v: f64) -> Tensor {
    f64t(vec![], vec![v])
}

fn data(t: &Tensor) -> Vec<f64> {
    match t {
        Tensor::F64(inner) => inner.host_data().to_vec(),
        _ => panic!("expected F64"),
    }
}

// ── DynamicTruncate: rounding, clamping, edge cases ──

#[test]
fn dynamic_truncate_rounds_non_integer_size() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let size = scalar(2.7); // rounds to 3
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0]);
}

#[test]
fn dynamic_truncate_clamps_negative_to_zero() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let size = scalar(-5.0);
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[0]);
}

#[test]
fn dynamic_truncate_clamps_oversize() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let size = scalar(100.0);
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0]);
}

#[test]
fn dynamic_truncate_nan_gives_empty() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let size = scalar(f64::NAN);
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[0]);
}

// ── PadToMatch: no-op and padding ──

#[test]
fn pad_to_match_noop_when_already_larger() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let reference = f64t(vec![3], vec![0.0; 3]);
    let result = exec_op_on_tensors(
        &StdTensorOp::PadToMatch { axis: 0 },
        &[&x, &reference],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn pad_to_match_pads_with_zeros() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![2], vec![1.0, 2.0]);
    let reference = f64t(vec![5], vec![0.0; 5]);
    let result = exec_op_on_tensors(
        &StdTensorOp::PadToMatch { axis: 0 },
        &[&x, &reference],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 0.0, 0.0, 0.0]);
}

// ── Constant: dtype byte parsing ──

#[test]
fn constant_f64_parses_bytes() {
    let mut b = CpuBackend::new();
    let result = exec_op_on_tensors(
        &StdTensorOp::Constant {
            dtype: DType::F64,
            bytes: 3.14_f64.to_le_bytes().to_vec(),
        },
        &[],
        &mut b,
    )
    .unwrap();
    assert!((data(&result[0])[0] - 3.14).abs() < 1e-12);
}

#[test]
fn constant_f32_parses_bytes() {
    let mut b = CpuBackend::new();
    let result = exec_op_on_tensors(
        &StdTensorOp::Constant {
            dtype: DType::F32,
            bytes: 2.5_f32.to_le_bytes().to_vec(),
        },
        &[],
        &mut b,
    )
    .unwrap();
    assert_eq!(result[0].dtype(), DType::F32);
}

// ── Multi-output linalg ──

#[test]
fn svd_returns_three_outputs() {
    let mut b = CpuBackend::new();
    let m = f64t(vec![3, 2], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let result = exec_op_on_tensors(
        &StdTensorOp::Svd {
            eps: 1e-12,
            input_shape: vec![DimExpr::Const(3), DimExpr::Const(2)],
        },
        &[&m],
        &mut b,
    )
    .unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[1].shape(), &[2]); // S = min(m,n) singular values
}

#[test]
fn lu_returns_four_outputs() {
    let mut b = CpuBackend::new();
    let m = f64t(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]);
    let result = exec_op_on_tensors(
        &StdTensorOp::Lu {
            input_shape: vec![DimExpr::Const(2), DimExpr::Const(2)],
        },
        &[&m],
        &mut b,
    )
    .unwrap();
    assert_eq!(result.len(), 4); // P, L, U, info
}

// ── DimExpr resolution ──

#[test]
fn reshape_resolves_dim_exprs() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = exec_op_on_tensors(
        &StdTensorOp::Reshape {
            from_shape: vec![DimExpr::Const(6)],
            to_shape: vec![DimExpr::Const(2), DimExpr::Const(3)],
        },
        &[&x],
        &mut b,
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[2, 3]);
}

// ── ShapeOf ──

#[test]
fn shape_of_each_axis() {
    let mut b = CpuBackend::new();
    let x = f64t(vec![3, 7, 5], vec![0.0; 105]);
    for (axis, expected) in [(0, 3.0), (1, 7.0), (2, 5.0)] {
        let result = exec_op_on_tensors(&StdTensorOp::ShapeOf { axis }, &[&x], &mut b).unwrap();
        assert_eq!(data(&result[0]), vec![expected]);
    }
}
