//! Value, error, and single-session-entry tests for the session-explicit
//! concrete-ops surface.
//!
//! Covers, per surface (dynamic [`Tensor`] and typed [`TypedTensor`]):
//! equal-shape ops, real broadcast, invalid-broadcast structured errors, dtype
//! conversion, typed output dtype validation (`into_typed_result`), and a
//! deterministic proof that a session chain executes inside exactly one
//! backend session entry.

use tenferro_cpu::CpuEngine;
use tenferro_runtime::{Tensor, TensorSessionOpsExt, TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::backend::{
    TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise, TensorFusion,
    TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, Engine, Error, GatherConfig, PadConfig,
    ScatterConfig, SessionCachedDot, ShapeMismatch, SliceConfig, TensorRead, TensorWrite,
    ValidationError,
};

type TensorResult = tenferro_tensor::Result<Tensor>;

/// Run one op chain on a fresh serial CPU engine.
fn on_engine<R>(f: impl FnOnce(&mut dyn Engine) -> R) -> R {
    let mut engine = CpuEngine::serial();
    f(&mut engine)
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error < 1.0e-12,
            "value {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}

/// Permute a rank-2 col-major layout with `perm = [1, 0]`, in plain math.
fn transpose_col_major(values: &[f64], shape: &[usize]) -> Vec<f64> {
    let (rows, cols) = (shape[0], shape[1]);
    let mut out = vec![0.0; values.len()];
    for row in 0..rows {
        for col in 0..cols {
            out[row * cols + col] = values[col * rows + row];
        }
    }
    out
}

fn assert_incompatible_shapes_error(error: Error) {
    match error {
        Error::Validation {
            op: "broadcast",
            source: ValidationError::ShapeMismatch(shape),
        } => assert!(
            matches!(
                shape.as_ref(),
                ShapeMismatch::IncompatibleShapes { lhs, rhs }
                    if lhs.as_slice() == [2] && rhs.as_slice() == [3]
            ),
            "unexpected shape payload: {shape:?}"
        ),
        other => panic!("expected broadcast ShapeMismatch, got {other:?}"),
    }
}

fn assert_rank_mismatch_error(error: Error) {
    match error {
        Error::Validation {
            op: "matmul",
            source: ValidationError::RankMismatch { expected, actual },
        } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        other => panic!("expected matmul RankMismatch, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Dynamic surface (TensorSessionOpsExt)
// ---------------------------------------------------------------------------

#[test]
fn session_dynamic_equal_shape_chain() {
    let a_values = vec![1.5_f64; 8];
    let b_values = vec![2.25_f64; 8];
    let a = Tensor::from_vec_col_major(vec![8], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let session = on_engine(|s| {
        let x = a.add(&b, s).unwrap();
        let x = x.exp(s).unwrap();
        let x = x.mul(&b, s).unwrap();
        x.reduce_sum(Some(&[0]), s).unwrap()
    });

    // Independent value check in plain scalar math: the chain computes
    // sum_i exp(a_i + b_i) * b_i without any tenferro op, so a shared
    // one-shot regression cannot satisfy it.
    let expected: f64 = a_values
        .iter()
        .zip(&b_values)
        .map(|(&x, &y)| (x + y).exp() * y)
        .sum();
    assert_close(session.as_slice::<f64>().unwrap(), &[expected]);
}

#[test]
fn session_dynamic_broadcast_chain() {
    let a_values = vec![1.0_f64];
    let b_values = vec![2.0_f64; 8];
    let a = Tensor::from_vec_col_major(vec![1], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let session = on_engine(|s| {
        let x = a.add(&b, s).unwrap();
        let x = x.exp(s).unwrap();
        let x = x.mul(&a, s).unwrap();
        x.reduce_sum(Some(&[0]), s).unwrap()
    });

    // Independent value check: the broadcast chain computes
    // sum_i exp(a_0 + b_i) * a_0 in plain scalar math.
    let expected: f64 = b_values
        .iter()
        .map(|&y| (a_values[0] + y).exp() * a_values[0])
        .sum();
    assert_close(session.as_slice::<f64>().unwrap(), &[expected]);
}

#[test]
fn session_dynamic_invalid_broadcast_error() {
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let session_error = on_engine(|s| a.add(&b, s)).unwrap_err();

    assert_incompatible_shapes_error(session_error);
}

#[test]
fn session_dynamic_dtype_error() {
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![1_i32; 2]).unwrap();

    let session_error = on_engine(|s| a.add(&b, s)).unwrap_err();

    // Assert the full payload (op name, source lhs dtype as expected, rhs
    // dtype as actual) for both paths, not just the error kind.
    let Error::Validation {
        op: "add",
        source: ValidationError::DTypeMismatch { expected, actual },
    } = &session_error
    else {
        panic!("expected add DTypeMismatch, got {session_error:?}");
    };
    assert_eq!(*expected, tenferro_tensor::core::DType::F64);
    assert_eq!(*actual, tenferro_tensor::core::DType::I32);
}

// ---------------------------------------------------------------------------
// Typed surface (TypedTensorSessionOpsExt)
// ---------------------------------------------------------------------------

#[test]
fn session_typed_equal_shape_chain() {
    let a_values = vec![1.5_f64; 8];
    let b_values = vec![2.25_f64; 8];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![8], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let session = on_engine(|s| {
        let x = a.add(&b, s).unwrap();
        let x = x.exp(s).unwrap();
        let x = x.mul(&b, s).unwrap();
        x.reduce_sum(Some(&[0]), s).unwrap()
    });

    // Independent value check in plain scalar math (see the dynamic twin).
    let expected: f64 = a_values
        .iter()
        .zip(&b_values)
        .map(|(&x, &y)| (x + y).exp() * y)
        .sum();
    assert_close(session.host_data().unwrap(), &[expected]);
}

#[test]
fn session_typed_broadcast_chain() {
    let a_values = vec![1.0_f64];
    let b_values = vec![2.0_f64; 8];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![1], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let session = on_engine(|s| {
        let x = a.add(&b, s).unwrap();
        let x = x.exp(s).unwrap();
        let x = x.mul(&a, s).unwrap();
        x.reduce_sum(Some(&[0]), s).unwrap()
    });

    // Independent value check: sum_i exp(a_0 + b_i) * a_0 in plain scalar math.
    let expected: f64 = b_values
        .iter()
        .map(|&y| (a_values[0] + y).exp() * a_values[0])
        .sum();
    assert_close(session.host_data().unwrap(), &[expected]);
}

#[test]
fn session_typed_invalid_broadcast_error() {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0; 2]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0; 3]).unwrap();

    let session_error = on_engine(|s| a.add(&b, s)).unwrap_err();

    assert_incompatible_shapes_error(session_error);
}

#[test]
fn session_in_typed_validates_output_dtype() {
    let mut backend = WrongDTypeSessionBackend;
    let a = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let lower = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![0, 0]).unwrap();
    let upper = TypedTensor::<i32>::from_vec_col_major(vec![2], vec![3, 3]).unwrap();
    let matrix = TypedTensor::<i32>::from_vec_col_major(vec![2, 2], vec![1, 2, 3, 4]).unwrap();

    let errors = [
        ("add", a.add(&a, &mut backend).unwrap_err()),
        ("mul", a.mul(&a, &mut backend).unwrap_err()),
        ("exp", a.exp(&mut backend).unwrap_err()),
        (
            "reduce_sum",
            a.reduce_sum(Some(&[0]), &mut backend).unwrap_err(),
        ),
        ("sub", a.sub(&a, &mut backend).unwrap_err()),
        ("div", a.div(&a, &mut backend).unwrap_err()),
        ("pow", a.pow(&a, &mut backend).unwrap_err()),
        ("maximum", a.maximum(&a, &mut backend).unwrap_err()),
        ("neg", a.neg(&mut backend).unwrap_err()),
        ("abs", a.abs(&mut backend).unwrap_err()),
        ("log", a.log(&mut backend).unwrap_err()),
        ("sqrt", a.sqrt(&mut backend).unwrap_err()),
        ("clamp", a.clamp(&lower, &upper, &mut backend).unwrap_err()),
        ("matmul", matrix.matmul(&matrix, &mut backend).unwrap_err()),
        ("reshape", matrix.reshape(&[4], &mut backend).unwrap_err()),
        (
            "transpose",
            matrix.transpose(&[1, 0], &mut backend).unwrap_err(),
        ),
        (
            "broadcast_in_dim",
            matrix
                .broadcast_in_dim(&[2, 2], &[0, 1], &mut backend)
                .unwrap_err(),
        ),
        (
            "compare",
            a.compare(&a, CompareDir::Gt, &mut backend).unwrap_err(),
        ),
    ];
    for (op, error) in errors {
        let Error::Validation {
            op: error_op,
            source: ValidationError::DTypeMismatch { expected, actual },
        } = &error
        else {
            panic!("expected {op} DTypeMismatch, got {error:?}");
        };
        assert_eq!(*error_op, op);
        // `compare` produces a bool tensor; every other op produces the input
        // scalar type.
        let expected_dtype = if op == "compare" {
            tenferro_tensor::core::DType::Bool
        } else {
            tenferro_tensor::core::DType::I32
        };
        assert_eq!(*expected, expected_dtype);
        assert_eq!(*actual, tenferro_tensor::core::DType::F64);
    }
}

// ---------------------------------------------------------------------------
// Single-session-entry proof
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 1 (issue #1680) full-surface direct session tests
// ---------------------------------------------------------------------------

#[test]
fn session_dynamic_binary_chain() {
    // Equal-shape arm: sub -> div -> pow -> maximum -> minimum.
    let a_values = vec![4.0_f64, 9.0, 6.0, 12.0];
    let b_values = vec![2.0_f64, 3.0, 2.0, 4.0];
    let a = Tensor::from_vec_col_major(vec![2, 2], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], b_values.clone()).unwrap();

    let session = on_engine(|s| {
        a.sub(&b, s)
            .unwrap()
            .div(&b, s)
            .unwrap()
            .pow(&b, s)
            .unwrap()
            .maximum(&b, s)
            .unwrap()
            .minimum(&b, s)
            .unwrap()
    });

    // Independent value check in plain scalar math.
    let expected: Vec<f64> = a_values
        .iter()
        .zip(&b_values)
        .map(|(&x, &y)| {
            let v = (x - y) / y;
            let v = v.powf(y);
            let v = v.max(y);
            v.min(y)
        })
        .collect();
    assert_close(session.as_slice::<f64>().unwrap(), &expected);

    // Real broadcast arm: [1] vs [4].
    let a = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    let b = Tensor::from_vec_col_major(vec![4], vec![2.0_f64; 4]).unwrap();

    let session = on_engine(|s| {
        a.sub(&b, s)
            .unwrap()
            .div(&b, s)
            .unwrap()
            .pow(&b, s)
            .unwrap()
            .maximum(&b, s)
            .unwrap()
            .minimum(&b, s)
            .unwrap()
    });
    let expected = [2.0_f64; 4];
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_dynamic_unary_chain() {
    let x_values = vec![2.0_f64, 3.0, 4.0, 5.0];
    let x = Tensor::from_vec_col_major(vec![4], x_values.clone()).unwrap();

    let session = on_engine(|s| {
        x.neg(s)
            .unwrap()
            .abs(s)
            .unwrap()
            .sqrt(s)
            .unwrap()
            .rsqrt(s)
            .unwrap()
            .sign(s)
            .unwrap()
            .conj(s)
            .unwrap()
            .log(s)
            .unwrap()
            .expm1(s)
            .unwrap()
            .log1p(s)
            .unwrap()
            .sin(s)
            .unwrap()
            .cos(s)
            .unwrap()
            .tanh(s)
            .unwrap()
    });

    // Independent value check in plain scalar math; `conj` is the identity
    // for real values.
    let mut expected: Vec<f64> = x_values.iter().map(|&v| -v).collect();
    expected = expected.iter().map(|&v| v.abs()).collect();
    expected = expected.iter().map(|&v| v.sqrt()).collect();
    expected = expected.iter().map(|&v| 1.0 / v).collect();
    expected = expected.iter().map(|&v| v.signum()).collect();
    expected = expected.iter().map(|&v| v.ln()).collect();
    expected = expected.iter().map(|&v| v.exp_m1()).collect();
    expected = expected.iter().map(|&v| (1.0 + v).ln()).collect();
    expected = expected.iter().map(|&v| v.sin()).collect();
    expected = expected.iter().map(|&v| v.cos()).collect();
    expected = expected.iter().map(|&v| v.tanh()).collect();
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_dynamic_ternary_chain() {
    let on_true = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let on_false = Tensor::from_vec_col_major(vec![4], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();

    // Equal-shape arm: where_select -> clamp.
    let condition = Tensor::from_vec_col_major(vec![4], vec![true, false, true, false]).unwrap();
    let lower = Tensor::from_vec_col_major(vec![], vec![0.0_f64]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![], vec![5.0_f64]).unwrap();

    let session = on_engine(|s| {
        condition
            .where_select(&on_true, &on_false, s)
            .unwrap()
            .clamp(&lower, &upper, s)
            .unwrap()
    });
    // select -> [1, 6, 3, 8], then clamp(0, 5).
    let expected = [1.0_f64, 5.0, 3.0, 5.0];
    assert_close(session.as_slice::<f64>().unwrap(), &expected);

    // Broadcast arm: singleton condition and bounds.
    let condition = Tensor::from_vec_col_major(vec![1], vec![true]).unwrap();
    let lower = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();

    let session = on_engine(|s| {
        condition
            .where_select(&on_true, &on_false, s)
            .unwrap()
            .clamp(&lower, &upper, s)
            .unwrap()
    });
    // select broadcasts the condition -> [1, 2, 3, 4], then clamp(2, 3).
    let expected = [2.0_f64, 2.0, 3.0, 3.0];
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_dynamic_structural_chain() {
    let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let session = on_engine(|s| {
        x.transpose(&[1, 0], s)
            .unwrap()
            .reshape(&[6], s)
            .unwrap()
            .reshape(&[2, 3], s)
            .unwrap()
            .transpose(&[1, 0], s)
            .unwrap()
    });

    // Independent permutation math: reshape preserves the col-major storage
    // order, so the chain is two transposes of [2,3] layouts on the original
    // storage, ending in a [3,2] layout.
    let transposed = transpose_col_major(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let expected = transpose_col_major(&transposed, &[2, 3]);
    assert_eq!(expected, vec![1.0, 5.0, 4.0, 3.0, 2.0, 6.0]);
    assert_eq!(session.shape(), &[3, 2]);
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_dynamic_dtype_chain() {
    let x = Tensor::from_vec_col_major(vec![4], vec![1.2_f64, -2.8, 3.5, 0.4]).unwrap();

    let session = on_engine(|s| x.cast(DType::I32, s)).unwrap();
    let expected = [1_i32, -2, 3, 0];
    assert_eq!(session.as_slice::<i32>().unwrap(), &expected);

    // convert uses the checked lattice: f64 -> C64 preserves values, and the
    // round trip back through cast reproduces them.
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();

    let session = on_engine(|s| y.convert(DType::C64, s)).unwrap();
    assert_eq!(session.dtype(), DType::C64);

    let back = on_engine(|s| session.cast(DType::F64, s)).unwrap();
    assert_close(back.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn session_dynamic_matmul_chain() {
    let a_values = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b_values = vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0];
    let a = Tensor::from_vec_col_major(vec![2, 3], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 2], b_values.clone()).unwrap();

    let session = on_engine(|s| a.matmul(&b, s)).unwrap();

    // Independent value check with plain triple loops over col-major indices.
    let mut expected = vec![0.0_f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                expected[j * 2 + i] += a_values[k * 2 + i] * b_values[j * 3 + k];
            }
        }
    }
    assert_eq!(session.shape(), &[2, 2]);
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_dynamic_errors() {
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    // Binary broadcast error.

    let session_error = on_engine(|s| a.sub(&b, s)).unwrap_err();

    assert_incompatible_shapes_error(session_error);

    // Ternary broadcast error through clamp ([2] vs [3] bounds).
    let lower = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let session_error = on_engine(|s| a.clamp(&lower, &upper, s)).unwrap_err();

    assert_incompatible_shapes_error(session_error);

    // Matmul rank error for rank-1 operands.

    let session_error = on_engine(|s| a.matmul(&b, s)).unwrap_err();

    assert_rank_mismatch_error(session_error);

    // Dtype mismatch error for sub.
    let c = Tensor::from_vec_col_major(vec![2], vec![1_i32; 2]).unwrap();

    let session_error = on_engine(|s| a.sub(&c, s)).unwrap_err();
    let Error::Validation {
        op: "sub",
        source: ValidationError::DTypeMismatch { expected, actual },
    } = &session_error
    else {
        panic!("expected sub DTypeMismatch, got {session_error:?}");
    };
    assert_eq!(*expected, tenferro_tensor::core::DType::F64);
    assert_eq!(*actual, tenferro_tensor::core::DType::I32);
}

#[test]
fn session_typed_binary_chain() {
    // Equal-shape arm: sub -> div -> pow -> maximum -> minimum.
    let a_values = vec![4.0_f64, 9.0, 6.0, 12.0];
    let b_values = vec![2.0_f64, 3.0, 2.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], b_values.clone()).unwrap();

    let session = on_engine(|s| {
        a.sub(&b, s)
            .unwrap()
            .div(&b, s)
            .unwrap()
            .pow(&b, s)
            .unwrap()
            .maximum(&b, s)
            .unwrap()
            .minimum(&b, s)
            .unwrap()
    });

    let expected: Vec<f64> = a_values
        .iter()
        .zip(&b_values)
        .map(|(&x, &y)| {
            let v = (x - y) / y;
            let v = v.powf(y);
            let v = v.max(y);
            v.min(y)
        })
        .collect();
    assert_close(session.host_data().unwrap(), &expected);

    // Real broadcast arm: [1] vs [4].
    let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![2.0; 4]).unwrap();

    let session = on_engine(|s| {
        a.sub(&b, s)
            .unwrap()
            .div(&b, s)
            .unwrap()
            .pow(&b, s)
            .unwrap()
            .maximum(&b, s)
            .unwrap()
            .minimum(&b, s)
            .unwrap()
    });
    let expected = [2.0_f64; 4];
    assert_close(session.host_data().unwrap(), &expected);
}

#[test]
fn session_typed_unary_chain() {
    let x_values = vec![2.0_f64, 3.0, 4.0, 5.0];
    let x = TypedTensor::<f64>::from_vec_col_major(vec![4], x_values.clone()).unwrap();

    let session = on_engine(|s| {
        x.neg(s)
            .unwrap()
            .abs(s)
            .unwrap()
            .sqrt(s)
            .unwrap()
            .rsqrt(s)
            .unwrap()
            .sign(s)
            .unwrap()
            .conj(s)
            .unwrap()
            .log(s)
            .unwrap()
            .expm1(s)
            .unwrap()
            .log1p(s)
            .unwrap()
            .sin(s)
            .unwrap()
            .cos(s)
            .unwrap()
            .tanh(s)
            .unwrap()
    });

    let mut expected: Vec<f64> = x_values.iter().map(|&v| -v).collect();
    expected = expected.iter().map(|&v| v.abs()).collect();
    expected = expected.iter().map(|&v| v.sqrt()).collect();
    expected = expected.iter().map(|&v| 1.0 / v).collect();
    expected = expected.iter().map(|&v| v.signum()).collect();
    expected = expected.iter().map(|&v| v.ln()).collect();
    expected = expected.iter().map(|&v| v.exp_m1()).collect();
    expected = expected.iter().map(|&v| (1.0 + v).ln()).collect();
    expected = expected.iter().map(|&v| v.sin()).collect();
    expected = expected.iter().map(|&v| v.cos()).collect();
    expected = expected.iter().map(|&v| v.tanh()).collect();
    assert_close(session.host_data().unwrap(), &expected);
}

#[test]
fn session_typed_clamp_chain() {
    // Equal-shape arm.
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![-2.0, 4.0, 1.0, 5.0]).unwrap();
    let lower = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]).unwrap();
    let upper = TypedTensor::<f64>::from_vec_col_major(vec![], vec![3.0]).unwrap();

    let session = on_engine(|s| x.clamp(&lower, &upper, s)).unwrap();
    assert_close(session.host_data().unwrap(), &[0.0, 3.0, 1.0, 3.0]);

    // Broadcast arm: singleton bounds.
    let x = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![-1.0, 0.0, 2.0, 5.0]).unwrap();
    let lower = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    let upper = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0]).unwrap();

    let session = on_engine(|s| x.clamp(&lower, &upper, s)).unwrap();
    assert_close(session.host_data().unwrap(), &[1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn session_typed_compare_chain() {
    // Equal-shape arm: the result is a bool typed tensor.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 5.0, 3.0, 8.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 6.0, 4.0, 7.0]).unwrap();

    let session = on_engine(|s| a.compare(&b, CompareDir::Gt, s)).unwrap();
    let expected = [true, false, false, true];
    assert_eq!(session.host_data().unwrap(), &expected);

    // Broadcast arm.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![1.0, 4.0, 2.0, 6.0]).unwrap();

    let session = on_engine(|s| a.compare(&b, CompareDir::Gt, s)).unwrap();
    let expected = [true, false, true, false];
    assert_eq!(session.host_data().unwrap(), &expected);
}

#[test]
fn session_typed_structural_chain() {
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    // Reshape preserves element order.

    let session = on_engine(|s| x.reshape(&[6], s)).unwrap();
    let expected = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(session.shape(), &[6]);
    assert_close(session.host_data().unwrap(), &expected);

    // Transpose permutes the col-major layout.

    let session = on_engine(|s| x.transpose(&[1, 0], s)).unwrap();
    let expected = [1.0_f64, 3.0, 5.0, 2.0, 4.0, 6.0];
    assert_eq!(session.shape(), &[3, 2]);
    assert_close(session.host_data().unwrap(), &expected);

    // broadcast_in_dim duplicates the row: [[1,2,3],[1,2,3]] in col-major
    // storage.
    let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();

    let session = on_engine(|s| row.broadcast_in_dim(&[2, 3], &[1], s)).unwrap();
    let expected = [1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0];
    assert_eq!(session.shape(), &[2, 3]);
    assert_close(session.host_data().unwrap(), &expected);
}

#[test]
fn session_typed_matmul_chain() {
    let a_values = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b_values = vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], b_values.clone()).unwrap();

    let session = on_engine(|s| a.matmul(&b, s)).unwrap();

    let mut expected = vec![0.0_f64; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                expected[j * 2 + i] += a_values[k * 2 + i] * b_values[j * 3 + k];
            }
        }
    }
    assert_eq!(session.shape(), &[2, 2]);
    assert_close(session.host_data().unwrap(), &expected);
}

#[test]
fn session_typed_errors() {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0; 2]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0; 3]).unwrap();

    // Ternary broadcast error through clamp ([2] vs [3] bounds).

    let session_error = on_engine(|s| a.clamp(&b, &a, s)).unwrap_err();

    assert_incompatible_shapes_error(session_error);

    // Matmul rank error for rank-1 operands.

    let session_error = on_engine(|s| a.matmul(&b, s)).unwrap_err();

    assert_rank_mismatch_error(session_error);
}

#[test]
fn session_new_ops_chain_computes_the_expected_values() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 5.0, 4.0, 6.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 1.0, 2.0]).unwrap();
    let m = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 3], vec![1.0_f64; 3]).unwrap();

    // A 5-op chain (sub, log, maximum, reshape, matmul) on one engine.
    let session = on_engine(|s| {
        a.sub(&b, s)
            .unwrap()
            .log(s)
            .unwrap()
            .maximum(&m, s)
            .unwrap()
            .reshape(&[4, 1], s)
            .unwrap()
            .matmul(&rhs, s)
            .unwrap()
    });

    // Independent scalar math: v_i = max(log(a_i - b_i), 1.0), and the
    // [4,1] x [1,3] matmul repeats each v_i across the three rhs columns.
    let values: Vec<f64> = [3.0_f64, 5.0, 4.0, 6.0]
        .iter()
        .zip([1.0_f64, 2.0, 1.0, 2.0])
        .map(|(&x, y)| (x - y).ln().max(1.0))
        .collect();
    let expected: Vec<f64> = (0..3).flat_map(|_| values.iter().copied()).collect();
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
}

// ---------------------------------------------------------------------------
// Test backends
// ---------------------------------------------------------------------------

/// Test backend whose session ops return an `F64` tensor regardless of the
/// requested dtype, so the typed session surface must reject the output through
/// `into_typed_result`.
struct WrongDTypeSessionBackend;

macro_rules! panic_backend_methods {
    ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
        $(
            fn $name(&mut self, $($arg: $argty),*) -> $ret {
                let _ = ($($arg),*);
                panic!(concat!(stringify!($name), " should not be called in this test"))
            }
        )+
    };
}

// WrongDTypeSessionBackend hand-writes the structural, dot, elementwise, and
// analytic impls so every op family the typed session surface routes through can
// return an F64 tensor regardless of the requested dtype.

impl TensorStructural for WrongDTypeSessionBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult {
        on_engine(|__s| __s.to_contiguous_read(input))
    }

    fn copy_read_into(
        &mut self,
        src: TensorRead<'_>,
        dst: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        on_engine(|__s| __s.copy_read_into(src, dst))
    }

    fn reshape_read(&mut self, _input: TensorRead<'_>, _shape: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn transpose_read(&mut self, _input: TensorRead<'_>, _perm: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn broadcast_in_dim_read(
        &mut self,
        _input: TensorRead<'_>,
        _shape: &[usize],
        _dims: &[usize],
    ) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    panic_backend_methods! {
        cast(input: &Tensor, to: DType) -> TensorResult;
        convert(input: &Tensor, to: DType) -> TensorResult;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
        tril(input: &Tensor, k: i64) -> TensorResult;
        triu(input: &Tensor, k: i64) -> TensorResult;
    }
}

impl TensorIndexing for WrongDTypeSessionBackend {
    panic_backend_methods! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> TensorResult;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> TensorResult;
        slice(input: &Tensor, config: &SliceConfig) -> TensorResult;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> TensorResult;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> TensorResult;
        pad(input: &Tensor, config: &PadConfig) -> TensorResult;
        concatenate(inputs: &[&Tensor], axis: usize) -> TensorResult;
        reverse(input: &Tensor, axes: &[usize]) -> TensorResult;
    }
}

impl TensorDot for WrongDTypeSessionBackend {
    // The previous read-half default delegated an owned pair to the one-shot
    // method and materialized borrowed views through to_contiguous_read before
    // contracting. Reproduce that exactly rather than forwarding a view.
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        _config: &DotGeneralConfig,
    ) -> TensorResult {
        match (lhs.as_tensor(), rhs.as_tensor()) {
            (Some(_), Some(_)) => Ok(wrong_dtype_tensor()),
            _ => {
                let _ = self.to_contiguous_read(lhs)?;
                let _ = self.to_contiguous_read(rhs)?;
                Ok(wrong_dtype_tensor())
            }
        }
    }
}

impl TensorFusion for WrongDTypeSessionBackend {}
impl TensorBuffer for WrongDTypeSessionBackend {}

impl TensorDeviceTransfer for WrongDTypeSessionBackend {
    fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
        Err(Error::unsupported(
            "WrongDTypeSessionBackend::download_to_host",
            "test backend does not transfer tensors",
        ))
    }

    fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
        Err(Error::unsupported(
            "WrongDTypeSessionBackend::upload_host_tensor",
            "test backend does not transfer tensors",
        ))
    }
}

impl SessionCachedDot for WrongDTypeSessionBackend {}

impl Engine for WrongDTypeSessionBackend {}

impl TensorElementwise for WrongDTypeSessionBackend {
    fn elementwise_read_into(
        &mut self,
        op: ElementwiseReadOp,
        inputs: &[TensorRead<'_>],
        out: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        let _ = (op, inputs, out);
        panic!("elementwise_read_into should not be called in this test")
    }

    panic_backend_methods! {
        rem(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
    }

    fn add_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn sub_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn mul_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn div_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn maximum_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn neg_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn abs_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn compare_read(
        &mut self,
        _lhs: TensorRead<'_>,
        _rhs: TensorRead<'_>,
        _dir: &CompareDir,
    ) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn clamp_read(
        &mut self,
        _input: TensorRead<'_>,
        _lower: TensorRead<'_>,
        _upper: TensorRead<'_>,
    ) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn conj_read(&mut self, input: TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("conj", input)?;
        panic!("conj should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sign_read(&mut self, input: TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("sign", input)?;
        panic!("sign should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn minimum_read(&mut self, lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("minimum", lhs)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("minimum", rhs)?;
        panic!("minimum should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn select_read(
        &mut self,
        pred: TensorRead<'_>,
        on_true: TensorRead<'_>,
        on_false: TensorRead<'_>,
    ) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("select", pred)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_true)?;
        let _ = tenferro_tensor::backend::read_owned_tensor("select", on_false)?;
        panic!("select should not be called in this test")
    }
}

impl TensorAnalytic for WrongDTypeSessionBackend {
    fn exp_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn log_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn sqrt_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn pow_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }
    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn sin_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("sin", input)?;
        panic!("sin should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn cos_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("cos", input)?;
        panic!("cos should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn tanh_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("tanh", input)?;
        panic!("tanh should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn rsqrt_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("rsqrt", input)?;
        panic!("rsqrt should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn expm1_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("expm1", input)?;
        panic!("expm1 should not be called in this test")
    }

    // Reproduce the previous read-half default: delegate an owned tensor and
    // reject a borrowed view.
    fn log1p_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("log1p", input)?;
        panic!("log1p should not be called in this test")
    }

    fn erf_read(&mut self, input: tenferro_tensor::TensorRead<'_>) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("erf", input)?;
        panic!("erf should not be called in this test")
    }
}

impl TensorReduction for WrongDTypeSessionBackend {
    fn reduce_sum_read(&mut self, _input: TensorRead<'_>, _axes: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    // The previous read-half default delegated owned tensors to the one-shot
    // method and rejected views. Reproduce it explicitly.
    fn reduce_prod_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("reduce_prod", input)?;
        let _ = axes;
        panic!("reduce_prod should not be called in this test")
    }

    fn reduce_max_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("reduce_max", input)?;
        let _ = axes;
        panic!("reduce_max should not be called in this test")
    }

    fn reduce_min_read(&mut self, input: TensorRead<'_>, axes: &[usize]) -> TensorResult {
        let _ = tenferro_tensor::backend::read_owned_tensor("reduce_min", input)?;
        let _ = axes;
        panic!("reduce_min should not be called in this test")
    }
}

fn wrong_dtype_tensor() -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![], vec![1.0]).unwrap())
}
