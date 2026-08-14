//! Parity and single-session-entry tests for the `_in` session surfaces.
//!
//! Covers, per surface (dynamic [`Tensor`] and typed [`TypedTensor`]):
//! equal-shape ops, real broadcast, invalid-broadcast structured-error parity
//! with the one-shot path, dtype parity, typed output dtype validation
//! (`into_typed_result`), and a deterministic proof that an `_in` chain
//! executes inside exactly one backend session entry while the one-shot
//! counterpart enters one session per op.

use std::cell::Cell;

use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    Tensor, TensorOpsExt, TensorSessionOpsExt, TypedTensor, TypedTensorOpsExt,
    TypedTensorSessionOpsExt,
};
use tenferro_tensor::backend::{
    BackendCachedDot, TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot,
    TensorElementwise, TensorFusion, TensorIndexing, TensorReduction, TensorStructural,
};
use tenferro_tensor::{
    BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir, DType, DotGeneralConfig,
    Error, GatherConfig, PadConfig, ScatterConfig, ShapeMismatch, SliceConfig, TensorBackend,
    TensorRead, TensorWrite, ValidationError,
};

type TensorResult = tenferro_tensor::Result<Tensor>;

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
fn session_in_dynamic_equal_shape_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.5_f64; 8];
    let b_values = vec![2.25_f64; 8];
    let a = Tensor::from_vec_col_major(vec![8], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&b, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&b, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
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

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(
        session.as_slice::<f64>().unwrap(),
        one_shot.as_slice::<f64>().unwrap(),
    );
}

#[test]
fn session_in_dynamic_broadcast_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.0_f64];
    let b_values = vec![2.0_f64; 8];
    let a = Tensor::from_vec_col_major(vec![1], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });

    // Independent value check: the broadcast chain computes
    // sum_i exp(a_0 + b_i) * a_0 in plain scalar math.
    let expected: f64 = b_values
        .iter()
        .map(|&y| (a_values[0] + y).exp() * a_values[0])
        .sum();
    assert_close(session.as_slice::<f64>().unwrap(), &[expected]);

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(
        session.as_slice::<f64>().unwrap(),
        one_shot.as_slice::<f64>().unwrap(),
    );
}

#[test]
fn session_in_dynamic_invalid_broadcast_matches_one_shot_error() {
    let mut backend = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    let one_shot_error = a.add(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.add_in(&b, s))
        .unwrap_err();

    assert_incompatible_shapes_error(one_shot_error);
    assert_incompatible_shapes_error(session_error);
}

#[test]
fn session_in_dynamic_dtype_error_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![1_i32; 2]).unwrap();

    let one_shot_error = a.add(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.add_in(&b, s))
        .unwrap_err();

    // Assert the full payload (op name, source lhs dtype as expected, rhs
    // dtype as actual) for both paths, not just the error kind.
    for error in [one_shot_error, session_error] {
        let Error::Validation {
            op: "add",
            source: ValidationError::DTypeMismatch { expected, actual },
        } = &error
        else {
            panic!("expected add DTypeMismatch, got {error:?}");
        };
        assert_eq!(*expected, tenferro_tensor::core::DType::F64);
        assert_eq!(*actual, tenferro_tensor::core::DType::I32);
    }
}

// ---------------------------------------------------------------------------
// Typed surface (TypedTensorSessionOpsExt)
// ---------------------------------------------------------------------------

#[test]
fn session_in_typed_equal_shape_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.5_f64; 8];
    let b_values = vec![2.25_f64; 8];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![8], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&b, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&b, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });

    // Independent value check in plain scalar math (see the dynamic twin).
    let expected: f64 = a_values
        .iter()
        .zip(&b_values)
        .map(|(&x, &y)| (x + y).exp() * y)
        .sum();
    assert_close(session.host_data().unwrap(), &[expected]);

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(session.host_data().unwrap(), one_shot.host_data().unwrap());
}

#[test]
fn session_in_typed_broadcast_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.0_f64];
    let b_values = vec![2.0_f64; 8];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![1], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![8], b_values.clone()).unwrap();

    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });

    // Independent value check: sum_i exp(a_0 + b_i) * a_0 in plain scalar math.
    let expected: f64 = b_values
        .iter()
        .map(|&y| (a_values[0] + y).exp() * a_values[0])
        .sum();
    assert_close(session.host_data().unwrap(), &[expected]);

    assert_eq!(session.shape(), one_shot.shape());
    assert_close(session.host_data().unwrap(), one_shot.host_data().unwrap());
}

#[test]
fn session_in_typed_invalid_broadcast_matches_one_shot_error() {
    let mut backend = CpuBackend::new();
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0; 2]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0; 3]).unwrap();

    let one_shot_error = a.add(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.add_in(&b, s))
        .unwrap_err();

    assert_incompatible_shapes_error(one_shot_error);
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
        (
            "add",
            backend
                .with_backend_session(|s| a.add_in(&a, s))
                .unwrap_err(),
        ),
        (
            "mul",
            backend
                .with_backend_session(|s| a.mul_in(&a, s))
                .unwrap_err(),
        ),
        (
            "exp",
            backend.with_backend_session(|s| a.exp_in(s)).unwrap_err(),
        ),
        (
            "reduce_sum",
            backend
                .with_backend_session(|s| a.reduce_sum_in(&[0], s))
                .unwrap_err(),
        ),
        (
            "sub",
            backend
                .with_backend_session(|s| a.sub_in(&a, s))
                .unwrap_err(),
        ),
        (
            "div",
            backend
                .with_backend_session(|s| a.div_in(&a, s))
                .unwrap_err(),
        ),
        (
            "pow",
            backend
                .with_backend_session(|s| a.pow_in(&a, s))
                .unwrap_err(),
        ),
        (
            "maximum",
            backend
                .with_backend_session(|s| a.maximum_in(&a, s))
                .unwrap_err(),
        ),
        (
            "neg",
            backend.with_backend_session(|s| a.neg_in(s)).unwrap_err(),
        ),
        (
            "abs",
            backend.with_backend_session(|s| a.abs_in(s)).unwrap_err(),
        ),
        (
            "log",
            backend.with_backend_session(|s| a.log_in(s)).unwrap_err(),
        ),
        (
            "sqrt",
            backend.with_backend_session(|s| a.sqrt_in(s)).unwrap_err(),
        ),
        (
            "clamp",
            backend
                .with_backend_session(|s| a.clamp_in(&lower, &upper, s))
                .unwrap_err(),
        ),
        (
            "matmul",
            backend
                .with_backend_session(|s| matrix.matmul_in(&matrix, s))
                .unwrap_err(),
        ),
        (
            "reshape",
            backend
                .with_backend_session(|s| matrix.reshape_in(&[4], s))
                .unwrap_err(),
        ),
        (
            "transpose",
            backend
                .with_backend_session(|s| matrix.transpose_in(&[1, 0], s))
                .unwrap_err(),
        ),
        (
            "broadcast_in_dim",
            backend
                .with_backend_session(|s| matrix.broadcast_in_dim_in(&[2, 2], &[0, 1], s))
                .unwrap_err(),
        ),
        (
            "compare",
            backend
                .with_backend_session(|s| a.compare_in(&a, CompareDir::Gt, s))
                .unwrap_err(),
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

#[test]
fn session_in_chain_enters_one_session_one_shot_enters_ten() {
    let mut backend = SessionCountingBackend::new();
    let a = Tensor::from_vec_col_major(vec![1], vec![0.5_f64]).unwrap();
    let b = Tensor::from_vec_col_major(vec![8], vec![1.0_f64; 8]).unwrap();

    // 10 one-shot ops (3x add->exp->mul + final reduce_sum): one session entry
    // per op.
    let one_shot = a
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .add(&b, &mut backend)
        .unwrap()
        .exp(&mut backend)
        .unwrap()
        .mul(&a, &mut backend)
        .unwrap()
        .reduce_sum(&[0], &mut backend)
        .unwrap();
    assert_eq!(
        backend.entries.get(),
        10,
        "one-shot chain must enter one session per op"
    );

    backend.entries.set(0);
    let session = backend.with_backend_session(|s| {
        let x = a.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        let x = x.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        let x = x.add_in(&b, s).unwrap();
        let x = x.exp_in(s).unwrap();
        let x = x.mul_in(&a, s).unwrap();
        x.reduce_sum_in(&[0], s).unwrap()
    });
    assert_eq!(
        backend.entries.get(),
        1,
        "session chain must enter exactly one session"
    );

    assert_close(
        session.as_slice::<f64>().unwrap(),
        one_shot.as_slice::<f64>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// Phase 1 (issue #1680): full `_in` surface parity
// ---------------------------------------------------------------------------

#[test]
fn session_in_dynamic_binary_matches_one_shot() {
    let mut backend = CpuBackend::new();

    // Equal-shape arm: sub -> div -> pow -> maximum -> minimum.
    let a_values = vec![4.0_f64, 9.0, 6.0, 12.0];
    let b_values = vec![2.0_f64, 3.0, 2.0, 4.0];
    let a = Tensor::from_vec_col_major(vec![2, 2], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], b_values.clone()).unwrap();

    let one_shot = a
        .sub(&b, &mut backend)
        .unwrap()
        .div(&b, &mut backend)
        .unwrap()
        .pow(&b, &mut backend)
        .unwrap()
        .maximum(&b, &mut backend)
        .unwrap()
        .minimum(&b, &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        a.sub_in(&b, s)
            .unwrap()
            .div_in(&b, s)
            .unwrap()
            .pow_in(&b, s)
            .unwrap()
            .maximum_in(&b, s)
            .unwrap()
            .minimum_in(&b, s)
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
    assert_close(one_shot.as_slice::<f64>().unwrap(), &expected);
    assert_eq!(session.shape(), one_shot.shape());

    // Real broadcast arm: [1] vs [4].
    let a = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();
    let b = Tensor::from_vec_col_major(vec![4], vec![2.0_f64; 4]).unwrap();
    let one_shot = a
        .sub(&b, &mut backend)
        .unwrap()
        .div(&b, &mut backend)
        .unwrap()
        .pow(&b, &mut backend)
        .unwrap()
        .maximum(&b, &mut backend)
        .unwrap()
        .minimum(&b, &mut backend)
        .unwrap();
    let session = backend.with_backend_session(|s| {
        a.sub_in(&b, s)
            .unwrap()
            .div_in(&b, s)
            .unwrap()
            .pow_in(&b, s)
            .unwrap()
            .maximum_in(&b, s)
            .unwrap()
            .minimum_in(&b, s)
            .unwrap()
    });
    let expected = [2.0_f64; 4];
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
    assert_close(one_shot.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_in_dynamic_unary_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let x_values = vec![2.0_f64, 3.0, 4.0, 5.0];
    let x = Tensor::from_vec_col_major(vec![4], x_values.clone()).unwrap();

    let one_shot = x
        .neg(&mut backend)
        .unwrap()
        .abs(&mut backend)
        .unwrap()
        .sqrt(&mut backend)
        .unwrap()
        .rsqrt(&mut backend)
        .unwrap()
        .sign(&mut backend)
        .unwrap()
        .conj(&mut backend)
        .unwrap()
        .log(&mut backend)
        .unwrap()
        .expm1(&mut backend)
        .unwrap()
        .log1p(&mut backend)
        .unwrap()
        .sin(&mut backend)
        .unwrap()
        .cos(&mut backend)
        .unwrap()
        .tanh(&mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        x.neg_in(s)
            .unwrap()
            .abs_in(s)
            .unwrap()
            .sqrt_in(s)
            .unwrap()
            .rsqrt_in(s)
            .unwrap()
            .sign_in(s)
            .unwrap()
            .conj_in(s)
            .unwrap()
            .log_in(s)
            .unwrap()
            .expm1_in(s)
            .unwrap()
            .log1p_in(s)
            .unwrap()
            .sin_in(s)
            .unwrap()
            .cos_in(s)
            .unwrap()
            .tanh_in(s)
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
    assert_close(one_shot.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_in_dynamic_ternary_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let on_true = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let on_false = Tensor::from_vec_col_major(vec![4], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap();

    // Equal-shape arm: where_select -> clamp.
    let condition = Tensor::from_vec_col_major(vec![4], vec![true, false, true, false]).unwrap();
    let lower = Tensor::from_vec_col_major(vec![], vec![0.0_f64]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![], vec![5.0_f64]).unwrap();

    let one_shot = condition
        .where_select(&on_true, &on_false, &mut backend)
        .unwrap()
        .clamp(&lower, &upper, &mut backend)
        .unwrap();
    let session = backend.with_backend_session(|s| {
        condition
            .where_select_in(&on_true, &on_false, s)
            .unwrap()
            .clamp_in(&lower, &upper, s)
            .unwrap()
    });
    // select -> [1, 6, 3, 8], then clamp(0, 5).
    let expected = [1.0_f64, 5.0, 3.0, 5.0];
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
    assert_close(one_shot.as_slice::<f64>().unwrap(), &expected);

    // Broadcast arm: singleton condition and bounds.
    let condition = Tensor::from_vec_col_major(vec![1], vec![true]).unwrap();
    let lower = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![1], vec![3.0_f64]).unwrap();

    let one_shot = condition
        .where_select(&on_true, &on_false, &mut backend)
        .unwrap()
        .clamp(&lower, &upper, &mut backend)
        .unwrap();
    let session = backend.with_backend_session(|s| {
        condition
            .where_select_in(&on_true, &on_false, s)
            .unwrap()
            .clamp_in(&lower, &upper, s)
            .unwrap()
    });
    // select broadcasts the condition -> [1, 2, 3, 4], then clamp(2, 3).
    let expected = [2.0_f64, 2.0, 3.0, 3.0];
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
    assert_close(one_shot.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_in_dynamic_structural_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let one_shot = x
        .transpose(&[1, 0], &mut backend)
        .unwrap()
        .reshape(&[6], &mut backend)
        .unwrap()
        .reshape(&[2, 3], &mut backend)
        .unwrap()
        .transpose(&[1, 0], &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        x.transpose_in(&[1, 0], s)
            .unwrap()
            .reshape_in(&[6], s)
            .unwrap()
            .reshape_in(&[2, 3], s)
            .unwrap()
            .transpose_in(&[1, 0], s)
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
    assert_close(one_shot.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_in_dynamic_dtype_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let x = Tensor::from_vec_col_major(vec![4], vec![1.2_f64, -2.8, 3.5, 0.4]).unwrap();

    let one_shot = x.cast(DType::I32, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| x.cast_in(DType::I32, s))
        .unwrap();
    let expected = [1_i32, -2, 3, 0];
    assert_eq!(session.as_slice::<i32>().unwrap(), &expected);
    assert_eq!(one_shot.as_slice::<i32>().unwrap(), &expected);

    // convert uses the checked lattice: f64 -> C64 preserves values, and the
    // round trip back through cast reproduces them.
    let y = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let one_shot = y.convert(DType::C64, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| y.convert_in(DType::C64, s))
        .unwrap();
    assert_eq!(session.dtype(), DType::C64);
    assert_eq!(one_shot.dtype(), DType::C64);
    let back = backend
        .with_backend_session(|s| session.cast_in(DType::F64, s))
        .unwrap();
    assert_close(back.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn session_in_dynamic_matmul_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b_values = vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0];
    let a = Tensor::from_vec_col_major(vec![2, 3], a_values.clone()).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 2], b_values.clone()).unwrap();

    let one_shot = a.matmul(&b, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| a.matmul_in(&b, s))
        .unwrap();

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
    assert_close(one_shot.as_slice::<f64>().unwrap(), &expected);
}

#[test]
fn session_in_dynamic_errors_match_one_shot() {
    let mut backend = CpuBackend::new();
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap();
    let b = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();

    // Binary broadcast error.
    let one_shot_error = a.sub(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.sub_in(&b, s))
        .unwrap_err();
    assert_incompatible_shapes_error(one_shot_error);
    assert_incompatible_shapes_error(session_error);

    // Ternary broadcast error through clamp ([2] vs [3] bounds).
    let lower = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let upper = Tensor::from_vec_col_major(vec![3], vec![1.0_f64; 3]).unwrap();
    let one_shot_error = a.clamp(&lower, &upper, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.clamp_in(&lower, &upper, s))
        .unwrap_err();
    assert_incompatible_shapes_error(one_shot_error);
    assert_incompatible_shapes_error(session_error);

    // Matmul rank error for rank-1 operands.
    let one_shot_error = a.matmul(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.matmul_in(&b, s))
        .unwrap_err();
    assert_rank_mismatch_error(one_shot_error);
    assert_rank_mismatch_error(session_error);

    // Dtype mismatch error for sub.
    let c = Tensor::from_vec_col_major(vec![2], vec![1_i32; 2]).unwrap();
    let one_shot_error = a.sub(&c, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.sub_in(&c, s))
        .unwrap_err();
    for error in [one_shot_error, session_error] {
        let Error::Validation {
            op: "sub",
            source: ValidationError::DTypeMismatch { expected, actual },
        } = &error
        else {
            panic!("expected sub DTypeMismatch, got {error:?}");
        };
        assert_eq!(*expected, tenferro_tensor::core::DType::F64);
        assert_eq!(*actual, tenferro_tensor::core::DType::I32);
    }
}

#[test]
fn session_in_typed_binary_matches_one_shot() {
    let mut backend = CpuBackend::new();

    // Equal-shape arm: sub -> div -> pow -> maximum -> minimum.
    let a_values = vec![4.0_f64, 9.0, 6.0, 12.0];
    let b_values = vec![2.0_f64, 3.0, 2.0, 4.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], b_values.clone()).unwrap();

    let one_shot = a
        .sub(&b, &mut backend)
        .unwrap()
        .div(&b, &mut backend)
        .unwrap()
        .pow(&b, &mut backend)
        .unwrap()
        .maximum(&b, &mut backend)
        .unwrap()
        .minimum(&b, &mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        a.sub_in(&b, s)
            .unwrap()
            .div_in(&b, s)
            .unwrap()
            .pow_in(&b, s)
            .unwrap()
            .maximum_in(&b, s)
            .unwrap()
            .minimum_in(&b, s)
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
    assert_close(one_shot.host_data().unwrap(), &expected);
    assert_eq!(session.shape(), one_shot.shape());

    // Real broadcast arm: [1] vs [4].
    let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![2.0; 4]).unwrap();
    let one_shot = a
        .sub(&b, &mut backend)
        .unwrap()
        .div(&b, &mut backend)
        .unwrap()
        .pow(&b, &mut backend)
        .unwrap()
        .maximum(&b, &mut backend)
        .unwrap()
        .minimum(&b, &mut backend)
        .unwrap();
    let session = backend.with_backend_session(|s| {
        a.sub_in(&b, s)
            .unwrap()
            .div_in(&b, s)
            .unwrap()
            .pow_in(&b, s)
            .unwrap()
            .maximum_in(&b, s)
            .unwrap()
            .minimum_in(&b, s)
            .unwrap()
    });
    let expected = [2.0_f64; 4];
    assert_close(session.host_data().unwrap(), &expected);
    assert_close(one_shot.host_data().unwrap(), &expected);
}

#[test]
fn session_in_typed_unary_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let x_values = vec![2.0_f64, 3.0, 4.0, 5.0];
    let x = TypedTensor::<f64>::from_vec_col_major(vec![4], x_values.clone()).unwrap();

    let one_shot = x
        .neg(&mut backend)
        .unwrap()
        .abs(&mut backend)
        .unwrap()
        .sqrt(&mut backend)
        .unwrap()
        .rsqrt(&mut backend)
        .unwrap()
        .sign(&mut backend)
        .unwrap()
        .conj(&mut backend)
        .unwrap()
        .log(&mut backend)
        .unwrap()
        .expm1(&mut backend)
        .unwrap()
        .log1p(&mut backend)
        .unwrap()
        .sin(&mut backend)
        .unwrap()
        .cos(&mut backend)
        .unwrap()
        .tanh(&mut backend)
        .unwrap();

    let session = backend.with_backend_session(|s| {
        x.neg_in(s)
            .unwrap()
            .abs_in(s)
            .unwrap()
            .sqrt_in(s)
            .unwrap()
            .rsqrt_in(s)
            .unwrap()
            .sign_in(s)
            .unwrap()
            .conj_in(s)
            .unwrap()
            .log_in(s)
            .unwrap()
            .expm1_in(s)
            .unwrap()
            .log1p_in(s)
            .unwrap()
            .sin_in(s)
            .unwrap()
            .cos_in(s)
            .unwrap()
            .tanh_in(s)
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
    assert_close(one_shot.host_data().unwrap(), &expected);
}

#[test]
fn session_in_typed_clamp_matches_one_shot() {
    let mut backend = CpuBackend::new();

    // Equal-shape arm.
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![-2.0, 4.0, 1.0, 5.0]).unwrap();
    let lower = TypedTensor::<f64>::from_vec_col_major(vec![], vec![0.0]).unwrap();
    let upper = TypedTensor::<f64>::from_vec_col_major(vec![], vec![3.0]).unwrap();
    let one_shot = x.clamp(&lower, &upper, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| x.clamp_in(&lower, &upper, s))
        .unwrap();
    assert_close(session.host_data().unwrap(), &[0.0, 3.0, 1.0, 3.0]);
    assert_close(one_shot.host_data().unwrap(), &[0.0, 3.0, 1.0, 3.0]);

    // Broadcast arm: singleton bounds.
    let x = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![-1.0, 0.0, 2.0, 5.0]).unwrap();
    let lower = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![1.0]).unwrap();
    let upper = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![2.0]).unwrap();
    let one_shot = x.clamp(&lower, &upper, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| x.clamp_in(&lower, &upper, s))
        .unwrap();
    assert_close(session.host_data().unwrap(), &[1.0, 1.0, 2.0, 2.0]);
    assert_close(one_shot.host_data().unwrap(), &[1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn session_in_typed_compare_matches_one_shot() {
    let mut backend = CpuBackend::new();

    // Equal-shape arm: the result is a bool typed tensor.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0, 5.0, 3.0, 8.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 6.0, 4.0, 7.0]).unwrap();
    let one_shot = a.compare(&b, CompareDir::Gt, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| a.compare_in(&b, CompareDir::Gt, s))
        .unwrap();
    let expected = [true, false, false, true];
    assert_eq!(session.host_data().unwrap(), &expected);
    assert_eq!(one_shot.host_data().unwrap(), &expected);

    // Broadcast arm.
    let a = TypedTensor::<f64>::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![4], vec![1.0, 4.0, 2.0, 6.0]).unwrap();
    let one_shot = a.compare(&b, CompareDir::Gt, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| a.compare_in(&b, CompareDir::Gt, s))
        .unwrap();
    let expected = [true, false, true, false];
    assert_eq!(session.host_data().unwrap(), &expected);
    assert_eq!(one_shot.host_data().unwrap(), &expected);
}

#[test]
fn session_in_typed_structural_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let x = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    // Reshape preserves element order.
    let one_shot = x.reshape(&[6], &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| x.reshape_in(&[6], s))
        .unwrap();
    let expected = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(session.shape(), &[6]);
    assert_close(session.host_data().unwrap(), &expected);
    assert_close(one_shot.host_data().unwrap(), &expected);

    // Transpose permutes the col-major layout.
    let one_shot = x.transpose(&[1, 0], &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| x.transpose_in(&[1, 0], s))
        .unwrap();
    let expected = [1.0_f64, 3.0, 5.0, 2.0, 4.0, 6.0];
    assert_eq!(session.shape(), &[3, 2]);
    assert_close(session.host_data().unwrap(), &expected);
    assert_close(one_shot.host_data().unwrap(), &expected);

    // broadcast_in_dim duplicates the row: [[1,2,3],[1,2,3]] in col-major
    // storage.
    let row = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let one_shot = row.broadcast_in_dim(&[2, 3], &[1], &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| row.broadcast_in_dim_in(&[2, 3], &[1], s))
        .unwrap();
    let expected = [1.0_f64, 1.0, 2.0, 2.0, 3.0, 3.0];
    assert_eq!(session.shape(), &[2, 3]);
    assert_close(session.host_data().unwrap(), &expected);
    assert_close(one_shot.host_data().unwrap(), &expected);
}

#[test]
fn session_in_typed_matmul_matches_one_shot() {
    let mut backend = CpuBackend::new();
    let a_values = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b_values = vec![7.0_f64, 9.0, 11.0, 8.0, 10.0, 12.0];
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], a_values.clone()).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], b_values.clone()).unwrap();

    let one_shot = a.matmul(&b, &mut backend).unwrap();
    let session = backend
        .with_backend_session(|s| a.matmul_in(&b, s))
        .unwrap();

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
    assert_close(one_shot.host_data().unwrap(), &expected);
}

#[test]
fn session_in_typed_errors_match_one_shot() {
    let mut backend = CpuBackend::new();
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0; 2]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0; 3]).unwrap();

    // Ternary broadcast error through clamp ([2] vs [3] bounds).
    let one_shot_error = a.clamp(&b, &a, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.clamp_in(&b, &a, s))
        .unwrap_err();
    assert_incompatible_shapes_error(one_shot_error);
    assert_incompatible_shapes_error(session_error);

    // Matmul rank error for rank-1 operands.
    let one_shot_error = a.matmul(&b, &mut backend).unwrap_err();
    let session_error = backend
        .with_backend_session(|s| a.matmul_in(&b, s))
        .unwrap_err();
    assert_rank_mismatch_error(one_shot_error);
    assert_rank_mismatch_error(session_error);
}

#[test]
fn session_in_new_ops_chain_enters_one_session_one_shot_enters_five() {
    let mut backend = SessionCountingBackend::new();
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 5.0, 4.0, 6.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 1.0, 2.0]).unwrap();
    let m = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1, 3], vec![1.0_f64; 3]).unwrap();

    // 5 one-shot ops (sub, log, maximum, reshape, matmul): one session entry
    // per op.
    let one_shot = a
        .sub(&b, &mut backend)
        .unwrap()
        .log(&mut backend)
        .unwrap()
        .maximum(&m, &mut backend)
        .unwrap()
        .reshape(&[4, 1], &mut backend)
        .unwrap()
        .matmul(&rhs, &mut backend)
        .unwrap();
    assert_eq!(
        backend.entries.get(),
        5,
        "one-shot chain must enter one session per op"
    );

    backend.entries.set(0);
    let session = backend.with_backend_session(|s| {
        a.sub_in(&b, s)
            .unwrap()
            .log_in(s)
            .unwrap()
            .maximum_in(&m, s)
            .unwrap()
            .reshape_in(&[4, 1], s)
            .unwrap()
            .matmul_in(&rhs, s)
            .unwrap()
    });
    assert_eq!(
        backend.entries.get(),
        1,
        "session chain must enter exactly one session"
    );

    // Independent scalar math: v_i = max(log(a_i - b_i), 1.0), and the
    // [4,1] x [1,3] matmul repeats each v_i across the three rhs columns.
    let values: Vec<f64> = [3.0_f64, 5.0, 4.0, 6.0]
        .iter()
        .zip([1.0_f64, 2.0, 1.0, 2.0])
        .map(|(&x, y)| (x - y).ln().max(1.0))
        .collect();
    let expected: Vec<f64> = (0..3).flat_map(|_| values.iter().copied()).collect();
    assert_close(session.as_slice::<f64>().unwrap(), &expected);
    assert_close(
        session.as_slice::<f64>().unwrap(),
        one_shot.as_slice::<f64>().unwrap(),
    );
}

// ---------------------------------------------------------------------------
// Test backends
// ---------------------------------------------------------------------------

/// Test backend counting `with_backend_session` entries while delegating real
/// execution to an inner [`CpuBackend`]. All direct op methods are unreachable
/// in these tests and panic.
struct SessionCountingBackend {
    inner: CpuBackend,
    entries: Cell<usize>,
}

impl SessionCountingBackend {
    fn new() -> Self {
        Self {
            inner: CpuBackend::new(),
            entries: Cell::new(0),
        }
    }
}

/// Test backend whose session ops return an `F64` tensor regardless of the
/// requested dtype, so the typed `_in` surface must reject the output through
/// `into_typed_result`.
struct WrongDTypeSessionBackend;

/// Session-type marker for [`WrongDTypeSessionBackend`].
struct WrongDTypeSessionBackendMarker;

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

/// Panic-only implementations of every op trait except the elementwise,
/// analytic, and reduction families (which some test backends override with
/// real ops). Also excludes `BackendSessionHost`, which each backend
/// implements explicitly.
macro_rules! test_backend_impls {
    ($ty:ident, $marker:ident) => {
        struct $marker;

        impl BackendRuntimeCache for $ty {
            type RuntimeCache = <CpuBackend as BackendRuntimeCache>::RuntimeCache;
        }

        impl TensorStructural for $ty {
            fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult {
                CpuBackend::new().to_contiguous_read(input)
            }

            fn copy_read_into(
                &mut self,
                src: TensorRead<'_>,
                dst: TensorWrite<'_>,
            ) -> tenferro_tensor::Result<()> {
                CpuBackend::new().copy_read_into(src, dst)
            }

            panic_backend_methods! {
                transpose(input: &Tensor, perm: &[usize]) -> TensorResult;
                reshape(input: &Tensor, shape: &[usize]) -> TensorResult;
                broadcast_in_dim(input: &Tensor, shape: &[usize], dims: &[usize]) -> TensorResult;
                cast(input: &Tensor, to: DType) -> TensorResult;
                convert(input: &Tensor, to: DType) -> TensorResult;
                extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
                embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> TensorResult;
                tril(input: &Tensor, k: i64) -> TensorResult;
                triu(input: &Tensor, k: i64) -> TensorResult;
            }
        }

        impl TensorIndexing for $ty {
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

        impl TensorDot for $ty {
            fn dot_general(
                &mut self,
                _lhs: &Tensor,
                _rhs: &Tensor,
                _config: &DotGeneralConfig,
            ) -> TensorResult {
                panic!("dot_general should not be called in this test")
            }
        }

        impl TensorFusion for $ty {}
        impl TensorBuffer for $ty {}

        impl TensorDeviceTransfer for $ty {
            fn download_to_host(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(Error::unsupported(
                    concat!(stringify!($ty), "::download_to_host"),
                    "test backend does not transfer tensors",
                ))
            }

            fn upload_host_tensor(&mut self, _tensor: TensorRead<'_>) -> TensorResult {
                Err(Error::unsupported(
                    concat!(stringify!($ty), "::upload_host_tensor"),
                    "test backend does not transfer tensors",
                ))
            }
        }

        impl BackendCachedDot for $ty {}

        impl BackendSession for $ty {
            fn session_type_id(&self) -> std::any::TypeId {
                std::any::TypeId::of::<$marker>()
            }

            unsafe fn session_data_mut(&mut self) -> *mut () {
                self as *mut Self as *mut ()
            }
        }

        impl TensorBackend for $ty {}
    };
}

macro_rules! panic_elementwise {
    ($ty:ident) => {
        impl TensorElementwise for $ty {
            panic_backend_methods! {
                add(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                sub(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                mul(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                neg(input: &Tensor) -> TensorResult;
                div(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                abs(input: &Tensor) -> TensorResult;
                sign(input: &Tensor) -> TensorResult;
                maximum(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                minimum(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                compare(lhs: &Tensor, rhs: &Tensor, dir: &CompareDir) -> TensorResult;
                select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult;
                clamp(input: &Tensor, lower: &Tensor, upper: &Tensor) -> TensorResult;
            }

            fn conj(&mut self, input: &Tensor) -> TensorResult {
                CpuBackend::new().conj(input)
            }
        }
    };
}

macro_rules! panic_analytic {
    ($ty:ident) => {
        impl TensorAnalytic for $ty {
            panic_backend_methods! {
                exp(input: &Tensor) -> TensorResult;
                log(input: &Tensor) -> TensorResult;
                sin(input: &Tensor) -> TensorResult;
                cos(input: &Tensor) -> TensorResult;
                tanh(input: &Tensor) -> TensorResult;
                sqrt(input: &Tensor) -> TensorResult;
                rsqrt(input: &Tensor) -> TensorResult;
                pow(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
                expm1(input: &Tensor) -> TensorResult;
                log1p(input: &Tensor) -> TensorResult;
            }
        }
    };
}

macro_rules! panic_reduction {
    ($ty:ident) => {
        impl TensorReduction for $ty {
            panic_backend_methods! {
                reduce_sum(input: &Tensor, axes: &[usize]) -> TensorResult;
                reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult;
                reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult;
                reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult;
            }
        }
    };
}

test_backend_impls!(SessionCountingBackend, SessionCountingBackendMarker);
panic_elementwise!(SessionCountingBackend);
panic_analytic!(SessionCountingBackend);
panic_reduction!(SessionCountingBackend);

impl BackendSessionHost for SessionCountingBackend {
    fn with_backend_session<R: Send>(
        &mut self,
        f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
    ) -> R {
        self.entries.set(self.entries.get() + 1);
        self.inner.with_backend_session(f)
    }
}

// WrongDTypeSessionBackend hand-writes the structural, dot, elementwise, and
// analytic impls so every op family the typed `_in` surface routes through can
// return an F64 tensor regardless of the requested dtype.

impl BackendRuntimeCache for WrongDTypeSessionBackend {
    type RuntimeCache = <CpuBackend as BackendRuntimeCache>::RuntimeCache;
}

impl TensorStructural for WrongDTypeSessionBackend {
    fn to_contiguous_read(&mut self, input: TensorRead<'_>) -> TensorResult {
        CpuBackend::new().to_contiguous_read(input)
    }

    fn copy_read_into(
        &mut self,
        src: TensorRead<'_>,
        dst: TensorWrite<'_>,
    ) -> tenferro_tensor::Result<()> {
        CpuBackend::new().copy_read_into(src, dst)
    }

    fn reshape(&mut self, _input: &Tensor, _shape: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn reshape_read(&mut self, _input: TensorRead<'_>, _shape: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn transpose(&mut self, _input: &Tensor, _perm: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn transpose_read(&mut self, _input: TensorRead<'_>, _perm: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn broadcast_in_dim(
        &mut self,
        _input: &Tensor,
        _shape: &[usize],
        _dims: &[usize],
    ) -> TensorResult {
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
    fn dot_general(
        &mut self,
        _lhs: &Tensor,
        _rhs: &Tensor,
        _config: &DotGeneralConfig,
    ) -> TensorResult {
        Ok(wrong_dtype_tensor())
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

impl BackendCachedDot for WrongDTypeSessionBackend {}

impl BackendSession for WrongDTypeSessionBackend {
    fn session_type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<WrongDTypeSessionBackendMarker>()
    }

    unsafe fn session_data_mut(&mut self) -> *mut () {
        self as *mut Self as *mut ()
    }
}

impl TensorBackend for WrongDTypeSessionBackend {}

impl TensorElementwise for WrongDTypeSessionBackend {
    panic_backend_methods! {
        rem(lhs: &Tensor, rhs: &Tensor) -> TensorResult;
        select(pred: &Tensor, on_true: &Tensor, on_false: &Tensor) -> TensorResult;
    }

    fn add(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn add_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn sub(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn sub_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn mul(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn mul_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn div(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn div_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn maximum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn maximum_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn minimum(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn neg(&mut self, _input: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn neg_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn abs(&mut self, _input: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn abs_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn sign(&mut self, _input: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn compare(&mut self, _lhs: &Tensor, _rhs: &Tensor, _dir: &CompareDir) -> TensorResult {
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

    fn clamp(&mut self, _input: &Tensor, _lower: &Tensor, _upper: &Tensor) -> TensorResult {
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

    fn conj(&mut self, input: &Tensor) -> TensorResult {
        CpuBackend::new().conj(input)
    }
}

impl TensorAnalytic for WrongDTypeSessionBackend {
    panic_backend_methods! {
        sin(input: &Tensor) -> TensorResult;
        cos(input: &Tensor) -> TensorResult;
        tanh(input: &Tensor) -> TensorResult;
        rsqrt(input: &Tensor) -> TensorResult;
        expm1(input: &Tensor) -> TensorResult;
        log1p(input: &Tensor) -> TensorResult;
    }

    fn exp(&mut self, _input: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn exp_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn log(&mut self, _input: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn log_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn sqrt(&mut self, _input: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn sqrt_read(&mut self, _input: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn pow(&mut self, _lhs: &Tensor, _rhs: &Tensor) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn pow_read(&mut self, _lhs: TensorRead<'_>, _rhs: TensorRead<'_>) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }
}

impl TensorReduction for WrongDTypeSessionBackend {
    panic_backend_methods! {
        reduce_prod(input: &Tensor, axes: &[usize]) -> TensorResult;
        reduce_max(input: &Tensor, axes: &[usize]) -> TensorResult;
        reduce_min(input: &Tensor, axes: &[usize]) -> TensorResult;
    }

    fn reduce_sum(&mut self, _input: &Tensor, _axes: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }

    fn reduce_sum_read(&mut self, _input: TensorRead<'_>, _axes: &[usize]) -> TensorResult {
        Ok(wrong_dtype_tensor())
    }
}

impl BackendSessionHost for WrongDTypeSessionBackend {}

fn wrong_dtype_tensor() -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![1.0]).unwrap())
}
