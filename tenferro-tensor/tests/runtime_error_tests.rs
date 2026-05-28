use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use tenferro_tensor::{
    cpu::CpuBackend, Buffer, BufferHandle, DeviceId, DeviceKind, DotGeneralConfig, Error,
    GpuBackendKind, MemoryKind, PadConfig, Placement, SliceConfig, Tensor, TensorAnalytic,
    TensorDot, TensorElementwise, TensorIndexing, TensorStructural, TypedTensor,
};

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data))
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data))
}

fn backend_f64_tensor(shape: Vec<usize>) -> Tensor {
    let len = shape.iter().product();
    Tensor::F64(TypedTensor::from_buffer_col_major(
        shape,
        Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(7, len))),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 0,
            }),
        },
    ))
}

#[test]
fn backend_failure_helper_preserves_op_and_message() {
    let err = Error::backend_failure("custom_op", "device rejected launch");

    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "custom_op",
            ref message,
        } if message == "device rejected launch"
    ));
}

#[test]
fn cpu_linalg_dispatch_does_not_use_panic_catching_as_error_handling() {
    let backend_dispatch = include_str!("../src/cpu/backend.rs");
    let exec_session_dispatch = include_str!("../src/cpu/exec_session.rs");

    assert!(
        !backend_dispatch.contains("catch_backend_panic"),
        "CpuBackend should return typed errors from linalg helpers instead of catching panics"
    );
    assert!(
        !exec_session_dispatch.contains("catch_backend_panic"),
        "CpuExecSession should return typed errors from linalg helpers instead of catching panics"
    );
    assert!(
        !backend_dispatch.contains("catch_unwind"),
        "CPU backend error handling should not depend on panic unwinding"
    );
}

#[test]
fn cpu_backend_try_with_threads_rejects_zero_without_panicking() {
    let err = match CpuBackend::try_with_threads(0) {
        Ok(_) => panic!("zero threads should be rejected"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "CpuBackend::try_with_threads",
            ..
        }
    ));
}

#[test]
fn dot_general_rejects_out_of_bounds_contracting_dim() {
    let lhs = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let rhs = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let mut backend = CpuBackend::new();

    let err = backend
        .dot_general(
            &lhs,
            &rhs,
            &DotGeneralConfig {
                lhs_contracting_dims: vec![2],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        )
        .unwrap_err();

    assert!(matches!(
        err,
        Error::AxisOutOfBounds {
            op: "dot_general",
            axis: 2,
            rank: 2,
        }
    ));
}

#[test]
fn add_rejects_shape_mismatch() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs = f64_tensor(vec![3], vec![3.0, 4.0, 5.0]);
    let mut backend = CpuBackend::new();

    let err = <CpuBackend as TensorElementwise>::add(&mut backend, &lhs, &rhs).unwrap_err();

    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "add",
            lhs,
            rhs,
        } if lhs == vec![2] && rhs == vec![3]
    ));
}

#[test]
fn cpu_backend_rejects_backend_buffers_without_panicking() {
    let lhs = backend_f64_tensor(vec![2]);
    let rhs = f64_tensor(vec![2], vec![3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| {
        <CpuBackend as TensorElementwise>::add(&mut backend, &lhs, &rhs)
    }));

    assert!(result.is_ok(), "CPU backend should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(err, Error::BackendFailure { op: "add", .. }));
    assert!(err.to_string().contains("download to host"));
}

#[test]
fn transpose_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.transpose(&input, &[0])));

    assert!(result.is_ok(), "transpose should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure {
            op: "transpose",
            ..
        } | Error::InvalidConfig {
            op: "transpose",
            ..
        } | Error::RankMismatch {
            op: "transpose",
            ..
        } | Error::AxisOutOfBounds {
            op: "transpose",
            ..
        } | Error::DuplicateAxis {
            op: "transpose",
            ..
        }
    ));
}

#[test]
fn reshape_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.reshape(&input, &[3])));

    assert!(result.is_ok(), "reshape should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "reshape", .. }
            | Error::InvalidConfig { op: "reshape", .. }
            | Error::ShapeMismatch { op: "reshape", .. }
    ));
}

#[test]
fn pow_returns_error_on_shape_mismatch_instead_of_panicking() {
    let lhs = f64_tensor(vec![2], vec![1.0, 2.0]);
    let rhs = f64_tensor(vec![1], vec![3.0]);
    let mut backend = CpuBackend::new();

    let result = catch_unwind(AssertUnwindSafe(|| backend.pow(&lhs, &rhs)));

    assert!(result.is_ok(), "pow should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch { op: "pow", .. } | Error::BackendFailure { op: "pow", .. }
    ));
}

#[test]
fn slice_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();
    let config = SliceConfig {
        starts: vec![0],
        limits: vec![2],
        strides: vec![1],
    };

    let result = catch_unwind(AssertUnwindSafe(|| backend.slice(&input, &config)));

    assert!(result.is_ok(), "slice should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "slice", .. }
            | Error::InvalidConfig { op: "slice", .. }
            | Error::RankMismatch { op: "slice", .. }
            | Error::AxisOutOfBounds { op: "slice", .. }
    ));
}

#[test]
fn pad_returns_error_instead_of_panicking() {
    let input = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let mut backend = CpuBackend::new();
    let config = PadConfig {
        edge_padding_low: vec![0, 0],
        edge_padding_high: vec![0],
        interior_padding: vec![0, 0],
    };

    let result = catch_unwind(AssertUnwindSafe(|| backend.pad(&input, &config)));

    assert!(result.is_ok(), "pad should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::BackendFailure { op: "pad", .. }
            | Error::InvalidConfig { op: "pad", .. }
            | Error::RankMismatch { op: "pad", .. }
    ));
}

#[test]
fn concatenate_returns_error_on_empty_inputs() {
    let mut backend = CpuBackend::new();
    let inputs: Vec<&Tensor> = vec![];

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&inputs, 0)));

    assert!(result.is_ok(), "concatenate should return Err, not panic");
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidConfig {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_dtype_mismatch() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f32_tensor(vec![2, 2], vec![5.0f32, 6.0, 7.0, 8.0]);

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 0)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on dtype mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::DTypeMismatch {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_rank_mismatch() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2], vec![5.0, 6.0]);

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 0)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on rank mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::RankMismatch {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_axis_out_of_bounds() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 5)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on axis out of bounds, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::AxisOutOfBounds {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_returns_error_on_shape_mismatch() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = f64_tensor(
        vec![2, 4],
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
    );

    let result = catch_unwind(AssertUnwindSafe(|| backend.concatenate(&[&a, &b], 0)));

    assert!(
        result.is_ok(),
        "concatenate should return Err on shape mismatch, not panic"
    );
    let err = result.unwrap().unwrap_err();
    assert!(matches!(
        err,
        Error::ShapeMismatch {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn concatenate_accepts_valid_inputs() {
    let mut backend = CpuBackend::new();
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = f64_tensor(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let result = backend.concatenate(&[&a, &b], 0);

    assert!(result.is_ok());
    let out = result.unwrap();
    assert_eq!(out.shape(), &[4, 2]);
}
