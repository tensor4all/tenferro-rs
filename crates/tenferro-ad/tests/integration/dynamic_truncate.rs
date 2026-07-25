use crate::support;
use std::sync::Arc;
use tenferro_ad::TracedTensorAdExt;

use support::{cpu_runtime, RunTraced};
use tenferro_runtime::{DType, Error as RuntimeError, Tensor, TracedTensor, TypedTensor};
use tenferro_tensor::{
    Buffer, BufferHandle, DeviceId, DeviceKind, Error as TensorError, ErrorKind, GpuBackendKind,
    MemoryKind, Placement,
};

const TOL: f64 = 1.0e-5;
const FD_H: f64 = 1.0e-5;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::F32(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn f64_scalar(value: f64) -> Tensor {
    Tensor::F64(TypedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
}

fn i64_scalar(value: i64) -> Tensor {
    Tensor::I64(TypedTensor::from_vec_col_major(vec![], vec![value]).unwrap())
}

fn backend_f64_scalar() -> Tensor {
    Tensor::F64(
        TypedTensor::from_buffer_col_major(
            vec![],
            Buffer::Backend(Arc::new(BufferHandle::<f64>::new_with_len(11, 1))),
            Placement {
                memory_kind: MemoryKind::Device,
                device: Some(DeviceId {
                    kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                    ordinal: 0,
                }),
                cpu_affinity: None,
            },
        )
        .unwrap(),
    )
}

fn get_f64_data(tensor: &Tensor) -> Vec<f64> {
    match tensor {
        Tensor::F64(inner) => inner.host_data().unwrap().to_vec(),
        other => panic!("expected F64 tensor, got {:?}", other.dtype()),
    }
}

fn get_f32_data(tensor: &Tensor) -> Vec<f32> {
    match tensor {
        Tensor::F32(inner) => inner.host_data().unwrap().to_vec(),
        other => panic!("expected F32 tensor, got {:?}", other.dtype()),
    }
}

#[test]
fn dynamic_truncate_basic() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    ))
    .unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0)).unwrap();

    let result = x.dynamic_truncate(&size, 0).unwrap();
    let data = get_f64_data(&result.run_with(&engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn dynamic_truncate_2d_axis1() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![2, 4],
        vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0],
    ))
    .unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(2.0)).unwrap();

    let result = x.dynamic_truncate(&size, 1).unwrap();
    let out = result.run_with(&engine).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(get_f64_data(&out), vec![1.0, 5.0, 2.0, 6.0]);
}

#[test]
fn dynamic_truncate_clamps_oversize() {
    let engine = cpu_runtime();
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(10.0)).unwrap();

    let result = x.dynamic_truncate(&size, 0).unwrap();
    let data = get_f64_data(&result.run_with(&engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0]);
}

#[test]
fn dynamic_truncate_accepts_i64_size() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(i64_scalar(2)).unwrap();

    let result = x.dynamic_truncate(&size, 0).unwrap();
    let out = result.run_with(&engine).unwrap();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(get_f64_data(&out), vec![1.0, 2.0]);
}

#[test]
fn dynamic_truncate_preserves_data_dtype_and_size_dtype() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f32_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(2.0)).unwrap();

    let result = x.dynamic_truncate(&size, 0).unwrap();
    assert_eq!(result.dtype, DType::F32);

    let out = result.run_with(&engine).unwrap();
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.shape(), &[2]);
    assert_eq!(get_f32_data(&out), vec![1.0, 2.0]);
}

#[test]
fn dynamic_truncate_rejects_backend_size_binding_on_cpu() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let size = TracedTensor::input_concrete_shape(DType::F64, &[]).unwrap();
    let result = x.dynamic_truncate(&size, 0).unwrap();

    let err = result
        .run_with_inputs_auto(&engine, &[(&size, &backend_f64_scalar())])
        .unwrap_err();

    assert_eq!(err.kind(), ErrorKind::RuntimeState);
    assert!(matches!(
        err,
        RuntimeError::TensorRuntime(TensorError::RuntimeState {
            op: "CpuBackend::download_to_host",
            ..
        })
    ));
}

#[test]
fn pad_to_match_basic() {
    let engine = cpu_runtime();
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let reference =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![0.0; 5])).unwrap();

    let result = x.pad_to_match(&reference, 0).unwrap();
    let data = get_f64_data(&result.run_with(&engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0, 0.0, 0.0]);
}

#[test]
fn pad_to_match_no_op_when_same_size() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let reference =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![4], vec![0.0; 4])).unwrap();

    let result = x.pad_to_match(&reference, 0).unwrap();
    let data = get_f64_data(&result.run_with(&engine).unwrap());
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn pad_to_match_preserves_data_dtype_and_infers_output_shape() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f32_tensor(
        vec![3, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ))
    .unwrap();
    let reference =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5, 7], vec![0.0; 35])).unwrap();

    let result = x.pad_to_match(&reference, 0).unwrap();
    assert_eq!(result.dtype, DType::F32);
    assert_eq!(result.try_concrete_shape().unwrap(), vec![5, 2]);

    let out = result.run_with(&engine).unwrap();
    assert_eq!(out.dtype(), DType::F32);
    assert_eq!(out.shape(), &[5, 2]);
    assert_eq!(
        get_f32_data(&out),
        vec![1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0]
    );
}

#[test]
fn dynamic_truncate_vjp_correct() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    ))
    .unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0)).unwrap();

    let truncated = x.dynamic_truncate(&size, 0).unwrap();
    let loss = (&truncated * &truncated)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();

    let grad = loss.grad(&x).unwrap();
    let grad_data = get_f64_data(&grad.run_with(&engine).unwrap());
    assert_eq!(grad_data, vec![2.0, 4.0, 6.0, 0.0, 0.0]);
}

#[test]
fn dynamic_truncate_jvp_correct() {
    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    ))
    .unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0)).unwrap();

    let truncated = x.dynamic_truncate(&size, 0).unwrap();
    let loss = (&truncated * &truncated)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();

    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![1.0; 5])).unwrap();
    let jvp_result = loss.jvp(&x, &v).unwrap();
    let jvp_value = get_f64_data(&jvp_result.run_with(&engine).unwrap())[0];
    assert!(
        (jvp_value - 12.0).abs() < TOL,
        "jvp={jvp_value}, expected=12.0"
    );
}

#[test]
fn dynamic_truncate_hvp_correct() {
    let engine = cpu_runtime();

    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(
        vec![5],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    ))
    .unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0)).unwrap();
    let truncated = x.dynamic_truncate(&size, 0).unwrap();
    let loss = (&truncated * &truncated)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();

    let grad = loss.grad(&x).unwrap();
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![1.0; 5])).unwrap();
    let hv = grad.jvp(&x, &v).unwrap();
    let hv_data = get_f64_data(&hv.run_with(&engine).unwrap());

    assert_eq!(hv_data.len(), 5);
    for (index, value) in hv_data.iter().copied().enumerate().take(3) {
        assert!(
            (value - 2.0).abs() < TOL,
            "hv[{index}]={value}, expected 2.0"
        );
    }
    for (index, value) in hv_data.iter().copied().enumerate().skip(3) {
        assert!(value.abs() < TOL, "hv[{index}]={value}, expected 0.0");
    }
}

#[test]
fn dynamic_truncate_hvp_finite_diff() {
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let v_data = vec![0.1, -0.2, 0.3, 0.4, -0.5];

    let compute_grad = |values: &[f64]| {
        let engine = cpu_runtime();
        let x =
            TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], values.to_vec())).unwrap();
        let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0)).unwrap();
        let truncated = x.dynamic_truncate(&size, 0).unwrap();
        let loss = (&truncated * &truncated)
            .unwrap()
            .reduce_sum(Some(&[0]))
            .unwrap();
        let grad = loss.grad(&x).unwrap();
        get_f64_data(&grad.run_with(&engine).unwrap())
    };

    let mut x_plus = x_data.clone();
    let mut x_minus = x_data.clone();
    for (index, direction) in v_data.iter().copied().enumerate() {
        x_plus[index] += FD_H * direction;
        x_minus[index] -= FD_H * direction;
    }
    let grad_plus = compute_grad(&x_plus);
    let grad_minus = compute_grad(&x_minus);
    let fd_hv: Vec<f64> = grad_plus
        .iter()
        .zip(&grad_minus)
        .map(|(plus, minus)| (plus - minus) / (2.0 * FD_H))
        .collect();

    let engine = cpu_runtime();
    let x = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], x_data)).unwrap();
    let size = TracedTensor::from_tensor_concrete_shape(f64_scalar(3.0)).unwrap();
    let truncated = x.dynamic_truncate(&size, 0).unwrap();
    let loss = (&truncated * &truncated)
        .unwrap()
        .reduce_sum(Some(&[0]))
        .unwrap();
    let grad = loss.grad(&x).unwrap();
    let v = TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], v_data)).unwrap();
    let hv = grad.jvp(&x, &v).unwrap();
    let hv_data = get_f64_data(&hv.run_with(&engine).unwrap());

    for (index, (actual, expected)) in hv_data.iter().zip(&fd_hv).enumerate() {
        assert!(
            (*actual - *expected).abs() < TOL,
            "HVP[{index}]: ad={actual}, fd={expected}"
        );
    }
}

// ═══════════════════════════════════════════
// PadToMatch AD tests
// ═══════════════════════════════════════════

#[test]
fn pad_to_match_vjp_correct() {
    // f(x) = sum(pad_to_match(x, ref, axis=0)^2)
    // x = [1,2,3], ref has size 5 → padded = [1,2,3,0,0]
    // loss = 1+4+9 = 14, grad = [2,4,6] (truncated back from [2,4,6,0,0])
    let engine = cpu_runtime();
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let reference =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![0.0; 5])).unwrap();

    let padded = x.pad_to_match(&reference, 0).unwrap();
    let loss = (&padded * &padded).unwrap().reduce_sum(Some(&[0])).unwrap();

    let grad = loss.grad(&x).unwrap();
    let grad_data = get_f64_data(&grad.run_with(&engine).unwrap());
    assert_eq!(grad_data, vec![2.0, 4.0, 6.0]);
}

#[test]
fn pad_to_match_jvp_correct() {
    let engine = cpu_runtime();
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let reference =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![0.0; 5])).unwrap();

    let padded = x.pad_to_match(&reference, 0).unwrap();
    let loss = (&padded * &padded).unwrap().reduce_sum(Some(&[0])).unwrap();

    let v =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 1.0, 1.0])).unwrap();
    let jvp_result = loss.jvp(&x, &v).unwrap();
    let jvp_val = get_f64_data(&jvp_result.run_with(&engine).unwrap())[0];
    // dot(grad, v) = 2+4+6 = 12
    assert!((jvp_val - 12.0).abs() < TOL, "jvp={jvp_val}, expected=12");
}

#[test]
fn pad_to_match_hvp_correct() {
    // f(x) = sum(pad(x,ref,0)^2) → Hessian = diag(2,2,2)
    // HVP with v=[1,1,1] → [2,2,2]
    let engine = cpu_runtime();
    let x =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 2.0, 3.0])).unwrap();
    let reference =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![5], vec![0.0; 5])).unwrap();

    let padded = x.pad_to_match(&reference, 0).unwrap();
    let loss = (&padded * &padded).unwrap().reduce_sum(Some(&[0])).unwrap();

    let grad = loss.grad(&x).unwrap();
    let v =
        TracedTensor::from_tensor_concrete_shape(f64_tensor(vec![3], vec![1.0, 1.0, 1.0])).unwrap();
    let hv = grad.jvp(&x, &v).unwrap();
    let hv_data = get_f64_data(&hv.run_with(&engine).unwrap());

    for (i, val) in hv_data.iter().enumerate() {
        assert!((*val - 2.0).abs() < TOL, "HVP[{i}]={val}, expected=2.0");
    }
}
