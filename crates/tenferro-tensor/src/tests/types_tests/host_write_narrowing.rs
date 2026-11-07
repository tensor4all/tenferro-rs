//! The write-side host narrowing: `TensorWrite`, `TensorViewMut` and
//! `TypedTensorViewMut::try_mut_host` refuse with the same variants and text as
//! the read-side `try_host`, and resolve placement labels the same way.

use super::{backend_tensor_f64, external_f64_tensor};
use crate::types::{
    DType, DeviceId, DeviceKind, GpuBackendKind, MemoryKind, Placement, Tensor, TensorRead,
    TensorViewMut, TensorWrite, TypedTensor,
};
use crate::Error;
use num_complex::{Complex32, Complex64};

fn device_placement() -> Placement {
    Placement {
        memory_kind: MemoryKind::Device,
        device: Some(DeviceId {
            kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
            ordinal: 0,
        }),
        cpu_affinity: None,
    }
}

#[test]
fn tensor_write_try_mut_host_covers_every_preset_scalar() {
    let mut cases = [
        Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![Complex32::new(1.0, 0.0); 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2], vec![Complex64::new(1.0, 0.0); 2]).unwrap(),
    ];
    let expected = [
        DType::F32,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::Bool,
        DType::C32,
        DType::C64,
    ];
    for (tensor, dtype) in cases.iter_mut().zip(expected) {
        let host = TensorWrite::from_tensor(tensor)
            .try_mut_host("example")
            .unwrap();
        assert_eq!(host.dtype(), dtype);
        assert_eq!(host.shape(), &[2]);
        assert_eq!(host.strides(), &[1]);
    }
}

#[test]
fn tensor_write_try_mut_host_writes_through_to_the_owner_and_the_view() {
    let mut tensor = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    match TensorWrite::from_tensor(&mut tensor)
        .try_mut_host("example")
        .unwrap()
    {
        TensorViewMut::F64(mut view) => view.as_host_slice_mut().copy_from_slice(&[3.0, 4.0]),
        other => panic!("expected an f64 arm, got {:?}", other.dtype()),
    }
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[3.0, 4.0]);

    let mut data = [0_i32, 9, 0, 9];
    let view = TensorViewMut::I32(
        crate::TypedTensorViewMut::from_slice(vec![2], vec![2], 0, &mut data).unwrap(),
    );
    match TensorWrite::from_view(view)
        .try_mut_host("example")
        .unwrap()
    {
        TensorViewMut::I32(mut view) => {
            assert_eq!(view.strides(), &[2]);
            *view.get_mut(&[1]).unwrap() = 7;
        }
        other => panic!("expected an i32 arm, got {:?}", other.dtype()),
    }
    assert_eq!(data, [0, 9, 7, 9]);
}

#[test]
fn try_mut_host_refuses_backend_storage_as_runtime_state_like_the_read_side() {
    let mut tensor = Tensor::from_typed(backend_tensor_f64(301, 2));
    let read_err = TensorRead::from_tensor(&tensor)
        .try_host("solve_read_into")
        .unwrap_err();
    let write_err = TensorWrite::from_tensor(&mut tensor)
        .try_mut_host("solve_read_into")
        .unwrap_err();
    match (&read_err, &write_err) {
        (
            Error::RuntimeState {
                op: read_op,
                message: read_message,
            },
            Error::RuntimeState {
                op: write_op,
                message: write_message,
            },
        ) => {
            assert_eq!(read_op, write_op);
            assert_eq!(read_message, write_message);
            assert_eq!(*write_op, "solve_read_into");
        }
        other => panic!("expected runtime-state refusals, got {other:?}"),
    }

    let mut typed = backend_tensor_f64(302, 2);
    let err = typed.as_view_mut().try_mut_host("typed").unwrap_err();
    assert!(
        matches!(err, Error::RuntimeState { op: "typed", .. }),
        "{err}"
    );

    let mut typed = backend_tensor_f64(303, 2);
    let err = TensorViewMut::F64(typed.as_view_mut())
        .try_mut_host("erased")
        .unwrap_err();
    assert!(
        matches!(err, Error::RuntimeState { op: "erased", .. }),
        "{err}"
    );
}

#[test]
fn try_mut_host_refuses_an_external_payload_as_unsupported_dtype_like_the_read_side() {
    let mut tensor = external_f64_tensor();
    let read_err = TensorRead::from_tensor(&tensor)
        .try_host("example")
        .unwrap_err();
    let write_err = TensorWrite::from_tensor(&mut tensor)
        .try_mut_host("example")
        .unwrap_err();
    assert_eq!(read_err.to_string(), write_err.to_string());
    assert!(
        matches!(write_err, Error::UnsupportedDType { op: "example", .. }),
        "{write_err}"
    );
}

#[test]
fn try_mut_host_resolves_a_device_label_on_host_storage_like_the_read_side() {
    let mut typed = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    typed.set_placement(Placement {
        cpu_affinity: None,
        ..device_placement()
    });
    let read_placement = typed
        .as_view()
        .try_host("example")
        .unwrap()
        .placement()
        .clone();
    let host = typed.as_view_mut().try_mut_host("example").unwrap();
    assert_eq!(host.placement(), &read_placement);
    assert_eq!(host.placement().memory_kind, MemoryKind::UnpinnedHost);
    assert!(host.placement().device.is_none());

    let dynamic = host.into_dynamic();
    assert_eq!(dynamic.placement().memory_kind, MemoryKind::UnpinnedHost);
}
