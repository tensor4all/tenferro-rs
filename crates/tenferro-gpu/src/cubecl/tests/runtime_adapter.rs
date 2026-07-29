use std::any::Any;
use std::sync::Arc;

use tenferro_tensor::{BackendBuffer, Buffer, DeviceId, Tensor, TypedTensor};

use super::*;
use crate::{gpu_available, upload_tensor, CudaRuntime};

#[derive(Debug)]
struct TestCudaBuffer {
    family: &'static str,
}

impl BackendBuffer<f32> for TestCudaBuffer {
    fn backend_family(&self) -> &'static str {
        self.family
    }

    fn len(&self) -> usize {
        1
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn input(family: &'static str, ordinal: usize) -> Tensor {
    TypedTensor::<f32>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::new(TestCudaBuffer { family })),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal,
            }),
            cpu_affinity: None,
        },
    )
    .expect("test tensor")
    .into()
}

#[test]
fn cuda_registration_ingress_rejects_forged_family_and_foreign_inputs() {
    let forged_family = input("cubecl", 3);
    let foreign_family = input("foreign-cuda", 3);
    let foreign_device = input("cubecl", 4);

    assert!(!cuda_input_tensor(
        &TensorRead::from_tensor(&forged_family),
        3
    ));
    assert!(!cuda_input_tensor(
        &TensorRead::from_tensor(&foreign_family),
        3
    ));
    assert!(!cuda_input_tensor(
        &TensorRead::from_tensor(&foreign_device),
        3
    ));
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_registration_ingress_accepts_backend_created_tensor() {
    if !gpu_available() {
        return;
    }
    let runtime = CudaRuntime::new(0).expect("CUDA runtime");
    let host = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).expect("host tensor");
    let input = upload_tensor(&runtime, &host).expect("CUDA upload");

    assert!(cuda_input_tensor(&TensorRead::from_tensor(&input), 0));

    let Tensor::F32(typed) = &input else {
        unreachable!("uploaded f32 tensor")
    };
    let Buffer::Backend(buffer) = typed.buffer() else {
        unreachable!("uploaded CUDA buffer")
    };
    let relabeled = TypedTensor::<f32>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::clone(buffer)),
        Placement {
            memory_kind: MemoryKind::Device,
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::Cuda),
                ordinal: 1,
            }),
            cpu_affinity: None,
        },
    )
    .expect("relabeled CUDA tensor")
    .into();
    assert!(!cuda_input_tensor(&TensorRead::from_tensor(&relabeled), 1));
}

#[test]
fn sum_squares_routes_through_runtime_reduction_preparation() {
    assert_eq!(
        cuda_operation_kind(&CoreSemanticOp::ReduceSumSquares { axes: vec![0] }),
        Some(CudaPreparedKind::Reduction)
    );
    assert_eq!(
        core_operation_name(&CoreSemanticOp::ReduceSumSquares { axes: vec![0] }),
        "reduce_sum_squares"
    );
}
