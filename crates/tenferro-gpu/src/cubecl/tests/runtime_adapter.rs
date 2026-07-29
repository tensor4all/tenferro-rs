use std::any::Any;
use std::sync::Arc;

use tenferro_tensor::{BackendBuffer, Buffer, DeviceId, Tensor, TypedTensor};

use super::*;

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
fn cuda_registration_ingress_accepts_own_device_and_rejects_foreign_inputs() {
    let valid = input("cubecl", 3);
    let foreign_family = input("foreign-cuda", 3);
    let foreign_device = input("cubecl", 4);

    assert!(cuda_input_tensor(&TensorRead::from_tensor(&valid), 3));
    assert!(!cuda_input_tensor(
        &TensorRead::from_tensor(&foreign_family),
        3
    ));
    assert!(!cuda_input_tensor(
        &TensorRead::from_tensor(&foreign_device),
        3
    ));
}
