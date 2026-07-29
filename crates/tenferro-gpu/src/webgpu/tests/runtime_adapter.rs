use std::any::Any;
use std::sync::Arc;

use tenferro_tensor::{BackendBuffer, Buffer, DeviceId, Tensor, TypedTensor};

use super::*;

#[derive(Debug)]
struct TestWebGpuBuffer {
    family: &'static str,
    domain: Option<AllocationDomainId>,
}

impl BackendBuffer<f32> for TestWebGpuBuffer {
    fn backend_family(&self) -> &'static str {
        self.family
    }

    fn len(&self) -> usize {
        1
    }

    fn allocation_domain(&self) -> Option<AllocationDomainId> {
        self.domain
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn input(family: &'static str, ordinal: usize, domain: Option<AllocationDomainId>) -> Tensor {
    TypedTensor::<f32>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::new(TestWebGpuBuffer { family, domain })),
        Placement {
            memory_kind: if domain.is_some() {
                MemoryKind::Managed
            } else {
                MemoryKind::Device
            },
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::WebGpu),
                ordinal,
            }),
            cpu_affinity: None,
        },
    )
    .expect("test tensor")
    .into()
}

#[test]
fn webgpu_registration_ingress_accepts_own_domain_and_rejects_foreign_inputs() {
    let domain = AllocationDomainId::fresh();
    let valid = input("cubecl-webgpu", 2, Some(domain));
    let valid_device = input("cubecl-webgpu", 2, None);
    let foreign_family = input("foreign-webgpu", 2, Some(domain));
    let foreign_domain = input("cubecl-webgpu", 2, Some(AllocationDomainId::fresh()));

    assert!(webgpu_input_tensor(
        &TensorRead::from_tensor(&valid),
        2,
        Some(domain)
    ));
    assert!(webgpu_input_tensor(
        &TensorRead::from_tensor(&valid_device),
        2,
        None
    ));
    assert!(!webgpu_input_tensor(
        &TensorRead::from_tensor(&foreign_family),
        2,
        Some(domain)
    ));
    assert!(!webgpu_input_tensor(
        &TensorRead::from_tensor(&foreign_domain),
        2,
        Some(domain)
    ));
}
