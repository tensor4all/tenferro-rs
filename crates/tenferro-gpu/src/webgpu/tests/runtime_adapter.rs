use std::any::Any;
use std::sync::Arc;

use tenferro_tensor::{BackendBuffer, Buffer, DeviceId, Tensor, TypedTensor};

use super::*;
use crate::{upload_webgpu_tensor, webgpu_available};

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
fn webgpu_registration_ingress_rejects_forged_family_domain_and_foreign_inputs() {
    let domain = AllocationDomainId::fresh();
    let forged_managed = input("cubecl-webgpu", 2, Some(domain));
    let forged_device = input("cubecl-webgpu", 2, None);
    let foreign_family = input("foreign-webgpu", 2, Some(domain));
    let foreign_domain = input("cubecl-webgpu", 2, Some(AllocationDomainId::fresh()));

    assert!(!webgpu_input_tensor(
        &TensorRead::from_tensor(&forged_managed),
        2,
        Some(domain)
    ));
    assert!(!webgpu_input_tensor(
        &TensorRead::from_tensor(&forged_device),
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

#[test]
fn webgpu_registration_ingress_accepts_backend_created_tensor() {
    if !webgpu_available() {
        return;
    }
    let backend = WebGpuBackend::new_default().expect("WebGPU backend");
    let host = Tensor::from_vec_col_major(vec![1], vec![1.0_f32]).expect("host tensor");
    let input = upload_webgpu_tensor(backend.runtime(), &host).expect("WebGPU upload");
    let ordinal = backend.runtime().device_ordinal();
    let domain = backend
        .runtime()
        .allocation_domain()
        .map(|domain| domain.id);

    assert!(webgpu_input_tensor(
        &TensorRead::from_tensor(&input),
        ordinal,
        domain
    ));

    let Tensor::F32(typed) = &input else {
        unreachable!("uploaded f32 tensor")
    };
    let Buffer::Backend(buffer) = typed.buffer() else {
        unreachable!("uploaded WebGPU buffer")
    };
    let foreign_ordinal = ordinal.saturating_add(1);
    let relabeled = TypedTensor::<f32>::from_buffer_col_major(
        vec![1],
        Buffer::Backend(Arc::clone(buffer)),
        Placement {
            memory_kind: input.placement().memory_kind.clone(),
            device: Some(DeviceId {
                kind: DeviceKind::Gpu(GpuBackendKind::WebGpu),
                ordinal: foreign_ordinal,
            }),
            cpu_affinity: None,
        },
    )
    .expect("relabeled WebGPU tensor")
    .into();
    assert!(!webgpu_input_tensor(
        &TensorRead::from_tensor(&relabeled),
        foreign_ordinal,
        domain
    ));
}
