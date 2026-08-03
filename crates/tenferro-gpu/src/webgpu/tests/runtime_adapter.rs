use std::any::Any;
use std::sync::Arc;

#[cfg(not(target_family = "wasm"))]
use tenferro_runtime::runtime::{EventDomainDriver, EventToken};
#[cfg(not(target_family = "wasm"))]
use tenferro_runtime::{
    assemble_preparation_only_engine_registration, CoreCapabilityBundle, EngineId,
    EngineRegistrationMetadata, EventDomainId, ExecutionContextIdentity, HardwareClassId,
    PreparationOnlyEngineRegistrationConfig, ProviderDeviceIdentity, ProviderId, Runtime,
    StorageClass,
};
#[cfg(not(target_family = "wasm"))]
use tenferro_tensor::TensorStructural;
use tenferro_tensor::{BackendBuffer, Buffer, DeviceId, Tensor, TypedTensor};

use super::*;

use crate::{download_webgpu_tensor, upload_webgpu_tensor, webgpu_available};

#[derive(Debug)]
struct TestWebGpuBuffer {
    family: &'static str,
    domain: Option<AllocationDomainId>,
}

#[derive(Debug)]
#[cfg(not(target_family = "wasm"))]
struct FailingEventToken {
    origin: EventDomainId,
}

#[cfg(not(target_family = "wasm"))]
impl EventToken for FailingEventToken {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn origin(&self) -> EventDomainId {
        self.origin
    }

    fn wait(&self) -> tenferro_runtime::Result<()> {
        Err(tenferro_runtime::Error::runtime_state(
            "failing_event_token",
            tenferro_runtime::ErrorPhase::Execution,
            "injected dependency failure",
        ))
    }
}

#[cfg(not(target_family = "wasm"))]
fn test_event_domain(suffix: &str) -> EventDomainId {
    let engine_id = EngineId::new(format!("tenferro.test.webgpu.event.{suffix}"))
        .expect("WebGPU event test engine id");
    let storage = StorageClass::new(format!("tenferro.test.webgpu.storage.{suffix}"))
        .expect("WebGPU event test storage class");
    let metadata = EngineRegistrationMetadata::new(
        engine_id.clone(),
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.webgpu").expect("WebGPU event test provider"),
            format!("target:{suffix}"),
        )
        .expect("WebGPU event test provider device"),
        HardwareClassId::new("tenferro.test.webgpu").expect("WebGPU event test hardware class"),
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    );
    let registration = assemble_preparation_only_engine_registration(
        PreparationOnlyEngineRegistrationConfig::new(
            metadata,
            ExecutionContextIdentity::of::<()>(),
        ),
    )
    .expect("WebGPU event test registration");
    let mut builder = Runtime::builder();
    builder
        .register_engine(registration)
        .expect("WebGPU event test engine registration");
    let runtime = builder.build().expect("WebGPU event test runtime");
    let snapshot = runtime.snapshot().expect("WebGPU event test snapshot");
    snapshot
        .engine(&engine_id)
        .expect("WebGPU event test engine snapshot")
        .event_domain_id()
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

#[test]
#[cfg(not(target_family = "wasm"))]
fn webgpu_event_domain_rejects_same_origin_incompatible_token_before_launch() {
    let domain = test_event_domain("incompatible-token");
    let dependency: Arc<dyn EventToken> = Arc::new(FailingEventToken { origin: domain });
    let mut launches = 0;

    let error = super::super::event_domain::admit_webgpu_tokens(
        std::slice::from_ref(&dependency),
        domain,
        |_| {
            launches += 1;
            Ok(())
        },
    )
    .expect_err("same-origin non-WebGPU token must be rejected");
    let tenferro_runtime::Error::EventDomain {
        source:
            tenferro_runtime::runtime::EventDomainError::IncompatibleTokenType {
                operation,
                node_index,
                expected,
                actual,
                token_type,
            },
    } = error
    else {
        panic!("same-origin non-WebGPU token must retain its typed admission error");
    };
    assert_eq!(
        operation,
        tenferro_runtime::runtime::EventDomainOperation::Enqueue
    );
    assert_eq!(node_index, None);
    assert_eq!(expected, domain);
    assert_eq!(actual, domain);
    assert_eq!(token_type, "non-WebGPU event token");
    assert_eq!(launches, 0);
}

#[test]
#[cfg(not(target_family = "wasm"))]
fn webgpu_registration_preserves_a_caller_selected_engine_id() {
    if !webgpu_available() {
        return;
    }
    let Ok(backend) = WebGpuBackend::new_default() else {
        return;
    };
    let engine_id =
        EngineId::new("tenferro-webgpu.test.selected.v1").expect("selected WebGPU engine ID");
    let registration = webgpu_runtime_engine_registration_with_id(&backend, engine_id.clone())
        .expect("selected WebGPU registration");
    assert_eq!(registration.engine_id(), &engine_id);
    assert_eq!(
        registration.hardware_class(),
        &webgpu_runtime_hardware_class().expect("WebGPU hardware class")
    );
}

#[test]
#[cfg(not(target_family = "wasm"))]
#[ignore = "requires a native WebGPU adapter"]
fn webgpu_event_domain_tokens_are_repeatable_and_order_native_dependencies() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().expect("WebGPU backend");
    let runtime = backend.runtime().clone();
    let host =
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).expect("host input");
    let input = upload_webgpu_tensor(&runtime, &host).expect("WebGPU upload");
    let driver = WebGpuEventDomainDriver::new(runtime.clone());
    let domain = test_event_domain("native");
    let run = driver.begin_run(domain).expect("WebGPU event-domain run");

    // A run may cross scheduler worker threads. Its captured CubeCL stream must
    // remain stable rather than following each worker's thread-local stream.
    let input_for_first = input.clone();
    let (mut run, mut backend, first_output, first_completion, first_launches) =
        std::thread::spawn(move || {
            let mut run = run;
            let mut first_output = None;
            let mut first_launches = 0;
            let mut first = || {
                first_launches += 1;
                first_output = Some(
                    backend
                        .transpose(&input_for_first, &[1, 0])
                        .map_err(tenferro_runtime::Error::from)?,
                );
                Ok(())
            };
            let first_completion = run.enqueue(&[], &mut first).expect("first enqueue");
            (
                run,
                backend,
                first_output.expect("first output"),
                first_completion,
                first_launches,
            )
        })
        .join()
        .expect("WebGPU launch worker");
    assert_eq!(first_launches, 1);

    let mut second_output = None;
    let mut second_launches = 0;
    let mut second = || {
        second_launches += 1;
        second_output = Some(
            backend
                .transpose(&first_output, &[1, 0])
                .map_err(tenferro_runtime::Error::from)?,
        );
        Ok(())
    };
    let second_completion = run
        .enqueue(&[first_completion], &mut second)
        .expect("dependent enqueue");
    assert_eq!(second_launches, 1);

    second_completion.wait().expect("first completion wait");
    second_completion.wait().expect("repeat completion wait");
    run.drain().expect("WebGPU event-domain drain");

    let mut panic_output = None;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut panicking = || -> tenferro_runtime::Result<()> {
            panic_output = Some(
                backend
                    .transpose(&input, &[1, 0])
                    .map_err(tenferro_runtime::Error::from)?,
            );
            panic!("injected post-launch panic");
        };
        let _ = run.enqueue(&[], &mut panicking);
    }));
    assert!(unwind.is_err());
    let panic_output = download_webgpu_tensor(
        &runtime,
        panic_output.as_ref().expect("panic-path output retained"),
    )
    .expect("panic-path work retired before unwind returned");
    let Tensor::F32(panic_output) = panic_output else {
        unreachable!("f32 panic-path output")
    };
    assert_eq!(
        panic_output.as_slice().expect("panic-path host slice"),
        &[1.0, 3.0, 2.0, 4.0]
    );

    let output = download_webgpu_tensor(&runtime, second_output.as_ref().expect("second output"))
        .expect("WebGPU download");
    let Tensor::F32(output) = output else {
        unreachable!("f32 elementwise output")
    };
    assert_eq!(
        output.as_slice().expect("host slice"),
        &[1.0, 2.0, 3.0, 4.0]
    );
}
