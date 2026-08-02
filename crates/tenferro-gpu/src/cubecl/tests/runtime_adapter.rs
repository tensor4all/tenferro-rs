use std::any::Any;
use std::sync::Arc;

use tenferro_runtime::runtime::{EventDomainDriver, EventToken};
use tenferro_runtime::{
    assemble_preparation_only_engine_registration, CoreCapabilityBundle, EngineId, EventDomainId,
    ExecutionContextIdentity, HardwareClassId, ProviderDeviceIdentity, ProviderId, Runtime,
    StorageClass,
};
use tenferro_tensor::{BackendBuffer, Buffer, DeviceId, Tensor, TensorElementwise, TypedTensor};

use super::*;
use crate::{
    cuda_devices, download_tensor, gpu_available, upload_tensor, CudaBackend, CudaDeviceError,
    CudaDeviceId, CudaDeviceInfo, CudaRuntime,
};

#[test]
fn cuda_public_constructors_and_registration_require_typed_selection() {
    let _: fn(CudaDeviceId) -> Result<CudaRuntime, CudaDeviceError> = CudaRuntime::new;
    let _: fn(CudaDeviceId) -> Result<CudaBackend, CudaDeviceError> = CudaBackend::new;
    let _: fn(&CudaBackend, EngineId) -> Result<EngineRegistration, RuntimeConfigError> =
        cuda_runtime_engine_registration;
    let _: fn(&CudaRuntime) -> CudaDeviceId = CudaRuntime::device_id;
    let _: fn(&CudaBackend) -> CudaDeviceId = CudaBackend::device_id;
}

#[test]
fn caller_selected_devices_and_engine_ids_flow_through_prepared_registration_identity() {
    let first_device = CudaDeviceId::from_ordinal(2);
    let second_device = CudaDeviceId::from_ordinal(7);
    let first_engine =
        EngineId::new("tenferro.test.cuda.selected.first.v1").expect("first test engine ID");
    let second_engine =
        EngineId::new("tenferro.test.cuda.selected.second.v1").expect("second test engine ID");

    let first_identity =
        super::prepare_cuda_registration_identity(first_engine.clone(), first_device)
            .expect("first prepared registration identity");
    let second_identity =
        super::prepare_cuda_registration_identity(second_engine.clone(), second_device)
            .expect("second prepared registration identity");
    assert_ne!(first_device, second_device);
    assert_eq!(first_identity.engine_id, first_engine);
    assert_eq!(second_identity.engine_id, second_engine);
    assert_eq!(
        first_identity.provider_device_identity.target_identity(),
        "device:2"
    );
    assert_eq!(
        second_identity.provider_device_identity.target_identity(),
        "device:7"
    );
}

#[test]
fn unavailable_selection_preserves_requested_id_and_discovered_records() {
    let requested = CudaDeviceId::from_ordinal(9);
    let discovered = vec![
        CudaDeviceInfo::new(CudaDeviceId::from_ordinal(2), "NVIDIA A100"),
        CudaDeviceInfo::new(CudaDeviceId::from_ordinal(7), "NVIDIA H100"),
    ];

    let error = super::super::device::unavailable_device_error(requested, discovered.clone());

    assert!(matches!(
        error,
        CudaDeviceError::Unavailable {
            requested: actual_requested,
            discovered: actual_discovered,
        } if actual_requested == requested && actual_discovered.as_ref() == discovered
    ));
}

#[derive(Debug)]
struct TestCudaBuffer {
    family: &'static str,
}

#[derive(Debug)]
struct FailingEventToken {
    origin: EventDomainId,
}

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

fn test_event_domain(suffix: &str) -> EventDomainId {
    let engine_id = EngineId::new(format!("tenferro.test.cuda.event.{suffix}"))
        .expect("CUDA event test engine id");
    let storage = StorageClass::new(format!("tenferro.test.cuda.storage.{suffix}"))
        .expect("CUDA event test storage class");
    let registration = assemble_preparation_only_engine_registration(
        engine_id.clone(),
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.cuda").expect("CUDA event test provider"),
            format!("target:{suffix}"),
        )
        .expect("CUDA event test provider device"),
        ExecutionContextIdentity::of::<()>(),
        HardwareClassId::new("tenferro.test.cuda").expect("CUDA event test hardware class"),
        Arc::from(vec![storage.clone()]),
        storage,
        CoreCapabilityBundle::default(),
    )
    .expect("CUDA event test registration");
    let mut builder = Runtime::builder();
    builder
        .register_engine(registration)
        .expect("CUDA event test engine registration");
    let runtime = builder.build().expect("CUDA event test runtime");
    let snapshot = runtime.snapshot().expect("CUDA event test snapshot");
    snapshot
        .engine(&engine_id)
        .expect("CUDA event test engine snapshot")
        .event_domain_id()
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
    let runtime = CudaRuntime::new(CudaDeviceId::from_ordinal(0)).expect("CUDA runtime");
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

#[test]
fn cuda_registration_installs_native_event_domain_driver() {
    let _: fn(CudaRuntime) -> super::CudaEventDomainDriver = super::CudaEventDomainDriver::new;
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_registration_preserves_two_caller_selected_engine_ids_and_devices() {
    if !gpu_available() {
        return;
    }
    let devices = cuda_devices().expect("CUDA device discovery");
    if devices.len() < 2 {
        return;
    }
    let first_device = devices[0].id();
    let second_device = devices[1].id();
    let first_backend = CudaBackend::new(first_device).expect("first CUDA backend");
    let second_backend = CudaBackend::new(second_device).expect("second CUDA backend");
    let first_engine =
        EngineId::new("tenferro-cuda.test.selected.first.v1").expect("first CUDA engine ID");
    let second_engine =
        EngineId::new("tenferro-cuda.test.selected.second.v1").expect("second CUDA engine ID");
    let first_registration = cuda_runtime_engine_registration(&first_backend, first_engine.clone())
        .expect("first selected CUDA registration");
    let second_registration =
        cuda_runtime_engine_registration(&second_backend, second_engine.clone())
            .expect("second selected CUDA registration");

    assert_eq!(first_backend.device_id(), first_device);
    assert_eq!(second_backend.device_id(), second_device);
    assert_eq!(first_registration.engine_id(), &first_engine);
    assert_eq!(second_registration.engine_id(), &second_engine);
    assert_eq!(
        first_registration
            .provider_device_identity()
            .target_identity(),
        format!("device:{}", first_device.ordinal())
    );
    assert_eq!(
        second_registration
            .provider_device_identity()
            .target_identity(),
        format!("device:{}", second_device.ordinal())
    );
    assert_eq!(
        first_registration.hardware_class(),
        &cuda_runtime_hardware_class().expect("CUDA hardware class")
    );
    assert_eq!(
        second_registration.hardware_class(),
        &cuda_runtime_hardware_class().expect("CUDA hardware class")
    );
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_event_domain_tokens_are_repeatable_and_order_native_dependencies() {
    if !gpu_available() {
        return;
    }

    let backend = CudaBackend::new(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let runtime = backend.runtime().clone();
    let host = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).expect("host input");
    let input = upload_tensor(&runtime, &host).expect("CUDA upload");
    let driver = CudaEventDomainDriver::new(runtime.clone());
    let domain = test_event_domain("native");
    let run = driver.begin_run(domain).expect("CUDA event-domain run");

    // A run may cross scheduler worker threads. Its captured CubeCL stream must
    // remain stable rather than following each worker's thread-local stream.
    let input_for_first = input.clone();
    let (mut run, mut backend, first_output, first_completion, first_launches) =
        std::thread::spawn(move || {
            let mut run = run;
            let mut backend = backend;
            let mut first_output = None;
            let mut first_launches = 0;
            let mut first = || {
                first_launches += 1;
                first_output = Some(
                    backend
                        .add(&input_for_first, &input_for_first)
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
        .expect("CUDA launch worker");
    assert_eq!(first_launches, 1);

    let mut second_output = None;
    let mut second_launches = 0;
    let mut second = || {
        second_launches += 1;
        second_output = Some(
            backend
                .add(&first_output, &input)
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
    run.drain().expect("CUDA event-domain drain");

    let mut forbidden_launches = 0;
    let mut forbidden = || {
        forbidden_launches += 1;
        Ok(())
    };
    let dependency_error = run.enqueue(
        &[Arc::new(FailingEventToken { origin: domain })],
        &mut forbidden,
    );
    assert!(matches!(
        dependency_error,
        Err(tenferro_runtime::Error::EventDomain {
            source: tenferro_runtime::runtime::EventDomainError::IncompatibleTokenType {
                operation: tenferro_runtime::runtime::EventDomainOperation::Enqueue,
                expected,
                actual,
                token_type: "non-CUDA event token",
                ..
            }
        }) if expected == domain && actual == domain
    ));
    assert_eq!(forbidden_launches, 0);

    let mut panic_output = None;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut panicking = || -> tenferro_runtime::Result<()> {
            panic_output = Some(
                backend
                    .add(&input, &input)
                    .map_err(tenferro_runtime::Error::from)?,
            );
            panic!("injected post-launch panic");
        };
        let _ = run.enqueue(&[], &mut panicking);
    }));
    assert!(unwind.is_err());
    let panic_output = download_tensor(
        &runtime,
        panic_output.as_ref().expect("panic-path output retained"),
    )
    .expect("panic-path work retired before unwind returned");
    let Tensor::F32(panic_output) = panic_output else {
        unreachable!("f32 panic-path output")
    };
    assert_eq!(
        panic_output.as_slice().expect("panic-path host slice"),
        &[2.0, 4.0]
    );

    let output = download_tensor(&runtime, second_output.as_ref().expect("second output"))
        .expect("CUDA download");
    let Tensor::F32(output) = output else {
        unreachable!("f32 elementwise output")
    };
    assert_eq!(output.as_slice().expect("host slice"), &[3.0, 6.0]);
}
