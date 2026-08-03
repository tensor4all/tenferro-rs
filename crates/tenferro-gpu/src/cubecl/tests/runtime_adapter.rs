use std::any::Any;
use std::sync::{Arc, Barrier};

use tenferro_runtime::runtime::{EventDomainDriver, EventToken};
use tenferro_runtime::{
    assemble_preparation_only_engine_registration, CoreCapabilityBundle, DType, EngineId,
    EngineRegistrationMetadata, EventDomainId, ExecutionContextIdentity, GraphCompiler,
    HardwareClassId, PreparationOnlyEngineRegistrationConfig, ProviderDeviceIdentity, ProviderId,
    Runtime, StorageClass, TracedTensor,
};
use tenferro_tensor::{
    BackendStorage, DeviceId, StorageBuffer, Tensor, TensorElementwise, TypedTensor,
};

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
    let metadata = EngineRegistrationMetadata::new(
        engine_id.clone(),
        ProviderDeviceIdentity::new(
            ProviderId::new("tenferro.test.cuda").expect("CUDA event test provider"),
            format!("target:{suffix}"),
        )
        .expect("CUDA event test provider device"),
        HardwareClassId::new("tenferro.test.cuda").expect("CUDA event test hardware class"),
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

impl BackendStorage<f32> for TestCudaBuffer {
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
        StorageBuffer::Backend(Arc::new(TestCudaBuffer { family })),
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
    let StorageBuffer::Backend(buffer) = typed.buffer() else {
        unreachable!("uploaded CUDA buffer")
    };
    let relabeled = TypedTensor::<f32>::from_buffer_col_major(
        vec![1],
        StorageBuffer::Backend(Arc::clone(buffer)),
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
fn cuda_event_domain_rejects_same_origin_incompatible_token_before_launch() {
    let domain = test_event_domain("incompatible-token");
    let dependency: Arc<dyn EventToken> = Arc::new(FailingEventToken { origin: domain });
    let mut launches = 0;

    let error = super::super::event_domain::admit_cuda_tokens(
        std::slice::from_ref(&dependency),
        domain,
        |_| {
            launches += 1;
            Ok(())
        },
    )
    .expect_err("same-origin non-CUDA token must be rejected");
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
        panic!("same-origin non-CUDA token must retain its typed admission error");
    };
    assert_eq!(
        operation,
        tenferro_runtime::runtime::EventDomainOperation::Enqueue
    );
    assert_eq!(node_index, None);
    assert_eq!(expected, domain);
    assert_eq!(actual, domain);
    assert_eq!(token_type, "non-CUDA event token");
    assert_eq!(launches, 0);
}

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_registration_preserves_two_caller_selected_engine_ids_and_devices() {
    let devices = cuda_devices().unwrap_or_else(|error| {
        panic!("CUDA device discovery failed: {error}");
    });
    if devices.len() < 2 {
        println!(
            "SKIP cuda_registration_preserves_two_caller_selected_engine_ids_and_devices: \
             reason=fewer-than-two-cuda-devices detected_count={}",
            devices.len()
        );
        return;
    }
    let first_device = devices[0].id();
    let second_device = devices[1].id();
    let first_backend = CudaBackend::new(first_device).expect("first CUDA backend");
    let second_backend = CudaBackend::new(second_device).expect("second CUDA backend");
    let first_runtime = first_backend.runtime().clone();
    let second_runtime = second_backend.runtime().clone();
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
    assert_eq!(first_runtime.device_id(), first_device);
    assert_eq!(second_runtime.device_id(), second_device);
    assert_ne!(first_device, second_device);
    assert_ne!(first_engine, second_engine);
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
        first_registration
            .provider_device_identity()
            .provider_id()
            .as_str(),
        "tenferro.cuda"
    );
    assert_eq!(
        second_registration
            .provider_device_identity()
            .provider_id()
            .as_str(),
        "tenferro.cuda"
    );
    assert_ne!(
        first_registration.provider_device_identity(),
        second_registration.provider_device_identity()
    );
    assert_eq!(
        first_registration.hardware_class(),
        &cuda_runtime_hardware_class().expect("CUDA hardware class")
    );
    assert_eq!(
        second_registration.hardware_class(),
        &cuda_runtime_hardware_class().expect("CUDA hardware class")
    );

    let mut builder = Runtime::builder();
    builder
        .register_engine(first_registration)
        .expect("register first selected CUDA engine");
    builder
        .register_engine(second_registration)
        .expect("register second selected CUDA engine");
    let runtime = builder.build().expect("build two-engine CUDA runtime");
    let snapshot = runtime.snapshot().expect("two-engine CUDA snapshot");
    assert_eq!(snapshot.engine_count(), 2);
    assert_eq!(snapshot.transfer_provider_count(), 0);

    let first_view = snapshot
        .engine(&first_engine)
        .expect("first selected CUDA engine snapshot");
    let second_view = snapshot
        .engine(&second_engine)
        .expect("second selected CUDA engine snapshot");
    assert_eq!(first_view.engine_id(), &first_engine);
    assert_eq!(second_view.engine_id(), &second_engine);
    assert_eq!(
        first_view.provider_device_identity().provider_id().as_str(),
        "tenferro.cuda"
    );
    assert_eq!(
        second_view
            .provider_device_identity()
            .provider_id()
            .as_str(),
        "tenferro.cuda"
    );
    assert_eq!(
        first_view.provider_device_identity().target_identity(),
        format!("device:{}", first_device.ordinal())
    );
    assert_eq!(
        second_view.provider_device_identity().target_identity(),
        format!("device:{}", second_device.ordinal())
    );
    assert_ne!(
        first_view.provider_device_identity(),
        second_view.provider_device_identity()
    );
    let first_event_domain = first_view.event_domain_id();
    let second_event_domain = second_view.event_domain_id();
    assert_ne!(first_event_domain, second_event_domain);

    let first_graph_input =
        TracedTensor::input_symbolic_shape(DType::F32, 1).expect("first graph input");
    let first_graph_output = first_graph_input
        .add(&first_graph_input)
        .expect("first graph elementwise add");
    let first_program = GraphCompiler::new()
        .compile_with_input_specs(
            &first_graph_output,
            &[(&first_graph_input, DType::F32, &[4])],
        )
        .expect("compile first CUDA elementwise graph");

    let second_graph_input =
        TracedTensor::input_symbolic_shape(DType::F32, 1).expect("second graph input");
    let second_graph_output = second_graph_input
        .add(&second_graph_input)
        .expect("second graph elementwise add");
    let second_program = GraphCompiler::new()
        .compile_with_input_specs(
            &second_graph_output,
            &[(&second_graph_input, DType::F32, &[4])],
        )
        .expect("compile second CUDA elementwise graph");

    let first_host = Tensor::from_vec_col_major(vec![4], vec![1.0_f32, 2.0, 3.0, 4.0])
        .expect("first host input");
    let second_host = Tensor::from_vec_col_major(vec![4], vec![-1.0_f32, 0.5, 2.0, 3.5])
        .expect("second host input");
    let first_input = upload_tensor(&first_runtime, &first_host).expect("upload first input");
    let second_input = upload_tensor(&second_runtime, &second_host).expect("upload second input");
    assert_eq!(
        TensorRead::from_tensor(&first_input)
            .placement()
            .device
            .as_ref()
            .expect("first input CUDA device")
            .ordinal,
        first_device.ordinal() as usize
    );
    assert_eq!(
        TensorRead::from_tensor(&second_input)
            .placement()
            .device
            .as_ref()
            .expect("second input CUDA device")
            .ordinal,
        second_device.ordinal() as usize
    );

    let first_prepared = runtime
        .prepare_compiled(&first_program, &[&first_input])
        .expect("prepare first CUDA graph");
    let second_prepared = runtime
        .prepare_compiled(&second_program, &[&second_input])
        .expect("prepare second CUDA graph");

    let submission_barrier = Barrier::new(2);
    let (first_values, second_values) = std::thread::scope(|scope| {
        let first_run = scope.spawn(|| {
            submission_barrier.wait();
            let outputs = runtime
                .run_prepared(&first_prepared, &[&first_input])
                .expect("execute first prepared CUDA graph");
            let output = outputs.into_iter().next().expect("first CUDA graph output");
            download_tensor(&first_runtime, &output)
                .expect("download first CUDA graph output")
                .as_slice::<f32>()
                .expect("first host output values")
                .to_vec()
        });
        let second_run = scope.spawn(|| {
            submission_barrier.wait();
            let outputs = runtime
                .run_prepared(&second_prepared, &[&second_input])
                .expect("execute second prepared CUDA graph");
            let output = outputs
                .into_iter()
                .next()
                .expect("second CUDA graph output");
            download_tensor(&second_runtime, &output)
                .expect("download second CUDA graph output")
                .as_slice::<f32>()
                .expect("second host output values")
                .to_vec()
        });
        (
            first_run.join().expect("first CUDA graph thread"),
            second_run.join().expect("second CUDA graph thread"),
        )
    });

    assert_eq!(first_values, vec![2.0_f32, 4.0, 6.0, 8.0]);
    assert_eq!(second_values, vec![-2.0_f32, 1.0, 4.0, 7.0]);
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
