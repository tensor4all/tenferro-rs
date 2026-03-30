use tenferro::{
    backward, forward_ad, runtime, AdMode, BackwardOptions, ComputeDevice, LogicalMemorySpace,
    RuntimeContext, Tensor,
};
use tenferro_prims::CpuContext;

#[test]
fn tensor_reports_memory_space_and_device_preference() {
    let mut x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    assert_eq!(x.memory_space(), LogicalMemorySpace::MainMemory);
    assert_eq!(x.preferred_compute_device(), None);

    x.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
    assert_eq!(
        x.preferred_compute_device(),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );
}

#[test]
fn placement_surface_covers_all_frontend_dtypes_and_gpu_sugar() {
    let mut f32_tensor = Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
    let mut f64_tensor = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    let mut c32_tensor =
        Tensor::from_slice(&[num_complex::Complex32::new(1.0, 0.5)], &[1]).unwrap();
    let mut c64_tensor =
        Tensor::from_slice(&[num_complex::Complex64::new(1.0, -0.5)], &[1]).unwrap();

    for tensor in [
        &mut f32_tensor,
        &mut f64_tensor,
        &mut c32_tensor,
        &mut c64_tensor,
    ] {
        assert_eq!(tensor.memory_space(), LogicalMemorySpace::MainMemory);
        tensor.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
        assert_eq!(
            tensor.preferred_compute_device(),
            Some(ComputeDevice::Cpu { device_id: 0 })
        );

        let sync = tensor.to_cpu().unwrap();
        let async_copy = tensor.to_cpu_async().unwrap();
        sync.wait();
        async_copy.wait();
        assert!(sync.is_ready());
        assert!(async_copy.is_ready());
    }

    let gpu_sync = f64_tensor.to_gpu();
    let gpu_async = f64_tensor.to_gpu_async();
    let gpu_sync_on = f64_tensor.to_gpu_on(1);
    let gpu_async_on = f64_tensor.to_gpu_async_on(1);

    assert!(
        matches!(
            gpu_sync,
            Ok(ref tensor)
                if tensor.memory_space() == LogicalMemorySpace::GpuMemory { device_id: 0 }
        ) || matches!(gpu_sync, Err(tenferro::Error::Backend(_)))
    );
    assert!(
        matches!(
            gpu_async,
            Ok(ref tensor)
                if tensor.memory_space() == LogicalMemorySpace::GpuMemory { device_id: 0 }
        ) || matches!(gpu_async, Err(tenferro::Error::Backend(_)))
    );
    assert!(
        matches!(
            gpu_sync_on,
            Ok(ref tensor)
                if tensor.memory_space() == LogicalMemorySpace::GpuMemory { device_id: 1 }
        ) || matches!(gpu_sync_on, Err(tenferro::Error::Backend(_)))
    );
    assert!(
        matches!(
            gpu_async_on,
            Ok(ref tensor)
                if tensor.memory_space() == LogicalMemorySpace::GpuMemory { device_id: 1 }
        ) || matches!(gpu_async_on, Err(tenferro::Error::Backend(_)))
    );
}

#[test]
fn to_memory_space_preserves_forward_mode_and_tangent() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    let dx = Tensor::from_slice(&[0.5_f64, -0.25], &[2]).unwrap();

    let (_, tangent) = forward_ad::dual_level(|fw| {
        let dual = fw.make_dual(&x, &dx)?;
        let moved = dual.to_memory_space(LogicalMemorySpace::MainMemory)?;
        assert_eq!(moved.mode(), AdMode::Forward);
        assert_eq!(moved.memory_space(), LogicalMemorySpace::MainMemory);
        fw.unpack_dual(&moved)
    })
    .unwrap();

    let tangent = tangent.unwrap();
    assert_eq!(tangent.mode(), AdMode::Primal);
    assert_eq!(tangent.dims(), &[2]);
    assert_eq!(
        tangent
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.5, -0.25]
    );
}

#[test]
fn forward_mode_device_preference_updates_primal_and_tangent() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
    let dx = Tensor::from_slice(&[0.5_f64, -0.25], &[2]).unwrap();

    let (_, tangent) = forward_ad::dual_level(|fw| {
        let mut dual = fw.make_dual(&x, &dx)?;
        dual.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));
        assert_eq!(
            dual.preferred_compute_device(),
            Some(ComputeDevice::Cpu { device_id: 0 })
        );
        fw.unpack_dual(&dual)
    })
    .unwrap();

    let tangent = tangent.unwrap();
    assert_eq!(
        tangent.preferred_compute_device(),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );
}

#[test]
fn to_memory_space_preserves_reverse_mode_and_grad_flow() {
    let _guard = runtime::with_runtime(RuntimeContext::Cpu(CpuContext::new(1)), || {
        let mut x = Tensor::from_slice(&[2.0_f64], &[])?;
        x.set_requires_grad(true)?;
        x.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 0 }));

        let moved = x.to_memory_space(LogicalMemorySpace::MainMemory)?;
        assert_eq!(moved.mode(), AdMode::Reverse);
        assert!(moved.requires_grad());
        assert_eq!(
            moved.preferred_compute_device(),
            Some(ComputeDevice::Cpu { device_id: 0 })
        );

        let out = moved.exp()?;
        backward(&[&out], None, &[&moved], BackwardOptions::default())?;
        assert!(moved.grad()?.is_some());
        Ok::<(), tenferro::Error>(())
    })
    .unwrap();
}

#[test]
fn to_cpu_surface_and_readiness_are_available_on_tensor() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3]).unwrap();

    let sync = x.to_cpu().unwrap();
    let async_copy = x.to_cpu_async().unwrap();

    assert_eq!(sync.memory_space(), LogicalMemorySpace::MainMemory);
    assert_eq!(async_copy.memory_space(), LogicalMemorySpace::MainMemory);
    sync.wait();
    async_copy.wait();
    assert!(sync.is_ready());
    assert!(async_copy.is_ready());
}
