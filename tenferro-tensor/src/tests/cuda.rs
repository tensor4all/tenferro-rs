use super::*;

fn cuda_device_zero_is_available() -> bool {
    std::panic::catch_unwind(|| cudarc::driver::CudaContext::new(0).is_ok()).unwrap_or(false)
}

#[test]
fn gpu_zeros_allocates_device_buffer_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let tensor = Tensor::<f32>::zeros(
        &[2, 3],
        LogicalMemorySpace::GpuMemory { device_id: 0 },
        MemoryOrder::ColumnMajor,
    );

    assert_eq!(
        tensor.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert!(tensor.buffer().is_gpu());
    assert_eq!(
        tensor.buffer().gpu_memory_space(),
        Some(LogicalMemorySpace::GpuMemory { device_id: 0 })
    );
    assert_eq!(tensor.buffer().len(), 6);
    assert!(tensor.buffer().as_device_ptr().is_some());
    assert!(tensor.buffer().as_slice().is_none());
}

#[test]
fn gpu_round_trip_preserves_view_layout_and_values_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let data: Vec<f32> = (1..=24).map(|value| value as f32).collect();
    let base = Tensor::<f32>::from_slice(&data, &[2, 3, 4], MemoryOrder::ColumnMajor).unwrap();
    let view = base.permute(&[2, 0, 1]).unwrap();
    assert!(!view.is_contiguous());

    let gpu = view
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let round_trip = gpu
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(round_trip.dims(), view.dims());
    assert_eq!(round_trip.strides(), view.strides());
    assert_eq!(round_trip.offset(), view.offset());
    assert_eq!(round_trip.buffer().len(), view.buffer().len());
    assert_eq!(
        round_trip.logical_memory_space(),
        LogicalMemorySpace::MainMemory
    );
    assert_eq!(
        round_trip
            .contiguous(MemoryOrder::ColumnMajor)
            .buffer()
            .as_slice(),
        view.contiguous(MemoryOrder::ColumnMajor)
            .buffer()
            .as_slice(),
    );
}
