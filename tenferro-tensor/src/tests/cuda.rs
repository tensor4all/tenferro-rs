use super::*;
use num_complex::Complex64;
use tenferro_device::ComputeDevice;

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
    )
    .unwrap();

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

#[test]
fn gpu_contiguous_matches_cpu_for_strided_views_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let data: Vec<f32> = (1..=24).map(|value| value as f32).collect();
    let base = Tensor::<f32>::from_slice(&data, &[2, 3, 4], MemoryOrder::ColumnMajor).unwrap();
    let view = base.permute(&[2, 0, 1]).unwrap();
    let gpu = view
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();

    let got = gpu
        .contiguous(MemoryOrder::ColumnMajor)
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();
    let expected = view.contiguous(MemoryOrder::ColumnMajor);

    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn gpu_logical_materialize_contiguous_resolves_lazy_conjugation_without_cuda_context() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let mut payload = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .conj();
    payload.set_preferred_compute_device(Some(ComputeDevice::Cpu { device_id: 7 }));
    let expected = Tensor::from_slice(
        &[
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let gpu_payload = payload
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = gpu_payload.materialize_logical_contiguous(MemoryOrder::ColumnMajor);
    assert_eq!(
        got.logical_memory_space(),
        LogicalMemorySpace::GpuMemory { device_id: 0 }
    );
    assert!(!got.is_conjugated());
    assert_eq!(got.preferred_compute_device(), None);
    assert!(got.is_col_major_contiguous());
    let got = got
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(got.dims(), expected.dims());
    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn gpu_zero_trailing_by_counts_matches_cpu_for_real_payload_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let payload = Tensor::from_slice(
        &[1.0_f64, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let keep_counts = Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let expected = payload.zero_trailing_by_counts(&keep_counts, 1, 2).unwrap();

    let gpu_payload = payload
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let gpu_keep_counts = keep_counts
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = gpu_payload
        .zero_trailing_by_counts(&gpu_keep_counts, 1, 2)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn gpu_zero_trailing_by_counts_matches_cpu_for_complex_payload_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let payload = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 2.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(4.0, 4.0),
            Complex64::new(5.0, 5.0),
            Complex64::new(6.0, 6.0),
            Complex64::new(7.0, 7.0),
            Complex64::new(8.0, 8.0),
            Complex64::new(9.0, 9.0),
            Complex64::new(10.0, 10.0),
            Complex64::new(11.0, 11.0),
            Complex64::new(12.0, 12.0),
        ],
        &[3, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let keep_counts = Tensor::from_slice(&[2.0_f32, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let expected = payload.zero_trailing_by_counts(&keep_counts, 0, 2).unwrap();

    let gpu_payload = payload
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let gpu_keep_counts = keep_counts
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = gpu_payload
        .zero_trailing_by_counts(&gpu_keep_counts, 0, 2)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn gpu_merge_strict_lower_and_upper_matches_cpu_for_real_payload_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let lower = Tensor::from_slice(
        &[10.0_f64, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let upper =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let expected = Tensor::merge_strict_lower_and_upper(&lower, &upper).unwrap();

    let gpu_lower = lower
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let gpu_upper = upper
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = Tensor::merge_strict_lower_and_upper(&gpu_lower, &gpu_upper)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn gpu_merge_strict_lower_and_upper_matches_cpu_for_complex_payload_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let lower = Tensor::from_slice(
        &[
            Complex64::new(10.0, 1.0),
            Complex64::new(20.0, 2.0),
            Complex64::new(30.0, 3.0),
            Complex64::new(40.0, 4.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let upper = Tensor::from_slice(
        &[
            Complex64::new(1.0, -1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, -3.0),
            Complex64::new(4.0, -4.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let expected = Tensor::merge_strict_lower_and_upper(&lower, &upper).unwrap();

    let gpu_lower = lower
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let gpu_upper = upper
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = Tensor::merge_strict_lower_and_upper(&gpu_lower, &gpu_upper)
        .unwrap()
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn gpu_tril_matches_cpu_for_batched_view_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let data: Vec<f64> = (1..=24).map(|value| value as f64).collect();
    let base = Tensor::<f64>::from_slice(&data, &[2, 3, 4], MemoryOrder::ColumnMajor).unwrap();
    let view = base.permute(&[1, 0, 2]).unwrap();
    let expected = view.tril(-1);

    let gpu = view
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = gpu
        .tril(-1)
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(got.dims(), expected.dims());
    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}

#[test]
fn gpu_triu_matches_cpu_for_complex_batched_view_when_cuda_is_available() {
    if !cuda_device_zero_is_available() {
        return;
    }

    let data: Vec<Complex64> = (1..=18)
        .map(|value| Complex64::new(value as f64, -(value as f64)))
        .collect();
    let base =
        Tensor::<Complex64>::from_slice(&data, &[3, 3, 2], MemoryOrder::ColumnMajor).unwrap();
    let view = base.permute(&[1, 0, 2]).unwrap();
    let expected = view.triu(1);

    let gpu = view
        .to_memory_space_async(LogicalMemorySpace::GpuMemory { device_id: 0 })
        .unwrap();
    let got = gpu
        .triu(1)
        .to_memory_space_async(LogicalMemorySpace::MainMemory)
        .unwrap();

    assert_eq!(got.dims(), expected.dims());
    assert_eq!(got.buffer().as_slice(), expected.buffer().as_slice());
}
