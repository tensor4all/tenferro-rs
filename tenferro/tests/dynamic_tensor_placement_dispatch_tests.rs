use num_complex::{Complex32, Complex64};
use tenferro::{ComputeDevice, LogicalMemorySpace, Tensor};

fn exercise_dynamic_placement_dispatch(tensor: &mut Tensor) {
    let memory_space: fn(&Tensor) -> LogicalMemorySpace = Tensor::memory_space;
    let preferred_compute_device: fn(&Tensor) -> Option<ComputeDevice> =
        Tensor::preferred_compute_device;
    let set_preferred_compute_device: fn(&mut Tensor, Option<ComputeDevice>) =
        Tensor::set_preferred_compute_device;
    let to_memory_space_async: fn(&Tensor, LogicalMemorySpace) -> tenferro::Result<Tensor> =
        Tensor::to_memory_space_async;
    let wait: fn(&Tensor) = Tensor::wait;
    let is_ready: fn(&Tensor) -> bool = Tensor::is_ready;

    assert_eq!(memory_space(tensor), LogicalMemorySpace::MainMemory);
    assert_eq!(preferred_compute_device(tensor), None);

    set_preferred_compute_device(tensor, Some(ComputeDevice::Cpu { device_id: 0 }));
    assert_eq!(
        preferred_compute_device(tensor),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );

    let moved_async = to_memory_space_async(tensor, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(memory_space(&moved_async), LogicalMemorySpace::MainMemory);
    assert_eq!(
        preferred_compute_device(&moved_async),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );
    wait(&moved_async);
    assert!(is_ready(&moved_async));
}

#[test]
fn dynamic_tensor_placement_dispatch_covers_all_frontend_dtypes() {
    let mut tensors = [
        Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap(),
        Tensor::from_slice(&[1.0_f64, 2.0], &[2]).unwrap(),
        Tensor::from_slice(&[Complex32::new(1.0, 0.5)], &[1]).unwrap(),
        Tensor::from_slice(&[Complex64::new(1.0, -0.5)], &[1]).unwrap(),
    ];

    for tensor in &mut tensors {
        exercise_dynamic_placement_dispatch(tensor);
    }
}
