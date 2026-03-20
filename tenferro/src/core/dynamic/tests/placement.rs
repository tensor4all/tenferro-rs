use super::*;
use crate::{ComputeDevice, LogicalMemorySpace};

fn exercise_dynamic_placement_dispatch(tensor: &mut Tensor) {
    let memory_space: fn(&Tensor) -> LogicalMemorySpace = Tensor::memory_space;
    let preferred_compute_device: fn(&Tensor) -> Option<ComputeDevice> =
        Tensor::preferred_compute_device;
    let set_preferred_compute_device: fn(&mut Tensor, Option<ComputeDevice>) =
        Tensor::set_preferred_compute_device;
    let to_memory_space_async: fn(&Tensor, LogicalMemorySpace) -> crate::Result<Tensor> =
        Tensor::to_memory_space_async;

    assert_eq!(memory_space(tensor), LogicalMemorySpace::MainMemory);
    assert_eq!(preferred_compute_device(tensor), None);

    set_preferred_compute_device(tensor, Some(ComputeDevice::Cpu { device_id: 0 }));
    assert_eq!(
        preferred_compute_device(tensor),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );

    let moved = to_memory_space_async(tensor, LogicalMemorySpace::MainMemory).unwrap();
    assert_eq!(memory_space(&moved), LogicalMemorySpace::MainMemory);
    assert_eq!(
        preferred_compute_device(&moved),
        Some(ComputeDevice::Cpu { device_id: 0 })
    );
}

#[test]
fn dyn_ad_tensor_placement_dispatch_covers_all_runtime_variants() {
    let mut tensors = [
        Tensor::from_tensor(super::vector_f32(&[1.0_f32, 2.0])),
        Tensor::from_tensor(
            DenseTensor::<f64>::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor)
                .unwrap(),
        ),
        Tensor::from_tensor(super::vector_c32(&[Complex32::new(1.0, 0.5)])),
        Tensor::from_tensor(super::vector_c64(&[Complex64::new(1.0, -0.5)])),
    ];

    for tensor in &mut tensors {
        exercise_dynamic_placement_dispatch(tensor);
    }
}
