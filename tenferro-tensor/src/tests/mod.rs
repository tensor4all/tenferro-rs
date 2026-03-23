use super::*;
use tenferro_device::LogicalMemorySpace;

mod buffer;
mod combine;
mod completion_event;
mod constructor_fallibility;
mod constructors_phase2;

#[cfg(feature = "cuda")]
mod cuda;
mod organization;
mod structural;
mod views;

#[test]
fn tensor_debug_is_summary_style() {
    let tensor = Tensor::<f32>::zeros(
        &[2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let dbg = format!("{:?}", tensor);
    assert!(dbg.contains("Tensor"));
    assert!(dbg.contains("f32"));
    assert!(dbg.contains("[2, 3]"));
    assert!(dbg.contains("logical_memory_space"));
    assert!(dbg.contains("is_contiguous"));
}
