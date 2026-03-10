pub(crate) mod semiring_backend;

use tenferro_device::{Error, LogicalMemorySpace};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::util::infer_memory_space;

#[test]
fn infer_memory_space_single_cpu() {
    let a = Tensor::<f64>::zeros(
        &[2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let space = infer_memory_space(&[&a]).unwrap();
    assert_eq!(space, LogicalMemorySpace::MainMemory);
}

#[test]
fn infer_memory_space_multiple_cpu() {
    let a = Tensor::<f64>::zeros(
        &[2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let b = Tensor::<f64>::zeros(
        &[3, 4],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let c = Tensor::<f64>::zeros(
        &[4, 5],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let space = infer_memory_space(&[&a, &b, &c]).unwrap();
    assert_eq!(space, LogicalMemorySpace::MainMemory);
}

#[test]
fn infer_memory_space_empty_operands_errors() {
    let operands: &[&Tensor<f64>] = &[];
    let result = infer_memory_space(operands);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got: {err:?}"
    );
}

#[test]
fn infer_memory_space_mixed_errors() {
    // We cannot construct GPU tensors in tests (assertion in Tensor::zeros
    // prevents GPU allocation), so we verify the logic by testing
    // the happy path (all CPU) and the error path (empty).
    // A true mixed-memory test requires GPU support which is not yet
    // available in the POC.
    //
    // This test documents the intended behaviour: calling einsum with
    // operands on different memory spaces returns
    // Error::CrossMemorySpaceOperation.
    let a = Tensor::<f64>::zeros(
        &[2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    // Verify that identical spaces produce Ok
    let space = infer_memory_space(&[&a, &a]).unwrap();
    assert_eq!(space, LogicalMemorySpace::MainMemory);
}
