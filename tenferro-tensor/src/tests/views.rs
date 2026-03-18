use super::*;
use tenferro_device::{Error, LogicalMemorySpace};

#[test]
fn diagonal_reports_stride_overflow_for_zero_extent_view() {
    let base = Tensor::<f64>::zeros(
        &[0, 0],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );
    let view = base
        .view_as_strided(vec![0, 0], vec![isize::MAX, 1])
        .unwrap();

    let err = view.diagonal(&[(0, 1)]).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref message) if message.contains("stride overflow")),
        "expected diagonal stride overflow error, got {err:?}"
    );
}
