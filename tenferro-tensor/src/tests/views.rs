use super::*;
use tenferro_device::{Error, LogicalMemorySpace};

const MEM: LogicalMemorySpace = LogicalMemorySpace::MainMemory;
const COL: MemoryOrder = MemoryOrder::ColumnMajor;

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

#[test]
fn unsqueeze_basic() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.dims(), &[1, 2, 3]);
    assert_eq!(u.ndim(), 3);
}

#[test]
fn unsqueeze_at_end() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let u = t.unsqueeze(2).unwrap();
    assert_eq!(u.dims(), &[2, 3, 1]);
}

#[test]
fn unsqueeze_negative_dim() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let u = t.unsqueeze(-1).unwrap();
    assert_eq!(u.dims(), &[2, 3, 1]);

    let u2 = t.unsqueeze(-3).unwrap();
    assert_eq!(u2.dims(), &[1, 2, 3]);
}

#[test]
fn unsqueeze_middle() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let u = t.unsqueeze(1).unwrap();
    assert_eq!(u.dims(), &[2, 1, 3]);
}

#[test]
fn unsqueeze_rank0() {
    let t = Tensor::<f64>::zeros(&[], MEM, COL);
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(u.dims(), &[1]);

    let u2 = t.unsqueeze(-1).unwrap();
    assert_eq!(u2.dims(), &[1]);
}

#[test]
fn unsqueeze_out_of_range() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let result = t.unsqueeze(3);
    assert!(result.is_err());
}

#[test]
fn unsqueeze_negative_out_of_range() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let result = t.unsqueeze(-4);
    assert!(result.is_err());
}

#[test]
fn unsqueeze_is_zero_copy() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let u = t.unsqueeze(0).unwrap();
    assert_eq!(t.buffer().as_ptr(), u.buffer().as_ptr());
}

#[test]
fn squeeze_all_unit_dims() {
    let t = Tensor::<f64>::zeros(&[1, 2, 1, 3, 1], MEM, COL);
    let s = t.squeeze().unwrap();
    assert_eq!(s.dims(), &[2, 3]);
}

#[test]
fn squeeze_no_unit_dims() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let s = t.squeeze().unwrap();
    assert_eq!(s.dims(), &[2, 3]);
}

#[test]
fn squeeze_all_dims_are_unit() {
    let t = Tensor::<f64>::zeros(&[1, 1, 1], MEM, COL);
    let s = t.squeeze().unwrap();
    assert_eq!(s.dims(), &[]);
    assert_eq!(s.ndim(), 0);
}

#[test]
fn squeeze_rank0() {
    let t = Tensor::<f64>::zeros(&[], MEM, COL);
    let s = t.squeeze().unwrap();
    assert_eq!(s.dims(), &[]);
}

#[test]
fn squeeze_is_zero_copy() {
    let t = Tensor::<f64>::zeros(&[1, 2, 1, 3], MEM, COL);
    let s = t.squeeze().unwrap();
    assert_eq!(t.buffer().as_ptr(), s.buffer().as_ptr());
}

#[test]
fn squeeze_dim_basic() {
    let t = Tensor::<f64>::zeros(&[2, 1, 3], MEM, COL);
    let s = t.squeeze_dim(1).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
}

#[test]
fn squeeze_dim_negative() {
    let t = Tensor::<f64>::zeros(&[2, 1, 3], MEM, COL);
    let s = t.squeeze_dim(-2).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
}

#[test]
fn squeeze_dim_first() {
    let t = Tensor::<f64>::zeros(&[1, 2, 3], MEM, COL);
    let s = t.squeeze_dim(0).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
}

#[test]
fn squeeze_dim_last() {
    let t = Tensor::<f64>::zeros(&[2, 3, 1], MEM, COL);
    let s = t.squeeze_dim(2).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
}

#[test]
fn squeeze_dim_not_size_1_error() {
    let t = Tensor::<f64>::zeros(&[2, 3, 4], MEM, COL);
    let result = t.squeeze_dim(1);
    assert!(result.is_err());
}

#[test]
fn squeeze_dim_out_of_range() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let result = t.squeeze_dim(2);
    assert!(result.is_err());
}

#[test]
fn squeeze_dim_negative_out_of_range() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let result = t.squeeze_dim(-3);
    assert!(result.is_err());
}

#[test]
fn squeeze_dim_rank0_error() {
    let t = Tensor::<f64>::zeros(&[], MEM, COL);
    let result = t.squeeze_dim(0);
    assert!(result.is_err());
}

#[test]
fn squeeze_dim_is_zero_copy() {
    let t = Tensor::<f64>::zeros(&[2, 1, 3], MEM, COL);
    let s = t.squeeze_dim(1).unwrap();
    assert_eq!(t.buffer().as_ptr(), s.buffer().as_ptr());
}

#[test]
fn unsqueeze_squeeze_roundtrip() {
    let t = Tensor::<f64>::zeros(&[2, 3], MEM, COL);
    let u = t.unsqueeze(1).unwrap();
    let s = u.squeeze_dim(1).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    assert_eq!(s.buffer().as_ptr(), t.buffer().as_ptr());
}
