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

fn col_tensor(data: &[f64], dims: &[usize]) -> Tensor<f64> {
    Tensor::from_slice(data, dims, MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn unsqueeze_inserts_unit_axes_without_copying_data() {
    let base = col_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let base_ptr = base.buffer().as_slice().unwrap().as_ptr();

    let leading = base.unsqueeze(0).unwrap();
    assert_eq!(leading.dims(), &[1, 2, 3]);
    assert_eq!(leading.strides(), &[1, 1, 2]);
    assert_eq!(leading.buffer().as_slice().unwrap().as_ptr(), base_ptr);
    assert_eq!(
        leading
            .contiguous(MemoryOrder::ColumnMajor)
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let middle = base.unsqueeze(1).unwrap();
    assert_eq!(middle.dims(), &[2, 1, 3]);
    assert_eq!(middle.strides(), &[1, 2, 2]);
    assert_eq!(middle.buffer().as_slice().unwrap().as_ptr(), base_ptr);

    let trailing = base.unsqueeze(-1).unwrap();
    assert_eq!(trailing.dims(), &[2, 3, 1]);
    assert_eq!(trailing.strides(), &[1, 2, 2]);
    assert_eq!(trailing.buffer().as_slice().unwrap().as_ptr(), base_ptr);
}

#[test]
fn unsqueeze_rejects_out_of_range_dimensions() {
    let base = Tensor::<f64>::zeros(
        &[2, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );

    let pos_err = base.unsqueeze(3).unwrap_err();
    assert!(
        matches!(pos_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected out-of-range error, got {pos_err:?}"
    );

    let neg_err = base.unsqueeze(-4).unwrap_err();
    assert!(
        matches!(neg_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected out-of-range error, got {neg_err:?}"
    );
}

#[test]
fn unsqueeze_supports_scalar_inputs() {
    let scalar = col_tensor(&[7.0], &[]);
    let expanded = scalar.unsqueeze(0).unwrap();
    assert_eq!(expanded.dims(), &[1]);
    assert_eq!(expanded.strides(), &[1]);
    assert_eq!(
        expanded.buffer().as_slice().unwrap().as_ptr(),
        scalar.buffer().as_slice().unwrap().as_ptr()
    );
}

#[test]
fn squeeze_removes_all_unit_axes_and_can_collapse_to_scalar() {
    let with_units = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 1, 2, 1]);
    let squeezed = with_units.squeeze().unwrap();
    assert_eq!(squeezed.dims(), &[2, 2]);
    assert_eq!(squeezed.strides(), &[1, 2]);
    assert_eq!(
        squeezed.buffer().as_slice().unwrap().as_ptr(),
        with_units.buffer().as_slice().unwrap().as_ptr()
    );

    let scalar_like = col_tensor(&[7.0], &[1, 1, 1]);
    let scalar = scalar_like.squeeze().unwrap();
    assert_eq!(scalar.dims(), &[] as &[usize]);
    assert_eq!(scalar.strides(), &[] as &[isize]);
    assert_eq!(
        scalar.buffer().as_slice().unwrap().as_ptr(),
        scalar_like.buffer().as_slice().unwrap().as_ptr()
    );
}

#[test]
fn squeeze_dim_supports_negative_dims_and_rejects_invalid_cases() {
    let base = col_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let squeezed = base.squeeze_dim(-2).unwrap();
    assert_eq!(squeezed.dims(), &[2, 2]);
    assert_eq!(squeezed.strides(), &[1, 2]);
    assert_eq!(
        squeezed.buffer().as_slice().unwrap().as_ptr(),
        base.buffer().as_slice().unwrap().as_ptr()
    );

    let non_unit_err = base.squeeze_dim(0).unwrap_err();
    assert!(
        matches!(non_unit_err, Error::InvalidArgument(ref msg) if msg.contains("expected 1")),
        "expected non-unit-dimension error, got {non_unit_err:?}"
    );

    let range_err = base.squeeze_dim(3).unwrap_err();
    assert!(
        matches!(range_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected out-of-range error, got {range_err:?}"
    );

    let negative_range_err = base.squeeze_dim(-4).unwrap_err();
    assert!(
        matches!(negative_range_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected negative out-of-range error, got {negative_range_err:?}"
    );

    let scalar = col_tensor(&[1.0], &[]);
    let scalar_err = scalar.squeeze_dim(0).unwrap_err();
    assert!(
        matches!(scalar_err, Error::InvalidArgument(ref msg) if msg.contains("rank-0 tensor")),
        "expected rank-0 error, got {scalar_err:?}"
    );
}

#[test]
fn broadcast_rejects_target_rank_mismatch() {
    let base = Tensor::<f64>::ones(
        &[1, 3],
        LogicalMemorySpace::MainMemory,
        MemoryOrder::ColumnMajor,
    );

    let err = base.broadcast(&[4, 3, 1]).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(ref msg) if msg.contains("target dims length")),
        "expected target rank mismatch, got {err:?}"
    );
}

#[test]
fn diagonal_preserves_unpaired_axes_and_validates_pairs() {
    let base = col_tensor(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 2, 2],
    );
    let diagonal = base.diagonal(&[(1, 2)]).unwrap();
    assert_eq!(diagonal.dims(), &[3, 2]);
    assert_eq!(diagonal.strides(), &[1, 9]);
    assert_eq!(
        diagonal
            .contiguous(MemoryOrder::ColumnMajor)
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    );

    let pair_base = col_tensor(&[1.0; 8], &[2, 2, 2]);
    let reused_axis_err = pair_base.diagonal(&[(0, 1), (1, 2)]).unwrap_err();
    assert!(
        matches!(reused_axis_err, Error::InvalidArgument(ref msg) if msg.contains("used in multiple diagonal pairs")),
        "expected reused-axis error, got {reused_axis_err:?}"
    );

    let out_of_range_err = base.diagonal(&[(0, 3)]).unwrap_err();
    assert!(
        matches!(out_of_range_err, Error::InvalidArgument(ref msg) if msg.contains("axis out of range")),
        "expected out-of-range error, got {out_of_range_err:?}"
    );
}

#[test]
fn view_as_strided_validates_layout_bounds() {
    let base = col_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let err = base.view_as_strided(vec![2, 2], vec![10, 1]).unwrap_err();
    assert!(
        matches!(err, Error::StrideError(ref msg) if msg.contains("layout accesses buffer positions")),
        "expected layout bounds error, got {err:?}"
    );
}

#[test]
fn select_and_narrow_cover_success_and_bounds_checks() {
    let base = col_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let selected = base.select(1, 1).unwrap();
    assert_eq!(selected.dims(), &[2]);
    assert_eq!(selected.strides(), &[1]);
    assert_eq!(selected.offset(), 2);
    assert_eq!(
        selected
            .contiguous(MemoryOrder::ColumnMajor)
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.0, 4.0]
    );

    let select_dim_err = base.select(2, 0).unwrap_err();
    assert!(
        matches!(select_dim_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected select dim error, got {select_dim_err:?}"
    );

    let select_index_err = base.select(1, 3).unwrap_err();
    assert!(
        matches!(select_index_err, Error::InvalidArgument(ref msg) if msg.contains("index 3 out of range")),
        "expected select index error, got {select_index_err:?}"
    );

    let narrowed = base.narrow(1, 1, 2).unwrap();
    assert_eq!(narrowed.dims(), &[2, 2]);
    assert_eq!(narrowed.strides(), &[1, 2]);
    assert_eq!(narrowed.offset(), 2);
    assert_eq!(
        narrowed
            .contiguous(MemoryOrder::ColumnMajor)
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.0, 4.0, 5.0, 6.0]
    );

    let narrow_dim_err = base.narrow(2, 0, 1).unwrap_err();
    assert!(
        matches!(narrow_dim_err, Error::InvalidArgument(ref msg) if msg.contains("out of range")),
        "expected narrow dim error, got {narrow_dim_err:?}"
    );

    let narrow_range_err = base.narrow(1, 2, 2).unwrap_err();
    assert!(
        matches!(narrow_range_err, Error::InvalidArgument(ref msg) if msg.contains("range out of bounds")),
        "expected narrow range error, got {narrow_range_err:?}"
    );
}
