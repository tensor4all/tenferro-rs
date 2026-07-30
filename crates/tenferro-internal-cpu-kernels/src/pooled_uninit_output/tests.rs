use super::{checked_compact_strides, PooledUninitOutput};
use crate::buffer_pool::BufferPool;

#[test]
fn compact_stride_validation_reports_dimension_and_product_overflow() {
    let dimension_error = checked_compact_strides(&[isize::MAX as usize + 1]).unwrap_err();
    assert!(dimension_error
        .to_string()
        .contains("dimension exceeds isize"));

    let stride_error = checked_compact_strides(&[isize::MAX as usize, 2]).unwrap_err();
    assert!(stride_error.to_string().contains("compact stride overflow"));
}

#[test]
fn pooled_uninit_output_public_contract_covers_zero_length_handoff() {
    let mut pool = BufferPool::new();
    let mut output = PooledUninitOutput::<i32>::new(&mut pool, vec![0]).unwrap();
    assert!(output.as_uninit_slice_mut().is_empty());
    assert!(output.as_uninit_bytes_mut().is_empty());
    assert_eq!(output.as_uninit_view_mut().unwrap().dims(), &[0]);

    let tensor = unsafe { output.assume_init() }.unwrap();
    assert_eq!(tensor.shape(), &[0]);
    assert!(pool.is_empty());
}
