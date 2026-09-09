use super::{checked_compact_strides, PooledUninitOutput};
use crate::buffer_pool::BufferPool;

fn recycled(pool: &mut BufferPool) -> tenferro_tensor::TypedTensor<f64> {
    let mut output = PooledUninitOutput::<f64>::new(pool, vec![4]).unwrap();
    for (i, value) in output.as_uninit_slice_mut().iter_mut().enumerate() {
        value.write(i as f64);
    }
    // SAFETY: every logical element was written above.
    unsafe { output.assume_init_recycled() }.unwrap()
}

#[test]
fn recycled_output_returns_only_after_final_group_owner_drops() {
    let mut pool = BufferPool::new();
    let tensor = recycled(&mut pool);
    let pointer = tensor.as_slice().unwrap().as_ptr();
    let (group, slots) =
        tenferro_tensor::AllocationGroup::from_tensors(vec![tenferro_tensor::Tensor::F64(tensor)])
            .unwrap();
    assert!(pool.is_empty());
    {
        let reads = group.read_views(&slots).unwrap();
        assert_eq!(reads[0].shape(), &[4]);
        assert!(pool.is_empty());
    }
    drop(group);
    assert_eq!(pool.len(), 1);
    let tensor = recycled(&mut pool);
    assert_eq!(tensor.as_slice().unwrap().as_ptr(), pointer);
    pool.replenish_in_flight_retained();
    assert!(
        pool.is_empty(),
        "live recycled outputs must not allocate replacements"
    );
    drop(tensor);
    assert_eq!(pool.len(), 1);
}

#[test]
fn recycled_drop_during_another_checkout_does_not_reenter_the_execution_lock() {
    let mut pool = BufferPool::new();
    let tensor = recycled(&mut pool);
    let output = PooledUninitOutput::<f64>::new(&mut pool, vec![4]).unwrap();
    drop(tensor);
    drop(output);
    assert_eq!(pool.len(), 1);
}

#[test]
fn recycled_output_extraction_disarms_return_and_pool_may_drop_first() {
    let mut pool = BufferPool::new();
    let tensor = recycled(&mut pool);
    let pointer = tensor.as_slice().unwrap().as_ptr();
    let data = tensor.into_host_vec().unwrap();
    assert_eq!(data.as_ptr(), pointer);
    assert!(pool.is_empty());
    drop(data);
    assert!(pool.is_empty());
    let tensor = recycled(&mut pool);
    drop(pool);
    assert_eq!(tensor.as_slice().unwrap(), &[0., 1., 2., 3.]);
    drop(tensor);
}

#[test]
fn partial_output_is_discarded_on_error_and_unwind() {
    let mut pool = BufferPool::new();
    for unwind in [false, true] {
        drop(recycled(&mut pool));
        assert_eq!(pool.len(), 1);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut output = PooledUninitOutput::<f64>::new(&mut pool, vec![4]).unwrap();
            output.as_uninit_slice_mut()[0].write(f64::NAN);
            if unwind {
                panic!("kernel stopped after a partial write");
            }
            Err::<(), _>("kernel returned an error after a partial write")
        }));
        if unwind {
            assert!(outcome.is_err());
        } else {
            assert!(outcome.unwrap().is_err());
        }
        assert!(
            pool.is_empty(),
            "partial initialization cannot become a retained tensor"
        );
        pool.replenish_in_flight_retained();
        assert!(
            pool.is_empty(),
            "aborted checkout must cancel replacement accounting"
        );
        let tensor = recycled(&mut pool);
        assert_eq!(tensor.as_slice().unwrap(), &[0., 1., 2., 3.]);
        drop(tensor);
        pool.clear();
    }
}

#[test]
fn recycled_output_respects_current_limit_and_clear() {
    let mut pool = BufferPool::new();
    let tensor = recycled(&mut pool);
    pool.set_max_retained_capacity_bytes(0);
    drop(tensor);
    assert!(pool.is_empty());
    pool.set_max_retained_capacity_bytes(1024);
    drop(recycled(&mut pool));
    assert_eq!(pool.len(), 1);
    pool.clear();
    assert!(pool.is_empty());
    assert_eq!(pool.retained_capacity_bytes(), 0);
}

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
