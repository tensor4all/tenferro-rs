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
    let (group, slots) = tenferro_tensor::AllocationGroup::from_tensors(vec![
        tenferro_tensor::Tensor::from_typed::<f64>(tensor),
    ])
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
    let output = PooledUninitOutput::<f64>::new(&pool, vec![4]).unwrap();
    drop(tensor);
    drop(output);
    // The recycled tensor and the abandoned checkout both return their storage, and neither
    // return needs the pool's execution lock while the other is held.
    assert_eq!(pool.len(), 2);
}

#[test]
fn output_and_scratch_checkouts_coexist_without_borrowing_the_pool() {
    let mut pool = BufferPool::new();
    let mut output = PooledUninitOutput::<f64>::new(&pool, vec![2]).unwrap();
    let mut scratch = PooledUninitOutput::<f64>::new(&pool, vec![2]).unwrap();
    output.as_uninit_slice_mut()[0].write(2.0);
    output.as_uninit_slice_mut()[1].write(3.0);
    scratch.as_uninit_slice_mut()[0].write(5.0);
    scratch.as_uninit_slice_mut()[1].write(7.0);
    // An active lease must not prevent using the pool for other resources.
    let zeroed = pool.acquire_zeroed::<f64>(1);
    assert_eq!(zeroed, vec![0.0]);
    // SAFETY: both elements of each lease were initialized above.
    let result = unsafe { output.assume_init_recycled() }.unwrap();
    let temporary = unsafe { scratch.assume_init_recycled() }.unwrap();
    assert_eq!(result.as_slice().unwrap(), &[2.0, 3.0]);
    assert_eq!(temporary.as_slice().unwrap(), &[5.0, 7.0]);
    drop(result);
    drop(temporary);
    assert_eq!(pool.len(), 2);
    assert!(pool.in_flight_is_empty());
}

#[test]
fn ranked_recycled_handoff_returns_once_and_rank_error_cancels_checkout() {
    use tenferro_tensor::Rank;

    let pool = BufferPool::new();
    let mut wrong = PooledUninitOutput::<f64>::new(&pool, vec![2]).unwrap();
    wrong.as_uninit_slice_mut()[0].write(1.0);
    wrong.as_uninit_slice_mut()[1].write(2.0);
    // SAFETY: both elements have been initialized, although the selected rank is wrong.
    assert!(unsafe { wrong.assume_init_as_recycled::<Rank<2>>() }.is_err());
    // A handoff that fails on its rank leaves the storage the pool's, so it comes straight back.
    assert_eq!(pool.len(), 1);
    assert!(pool.in_flight_is_empty());

    let mut output = PooledUninitOutput::<f64>::new(&pool, vec![2]).unwrap();
    output.as_uninit_slice_mut()[0].write(3.0);
    output.as_uninit_slice_mut()[1].write(4.0);
    // SAFETY: both elements have been initialized.
    let owner = unsafe { output.assume_init_as_recycled::<Rank<1>>() }.unwrap();
    assert_eq!(owner.as_slice().unwrap(), &[3.0, 4.0]);
    assert!(pool.is_empty());
    drop(owner);
    assert_eq!(pool.len(), 1);
    assert!(pool.in_flight_is_empty());
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
fn partial_output_returns_to_the_pool_on_error_and_unwind() {
    let mut pool = BufferPool::new();
    for unwind in [false, true] {
        drop(recycled(&mut pool));
        assert_eq!(pool.len(), 1);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut output = PooledUninitOutput::<f64>::new(&pool, vec![4]).unwrap();
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
        // An aborted destination never became a tensor, so its storage is still the pool's and
        // goes back for the next call rather than to the allocator.
        assert_eq!(
            pool.len(),
            1,
            "an aborted checkout returns its buffer to the pool"
        );
        assert_eq!(
            pool.retained_capacity_bytes(),
            4 * std::mem::size_of::<f64>()
        );
        assert!(pool.in_flight_is_empty());
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
    let pool = BufferPool::new();
    let mut output = PooledUninitOutput::<i32>::new(&pool, vec![0]).unwrap();
    assert!(output.as_uninit_slice_mut().is_empty());
    assert!(output.as_uninit_bytes_mut().is_empty());
    assert_eq!(output.as_uninit_view_mut().unwrap().dims(), &[0]);

    let tensor = unsafe { output.assume_init_recycled() }.unwrap();
    assert_eq!(tensor.shape(), &[0]);
    assert!(pool.is_empty());
}
