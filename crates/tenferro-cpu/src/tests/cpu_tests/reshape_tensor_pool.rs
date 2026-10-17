//! `CpuExecutionContext::reshape_tensor` shares the session's reshape policy.
//!
//! The entered linalg seam (`tenferro-linalg`'s `solve_entered`) reshapes a
//! vector right-hand side and restores the solution shape through
//! `reshape_tensor`. This module pins that the helper keeps exact column-major
//! values across the preset dtypes and both thread counts, that an input above
//! the session's serial threshold takes the pooled copy, and that a
//! non-contiguous source is read in logical order rather than raw storage order.

use super::*;

/// Reshape through the entered-context seam the linalg caller uses.
fn reshape_via_context(
    backend: &mut CpuBackend,
    input: &Tensor,
    shape: &[usize],
) -> tenferro_tensor::Result<Tensor> {
    with_cpu_session(backend, |cpu| {
        cpu.with_linalg_pool(|context, buffers| context.reshape_tensor(buffers, input, shape))
    })
}

/// Whether a freshly entered context can run the pooled copy.
fn parallel_capable(threads: usize) -> bool {
    let mut backend = CpuBackend::with_threads(threads).unwrap();
    with_cpu_session(&mut backend, |cpu| {
        cpu.with_linalg_pool(|context, _| Ok(context.uses_inner_parallelism()))
    })
    .unwrap()
}

#[test]
fn a_single_thread_context_cannot_pool_and_a_four_thread_context_can() {
    // The threshold branch in `reshape_tensor` is only reachable when the
    // context owns an inner parallel region; pin that the test's thread counts
    // select the serial and pooled branches respectively.
    assert!(!parallel_capable(1));
    assert!(parallel_capable(4));
}

/// Build a compact owner, reshape it through the seam, and compare exactly.
fn assert_exact<T>(threads: usize, from_shape: &[usize], to_shape: &[usize], values: Vec<T>)
where
    T: tenferro_tensor::TensorScalar + PartialEq + core::fmt::Debug,
{
    let mut backend = CpuBackend::with_threads(threads).unwrap();
    let input = Tensor::from_typed::<T>(
        TypedTensor::from_vec_col_major(from_shape.to_vec(), values.clone()).unwrap(),
    );
    let out = reshape_via_context(&mut backend, &input, to_shape)
        .unwrap_or_else(|error| panic!("threads={threads}: {error}"));
    assert_eq!(out.shape(), to_shape, "threads={threads}");
    assert_eq!(out.dtype(), input.dtype(), "threads={threads}");
    assert_eq!(
        out.as_slice::<T>().unwrap(),
        values.as_slice(),
        "threads={threads}"
    );
}

#[test]
fn reshape_tensor_is_exact_for_every_preset_dtype_below_the_threshold() {
    // Small inputs keep the serial host copy; the values must not depend on the
    // dtype or the session's thread count.
    for threads in [1, 4] {
        assert_exact::<f32>(threads, &[2, 3], &[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_exact::<f64>(
            threads,
            &[2, 3],
            &[3, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert_exact::<i32>(threads, &[2, 3], &[6], vec![-1, 2, -3, 4, -5, 6]);
        assert_exact::<i64>(threads, &[2, 3], &[6], vec![-1, 2, -3, 4, -5, 6]);
        assert_exact::<bool>(
            threads,
            &[2, 3],
            &[6],
            vec![true, false, true, false, true, false],
        );
        assert_exact::<Complex32>(
            threads,
            &[2, 3],
            &[6],
            (0..6)
                .map(|i| Complex32::new(i as f32, -(i as f32)))
                .collect(),
        );
        assert_exact::<Complex64>(
            threads,
            &[2, 3],
            &[6],
            (0..6)
                .map(|i| Complex64::new(i as f64, -(i as f64)))
                .collect(),
        );
    }
}

#[test]
fn reshape_tensor_above_the_threshold_preserves_column_major_order() {
    // 256 x 256 is above the strided parallel threshold, so a four-thread
    // session materializes through the pool; a one-thread session cannot
    // parallelize and keeps the serial copy. Both must preserve order.
    const N: usize = 256;
    let values: Vec<f64> = (0..N * N).map(|i| i as f64).collect();
    for threads in [1, 4] {
        assert_exact::<f64>(threads, &[N, N], &[512, 128], values.clone());
    }
}

#[test]
fn reshape_read_reads_a_non_contiguous_view_in_logical_order() {
    // An owned `Tensor` is always compact: the group entry point refuses a
    // non-compact descriptor (`AllocationGroup::into_tensor` ->
    // `NonCompactDescriptor`), so a non-contiguous source can only reach the
    // shared pooled copy as a borrowed view. That is exactly the
    // `reshape_read_with_pool` -> `typed_reshape_view_with_pool` copy
    // `reshape_tensor` now routes its materializing case through, so this pins
    // the non-contiguous case on the same code path for both thread counts.
    const N: usize = 256;
    let storage: Vec<f64> = (0..N * N).map(|i| i as f64).collect();
    let expected: Vec<f64> = (0..N * N).map(|k| ((k / N) + N * (k % N)) as f64).collect();
    for threads in [1, 4] {
        let mut backend = CpuBackend::with_threads(threads).unwrap();
        let view = TypedTensorView::from_col_major(&[N, N], &storage)
            .unwrap()
            .transpose_view([1, 0])
            .unwrap();
        let out = backend
            .with_backend_session(|session| {
                session.reshape_read(TensorRead::from_view(TensorView::F64(view)), &[N * N])
            })
            .unwrap()
            .unwrap();
        assert_eq!(out.shape(), &[N * N], "threads={threads}");
        assert_eq!(
            out.as_slice::<f64>().unwrap(),
            expected.as_slice(),
            "threads={threads}"
        );
    }
}
