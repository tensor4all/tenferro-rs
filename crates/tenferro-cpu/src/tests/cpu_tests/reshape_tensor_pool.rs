//! `CpuExecutionContext::reshape_tensor` shares the session's reshape policy.
//!
//! The entered linalg seam (`tenferro-linalg`'s `solve_entered`) reshapes a
//! vector right-hand side and restores the solution shape through
//! `reshape_tensor`. This module pins that the helper keeps exact column-major
//! values across the preset dtypes above the session's serial threshold, that
//! the parallel branch really checks out and recycles a pool buffer while the
//! serial branch does not, and that a non-contiguous source is read in logical
//! order rather than raw storage order.

use super::*;

/// Reshape through the entered-context seam the linalg caller uses.
fn reshape_via_context(
    backend: &mut CpuEngine,
    input: &Tensor,
    shape: &[usize],
) -> tenferro_tensor::Result<Tensor> {
    with_cpu_session(backend, |cpu| {
        cpu.with_linalg_pool(|exec, buffers| crate::reshape_tensor(exec, buffers, input, shape))
    })
}

/// Build a compact owner, reshape it through the seam, and compare exactly.
fn assert_exact<T>(threads: usize, from_shape: &[usize], to_shape: &[usize], values: Vec<T>)
where
    T: tenferro_tensor::TensorScalar + PartialEq + core::fmt::Debug,
{
    let mut backend = crate::test_support::cpu_engine(threads).unwrap();
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

fn patterned<T>(len: usize, make: impl Fn(usize) -> T) -> Vec<T> {
    (0..len).map(make).collect()
}

#[test]
fn reshape_tensor_is_exact_for_every_preset_dtype_above_the_threshold() {
    // Every dtype must round-trip a patterned payload larger than the serial
    // threshold: the four-thread session takes the pooled copy and the
    // one-thread session keeps the serial copy, so both branches are checked
    // for exact column-major values.
    const N: usize = 256;
    let from = [N, N];
    let to = [2 * N, N / 2];
    let len = N * N;
    assert!(len > crate::structural::SERIAL_RESHAPE_MAX_ELEMS);
    for threads in [1, 4] {
        assert_exact::<f32>(threads, &from, &to, patterned(len, |i| i as f32));
        assert_exact::<f64>(threads, &from, &to, patterned(len, |i| i as f64));
        assert_exact::<i32>(threads, &from, &to, patterned(len, |i| i as i32 - 1000));
        assert_exact::<i64>(threads, &from, &to, patterned(len, |i| i as i64 - 1000));
        assert_exact::<bool>(threads, &from, &to, patterned(len, |i| i % 3 == 0));
        assert_exact::<Complex32>(
            threads,
            &from,
            &to,
            patterned(len, |i| Complex32::new(i as f32, -(i as f32) / 2.0)),
        );
        assert_exact::<Complex64>(
            threads,
            &from,
            &to,
            patterned(len, |i| Complex64::new(i as f64, -(i as f64) / 2.0)),
        );
    }
}

#[test]
fn reshape_tensor_recycles_a_pool_buffer_only_on_the_parallel_branch() {
    // The regression signal for the routing itself: an owned copy above the
    // threshold must check a buffer out of the caller's pool and hand it back
    // when the output drops on a parallel-capable context, and must not touch
    // the pool on a one-thread context. Reverting `reshape_tensor` to the
    // serial `to_vec` copy leaves the pool empty in both cases and fails here.
    const N: usize = 256;
    let values: Vec<f64> = (0..N * N).map(|i| i as f64).collect();
    let input =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![N, N], values).unwrap());
    for (threads, expect_checkout) in [(1usize, false), (4usize, true)] {
        let mut backend = crate::test_support::cpu_engine(threads).unwrap();
        assert_eq!(
            backend.buffer_pool_len(),
            0,
            "threads={threads}: a fresh engine pool starts empty"
        );
        let out = reshape_via_context(&mut backend, &input, &[512, 128]).unwrap();
        assert_eq!(out.as_slice::<f64>().unwrap().len(), N * N);
        drop(out);
        let retained = backend.buffer_pool_len();
        assert_eq!(
            retained == 1,
            expect_checkout,
            "threads={threads}: retained={retained}; the pooled branch returns one buffer and \
             the serial branch returns none"
        );
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
        let mut backend = crate::test_support::cpu_engine(threads).unwrap();
        let view = TypedTensorView::from_col_major(&[N, N], &storage)
            .unwrap()
            .transpose_view([1, 0])
            .unwrap();
        let out = crate::test_support::on_engine(&mut backend, |session| {
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
