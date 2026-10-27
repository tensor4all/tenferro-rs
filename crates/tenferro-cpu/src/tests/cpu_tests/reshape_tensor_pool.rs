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
    // Every dtype must round-trip a patterned payload with exact column-major
    // values on a one-thread and a four-thread session. Which branch the
    // four-thread session takes is the token's call (`reshape_uses_pool`); the
    // payload is exact on either, and the pooled branch is pinned by size in
    // `reshape_tensor_recycles_a_pool_buffer_only_on_the_parallel_branch`.
    const N: usize = 256;
    let from = [N, N];
    let to = [2 * N, N / 2];
    let len = N * N;
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
    // The payload is sized so a four-thread token's width for the copy is above
    // one on every host class; the first assertion checks that premise.
    const R: usize = 2048;
    const C: usize = 1024;
    let values: Vec<f64> = (0..R * C).map(|i| i as f64).collect();
    let input =
        Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![R, C], values).unwrap());
    {
        let pool = threadteam::Pool::builder().threads(4).build().unwrap();
        let token = threadteam::Exec::new(&pool, 4).unwrap();
        assert!(
            crate::structural::reshape_uses_pool(&input, &token).unwrap(),
            "the payload must pay for a four-thread fork"
        );
    }
    for (threads, expect_checkout) in [(1usize, false), (4usize, true)] {
        let mut backend = crate::test_support::cpu_engine(threads).unwrap();
        assert_eq!(
            backend.buffer_pool_len(),
            0,
            "threads={threads}: a fresh engine pool starts empty"
        );
        let out = reshape_via_context(&mut backend, &input, &[C, R]).unwrap();
        assert_eq!(out.as_slice::<f64>().unwrap().len(), R * C);
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

#[cfg(feature = "ledger")]
#[test]
fn a_small_compact_reshape_on_a_multi_thread_engine_takes_no_engine_entry() {
    // The pre-entry gate keeps a copy that cannot pay for a fork off the engine
    // entry: no token is minted (the ledger records every minted token's root),
    // the output matches the serial copy, and no pool buffer is taken.
    let input = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![4, 6], (0..24).map(f64::from).collect()).unwrap(),
    );
    let mut backend = crate::test_support::cpu_engine(4).unwrap();
    let _ = backend.take_ledger_roots();
    let out = crate::test_support::on_engine(&mut backend, |session| {
        session.reshape_read(TensorRead::from_tensor(&input), &[6, 4])
    })
    .unwrap()
    .unwrap();
    assert_eq!(out.shape(), &[6, 4]);
    assert_eq!(
        out.as_slice::<f64>().unwrap(),
        input.as_slice::<f64>().unwrap()
    );
    assert!(
        backend.take_ledger_roots().is_empty(),
        "a small compact reshape must not mint a token"
    );
    drop(out);
    assert_eq!(backend.buffer_pool_len(), 0);

    // A payload that pays does enter the engine and mint a token.
    let big = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2048, 1024], vec![1.0; 2048 * 1024]).unwrap(),
    );
    let _ = crate::test_support::on_engine(&mut backend, |session| {
        session.reshape_read(TensorRead::from_tensor(&big), &[1024, 2048])
    })
    .unwrap()
    .unwrap();
    assert!(!backend.take_ledger_roots().is_empty());
}
