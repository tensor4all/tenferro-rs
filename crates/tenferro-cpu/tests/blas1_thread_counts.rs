//! `vdot_read` and `norm_squared_read` against a scalar reference across
//! thread counts and input lengths.
//!
//! The existing in-crate BLAS1 coverage exercises one and two worker threads.
//! A default `CpuBackend` on a large host takes one worker per core, so a
//! reduction whose chunking is wrong only when the worker count rivals or
//! exceeds the element count is invisible to it. These cases sweep both.

use num_complex::Complex64;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::BackendSessionHost;
use tenferro_tensor::{BackendSession, Tensor, TensorRead, TensorView, TypedTensorView};

fn sample(len: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| Complex64::new(0.25 + 0.5 * i as f64, 1.0 - 0.125 * i as f64))
        .collect()
}

fn reference_vdot(lhs: &[Complex64], rhs: &[Complex64]) -> Complex64 {
    lhs.iter().zip(rhs).map(|(a, b)| a.conj() * b).sum()
}

fn reference_norm_squared(input: &[Complex64]) -> f64 {
    input.iter().map(|z| z.norm_sqr()).sum()
}

/// A flat view over an owned tensor, which is how a sweep hands a slice of a
/// larger buffer to BLAS1 without materializing it.
fn flat_view(tensor: &Tensor, len: usize) -> TensorRead<'_> {
    let Tensor::C64(typed) = tensor else {
        panic!("these cases are C64")
    };
    let view = TypedTensorView::from_slice(
        vec![len],
        vec![1_isize],
        0,
        typed.host_data().expect("host tensor"),
    )
    .expect("a flat view of the whole buffer is in bounds");
    TensorRead::from_view(TensorView::C64(view))
}

#[test]
fn vdot_and_norm_squared_hold_across_thread_counts_and_lengths() {
    // Lengths that straddle the worker count, which is where a chunking bug
    // lives: fewer elements than workers, exactly as many, and many more.
    for threads in [1_usize, 2, 4, 16, 64, 128] {
        let mut backend = CpuBackend::with_threads(threads).expect("backend builds");
        for len in [1_usize, 3, 16, 17, 64, 128, 129, 4096] {
            let lhs_host = sample(len);
            let rhs_host: Vec<Complex64> = sample(len).iter().map(|z| z.conj() * 0.5).collect();
            let lhs = Tensor::from_vec_col_major(vec![len], lhs_host.clone()).unwrap();
            let rhs = Tensor::from_vec_col_major(vec![len], rhs_host.clone()).unwrap();

            let expected_dot = reference_vdot(&lhs_host, &rhs_host);
            let expected_norm = reference_norm_squared(&lhs_host);
            let tolerance = 1e-9 * (1.0 + expected_norm);

            // Owned operands.
            let dot = backend
                .with_backend_session(|session| {
                    session.vdot_read(
                        TensorRead::from_tensor(&lhs),
                        TensorRead::from_tensor(&rhs),
                    )
                })
                .unwrap();
            let dot = dot.as_slice::<Complex64>().unwrap()[0];
            assert!(
                (dot - expected_dot).norm() <= tolerance,
                "vdot(owned) threads={threads} len={len}: {dot} vs {expected_dot}"
            );

            let norm = backend
                .with_backend_session(|session| {
                    session.norm_squared_read(TensorRead::from_tensor(&lhs))
                })
                .unwrap();
            let norm = norm.as_slice::<f64>().unwrap()[0];
            assert!(
                (norm - expected_norm).abs() <= tolerance,
                "norm_squared(owned) threads={threads} len={len}: {norm} vs {expected_norm}"
            );

            // The same operands as views, which is the path a backend adapter
            // takes when the buffer it holds is larger than the operand.
            let dot = backend
                .with_backend_session(|session| {
                    session.vdot_read(flat_view(&lhs, len), flat_view(&rhs, len))
                })
                .unwrap();
            let dot = dot.as_slice::<Complex64>().unwrap()[0];
            assert!(
                (dot - expected_dot).norm() <= tolerance,
                "vdot(view) threads={threads} len={len}: {dot} vs {expected_dot}"
            );

            let norm = backend
                .with_backend_session(|session| session.norm_squared_read(flat_view(&lhs, len)))
                .unwrap();
            let norm = norm.as_slice::<f64>().unwrap()[0];
            assert!(
                (norm - expected_norm).abs() <= tolerance,
                "norm_squared(view) threads={threads} len={len}: {norm} vs {expected_norm}"
            );
        }
    }
}
