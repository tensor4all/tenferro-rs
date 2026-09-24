//! Issue #1884: the faer batch lanes must reproduce the serial batch.
//!
//! These cases drive the batched faer kernels on a four-thread backend, so the
//! lane branches in `faer_linalg/packed_lu.rs` run here even though the resident
//! oracle sweeps use one thread.

use super::*;
use tenferro_cpu::CpuBackendKind;

fn faer_tensor(shape: &[usize], data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data.to_vec()).unwrap()
}

/// Cyclic dominant matrices, so every batch member needs pivoting.
fn pivoting(n: usize, batch: usize, seed: u64) -> Vec<f64> {
    let mut values: Vec<f64> = (0..n * n * batch)
        .map(|index| {
            let value = (index as u64)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed);
            ((value % 4096) as f64 / 2048.0) - 1.0
        })
        .collect();
    for matrix in 0..batch {
        for col in 0..n {
            values[matrix * n * n + col + col * n] += n as f64 + 2.0;
        }
    }
    values
}

fn rhs(n: usize, nrhs: usize, batch: usize) -> Vec<f64> {
    (0..n * nrhs * batch)
        .map(|index| ((index % 7) as f64) - 3.0)
        .collect()
}

/// Numeric contents as `f64`, so factor, pivot, and solution tensors compare
/// with the same helper.
fn data(tensor: &Tensor) -> Vec<f64> {
    match tensor.dtype() {
        DType::F64 => tensor
            .as_typed::<f64>()
            .unwrap()
            .host_data()
            .unwrap()
            .to_vec(),
        DType::I32 => tensor
            .as_typed::<i32>()
            .unwrap()
            .host_data()
            .unwrap()
            .iter()
            .map(|&value| f64::from(value))
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// A four-thread batch must agree with a one-thread batch bit for bit.
///
/// Each lane factorizes one matrix at a time with `Par::Seq`, so the per-matrix
/// arithmetic is identical to the serial loop and the results must match
/// exactly, not approximately.
#[test]
fn batched_faer_lanes_reproduce_the_serial_batch() {
    let n = 8usize;
    let batch = 8usize;
    let a_data = pivoting(n, batch, 5);
    let b_data = rhs(n, 1, batch);
    let a = || faer_tensor(&[n, n, batch], &a_data);
    let b = || faer_tensor(&[n, batch], &b_data);

    let mut serial = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
    let mut parallel = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer).unwrap();

    let serial_factors = with_cpu_linalg(&mut serial, |s| s.lu_factor(&a())).unwrap();
    let parallel_factors = with_cpu_linalg(&mut parallel, |s| s.lu_factor(&a())).unwrap();
    assert_eq!(data(&parallel_factors[0]), data(&serial_factors[0]));
    assert_eq!(data(&parallel_factors[1]), data(&serial_factors[1]));

    let serial_x = with_cpu_linalg(&mut serial, |s| {
        s.lu_solve_prepared(
            &a(),
            &serial_factors[0],
            &serial_factors[1],
            &b(),
            false,
            false,
        )
    })
    .unwrap();
    let parallel_x = with_cpu_linalg(&mut parallel, |s| {
        s.lu_solve_prepared(
            &a(),
            &parallel_factors[0],
            &parallel_factors[1],
            &b(),
            false,
            false,
        )
    })
    .unwrap();
    assert_eq!(data(&parallel_x), data(&serial_x));

    let serial_fused = with_cpu_linalg(&mut serial, |s| s.lu_factor_solve(&a(), &b())).unwrap();
    let parallel_fused = with_cpu_linalg(&mut parallel, |s| s.lu_factor_solve(&a(), &b())).unwrap();
    for (parallel_part, serial_part) in parallel_fused.iter().zip(&serial_fused) {
        assert_eq!(data(parallel_part), data(serial_part));
    }

    // A zero-column RHS exercises the factor-only lane path.
    let empty = faer_tensor(&[n, 0, batch], &[]);
    let serial_only = with_cpu_linalg(&mut serial, |s| s.lu_factor_solve(&a(), &empty)).unwrap();
    let parallel_only =
        with_cpu_linalg(&mut parallel, |s| s.lu_factor_solve(&a(), &empty)).unwrap();
    for (parallel_part, serial_part) in parallel_only.iter().zip(&serial_only) {
        assert_eq!(data(parallel_part), data(serial_part));
    }
}

/// A singular batch member keeps reporting the same typed error with lanes.
#[test]
fn batched_faer_lanes_report_a_singular_member() {
    let n = 8usize;
    let batch = 8usize;
    for threads in [1usize, 4] {
        let mut a_data = pivoting(n, batch, 3);
        // Make the last matrix exactly singular.
        let offset = (batch - 1) * n * n;
        for index in 0..n * n {
            a_data[offset + index] = 0.0;
        }
        let a = faer_tensor(&[n, n, batch], &a_data);
        let b = faer_tensor(&[n, batch], &rhs(n, 1, batch));
        let mut backend = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Faer).unwrap();
        let err = with_cpu_linalg(&mut backend, |s| s.lu_factor_solve(&a, &b)).unwrap_err();
        assert!(
            err.to_string().contains("singular"),
            "{threads}T: expected a singular error, got {err}"
        );
    }
}
