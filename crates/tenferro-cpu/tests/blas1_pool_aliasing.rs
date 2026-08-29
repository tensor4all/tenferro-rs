//! A BLAS1 scalar result must not alias a buffer that a live tensor still owns.
//!
//! `vdot_read` and `norm_squared_read` allocate their rank-0 output from the
//! backend's buffer pool. Tensors returned by earlier session operations come
//! from the same pool. If the pool can hand the same storage to both, a BLAS1
//! call returns the correct scalar while silently overwriting data a caller is
//! still holding — which is invisible to any test that only checks the scalar.
//!
//! This is the shape a Krylov loop produces: contraction results are kept alive
//! as basis vectors across many interleaved inner products, each in its own
//! session.

use num_complex::Complex64;

use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::BackendSessionHost;
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead};

fn matrix(rows: usize, columns: usize, seed: f64) -> Tensor {
    let data: Vec<Complex64> = (0..rows * columns)
        .map(|i| Complex64::new(seed + 0.5 * i as f64, 1.0 - 0.125 * i as f64))
        .collect();
    Tensor::from_vec_col_major(vec![rows, columns], data).unwrap()
}

#[test]
fn blas1_scalars_do_not_overwrite_live_session_results() {
    let mut backend = CpuBackend::new();

    let lhs = matrix(8, 6, 0.25);
    let rhs = matrix(6, 4, -1.5);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    // Several live results from the pool, exactly as a sweep accumulates a
    // Krylov basis.
    let mut live: Vec<Tensor> = Vec::new();
    for _ in 0..4 {
        let product = backend
            .with_backend_session(|session| {
                session.dot_general_read(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                )
            })
            .unwrap();
        live.push(product);
    }

    let expected: Vec<Vec<Complex64>> = live
        .iter()
        .map(|tensor| tensor.as_slice::<Complex64>().unwrap().to_vec())
        .collect();

    // Interleave BLAS1 scalar reductions, each in its own session, while those
    // results stay alive.
    let operand = matrix(16, 1, 0.75);
    for _ in 0..8 {
        let _dot = backend
            .with_backend_session(|session| {
                session.vdot_read(
                    TensorRead::from_tensor(&operand),
                    TensorRead::from_tensor(&operand),
                )
            })
            .unwrap();
        let _norm = backend
            .with_backend_session(|session| {
                session.norm_squared_read(TensorRead::from_tensor(&operand))
            })
            .unwrap();
    }

    for (index, (tensor, expected)) in live.iter().zip(&expected).enumerate() {
        let actual = tensor.as_slice::<Complex64>().unwrap();
        assert_eq!(
            actual,
            expected.as_slice(),
            "contraction result {index} changed while only BLAS1 scalars ran"
        );
    }
}
