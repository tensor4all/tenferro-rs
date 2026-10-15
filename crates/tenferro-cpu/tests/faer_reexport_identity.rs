//! #1939: the faer a CPU session hands to `with_faer_parallelism` is the one this
//! crate re-exports, so downstream direct-call code cannot resolve a different
//! build and meet a type error at the callback boundary.
#![cfg(feature = "native")]

use tenferro_cpu::faer::{linalg::matmul::matmul, Accum, Mat, MatRef, Par};
use tenferro_cpu::{CpuBackend, FaerParallelismExt};
use tenferro_tensor::{BackendSessionHost, TypedTensor};

#[test]
fn the_reexported_faer_is_the_session_identity() -> tenferro_tensor::Result<()> {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])?;
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    let mut product = Mat::<f64>::zeros(2, 2);

    let mut backend = CpuBackend::with_threads(1)?;
    backend.with_backend_session(|session| {
        // The callback parameter is spelled with the re-export: this compiles
        // only while the re-export and the session's `faer::Par` are one type.
        session.with_faer_parallelism(|parallel: Par| {
            matmul(
                product.as_mut(),
                Accum::Replace,
                MatRef::from_column_major_slice(&identity, 2, 2),
                MatRef::from_column_major_slice(a.as_slice()?, 2, 2),
                1.0,
                parallel,
            );
            Ok(())
        })
    })??;

    let product = product.as_ref();
    assert_eq!(*product.get(0, 0), 1.0);
    assert_eq!(*product.get(1, 0), 3.0);
    assert_eq!(*product.get(0, 1), 2.0);
    assert_eq!(*product.get(1, 1), 4.0);
    Ok(())
}
