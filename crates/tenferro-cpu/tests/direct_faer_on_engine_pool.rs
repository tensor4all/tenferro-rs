//! Direct faer work on tenferro-owned storage takes column-major slices and
//! runs on the engine's own pool, so tenferro hands out no faer value and a
//! downstream crate cannot meet a build mismatch at a tenferro boundary (#1939).
#![cfg(feature = "native")]

use tenferro_cpu::faer::{linalg::matmul::matmul, Accum, Mat, MatRef, Par};
use tenferro_tensor::TypedTensor;

fn engine(threads: usize) -> tenferro_cpu::CpuEngine {
    if threads <= 1 {
        tenferro_cpu::CpuEngine::serial()
    } else {
        tenferro_cpu::CpuEngine::new(
            tenferro_cpu::Pool::builder()
                .threads(threads)
                .build()
                .expect("test pool"),
        )
    }
}

#[test]
fn direct_faer_on_tenferro_storage_runs_on_the_engine_pool() -> tenferro_tensor::Result<()> {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])?;
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    let mut product = Mat::<f64>::zeros(2, 2);

    let mut backend = engine(2);
    assert_eq!(backend.num_threads(), 2);

    // The engine's token hands its budget to faer as a plain width and enters
    // the engine's pool once: faer's `Par::rayon` then schedules on the pool of
    // the calling thread. The direct call borrows tenferro storage as plain
    // column-major slices. Everything is spelled with `tenferro_cpu::faer`.
    let slice = a.as_slice()?;
    backend.exec()?.into_token().external(|width| {
        matmul(
            product.as_mut(),
            Accum::Replace,
            MatRef::from_column_major_slice(&identity, 2, 2),
            MatRef::from_column_major_slice(slice, 2, 2),
            1.0,
            if width.get() == 1 {
                Par::Seq
            } else {
                Par::rayon(width.get())
            },
        );
    });

    let product = product.as_ref();
    assert_eq!(*product.get(0, 0), 1.0);
    assert_eq!(*product.get(1, 0), 3.0);
    assert_eq!(*product.get(0, 1), 2.0);
    assert_eq!(*product.get(1, 1), 4.0);
    Ok(())
}

#[test]
fn a_serial_engine_has_no_pool_and_direct_faer_stays_sequential() -> tenferro_tensor::Result<()> {
    let mut backend = engine(1);
    assert_eq!(backend.num_threads(), 1);
    // A serial engine's token hands faer a width of one and never enters a pool.
    assert_eq!(
        backend.exec()?.into_token().external(|width| width.get()),
        1
    );

    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])?;
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    let mut product = Mat::<f64>::zeros(2, 2);
    matmul(
        product.as_mut(),
        Accum::Replace,
        MatRef::from_column_major_slice(&identity, 2, 2),
        MatRef::from_column_major_slice(a.as_slice()?, 2, 2),
        1.0,
        Par::Seq,
    );
    assert_eq!(*product.as_ref().get(1, 0), 3.0);
    Ok(())
}
