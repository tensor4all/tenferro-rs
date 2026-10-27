//! Direct faer work on tenferro-owned storage takes column-major slices and
//! runs on the engine's own team, so tenferro hands out no faer value and a
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
fn direct_faer_on_tenferro_storage_runs_on_the_engine_team() -> tenferro_tensor::Result<()> {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])?;
    let identity = [1.0_f64, 0.0, 0.0, 1.0];
    let mut product = Mat::<f64>::zeros(2, 2);

    let mut backend = engine(2);
    assert_eq!(backend.num_threads(), 2);

    // The engine's token carries its team: faer runs on it as `Par::Team`,
    // never on Rayon. The direct call borrows tenferro storage as plain
    // column-major slices. Everything is spelled with `tenferro_cpu::faer`.
    let slice = a.as_slice()?;
    let mut exec = backend.exec()?;
    let mut token = exec.token();
    assert!(token.carries_team());
    let mut multiply = |par: Par<'_>| {
        matmul(
            product.as_mut(),
            Accum::Replace,
            MatRef::from_column_major_slice(&identity, 2, 2),
            MatRef::from_column_major_slice(slice, 2, 2),
            1.0,
            par,
        );
    };
    token
        .team(|team| multiply(Par::Team(team)))
        .expect("a team token carries its team");

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
    // A serial engine's token carries no team and never enters a pool.
    assert!(!backend.exec()?.into_token().carries_team());

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
