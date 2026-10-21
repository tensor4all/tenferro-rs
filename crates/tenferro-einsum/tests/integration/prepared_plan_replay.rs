//! #2139 item 2, #2141 item 1: a repeated prepared `execute_into` replays the
//! engine's binary contraction plan instead of rebuilding it.
//!
//! The public observer is the engine's contraction plan cache. `misses` counts
//! plan builds and `hits` counts replays, so a first call followed by repeats
//! must show one build and one hit per repeat. The benchmark case
//! `einsum_prepared_repeat_f64_counter` reports the same delta as
//! `binary_plan_builds_per_call`.

use tenferro_cpu::CpuEngine;
use tenferro_einsum::ConcreteEinsumPlan;
use tenferro_tensor::{Tensor, TensorWrite};

fn matrix(n: usize, value: f64) -> Tensor {
    Tensor::from_vec_col_major(vec![n, n], vec![value; n * n]).expect("square matrix")
}

#[test]
fn einsum_prepared_repeat_f64_counter() -> Result<(), Box<dyn std::error::Error>> {
    let n = 32;
    let lhs = matrix(n, 1.0);
    let rhs = matrix(n, 2.0);
    let plan = ConcreteEinsumPlan::prepare([&lhs, &rhs], "ij,jk->ik")?;
    let mut out = matrix(n, 0.0);
    let mut engine = CpuEngine::serial();

    let before = engine.plan_cache().contraction_stats();
    plan.execute_into(
        [&lhs, &rhs],
        &mut engine,
        TensorWrite::from_tensor(&mut out),
    )?;
    let built = engine.plan_cache().contraction_stats();
    assert_eq!(
        built.misses,
        before.misses + 1,
        "the first execution must build exactly one binary contraction plan"
    );
    assert_eq!(
        built.hits, before.hits,
        "the first execution cannot replay a plan it just built"
    );

    for _ in 0..4 {
        plan.execute_into(
            [&lhs, &rhs],
            &mut engine,
            TensorWrite::from_tensor(&mut out),
        )?;
    }
    let replayed = engine.plan_cache().contraction_stats();
    assert_eq!(
        replayed.misses, built.misses,
        "a repeated execute_into must not rebuild the binary contraction plan"
    );
    assert_eq!(
        replayed.hits,
        built.hits + 4,
        "every repeat must replay the cached binary contraction plan"
    );
    assert_eq!(
        replayed.entries, built.entries,
        "replaying a plan must not grow the cache"
    );

    let expected = 2.0 * n as f64;
    for &value in out.as_slice::<f64>()? {
        assert_eq!(value, expected, "the replayed plan must produce the matmul");
    }
    Ok(())
}
