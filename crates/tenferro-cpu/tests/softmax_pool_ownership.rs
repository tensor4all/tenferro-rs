//! CPU softmax and fused activations fan out on the engine's pool, not on
//! Rayon's global one (#2273).
//!
//! Rayon's global pool is process-wide, so "the global pool was never started"
//! is only observable in a binary that does nothing else; `autotests = false`
//! keeps every test file of this crate in its own process.
//!
//! A serial engine must stay on the calling thread and a pooled engine must use
//! only its own pool, per the caller-held-engine contract of #2206.

use tenferro_cpu::{CpuEngine, Pool};
use tenferro_tensor::backend::ActivationOp;
use tenferro_tensor::{Engine, Tensor, TensorElementwise, TensorRead};

/// A `[4096, 1024]` column-major input, large enough to fan out over workers.
fn input() -> Tensor {
    let n = 4096 * 1024;
    Tensor::from_vec_col_major(vec![4096, n / 4096], vec![0.5f64; n]).unwrap()
}

/// Run one operation on `engine`; every one of them must keep its fast path.
fn run(engine: &mut CpuEngine, op: &str) {
    let x = input();
    if op == "control" {
        // A BLAS1 read that already fanned out correctly; the baseline.
        engine
            .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
            .unwrap();
        return;
    }
    let output = match op {
        "softmax" => engine.softmax_read(TensorRead::from_tensor(&x), 0),
        _ => engine.fused_activation_read(ActivationOp::Sigmoid, TensorRead::from_tensor(&x)),
    }
    .unwrap();
    assert!(output.is_some(), "{op} took its fused path on f64");
}

#[test]
fn fast_paths_never_enter_the_global_rayon_pool() {
    let mut serial = CpuEngine::serial();
    let mut pooled = CpuEngine::new(Pool::builder().threads(4).build().expect("test pool"));
    for op in ["control", "softmax", "activation"] {
        run(&mut serial, op);
        run(&mut pooled, op);
    }
    // The ledger reports whether anything started Rayon's global pool, which
    // no engine-held operation may do (`--features ledger`, in this binary alone).
    #[cfg(feature = "ledger")]
    assert!(
        threadteam::ledger::global_pool_unstarted(),
        "a CPU fast path entered Rayon's global pool"
    );
}

/// The tasks running at once under one operation never exceed the engine's
/// budget, for the fused paths that fan out.
#[cfg(feature = "ledger")]
#[test]
fn fast_paths_stay_within_the_engine_budget() {
    let mut pooled = CpuEngine::new(Pool::builder().threads(3).build().expect("test pool"));
    for op in ["control", "softmax", "activation"] {
        run(&mut pooled, op);
        let roots = pooled.take_ledger_roots();
        assert!(!roots.is_empty(), "{op} minted a token");
        for root in roots {
            assert!(
                threadteam::ledger::max_live(root) <= 3,
                "{op} ran more than 3 tasks at once"
            );
        }
    }
}

/// A serial engine and a pooled engine compute the same normalized columns.
#[test]
fn serial_and_pooled_engines_agree() {
    let x = input();
    let mut serial = CpuEngine::serial();
    let mut pooled = CpuEngine::new(Pool::builder().threads(4).build().expect("test pool"));
    let a = serial
        .softmax_read(TensorRead::from_tensor(&x), 0)
        .unwrap()
        .unwrap();
    let b = pooled
        .softmax_read(TensorRead::from_tensor(&x), 0)
        .unwrap()
        .unwrap();
    assert_eq!(a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap());
    for column in a.as_slice::<f64>().unwrap().chunks(4096).take(4) {
        let sum: f64 = column.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "column sums to {sum}");
    }
}
