//! `CpuEngine::with_lanes`: split validation, panic semantics and concurrent lanes (#2552 S2b).
//!
//! The factorizations that run inside a lane live in tenferro-linalg, so these tests use the
//! lane bridge (`with_borrowed_cpu_engine`) directly: a lane is an engine over its share of the
//! thread budget and its own pool, and nothing else.

use std::sync::mpsc;
use std::sync::Barrier;
use std::time::Duration;

use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::{with_borrowed_cpu_engine, CpuEngine};
use tenferro_tensor::{Engine, ErrorKind, Tensor, TensorElementwise, TensorRead};

fn team_engine(threads: usize) -> CpuEngine {
    let pool = tenferro_cpu::Pool::builder()
        .threads(threads)
        .acquire_timeout(Duration::from_secs(10))
        .build()
        .expect("pool");
    CpuEngine::new(pool)
}

/// Run `body` on a helper thread and fail instead of hanging if it deadlocks.
fn within_a_minute<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> R {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(body());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("no result within 60 s: deadlock or lost lane")
}

fn lane_budget(lane: &mut dyn Engine) -> usize {
    with_borrowed_cpu_engine(lane, |exec, _| exec.budget()).expect("a lane is a borrowed engine")
}

/// An engine still runs an ordinary operation: it was neither broken nor left locked.
fn assert_usable(engine: &mut CpuEngine) {
    let a = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).expect("tensor");
    let sum = engine
        .add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
        .expect("an add on the engine after lanes");
    assert_eq!(sum.shape(), &[2]);
    assert!(!engine.is_broken());
}

#[test]
fn invalid_splits_are_validation_errors_before_any_work_and_leave_the_engine_usable() {
    let mut engine = team_engine(4);
    let mut two = [BufferPool::new(), BufferPool::new()];
    let mut one = [BufferPool::new()];
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let count = |_: usize, _: &mut dyn Engine| {
        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    };
    // Sum above the budget, a zero share, and a length mismatch in both directions.
    for shares in [&[3_usize, 2][..], &[0, 1][..], &[1][..]] {
        let err = engine.with_lanes(shares, &mut two, count).unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::Validation(_)),
            "{shares:?}: {err}"
        );
        assert_usable(&mut engine);
    }
    let err = engine.with_lanes(&[1, 1], &mut one, count).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Validation(_)), "{err}");
    assert_usable(&mut engine);
    // Overflowing shares are a validation error too, not a wrap-around.
    let err = engine
        .with_lanes(&[usize::MAX, 2], &mut two, count)
        .unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Validation(_)), "{err}");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_usable(&mut engine);
    // A valid split still runs afterwards.
    let ids = engine
        .with_lanes(&[2, 2], &mut two, |i, _| i)
        .expect("lanes");
    assert_eq!(ids, vec![0, 1]);
}

#[test]
fn a_panic_in_a_lane_breaks_the_engine_like_a_panic_in_with_exec() {
    let mut host = team_engine(2);
    let mut pools = [BufferPool::new(), BufferPool::new()];
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = host.with_lanes(&[1, 1], &mut pools, |lane, _| {
            if lane == 1 {
                panic!("a failing lane");
            }
        });
    }));
    assert!(unwind.is_err());
    assert!(host.is_broken());
    assert!(host.with_lanes(&[1, 1], &mut pools, |_, _| ()).is_err());
}

/// Two lanes meet at a barrier: the call finishes only if they really run at the same time.
#[test]
fn lanes_run_concurrently_on_a_four_thread_team_for_every_split() {
    for shares in [[3_usize, 1], [2, 2]] {
        within_a_minute(move || {
            let mut host = team_engine(4);
            assert!(host.has_team(), "the engine must own a team for this test");
            let before = host.buffer_pool_stats();
            let mut pools = [BufferPool::new(), BufferPool::new()];
            let meeting = Barrier::new(2);
            let budgets = host
                .with_lanes(&shares, &mut pools, |lane, engine| {
                    let budget = lane_budget(engine);
                    meeting.wait();
                    (lane, budget)
                })
                .expect("lanes");
            assert_eq!(budgets, vec![(0, shares[0]), (1, shares[1])]);
            assert_eq!(
                host.buffer_pool_stats(),
                before,
                "the engine pool is untouched"
            );
            assert_usable(&mut host);
        });
    }
}

#[test]
fn a_lane_refuses_every_operation_but_the_factorizations_and_a_serial_engine_runs_inline() {
    let mut serial = CpuEngine::serial();
    let mut pools = [BufferPool::new()];
    let a = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).expect("tensor");
    let kinds = serial
        .with_lanes(&[1], &mut pools, |_, lane| {
            lane.add_read(TensorRead::from_tensor(&a), TensorRead::from_tensor(&a))
                .map(|_| ())
                .unwrap_err()
                .kind()
        })
        .expect("lanes");
    assert_eq!(kinds, vec![ErrorKind::Unsupported]);
    assert_usable(&mut serial);
}

#[test]
fn lane_results_outlive_the_pools_and_the_engine() {
    let mut host = team_engine(2);
    let mut pools = [BufferPool::new(), BufferPool::new()];
    let outs = host
        .with_lanes(&[1, 1], &mut pools, |lane, _| vec![lane; 3])
        .expect("lanes");
    drop(pools);
    drop(host);
    assert_eq!(outs, vec![vec![0; 3], vec![1; 3]]);
}
