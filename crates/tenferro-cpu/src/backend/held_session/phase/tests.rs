use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tenferro_tensor::{Tensor, TensorRead};

use super::*;
use crate::context::CpuContext;

fn backend(threads: usize) -> CpuBackend {
    CpuBackend::with_threads(threads).expect("CPU backend")
}

fn vector(value: f64) -> Tensor {
    Tensor::from_vec_col_major(vec![1], vec![value]).expect("vector")
}

/// Every lane runs, and the per-lane results agree with the single-threaded one.
#[test]
fn phase_runs_every_lane_and_matches_a_single_threaded_result() {
    let backend = backend(2);
    let x = vector(3.0);
    let lane_count = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    let session = backend.open_session().expect("held session");
    let mut session = session;
    session
        .phase(|phase| -> Result<(), Box<dyn std::error::Error>> {
            assert_eq!(phase.lanes(), 2);
            phase.run(|index, lane| -> Result<(), tenferro_tensor::Error> {
                lane_count.fetch_add(1, Ordering::Relaxed);
                let value = lane
                    .session()
                    .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))?;
                results
                    .lock()
                    .expect("results lock")
                    .push((index, value.as_slice::<f64>()?[0]));
                Ok(())
            })?;
            Ok(())
        })
        .expect("phase starts")
        .expect("phase runs");
    assert_eq!(lane_count.load(Ordering::Relaxed), 2);
    let mut results = results.into_inner().expect("results");
    results.sort_by_key(|(index, _)| *index);
    assert_eq!(results.len(), 2);
    for (_, value) in results {
        assert_eq!(value, 6.0);
    }
    session.close().expect("affinity restores");
}

/// A context without an inner execution pool drives one lane inline, on the
/// calling thread, without a second admission.
#[test]
fn phase_on_a_one_worker_context_runs_one_lane_on_the_caller() {
    let backend = backend(1);
    let x = vector(4.0);
    let caller = thread::current().id();
    let lane_thread = Mutex::new(None);
    let mut session = backend.open_session().expect("held session");
    session
        .phase(|phase| -> Result<(), Box<dyn std::error::Error>> {
            assert_eq!(phase.lanes(), 1);
            phase.run(|_index, lane| -> Result<(), tenferro_tensor::Error> {
                *lane_thread.lock().expect("lane thread lock") = Some(thread::current().id());
                let value = lane
                    .session()
                    .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))?;
                assert_eq!(value.as_slice::<f64>()?, &[8.0]);
                Ok(())
            })?;
            Ok(())
        })
        .expect("phase starts")
        .expect("phase runs");
    assert_eq!(
        *lane_thread.lock().expect("lane thread lock"),
        Some(caller),
        "a one-worker phase runs on the calling thread"
    );
    session.close().expect("affinity restores");
    assert_eq!(backend.buffer_pool_len().expect("resources returned"), 0);
}

/// A worker of the pool the phase would broadcast to can never drive the blocking
/// broadcast, so the phase is rejected before any work starts.
#[test]
fn phase_rejects_a_caller_inside_its_own_pool() {
    let context = Arc::new(CpuContext::with_threads(2).expect("two-worker context"));
    let backend = Arc::new(CpuBackend::from_context(Arc::clone(&context)));
    let outcome = context.install(move || {
        // The admission arbiter is process-global, so a concurrently running test can
        // hold the same CPU set; wait for a free window instead of assuming one.
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut session = loop {
            match backend.open_session() {
                Ok(session) => break session,
                Err(SessionEntryError::Contended { .. }) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("held session on a pool worker: {error:?}"),
            }
        };
        session
            .phase(|_| ())
            .expect_err("a target-pool worker must not drive a phase")
    });
    assert!(matches!(outcome, CpuPhaseError::TargetPoolCaller { .. }));
}

/// Every lane is joined before the error is returned, and the peers see the
/// cancellation the failing lane raised.
#[test]
fn phase_reports_one_lane_error_after_joining_and_cancels_peers() {
    let backend = backend(2);
    let barrier = Arc::new(Barrier::new(2));
    let peers_finished = Arc::new(AtomicUsize::new(0));
    let observed_cancel = Arc::new(AtomicBool::new(false));
    let mut session = backend.open_session().expect("held session");
    let outcome: Result<(), PhaseRunError<&'static str>> = session
        .phase(|phase| -> Result<(), PhaseRunError<&'static str>> {
            phase.run(|index, lane| -> Result<(), &'static str> {
                barrier.wait();
                if index == 0 {
                    return Err("lane zero failed");
                }
                let deadline = Instant::now() + Duration::from_secs(30);
                while !lane.cancelled() {
                    assert!(
                        Instant::now() < deadline,
                        "the peer lane never observed cancellation"
                    );
                    thread::sleep(Duration::from_millis(1));
                }
                observed_cancel.store(true, Ordering::Relaxed);
                peers_finished.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        })
        .expect("phase starts");
    assert!(matches!(
        outcome,
        Err(PhaseRunError::Lane("lane zero failed"))
    ));
    assert!(
        observed_cancel.load(Ordering::Relaxed),
        "the peer lane observed the cancellation"
    );
    assert_eq!(peers_finished.load(Ordering::Relaxed), 1);
    session.close().expect("affinity restores");
}

/// A lane panic is resumed only after every lane has finished, so the phase never
/// abandons a peer.
#[test]
fn phase_joins_every_lane_before_a_lane_panic_surfaces() {
    let backend = backend(2);
    let barrier = Arc::new(Barrier::new(2));
    let survivor_finished = Arc::new(AtomicBool::new(false));
    let mut session = backend.open_session().expect("held session");
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = session.phase(
            |phase| -> Result<(), PhaseRunError<std::convert::Infallible>> {
                phase.run(|index, _lane| -> Result<(), std::convert::Infallible> {
                    if index == 0 {
                        barrier.wait();
                        panic!("lane zero panicked");
                    }
                    barrier.wait();
                    thread::sleep(Duration::from_millis(200));
                    survivor_finished.store(true, Ordering::Relaxed);
                    Ok(())
                })
            },
        );
    }));
    assert!(panic.is_err(), "the lane panic surfaces on the driver");
    assert!(
        survivor_finished.load(Ordering::Relaxed),
        "the surviving lane finished before the panic surfaced"
    );
}

/// After a caught lane failure the session is unchanged: a root operation and a
/// fresh phase both work, and the phase is not left permanently cancelled.
#[test]
fn phase_leaves_the_session_usable_after_a_caught_lane_failure() {
    let backend = backend(1);
    let x = vector(5.0);
    let mut session = backend.open_session().expect("held session");
    let failed = session
        .phase(|phase| -> Result<(), PhaseRunError<&'static str>> {
            phase.run(|_index, _lane| -> Result<(), &'static str> { Err("lane failed") })
        })
        .expect("phase starts");
    assert!(matches!(failed, Err(PhaseRunError::Lane("lane failed"))));

    let value = session
        .with_session(|view| {
            view.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
        })
        .expect("root operation after a failed phase");
    assert_eq!(value.as_slice::<f64>().expect("f64 payload"), &[10.0]);

    let lane_count = AtomicUsize::new(0);
    session
        .phase(|phase| -> Result<(), PhaseRunError<&'static str>> {
            phase.run(|_index, _lane| -> Result<(), &'static str> {
                lane_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        })
        .expect("a later phase starts fresh")
        .expect("a later phase runs");
    assert_eq!(lane_count.load(Ordering::Relaxed), 1);
    session.close().expect("affinity restores");
}

/// An unrelated owner that cannot wait is rejected while the lanes are running, and
/// the lanes' inherited child work still succeeds.
#[test]
fn unrelated_owner_is_rejected_while_lanes_run() {
    let backend = Arc::new(backend(2));
    let (probe_tx, probe_rx) = mpsc::channel::<()>();
    let (done_tx, done_rx) = mpsc::channel::<&'static str>();
    let outsider_reported = Arc::new(AtomicBool::new(false));
    let outsider_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("outsider pool"),
    );
    let outsider_backend = Arc::clone(&backend);
    let outsider_reported_flag = Arc::clone(&outsider_reported);
    let outsider = std::thread::spawn(move || {
        // `install` runs the closure on a worker of this pool, which is what makes
        // the probe a non-waiting owner rather than an ordinary queued caller.
        let outcome = outsider_pool.install(move || {
            probe_rx.recv().expect("probe signal");
            let outcome = match outsider_backend.open_session() {
                Ok(_) => "admitted",
                Err(SessionEntryError::Contended { .. }) => "contended",
                Err(other) => panic!("unexpected entry error: {other}"),
            };
            outsider_reported_flag.store(true, Ordering::Relaxed);
            outcome
        });
        done_tx.send(outcome).expect("send probe result");
    });

    let lane_count = Arc::new(AtomicUsize::new(0));
    let lanes = Arc::clone(&lane_count);
    let reported = Arc::clone(&outsider_reported);
    let mut session = backend.open_session().expect("held session");
    session
        .phase(
            |phase| -> Result<(), PhaseRunError<std::convert::Infallible>> {
                phase.run(|_index, lane| -> Result<(), std::convert::Infallible> {
                    let started = lanes.fetch_add(1, Ordering::Relaxed) + 1;
                    if started == 2 {
                        probe_tx.send(()).expect("signal the probe");
                    }
                    // Stay inside the phase until the probe reported, so the entry
                    // attempt really overlaps running lanes.
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while !reported.load(Ordering::Relaxed) {
                        assert!(
                            Instant::now() < deadline,
                            "the probe never reported while the lanes were running"
                        );
                        assert!(!lane.cancelled(), "no lane failed, so none is cancelled");
                        thread::sleep(Duration::from_millis(1));
                    }
                    Ok(())
                })
            },
        )
        .expect("phase starts")
        .expect("phase runs");
    assert_eq!(lane_count.load(Ordering::Relaxed), 2);
    assert_eq!(
        done_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("probe result"),
        "contended",
        "an unrelated non-waiting owner is rejected while the phase runs"
    );
    outsider.join().expect("outsider thread");
    session.close().expect("affinity restores");
}
