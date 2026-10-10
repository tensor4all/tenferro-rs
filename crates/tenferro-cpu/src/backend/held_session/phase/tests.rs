use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead};

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
    let worker_indices = Mutex::new(Vec::new());
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
                worker_indices
                    .lock()
                    .expect("worker index lock")
                    .push(rayon::current_thread_index());
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
    let worker_indices = worker_indices.into_inner().expect("worker indices");
    assert_eq!(worker_indices.len(), 2);
    for index in worker_indices {
        assert!(
            index.is_some(),
            "every multi-thread lane runs on a worker of the context pool"
        );
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
                assert!(
                    rayon::current_thread_index().is_none(),
                    "the one-worker lane runs on the caller, not on a pool worker"
                );
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

/// A lane panic is resumed only after every lane has finished, and it cancels its
/// peers while it unwinds.
#[test]
fn phase_joins_every_lane_before_a_lane_panic_surfaces() {
    let backend = backend(2);
    let x = vector(2.0);
    let barrier = Arc::new(Barrier::new(2));
    let survivor_saw_cancel = Arc::new(AtomicBool::new(false));
    let survivor_finished = Arc::new(AtomicBool::new(false));
    let mut session = backend.open_session().expect("held session");
    let saw_cancel = Arc::clone(&survivor_saw_cancel);
    let finished = Arc::clone(&survivor_finished);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = session.phase(
            |phase| -> Result<(), PhaseRunError<tenferro_tensor::Error>> {
                phase.run(|index, lane| -> Result<(), tenferro_tensor::Error> {
                    if index == 0 {
                        barrier.wait();
                        panic!("lane zero panicked");
                    }
                    barrier.wait();
                    // The panic arms the stop signal while its lane unwinds, before the
                    // child session is cleaned up.
                    let deadline = Instant::now() + Duration::from_secs(30);
                    while !lane.cancelled() {
                        assert!(
                            Instant::now() < deadline,
                            "the survivor never observed the panicking lane's cancellation"
                        );
                        thread::sleep(Duration::from_millis(1));
                    }
                    saw_cancel.store(true, Ordering::Relaxed);
                    // Real numerical work still runs through the survivor's child session.
                    let value = lane
                        .session()
                        .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))?;
                    assert_eq!(value.as_slice::<f64>()?, &[4.0]);
                    finished.store(true, Ordering::Relaxed);
                    Ok(())
                })
            },
        );
    }));
    assert!(panic.is_err(), "the lane panic surfaces on the driver");
    assert!(
        survivor_saw_cancel.load(Ordering::Relaxed),
        "the surviving lane observed the panic's cancellation"
    );
    assert!(
        survivor_finished.load(Ordering::Relaxed),
        "the surviving lane finished before the panic surfaced"
    );

    // The session survives the caught unwind, and a later run is not left cancelled.
    let value = session
        .with_session(|view| {
            view.add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))
        })
        .expect("root operation after a caught lane panic");
    assert_eq!(value.as_slice::<f64>().expect("f64 payload"), &[4.0]);
    session.close().expect("affinity restores");
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

    let x = vector(2.0);
    let lane_count = Arc::new(AtomicUsize::new(0));
    let lanes = Arc::clone(&lane_count);
    let reported = Arc::clone(&outsider_reported);
    let mut session = backend.open_session().expect("held session");
    session
        .phase(
            |phase| -> Result<(), PhaseRunError<tenferro_tensor::Error>> {
                phase.run(|_index, lane| -> Result<(), tenferro_tensor::Error> {
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
                        // Inherited child numerical work succeeds while the unrelated
                        // owner is rejected.
                        let value = lane
                            .session()
                            .add_read(TensorRead::from_tensor(&x), TensorRead::from_tensor(&x))?;
                        assert_eq!(value.as_slice::<f64>()?, &[4.0]);
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

/// A failed run does not cancel the lease: a later run on the *same* phase does real
/// work.
#[test]
fn phase_repeats_on_the_same_lease_after_a_failure() {
    let backend = backend(1);
    let mut session = backend.open_session().expect("held session");
    let ran = AtomicUsize::new(0);
    let outcome: Result<(), PhaseRunError<&'static str>> = session
        .phase(|phase| -> Result<(), PhaseRunError<&'static str>> {
            let first = phase.run(|_index, _lane| Err("first run failed"));
            assert!(matches!(
                first,
                Err(PhaseRunError::Lane("first run failed"))
            ));
            ran.fetch_add(1, Ordering::Relaxed);
            phase.run(|_index, lane| -> Result<(), &'static str> {
                assert!(
                    !lane.cancelled(),
                    "a fresh run starts with a cleared stop signal"
                );
                ran.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        })
        .expect("phase starts");
    assert!(outcome.is_ok(), "the second run on the same lease succeeds");
    assert_eq!(ran.load(Ordering::Relaxed), 2);
    session.close().expect("affinity restores");
}

/// A session (entry/cleanup) failure replaces a lane error, whatever order the lanes
/// finish in.
#[test]
fn session_failures_take_precedence_over_lane_errors() {
    let session_error = SessionEntryError::Reentered {
        backend: "CpuBackend",
    };
    let reduced: Result<(), PhaseRunError<&'static str>> = reduce_outcomes([
        Some(PhaseRunError::Lane("lane error")),
        Some(PhaseRunError::Session(session_error)),
    ]);
    assert!(matches!(reduced, Err(PhaseRunError::Session(_))));

    let reduced: Result<(), PhaseRunError<&'static str>> =
        reduce_outcomes([Some(PhaseRunError::Lane("lane error")), None]);
    assert!(matches!(reduced, Err(PhaseRunError::Lane("lane error"))));

    let reduced: Result<(), PhaseRunError<&'static str>> = reduce_outcomes([None, None]);
    assert!(reduced.is_ok());
}

/// A worker of a *different* Rayon pool is admitted: only a worker of the pool the
/// phase would broadcast to is rejected.
#[test]
fn phase_runs_from_a_foreign_pool_worker() {
    let backend = Arc::new(backend(2));
    let foreign_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("foreign pool"),
    );
    let lane_count = Arc::new(AtomicUsize::new(0));
    let counts = Arc::clone(&lane_count);
    let outcome = foreign_pool.install(move || {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut session = loop {
            match backend.open_session() {
                Ok(session) => break session,
                Err(SessionEntryError::Contended { .. }) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("held session on a foreign worker: {error:?}"),
            }
        };
        let result = session.phase(
            |phase| -> Result<(), PhaseRunError<std::convert::Infallible>> {
                phase.run(|_index, _lane| {
                    counts.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })
            },
        );
        session.close().expect("affinity restores");
        result
    });
    outcome
        .expect("phase starts from a foreign pool worker")
        .expect("phase runs");
    assert_eq!(lane_count.load(Ordering::Relaxed), 2);
}

/// A phase lease is `!Send + !Sync`: its inline lane runs under the caller-affinity
/// guard and owner marker the root session installed on its opening thread.
///
/// The probe is the `assert_not_impl_any` ambiguity idiom from the `static_assertions`
/// crate.
#[test]
fn phase_lease_is_neither_send_nor_sync() {
    const _: fn() = || {
        trait AmbiguousIfImpl<A> {
            fn item() {}
        }
        struct InvalidSend;
        struct InvalidSync;
        impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfImpl<InvalidSend> for T {}
        impl<T: ?Sized + Sync> AmbiguousIfImpl<InvalidSync> for T {}
        let _ = <CpuPhase<'static> as AmbiguousIfImpl<_>>::item;
    };
}

/// More lanes than work items: every lane runs, the unclaimed ones find nothing, and a lane
/// whose work is much larger than its peers' still completes.
#[test]
fn phase_with_empty_and_skewed_lanes_completes() {
    let backend = backend(4);
    let work = Arc::new(Mutex::new(vec![0usize, 0, 0, 0]));
    let runs = Arc::new(Mutex::new(Vec::new()));
    let mut session = backend.open_session().expect("held session");

    session
        .phase(
            |phase| -> Result<(), PhaseRunError<std::convert::Infallible>> {
                assert_eq!(phase.lanes(), 4);
                phase.run(|index, _lane| {
                    runs.lock().expect("runs lock").push(index);
                    // Only lane 0 pulls work; the queue is empty for the rest.
                    if index == 0 {
                        for _ in 0..64 {
                            work.lock().expect("work lock")[0] += 1;
                        }
                    }
                    Ok(())
                })
            },
        )
        .expect("phase starts")
        .expect("phase runs");

    let mut runs = runs.lock().expect("runs lock").clone();
    runs.sort_unstable();
    assert_eq!(
        runs,
        vec![0, 1, 2, 3],
        "an empty queue still reaches every lane"
    );
    assert_eq!(work.lock().expect("work lock")[0], 64);
    session.close().expect("affinity restores");
}

/// Concurrent lane execution stays within the context's thread budget, observed inside the
/// lanes around their numerical work.
///
/// The bound is a lane-level one: how many *lower-library* threads a lane fans out to is the
/// lower libraries' own contract, which this crate bounds only through the token it passes.
#[test]
fn phase_bounds_concurrent_lane_execution_by_the_thread_budget() {
    let budget = 2;
    let backend = backend(budget);
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let workers = Arc::new(Mutex::new(Vec::new()));
    let lhs = vector(3.0);
    let rhs = vector(4.0);

    let mut session = backend.open_session().expect("held session");
    let (active_in_lane, peak_in_lane, workers_in_lane) =
        (Arc::clone(&active), Arc::clone(&peak), Arc::clone(&workers));
    let lhs_in_lane = &lhs;
    let rhs_in_lane = &rhs;
    session
        .phase(
            move |phase| -> Result<(), PhaseRunError<std::convert::Infallible>> {
                phase.run(move |_index, lane| {
                    let now = active_in_lane.fetch_add(1, Ordering::SeqCst) + 1;
                    peak_in_lane.fetch_max(now, Ordering::SeqCst);
                    workers_in_lane
                        .lock()
                        .expect("workers lock")
                        .push(rayon::current_thread_index());

                    let product = lane
                        .session()
                        .add_read(
                            TensorRead::from_tensor(lhs_in_lane),
                            TensorRead::from_tensor(rhs_in_lane),
                        )
                        .expect("lane numerical work");
                    assert_eq!(product.as_slice::<f64>().expect("f64 payload"), &[7.0]);

                    active_in_lane.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
            },
        )
        .expect("phase starts")
        .expect("phase runs");

    assert!(
        peak.load(Ordering::SeqCst) <= budget,
        "at most one numerical lane per budgeted worker"
    );
    let workers = workers.lock().expect("workers lock").clone();
    assert_eq!(workers.len(), budget);
    assert!(
        workers.iter().all(Option::is_some),
        "every lane ran on a worker of the context pool"
    );
    session.close().expect("affinity restores");
}

/// A lane runs the concrete route families, not only one primitive: contraction, reduction,
/// transpose and reshape all work inside a lane and agree with their single-threaded values.
#[test]
fn phase_lane_runs_concrete_route_families() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    let backend = backend(2);
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).expect("matrix");
    let ones = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).expect("matrix");
    let mut session = backend.open_session().expect("held session");

    let sums = Mutex::new(Vec::new());
    session
        .phase(
            |phase| -> Result<(), PhaseRunError<std::convert::Infallible>> {
                phase.run(|_index, lane| {
                    let product = lane
                        .session()
                        .dot_general_read(
                            TensorRead::from_tensor(&a),
                            TensorRead::from_tensor(&ones),
                            &config,
                        )
                        .expect("lane contraction");
                    assert_eq!(product.shape(), &[2, 3]);

                    let transposed = lane
                        .session()
                        .transpose_read(TensorRead::from_tensor(&product), &[1, 0])
                        .expect("lane transpose");
                    assert_eq!(transposed.shape(), &[3, 2]);

                    let reshaped = lane
                        .session()
                        .reshape_read(TensorRead::from_tensor(&transposed), &[6])
                        .expect("lane reshape");
                    assert_eq!(reshaped.shape(), &[6]);

                    let total = lane
                        .session()
                        .reduce_sum_read(TensorRead::from_tensor(&reshaped), &[0])
                        .expect("lane reduction");
                    sums.lock()
                        .expect("sums lock")
                        .push(total.as_slice::<f64>().expect("f64 payload")[0]);
                    Ok(())
                })
            },
        )
        .expect("phase starts")
        .expect("phase runs");

    // `A` is `[[1, 3], [2, 4]]` column-major, so `A * ones(2, 3)` is `[[4, 4, 4], [6, 6, 6]]`.
    let sums = sums.into_inner().expect("sums");
    assert_eq!(sums.len(), 2);
    for sum in sums {
        assert!((sum - 30.0).abs() < 1e-12, "got {sum}");
    }
    session.close().expect("affinity restores");
}
