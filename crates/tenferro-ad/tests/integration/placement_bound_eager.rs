use std::error::Error as StdError;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use tenferro_ad::{CpuPlacementBoundEager, EagerRuntime, EagerTensor};
use tenferro_cpu::{
    discover_cpu_topology, CpuBackend, CpuDomainExecutor, CpuDomainExecutorCapabilities,
    CpuDomainExecutorError, CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown,
    CpuInnerParallelism, CpuPlacement, CpuPlacementError, CpuPlacementGuarantee, ExternalCpuDomain,
    NumaNodeId, ResolvedCpuPlacement, ScopedCpuJob, ScopedCpuJobs,
};
use tenferro_runtime::{Error as RuntimeError, ErrorPhase, GraphCompiler, Runtime, TracedTensor};
use tenferro_tensor::{CpuDomainId, ErrorKind, Tensor, TensorElementwise};

#[derive(Debug, Default)]
struct ExecutorCounters {
    installs: AtomicUsize,
    submits: AtomicUsize,
    drops: AtomicUsize,
    fail_next: AtomicBool,
}

#[derive(Debug)]
struct CountingExecutor {
    counters: Arc<ExecutorCounters>,
}

impl Drop for CountingExecutor {
    fn drop(&mut self) {
        self.counters.drops.fetch_add(1, Ordering::Relaxed);
    }
}

impl CpuDomainExecutor for CountingExecutor {
    fn capabilities(&self) -> CpuDomainExecutorCapabilities {
        CpuDomainExecutorCapabilities {
            worker_count: NonZeroUsize::new(1).unwrap(),
            outer_parallelism: false,
            inner_parallelism: CpuInnerParallelism::None,
            reentrancy: CpuExecutorReentrancy::Rejected,
            affinity: CpuExecutorAffinity::CallerDeclaredUnverified,
            shutdown: CpuExecutorShutdown::CallerOwned,
        }
    }

    fn submit(&self, jobs: &dyn ScopedCpuJobs) -> Result<(), CpuDomainExecutorError> {
        self.counters.submits.fetch_add(1, Ordering::Relaxed);
        for index in 0..jobs.len() {
            jobs.run(index)?;
        }
        Ok(())
    }

    fn install(&self, job: &mut dyn ScopedCpuJob) -> Result<(), CpuDomainExecutorError> {
        self.counters.installs.fetch_add(1, Ordering::Relaxed);
        if self.counters.fail_next.swap(false, Ordering::AcqRel) {
            return Err(CpuDomainExecutorError::Admission {
                message: "placement-bound test rejection".to_owned(),
            });
        }
        job.run()
    }
}

fn placement() -> CpuPlacement {
    CpuPlacement::NumaNode(NumaNodeId::new(0))
}

fn external_backend(counters: Arc<ExecutorCounters>) -> CpuBackend {
    let allowed = discover_cpu_topology().unwrap().allowed_cpus().clone();
    let domain = ExternalCpuDomain::new(
        CpuDomainId::new(41),
        ResolvedCpuPlacement::NumaNode {
            id: NumaNodeId::new(0),
            cpus: allowed,
        },
        Arc::new(CountingExecutor { counters }),
        NonZeroUsize::new(1).unwrap(),
        CpuPlacementGuarantee::AdvisoryDeclared,
    )
    .unwrap();
    CpuBackend::from_external_managed_domains(CpuDomainId::new(41), [domain]).unwrap()
}

fn add_one(session: &mut dyn tenferro_tensor::BackendSession) -> tenferro_ad::Result<Tensor> {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    TensorElementwise::add(session, &lhs, &rhs).map_err(RuntimeError::from)
}

fn source_chain_contains<E: StdError + 'static>(error: &(dyn StdError + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(candidate) = current {
        if candidate.downcast_ref::<E>().is_some() {
            return true;
        }
        current = candidate.source();
    }
    false
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn placement_bound_type_is_send_sync_and_callback_may_borrow_stack_data() {
    assert_send_sync::<CpuPlacementBoundEager>();
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap());
    let mut cpu = runtime.on_cpu(CpuPlacement::Auto).unwrap();
    let label = String::from("borrowed callback result");
    let mut calls = 0;

    let borrowed = cpu
        .with_eager_session(|_| {
            calls += 1;
            Ok(label.as_str())
        })
        .unwrap();

    assert_eq!(borrowed, label);
    assert_eq!(calls, 1);
}

#[test]
fn placement_binding_is_idle_until_a_session_operation_runs() {
    let counters = Arc::new(ExecutorCounters::default());
    let backend = external_backend(Arc::clone(&counters));
    let probe = backend.clone();
    let runtime = EagerRuntime::with_cpu_backend(backend);
    let cpu = runtime.on_cpu(placement()).unwrap();
    assert_eq!(counters.installs.load(Ordering::Relaxed), 0);

    let (sent, received) = mpsc::channel();
    let worker = thread::spawn(move || {
        probe.install(|| {});
        sent.send(()).unwrap();
    });
    let completed = received.recv_timeout(Duration::from_secs(1));
    drop(cpu);
    worker.join().unwrap();

    assert!(completed.is_ok(), "idle placement view retained a permit");
    assert_eq!(counters.installs.load(Ordering::Relaxed), 1);
}

#[test]
fn fallible_external_session_enters_each_core_operation_exactly_once() {
    let counters = Arc::new(ExecutorCounters::default());
    let runtime = EagerRuntime::with_cpu_backend(external_backend(Arc::clone(&counters)));
    let mut cpu = runtime.on_cpu(placement()).unwrap();

    cpu.with_eager_session(|_| Ok(())).unwrap();
    assert_eq!(counters.installs.load(Ordering::Relaxed), 0);
    cpu.with_eager_session(add_one).unwrap();
    assert_eq!(counters.installs.load(Ordering::Relaxed), 1);
    cpu.with_eager_session(|session| {
        add_one(session)?;
        add_one(session)?;
        Ok(())
    })
    .unwrap();

    assert_eq!(counters.installs.load(Ordering::Relaxed), 3);
    assert_eq!(counters.submits.load(Ordering::Relaxed), 0);
}

#[test]
fn placement_and_executor_failures_retain_typed_sources() {
    let counters = Arc::new(ExecutorCounters::default());
    let runtime = EagerRuntime::with_cpu_backend(external_backend(Arc::clone(&counters)));
    let placement_error = runtime
        .on_cpu(CpuPlacement::NumaNode(NumaNodeId::new(99)))
        .unwrap_err();
    assert_eq!(placement_error.kind(), ErrorKind::RuntimeState);
    assert!(source_chain_contains::<CpuPlacementError>(&placement_error));

    let mut cpu = runtime.on_cpu(placement()).unwrap();
    counters.fail_next.store(true, Ordering::Release);
    let executor_error = cpu.with_eager_session(add_one).unwrap_err();
    assert!(source_chain_contains::<CpuDomainExecutorError>(
        &executor_error
    ));

    let recovered = cpu.with_eager_session(add_one).unwrap();
    assert_eq!(recovered.as_slice::<f64>().unwrap(), &[3.0]);
}

#[test]
fn callback_error_and_panic_release_the_session_for_reuse() {
    let counters = Arc::new(ExecutorCounters::default());
    let runtime = EagerRuntime::with_cpu_backend(external_backend(Arc::clone(&counters)));
    let mut cpu = runtime.on_cpu(placement()).unwrap();

    let error = cpu
        .with_eager_session::<()>(|_| {
            Err(RuntimeError::unsupported(
                "placement_bound_callback",
                ErrorPhase::Execution,
                "intentional callback error",
            ))
        })
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    cpu.with_eager_session(add_one).unwrap();

    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = cpu.with_eager_session::<()>(|_| panic!("intentional callback panic"));
    }));
    assert!(panicked.is_err());
    cpu.with_eager_session(add_one).unwrap();
}

#[test]
fn same_runtime_eager_reentry_panics_without_deadlock_and_then_recovers() {
    let counters = Arc::new(ExecutorCounters::default());
    let runtime = EagerRuntime::with_cpu_backend(external_backend(counters));
    let eager = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap(),
        Arc::clone(&runtime),
    )
    .unwrap();
    let mut cpu = runtime.on_cpu(placement()).unwrap();

    let panicked = catch_unwind(AssertUnwindSafe(|| {
        let _ = cpu.with_eager_session::<()>(|_| {
            let _ = eager.add(&eager).unwrap();
            Ok(())
        });
    }));
    let payload = panicked.expect_err("same-runtime eager re-entry must be rejected");
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or_default();
    assert!(message.contains("CpuBackend cannot be re-entered"));

    cpu.with_eager_session(add_one).unwrap();
}

#[test]
fn placement_session_matches_graph_execution_for_core_operations() {
    let counters = Arc::new(ExecutorCounters::default());
    let backend = external_backend(counters);
    let graph_backend = backend.for_placement(placement()).unwrap();
    let runtime = EagerRuntime::with_cpu_backend(backend);
    let mut cpu = runtime.on_cpu(placement()).unwrap();
    let session_output = cpu.with_eager_session(add_one).unwrap();

    let x = TracedTensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let y = TracedTensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    let sum = (&x + &y).unwrap();
    let program = GraphCompiler::new().compile(&sum).unwrap();
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&graph_backend).unwrap())
        .unwrap();
    let graph_runtime = builder.build().unwrap();
    let mut graph_outputs = graph_runtime.run_compiled(&program, &[]).unwrap();
    assert_eq!(graph_outputs.len(), 1);
    let graph_output = graph_outputs.pop().unwrap();

    assert_eq!(session_output.as_slice::<f64>().unwrap(), &[3.0]);
    assert_eq!(
        session_output.as_slice::<f64>().unwrap(),
        graph_output.as_slice::<f64>().unwrap()
    );
}
