use std::error::Error as StdError;
use std::num::NonZeroUsize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;

use tenferro_ad::{CpuPlacementBoundEager, EagerBackend, EagerRuntime, EagerTensor};
use tenferro_cpu::provider::{
    CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest, CpuProviderOutcome,
};
use tenferro_cpu::{
    discover_cpu_topology, CpuBackend, CpuDomainExecutor, CpuDomainExecutorCapabilities,
    CpuDomainExecutorError, CpuExecutorAffinity, CpuExecutorReentrancy, CpuExecutorShutdown,
    CpuInnerParallelism, CpuPlacement, CpuPlacementError, CpuPlacementGuarantee, CpuProviderBundle,
    ExternalCpuDomain, NumaNodeId, ResolvedCpuPlacement, ScopedCpuJob, ScopedCpuJobs,
};
use tenferro_runtime::{
    Error as RuntimeError, ErrorPhase, GraphCompiler, GraphExecutor, TracedTensor,
};
use tenferro_tensor::{
    BackendSessionHost, CpuDomainId, DotGeneralConfig, Error as TensorError, ErrorKind, Tensor,
    TensorDot, TensorElementwise,
};

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

#[derive(Debug)]
struct MarkerGemmProvider {
    marker: &'static str,
    calls: Arc<AtomicUsize>,
}

impl MarkerGemmProvider {
    fn fail(&self) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(TensorError::backend_failure(
            "placement_bound_snapshot",
            self.marker,
        ))
    }
}

impl CpuGemmProvider for MarkerGemmProvider {
    fn gemm(
        &self,
        _context: &tenferro_cpu::CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.fail()
    }

    fn strided_batched_gemm(
        &self,
        _context: &tenferro_cpu::CpuExecutionContext<'_>,
        _request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.fail()
    }

    fn grouped_gemm(
        &self,
        _context: &tenferro_cpu::CpuExecutionContext<'_>,
        _request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.fail()
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

fn with_marker_provider(
    backend: CpuBackend,
    marker: &'static str,
    calls: Arc<AtomicUsize>,
) -> CpuBackend {
    let bundle = CpuProviderBundle::builder(backend.kind())
        .gemm_provider(Arc::new(MarkerGemmProvider { marker, calls }))
        .build()
        .unwrap();
    backend.with_provider_bundle(bundle).unwrap()
}

fn add_one(session: &mut dyn tenferro_tensor::BackendSession) -> tenferro_ad::Result<Tensor> {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    TensorElementwise::add(session, &lhs, &rhs).map_err(RuntimeError::from)
}

fn dot_error(session: &mut dyn tenferro_tensor::BackendSession) -> RuntimeError {
    let lhs = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 7.0, 6.0, 8.0]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    TensorDot::dot_general(session, &lhs, &rhs, &config)
        .map_err(RuntimeError::from)
        .unwrap_err()
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
fn one_session_enters_each_core_operation_exactly_once() {
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
fn binding_snapshots_cpu_coordinator_and_provider_before_runtime_replacement() {
    let old_executor = Arc::new(ExecutorCounters::default());
    let new_executor = Arc::new(ExecutorCounters::default());
    let old_provider_calls = Arc::new(AtomicUsize::new(0));
    let new_provider_calls = Arc::new(AtomicUsize::new(0));
    let old_backend = with_marker_provider(
        external_backend(Arc::clone(&old_executor)),
        "old provider snapshot",
        Arc::clone(&old_provider_calls),
    );
    let runtime = EagerRuntime::with_cpu_backend(old_backend);
    let mut cpu = runtime.on_cpu(placement()).unwrap();
    let replacement = with_marker_provider(
        external_backend(Arc::clone(&new_executor)),
        "new provider snapshot",
        Arc::clone(&new_provider_calls),
    );
    runtime
        .with_backend_mut(|backend| *backend = EagerBackend::Cpu(replacement))
        .unwrap();

    cpu.with_eager_session(add_one).unwrap();
    let old_error = cpu
        .with_eager_session::<()>(|session| Err(dot_error(session)))
        .unwrap_err();
    assert!(old_error.to_string().contains("old provider snapshot"));
    assert_eq!(old_executor.installs.load(Ordering::Relaxed), 2);
    assert_eq!(new_executor.installs.load(Ordering::Relaxed), 0);
    assert_eq!(old_provider_calls.load(Ordering::Relaxed), 1);
    assert_eq!(new_provider_calls.load(Ordering::Relaxed), 0);

    let new_error = runtime
        .with_backend_mut(|backend| backend.with_backend_session(dot_error))
        .unwrap();
    assert!(new_error.to_string().contains("new provider snapshot"));
    assert_eq!(new_provider_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn binding_retains_selected_executor_owner_after_runtime_replacement() {
    let old = Arc::new(ExecutorCounters::default());
    let runtime = EagerRuntime::with_cpu_backend(external_backend(Arc::clone(&old)));
    let cpu = runtime.on_cpu(placement()).unwrap();
    runtime
        .with_backend_mut(|backend| *backend = EagerBackend::Cpu(CpuBackend::new()))
        .unwrap();

    assert_eq!(old.drops.load(Ordering::Relaxed), 0);
    drop(cpu);
    assert_eq!(old.drops.load(Ordering::Relaxed), 1);
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
    let graph_output = GraphExecutor::new(graph_backend).run(&program).unwrap();

    assert_eq!(session_output.as_slice::<f64>().unwrap(), &[3.0]);
    assert_eq!(
        session_output.as_slice::<f64>().unwrap(),
        graph_output.as_slice::<f64>().unwrap()
    );
}
