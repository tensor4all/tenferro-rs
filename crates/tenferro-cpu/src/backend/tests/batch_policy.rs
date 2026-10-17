//! Batch-policy precedence, restoration and end-to-end policy delivery (#2142).

use super::*;
use crate::{CpuBatchPolicy, CpuBatchStrategy};

/// Read the effective policy of a live CPU session through its linalg context.
fn observed(session: &mut dyn tenferro_tensor::BackendSession) -> CpuBatchStrategy {
    crate::with_cpu_exec_session(session, |cpu| {
        cpu.with_linalg_pool(|context, _| Ok(context.batch_policy().strategy))
    })
    .expect("a CPU backend session")
    .unwrap()
}

#[test]
fn scoped_overrides_nest_over_the_backend_default_and_restore() {
    let mut backend = CpuBackend::with_threads(1)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Sequential));

    let seen = with_cpu_session(&mut backend, |session| {
        let mut seen = vec![observed(session)];
        crate::with_batch_policy(
            session,
            CpuBatchPolicy::new(CpuBatchStrategy::Outer),
            |session| {
                seen.push(observed(session));
                crate::with_batch_policy(session, CpuBatchPolicy::default(), |session| {
                    seen.push(observed(session));
                })
                .unwrap();
                seen.push(observed(session));
            },
        )
        .unwrap();
        seen.push(observed(session));
        seen
    });

    assert_eq!(
        seen,
        [
            CpuBatchStrategy::Sequential,
            CpuBatchStrategy::Outer,
            CpuBatchStrategy::Auto,
            CpuBatchStrategy::Outer,
            CpuBatchStrategy::Sequential,
        ]
    );
}

#[test]
fn scoped_override_is_restored_after_an_error() {
    let mut backend = CpuBackend::with_threads(1).unwrap();

    with_cpu_session(&mut backend, |session| {
        let returned = crate::with_batch_policy(
            session,
            CpuBatchPolicy::new(CpuBatchStrategy::Sequential),
            |_| -> Result<(), &'static str> { Err("callback error") },
        )
        .unwrap();
        assert_eq!(returned, Err("callback error"));
        assert_eq!(observed(session), CpuBatchStrategy::Auto);
    });
}

#[test]
fn scoped_override_is_restored_after_unwind() {
    let mut backend = CpuBackend::with_threads(1).unwrap();

    with_cpu_session(&mut backend, |session| {
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = crate::with_batch_policy(
                session,
                CpuBatchPolicy::new(CpuBatchStrategy::Outer),
                |_| panic!("forced scope panic"),
            );
        }));
        assert!(unwound.is_err());
        assert_eq!(observed(session), CpuBatchStrategy::Auto);
    });
}

#[test]
fn a_non_cpu_session_is_rejected() {
    let mut session = NonCpuSession;
    let error = crate::with_batch_policy(
        &mut session,
        CpuBatchPolicy::new(CpuBatchStrategy::Auto),
        |_| (),
    )
    .expect_err("a non-CPU session must be rejected");
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::Unsupported);
}

#[test]
fn backend_default_policy_follows_clones() {
    let backend = CpuBackend::with_threads(1)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::WholeBatchVendor));
    let clone = backend.clone();
    assert_eq!(
        clone.batch_policy().strategy,
        CpuBatchStrategy::WholeBatchVendor
    );
    let mut clone = clone;
    let strategy = with_cpu_session(&mut clone, |session| observed(session));
    assert_eq!(strategy, CpuBatchStrategy::WholeBatchVendor);
}

/// A forced `Outer` reaches the `Exec` the contraction adapter hands to
/// `cpueinsum`: the token carries the resolved strategy and its decision fans
/// the batch over the budget instead of the route's `Auto` result.
#[test]
fn forced_outer_reaches_the_contraction_exec() {
    let mut backend = CpuBackend::with_threads(4).unwrap();
    let decision = with_cpu_session(&mut backend, |session| {
        crate::with_batch_policy(
            session,
            CpuBatchPolicy::new(CpuBatchStrategy::Outer),
            |session| {
                crate::with_cpu_exec_session(session, |cpu| {
                    cpu.with_contraction_exec(|exec, _, _| {
                        assert_eq!(exec.batch_policy().strategy, CpuBatchStrategy::Outer);
                        Ok(exec
                            .batch_decision(&cpueinsum::BatchQuery::new(8, 4096.0))
                            .expect("forced Outer is available on the contraction route"))
                    })
                    .expect("a CPU session")
                })
            },
        )
        .expect("the CPU session accepts the policy")
    })
    .expect("the contraction exec carries the resolved policy");
    assert_eq!(decision.lanes, 4, "{decision:?}");
}

/// A session that is not a CPU execution session and forwards none.
struct NonCpuSession;

mod non_cpu_session {
    use super::NonCpuSession;
    use tenferro_tensor::{
        BackendCachedDot, BackendRuntimeCache, BackendSession, BackendSessionHost, CompareDir,
        DType, DotGeneralConfig, ElementwiseReadOp, GatherConfig, PadConfig, ScatterConfig,
        SliceConfig, Tensor, TensorAnalytic, TensorBackend, TensorBuffer, TensorDeviceTransfer,
        TensorDot, TensorElementwise, TensorFusion, TensorIndexing, TensorRead, TensorReduction,
        TensorStructural, TensorWrite,
    };

    type Result<T> = tenferro_tensor::Result<T>;

    macro_rules! unreachable_methods {
        ($($name:ident($($arg:ident : $argty:ty),*) -> $ret:ty;)+) => {
            $(
                fn $name(&mut self, $($arg: $argty),*) -> $ret {
                    $(let _ = &$arg;)*
                    panic!(concat!(stringify!($name), " is not used by this test"))
                }
            )+
        };
    }

    impl BackendRuntimeCache for NonCpuSession {
        type RuntimeCache = ();
    }

    impl TensorElementwise for NonCpuSession {
        unreachable_methods! {
            elementwise_read_into(op: ElementwiseReadOp, inputs: &[TensorRead<'_>], out: TensorWrite<'_>) -> Result<()>;
            add_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
            neg_read(input: TensorRead<'_>) -> Result<Tensor>;
            conj_read(input: TensorRead<'_>) -> Result<Tensor>;
            abs_read(input: TensorRead<'_>) -> Result<Tensor>;
            sign_read(input: TensorRead<'_>) -> Result<Tensor>;
            compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) -> Result<Tensor>;
            select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> Result<Tensor>;
            clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> Result<Tensor>;
        }
    }

    impl TensorAnalytic for NonCpuSession {
        unreachable_methods! {
            exp_read(input: TensorRead<'_>) -> Result<Tensor>;
            log_read(input: TensorRead<'_>) -> Result<Tensor>;
            sin_read(input: TensorRead<'_>) -> Result<Tensor>;
            cos_read(input: TensorRead<'_>) -> Result<Tensor>;
            tanh_read(input: TensorRead<'_>) -> Result<Tensor>;
            sqrt_read(input: TensorRead<'_>) -> Result<Tensor>;
            rsqrt_read(input: TensorRead<'_>) -> Result<Tensor>;
            expm1_read(input: TensorRead<'_>) -> Result<Tensor>;
            log1p_read(input: TensorRead<'_>) -> Result<Tensor>;
            erf_read(input: TensorRead<'_>) -> Result<Tensor>;
            pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> Result<Tensor>;
        }
    }

    impl TensorStructural for NonCpuSession {
        unreachable_methods! {
            cast(input: &Tensor, to: DType) -> Result<Tensor>;
            extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> Result<Tensor>;
            embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> Result<Tensor>;
            tril(input: &Tensor, k: i64) -> Result<Tensor>;
            triu(input: &Tensor, k: i64) -> Result<Tensor>;
            transpose_read(input: TensorRead<'_>, perm: &[usize]) -> Result<Tensor>;
            reshape_read(input: TensorRead<'_>, shape: &[usize]) -> Result<Tensor>;
            broadcast_in_dim_read(input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> Result<Tensor>;
        }
    }

    impl TensorReduction for NonCpuSession {
        unreachable_methods! {
            reduce_sum_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
            reduce_prod_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
            reduce_max_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
            reduce_min_read(input: TensorRead<'_>, axes: &[usize]) -> Result<Tensor>;
        }
    }

    impl TensorIndexing for NonCpuSession {
        unreachable_methods! {
            gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> Result<Tensor>;
            scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> Result<Tensor>;
            slice(input: &Tensor, config: &SliceConfig) -> Result<Tensor>;
            dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> Result<Tensor>;
            dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> Result<Tensor>;
            pad(input: &Tensor, config: &PadConfig) -> Result<Tensor>;
            concatenate(inputs: &[&Tensor], axis: usize) -> Result<Tensor>;
            reverse(input: &Tensor, axes: &[usize]) -> Result<Tensor>;
        }
    }

    impl TensorDot for NonCpuSession {
        unreachable_methods! {
            dot_general_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, config: &DotGeneralConfig) -> Result<Tensor>;
        }
    }
    impl TensorFusion for NonCpuSession {}
    impl TensorBuffer for NonCpuSession {}

    impl TensorDeviceTransfer for NonCpuSession {
        unreachable_methods! {
            download_to_host(tensor: TensorRead<'_>) -> Result<Tensor>;
            upload_host_tensor(tensor: TensorRead<'_>) -> Result<Tensor>;
        }
    }

    impl BackendCachedDot for NonCpuSession {}
    impl BackendSession for NonCpuSession {}

    impl BackendSessionHost for NonCpuSession {
        fn with_backend_session<R>(
            &mut self,
            f: impl FnOnce(&mut dyn BackendSession) -> R,
        ) -> std::result::Result<R, tenferro_tensor::SessionEntryError> {
            tenferro_tensor::with_session_entry_guard("non-CPU test session", || f(self))
        }
    }

    impl TensorBackend for NonCpuSession {}
}
