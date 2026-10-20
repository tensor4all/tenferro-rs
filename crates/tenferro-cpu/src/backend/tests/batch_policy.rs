//! Batch-policy precedence, restoration and end-to-end policy delivery (#2142).

use super::*;
use crate::{CpuBatchPolicy, CpuBatchStrategy};

/// Read the effective policy of a live CPU session through its linalg context.
fn observed(session: &mut dyn tenferro_tensor::Engine) -> CpuBatchStrategy {
    crate::with_cpu_engine(session, |cpu| {
        cpu.with_linalg_pool(|context, _| Ok(context.batch_policy().strategy))
    })
    .expect("a CPU backend session")
    .unwrap()
}

#[test]
fn scoped_overrides_nest_over_the_backend_default_and_restore() {
    let mut backend = crate::test_support::cpu_engine(1)
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
    let mut backend = crate::test_support::cpu_engine(1).unwrap();

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
    let mut backend = crate::test_support::cpu_engine(1).unwrap();

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

fn batched_matmul_config() -> tenferro_tensor::DotGeneralConfig {
    tenferro_tensor::DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    }
}

fn contraction_inputs() -> (Tensor, Tensor) {
    let (rows, inner, cols, batch) = (8usize, 8usize, 8usize, 16usize);
    let lhs = Tensor::from_vec_col_major(
        vec![rows, inner, batch],
        (0..rows * inner * batch)
            .map(|index| ((index % 7) as f64) - 3.0)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let rhs = Tensor::from_vec_col_major(
        vec![inner, cols, batch],
        (0..inner * cols * batch)
            .map(|index| ((index % 5) as f64) - 2.0)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    (lhs, rhs)
}

fn dot_with_strategy(threads: usize, strategy: CpuBatchStrategy) -> Tensor {
    use tenferro_tensor::TensorRead;

    let (lhs, rhs) = contraction_inputs();
    let config = batched_matmul_config();
    let mut backend = crate::test_support::cpu_engine(threads).unwrap();
    with_cpu_session(&mut backend, |session| {
        crate::with_batch_policy(session, CpuBatchPolicy::new(strategy), |session| {
            session
                .dot_general_read(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                )
                .unwrap()
        })
        .expect("the CPU session accepts the policy")
    })
}

/// A forced `Outer` executes the same binary contraction as `Sequential`, bit
/// for bit, over a batch large enough to fan out.
#[test]
fn forced_outer_binary_contraction_matches_sequential() {
    let sequential = dot_with_strategy(1, CpuBatchStrategy::Sequential);
    let outer = dot_with_strategy(4, CpuBatchStrategy::Outer);
    assert_eq!(
        outer.as_slice::<f64>().unwrap(),
        sequential.as_slice::<f64>().unwrap()
    );
}

/// The forced `Outer` route is observable on the native alternatives: it must
/// not stay on the coordinator-inline vendor loop.
#[test]
fn forced_outer_binary_contraction_takes_the_native_route() {
    let _ = dot_with_strategy(4, CpuBatchStrategy::Outer);
    assert_eq!(
        crate::contraction::take_last_contraction_route(),
        Some(crate::contraction::ContractionRoute::Native)
    );
}

/// `WholeBatchVendor` is unavailable on the per-job BLAS contraction route; the
/// refusal must precede the caller-owned output write.
#[cfg(feature = "blas")]
#[test]
fn forced_whole_batch_vendor_on_blas_contraction_is_refused_before_writes() {
    use tenferro_tensor::{TensorRead, TensorWrite};

    let (lhs, rhs) = contraction_inputs();
    let config = batched_matmul_config();
    let mut out = Tensor::from_vec_col_major(vec![8, 8, 16], vec![-71.0_f64; 8 * 8 * 16]).unwrap();
    let mut backend = crate::test_support::cpu_engine(4).unwrap();
    let result = with_cpu_session(&mut backend, |session| {
        crate::with_batch_policy(
            session,
            CpuBatchPolicy::new(CpuBatchStrategy::WholeBatchVendor),
            |session| {
                session.dot_general_read_into(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                    TensorWrite::from_tensor(&mut out),
                )
            },
        )
        .expect("the CPU session accepts the policy")
    });
    let error = result.unwrap_err();
    assert_eq!(
        error.kind(),
        tenferro_tensor::ErrorKind::Unsupported,
        "{error}"
    );
    assert!(
        matches!(
            std::error::Error::source(&error).and_then(|source| source.downcast_ref()),
            Some(cpueinsum_blas::BlasError::StrategyUnavailable)
        ),
        "the provider refusal must stay the typed source: {error}"
    );
    assert_eq!(out.as_slice::<f64>().unwrap(), &[-71.0_f64; 8 * 8 * 16]);
}

/// A session that is not a CPU execution session and forwards none.
struct NonCpuSession;

mod non_cpu_session {
    use super::NonCpuSession;
    use tenferro_tensor::{
        CompareDir, DType, DotGeneralConfig, ElementwiseReadOp, Engine, GatherConfig, PadConfig,
        ScatterConfig, SliceConfig, Tensor, TensorAnalytic, TensorBuffer, TensorDeviceTransfer,
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

    impl tenferro_tensor::SessionCachedDot for NonCpuSession {}
    impl Engine for NonCpuSession {}

    impl crate::test_support::TestEngine for NonCpuSession {
        fn as_engine(&mut self) -> &mut dyn Engine {
            self
        }
    }
}
