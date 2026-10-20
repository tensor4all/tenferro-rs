//! #2142: the linalg adapters carry the resolved batch policy, and the BLAS
//! lane refuses a forced strategy it cannot honor.

use super::*;
use tenferro_cpu::{with_batch_policy, CpuBatchPolicy, CpuBatchStrategy};

/// A forced `Outer` reaches the `Exec` the tlinalg (faer) adapter hands the
/// provider: the token carries the resolved strategy and fans the batch over
/// the budget.
#[cfg(feature = "native")]
#[test]
fn forced_outer_reaches_the_faer_exec() {
    use crate::cpu::tlinalg::exec_from;

    let mut backend = tenferro_cpu::CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(4)
            .build()
            .unwrap(),
    );
    let decision = with_cpu_linalg(&mut backend, |session| {
        with_batch_policy(
            session,
            CpuBatchPolicy::new(CpuBatchStrategy::Outer),
            |session| {
                with_cpu_engine(session, |cpu| {
                    cpu.with_linalg_pool(|context, _| {
                        let exec = exec_from(context)?;
                        assert_eq!(exec.batch_policy().strategy, CpuBatchStrategy::Outer);
                        Ok(exec
                            .batch_decision(&tcontract_exec::BatchQuery::new(8, 4096.0))
                            .expect("forced Outer is available on the faer route"))
                    })
                })
                .expect("a CPU session")
            },
        )
        .expect("the CPU session accepts the policy")
    })
    .expect("the faer exec carries the resolved policy");
    assert_eq!(decision.lanes, 4, "{decision:?}");
}

/// Every forced route the BLAS linalg lane cannot honor is a typed unsupported
/// error raised before any write, even for a degenerate empty batch.
#[cfg(feature = "blas")]
#[test]
fn forced_strategy_on_the_blas_linalg_lane_is_refused_before_writes() {
    use std::num::NonZeroUsize;

    let n = 4usize;
    let batch = 3usize;
    let a_data: Vec<f64> = (0..n * n * batch)
        .map(|index| ((index % 7) as f64) - 3.0)
        .collect();
    let mut backend = tenferro_cpu::CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(2)
            .build()
            .unwrap(),
    );

    let strategies = [
        CpuBatchStrategy::Outer,
        CpuBatchStrategy::Inner,
        CpuBatchStrategy::Split {
            lanes: NonZeroUsize::new(2).unwrap(),
        },
        CpuBatchStrategy::WholeBatchVendor,
    ];

    for shape in [vec![n, n, batch], vec![0, 0, 0]] {
        let len: usize = shape.iter().product();
        let data = if len == 0 { Vec::new() } else { a_data.clone() };
        let input = Tensor::from_vec_col_major(shape, data.clone()).unwrap();
        for strategy in strategies {
            let result = with_cpu_linalg(&mut backend, |session| {
                with_batch_policy(session, CpuBatchPolicy::new(strategy), |session| {
                    with_cpu_engine(session, |cpu| cpu.lu_factor(&input)).expect("a CPU session")
                })
                .expect("the CPU session accepts the policy")
            });
            let error = match result {
                Ok(_) => panic!("{strategy:?} must be refused"),
                Err(error) => error,
            };
            assert_eq!(
                error.kind(),
                tenferro_tensor::ErrorKind::Unsupported,
                "{strategy:?}: {error}"
            );
            assert_eq!(
                input.as_slice::<f64>().unwrap(),
                &data[..],
                "{strategy:?}: the refusal must precede every write"
            );
        }
    }
}

/// The direct caller-output solve takes the same refusal gate as the
/// allocating entries: every forced strategy the BLAS linalg lane cannot honor
/// is refused before the caller's sentinels are touched.
#[cfg(feature = "blas")]
#[test]
fn forced_strategy_on_blas_solve_read_into_is_refused_before_writes() {
    use std::num::NonZeroUsize;

    let a = Tensor::from_vec_col_major([2, 2], vec![3.0_f64, 1.0, 0.0, 2.0]).unwrap();
    let b = Tensor::from_vec_col_major([2, 1], vec![7.0_f64, 4.0]).unwrap();
    let mut backend = tenferro_cpu::CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(2)
            .build()
            .unwrap(),
    );

    for strategy in [
        CpuBatchStrategy::Outer,
        CpuBatchStrategy::Inner,
        CpuBatchStrategy::Split {
            lanes: NonZeroUsize::new(2).unwrap(),
        },
        CpuBatchStrategy::WholeBatchVendor,
    ] {
        let mut out = Tensor::from_vec_col_major([2, 1], vec![-41.0_f64; 2]).unwrap();
        let result = with_cpu_linalg(&mut backend, |session| {
            with_batch_policy(session, CpuBatchPolicy::new(strategy), |session| {
                with_cpu_engine(session, |cpu| {
                    cpu.solve_read_into(
                        TensorRead::from_tensor(&a),
                        TensorRead::from_tensor(&b),
                        tenferro_tensor::TensorWrite::from_tensor(&mut out),
                    )
                })
                .expect("a CPU session")
            })
            .expect("the CPU session accepts the policy")
        });
        let error = result.unwrap_err();
        assert_eq!(
            error.kind(),
            tenferro_tensor::ErrorKind::Unsupported,
            "{strategy:?}: {error}"
        );
        assert_eq!(
            out.as_slice::<f64>().unwrap(),
            &[-41.0, -41.0],
            "{strategy:?}: the refusal must precede the caller-output write"
        );
    }
}

/// `tlinalg` resolves strategy availability before its own empty-batch return;
/// the native empty shortcuts must not bypass that validation.
#[cfg(feature = "native")]
#[test]
fn forced_whole_batch_vendor_on_native_empty_inputs_is_refused() {
    use std::error::Error as _;

    let mut backend = tenferro_cpu::CpuEngine::new(
        tenferro_cpu::thread_pool_builder()
            .num_threads(2)
            .build()
            .unwrap(),
    );
    for shape in [vec![2, 2, 0], vec![0, 0]] {
        let input = Tensor::from_vec_col_major(shape.clone(), Vec::<f64>::new()).unwrap();
        let result = with_cpu_linalg(&mut backend, |session| {
            with_batch_policy(
                session,
                CpuBatchPolicy::new(CpuBatchStrategy::WholeBatchVendor),
                |session| {
                    with_cpu_engine(session, |cpu| cpu.cholesky(&input)).expect("a CPU session")
                },
            )
            .expect("the CPU session accepts the policy")
        });
        let error = result.unwrap_err();
        assert_eq!(
            error.kind(),
            tenferro_tensor::ErrorKind::Unsupported,
            "{shape:?}: {error}"
        );
        assert!(
            matches!(
                error.source().and_then(|source| source.downcast_ref()),
                Some(tlinalg::Error::StrategyUnavailable { .. })
            ),
            "{shape:?}: the provider refusal must stay the typed source: {error}"
        );
    }
}
