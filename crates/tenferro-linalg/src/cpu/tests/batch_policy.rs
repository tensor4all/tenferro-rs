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

    let mut backend = CpuBackend::with_threads(4).unwrap();
    let decision = with_cpu_linalg(&mut backend, |session| {
        with_batch_policy(
            session,
            CpuBatchPolicy::new(CpuBatchStrategy::Outer),
            |session| {
                with_cpu_exec_session(session, |cpu| {
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
    let mut backend = CpuBackend::with_threads(2).unwrap();

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
                    with_cpu_exec_session(session, |cpu| cpu.lu_factor(&input))
                        .expect("a CPU session")
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
