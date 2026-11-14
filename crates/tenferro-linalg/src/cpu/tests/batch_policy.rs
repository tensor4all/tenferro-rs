//! #2142: the linalg adapters carry the resolved batch policy, and a forced
//! strategy is honored or refused, never silently downgraded.

use super::*;
use tenferro_cpu::{with_batch_policy, CpuBatchPolicy, CpuBatchStrategy};

/// A forced `Outer` reaches the `Exec` the tlinalg (faer) adapter hands the
/// provider: the token carries the resolved strategy and fans the batch over
/// the budget.
#[test]
fn forced_outer_reaches_the_faer_exec() {
    let mut backend =
        tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(4).build().unwrap());
    let decision = with_cpu_linalg(&mut backend, |session| {
        with_batch_policy(
            session,
            CpuBatchPolicy::new(CpuBatchStrategy::Outer),
            |session| {
                with_cpu_engine(session, |cpu| {
                    cpu.with_linalg_pool(|exec, _| {
                        assert_eq!(exec.batch_policy().strategy, CpuBatchStrategy::Outer);
                        Ok(exec.batch_decision(&tcontract_exec::BatchQuery::new(8, 4096.0)))
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
