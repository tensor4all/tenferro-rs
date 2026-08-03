use tenferro_ad::{EagerRuntime, Error};
use tenferro_cpu::{CpuBackend, CpuPlacement};
use tenferro_tensor::{Tensor, TensorElementwise};

fn add_values(session: &mut dyn tenferro_tensor::BackendSession) -> tenferro_ad::Result<Tensor> {
    let lhs = Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![1], vec![2.0_f64]).unwrap();
    TensorElementwise::add(session, &lhs, &rhs).map_err(Error::from)
}

#[test]
fn runtime_snapshot_bridge_preserves_public_identity_placement_and_results() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let mut cpu = runtime.on_cpu(CpuPlacement::Auto).unwrap();

    assert_eq!(cpu.runtime_id(), runtime.id());
    assert_eq!(cpu.placement(), CpuPlacement::Auto);
    assert_eq!(
        cpu.with_eager_session(add_values)
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[3.0]
    );
    assert_eq!(
        cpu.with_eager_session(add_values)
            .unwrap()
            .as_slice::<f64>()
            .unwrap(),
        &[3.0]
    );
}
