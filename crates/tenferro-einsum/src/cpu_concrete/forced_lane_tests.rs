//! Forced batch-lane execution through the concrete CPU einsum route.

use super::*;

fn nary_inputs() -> (Vec<Tensor>, ContractionTree) {
    let n = 32usize;
    let batch = 4usize;
    let tensor = |offset: usize| {
        Tensor::from_typed::<f64>(
            tenferro_tensor::TypedTensor::from_vec_col_major(
                vec![n, n, batch],
                (0..n * n * batch)
                    .map(|index| ((index + offset) % 11) as f64 - 5.0)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
    };
    let inputs = vec![tensor(0), tensor(3), tensor(7)];
    let subs = crate::Subscripts::parse("ijb,jkb,klb->ilb").unwrap();
    let shape_refs: Vec<&[usize]> = inputs.iter().map(Tensor::shape).collect();
    let tree = ContractionTree::optimize(&subs, &shape_refs).unwrap();
    (inputs, tree)
}

fn execute_nary_with_strategy(
    strategy: tenferro_cpu::CpuBatchStrategy,
) -> tenferro_tensor::Result<Tensor> {
    let (inputs, tree) = nary_inputs();
    let reads: Vec<TensorRead<'_>> = inputs.iter().map(TensorRead::from_tensor).collect();
    let mut backend =
        tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(4).build().unwrap());
    {
        let session = &mut backend;
        {
            tenferro_cpu::with_batch_policy(
                session,
                tenferro_cpu::CpuBatchPolicy::new(strategy),
                |session| {
                    execute_cached(session, &reads, &tree, None)
                        .expect("concrete CPU route must be selected")
                },
            )
            .expect("the CPU session accepts the policy")
        }
    }
}

/// A forced `Outer` N-ary einsum executes the same contraction as the forced
/// serial route, bit for bit, on both compiled CPU lanes.
#[test]
fn forced_outer_nary_einsum_matches_sequential() {
    let sequential =
        execute_nary_with_strategy(tenferro_cpu::CpuBatchStrategy::Sequential).unwrap();
    let outer = execute_nary_with_strategy(tenferro_cpu::CpuBatchStrategy::Outer).unwrap();
    assert_eq!(
        outer.as_slice::<f64>().unwrap(),
        sequential.as_slice::<f64>().unwrap()
    );
}
