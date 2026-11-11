#![cfg(feature = "autodiff")]
//! Traced VJP of one einsum node with more than 64 operands.
//!
//! The einsum AD rule declares every input as a residual
//! (`ResidualSpec::all_inputs`) and reads every input through the checked
//! accessor. The declaration has to cover the operand past index 63, or the
//! transpose fails with `UndeclaredResidualValue` at index 64.

use tenferro_ad::AdContext;
use tenferro_einsum::{EinsumSubscripts, TraceContextEinsumExt};
use tenferro_runtime::{GraphCompiler, TraceContext};
use tenferro_tensor::Tensor;

use super::support;

const OPERANDS: usize = 65;

fn vector(values: [f64; 2]) -> Tensor {
    Tensor::from_vec_col_major(vec![2], values.to_vec()).unwrap()
}

/// `out[i] = prod_k x_k[i]` over 65 operands, every one `[1, 1]` except the
/// last, `[2, 3]`. With an all-ones cotangent, `d/d x_0 = [2, 3]` (it needs the
/// last operand as a residual) and `d/d x_64 = [1, 1]`.
#[test]
fn traced_einsum_vjp_with_65_operands_reads_every_residual() {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_einsum::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    let mut trace = TraceContext::new();
    let leaves: Vec<_> = (0..OPERANDS)
        .map(|k| {
            let values = if k + 1 == OPERANDS {
                [2.0, 3.0]
            } else {
                [1.0, 1.0]
            };
            trace.from_tensor_concrete_shape(vector(values)).unwrap()
        })
        .collect();
    let values: Vec<_> = leaves.iter().map(|leaf| leaf.trace_value()).collect();
    let labels: Vec<&[u32]> = vec![&[0]; OPERANDS];
    let output = trace
        .einsum_subscripts(&values, &EinsumSubscripts::new(&labels, &[0]))
        .unwrap();
    let output = trace.derive_tensor(output).unwrap();
    let cotangent = trace
        .from_tensor_concrete_shape(vector([1.0, 1.0]))
        .unwrap();

    let forward = support::run_one(&GraphCompiler::new().compile(&output).unwrap(), &[]).unwrap();
    assert_eq!(forward.as_slice::<f64>().unwrap(), &[2.0, 3.0]);

    for (k, expected) in [(0, [2.0, 3.0]), (OPERANDS - 1, [1.0, 1.0])] {
        let grad = ad.vjp(&output, &leaves[k], &cotangent).unwrap();
        let grad = support::run_one(&GraphCompiler::new().compile(&grad).unwrap(), &[]).unwrap();
        assert_eq!(grad.as_slice::<f64>().unwrap(), &expected, "operand {k}");
    }
}
