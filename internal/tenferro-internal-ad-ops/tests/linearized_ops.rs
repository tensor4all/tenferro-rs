use tenferro_internal_ad_core::{new_reverse_leaf, LinearizableOp, LinearizedOp};
use tenferro_internal_frontend_core::{DynTensor, StructuredTensor};
use tenferro_internal_runtime::{set_default_runtime, RuntimeContext};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor};

use tenferro_internal_ad_ops::{
    add_dyn_values, einsum_dyn_values, sum_dyn_value, AddOp, EinsumOp, ExpOp,
};

fn dyn_vec(values: &[f64], dims: &[usize]) -> DynTensor {
    let dense = Tensor::<f64>::from_slice(values, dims, MemoryOrder::ColumnMajor).unwrap();
    DynTensor::F64(StructuredTensor::from(dense))
}

fn dyn_scalar(value: f64) -> DynTensor {
    dyn_vec(&[value], &[])
}

fn f64_values(tensor: &DynTensor) -> Vec<f64> {
    match tensor {
        DynTensor::F64(value) => value
            .to_dense()
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap()
            .to_vec(),
        other => panic!("expected f64 dyn tensor, got {other:?}"),
    }
}

#[test]
fn linearized_ops_add_and_sum_dyn_values_use_linearized_runtime() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = new_reverse_leaf(dyn_vec(&[1.0, 2.0], &[2]));
    let y = new_reverse_leaf(dyn_vec(&[3.0, 4.0], &[2]));

    let added = add_dyn_values(&x, &y).unwrap();
    let summed = sum_dyn_value(&added).unwrap();
    let grads = summed
        .grad_wrt_with_seed(dyn_scalar(1.0), &[&x, &y])
        .unwrap();

    assert_eq!(f64_values(&grads[0].clone().unwrap()), vec![1.0, 1.0]);
    assert_eq!(f64_values(&grads[1].clone().unwrap()), vec![1.0, 1.0]);

    let op = AddOp;
    let outputs = op.primal(&[x.primal(), y.primal()]).unwrap();
    let linearized = op.linearize(&[x.primal(), y.primal()], &outputs).unwrap();
    let jvp = linearized
        .jvp(&[Some(dyn_vec(&[0.5, 1.5], &[2])), None])
        .unwrap();

    assert_eq!(f64_values(&jvp[0].clone().unwrap()), vec![0.5, 1.5]);
}

#[test]
fn linearized_ops_einsum_dyn_values_expose_vjp_and_jvp() {
    let _runtime = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let x = new_reverse_leaf(dyn_vec(&[1.0, 2.0], &[2]));
    let y = new_reverse_leaf(dyn_vec(&[3.0, 5.0], &[2]));

    let out = einsum_dyn_values("i,i->", &[&x, &y]).unwrap();
    let grads = out.grad_wrt_with_seed(dyn_scalar(1.0), &[&x, &y]).unwrap();

    assert_eq!(f64_values(&grads[0].clone().unwrap()), vec![3.0, 5.0]);
    assert_eq!(f64_values(&grads[1].clone().unwrap()), vec![1.0, 2.0]);

    let op = EinsumOp::new("i,i->");
    let outputs = op.primal(&[x.primal(), y.primal()]).unwrap();
    let linearized = op.linearize(&[x.primal(), y.primal()], &outputs).unwrap();
    let jvp = linearized
        .jvp(&[Some(dyn_vec(&[1.0, 0.0], &[2])), None])
        .unwrap();

    assert_eq!(f64_values(&jvp[0].clone().unwrap()), vec![3.0]);
}

#[test]
fn linearized_ops_exp_linearized_jvp_uses_saved_output_value() {
    let x = new_reverse_leaf(dyn_vec(&[0.0, 1.0], &[2]));

    let op = ExpOp;
    let outputs = op.primal(&[x.primal()]).unwrap();
    let output_values = f64_values(&outputs[0]);
    assert_eq!(output_values, vec![1.0, std::f64::consts::E]);

    let linearized = op.linearize(&[x.primal()], &outputs).unwrap();
    let jvp = linearized.jvp(&[Some(dyn_vec(&[1.0, 2.0], &[2]))]).unwrap();

    assert_eq!(
        f64_values(&jvp[0].clone().unwrap()),
        vec![1.0, 2.0 * std::f64::consts::E]
    );
}
