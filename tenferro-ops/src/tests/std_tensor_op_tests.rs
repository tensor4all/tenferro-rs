use crate::config::DotGeneralConfig;
use crate::std_tensor_op::StdTensorOp;
use computegraph::GraphOp;
use tenferro_tensor::{Tensor, TypedTensor};

fn eval_op(op: &StdTensorOp, inputs: &[&Tensor]) -> Vec<Tensor> {
    op.eval(&mut (), inputs)
}

fn get_f64(t: &Tensor, idx: &[usize]) -> f64 {
    match t {
        Tensor::F64(inner) => *inner.get(idx),
        _ => panic!("expected F64 tensor"),
    }
}

#[test]
fn test_add_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![1.0, 2.0]));
    let b = Tensor::F64(TypedTensor::from_vec(vec![2], vec![10.0, 20.0]));
    let out = eval_op(&StdTensorOp::Add, &[&a, &b]);
    assert_eq!(out.len(), 1);
    assert_eq!(get_f64(&out[0], &[0]), 11.0);
    assert_eq!(get_f64(&out[0], &[1]), 22.0);
}

#[test]
fn test_mul_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, 4.0]));
    let b = Tensor::F64(TypedTensor::from_vec(vec![2], vec![5.0, 6.0]));
    let out = eval_op(&StdTensorOp::Mul, &[&a, &b]);
    assert_eq!(get_f64(&out[0], &[0]), 15.0);
    assert_eq!(get_f64(&out[0], &[1]), 24.0);
}

#[test]
fn test_neg_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, -7.0]));
    let out = eval_op(&StdTensorOp::Neg, &[&a]);
    assert_eq!(get_f64(&out[0], &[0]), -3.0);
    assert_eq!(get_f64(&out[0], &[1]), 7.0);
}

#[test]
fn test_conj_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![2], vec![3.0, -7.0]));
    let out = eval_op(&StdTensorOp::Conj, &[&a]);
    assert_eq!(get_f64(&out[0], &[0]), 3.0);
    assert_eq!(get_f64(&out[0], &[1]), -7.0);
}

#[test]
fn test_dot_general_eval() {
    // Matrix multiply [2,3] x [3,2] -> [2,2]
    let a = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    ));
    let b = Tensor::F64(TypedTensor::from_vec(
        vec![3, 2],
        vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0],
    ));
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let out = eval_op(&StdTensorOp::DotGeneral(config), &[&a, &b]);
    assert_eq!(out[0].shape(), &[2, 2]);
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
    assert_eq!(get_f64(&out[0], &[0, 0]), 58.0);
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
    assert_eq!(get_f64(&out[0], &[1, 0]), 139.0);
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
    assert_eq!(get_f64(&out[0], &[0, 1]), 64.0);
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
    assert_eq!(get_f64(&out[0], &[1, 1]), 154.0);
}

#[test]
fn test_transpose_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let out = eval_op(&StdTensorOp::Transpose { perm: vec![1, 0] }, &[&a]);
    assert_eq!(out[0].shape(), &[3, 2]);
}

#[test]
fn test_reshape_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let out = eval_op(&StdTensorOp::Reshape { shape: vec![6] }, &[&a]);
    assert_eq!(out[0].shape(), &[6]);
}

#[test]
fn test_broadcast_in_dim_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(vec![1], vec![5.0]));
    let out = eval_op(
        &StdTensorOp::BroadcastInDim {
            shape: vec![3],
            dims: vec![0],
        },
        &[&a],
    );
    assert_eq!(out[0].shape(), &[3]);
    assert_eq!(get_f64(&out[0], &[0]), 5.0);
    assert_eq!(get_f64(&out[0], &[1]), 5.0);
    assert_eq!(get_f64(&out[0], &[2]), 5.0);
}

#[test]
fn test_reduce_sum_eval() {
    let a = Tensor::F64(TypedTensor::from_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ));
    let out = eval_op(&StdTensorOp::ReduceSum { axes: vec![0] }, &[&a]);
    assert_eq!(out[0].shape(), &[3]);
    assert_eq!(get_f64(&out[0], &[0]), 3.0);
    assert_eq!(get_f64(&out[0], &[1]), 7.0);
    assert_eq!(get_f64(&out[0], &[2]), 11.0);
}

#[test]
fn test_n_inputs_outputs() {
    assert_eq!(StdTensorOp::Add.n_inputs(), 2);
    assert_eq!(StdTensorOp::Mul.n_inputs(), 2);
    assert_eq!(StdTensorOp::Neg.n_inputs(), 1);
    assert_eq!(StdTensorOp::Conj.n_inputs(), 1);
    assert_eq!(
        StdTensorOp::DotGeneral(DotGeneralConfig {
            lhs_contracting_dims: vec![],
            rhs_contracting_dims: vec![],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        })
        .n_inputs(),
        2
    );
    assert_eq!(StdTensorOp::Transpose { perm: vec![0] }.n_inputs(), 1);
    assert_eq!(StdTensorOp::Reshape { shape: vec![1] }.n_inputs(), 1);
    assert_eq!(
        StdTensorOp::BroadcastInDim {
            shape: vec![1],
            dims: vec![0]
        }
        .n_inputs(),
        1
    );
    assert_eq!(StdTensorOp::ReduceSum { axes: vec![0] }.n_inputs(), 1);

    // All Tier 1 ops produce 1 output
    assert_eq!(StdTensorOp::Add.n_outputs(), 1);
    assert_eq!(StdTensorOp::Neg.n_outputs(), 1);
    assert_eq!(StdTensorOp::ReduceSum { axes: vec![0] }.n_outputs(), 1);
}
