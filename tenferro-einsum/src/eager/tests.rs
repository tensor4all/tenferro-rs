use tenferro_tensor::{cpu::CpuBackend, Tensor, TensorBackend, TensorRead, TensorView};

use super::{
    binary_contract, eager_einsum_exec_read, try_eager_einsum_binary_read_fast, LabeledTensor,
    TensorValue,
};
use crate::{ContractionTree, Subscripts};

#[test]
fn tensor_value_view_paths_materialize_and_read() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]);
    let view_shape = [2usize];
    let view_data = [3.0_f64, 4.0];
    let view = TensorView::f64(&view_shape, &view_data).unwrap();

    let borrowed = TensorValue::Borrowed(&tensor);
    assert_eq!(borrowed.as_tensor().unwrap().shape(), &[2]);
    assert_eq!(borrowed.tensor_read().shape(), &[2]);

    let owned = TensorValue::Owned(tensor.clone());
    assert_eq!(owned.as_tensor().unwrap().shape(), &[2]);
    assert_eq!(owned.tensor_read().shape(), &[2]);

    let view_value = TensorValue::View(view);
    assert!(view_value.as_tensor().is_none());
    assert_eq!(view_value.tensor_read().shape(), &[2]);
    assert_eq!(
        view_value.into_tensor().as_slice::<f64>().unwrap(),
        &[3.0, 4.0]
    );
}

#[test]
fn generic_outer_product_with_views_uses_broadcast_path() {
    let lhs_shape = [2usize];
    let lhs_data = [1.0_f64, 2.0];
    let rhs_shape = [3usize];
    let rhs_data = [3.0_f64, 4.0, 5.0];
    let lhs_view = TensorView::f64(&lhs_shape, &lhs_data).unwrap();
    let rhs_view = TensorView::f64(&rhs_shape, &rhs_data).unwrap();
    let lhs = LabeledTensor {
        tensor: TensorValue::View(lhs_view),
        labels: vec![0],
    };
    let rhs = LabeledTensor {
        tensor: TensorValue::View(rhs_view),
        labels: vec![1],
    };

    let mut ctx = CpuBackend::new();
    let result = ctx
        .with_exec_session(|exec| binary_contract(exec, lhs, rhs, &[0, 1], true))
        .unwrap();
    let labels = result.labels;
    let tensor = result.tensor.into_tensor();

    assert_eq!(labels, vec![0, 1]);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(
        tensor.as_slice::<f64>().unwrap(),
        &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0]
    );
}

#[test]
fn generic_binary_contract_reduces_then_builds_dot_config() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 4], vec![1.0_f64; 24]);
    let rhs = Tensor::from_vec_col_major(vec![3, 5], vec![2.0_f64; 15]);
    let lhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&lhs),
        labels: vec![0, 1, 9],
    };
    let rhs = LabeledTensor {
        tensor: TensorValue::Borrowed(&rhs),
        labels: vec![1, 2],
    };

    let mut ctx = CpuBackend::new();
    let result = ctx
        .with_exec_session(|exec| binary_contract(exec, lhs, rhs, &[0, 2], false))
        .unwrap();
    let labels = result.labels;
    let tensor = result.tensor.into_tensor();

    assert_eq!(labels, vec![0, 2]);
    assert_eq!(tensor.shape(), &[2, 5]);
    assert_eq!(tensor.as_slice::<f64>().unwrap(), &[24.0; 10]);
}

#[test]
fn generic_read_exec_reduces_single_view_input() {
    let shape = [2usize, 3];
    let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let view = TensorView::f64(&shape, &data).unwrap();
    let inputs = [TensorRead::from_view(view)];
    let subscripts = Subscripts::parse("ij->i").unwrap();
    let tree = ContractionTree::optimize(&subscripts, &[&shape]).unwrap();

    let mut ctx = CpuBackend::new();
    let result = ctx
        .with_exec_session(|exec| eager_einsum_exec_read(exec, &inputs, &tree))
        .unwrap();

    assert_eq!(result.shape(), &[2]);
    assert_eq!(result.as_slice::<f64>().unwrap(), &[9.0, 12.0]);
}

#[test]
fn binary_read_fast_path_rejects_non_fast_shapes_and_labels() {
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]);
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64; 6]);
    let mut ctx = CpuBackend::new();

    let subscripts = Subscripts::parse("ij,jk->ik").unwrap();
    let one_input = [TensorRead::from_tensor(&lhs)];
    assert!(try_eager_einsum_binary_read_fast(&mut ctx, &one_input, &subscripts).is_none());

    let flat_rhs = Tensor::from_vec_col_major(vec![6], vec![1.0_f64; 6]);
    let rank_mismatch = [
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&flat_rhs),
    ];
    assert!(try_eager_einsum_binary_read_fast(&mut ctx, &rank_mismatch, &subscripts).is_none());

    let duplicate_labels = Subscripts::parse("ii,jk->ik").unwrap();
    let inputs = [TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs)];
    assert!(try_eager_einsum_binary_read_fast(&mut ctx, &inputs, &duplicate_labels).is_none());
}
