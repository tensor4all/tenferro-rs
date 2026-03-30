use super::*;
use crate::Tensor;
use tenferro::{grad, GradOptions};
use tenferro_tensor::Tensor as DenseTensor;

fn diag_structured<T: tenferro_algebra::Scalar>(
    tensor: DenseTensor<T>,
    logical_rank: usize,
) -> StructuredTensor<T> {
    StructuredTensor(
        tenferro_tensor::StructuredTensor::from_diagonal_vector(tensor, logical_rank).unwrap(),
    )
}

#[test]
fn structured_reverse_pullback_accepts_dense_cotangent() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(
        diag_structured(
            DenseTensor::<f64>::from_slice(&[2.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
            2,
        ),
        &tape,
    );
    let alpha = reverse_leaf_f64(scalar_f64(2.0), &tape);
    let x_tensor = Tensor::from(x.clone());
    let alpha_tensor = Tensor::from(alpha.clone());
    let y = x_tensor.scale(&alpha_tensor).unwrap();
    let dense_cotangent = Tensor::from_tensor(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let grads = grad(
        &[&y],
        &[&x_tensor, &alpha_tensor],
        Some(&[dense_cotangent]),
        GradOptions::default(),
    )
    .unwrap();
    let grad_x = grads[0].as_ref().expect("missing structured wrt gradient");
    assert_eq!(
        tensor_to_vec_f64(grad_x.as_f64().unwrap().structured_primal()),
        vec![2.0, 2.0]
    );
    assert_eq!(grad_x.as_f64().unwrap().axis_classes(), &[0, 0]);
    assert_eq!(
        tensor_to_vec_f64(
            grads[1]
                .as_ref()
                .unwrap()
                .as_f64()
                .unwrap()
                .structured_primal()
        ),
        vec![5.0]
    );
}

#[test]
fn reshape_reverse_pullback_accepts_non_contiguous_cotangent_view() {
    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([1.0, 2.0, 3.0, 4.0]), &tape);
    let reshaped = Tensor::from(x.clone()).reshape(&[4]).unwrap();

    let base = DenseTensor::<f64>::from_slice(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[4, 4],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let cotangent_view = base.diagonal(&[(0, 1)]).unwrap();
    let expected = cotangent_view
        .contiguous(MemoryOrder::ColumnMajor)
        .reshape(&[2, 2])
        .unwrap();

    let x_tensor = Tensor::from(x.clone());
    let grads = grad(
        &[&reshaped],
        &[&x_tensor],
        Some(&[Tensor::from_tensor(cotangent_view)]),
        GradOptions::default(),
    )
    .unwrap();
    let grad_x = grads[0].as_ref().expect("missing reshape gradient");
    assert_eq!(
        tensor_to_vec_f64(grad_x.as_f64().unwrap().structured_primal()),
        tensor_to_vec_f64(&expected)
    );
}

#[test]
fn dense_reverse_pullback_accepts_structured_cotangent() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([2.0, 5.0, 7.0, 3.0]), &tape);
    let alpha = reverse_leaf_f64(scalar_f64(2.0), &tape);
    let x_tensor = Tensor::from(x.clone());
    let alpha_tensor = Tensor::from(alpha.clone());
    let y = x_tensor.scale(&alpha_tensor).unwrap();
    assert!(y.as_f64().unwrap().is_dense());

    let structured_cotangent = Tensor::from(diag_structured(
        DenseTensor::<f64>::from_slice(&[1.0, 1.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
        2,
    ));

    let grads = grad(
        &[&y],
        &[&x_tensor, &alpha_tensor],
        Some(&[structured_cotangent]),
        GradOptions::default(),
    )
    .unwrap();
    assert_eq!(
        tensor_to_vec_f64(
            grads[0]
                .as_ref()
                .unwrap()
                .as_f64()
                .unwrap()
                .structured_primal()
        ),
        vec![2.0, 2.0, 2.0, 2.0]
    );
    assert_eq!(
        tensor_to_vec_f64(
            grads[1]
                .as_ref()
                .unwrap()
                .as_f64()
                .unwrap()
                .structured_primal()
        ),
        vec![17.0]
    );
}

#[test]
fn pullback_helpers_preserve_none_for_untracked_wrt_inputs() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([1.0, 2.0, 3.0, 4.0]), &tape);
    let alpha = reverse_leaf_f64(scalar_f64(3.0), &tape);
    let x_tensor = Tensor::from(x.clone());
    let alpha_tensor = Tensor::from(alpha);
    let y = x_tensor.scale(&alpha_tensor).unwrap();
    let cotangent = Tensor::from_tensor(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let primal_tensor = AdTensor::new_primal(f64_2x2([9.0, 8.0, 7.0, 6.0]));
    let primal_scalar = AdTensor::new_primal(scalar_f64(1.0));
    let primal_tensor = Tensor::from(primal_tensor);
    let primal_scalar = Tensor::from(primal_scalar);
    let grads = grad(
        &[&y],
        &[&primal_tensor, &primal_scalar],
        Some(&[cotangent]),
        GradOptions::default(),
    )
    .unwrap();
    assert_eq!(grads.len(), 2);
    assert!(grads[0].is_none());
    assert!(grads[1].is_none());
}

#[test]
fn pullback_wrt_returns_none_for_disconnected_reverse_tensor() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let tape = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([1.0, 2.0, 3.0, 4.0]), &tape);
    let alpha = reverse_leaf_f64(scalar_f64(3.0), &tape);
    let disconnected = reverse_leaf_f64(f64_2x2([9.0, 8.0, 7.0, 6.0]), &tape);
    let x_tensor = Tensor::from(x);
    let alpha_tensor = Tensor::from(alpha);
    let y = x_tensor.scale(&alpha_tensor).unwrap();
    let cotangent = Tensor::from_tensor(f64_2x2([1.0, 0.0, 0.0, 1.0]));
    let disconnected = Tensor::from(disconnected);

    let grads = grad(
        &[&y],
        &[&disconnected],
        Some(&[cotangent]),
        GradOptions::default(),
    )
    .unwrap();
    assert_eq!(grads.len(), 1);
    assert!(grads[0].is_none());
}

#[test]
fn pullback_helpers_reject_primal_outputs_and_mixed_tapes() {
    let _guard = crate::set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));

    let primal_output = Tensor::from(AdTensor::new_primal(f64_2x2([1.0, 2.0, 3.0, 4.0])));
    let cotangent = Tensor::from(AdTensor::new_primal(f64_2x2([1.0, 0.0, 0.0, 1.0])));

    let err = grad(
        &[&primal_output],
        &[&primal_output],
        Some(&[cotangent.clone()]),
        GradOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        Error::InvalidAdTensor { message }
            if message.contains("reverse-mode output tensor")
    ));

    let tape_output = Tape::<crate::DynTensor>::new();
    let tape_wrt = Tape::<crate::DynTensor>::new();
    let x = reverse_leaf_f64(f64_2x2([4.0, 3.0, 2.0, 1.0]), &tape_output);
    let alpha = reverse_leaf_f64(scalar_f64(2.0), &tape_output);
    let x_tensor = Tensor::from(x);
    let alpha_tensor = Tensor::from(alpha);
    let output = x_tensor.scale(&alpha_tensor).unwrap();
    let wrt_tensor = reverse_leaf_f64(f64_2x2([1.0, 0.0, 0.0, 1.0]), &tape_wrt);
    let wrt_tensor = Tensor::from(wrt_tensor);
    let cotangent = Tensor::from_tensor(f64_2x2([1.0, 0.0, 0.0, 1.0]));

    let err = grad(
        &[&output],
        &[&wrt_tensor],
        Some(&[cotangent]),
        GradOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::MixedReverseTape { .. }));
}
