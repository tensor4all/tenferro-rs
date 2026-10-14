//! AD tests over placeholder-input graphs.
//!
//! Demonstrates that `grad` built over an `input_concrete_shape` placeholder
//! as well as `input_symbolic_shape` placeholders produces graphs that can be
//! evaluated via `run_with_inputs`.
//!
//! Previously the `input_symbolic_shape` variants were ignored because the
//! VJP code materialised zero cotangents with a concrete `vec![0; rank]`
//! shape when `wrt.shape_hint` was `None`. That has been replaced with a
//! deferred zero-tangent approach: the zero tensor is synthesised at
//! `run_with_inputs` time once the caller supplies the concrete binding.
//! See the deferred zero-tangent policy in `docs/spec/extension-op.md` and the
//! symbolic-shape tests in this file.

use crate::support;
use support::{cpu_runtime, RunTraced};
use tenferro_ad::TracedTensorAdExt;
use tenferro_runtime::{Tensor, TracedTensor};
use tenferro_tensor::DType;

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64 tensor")
}

#[test]
fn grad_of_sum_of_squares_against_symbolic_input() {
    // f(x) = sum(x * x), df/dx = 2 * x
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let sq = (&x * &x).unwrap();
    let loss = sq.reduce_sum(Some(&[0])).unwrap();

    let g = loss.grad(&x).expect("grad build");

    let bound = Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let engine = cpu_runtime();
    let grad_out = g
        .run_with_inputs_auto(&engine, &[(&x, &bound)])
        .expect("grad eval");

    assert_eq!(grad_out.shape(), &[4]);
    assert_eq!(f64_data(&grad_out), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn grad_evaluates_with_different_shapes_from_same_symbolic_graph() {
    // Same symbolic grad graph, eval with shape [3] then shape [5].
    let x = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let sq = (&x * &x).unwrap();
    let loss = sq.reduce_sum(Some(&[0])).unwrap();
    let g_template = loss.grad(&x).expect("grad build");

    let engine = cpu_runtime();

    let g1 = g_template.clone();
    let b1 = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let out1 = g1
        .run_with_inputs_auto(&engine, &[(&x, &b1)])
        .expect("eval1");
    assert_eq!(f64_data(&out1), &[2.0, 4.0, 6.0]);

    let g2 = g_template.clone();
    let b2 = Tensor::from_vec_col_major(vec![5], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let out2 = g2
        .run_with_inputs_auto(&engine, &[(&x, &b2)])
        .expect("eval2");
    assert_eq!(f64_data(&out2), &[2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn grad_of_dot_product_against_two_symbolic_inputs() {
    // f(a, b) = sum(a * b)
    // df/da = b, df/db = a
    let a = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let b = TracedTensor::input_symbolic_shape(DType::F64, 1).unwrap();
    let loss = (&a * &b).unwrap().reduce_sum(Some(&[0])).unwrap();

    let grad_a = loss.grad(&a).expect("grad a");
    let grad_b = loss.grad(&b).expect("grad b");

    let engine = cpu_runtime();
    let ta = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let tb = Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap();

    let out_a = grad_a
        .run_with_inputs_auto(&engine, &[(&a, &ta), (&b, &tb)])
        .expect("eval grad a");
    assert_eq!(f64_data(&out_a), &[10.0, 20.0, 30.0]);

    let out_b = grad_b
        .run_with_inputs_auto(&engine, &[(&a, &ta), (&b, &tb)])
        .expect("eval grad b");
    assert_eq!(f64_data(&out_b), &[1.0, 2.0, 3.0]);
}

#[test]
fn grad_with_concrete_shape_placeholder() {
    // input_concrete_shape placeholder works for AD too.
    let x = TracedTensor::input_concrete_shape(DType::F64, &[3]).unwrap();
    let sq = (&x * &x).unwrap();
    let loss = sq.reduce_sum(Some(&[0])).unwrap();
    let g = loss.grad(&x).expect("grad build");

    let engine = cpu_runtime();
    let bound = Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let out = g
        .run_with_inputs_auto(&engine, &[(&x, &bound)])
        .expect("eval grad");

    assert_eq!(f64_data(&out), &[2.0, 4.0, 6.0]);
}

#[test]
fn grad_embed_diag_shifted_axis_restores_input_layout() {
    // Shape alone is not enough: a permutation between equal-sized axes is
    // invisible to it, and a uniform `reduce_sum` cotangent is invariant under
    // every permutation. First evaluate the VJP with a distinct cotangent `w`
    // and assert each entry. For `embed_diag(1, 0)` on `x: [m, n]` the primal
    // writes `x[i, j]` to `out[j, i, j]` (output shape `[n, m, n]`), so
    // `d(sum(w * embed_diag(x))) / d x[i, j] = w[j, i, j]`.
    fn assert_shifted_axis_vjp(m: usize, n: usize) {
        let w_data: Vec<f64> = (0..n * m * n).map(|k| k as f64 + 1.0).collect();

        let x = TracedTensor::input_concrete_shape(DType::F64, &[m, n]).unwrap();
        let w = TracedTensor::from_vec_col_major(vec![n, m, n], w_data.clone()).unwrap();
        let loss = (&x.embed_diag(1, 0).unwrap() * &w)
            .unwrap()
            .reduce_sum(Some(&[0, 1, 2]))
            .unwrap();
        let grad = loss.grad(&x).unwrap();
        assert_eq!(grad.try_concrete_shape(), Some(vec![m, n]));

        // The adjoint of a linear map is the (constant) cotangent push-through,
        // so the gradient graph does not depend on the primal and takes no
        // bindings.
        let engine = cpu_runtime();
        let out = grad.run_with_inputs_auto(&engine, &[]).expect("grad eval");
        assert_eq!(out.shape(), &[m, n]);

        // `w` is col-major `[n, m, n]`, so `w[j, i, j]` lives at
        // `j + n * i + n * m * j` and the input-layout gradient is col-major
        // `[m, n]` at `i + m * j`.
        let mut expected = vec![0.0_f64; m * n];
        for i in 0..m {
            for j in 0..n {
                expected[i + m * j] = w_data[j + n * i + n * m * j];
            }
        }
        assert_eq!(f64_data(&out), expected.as_slice());
    }

    // Equal-sized axes: restoring the trailing `Transpose` is shape-preserving
    // here, so only the values expose it - exactly the gap review found.
    assert_shifted_axis_vjp(3, 3);
    assert_shifted_axis_vjp(2, 3);

    // Regression for #2107: the adjoint of `EmbedDiag { axis_a: 1, axis_b: 0 }`
    // must land back in the `[2, 3]` input layout, not a transposed `[3, 2]`.
    // The spurious trailing `Transpose` made the VJP metadata `[3, 2]`.
    let x = TracedTensor::input_concrete_shape(DType::F64, &[2, 3]).unwrap();
    let embedded = x.embed_diag(1, 0).unwrap();
    let loss = embedded.reduce_sum(Some(&[0, 1, 2])).unwrap();
    let grad = loss.grad(&x).unwrap();
    assert_eq!(grad.try_concrete_shape(), Some(vec![2, 3]));

    // Higher rank: the reported failure mode permuted the trailing axes.
    let cube = TracedTensor::input_concrete_shape(DType::F64, &[2, 3, 4]).unwrap();
    let cube_loss = cube
        .embed_diag(1, 0)
        .unwrap()
        .reduce_sum(Some(&[0, 1, 2, 3]))
        .unwrap();
    let cube_grad = cube_loss.grad(&cube).unwrap();
    assert_eq!(cube_grad.try_concrete_shape(), Some(vec![2, 3, 4]));

    // The other branches (`axis_b >= axis_a`, including `axis_b == axis_a`)
    // never emitted a transpose and must keep restoring the input layout.
    for (axis_a, axis_b, loss_rank) in [(0, 1, 3usize), (1, 1, 3)] {
        let y = x.embed_diag(axis_a, axis_b).unwrap();
        let loss = y
            .reduce_sum(Some(&(0..loss_rank).collect::<Vec<_>>()))
            .unwrap();
        let grad = loss.grad(&x).unwrap();
        assert_eq!(grad.try_concrete_shape(), Some(vec![2, 3]));
    }
}
