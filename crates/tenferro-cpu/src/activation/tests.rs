use super::*;

/// Scalar f64 definitions, as `tenferro_runtime::composite` states them.
fn reference(op: ActivationOp, x: f64) -> f64 {
    let sigmoid = 1.0 / (1.0 + (-x).exp());
    match op {
        ActivationOp::Sigmoid => sigmoid,
        ActivationOp::Silu => x * sigmoid,
        ActivationOp::Softplus => x.max(0.0) + (-x.abs()).exp().ln_1p(),
        ActivationOp::Gelu => 0.5 * x * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2)),
        ActivationOp::GeluTanh => {
            let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
    }
}

const OPS: [ActivationOp; 5] = [
    ActivationOp::Sigmoid,
    ActivationOp::Silu,
    ActivationOp::Softplus,
    ActivationOp::Gelu,
    ActivationOp::GeluTanh,
];

#[test]
fn every_op_matches_the_scalar_definition_in_both_dtypes() {
    // Enough elements for several parallel tasks and a ragged tail.
    let n = 3 * CHUNK + 37;
    let xs: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) * 16.0 - 8.0).collect();
    let mut pool = BufferPool::new();
    for op in OPS {
        let input = Tensor::from_vec_col_major(vec![n], xs.clone()).unwrap();
        let out = fused_activation_read_with_pool(&mut pool, op, TensorRead::from_tensor(&input))
            .unwrap()
            .expect("f64 has a fused kernel");
        for (&got, &x) in out.as_slice::<f64>().unwrap().iter().zip(&xs) {
            let want = reference(op, x);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "{op:?} f64 at {x}: got {got}, want {want}"
            );
        }
        let input32 =
            Tensor::from_vec_col_major(vec![n], xs.iter().map(|&v| v as f32).collect()).unwrap();
        let out = fused_activation_read_with_pool(&mut pool, op, TensorRead::from_tensor(&input32))
            .unwrap()
            .expect("f32 has a fused kernel");
        for (&got, &x) in out.as_slice::<f32>().unwrap().iter().zip(&xs) {
            let want = reference(op, x as f32 as f64);
            assert!(
                (got as f64 - want).abs() <= 2e-6 * want.abs().max(1.0),
                "{op:?} f32 at {x}: got {got}, want {want}"
            );
        }
    }
}

#[test]
fn strided_input_is_materialized_once_and_integers_decline() {
    let mut pool = BufferPool::new();
    let base = tenferro_tensor::TypedTensor::<f32>::from_vec_col_major(
        vec![4, 3],
        (0..12).map(|i| i as f32 - 6.0).collect(),
    )
    .unwrap();
    let view = tenferro_tensor::TensorView::F32(base.as_view().transpose_view([1, 0]).unwrap());
    let out = fused_activation_read_with_pool(
        &mut pool,
        ActivationOp::Sigmoid,
        TensorRead::from_view(view),
    )
    .unwrap()
    .expect("f32 view");
    assert_eq!(out.shape(), &[3, 4]);
    let got = out.as_slice::<f32>().unwrap();
    for j in 0..4 {
        for i in 0..3 {
            let x = (i * 4 + j) as f32 - 6.0;
            let want = 1.0 / (1.0 + (-x).exp());
            assert!((got[i + j * 3] - want).abs() < 1e-6);
        }
    }
    let ints = Tensor::from_vec_col_major(vec![2], vec![1i32, 2]).unwrap();
    assert!(fused_activation_read_with_pool(
        &mut pool,
        ActivationOp::Silu,
        TensorRead::from_tensor(&ints)
    )
    .unwrap()
    .is_none());
}
