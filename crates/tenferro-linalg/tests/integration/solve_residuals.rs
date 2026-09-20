#![cfg(feature = "autodiff")]

use num_complex::Complex64;
use std::sync::Arc;
use tenferro_ad::{AdContext, EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::EagerTensorLinalgExt;

fn context() -> Arc<EagerRuntime> {
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    EagerRuntime::with_cpu_backend_and_ad_context(CpuBackend::with_threads(1).unwrap(), &ad)
        .unwrap()
}

fn values(tensor: &EagerTensor, complex: bool) -> Vec<Complex64> {
    let value = tensor.value().unwrap();
    if complex {
        value.as_slice::<Complex64>().unwrap().to_vec()
    } else {
        value
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect()
    }
}

fn close(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        error < tolerance,
        "max absolute error {error}: {actual:?} != {expected:?}"
    );
}

#[test]
fn solve_saved_lu_preserves_real_complex_batched_higher_derivatives() {
    for complex in [false, true] {
        for batched in [false, true] {
            let ctx = context();
            let shape = if batched { vec![2, 2, 2] } else { vec![2, 2] };
            let data = |real: &[f64], imag: &[f64]| {
                real.iter()
                    .zip(imag)
                    .map(|(&re, &im)| Complex64::new(re, if complex { im } else { 0.0 }))
                    .cycle()
                    .take(if batched { 8 } else { 4 })
                    .collect::<Vec<_>>()
            };
            // The first column requires a pivot; perturbations do not cross a pivot boundary.
            let av = data(&[0.2, 3.0, 2.0, 4.0], &[0.1, -0.4, 0.3, -0.1]);
            let bv = data(&[1.0, 2.0, -1.0, 0.5], &[0.5, -0.3, 0.2, -0.1]);
            let dv = data(&[0.1, -0.2, 0.3, 0.4], &[-0.2, 0.1, 0.05, -0.1]);
            let input = |data: Vec<Complex64>| {
                let tensor = if complex {
                    Tensor::from_vec_col_major(shape.clone(), data).unwrap()
                } else {
                    Tensor::from_vec_col_major(
                        shape.clone(),
                        data.iter().map(|x| x.re).collect::<Vec<_>>(),
                    )
                    .unwrap()
                };
                EagerTensor::requires_grad_in(tensor, Arc::clone(&ctx)).unwrap()
            };
            let a = input(av.clone());
            let b = input(bv.clone());
            let direction = input(dv.clone()).detach();
            let axes: Vec<_> = (0..shape.len()).collect();
            let solution = a.solve(&b).unwrap();
            let loss = solution.reduce_sum(Some(&axes)).unwrap();
            drop(solution); // LU/pivots and X must survive without user handles.
            let ga = ctx.grad(&loss, &a).unwrap();
            let gb = ctx.grad(&loss, &b).unwrap();
            let haa = ctx.jvp(&ga, &a, &direction).unwrap();
            let hba = ctx.jvp(&gb, &a, &direction).unwrap();
            let first_at = |sign: f64| {
                let a = input(
                    av.iter()
                        .zip(&dv)
                        .map(|(a, d)| a + sign * 1e-5 * d)
                        .collect(),
                );
                let b = input(bv.clone());
                let loss = a.solve(&b).unwrap().reduce_sum(Some(&axes)).unwrap();
                (
                    values(&ctx.grad(&loss, &a).unwrap(), complex),
                    values(&ctx.grad(&loss, &b).unwrap(), complex),
                )
            };
            let plus = first_at(1.0);
            let minus = first_at(-1.0);
            let fd = |p: &[Complex64], m: &[Complex64]| {
                p.iter()
                    .zip(m)
                    .map(|(p, m)| (p - m) / 2e-5)
                    .collect::<Vec<_>>()
            };
            close(&values(&haa, complex), &fd(&plus.0, &minus.0), 2e-7);
            close(&values(&hba, complex), &fd(&plus.1, &minus.1), 2e-7);
            // Real Hessian symmetry: reverse-over-reverse must match the JVP,
            // including conjugation for a real loss on complex inputs.
            let directional = ga
                .mul(&direction.conj().unwrap())
                .unwrap()
                .reduce_sum(Some(&axes))
                .unwrap();
            close(
                &values(&ctx.grad(&directional, &a).unwrap(), complex),
                &values(&haa, complex),
                2e-10,
            );
            close(
                &values(&ctx.grad(&directional, &b).unwrap(), complex),
                &values(&hba, complex),
                2e-10,
            );
        }
    }
}

#[test]
fn solve_handles_single_and_multiple_tracked_inputs_and_no_grad() {
    for (track_a, track_b) in [(false, false), (true, false), (false, true), (true, true)] {
        let ctx = context();
        let make = |shape, data, tracked| {
            let tensor = Tensor::from_vec_col_major(shape, data).unwrap();
            if tracked {
                EagerTensor::requires_grad_in(tensor, Arc::clone(&ctx)).unwrap()
            } else {
                EagerTensor::from_tensor_in(tensor, Arc::clone(&ctx)).unwrap()
            }
        };
        let a = make(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 3.0], track_a);
        let b = make(vec![2, 1], vec![4.0_f64, 9.0], track_b);
        let solution = a.solve(&b).unwrap();
        close(&values(&solution, false), &[2.0.into(), 3.0.into()], 1e-12);
        if track_a || track_b {
            let loss = solution.reduce_sum(Some(&[0, 1])).unwrap();
            drop(solution);
            let gradients = loss.backward().unwrap();
            assert_eq!(gradients.len(), usize::from(track_a) + usize::from(track_b));
            if track_a {
                assert_eq!(
                    a.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
                    &[-1.0, -2.0 / 3.0, -1.5, -1.0]
                );
            }
            if track_b {
                assert_eq!(
                    b.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
                    &[0.5, 1.0 / 3.0]
                );
            }
        }
        let _guard = ctx.no_grad();
        assert!(!a.solve(&b).unwrap().tracks_grad());
    }
}

#[test]
fn solve_validates_both_operands_before_factorization() {
    let ctx = context();
    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f64; 4]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    let bad_b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 1], vec![1.0_f64; 3]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    assert!(matches!(
        a.solve(&bad_b),
        Err(tenferro_ad::Error::TensorRuntime(
            tenferro_tensor::Error::Validation { .. }
        ))
    ));
    for invalid in [
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64; 2]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 1, 2], vec![1.0_f64; 4]).unwrap(),
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f32; 2]).unwrap(),
    ] {
        let invalid = EagerTensor::from_tensor_in(invalid, Arc::clone(&ctx)).unwrap();
        assert!(matches!(
            a.solve(&invalid),
            Err(tenferro_ad::Error::TensorRuntime(
                tenferro_tensor::Error::Validation { .. }
            ))
        ));
    }
    let empty = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 0], Vec::<f64>::new()).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    assert_eq!(a.solve(&empty).unwrap().shape(), &[2, 0]);
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64; 2]).unwrap(),
        Arc::clone(&ctx),
    )
    .unwrap();
    assert!(
        a.solve(&b).is_err(),
        "singular valid-shape system must still fail"
    );
    let other = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![1.0_f64; 2]).unwrap(),
        context(),
    )
    .unwrap();
    assert!(matches!(
        a.solve(&other),
        Err(tenferro_ad::Error::ContextMismatch { .. })
    ));
}
