#![cfg(feature = "pjrt")]

use std::sync::{Mutex, MutexGuard, OnceLock};

use tenferro_einsum::GraphCompilerEinsumExt;
use tenferro_runtime::{DType, GraphCompiler, TracedTensor};
use tenferro_tensor::Tensor;
use tenferro_xla::{XlaExecutor, TENFERRO_PJRT_PLUGIN_ENV};

#[test]
fn pjrt_executes_phase_one_elementwise_from_rust_when_configured() {
    let Some((_guard, executor)) = configured_executor() else {
        return;
    };

    let x = TracedTensor::input_symbolic_shape(DType::F32, 1).unwrap();
    let positive = x.abs().unwrap().exp().unwrap();
    let analytic = positive
        .log()
        .unwrap()
        .sqrt()
        .unwrap()
        .rsqrt()
        .unwrap()
        .expm1()
        .unwrap()
        .log1p()
        .unwrap();
    let trig = positive.sin().unwrap().cos().unwrap().tanh().unwrap();
    let combined = (&analytic + &trig).unwrap();
    let divided = combined.div(&positive).unwrap();
    let y = divided.abs().unwrap().pow(&positive).unwrap();

    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&y, &[(&x, DType::F32, &[4])])
        .unwrap();
    let input_values = vec![-1.0_f32, 0.25, 0.5, 1.25];
    let input = Tensor::from_vec_col_major(vec![4], input_values.clone()).unwrap();
    let output = executor.run_with_inputs(&program, &[&input]).unwrap();

    let expected = input_values
        .into_iter()
        .map(|x| {
            let positive = x.abs().exp();
            let analytic = positive.ln().sqrt().sqrt().recip().exp_m1().ln_1p();
            let trig = positive.sin().cos().tanh();
            ((analytic + trig) / positive).abs().powf(positive)
        })
        .collect::<Vec<_>>();
    assert_close_f32(output.as_slice::<f32>().unwrap(), &expected, 1.0e-4);
}

#[test]
fn pjrt_executes_nary_einsum_from_rust_when_configured() {
    let Some((_guard, executor)) = configured_executor() else {
        return;
    };

    let lhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let mid = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let mut compiler = GraphCompiler::new();
    let product = compiler
        .einsum(&[&lhs, &mid, &rhs], "ij,jk,kl->il")
        .unwrap();
    let program = compiler
        .compile_with_input_specs(
            &product,
            &[
                (&lhs, DType::F32, &[2, 3]),
                (&mid, DType::F32, &[3, 4]),
                (&rhs, DType::F32, &[4, 2]),
            ],
        )
        .unwrap();

    let lhs_values = vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    let mid_values = vec![
        1.0_f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let rhs_values = vec![1.0_f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let lhs_input = Tensor::from_vec_col_major(vec![2, 3], lhs_values.clone()).unwrap();
    let mid_input = Tensor::from_vec_col_major(vec![3, 4], mid_values.clone()).unwrap();
    let rhs_input = Tensor::from_vec_col_major(vec![4, 2], rhs_values.clone()).unwrap();

    let output = executor
        .run_with_inputs(&program, &[&lhs_input, &mid_input, &rhs_input])
        .unwrap();
    let expected = expected_ij_jk_kl_to_il(&lhs_values, &mid_values, &rhs_values);
    assert_eq!(output.shape(), &[2, 2]);
    assert_close_f32(output.as_slice::<f32>().unwrap(), &expected, 1.0e-4);
}

#[test]
fn pjrt_executes_nary_einsum_plus_elementwise_from_rust_when_configured() {
    let Some((_guard, executor)) = configured_executor() else {
        return;
    };

    let lhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let mid = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let rhs = TracedTensor::input_symbolic_shape(DType::F32, 2).unwrap();
    let mut compiler = GraphCompiler::new();
    let product = compiler
        .einsum(&[&lhs, &mid, &rhs], "ij,jk,kl->il")
        .unwrap();
    let y = product
        .abs()
        .unwrap()
        .sqrt()
        .unwrap()
        .log1p()
        .unwrap()
        .exp()
        .unwrap();
    let program = compiler
        .compile_with_input_specs(
            &y,
            &[
                (&lhs, DType::F32, &[2, 3]),
                (&mid, DType::F32, &[3, 4]),
                (&rhs, DType::F32, &[4, 2]),
            ],
        )
        .unwrap();

    let lhs_values = vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0];
    let mid_values = vec![
        1.0_f32, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let rhs_values = vec![1.0_f32, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0];
    let lhs_input = Tensor::from_vec_col_major(vec![2, 3], lhs_values.clone()).unwrap();
    let mid_input = Tensor::from_vec_col_major(vec![3, 4], mid_values.clone()).unwrap();
    let rhs_input = Tensor::from_vec_col_major(vec![4, 2], rhs_values.clone()).unwrap();

    let output = executor
        .run_with_inputs(&program, &[&lhs_input, &mid_input, &rhs_input])
        .unwrap();
    let expected = expected_ij_jk_kl_to_il(&lhs_values, &mid_values, &rhs_values)
        .into_iter()
        .map(|value| value.abs().sqrt().ln_1p().exp())
        .collect::<Vec<_>>();
    assert_eq!(output.shape(), &[2, 2]);
    assert_close_f32(output.as_slice::<f32>().unwrap(), &expected, 1.0e-3);
}

fn configured_executor() -> Option<(MutexGuard<'static, ()>, XlaExecutor)> {
    let guard = pjrt_lock();
    if std::env::var_os(TENFERRO_PJRT_PLUGIN_ENV).is_none() {
        eprintln!("skipping PJRT execution check; set {TENFERRO_PJRT_PLUGIN_ENV}");
        return None;
    }
    Some((guard, XlaExecutor::from_env().unwrap()))
}

fn pjrt_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn expected_ij_jk_kl_to_il(lhs: &[f32], mid: &[f32], rhs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; 4];
    for i in 0..2 {
        for l in 0..2 {
            let mut sum = 0.0_f32;
            for j in 0..3 {
                for k in 0..4 {
                    let lhs_ij = lhs[i + 2 * j];
                    let mid_jk = mid[j + 3 * k];
                    let rhs_kl = rhs[k + 4 * l];
                    sum += lhs_ij * mid_jk * rhs_kl;
                }
            }
            out[i + 2 * l] = sum;
        }
    }
    out
}

fn assert_close_f32(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let residual = (actual - expected).abs();
        assert!(
            residual <= tolerance,
            "value {index} differs: actual={actual}, expected={expected}, residual={residual}"
        );
    }
}
