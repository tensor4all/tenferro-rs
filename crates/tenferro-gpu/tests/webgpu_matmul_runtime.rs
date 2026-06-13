#![cfg(feature = "webgpu")]

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_gpu::{webgpu_available, WebGpuBackend};
use tenferro_tensor::{DotGeneralConfig, Tensor, TensorDeviceTransfer, TensorDot};

fn matmul2_col_major(lhs: &[Complex32], rhs: &[Complex32]) -> [Complex32; 4] {
    let a00 = lhs[0];
    let a10 = lhs[1];
    let a01 = lhs[2];
    let a11 = lhs[3];
    let b00 = rhs[0];
    let b10 = rhs[1];
    let b01 = rhs[2];
    let b11 = rhs[3];
    [
        a00 * b00 + a01 * b10,
        a10 * b00 + a11 * b10,
        a00 * b01 + a01 * b11,
        a10 * b01 + a11 * b11,
    ]
}

fn assert_complex_close(actual: &[Complex32], expected: &[Complex32]) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (actual.re - expected.re)
                .abs()
                .max((actual.im - expected.im).abs())
        })
        .fold(0.0_f32, f32::max);
    assert!(
        max_error <= 1e-4,
        "max complex mismatch {max_error} exceeds tolerance\nactual={actual:?}\nexpected={expected:?}"
    );
}

fn assert_f32_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error <= 1e-4,
        "max f32 mismatch {max_error} exceeds tolerance\nactual={actual:?}\nexpected={expected:?}"
    );
}

fn dot_general_config_for_shapes(lhs_shape: &[usize], rhs_shape: &[usize]) -> DotGeneralConfig {
    if lhs_shape.len() == 3 && rhs_shape.len() == 3 {
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
        }
    } else {
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        }
    }
}

fn c32_values(len: usize, seed: f32) -> Vec<Complex32> {
    (0..len)
        .map(|index| {
            let x = seed + index as f32 * 0.375;
            Complex32::new(x, 0.25 - 0.5 * x)
        })
        .collect()
}

fn assert_c32_dot_general_with_conj_matches_cpu(
    backend: &mut WebGpuBackend,
    lhs_conj: bool,
    rhs_conj: bool,
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
) {
    let lhs_len = lhs_shape.iter().product();
    let rhs_len = rhs_shape.iter().product();
    let lhs = Tensor::from_vec_col_major(lhs_shape.clone(), c32_values(lhs_len, 0.25));
    let rhs = Tensor::from_vec_col_major(rhs_shape.clone(), c32_values(rhs_len, -0.75));
    let config = dot_general_config_for_shapes(&lhs_shape, &rhs_shape);

    let mut cpu = CpuBackend::new();
    let expected = cpu
        .dot_general_with_conj(&lhs, &rhs, &config, lhs_conj, rhs_conj)
        .unwrap();

    let gpu_lhs = backend.upload_host_tensor(&lhs).unwrap();
    let gpu_rhs = backend.upload_host_tensor(&rhs).unwrap();
    let gpu_out = backend
        .dot_general_with_conj(&gpu_lhs, &gpu_rhs, &config, lhs_conj, rhs_conj)
        .unwrap();
    let out = backend.download_to_host(&gpu_out).unwrap();

    assert_eq!(out.shape(), expected.shape());
    assert_complex_close(
        out.as_slice::<Complex32>().unwrap(),
        expected.as_slice::<Complex32>().unwrap(),
    );
}

fn noncontiguous_lhs_free_axes_reference(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; 2 * 2 * 2];
    for n in 0..2 {
        for m1 in 0..2 {
            for m0 in 0..2 {
                let mut acc = 0.0_f32;
                for k in 0..3 {
                    let lhs_offset = m0 + 2 * k + 2 * 3 * m1;
                    let rhs_offset = k + 3 * n;
                    acc += lhs[lhs_offset] * rhs[rhs_offset];
                }
                let out_offset = m0 + 2 * m1 + 2 * 2 * n;
                out[out_offset] = acc;
            }
        }
    }
    out
}

fn batched_matmul_c32_reference(lhs: &[Complex32], rhs: &[Complex32]) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); 2 * 2 * 2];
    for batch in 0..2 {
        for n in 0..2 {
            for m in 0..2 {
                let mut acc = Complex32::new(0.0, 0.0);
                for k in 0..3 {
                    let lhs_offset = m + 2 * k + 2 * 3 * batch;
                    let rhs_offset = k + 3 * n + 3 * 2 * batch;
                    acc += lhs[lhs_offset] * rhs[rhs_offset];
                }
                let out_offset = m + 2 * n + 2 * 2 * batch;
                out[out_offset] = acc;
            }
        }
    }
    out
}

#[test]
fn webgpu_c32_dot_general_with_lhs_conj_matches_cpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    assert_c32_dot_general_with_conj_matches_cpu(&mut backend, true, false, vec![2, 3], vec![3, 2]);
}

#[test]
fn webgpu_c32_dot_general_with_rhs_conj_matches_cpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    assert_c32_dot_general_with_conj_matches_cpu(&mut backend, false, true, vec![2, 3], vec![3, 2]);
}

#[test]
fn webgpu_c32_batched_dot_general_with_both_conj_matches_cpu_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    assert_c32_dot_general_with_conj_matches_cpu(
        &mut backend,
        true,
        true,
        vec![2, 3, 2],
        vec![3, 2, 2],
    );
}

#[test]
fn webgpu_f32_dot_general_with_conj_is_identity_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let mut cpu = CpuBackend::new();
    let expected = cpu
        .dot_general_with_conj(&lhs, &rhs, &config, true, true)
        .unwrap();

    let gpu_lhs = backend.upload_host_tensor(&lhs).unwrap();
    let gpu_rhs = backend.upload_host_tensor(&rhs).unwrap();
    let gpu_out = backend
        .dot_general_with_conj(&gpu_lhs, &gpu_rhs, &config, true, true)
        .unwrap();
    let out = backend.download_to_host(&gpu_out).unwrap();

    assert_eq!(out.shape(), expected.shape());
    assert_f32_close(
        out.as_slice::<f32>().unwrap(),
        expected.as_slice::<f32>().unwrap(),
    );
}

#[test]
fn webgpu_dot_general_runs_rank2_f32_matmul_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let lhs = Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0]);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let gpu_lhs = backend.upload_host_tensor(&lhs).unwrap();
    let gpu_rhs = backend.upload_host_tensor(&rhs).unwrap();
    let gpu_out = backend.dot_general(&gpu_lhs, &gpu_rhs, &config).unwrap();
    let out = backend.download_to_host(&gpu_out).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    let actual = out.as_slice::<f32>().unwrap();
    let expected = [58.0_f32, 139.0, 64.0, 154.0];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-4);
    }
}

#[test]
fn webgpu_dot_general_supports_batched_f32_contract_shape_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let lhs = Tensor::from_vec_col_major(
        vec![2, 3, 2],
        vec![
            1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0, 10.0, 40.0, 20.0, 50.0, 30.0, 60.0,
        ],
    );
    let rhs = Tensor::from_vec_col_major(
        vec![3, 2, 2],
        vec![
            7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0, 70.0, 90.0, 110.0, 80.0, 100.0, 120.0,
        ],
    );
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![2],
        rhs_batch_dims: vec![2],
    };

    let gpu_lhs = backend.upload_host_tensor(&lhs).unwrap();
    let gpu_rhs = backend.upload_host_tensor(&rhs).unwrap();
    let gpu_out = backend.dot_general(&gpu_lhs, &gpu_rhs, &config).unwrap();
    let out = backend.download_to_host(&gpu_out).unwrap();

    assert_eq!(out.shape(), &[2, 2, 2]);
    let actual = out.as_slice::<f32>().unwrap();
    let expected = [
        58.0_f32, 139.0, 64.0, 154.0, 5800.0, 13900.0, 6400.0, 15400.0,
    ];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-4);
    }
}

#[test]
fn webgpu_dot_general_packs_noncontiguous_lhs_free_axes_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let lhs_data = vec![
        1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0, 10.0, 40.0, 20.0, 50.0, 30.0, 60.0,
    ];
    let rhs_data = vec![7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0];
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 2], lhs_data.clone());
    let rhs = Tensor::from_vec_col_major(vec![3, 2], rhs_data.clone());
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let gpu_lhs = backend.upload_host_tensor(&lhs).unwrap();
    let gpu_rhs = backend.upload_host_tensor(&rhs).unwrap();
    let gpu_out = backend.dot_general(&gpu_lhs, &gpu_rhs, &config).unwrap();
    let out = backend.download_to_host(&gpu_out).unwrap();

    assert_eq!(out.shape(), &[2, 2, 2]);
    let actual = out.as_slice::<f32>().unwrap();
    let expected = noncontiguous_lhs_free_axes_reference(&lhs_data, &rhs_data);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-4);
    }
}

#[test]
fn webgpu_dot_general_supports_batched_c32_contract_shape_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let lhs_data = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(4.0, -1.0),
        Complex32::new(2.0, 0.25),
        Complex32::new(5.0, 1.0),
        Complex32::new(3.0, -0.75),
        Complex32::new(6.0, 0.5),
        Complex32::new(10.0, 0.5),
        Complex32::new(40.0, -1.0),
        Complex32::new(20.0, 0.25),
        Complex32::new(50.0, 1.0),
        Complex32::new(30.0, -0.75),
        Complex32::new(60.0, 0.5),
    ];
    let rhs_data = vec![
        Complex32::new(7.0, -0.5),
        Complex32::new(9.0, 0.25),
        Complex32::new(11.0, 1.0),
        Complex32::new(8.0, -0.75),
        Complex32::new(10.0, 0.5),
        Complex32::new(12.0, -0.25),
        Complex32::new(70.0, -0.5),
        Complex32::new(90.0, 0.25),
        Complex32::new(110.0, 1.0),
        Complex32::new(80.0, -0.75),
        Complex32::new(100.0, 0.5),
        Complex32::new(120.0, -0.25),
    ];
    let lhs = Tensor::from_vec_col_major(vec![2, 3, 2], lhs_data.clone());
    let rhs = Tensor::from_vec_col_major(vec![3, 2, 2], rhs_data.clone());
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![2],
        rhs_batch_dims: vec![2],
    };

    let gpu_lhs = backend.upload_host_tensor(&lhs).unwrap();
    let gpu_rhs = backend.upload_host_tensor(&rhs).unwrap();
    let gpu_out = backend.dot_general(&gpu_lhs, &gpu_rhs, &config).unwrap();
    let out = backend.download_to_host(&gpu_out).unwrap();

    assert_eq!(out.shape(), &[2, 2, 2]);
    let actual = out.as_slice::<Complex32>().unwrap();
    let expected = batched_matmul_c32_reference(&lhs_data, &rhs_data);
    assert_complex_close(actual, &expected);
}

#[test]
fn webgpu_dot_general_rejects_f64_and_c64_without_cpu_fallback_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let lhs_f64 = Tensor::from_vec_col_major(vec![1, 1], vec![1.0_f64]);
    let rhs_f64 = Tensor::from_vec_col_major(vec![1, 1], vec![2.0_f64]);
    let lhs_f64 = backend.upload_host_tensor(&lhs_f64).unwrap();
    let rhs_f64 = backend.upload_host_tensor(&rhs_f64).unwrap();
    let err = backend
        .dot_general(&lhs_f64, &rhs_f64, &config)
        .expect_err("f64 WebGPU dot_general must stay unsupported");
    assert!(
        err.to_string().contains("WebGPU"),
        "unexpected f64 error: {err}"
    );

    let lhs_c64 = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(1.0, 0.5)]);
    let rhs_c64 = Tensor::from_vec_col_major(vec![1, 1], vec![Complex64::new(2.0, -0.25)]);
    let lhs_c64 = backend.upload_host_tensor(&lhs_c64).unwrap();
    let rhs_c64 = backend.upload_host_tensor(&rhs_c64).unwrap();
    let err = backend
        .dot_general(&lhs_c64, &rhs_c64, &config)
        .expect_err("c64 WebGPU dot_general must stay unsupported");
    assert!(
        err.to_string().contains("WebGPU"),
        "unexpected c64 error: {err}"
    );
}

#[test]
fn webgpu_dot_general_runs_rank2_c32_matmul_when_adapter_available() {
    if !webgpu_available() {
        return;
    }

    let mut backend = WebGpuBackend::new_default().unwrap();
    let lhs_data = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(2.0, -1.0),
        Complex32::new(3.0, 0.25),
        Complex32::new(4.0, 1.0),
    ];
    let rhs_data = vec![
        Complex32::new(5.0, -0.5),
        Complex32::new(6.0, 0.25),
        Complex32::new(7.0, 1.0),
        Complex32::new(8.0, -0.75),
    ];
    let lhs = Tensor::from_vec_col_major(vec![2, 2], lhs_data.clone());
    let rhs = Tensor::from_vec_col_major(vec![2, 2], rhs_data.clone());
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let gpu_lhs = backend.upload_host_tensor(&lhs).unwrap();
    let gpu_rhs = backend.upload_host_tensor(&rhs).unwrap();
    let gpu_out = backend.dot_general(&gpu_lhs, &gpu_rhs, &config).unwrap();
    let out = backend.download_to_host(&gpu_out).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    let actual = out.as_slice::<Complex32>().unwrap();
    let expected = matmul2_col_major(&lhs_data, &rhs_data);
    assert_complex_close(actual, &expected);
}
