use super::*;
use tenferro_linalg::TracedTensorLinalgExt;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TracedTensor};

fn complex_input(m: usize, n: usize, batches: usize) -> Tensor {
    tensor_c64(
        vec![m, n, batches],
        (0..m * n * batches)
            .map(|i| {
                let x = i as f64;
                Complex64::new(
                    (0.71 * x).sin() + if i % (m + 1) == 0 { 2.0 } else { 0.0 },
                    (0.37 * x + 0.2).cos(),
                )
            })
            .collect(),
    )
}

fn check_factors(input: &Tensor, factors: &[Tensor]) {
    let (m, n) = (input.shape()[0], input.shape()[1]);
    let k = m.min(n);
    let a = input.as_slice::<Complex64>().unwrap();
    let u = factors[0].as_slice::<Complex64>().unwrap();
    let s = factors[1].as_slice::<f64>().unwrap();
    let vt = factors[2].as_slice::<Complex64>().unwrap();
    assert_eq!(&factors[0].shape()[..2], &[m, k]);
    assert_eq!(&factors[2].shape()[..2], &[k, n]);
    for b in 0..a.len() / (m * n) {
        let u = &u[b * m * k..(b + 1) * m * k];
        let s = &s[b * k..(b + 1) * k];
        let vt = &vt[b * k * n..(b + 1) * k * n];
        let mut us = u.to_vec();
        for col in 0..k {
            for row in 0..m {
                us[row + m * col] *= s[col];
            }
        }
        let reconstructed = matmul_c64(&us, vt, m, k, n);
        assert_relative_error_c64(&reconstructed, &a[b * m * n..(b + 1) * m * n], 1e-12);
        for i in 0..k {
            for j in 0..k {
                let gram_u: Complex64 = (0..m).map(|r| u[r + m * i].conj() * u[r + m * j]).sum();
                let gram_v: Complex64 = (0..n).map(|c| vt[i + k * c] * vt[j + k * c].conj()).sum();
                let expected = Complex64::new(f64::from(i == j), 0.0);
                assert!(
                    (gram_u - expected).norm() < 1e-12,
                    "U orthogonality {i},{j}: {gram_u}"
                );
                assert!(
                    (gram_v - expected).norm() < 1e-12,
                    "V orthogonality {i},{j}: {gram_v}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA and cuSOLVER Xgesvdp"]
fn test_cubecl_svd_xgesvdp_rectangular_batched_and_values() {
    for (m, n) in [(7, 3), (3, 7), (5, 5)] {
        check_forced_svd_driver(m, n, SvdDriver::Xgesvdp);
        // Different matrices across eight outstanding calls stress host scratch
        // isolation as well as device offsets in the shared batch loop.
        let input = complex_input(m, n, 8);
        let mut gpu = gpu_backend();
        let device = upload(&gpu, &input);
        let options = SvdOptions::default().driver(SvdDriver::Xgesvdp);
        let outputs = with_cuda_linalg_session(&mut gpu, |s| {
            s.svd_with_options_read(TensorRead::from_tensor(&device), options)
        })
        .unwrap();
        let factors: Vec<_> = outputs.iter().map(|x| download(&gpu, x)).collect();
        check_factors(&input, &factors);
        let baseline = with_cuda_linalg_session(&mut gpu, |s| {
            s.svd_with_options(&device, SvdOptions::default().driver(SvdDriver::Gesvd))
        })
        .unwrap();
        assert_tensor_close(&factors[1], &download(&gpu, &baseline[1]), 1e-12);
        let values = with_cuda_linalg_session(&mut gpu, |s| {
            s.svd_values_with_driver_read(TensorRead::from_tensor(&device), SvdDriver::Xgesvdp)
        })
        .unwrap();
        assert_tensor_close(&download(&gpu, &values), &factors[1], 1e-12);
        assert_tensor_close(&download(&gpu, &device), &input, 0.0);
    }
}

#[test]
#[ignore = "requires CUDA and cuSOLVER Xgesvdp"]
fn test_cubecl_svd_xgesvdp_traced_and_pruned() {
    use tenferro_gpu::cuda::cuda_runtime_engine_registration;
    let input = complex_input(3, 7, 1);
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &input);
    let options = SvdOptions::default().driver(SvdDriver::Xgesvdp);
    let expected =
        with_cuda_linalg_session(&mut gpu, |s| s.svd_with_options(&device, options)).unwrap();
    let traced = TracedTensor::from_tensor_concrete_shape(complex_input(3, 7, 1)).unwrap();
    let (u, s, vt) = traced.svd_with_options(options).unwrap();
    // A separate identity forces an S-only op instead of reusing the full SVD.
    let (_, only_s, _) = traced
        .svd_with_options(options.derivative_eps(2e-12))
        .unwrap();
    let (_, qr_s, _) = traced
        .svd_with_options(SvdOptions::default().driver(SvdDriver::Gesvd))
        .unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&u, &s, &vt, &only_s, &qr_s])
        .unwrap();
    let engine = gpu.runtime_engine_id().clone();
    let mut builder = Runtime::builder();
    builder
        .register_engine(cuda_runtime_engine_registration(&gpu).unwrap())
        .unwrap();
    builder
        .install_extension_module(tenferro_linalg::extension_module::<CudaEngine>(engine).unwrap())
        .unwrap();
    let runtime = builder.build().unwrap();
    let reads = [TensorRead::from_tensor(&device)];
    let prepared = runtime.prepare(&program, &reads, &[gpu.target()]).unwrap();
    let outputs = {
        let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut gpu];
        runtime.run(&mut engines, &prepared, &reads).unwrap()
    };
    let owned: Vec<_> = outputs
        .into_iter()
        .map(|output| output.into_tensor(&mut gpu).unwrap())
        .collect();
    let host: Vec<_> = owned.iter().map(|x| download(&gpu, x)).collect();
    check_factors(&input, &host[..3]);
    for (actual, expected) in host[..3].iter().zip(&expected) {
        assert_tensor_close(actual, &download(&gpu, expected), 0.0);
    }
    assert_tensor_close(&host[3], &host[1], 1e-12);
    assert_tensor_close(&host[4], &host[1], 1e-12);
}

#[cfg(feature = "autodiff")]
#[test]
#[ignore = "requires CUDA and cuSOLVER Xgesvdp"]
fn test_cubecl_svd_xgesvdp_eager() {
    use tenferro_ad::EagerRuntime;
    use tenferro_linalg::EagerSessionLinalgExt;
    let input = complex_input(7, 3, 1);
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &input);
    let options = SvdOptions::default().driver(SvdDriver::Xgesvdp);
    let expected =
        with_cuda_linalg_session(&mut gpu, |s| s.svd_with_options(&device, options)).unwrap();
    let runtime = EagerRuntime::new().unwrap();
    let eager = runtime
        .with_eager_session(&mut gpu, |s| s.constant_from(device))
        .unwrap();
    let (u, s, vt) = runtime
        .with_eager_session(&mut gpu, |session| {
            session.svd_with_options(&eager, options)
        })
        .unwrap();
    let factors: Vec<_> = [&u, &s, &vt]
        .into_iter()
        .map(|x| {
            let resident = x.to_tensor(&mut gpu).unwrap();
            download(&gpu, &resident)
        })
        .collect();
    check_factors(&input, &factors);
    for (actual, expected) in factors.iter().zip(&expected) {
        assert_tensor_close(actual, &download(&gpu, expected), 0.0);
    }
}

#[test]
#[ignore = "requires CUDA; issue #1849 acceptance sizes"]
fn test_cubecl_svd_xgesvdp_large_spectrum_agrees_with_gesvd() {
    let mut gpu = gpu_backend();
    for m in [400, 800, 1024] {
        // A = (I - 2wwᴴ) diag(s) (I - 2zzᴴ) with unit w, z: the issue's
        // `U diag(s) Vᴴ` family, but built from two Householder reflectors so
        // the unitary factors are exact to machine precision and the whole
        // matrix costs O(m²) instead of a Gram-Schmidt O(m³).
        //
        // The construction matters, not just the spectrum. An earlier revision
        // used a circulant-like closed form with the same ten decades, and on
        // that matrix Xgesvdp lost about two digits against `gesvd`
        // (4.1e-13 versus 4.4e-15 at m = 400 on an A100), so the tolerance
        // below — measured in #1849 on `U diag(s) Vᴴ` — did not hold. Polar
        // SVD accuracy is input-dependent; see the work log.
        let ratio = 10f64.powf(-10.0 / (m - 1) as f64);
        let unit_vector = |phase: f64, shift: f64| -> Vec<Complex64> {
            let raw: Vec<Complex64> = (0..m)
                .map(|i| Complex64::from_polar(1.0 + ((i as f64) * shift).sin(), phase * i as f64))
                .collect();
            let norm = raw.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            raw.into_iter().map(|z| z / norm).collect()
        };
        let w = unit_vector(0.17, 0.41);
        let z = unit_vector(0.31, 0.23);
        // Expand once: A = D - 2w(wᴴD) - 2(Dz)zᴴ + 4(wᴴDz)wzᴴ with D = diag(s).
        let wh_d: Vec<Complex64> = (0..m).map(|j| w[j].conj() * ratio.powi(j as i32)).collect();
        let d_z: Vec<Complex64> = (0..m).map(|i| z[i] * ratio.powi(i as i32)).collect();
        let wh_d_z: Complex64 = (0..m).map(|j| wh_d[j] * z[j]).sum();
        let data = (0..m * m)
            .map(|idx| {
                let (i, j) = (idx % m, idx / m);
                let mut value = -2.0 * w[i] * wh_d[j] - 2.0 * d_z[i] * z[j].conj()
                    + 4.0 * wh_d_z * w[i] * z[j].conj();
                if i == j {
                    value += Complex64::new(ratio.powi(i as i32), 0.0);
                }
                value
            })
            .collect();
        let input = tensor_c64(vec![m, m], data);
        let device = upload(&gpu, &input);
        let mut spectra = Vec::new();
        for driver in [SvdDriver::Gesvd, SvdDriver::Xgesvdp] {
            let result = with_cuda_linalg_session(&mut gpu, |s| {
                s.svd_with_options_read(
                    TensorRead::from_tensor(&device),
                    SvdOptions::default().driver(driver),
                )
            })
            .unwrap();
            spectra.push(
                download(&gpu, &result[1])
                    .as_slice::<f64>()
                    .unwrap()
                    .to_vec(),
            );
        }
        let error = spectra[0]
            .iter()
            .zip(&spectra[1])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
            / spectra[0][0];
        assert!(error < 1e-13, "{m}: max|Δs|/s₁ = {error}");
        for (i, &s) in spectra[1].iter().enumerate() {
            assert!(
                (s - ratio.powi(i as i32)).abs() < 1e-13,
                "{m}: known spectrum at {i}"
            );
        }
    }
}
