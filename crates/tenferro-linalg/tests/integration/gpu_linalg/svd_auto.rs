//! CUDA SVD validation for #2182: the default `SvdDriver::Auto` path at
//! `min(m, n) >= 256` (which stays accuracy-preserving), the default
//! pseudoinverse cutoff on a rank-deficient spectrum, and the explicit
//! `Xgesvdp` `h_err_sigma` guard.
//!
//! These are hardware-gated (`#[ignore]`) and registered in
//! `scripts/ci/cuda_test_partition.tsv`. They cover dense square/tall/wide/
//! batched reconstruction, a rank-deficient spectrum with a pseudoinverse
//! residual, the pseudoinverse mask near its cutoff, the values-only path, the
//! default `pinv` retained rank, and the per-matrix rejection of an explicit
//! `Xgesvdp` request on rank-deficient single precision or inside a mixed-scale
//! batch.

use super::*;

/// Fixed-seed splitmix64 stream, the same shape as the CUDA SVD driver bench.
/// A dense pseudo-random matrix has no low-rank structure, unlike the rank-31
/// periodic pattern older SVD tests use.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_unit(&mut self) -> f64 {
        // 53-bit mantissa in [0, 1), mapped to [-1, 1).
        ((self.next_u64() >> 11) as f64) * (2.0 / (1u64 << 53) as f64) - 1.0
    }
}

fn dense_input(dtype: DType, shape: &[usize], seed: u64) -> Tensor {
    let len: usize = shape.iter().product();
    let mut rng = SplitMix64::new(seed);
    match dtype {
        DType::F32 => tensor_f32(
            shape.to_vec(),
            (0..len).map(|_| rng.next_unit() as f32).collect(),
        ),
        DType::F64 => tensor_f64(shape.to_vec(), (0..len).map(|_| rng.next_unit()).collect()),
        DType::C32 => tensor_c32(
            shape.to_vec(),
            (0..len)
                .map(|_| Complex32::new(rng.next_unit() as f32, rng.next_unit() as f32))
                .collect(),
        ),
        DType::C64 => tensor_c64(
            shape.to_vec(),
            (0..len)
                .map(|_| Complex64::new(rng.next_unit(), rng.next_unit()))
                .collect(),
        ),
        other => panic!("dense_input has no dtype arm for {other:?}"),
    }
}

/// Rebuild a tensor of `dtype` from Complex64 host values, discarding the
/// imaginary part for real dtypes.
fn complex_tensor(dtype: DType, shape: Vec<usize>, data: &[Complex64]) -> Tensor {
    match dtype {
        DType::F32 => tensor_f32(shape, data.iter().map(|z| z.re as f32).collect()),
        DType::F64 => tensor_f64(shape, data.iter().map(|z| z.re).collect()),
        DType::C32 => tensor_c32(
            shape,
            data.iter()
                .map(|z| Complex32::new(z.re as f32, z.im as f32))
                .collect(),
        ),
        DType::C64 => tensor_c64(shape, data.to_vec()),
        other => panic!("complex_tensor has no dtype arm for {other:?}"),
    }
}

/// `A = U V` with `U: n x r`, `V: r x n` dense pseudo-random, so `rank(A) = r`
/// and the singular values past `r` are at roundoff. Real dtypes are built from
/// real factors: taking the real part of a complex low-rank product would
/// roughly double its rank.
fn low_rank_input(dtype: DType, n: usize, r: usize, seed: u64) -> Tensor {
    let mut rng = SplitMix64::new(seed);
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    if matches!(dtype, DType::F32 | DType::F64) {
        let u: Vec<f64> = (0..n * r).map(|_| rng.next_unit()).collect();
        let v: Vec<f64> = (0..r * n).map(|_| rng.next_unit()).collect();
        for col in 0..n {
            for k in 0..r {
                let v_kj = v[col_major_index(r, k, col)];
                for row in 0..n {
                    a[col_major_index(n, row, col)].re += u[col_major_index(n, row, k)] * v_kj;
                }
            }
        }
    } else {
        let u: Vec<Complex64> = (0..n * r)
            .map(|_| Complex64::new(rng.next_unit(), rng.next_unit()))
            .collect();
        let v: Vec<Complex64> = (0..r * n)
            .map(|_| Complex64::new(rng.next_unit(), rng.next_unit()))
            .collect();
        for col in 0..n {
            for k in 0..r {
                let v_kj = v[col_major_index(r, k, col)];
                for row in 0..n {
                    a[col_major_index(n, row, col)] += u[col_major_index(n, row, k)] * v_kj;
                }
            }
        }
    }
    complex_tensor(dtype, vec![n, n], &a)
}

fn unit_vector(n: usize, phase: f64, shift: f64) -> Vec<Complex64> {
    let raw: Vec<Complex64> = (0..n)
        .map(|i| Complex64::from_polar(1.0 + ((i as f64) * shift).sin(), phase * i as f64))
        .collect();
    let norm = raw.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    raw.into_iter().map(|z| z / norm).collect()
}

/// `A = (I - 2wwᴴ) diag(s) (I - 2zzᴴ)` with unit `w, z`. Both Householder
/// factors are unitary and Hermitian, so `AᴴA = H_z diag(s)² H_z` and the
/// singular values are exactly `s`, at O(n²) construction cost.
fn spectrum_matrix(n: usize, s: &[f64]) -> Tensor {
    assert_eq!(s.len(), n);
    let w = unit_vector(n, 0.17, 0.41);
    let z = unit_vector(n, 0.31, 0.23);
    let wh_d: Vec<Complex64> = (0..n).map(|j| w[j].conj() * s[j]).collect();
    let d_z: Vec<Complex64> = (0..n).map(|i| z[i] * s[i]).collect();
    let wh_d_z: Complex64 = (0..n).map(|j| wh_d[j] * z[j]).sum();
    let data = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx % n, idx / n);
            let mut value = -2.0 * w[i] * wh_d[j] - 2.0 * d_z[i] * z[j].conj()
                + 4.0 * wh_d_z * w[i] * z[j].conj();
            if i == j {
                value += Complex64::new(s[i], 0.0);
            }
            value
        })
        .collect();
    tensor_c64(vec![n, n], data)
}

fn dtype_tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => 3.0e-4,
        DType::F64 | DType::C64 => 1.0e-10,
        other => panic!("no SVD tolerance for {other:?}"),
    }
}

/// `||A - U diag(S) Vᴴ||_F / ||A||_F` on host copies, over every batch.
fn relative_reconstruction_error(input: &Tensor, factors: &[Tensor]) -> f64 {
    let (m, n) = (input.shape()[0], input.shape()[1]);
    let k = m.min(n);
    let batches: usize = input.shape()[2..].iter().product();
    let a = complex_values(input);
    let u = complex_values(&factors[0]);
    let s = real_values(&factors[1]);
    let vt = complex_values(&factors[2]);
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for batch in 0..batches {
        let a_b = &a[batch * m * n..(batch + 1) * m * n];
        let u_b = &u[batch * m * k..(batch + 1) * m * k];
        let s_b = &s[batch * k..(batch + 1) * k];
        let vt_b = &vt[batch * k * n..(batch + 1) * k * n];
        let mut scaled_u = u_b.to_vec();
        for col in 0..k {
            for row in 0..m {
                scaled_u[col_major_index(m, row, col)] *= s_b[col];
            }
        }
        let rebuilt = matmul_c64(&scaled_u, vt_b, m, k, n);
        for (actual, expected) in rebuilt.iter().zip(a_b) {
            numerator += (*actual - *expected).norm_sqr();
        }
        for expected in a_b {
            denominator += expected.norm_sqr();
        }
    }
    (numerator / denominator.max(f64::MIN_POSITIVE)).sqrt()
}

/// `||A pinv(A) A - A||_F / ||A||_F` on host copies.
fn pinv_residual_error(a: &Tensor, pseudoinverse: &Tensor) -> f64 {
    let n = a.shape()[0];
    let a_c = complex_values(a);
    let p_c = complex_values(pseudoinverse);
    let ap = matmul_c64(&a_c, &p_c, n, n, n);
    let apa = matmul_c64(&ap, &a_c, n, n, n);
    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;
    for (actual, expected) in apa.iter().zip(&a_c) {
        numerator += (*actual - *expected).norm_sqr();
    }
    for expected in &a_c {
        denominator += expected.norm_sqr();
    }
    (numerator / denominator.max(f64::MIN_POSITIVE)).sqrt()
}

/// (a) Dense square, tall, wide, and batched reconstruction through `Auto` for
/// every supported SVD dtype, all at or above the `min(m, n) = 256` threshold.
#[test]
#[ignore = "requires CUDA; validates the Auto Xgesvdp branch at min(m, n) >= 256"]
fn cuda_svd_auto_reconstructs_dense_dtypes_and_shapes_from_min_dim_256() {
    if !gpu_available() {
        return;
    }
    let shapes: Vec<Vec<usize>> = vec![
        vec![256, 256],
        vec![512, 256],
        vec![256, 512],
        vec![256, 256, 2],
    ];
    let mut gpu = gpu_backend();
    for (case, dtype) in [DType::F32, DType::F64, DType::C32, DType::C64]
        .into_iter()
        .enumerate()
    {
        for (shape_case, shape) in shapes.iter().enumerate() {
            let (m, n) = (shape[0], shape[1]);
            let input = dense_input(dtype, shape, 0x2182_0000 + (case * 16 + shape_case) as u64);
            let device = upload(&gpu, &input);
            let factors =
                with_cuda_linalg_session(&mut gpu, |session| session.svd(&device)).unwrap();
            for factor in &factors {
                assert_eq!(
                    factor.placement().memory_kind,
                    tenferro_tensor::MemoryKind::Device,
                    "Auto SVD factors must stay resident on the device"
                );
            }
            let host: Vec<Tensor> = factors.iter().map(|t| download(&gpu, t)).collect();
            let k = m.min(n);
            assert_eq!(&host[0].shape()[..2], &[m, k]);
            assert_eq!(&host[2].shape()[..2], &[k, n]);
            let error = relative_reconstruction_error(&input, &host);
            assert!(
                error <= dtype_tolerance(dtype),
                "{dtype:?} shape {shape:?}: Auto reconstruction {error:e} exceeds {:e}",
                dtype_tolerance(dtype)
            );
            assert!(error.is_finite());
        }
    }
}

/// `Auto` stays the accuracy-preserving JAX rule: `gesvdj` up to the 1024
/// boundary, `gesvd` above it, and never the polar `Xgesvdp`.
#[test]
#[ignore = "requires CUDA; Auto keeps the accurate driver across the 1024 boundary"]
fn cuda_svd_auto_keeps_the_accurate_driver_across_the_1024_boundary() {
    if !gpu_available() {
        return;
    }
    let mut gpu = gpu_backend();
    for (order, expected) in [(256_usize, SvdDriver::Gesvdj), (1025, SvdDriver::Gesvd)] {
        let input = dense_input(DType::F64, &[order, order], 0x2182_1000 + order as u64);
        let device = upload(&gpu, &input);
        let auto = with_cuda_linalg_session(&mut gpu, |session| {
            session.svd_with_options(&device, SvdOptions::default().driver(SvdDriver::Auto))
        })
        .unwrap();
        let explicit = with_cuda_linalg_session(&mut gpu, |session| {
            session.svd_with_options(&device, SvdOptions::default().driver(expected))
        })
        .unwrap();
        for (actual, expected) in auto.iter().zip(&explicit) {
            assert_tensor_close(&download(&gpu, actual), &download(&gpu, expected), 0.0);
        }
    }
}

/// (b) Rank-deficient input: the spectrum past the rank is roundoff, and the
/// pseudoinverse reconstructs the range, `||A pinv(A) A - A|| / ||A||`.
#[test]
#[ignore = "requires CUDA; rank-deficient Auto SVD and pseudoinverse residual"]
fn cuda_svd_auto_rank_deficient_spectrum_and_pinv_residual() {
    if !gpu_available() {
        return;
    }
    const N: usize = 256;
    const RANK: usize = 64;
    let mut gpu = gpu_backend();
    for (case, dtype) in [DType::F32, DType::F64, DType::C32, DType::C64]
        .into_iter()
        .enumerate()
    {
        let input = low_rank_input(dtype, N, RANK, 0x2182_2000 + case as u64);
        // The CPU reference proves the matrix is rank-deficient, so a bad
        // device tail is the driver's fault and not the input's.
        let mut cpu = cpu_backend();
        let cpu_values = real_values(
            &with_cpu_linalg_session(&mut cpu, |session| session.svd_values(&input)).unwrap(),
        );
        let cpu_smax = cpu_values[0];
        let cpu_tail = cpu_values[RANK..]
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!(
            cpu_tail <= 1.0e-5 * cpu_smax,
            "{dtype:?}: the test input is not rank {RANK}: cpu tail {cpu_tail:e} against {cpu_smax:e}"
        );

        let device = upload(&gpu, &input);
        let factors = with_cuda_linalg_session(&mut gpu, |session| session.svd(&device)).unwrap();
        let values = real_values(&download(&gpu, &factors[1]));
        let smax = values[0];
        let tail = values[RANK..]
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        eprintln!(
            "{dtype:?}: Auto SVD smax {smax:e} tail {tail:e} (cpu smax {cpu_smax:e} tail {cpu_tail:e})"
        );
        // Auto is the accuracy-preserving JAX rule, so the tail is at machine
        // precision for every dtype.
        let tail_tolerance = if matches!(dtype, DType::F32 | DType::C32) {
            1.0e-3
        } else {
            1.0e-10
        };
        assert!(
            tail <= tail_tolerance * smax,
            "{dtype:?}: singular values past rank {RANK} reach {tail:e}, above {tail_tolerance:e} * {smax:e}"
        );

        let rtol = if matches!(dtype, DType::F32 | DType::C32) {
            1.0e-4
        } else {
            1.0e-8
        };
        let pseudoinverse =
            with_cuda_linalg_session(&mut gpu, |session| device.pinv_with_rtol(rtol, session))
                .unwrap();
        let residual = pinv_residual_error(&input, &download(&gpu, &pseudoinverse));
        let pinv_tolerance = if matches!(dtype, DType::F32 | DType::C32) {
            1.0e-3
        } else {
            1.0e-9
        };
        assert!(
            residual <= pinv_tolerance,
            "{dtype:?}: ||A pinv(A) A - A|| / ||A|| = {residual:e} exceeds {pinv_tolerance:e}"
        );
    }
}

/// (c) A spectrum with values on both sides of the pseudoinverse cutoff: the
/// `Auto` and `Gesvd` spectra must keep and drop the same singular values.
#[test]
#[ignore = "requires CUDA; Auto pseudoinverse mask agrees with the Gesvd reference near the cutoff"]
fn cuda_svd_auto_pinv_mask_matches_gesvd_near_cutoff() {
    if !gpu_available() {
        return;
    }
    const N: usize = 256;
    const RTOL: f64 = 1.0e-3;
    // Five above the cutoff (1, 0.5, 0.2, 2e-3, 1.1e-3) and three below
    // (0.9e-3, 5e-4, 1e-5); the rest are exact zeros. The 10% gap is far above
    // the polar driver's perturbation, so the mask is a driver comparison and
    // not a coin flip.
    let mut spectrum = vec![0.0_f64; N];
    for (index, value) in [1.0, 0.5, 0.2, 2.0e-3, 1.1e-3, 0.9e-3, 5.0e-4, 1.0e-5]
        .into_iter()
        .enumerate()
    {
        spectrum[index] = value;
    }
    let input = spectrum_matrix(N, &spectrum);
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &input);
    let auto = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_values_with_driver(&device, SvdDriver::Auto)
    })
    .unwrap();
    let reference = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_values_with_driver(&device, SvdDriver::Gesvd)
    })
    .unwrap();
    let auto = real_values(&download(&gpu, &auto));
    let reference = real_values(&download(&gpu, &reference));
    let mask = |values: &[f64]| -> Vec<usize> {
        let smax = values[0];
        values
            .iter()
            .enumerate()
            .filter(|(_, value)| **value > RTOL * smax)
            .map(|(index, _)| index)
            .collect()
    };
    assert_eq!(
        mask(&auto),
        mask(&reference),
        "Auto and Gesvd keep different singular values at the {RTOL:e} cutoff"
    );
    assert_eq!(mask(&reference), vec![0, 1, 2, 3, 4]);
}

/// The values-only (`svdvals`) path uses the same Auto branch and matches the
/// spectrum of the full factorization.
#[test]
#[ignore = "requires CUDA; values-only Auto path at min(m, n) >= 256"]
fn cuda_svd_auto_values_only_matches_full_factors_for_all_dtypes() {
    if !gpu_available() {
        return;
    }
    const N: usize = 256;
    let mut gpu = gpu_backend();
    for (case, dtype) in [DType::F32, DType::F64, DType::C32, DType::C64]
        .into_iter()
        .enumerate()
    {
        let input = dense_input(dtype, &[N, N], 0x2182_3000 + case as u64);
        let device = upload(&gpu, &input);
        let factors = with_cuda_linalg_session(&mut gpu, |session| session.svd(&device)).unwrap();
        let s = download(&gpu, &factors[1]);
        let smax = real_values(&s)[0];
        let values = with_cuda_linalg_session(&mut gpu, |session| {
            session.svd_values_with_driver(&device, SvdDriver::Auto)
        })
        .unwrap();
        assert_tensor_close(&download(&gpu, &values), &s, dtype_tolerance(dtype));
        let reference = with_cuda_linalg_session(&mut gpu, |session| {
            session.svd_values_with_driver(&device, SvdDriver::Gesvd)
        })
        .unwrap();
        // Compare against the distinct driver with a relative tolerance: the
        // polar and QR drivers differ at their own roundoff level, not zero.
        let reference_tolerance = dtype_tolerance(dtype) * smax.max(1.0);
        assert_tensor_close(
            &download(&gpu, &values),
            &download(&gpu, &reference),
            reference_tolerance,
        );
    }
}

/// An explicit `Xgesvdp` request on single-precision rank-deficient input must
/// fail with a typed backend error instead of returning cuSOLVER's perturbed
/// spectrum. The values-only path shares the same guard.
#[test]
#[ignore = "requires CUDA; explicit Xgesvdp rejects unreliable h_err_sigma"]
fn cuda_svd_xgesvdp_rejects_unreliable_single_precision_factorization() {
    if !gpu_available() {
        return;
    }
    const N: usize = 256;
    const RANK: usize = 64;
    let input = low_rank_input(DType::F32, N, RANK, 0x2182_2000);
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &input);
    let error = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_with_options(&device, SvdOptions::default().driver(SvdDriver::Xgesvdp))
    })
    .unwrap_err();
    assert!(
        matches!(error, tenferro_tensor::Error::BackendFailure { .. }),
        "explicit Xgesvdp on rank-deficient f32 should fail with a typed backend error, got {error:?}"
    );
    let error = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_values_with_driver(&device, SvdDriver::Xgesvdp)
    })
    .unwrap_err();
    assert!(
        matches!(error, tenferro_tensor::Error::BackendFailure { .. }),
        "explicit Xgesvdp values-only on rank-deficient f32 should fail with a typed backend error, got {error:?}"
    );
}

/// The error bound is per matrix: a well-conditioned batch member with a huge
/// spectrum must not mask a bad rank-deficient member.
#[test]
#[ignore = "requires CUDA; Xgesvdp rejects a bad batched member beside a large one"]
fn cuda_svd_xgesvdp_rejects_a_bad_batch_member_beside_a_large_one() {
    if !gpu_available() {
        return;
    }
    const N: usize = 256;
    // Batch 0 is the f32 rank-deficient matrix; batch 1 is 1e10 I, whose
    // singular values dwarf any batch-global bound.
    let bad = low_rank_input(DType::F32, N, 64, 0x2182_2000);
    let mut data = bad.as_slice::<f32>().unwrap().to_vec();
    let mut large = vec![0.0_f32; N * N];
    for index in 0..N {
        large[col_major_index(N, index, index)] = 1.0e10;
    }
    data.extend_from_slice(&large);
    let input = tensor_f32(vec![N, N, 2], data);
    let mut gpu = gpu_backend();
    let device = upload(&gpu, &input);
    let error = with_cuda_linalg_session(&mut gpu, |session| {
        session.svd_with_options(&device, SvdOptions::default().driver(SvdDriver::Xgesvdp))
    })
    .unwrap_err();
    assert!(
        matches!(error, tenferro_tensor::Error::BackendFailure { .. }),
        "a bad batch member must be rejected even beside a large one, got {error:?}"
    );
    assert!(
        format!("{error}").contains("batch 0"),
        "the error should name the failing batch, got {error}"
    );
}

/// The unchanged default `pinv()` cutoff (`eps * max_dim`) must keep the same
/// rank as the CPU reference when `Auto` picks Xgesvdp for double precision.
/// If Xgesvdp's perturbation lifts the zero singular values above that cutoff,
/// the pseudoinverse itself is wrong even though `A P A ~= A` still holds.
#[test]
#[ignore = "requires CUDA; default eps-cutoff pinv on the Auto Xgesvdp rank-deficient spectrum"]
fn cuda_svd_auto_default_pinv_rank_deficient_double_precision() {
    if !gpu_available() {
        return;
    }
    const N: usize = 256;
    const RANK: usize = 64;
    let mut gpu = gpu_backend();
    let mut cpu = cpu_backend();
    for (case, dtype) in [DType::F64, DType::C64].into_iter().enumerate() {
        let input = low_rank_input(dtype, N, RANK, 0x2182_4000 + case as u64);
        let rtol = f64::EPSILON * N as f64;
        let device = upload(&gpu, &input);
        // `Auto` is the accuracy-preserving JAX rule, so the default `pinv`
        // cutoff must retain the same rank as the CPU reference.
        let auto_values = with_cuda_linalg_session(&mut gpu, |session| {
            session.svd_values_with_driver(&device, SvdDriver::Auto)
        })
        .unwrap();
        let auto_values = real_values(&download(&gpu, &auto_values));
        let cpu_values = real_values(
            &with_cpu_linalg_session(&mut cpu, |session| session.svd_values(&input)).unwrap(),
        );
        let mask = |values: &[f64]| -> usize {
            values
                .iter()
                .filter(|value| **value > rtol * values[0])
                .count()
        };
        let auto_rank = mask(&auto_values);
        let cpu_rank = mask(&cpu_values);
        let auto_pinv =
            with_cuda_linalg_session(&mut gpu, |session| device.pinv_with_rtol(rtol, session))
                .unwrap();
        let cpu_pinv =
            with_cpu_linalg_session(&mut cpu, |session| input.pinv_with_rtol(rtol, session))
                .unwrap();
        let p_auto = complex_values(&download(&gpu, &auto_pinv));
        let p_cpu = complex_values(&cpu_pinv);
        let p_error = {
            let number: f64 = p_auto
                .iter()
                .zip(&p_cpu)
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum();
            let denominator: f64 = p_cpu.iter().map(|b| b.norm_sqr()).sum();
            (number / denominator.max(f64::MIN_POSITIVE)).sqrt()
        };
        let a = complex_values(&input);
        let pap = matmul_c64(&matmul_c64(&p_auto, &a, N, N, N), &p_auto, N, N, N);
        let pap_error = {
            let number: f64 = pap
                .iter()
                .zip(&p_auto)
                .map(|(a, b)| (*a - *b).norm_sqr())
                .sum();
            let denominator: f64 = p_auto.iter().map(|b| b.norm_sqr()).sum();
            (number / denominator.max(f64::MIN_POSITIVE)).sqrt()
        };
        eprintln!(
            "{dtype:?}: auto rank {auto_rank}, cpu rank {cpu_rank}; ||P - P_cpu||/||P_cpu|| = \
             {p_error:e}; ||P A P - P||/||P|| = {pap_error:e}"
        );
        assert_eq!(
            auto_rank, cpu_rank,
            "{dtype:?}: the default pinv must retain the reference rank"
        );
        assert!(
            p_error <= 1.0e-6,
            "{dtype:?}: pinv vs reference {p_error:e}"
        );
        assert!(
            pap_error <= 1.0e-6,
            "{dtype:?}: ||P A P - P|| {pap_error:e}"
        );
    }
}
