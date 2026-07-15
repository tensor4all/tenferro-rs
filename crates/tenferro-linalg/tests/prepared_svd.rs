use num_complex::{Complex32, Complex64};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::{
    LinalgBackend, PreparedSvdBackendExt, SvdGauge, SvdOptions, SvdOutputWrites,
};
use tenferro_tensor::{
    BackendId, DType, Error, Tensor, TensorRead, TensorView, TensorViewMut, TensorWrite,
    TypedTensorView, TypedTensorViewMut,
};

fn f64_outputs(m: usize, n: usize, fill: f64) -> (Tensor, Tensor, Tensor) {
    let k = m.min(n);
    (
        Tensor::from_vec_col_major(vec![m, k], vec![fill; m * k]).unwrap(),
        Tensor::from_vec_col_major(vec![k], vec![fill; k]).unwrap(),
        Tensor::from_vec_col_major(vec![k, n], vec![fill; k * n]).unwrap(),
    )
}

fn assert_f64_svd_residual(input: &[f64], u: &[f64], s: &[f64], vt: &[f64], m: usize, n: usize) {
    let k = m.min(n);
    let mut max_reconstruction = 0.0_f64;
    for col in 0..n {
        for row in 0..m {
            let mut reconstructed = 0.0;
            for inner in 0..k {
                reconstructed += u[row + m * inner] * s[inner] * vt[inner + k * col];
            }
            max_reconstruction =
                max_reconstruction.max((reconstructed - input[row + m * col]).abs());
        }
    }
    let mut max_orthogonality = 0.0_f64;
    for lhs in 0..k {
        for rhs in 0..k {
            let mut dot = 0.0;
            for row in 0..m {
                dot += u[row + m * lhs] * u[row + m * rhs];
            }
            let expected = if lhs == rhs { 1.0 } else { 0.0 };
            max_orthogonality = max_orthogonality.max((dot - expected).abs());
        }
    }
    let mut max_vt_orthogonality = 0.0_f64;
    for lhs in 0..k {
        for rhs in 0..k {
            let mut dot = 0.0;
            for col in 0..n {
                dot += vt[lhs + k * col] * vt[rhs + k * col];
            }
            let expected = if lhs == rhs { 1.0 } else { 0.0 };
            max_vt_orthogonality = max_vt_orthogonality.max((dot - expected).abs());
        }
    }
    assert!(
        max_reconstruction < 1.0e-10,
        "reconstruction residual {max_reconstruction}"
    );
    assert!(
        max_orthogonality < 1.0e-10,
        "U orthogonality residual {max_orthogonality}"
    );
    assert!(
        max_vt_orthogonality < 1.0e-10,
        "Vt row orthogonality residual {max_vt_orthogonality}"
    );
}

fn assert_c64_svd_residual(
    input: &[Complex64],
    u: &[Complex64],
    s: &[f64],
    vt: &[Complex64],
    m: usize,
    n: usize,
    tolerance: f64,
) {
    let k = m.min(n);
    let mut max_reconstruction = 0.0_f64;
    for col in 0..n {
        for row in 0..m {
            let mut reconstructed = Complex64::new(0.0, 0.0);
            for inner in 0..k {
                reconstructed += u[row + m * inner] * s[inner] * vt[inner + k * col];
            }
            max_reconstruction =
                max_reconstruction.max((reconstructed - input[row + m * col]).norm());
        }
    }
    let mut max_orthogonality = 0.0_f64;
    for lhs in 0..k {
        for rhs in 0..k {
            let mut dot = Complex64::new(0.0, 0.0);
            for row in 0..m {
                dot += u[row + m * lhs].conj() * u[row + m * rhs];
            }
            let expected = if lhs == rhs {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            max_orthogonality = max_orthogonality.max((dot - expected).norm());
        }
    }
    let mut max_vt_orthogonality = 0.0_f64;
    for lhs in 0..k {
        for rhs in 0..k {
            let mut dot = Complex64::new(0.0, 0.0);
            for col in 0..n {
                dot += vt[lhs + k * col] * vt[rhs + k * col].conj();
            }
            let expected = if lhs == rhs {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            max_vt_orthogonality = max_vt_orthogonality.max((dot - expected).norm());
        }
    }
    assert!(
        max_reconstruction < tolerance,
        "reconstruction residual {max_reconstruction}"
    );
    assert!(
        max_orthogonality < tolerance,
        "U orthogonality residual {max_orthogonality}"
    );
    assert!(
        max_vt_orthogonality < tolerance,
        "Vt row orthogonality residual {max_vt_orthogonality}"
    );
}

fn assert_f32_svd_residual(input: &[f32], u: &[f32], s: &[f32], vt: &[f32], m: usize, n: usize) {
    let input = input.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let u = u.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let s = s.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let vt = vt.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let k = m.min(n);
    let mut reconstruction = 0.0_f64;
    for col in 0..n {
        for row in 0..m {
            let actual = (0..k)
                .map(|inner| u[row + m * inner] * s[inner] * vt[inner + k * col])
                .sum::<f64>();
            reconstruction = reconstruction.max((actual - input[row + m * col]).abs());
        }
    }
    let mut unitary = 0.0_f64;
    for lhs in 0..k {
        for rhs in 0..k {
            let expected = f64::from(lhs == rhs);
            let u_dot = (0..m)
                .map(|row| u[row + m * lhs] * u[row + m * rhs])
                .sum::<f64>();
            let vt_dot = (0..n)
                .map(|col| vt[lhs + k * col] * vt[rhs + k * col])
                .sum::<f64>();
            unitary = unitary.max((u_dot - expected).abs());
            unitary = unitary.max((vt_dot - expected).abs());
        }
    }
    assert!(
        reconstruction < 2.0e-4,
        "F32 reconstruction residual {reconstruction}"
    );
    assert!(unitary < 2.0e-4, "F32 unitarity residual {unitary}");
}

fn assert_c32_svd_residual(
    input: &[Complex32],
    u: &[Complex32],
    s: &[f32],
    vt: &[Complex32],
    m: usize,
    n: usize,
) {
    let lift = |value: &Complex32| Complex64::new(f64::from(value.re), f64::from(value.im));
    let input = input.iter().map(lift).collect::<Vec<_>>();
    let u = u.iter().map(lift).collect::<Vec<_>>();
    let s = s.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let vt = vt.iter().map(lift).collect::<Vec<_>>();
    assert_c64_svd_residual(&input, &u, &s, &vt, m, n, 2.0e-4);
}

#[test]
fn prepared_svd_all_dtypes_cover_shapes_repeated_values_and_gauges() {
    let cases = [
        (2, 2, vec![2.0_f64, 0.0, 0.0, 2.0]),
        (3, 2, vec![3.0, 1.0, -2.0, 4.0, -1.0, 2.0]),
        (2, 3, vec![3.0, 1.0, -2.0, 4.0, -1.0, 2.0]),
    ];
    for gauge in [SvdGauge::Raw, SvdGauge::CanonicalPivot] {
        for (m, n, real_data) in &cases {
            let (m, n) = (*m, *n);
            let options = SvdOptions::default().gauge(gauge);

            let f32_data = real_data.iter().map(|&x| x as f32).collect::<Vec<_>>();
            let input = Tensor::from_vec_col_major(vec![m, n], f32_data.clone()).unwrap();
            let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
            let plan = backend.prepare_svd([m, n], DType::F32, options).unwrap();
            let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
            let k = m.min(n);
            let mut u = Tensor::from_vec_col_major(vec![m, k], vec![0.0_f32; m * k]).unwrap();
            let mut s = Tensor::from_vec_col_major(vec![k], vec![0.0_f32; k]).unwrap();
            let mut vt = Tensor::from_vec_col_major(vec![k, n], vec![0.0_f32; k * n]).unwrap();
            plan.execute_into(
                &mut backend,
                &mut workspace,
                TensorRead::from_tensor(&input),
                SvdOutputWrites::new(
                    TensorWrite::from_tensor(&mut u),
                    TensorWrite::from_tensor(&mut s),
                    TensorWrite::from_tensor(&mut vt),
                ),
            )
            .unwrap();
            assert_f32_svd_residual(
                &f32_data,
                u.as_slice().unwrap(),
                s.as_slice().unwrap(),
                vt.as_slice().unwrap(),
                m,
                n,
            );

            let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
            let input = Tensor::from_vec_col_major(vec![m, n], real_data.clone()).unwrap();
            let plan = backend.prepare_svd([m, n], DType::F64, options).unwrap();
            let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
            let (mut u, mut s, mut vt) = f64_outputs(m, n, 0.0);
            plan.execute_into(
                &mut backend,
                &mut workspace,
                TensorRead::from_tensor(&input),
                SvdOutputWrites::new(
                    TensorWrite::from_tensor(&mut u),
                    TensorWrite::from_tensor(&mut s),
                    TensorWrite::from_tensor(&mut vt),
                ),
            )
            .unwrap();
            assert_f64_svd_residual(
                real_data,
                u.as_slice().unwrap(),
                s.as_slice().unwrap(),
                vt.as_slice().unwrap(),
                m,
                n,
            );

            let c32_data = real_data
                .iter()
                .enumerate()
                .map(|(i, &x)| Complex32::new(x as f32, (i % 3) as f32 * 0.125))
                .collect::<Vec<_>>();
            let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
            let input = Tensor::from_vec_col_major(vec![m, n], c32_data.clone()).unwrap();
            let plan = backend.prepare_svd([m, n], DType::C32, options).unwrap();
            let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
            let mut u =
                Tensor::from_vec_col_major(vec![m, k], vec![Complex32::default(); m * k]).unwrap();
            let mut s = Tensor::from_vec_col_major(vec![k], vec![0.0_f32; k]).unwrap();
            let mut vt =
                Tensor::from_vec_col_major(vec![k, n], vec![Complex32::default(); k * n]).unwrap();
            plan.execute_into(
                &mut backend,
                &mut workspace,
                TensorRead::from_tensor(&input),
                SvdOutputWrites::new(
                    TensorWrite::from_tensor(&mut u),
                    TensorWrite::from_tensor(&mut s),
                    TensorWrite::from_tensor(&mut vt),
                ),
            )
            .unwrap();
            assert_c32_svd_residual(
                &c32_data,
                u.as_slice().unwrap(),
                s.as_slice().unwrap(),
                vt.as_slice().unwrap(),
                m,
                n,
            );

            let c64_data = real_data
                .iter()
                .enumerate()
                .map(|(i, &x)| Complex64::new(x, (i % 3) as f64 * 0.125))
                .collect::<Vec<_>>();
            let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
            let input = Tensor::from_vec_col_major(vec![m, n], c64_data.clone()).unwrap();
            let plan = backend.prepare_svd([m, n], DType::C64, options).unwrap();
            let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
            let mut u =
                Tensor::from_vec_col_major(vec![m, k], vec![Complex64::default(); m * k]).unwrap();
            let mut s = Tensor::from_vec_col_major(vec![k], vec![0.0_f64; k]).unwrap();
            let mut vt =
                Tensor::from_vec_col_major(vec![k, n], vec![Complex64::default(); k * n]).unwrap();
            plan.execute_into(
                &mut backend,
                &mut workspace,
                TensorRead::from_tensor(&input),
                SvdOutputWrites::new(
                    TensorWrite::from_tensor(&mut u),
                    TensorWrite::from_tensor(&mut s),
                    TensorWrite::from_tensor(&mut vt),
                ),
            )
            .unwrap();
            assert_c64_svd_residual(
                &c64_data,
                u.as_slice().unwrap(),
                s.as_slice().unwrap(),
                vt.as_slice().unwrap(),
                m,
                n,
                1.0e-10,
            );
        }
    }
}

#[test]
fn prepared_svd_supports_f32_c32_and_c64_with_owned_semantics() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();

    let input_f32 =
        Tensor::from_vec_col_major(vec![3, 2], vec![3.0_f32, 1.0, -2.0, 4.0, -1.0, 2.0]).unwrap();
    let plan_f32 = backend
        .prepare_svd([3, 2], DType::F32, SvdOptions::default())
        .unwrap();
    let mut workspace_f32 = plan_f32.allocate_workspace(&mut backend).unwrap();
    let mut u_f32 = Tensor::from_vec_col_major(vec![3, 2], vec![0.0_f32; 6]).unwrap();
    let mut s_f32 = Tensor::from_vec_col_major(vec![2], vec![0.0_f32; 2]).unwrap();
    let mut vt_f32 = Tensor::from_vec_col_major(vec![2, 2], vec![0.0_f32; 4]).unwrap();
    plan_f32
        .execute_into(
            &mut backend,
            &mut workspace_f32,
            TensorRead::from_tensor(&input_f32),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u_f32),
                TensorWrite::from_tensor(&mut s_f32),
                TensorWrite::from_tensor(&mut vt_f32),
            ),
        )
        .unwrap();
    let owned_f32 = backend.svd(&input_f32).unwrap();
    for (prepared, owned) in [
        (
            u_f32.as_slice::<f32>().unwrap(),
            owned_f32[0].as_slice::<f32>().unwrap(),
        ),
        (
            s_f32.as_slice::<f32>().unwrap(),
            owned_f32[1].as_slice::<f32>().unwrap(),
        ),
        (
            vt_f32.as_slice::<f32>().unwrap(),
            owned_f32[2].as_slice::<f32>().unwrap(),
        ),
    ] {
        assert!(prepared
            .iter()
            .zip(owned)
            .all(|(a, b)| (*a - *b).abs() < 2.0e-5));
    }

    let c32_data = vec![
        Complex32::new(2.0, 1.0),
        Complex32::new(-1.0, 0.5),
        Complex32::new(3.0, -2.0),
        Complex32::new(0.25, 1.5),
    ];
    let input_c32 = Tensor::from_vec_col_major(vec![2, 2], c32_data).unwrap();
    let plan_c32 = backend
        .prepare_svd([2, 2], DType::C32, SvdOptions::default())
        .unwrap();
    let mut workspace_c32 = plan_c32.allocate_workspace(&mut backend).unwrap();
    let mut u_c32 = Tensor::from_vec_col_major(vec![2, 2], vec![Complex32::default(); 4]).unwrap();
    let mut s_c32 = Tensor::from_vec_col_major(vec![2], vec![0.0_f32; 2]).unwrap();
    let mut vt_c32 = Tensor::from_vec_col_major(vec![2, 2], vec![Complex32::default(); 4]).unwrap();
    plan_c32
        .execute_into(
            &mut backend,
            &mut workspace_c32,
            TensorRead::from_tensor(&input_c32),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u_c32),
                TensorWrite::from_tensor(&mut s_c32),
                TensorWrite::from_tensor(&mut vt_c32),
            ),
        )
        .unwrap();
    let owned_c32 = backend.svd(&input_c32).unwrap();
    for (prepared, owned) in [
        (
            u_c32.as_slice::<Complex32>().unwrap(),
            owned_c32[0].as_slice::<Complex32>().unwrap(),
        ),
        (
            vt_c32.as_slice::<Complex32>().unwrap(),
            owned_c32[2].as_slice::<Complex32>().unwrap(),
        ),
    ] {
        assert!(prepared
            .iter()
            .zip(owned)
            .all(|(a, b)| (*a - *b).norm() < 2.0e-5));
    }
    assert!(s_c32
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .zip(owned_c32[1].as_slice::<f32>().unwrap())
        .all(|(a, b)| (*a - *b).abs() < 2.0e-5));

    let c64_data = vec![
        Complex64::new(2.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let input_c64 = Tensor::from_vec_col_major(vec![2, 2], c64_data.clone()).unwrap();
    let plan_c64 = backend
        .prepare_svd([2, 2], DType::C64, SvdOptions::default())
        .unwrap();
    let mut workspace_c64 = plan_c64.allocate_workspace(&mut backend).unwrap();
    let mut u_c64 = Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::default(); 4]).unwrap();
    let mut s_c64 = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let mut vt_c64 = Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::default(); 4]).unwrap();
    plan_c64
        .execute_into(
            &mut backend,
            &mut workspace_c64,
            TensorRead::from_tensor(&input_c64),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u_c64),
                TensorWrite::from_tensor(&mut s_c64),
                TensorWrite::from_tensor(&mut vt_c64),
            ),
        )
        .unwrap();
    assert_c64_svd_residual(
        &c64_data,
        u_c64.as_slice::<Complex64>().unwrap(),
        s_c64.as_slice::<f64>().unwrap(),
        vt_c64.as_slice::<Complex64>().unwrap(),
        2,
        2,
        1.0e-10,
    );
}

#[test]
fn prepared_svd_canonical_complex_gauge_matches_owned_path() {
    let options = SvdOptions::default().gauge(SvdGauge::CanonicalPivot);
    let data = vec![
        Complex64::new(2.0, 1.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(0.25, -1.0),
        Complex64::new(3.0, 2.0),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.5, -0.25),
    ];
    let input = Tensor::from_vec_col_major(vec![3, 2], data.clone()).unwrap();
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend.prepare_svd([3, 2], DType::C64, options).unwrap();
    let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
    let mut u = Tensor::from_vec_col_major(vec![3, 2], vec![Complex64::default(); 6]).unwrap();
    let mut s = Tensor::from_vec_col_major(vec![2], vec![0.0_f64; 2]).unwrap();
    let mut vt = Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::default(); 4]).unwrap();
    plan.execute_into(
        &mut backend,
        &mut workspace,
        TensorRead::from_tensor(&input),
        SvdOutputWrites::new(
            TensorWrite::from_tensor(&mut u),
            TensorWrite::from_tensor(&mut s),
            TensorWrite::from_tensor(&mut vt),
        ),
    )
    .unwrap();
    let owned = backend.svd_with_options(&input, options).unwrap();
    assert_eq!(
        u.as_slice::<Complex64>().unwrap(),
        owned[0].as_slice::<Complex64>().unwrap()
    );
    assert_eq!(
        s.as_slice::<f64>().unwrap(),
        owned[1].as_slice::<f64>().unwrap()
    );
    assert_eq!(
        vt.as_slice::<Complex64>().unwrap(),
        owned[2].as_slice::<Complex64>().unwrap()
    );
    assert_c64_svd_residual(
        &data,
        u.as_slice::<Complex64>().unwrap(),
        s.as_slice::<f64>().unwrap(),
        vt.as_slice::<Complex64>().unwrap(),
        3,
        2,
        1.0e-10,
    );
}

#[test]
fn prepared_svd_f64_raw_reconstructs_square_tall_and_wide_inputs() {
    for (m, n, data) in [
        (2, 2, vec![3.0, 1.0, -2.0, 4.0]),
        (4, 2, vec![3.0, 1.0, -2.0, 0.5, 4.0, -1.0, 2.0, 5.0]),
        (2, 4, vec![3.0, 1.0, -2.0, 4.0, 0.5, -3.0, 2.0, 1.5]),
    ] {
        let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
        let plan = backend
            .prepare_svd([m, n], DType::F64, SvdOptions::default())
            .unwrap();
        let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
        let input = Tensor::from_vec_col_major(vec![m, n], data.clone()).unwrap();
        let (mut u, mut s, mut vt) = f64_outputs(m, n, 0.0);
        plan.execute_into(
            &mut backend,
            &mut workspace,
            TensorRead::from_tensor(&input),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap();
        assert_f64_svd_residual(
            &data,
            u.as_slice::<f64>().unwrap(),
            s.as_slice::<f64>().unwrap(),
            vt.as_slice::<f64>().unwrap(),
            m,
            n,
        );
    }
}

#[test]
fn prepared_svd_f64_accepts_positive_negative_and_zero_stride_inputs() {
    let cases = [
        ([3, 2], [2, 7], 1_isize, 13_usize),
        ([3, 2], [-2, 7], 5_isize, 13_usize),
        ([1, 3], [0, 2], 1_isize, 6_usize),
    ];
    for (shape, strides, offset, storage_len) in cases {
        let [m, n] = shape;
        let mut storage = vec![97.0_f64; storage_len];
        let mut expected = vec![0.0; m * n];
        for col in 0..n {
            for row in 0..m {
                let value = 1.0 + row as f64 + 2.5 * col as f64;
                let physical = offset + row as isize * strides[0] + col as isize * strides[1];
                storage[usize::try_from(physical).unwrap()] = value;
                expected[row + m * col] = value;
            }
        }
        let view = TypedTensorView::from_slice(shape, strides, offset, &storage).unwrap();
        let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
        let plan = backend
            .prepare_svd(shape, DType::F64, SvdOptions::default())
            .unwrap();
        let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
        let (mut u, mut s, mut vt) = f64_outputs(m, n, 0.0);
        plan.execute_into(
            &mut backend,
            &mut workspace,
            TensorRead::from_view(TensorView::F64(view)),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap();
        assert_f64_svd_residual(
            &expected,
            u.as_slice::<f64>().unwrap(),
            s.as_slice::<f64>().unwrap(),
            vt.as_slice::<f64>().unwrap(),
            m,
            n,
        );
    }
}

#[test]
fn prepared_svd_writes_compact_output_subviews_without_touching_guards() {
    let (m, n, k) = (3, 2, 2);
    let input_data = vec![3.0_f64, 1.0, -2.0, 4.0, -1.0, 2.0];
    let input = Tensor::from_vec_col_major(vec![m, n], input_data.clone()).unwrap();
    let mut u_storage = vec![41.0_f64; 2 + m * k + 2];
    let mut s_storage = vec![42.0_f64; 2 + k + 2];
    let mut vt_storage = vec![43.0_f64; 2 + k * n + 2];
    let u_view =
        TypedTensorViewMut::from_slice([m, k], [1, m as isize], 2, &mut u_storage).unwrap();
    let s_view = TypedTensorViewMut::from_slice([k], [1], 2, &mut s_storage).unwrap();
    let vt_view =
        TypedTensorViewMut::from_slice([k, n], [1, k as isize], 2, &mut vt_storage).unwrap();
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend
        .prepare_svd([m, n], DType::F64, SvdOptions::default())
        .unwrap();
    let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
    plan.execute_into(
        &mut backend,
        &mut workspace,
        TensorRead::from_tensor(&input),
        SvdOutputWrites::new(
            TensorWrite::from_view(TensorViewMut::F64(u_view)),
            TensorWrite::from_view(TensorViewMut::F64(s_view)),
            TensorWrite::from_view(TensorViewMut::F64(vt_view)),
        ),
    )
    .unwrap();
    assert_eq!(&u_storage[..2], &[41.0; 2]);
    assert_eq!(&u_storage[2 + m * k..], &[41.0; 2]);
    assert_eq!(&s_storage[..2], &[42.0; 2]);
    assert_eq!(&s_storage[2 + k..], &[42.0; 2]);
    assert_eq!(&vt_storage[..2], &[43.0; 2]);
    assert_eq!(&vt_storage[2 + k * n..], &[43.0; 2]);
    assert_f64_svd_residual(
        &input_data,
        &u_storage[2..2 + m * k],
        &s_storage[2..2 + k],
        &vt_storage[2..2 + k * n],
        m,
        n,
    );
}

#[test]
fn prepared_svd_reports_compact_output_specs() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend
        .prepare_svd([3, 2], DType::C64, SvdOptions::default())
        .unwrap();
    assert_eq!(plan.output_specs().u().shape(), &[3, 2]);
    assert_eq!(plan.output_specs().u().dtype(), DType::C64);
    assert_eq!(plan.output_specs().s().shape(), &[2]);
    assert_eq!(plan.output_specs().s().dtype(), DType::F64);
    assert_eq!(plan.output_specs().vt().shape(), &[2, 2]);
}

#[cfg(feature = "cpu-blas")]
#[test]
fn prepared_svd_rejects_unsupported_cpu_provider_without_fallback() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Blas).unwrap();
    let error = backend
        .prepare_svd([2, 2], DType::F64, SvdOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Error::UnsupportedCapability {
            op: "prepare_svd",
            backend: BackendId::Cpu,
            provider: "blas",
            dtype: DType::F64,
            capability: "prepared compact SVD",
        }
    ));
}

#[test]
fn prepared_svd_reports_unsupported_dtype_as_typed_capability() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let error = backend
        .prepare_svd([2, 2], DType::I32, SvdOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        Error::UnsupportedCapability {
            op: "prepare_svd",
            backend: BackendId::Cpu,
            provider: "faer",
            dtype: DType::I32,
            capability: "prepared compact SVD",
        }
    ));
}

#[test]
fn prepared_svd_binding_accepts_clone_and_rejects_distinct_backend_before_writes() {
    let mut creator = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = creator
        .prepare_svd([2, 2], DType::F64, SvdOptions::default())
        .unwrap();
    let mut workspace = plan.allocate_workspace(&mut creator).unwrap();
    let mut clone = creator.clone();
    drop(creator);

    let input = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 2.0]).unwrap();
    let (mut u, mut s, mut vt) = f64_outputs(2, 2, -7.0);
    plan.execute_into(
        &mut clone,
        &mut workspace,
        TensorRead::from_tensor(&input),
        SvdOutputWrites::new(
            TensorWrite::from_tensor(&mut u),
            TensorWrite::from_tensor(&mut s),
            TensorWrite::from_tensor(&mut vt),
        ),
    )
    .unwrap();
    assert_ne!(s.as_slice::<f64>().unwrap(), &[-7.0, -7.0]);

    let mut distinct = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let before_u = vec![-11.0; 4];
    let before_s = vec![-11.0; 2];
    let before_vt = vec![-11.0; 4];
    let mut u = Tensor::from_vec_col_major(vec![2, 2], before_u.clone()).unwrap();
    let mut s = Tensor::from_vec_col_major(vec![2], before_s.clone()).unwrap();
    let mut vt = Tensor::from_vec_col_major(vec![2, 2], before_vt.clone()).unwrap();
    let error = plan
        .execute_into(
            &mut distinct,
            &mut workspace,
            TensorRead::from_tensor(&input),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::UnsupportedCapability {
            op: "PreparedSvd::execute_into",
            backend: BackendId::Cpu,
            provider: "faer",
            dtype: DType::F64,
            capability: "prepared SVD backend/context binding",
        }
    ));
    assert_eq!(u.as_slice::<f64>().unwrap(), before_u);
    assert_eq!(s.as_slice::<f64>().unwrap(), before_s);
    assert_eq!(vt.as_slice::<f64>().unwrap(), before_vt);
}

#[test]
fn prepared_svd_validation_failure_leaves_all_destinations_unchanged() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend
        .prepare_svd([2, 2], DType::F64, SvdOptions::default())
        .unwrap();
    let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4]).unwrap();
    let mut u = Tensor::from_vec_col_major(vec![4], vec![19.0_f64; 4]).unwrap();
    let mut s = Tensor::from_vec_col_major(vec![2], vec![19.0_f64; 2]).unwrap();
    let mut vt = Tensor::from_vec_col_major(vec![2, 2], vec![19.0_f64; 4]).unwrap();
    assert!(plan
        .execute_into(
            &mut backend,
            &mut workspace,
            TensorRead::from_tensor(&input),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .is_err());
    assert_eq!(u.as_slice::<f64>().unwrap(), &[19.0; 4]);
    assert_eq!(s.as_slice::<f64>().unwrap(), &[19.0; 2]);
    assert_eq!(vt.as_slice::<f64>().unwrap(), &[19.0; 4]);
}

#[test]
fn prepared_svd_rejects_unsupported_destination_stride_before_writes() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend
        .prepare_svd([2, 1], DType::F64, SvdOptions::default())
        .unwrap();
    let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
    let input = Tensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 1.0]).unwrap();
    let mut u_storage = vec![31.0_f64; 3];
    let before_u = u_storage.clone();
    let u = TypedTensorViewMut::from_slice([2, 1], [2, 3], 0, &mut u_storage).unwrap();
    let mut s = Tensor::from_vec_col_major(vec![1], vec![32.0_f64]).unwrap();
    let mut vt = Tensor::from_vec_col_major(vec![1, 1], vec![33.0_f64]).unwrap();
    let error = plan
        .execute_into(
            &mut backend,
            &mut workspace,
            TensorRead::from_tensor(&input),
            SvdOutputWrites::new(
                TensorWrite::from_view(TensorViewMut::F64(u)),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::UnsupportedCapability {
            op: "PreparedSvd::execute_into",
            backend: BackendId::Cpu,
            provider: "faer",
            dtype: DType::F64,
            capability: "compact column-major prepared SVD destination",
        }
    ));
    assert_eq!(u_storage, before_u);
    assert_eq!(s.as_slice::<f64>().unwrap(), &[32.0]);
    assert_eq!(vt.as_slice::<f64>().unwrap(), &[33.0]);
}

#[test]
fn prepared_svd_supports_multiple_workspaces_and_rejects_cross_plan_workspace() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend
        .prepare_svd([2, 2], DType::F64, SvdOptions::default())
        .unwrap();
    let other_plan = backend
        .prepare_svd([2, 2], DType::F64, SvdOptions::default())
        .unwrap();
    let mut first_workspace = plan.allocate_workspace(&mut backend).unwrap();
    let mut second_workspace = plan.allocate_workspace(&mut backend).unwrap();
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![3.0_f64, 0.0, 0.0, 2.0]).unwrap();

    for workspace in [&mut first_workspace, &mut second_workspace] {
        let (mut u, mut s, mut vt) = f64_outputs(2, 2, -1.0);
        plan.execute_into(
            &mut backend,
            workspace,
            TensorRead::from_tensor(&input),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap();
        assert_f64_svd_residual(
            input.as_slice::<f64>().unwrap(),
            u.as_slice::<f64>().unwrap(),
            s.as_slice::<f64>().unwrap(),
            vt.as_slice::<f64>().unwrap(),
            2,
            2,
        );
    }

    let (mut u, mut s, mut vt) = f64_outputs(2, 2, 37.0);
    let error = other_plan
        .execute_into(
            &mut backend,
            &mut first_workspace,
            TensorRead::from_tensor(&input),
            SvdOutputWrites::new(
                TensorWrite::from_tensor(&mut u),
                TensorWrite::from_tensor(&mut s),
                TensorWrite::from_tensor(&mut vt),
            ),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        Error::UnsupportedCapability {
            capability: "prepared SVD backend/context binding",
            ..
        }
    ));
    assert_eq!(u.as_slice::<f64>().unwrap(), &[37.0; 4]);
    assert_eq!(s.as_slice::<f64>().unwrap(), &[37.0; 2]);
    assert_eq!(vt.as_slice::<f64>().unwrap(), &[37.0; 4]);
}

#[test]
fn prepared_svd_rejects_every_alias_pair_before_writes() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend
        .prepare_svd([2, 1], DType::F64, SvdOptions::default())
        .unwrap();
    let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
    let cases = [
        ("input and U", [0_usize, 0, 8, 12]),
        ("input and S", [0, 4, 0, 12]),
        ("input and Vt", [0, 4, 8, 0]),
        ("U and S", [0, 4, 4, 12]),
        ("U and Vt", [0, 4, 8, 4]),
        ("S and Vt", [0, 4, 8, 8]),
    ];
    for (expected_pair, [input_offset, u_offset, s_offset, vt_offset]) in cases {
        let mut shared = vec![23.0_f64; 16];
        shared[input_offset] = 2.0;
        shared[input_offset + 1] = 1.0;
        let before = shared.clone();
        let ptr = shared.as_mut_ptr();
        // SAFETY: each case deliberately creates exactly one overlapping pair.
        // The operation must detect pointer regions before provider or output access.
        let input_storage = unsafe { std::slice::from_raw_parts(ptr.add(input_offset), 2) };
        // SAFETY: see the deliberate preflight-only alias contract above.
        let u_storage = unsafe { std::slice::from_raw_parts_mut(ptr.add(u_offset), 2) };
        // SAFETY: see the deliberate preflight-only alias contract above.
        let s_storage = unsafe { std::slice::from_raw_parts_mut(ptr.add(s_offset), 1) };
        // SAFETY: see the deliberate preflight-only alias contract above.
        let vt_storage = unsafe { std::slice::from_raw_parts_mut(ptr.add(vt_offset), 1) };
        let input = TypedTensorView::from_slice([2, 1], [1, 2], 0, input_storage).unwrap();
        let u = TypedTensorViewMut::from_slice([2, 1], [1, 2], 0, u_storage).unwrap();
        let s = TypedTensorViewMut::from_slice([1], [1], 0, s_storage).unwrap();
        let vt = TypedTensorViewMut::from_slice([1, 1], [1, 1], 0, vt_storage).unwrap();
        let error = plan
            .execute_into(
                &mut backend,
                &mut workspace,
                TensorRead::from_view(TensorView::F64(input)),
                SvdOutputWrites::new(
                    TensorWrite::from_view(TensorViewMut::F64(u)),
                    TensorWrite::from_view(TensorViewMut::F64(s)),
                    TensorWrite::from_view(TensorViewMut::F64(vt)),
                ),
            )
            .unwrap_err();
        assert!(
            error.to_string().contains(expected_pair),
            "expected {expected_pair}, got {error}"
        );
        assert_eq!(shared, before, "{expected_pair} changed backing storage");
    }
}

#[test]
fn prepared_svd_zero_size_validates_and_skips_provider() {
    let mut backend = CpuBackend::with_kind(CpuBackendKind::Faer).unwrap();
    let plan = backend
        .prepare_svd([0, 3], DType::F64, SvdOptions::default())
        .unwrap();
    let mut workspace = plan.allocate_workspace(&mut backend).unwrap();
    let input_data: [f64; 0] = [];
    let input = TypedTensorView::from_slice([0, 3], [1, 0], 0, &input_data).unwrap();
    let (mut u, mut s, mut vt) = f64_outputs(0, 3, 0.0);
    plan.execute_into(
        &mut backend,
        &mut workspace,
        TensorRead::from_view(TensorView::F64(input)),
        SvdOutputWrites::new(
            TensorWrite::from_tensor(&mut u),
            TensorWrite::from_tensor(&mut s),
            TensorWrite::from_tensor(&mut vt),
        ),
    )
    .unwrap();
    assert!(u.as_slice::<f64>().unwrap().is_empty());
    assert!(s.as_slice::<f64>().unwrap().is_empty());
    assert!(vt.as_slice::<f64>().unwrap().is_empty());
}
