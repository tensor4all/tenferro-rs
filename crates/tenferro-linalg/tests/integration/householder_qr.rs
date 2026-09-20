use std::ops::{Add, Mul, Sub};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuBackend;
use tenferro_linalg::{HouseholderQr, QrOptions, TensorLinalgExt};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorScalar};

fn product(a: &[f64], rows: usize, inner: usize, b: &[f64], cols: usize) -> Vec<f64> {
    let mut output = vec![0.0; rows * cols];
    for col in 0..cols {
        for k in 0..inner {
            for row in 0..rows {
                output[row + col * rows] += a[row + k * rows] * b[k + col * inner];
            }
        }
    }
    output
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert_eq!(actual.len(), expected.len());
    assert!(error < 1.0e-10, "maximum reconstruction error: {error}");
}

trait SampleScalar:
    TensorScalar + Copy + Default + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
{
    fn from_parts(real: f64, imaginary: f64) -> Self;
    fn from_real(value: f64) -> Self {
        Self::from_parts(value, 0.0)
    }
    fn magnitude(self) -> f64;
}

macro_rules! real_sample {
    ($scalar:ty) => {
        impl SampleScalar for $scalar {
            fn from_parts(real: f64, _imaginary: f64) -> Self {
                real as Self
            }
            fn magnitude(self) -> f64 {
                self.abs() as f64
            }
        }
    };
}
real_sample!(f32);
real_sample!(f64);

macro_rules! complex_sample {
    ($scalar:ty, $real:ty) => {
        impl SampleScalar for $scalar {
            fn from_parts(real: f64, imaginary: f64) -> Self {
                Self::new(real as $real, imaginary as $real)
            }
            fn magnitude(self) -> f64 {
                self.norm() as f64
            }
        }
    };
}
complex_sample!(Complex32, f32);
complex_sample!(Complex64, f64);

fn check_factor_dtype<T: SampleScalar>() {
    let values = [1.0, 2.0, 3.0, 2.0, -1.0, 4.0]
        .into_iter()
        .map(T::from_real)
        .collect::<Vec<_>>();
    let input = Tensor::from_vec_col_major(vec![3, 2], values.clone()).unwrap();
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|session| {
        let state = input.householder_qr(session).unwrap();
        let q = state
            .q_columns(0..2, QrOptions::default(), session)
            .unwrap();
        let r = state.r(QrOptions::default(), session).unwrap();
        let reconstructed = product_generic(
            q.as_slice::<T>().unwrap(),
            3,
            2,
            r.as_slice::<T>().unwrap(),
            2,
        );
        let error = reconstructed
            .iter()
            .zip(values)
            .map(|(actual, expected)| (*actual - expected).magnitude())
            .fold(0.0, f64::max);
        assert!(error < 2.0e-5, "maximum reconstruction error: {error}");
    });
}

fn product_generic<T: SampleScalar>(
    a: &[T],
    rows: usize,
    inner: usize,
    b: &[T],
    cols: usize,
) -> Vec<T> {
    let mut output = vec![T::default(); rows * cols];
    for col in 0..cols {
        for k in 0..inner {
            for row in 0..rows {
                output[row + col * rows] =
                    output[row + col * rows] + a[row + k * rows] * b[k + col * inner];
            }
        }
    }
    output
}

#[test]
fn compact_qr_reconstructs_all_supported_dtypes() {
    check_factor_dtype::<f32>();
    check_factor_dtype::<f64>();
    check_factor_dtype::<Complex32>();
    check_factor_dtype::<Complex64>();
}

fn check_append_dtype<T: SampleScalar>() {
    let a_values = [(1.0, 0.5), (2.0, -1.0), (3.0, 0.25)]
        .into_iter()
        .map(|(real, imaginary)| T::from_parts(real, imaginary))
        .collect::<Vec<_>>();
    let b_values = [(0.5, -0.75), (-1.0, 0.5), (2.0, 1.25)]
        .into_iter()
        .map(|(real, imaginary)| T::from_parts(real, imaginary))
        .collect::<Vec<_>>();
    let expected = [a_values.as_slice(), b_values.as_slice()].concat();
    let a = Tensor::from_vec_col_major(vec![3, 1], a_values).unwrap();
    let b = Tensor::from_vec_col_major(vec![3, 1], b_values).unwrap();
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|session| {
        let state = a
            .householder_qr(session)
            .unwrap()
            .append_columns(&b, session)
            .unwrap();
        let q = state
            .q_columns(0..2, QrOptions::default(), session)
            .unwrap();
        let r = state.r(QrOptions::default(), session).unwrap();
        let reconstructed = product_generic(
            q.as_slice::<T>().unwrap(),
            3,
            2,
            r.as_slice::<T>().unwrap(),
            2,
        );
        let error = reconstructed
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual - expected).magnitude())
            .fold(0.0, f64::max);
        assert!(
            error < 2.0e-5,
            "maximum append reconstruction error: {error}"
        );
    });
}

#[test]
fn compact_qr_append_reconstructs_all_supported_dtypes() {
    check_append_dtype::<f32>();
    check_append_dtype::<f64>();
    check_append_dtype::<Complex32>();
    check_append_dtype::<Complex64>();
}

#[test]
fn concrete_compact_qr_appends_and_reconstructs() {
    let a =
        Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0])
            .unwrap();
    let b = Tensor::from_vec_col_major(
        vec![4, 2],
        vec![3.0_f64, -1.0, 2.0, 1.0, 0.5, 2.0, -2.0, 4.0],
    )
    .unwrap();
    let expected = [a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap()].concat();
    let mut backend = CpuBackend::new();

    backend
        .with_backend_session(|session| {
            let state = a.householder_qr(session)?.append_columns(&b, session)?;
            // A handle renders its own surface without walking the factors it holds.
            let rendered = format!("{state:?}");
            assert!(rendered.contains("HouseholderQr"), "{rendered}");
            let q = state.q_columns(0..4, QrOptions::default(), session)?;
            let r = state.r(QrOptions::default(), session)?;
            assert_close(
                &product(q.as_slice::<f64>()?, 4, 4, r.as_slice::<f64>()?, 4),
                &expected,
            );
            Ok::<(), tenferro_tensor::Error>(())
        })
        .unwrap();
}

#[test]
fn rank_deficient_zero_append_and_tall_to_wide_transition_reconstruct() {
    let a = Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 2.0, 3.0, 2.0, 4.0, 6.0]).unwrap();
    let empty = Tensor::from_vec_col_major(vec![3, 0], Vec::<f64>::new()).unwrap();
    let b =
        Tensor::from_vec_col_major(vec![3, 2], vec![0.0_f64, 1.0, -1.0, 2.0, 0.5, 3.0]).unwrap();
    let expected = [a.as_slice::<f64>().unwrap(), b.as_slice::<f64>().unwrap()].concat();
    let mut backend = CpuBackend::new();

    backend
        .with_backend_session(|session| {
            let state = a
                .householder_qr(session)?
                .append_columns(&empty, session)?
                .append_columns(&b, session)?;
            let q = state.q_columns(0..3, QrOptions::default(), session)?;
            let r = state.r(QrOptions::default(), session)?;
            assert_close(
                &product(q.as_slice::<f64>()?, 3, 3, r.as_slice::<f64>()?, 4),
                &expected,
            );
            Ok::<(), tenferro_tensor::Error>(())
        })
        .unwrap();
}

#[test]
fn concrete_from_factors_requires_upper_trapezoidal_r() {
    let q =
        Tensor::from_vec_col_major(vec![4, 2], vec![1.0_f64, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0])
            .unwrap();
    let invalid_r = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 3.0, 4.0]).unwrap();
    let mut backend = CpuBackend::new();

    backend.with_backend_session(|session| {
        let error = HouseholderQr::<Tensor>::from_factors(&q, &invalid_r, session)
            .expect_err("non-trapezoidal R must be rejected");
        assert!(matches!(error, tenferro_tensor::Error::Validation { .. }));
    });
}

#[test]
fn traced_compact_qr_preserves_known_shapes() {
    use tenferro_linalg::TracedTensorLinalgExt;
    use tenferro_runtime::{GraphCompiler, TracedTensor};

    let a = TracedTensor::from_vec_col_major(
        vec![4, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0],
    )
    .unwrap();
    let b = TracedTensor::from_vec_col_major(vec![4, 1], vec![3.0_f64, -1.0, 2.0, 1.0]).unwrap();
    let state = a.householder_qr().unwrap().append_columns(&b).unwrap();
    let q = state.q_columns(1..3, QrOptions::default()).unwrap();
    let r = state.r(QrOptions::default()).unwrap();
    let program = GraphCompiler::new().compile_many(&[&q, &r]).unwrap();
    let outputs = super::support::run_all(&program, &[]).unwrap();
    assert_eq!(outputs[0].shape(), &[4, 2]);
    assert_eq!(outputs[1].shape(), &[3, 3]);
}

#[cfg(feature = "autodiff")]
#[test]
fn householder_qr_r_grad_matches_finite_difference() {
    use tenferro_ad::AdContext;
    use tenferro_linalg::TracedTensorLinalgExt;
    use tenferro_runtime::{GraphCompiler, TracedTensor};

    let values = vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0];
    let a = TracedTensor::from_vec_col_major(vec![3, 2], values.clone()).unwrap();
    let r = a.householder_qr().unwrap().r(QrOptions::default()).unwrap();
    let loss = r.reduce_sum(None).unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    let grad = ad.grad(&loss, &a).unwrap();
    let program = GraphCompiler::new().compile(&grad).unwrap();
    let actual = super::support::run_all(&program, &[]).unwrap().remove(0);
    let actual = actual.as_slice::<f64>().unwrap();

    let scalar_loss = |data: Vec<f64>| {
        let input = Tensor::from_vec_col_major(vec![3, 2], data).unwrap();
        let mut backend = CpuBackend::new();
        backend.with_backend_session(|session| {
            let r = input
                .householder_qr(session)
                .unwrap()
                .r(QrOptions::default(), session)
                .unwrap();
            r.as_slice::<f64>().unwrap().iter().sum::<f64>()
        })
    };
    let step = 1.0e-6;
    for (index, &gradient) in actual.iter().enumerate() {
        let mut plus = values.clone();
        let mut minus = values.clone();
        plus[index] += step;
        minus[index] -= step;
        let expected = (scalar_loss(plus) - scalar_loss(minus)) / (2.0 * step);
        assert!(
            (gradient - expected).abs() < 2.0e-5,
            "gradient[{index}] expected {expected}, got {gradient}"
        );
    }
}

#[cfg(feature = "autodiff")]
#[test]
fn householder_qr_two_appends_produce_all_input_gradients() {
    use tenferro_ad::AdContext;
    use tenferro_linalg::{QrGauge, TracedTensorLinalgExt};
    use tenferro_runtime::{GraphCompiler, TracedTensor};

    let a = TracedTensor::from_vec_col_major(vec![4, 1], vec![1.0_f64, 2.0, 0.0, -1.0]).unwrap();
    let b = TracedTensor::from_vec_col_major(vec![4, 1], vec![0.0_f64, 1.0, 2.0, 1.0]).unwrap();
    let c = TracedTensor::from_vec_col_major(vec![4, 1], vec![2.0_f64, -1.0, 1.0, 0.5]).unwrap();
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    let state = a
        .householder_qr()
        .unwrap()
        .append_columns(&b)
        .unwrap()
        .append_columns(&c)
        .unwrap();
    let q = state.q_columns(0..3, options).unwrap();
    let r = state.r(options).unwrap();
    let loss = q
        .reduce_sum(Some(&[0, 1]))
        .unwrap()
        .add(&r.reduce_sum(Some(&[0, 1])).unwrap())
        .unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    let gradients = [
        ad.grad(&loss, &a).unwrap(),
        ad.grad(&loss, &b).unwrap(),
        ad.grad(&loss, &c).unwrap(),
    ];
    let refs = gradients.iter().collect::<Vec<_>>();
    let program = GraphCompiler::new().compile_many(&refs).unwrap();
    let outputs = super::support::run_all(&program, &[]).unwrap();
    for output in outputs {
        assert_eq!(output.shape(), &[4, 1]);
        assert!(output
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
    }
}

#[cfg(feature = "autodiff")]
#[test]
fn eager_compact_qr_executes_on_cpu() {
    use tenferro_ad::{EagerRuntime, EagerTensor};
    use tenferro_linalg::EagerTensorLinalgExt;

    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap();
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0]).unwrap(),
        runtime,
    )
    .unwrap();
    let state = a.householder_qr().unwrap();
    assert_eq!(state.r(QrOptions::default()).unwrap().shape(), &[2, 2]);
    assert_eq!(
        state.q_columns(0..2, QrOptions::default()).unwrap().shape(),
        &[3, 2]
    );
}

// ---------------------------------------------------------------------------
// Full-Q width. Columns `k..m` of Q span the orthogonal complement of the
// input's column space, which the thin factor cannot represent. They come from
// the same compact reflectors, so no extra factorization is needed.
// ---------------------------------------------------------------------------

/// Every CPU linalg provider compiled into this build.
fn full_q_providers() -> Vec<(&'static str, CpuBackend)> {
    const KINDS: &[(&str, tenferro_cpu::CpuBackendKind)] = &[
        #[cfg(feature = "cpu-faer")]
        ("faer", tenferro_cpu::CpuBackendKind::Faer),
        #[cfg(feature = "cpu-blas")]
        ("blas", tenferro_cpu::CpuBackendKind::Blas),
    ];
    KINDS
        .iter()
        .map(|&(name, kind)| {
            let backend = CpuBackend::with_threads_and_kind(1, kind)
                .unwrap_or_else(|error| panic!("{name} CPU backend: {error}"));
            (name, backend)
        })
        .collect()
}

/// A tall, full-column-rank sample whose complement is two-dimensional.
fn tall_sample(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = (index % m) as f64;
            let col = (index / m) as f64;
            1.0 + row * 1.25 - col * 0.5 + row * col * 0.125
        })
        .collect()
}

#[test]
fn full_q_columns_are_orthonormal_and_complete_the_thin_factor() {
    let (m, n) = (5_usize, 3_usize);
    let data = tall_sample(m, n);
    for (provider, mut host) in full_q_providers() {
        let input = Tensor::from_vec_col_major(vec![m, n], data.clone()).unwrap();
        host.with_backend_session(|session| {
            let state = input.householder_qr(session).unwrap();
            let full = state
                .q_columns(0..m, QrOptions::default(), session)
                .unwrap();
            assert_eq!(full.shape(), &[m, m], "{provider}: full-Q shape");
            let full_data = full.as_slice::<f64>().unwrap();

            // `Qᵀ Q = I (m x m)`: the identity a thin factor cannot satisfy.
            for left in 0..m {
                for right in 0..m {
                    let inner: f64 = (0..m)
                        .map(|row| full_data[row + left * m] * full_data[row + right * m])
                        .sum();
                    let expected = if left == right { 1.0 } else { 0.0 };
                    assert!(
                        (inner - expected).abs() < 1.0e-10,
                        "{provider}: column pair ({left}, {right}) inner product {inner}"
                    );
                }
            }

            // The leading columns still are the thin factor.
            let thin = state
                .q_columns(0..n, QrOptions::default(), session)
                .unwrap();
            assert_eq!(thin.shape(), &[m, n]);
            assert_close(thin.as_slice::<f64>().unwrap(), &full_data[..m * n]);

            // `Aᵀ Q[:, k..m] = 0`: the complement is the input's left nullspace.
            for col in n..m {
                for input_col in 0..n {
                    let inner: f64 = (0..m)
                        .map(|row| data[row + input_col * m] * full_data[row + col * m])
                        .sum();
                    assert!(
                        inner.abs() < 1.0e-10,
                        "{provider}: complement column {col} is not orthogonal to input column \
                         {input_col}: {inner}"
                    );
                }
            }

            // A complement-only range is reachable directly.
            let complement = state
                .q_columns(n..m, QrOptions::default(), session)
                .unwrap();
            assert_eq!(complement.shape(), &[m, m - n]);
            assert_close(
                complement.as_slice::<f64>().unwrap(),
                &full_data[n * m..m * m],
            );
        });
    }
}

#[test]
fn full_q_positive_diagonal_gauge_fixes_only_the_thin_columns() {
    // The gauge is defined by R's diagonal, so it is a no-op for the
    // complement: the gauged and ungauged complement columns must be identical
    // while the leading columns may be re-phased.
    let (m, n) = (4_usize, 2_usize);
    for (provider, mut host) in full_q_providers() {
        let input = Tensor::from_vec_col_major(vec![m, n], tall_sample(m, n)).unwrap();
        host.with_backend_session(|session| {
            let state = input.householder_qr(session).unwrap();
            let raw = state
                .q_columns(0..m, QrOptions::default(), session)
                .unwrap();
            let gauged = state
                .q_columns(
                    0..m,
                    QrOptions::default().gauge(tenferro_linalg::QrGauge::PositiveDiagonal),
                    session,
                )
                .unwrap();
            assert_eq!(gauged.shape(), &[m, m]);
            assert_close(
                &gauged.as_slice::<f64>().unwrap()[n * m..],
                &raw.as_slice::<f64>().unwrap()[n * m..],
            );

            // The gauged full factor is still orthonormal.
            let gauged_data = gauged.as_slice::<f64>().unwrap();
            for col in 0..m {
                let norm: f64 = (0..m).map(|row| gauged_data[row + col * m].powi(2)).sum();
                assert!(
                    (norm - 1.0).abs() < 1.0e-10,
                    "{provider}: gauged column {col} norm {norm}"
                );
            }
        });
    }
}

#[test]
fn square_and_empty_full_q_ranges_stay_consistent() {
    for (provider, mut host) in full_q_providers() {
        // Square input: full Q and thin Q coincide (k == m).
        let square = Tensor::from_vec_col_major(vec![3, 3], tall_sample(3, 3)).unwrap();
        host.with_backend_session(|session| {
            let state = square.householder_qr(session).unwrap();
            let full = state
                .q_columns(0..3, QrOptions::default(), session)
                .unwrap();
            assert_eq!(full.shape(), &[3, 3], "{provider}: square full-Q shape");

            // An empty range is still a valid request.
            let empty = state
                .q_columns(2..2, QrOptions::default(), session)
                .unwrap();
            assert_eq!(empty.shape(), &[3, 0]);
        });

        // Wide input: k == m, so the full-Q width is m and there is no complement.
        let wide = Tensor::from_vec_col_major(vec![2, 4], tall_sample(2, 4)).unwrap();
        host.with_backend_session(|session| {
            let state = wide.householder_qr(session).unwrap();
            let full = state
                .q_columns(0..2, QrOptions::default(), session)
                .unwrap();
            assert_eq!(full.shape(), &[2, 2], "{provider}: wide full-Q shape");
        });
    }
}

#[test]
fn q_column_ranges_past_the_full_q_width_are_rejected() {
    let (m, n) = (4_usize, 2_usize);
    for (provider, mut host) in full_q_providers() {
        let input = Tensor::from_vec_col_major(vec![m, n], tall_sample(m, n)).unwrap();
        host.with_backend_session(|session| {
            let state = input.householder_qr(session).unwrap();
            let error = state
                .q_columns(0..m + 1, QrOptions::default(), session)
                .unwrap_err();
            assert!(
                format!("{error}").contains("range"),
                "{provider}: expected a range validation error, got {error}"
            );
            // An inverted range is still a range error, not a silent empty
            // result. `Range` is built explicitly because a literal `3..1`
            // trips the reversed-empty-range lint.
            let inverted = state
                .q_columns(
                    std::ops::Range { start: 3, end: 1 },
                    QrOptions::default(),
                    session,
                )
                .unwrap_err();
            assert!(format!("{inverted}").contains("range"));
        });
    }
}

#[test]
fn traced_full_q_columns_carry_the_full_width_shape() {
    use tenferro_linalg::TracedTensorLinalgExt;
    use tenferro_runtime::{GraphCompiler, TracedTensor};

    let a = TracedTensor::from_vec_col_major(
        vec![4, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0],
    )
    .unwrap();
    let state = a.householder_qr().unwrap();
    let full = state.q_columns(0..4, QrOptions::default()).unwrap();
    let complement = state.q_columns(2..4, QrOptions::default()).unwrap();
    let program = GraphCompiler::new()
        .compile_many(&[&full, &complement])
        .unwrap();
    let outputs = super::support::run_all(&program, &[]).unwrap();
    assert_eq!(outputs[0].shape(), &[4, 4]);
    assert_eq!(outputs[1].shape(), &[4, 2]);

    // `Aᵀ Q[:, 2..4] = 0` through the traced surface too.
    let input = [1.0_f64, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0];
    let complement = outputs[1].as_slice::<f64>().unwrap();
    for col in 0..2 {
        for input_col in 0..2 {
            let inner: f64 = (0..4)
                .map(|row| input[row + input_col * 4] * complement[row + col * 4])
                .sum();
            assert!(inner.abs() < 1.0e-10, "traced complement residual {inner}");
        }
    }
}

#[cfg(feature = "autodiff")]
#[test]
fn differentiating_through_complement_q_columns_is_refused() {
    use tenferro_ad::AdContext;
    use tenferro_linalg::TracedTensorLinalgExt;
    use tenferro_runtime::TracedTensor;

    // The complement basis is only defined up to a rotation inside the
    // nullspace, and the thin `dQ` this rule builds has no column there, so the
    // AD rule must refuse rather than return a silently wrong derivative.
    let a = TracedTensor::from_vec_col_major(
        vec![4, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0, 2.0, 0.0, 1.0, 3.0],
    )
    .unwrap();
    let state = a.householder_qr().unwrap();
    let q = state.q_columns(0..4, QrOptions::default()).unwrap();
    let loss = q.reduce_sum(Some(&[0, 1])).unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    let error = ad
        .grad(&loss, &a)
        .expect_err("full-Q differentiation must be refused");
    let text = format!("{error}");
    assert!(
        text.contains("householder_qr_q_columns"),
        "expected a typed q_columns AD refusal, got {text}"
    );

    // The thin range still differentiates.
    let thin = state.q_columns(0..2, QrOptions::default()).unwrap();
    let thin_loss = thin.reduce_sum(Some(&[0, 1])).unwrap();
    ad.grad(&thin_loss, &a)
        .expect("thin-Q differentiation must keep working");
}
