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
fn householder_qr_ad_rejects_before_oracles() {
    use tenferro_ad::AdContext;
    use tenferro_linalg::TracedTensorLinalgExt;
    use tenferro_runtime::TracedTensor;

    let a = TracedTensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 1.0])
        .unwrap();
    let r = a.householder_qr().unwrap().r(QrOptions::default()).unwrap();
    let loss = r.reduce_sum(None).unwrap();
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules().unwrap())
        .unwrap()
        .build()
        .unwrap();
    let error = ad
        .grad(&loss, &a)
        .expect_err("incremental QR AD must reject until its oracle-backed rules land");
    assert!(error.to_string().contains("unsupported"));
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
