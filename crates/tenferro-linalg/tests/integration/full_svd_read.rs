//! Public borrowed and concrete full-matrices SVD (`svd_full` / `svd_full_read`).
//!
//! Full SVD is currently implemented by the CPU faer provider only, so the
//! numeric tests build a faer backend explicitly instead of relying on the
//! default provider selection. The LAPACK boundary is covered separately in
//! `full_svd_lstsq.rs`.

#![cfg(feature = "cpu-faer")]

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::{TensorLinalgExt, TensorReadLinalgExt, TypedTensorLinalgExt};
use tenferro_tensor::{
    BackendSessionHost, DType, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorView,
    TypedTensor,
};

fn faer_backend() -> CpuBackend {
    CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).expect("faer CPU backend")
}

/// Deterministic, well-scaled column-major test data with no repeated singular
/// values, so the factors are unique up to phase and the checks below are sharp.
fn sample_real(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = (index % m) as f64;
            let col = (index / m) as f64;
            1.0 + row * 1.5 - col * 0.75 + (row * col) * 0.25
        })
        .collect()
}

fn sample_imag(m: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|index| {
            let row = (index % m) as f64;
            let col = (index / m) as f64;
            0.5 - row * 0.25 + col * 0.375
        })
        .collect()
}

/// Build a column-major tensor of `dtype` from real/imaginary parts.
fn tensor_of(dtype: DType, shape: &[usize], real: &[f64], imag: &[f64]) -> Tensor {
    match dtype {
        DType::F32 => Tensor::from_typed::<f32>(
            TypedTensor::from_vec_col_major(
                shape.to_vec(),
                real.iter().map(|&value| value as f32).collect(),
            )
            .unwrap(),
        ),
        DType::F64 => Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(shape.to_vec(), real.to_vec()).unwrap(),
        ),
        DType::C32 => Tensor::from_typed::<Complex32>(
            TypedTensor::from_vec_col_major(
                shape.to_vec(),
                real.iter()
                    .zip(imag)
                    .map(|(&re, &im)| Complex32::new(re as f32, im as f32))
                    .collect(),
            )
            .unwrap(),
        ),
        DType::C64 => Tensor::from_typed::<Complex64>(
            TypedTensor::from_vec_col_major(
                shape.to_vec(),
                real.iter()
                    .zip(imag)
                    .map(|(&re, &im)| Complex64::new(re, im))
                    .collect(),
            )
            .unwrap(),
        ),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

/// Read any supported factor dtype as complex values, so one checker covers all four.
fn complex_values(tensor: &Tensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DType::F32 => tensor
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| Complex64::new(value as f64, 0.0))
            .collect(),
        DType::F64 => tensor
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        DType::C32 => tensor
            .as_slice::<Complex32>()
            .unwrap()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        DType::C64 => tensor.as_slice::<Complex64>().unwrap().to_vec(),
        other => panic!("unsupported factor dtype {other:?}"),
    }
}

/// Singular values keep the real counterpart dtype of the input.
fn real_values(tensor: &Tensor) -> Vec<f64> {
    match tensor.dtype() {
        DType::F32 => tensor
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&value| value as f64)
            .collect(),
        DType::F64 => tensor.as_slice::<f64>().unwrap().to_vec(),
        other => panic!("singular values must be real, got {other:?}"),
    }
}

fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => 2.0e-4,
        _ => 1.0e-10,
    }
}

/// `lhs^H lhs == I` for a `rows x cols` column-major matrix.
fn assert_isometric(matrix: &[Complex64], rows: usize, cols: usize, tol: f64, label: &str) {
    for left in 0..cols {
        for right in 0..cols {
            let inner: Complex64 = (0..rows)
                .map(|row| matrix[row + left * rows].conj() * matrix[row + right * rows])
                .sum();
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!(
                (inner - Complex64::new(expected, 0.0)).norm() < tol,
                "{label}: column pair ({left}, {right}) inner product {inner} is not {expected}"
            );
        }
    }
}

/// Every acceptance identity of the full decomposition, in one place.
fn assert_full_svd(
    dtype: DType,
    m: usize,
    n: usize,
    source: &[Complex64],
    u: &Tensor,
    s: &Tensor,
    vt: &Tensor,
) {
    let k = m.min(n);
    assert_eq!(u.shape(), &[m, m], "U must be square m x m");
    assert_eq!(s.shape(), &[k], "S must hold min(m, n) singular values");
    assert_eq!(vt.shape(), &[n, n], "Vt must be square n x n");
    assert_eq!(u.dtype(), dtype);
    assert_eq!(vt.dtype(), dtype);

    let tol = tolerance(dtype);
    let u = complex_values(u);
    let vt = complex_values(vt);
    let values = real_values(s);

    // Both unitarity directions: a square unitary satisfies U^H U = U U^H = I.
    assert_isometric(&u, m, m, tol, "U^H U");
    assert_isometric(&vt, n, n, tol, "Vt^H Vt");
    let u_adjoint: Vec<Complex64> = (0..m * m)
        .map(|index| u[index / m + (index % m) * m].conj())
        .collect();
    assert_isometric(&u_adjoint, m, m, tol, "U U^H");
    let v: Vec<Complex64> = (0..n * n)
        .map(|index| vt[index / n + (index % n) * n].conj())
        .collect();
    assert_isometric(&v, n, n, tol, "Vt Vt^H");

    assert!(
        values.windows(2).all(|pair| pair[0] >= pair[1] - tol),
        "singular values must be non-increasing: {values:?}"
    );

    // A = U[:, :k] diag(S) Vt[:k, :]; the trailing U columns and Vt rows span
    // the left and right nullspaces and contribute nothing to the product.
    for col in 0..n {
        for row in 0..m {
            let reconstructed: Complex64 = (0..k)
                .map(|index| u[row + index * m] * values[index] * vt[index + col * n])
                .sum();
            assert!(
                (reconstructed - source[row + col * m]).norm() < tol,
                "reconstruction mismatch at ({row}, {col}): {reconstructed} != {}",
                source[row + col * m]
            );
        }
    }
}

const DTYPES: [DType; 4] = [DType::F32, DType::F64, DType::C32, DType::C64];
/// Tall, wide, square, and both degenerate one-dimensional cases.
const SHAPES: [(usize, usize); 5] = [(4, 2), (2, 4), (3, 3), (1, 3), (3, 1)];

#[test]
fn owned_and_borrowed_full_svd_agree_for_every_dtype_and_shape() {
    let mut host = faer_backend();
    for dtype in DTYPES {
        for (m, n) in SHAPES {
            let real = sample_real(m, n);
            let imag = sample_imag(m, n);
            let input = tensor_of(dtype, &[m, n], &real, &imag);
            let source = complex_values(&input);

            host.with_backend_session(|session| {
                let (u, s, vt) = input.svd_full(session).unwrap();
                assert_full_svd(dtype, m, n, &source, &u, &s, &vt);

                let (ru, rs, rvt) = TensorRead::from_tensor(&input)
                    .svd_full_read(session)
                    .unwrap();
                assert_full_svd(dtype, m, n, &source, &ru, &rs, &rvt);

                // The borrowed and owned entry points must produce the same
                // spectrum, and it must match the thin decomposition's.
                assert_eq!(real_values(&s), real_values(&rs));
                let thin = input.svdvals(session).unwrap();
                let tol = tolerance(dtype);
                for (full, thin) in real_values(&s).iter().zip(real_values(&thin)) {
                    assert!(
                        (full - thin).abs() < tol,
                        "full and thin spectra disagree: {full} vs {thin}"
                    );
                }
            });
        }
    }
}

#[test]
fn full_svd_read_consumes_strided_offset_and_reversed_views_unchanged() {
    // Three borrowed layouts with one eligibility story each: the transposed
    // view reaches faer directly, the offset slice keeps positive strides and
    // also reaches faer, and the reversed view has negative strides so the
    // provider must pack it first. All three must agree with the owned path
    // and leave the source bytes untouched.
    let mut host = faer_backend();
    let base = TypedTensor::<f64>::from_vec_col_major(vec![3, 4], sample_real(3, 4)).unwrap();
    let original = base.host_data().unwrap().to_vec();

    let transposed = base.as_view().transpose_view([1, 0]).unwrap();
    let offset = base
        .as_view()
        .try_slice(&[
            StridedSliceSpec::new(1, Some(3), 1),
            StridedSliceSpec::new(1, Some(4), 1),
        ])
        .unwrap();
    let reversed = base
        .as_view()
        .try_slice(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])
        .unwrap();

    for (label, view, m, n) in [
        ("transposed", transposed, 4, 3),
        ("offset", offset, 2, 3),
        ("reversed", reversed, 3, 4),
    ] {
        let expected = {
            // Materialize the same elements into a compact tensor so the owned
            // path sees exactly the values the view exposes.
            let mut data = vec![0.0_f64; m * n];
            for col in 0..n {
                for row in 0..m {
                    data[row + col * m] = *view.get(&[row, col]).unwrap();
                }
            }
            Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![m, n], data.clone()).unwrap(),
            )
        };
        let source = complex_values(&expected);

        host.with_backend_session(|session| {
            let (u, s, vt) = TensorRead::from_view(TensorView::F64(view.clone()))
                .svd_full_read(session)
                .unwrap();
            assert_full_svd(DType::F64, m, n, &source, &u, &s, &vt);

            let owned = expected.svd_full(session).unwrap();
            for (borrowed, owned) in real_values(&s).iter().zip(real_values(&owned.1)) {
                assert!(
                    (borrowed - owned).abs() < 1.0e-10,
                    "{label}: borrowed and owned spectra disagree"
                );
            }
        });
    }

    assert_eq!(
        base.host_data().unwrap(),
        original.as_slice(),
        "full_svd_read must not modify its borrowed source"
    );
}

#[test]
fn full_svd_keeps_square_factors_when_a_core_dimension_is_empty() {
    // The full variant keeps `m x m` and `n x n` shapes even when the other
    // core dimension is zero, so the non-degenerate factor is the identity
    // rather than an empty tensor.
    let mut host = faer_backend();
    for (m, n) in [(0_usize, 3_usize), (3, 0), (0, 0)] {
        let input = tensor_of(DType::F64, &[m, n], &[], &[]);
        host.with_backend_session(|session| {
            for (label, (u, s, vt)) in [
                ("owned", input.svd_full(session).unwrap()),
                (
                    "borrowed",
                    TensorRead::from_tensor(&input)
                        .svd_full_read(session)
                        .unwrap(),
                ),
            ] {
                assert_eq!(u.shape(), &[m, m], "{label}: U shape");
                assert_eq!(s.shape(), &[m.min(n)], "{label}: S shape");
                assert_eq!(vt.shape(), &[n, n], "{label}: Vt shape");
                let u = complex_values(&u);
                let vt = complex_values(&vt);
                assert_isometric(&u, m, m, 1.0e-12, "empty U");
                assert_isometric(&vt, n, n, 1.0e-12, "empty Vt");
            }
        });
    }
}

#[test]
fn full_svd_handles_batched_inputs_per_matrix() {
    // Shape `[m, n, batch]`: each trailing slab is decomposed independently and
    // the outputs carry the same batch suffix.
    let mut host = faer_backend();
    let (m, n, batch) = (3_usize, 2_usize, 2_usize);
    let real = sample_real(m, n * batch);
    let input = tensor_of(DType::F64, &[m, n, batch], &real, &[]);

    host.with_backend_session(|session| {
        let (u, s, vt) = input.svd_full(session).unwrap();
        assert_eq!(u.shape(), &[m, m, batch]);
        assert_eq!(s.shape(), &[m.min(n), batch]);
        assert_eq!(vt.shape(), &[n, n, batch]);

        let (ru, rs, rvt) = TensorRead::from_tensor(&input)
            .svd_full_read(session)
            .unwrap();
        assert_eq!(ru.shape(), u.shape());
        assert_eq!(rs.shape(), s.shape());
        assert_eq!(rvt.shape(), vt.shape());

        let source = complex_values(&input);
        let u = complex_values(&u);
        let s = real_values(&s);
        let vt = complex_values(&vt);
        for slab in 0..batch {
            let slab_source = source[slab * m * n..(slab + 1) * m * n].to_vec();
            let slab_u = &u[slab * m * m..(slab + 1) * m * m];
            let slab_vt = &vt[slab * n * n..(slab + 1) * n * n];
            let slab_s = &s[slab * m.min(n)..(slab + 1) * m.min(n)];
            assert_isometric(slab_u, m, m, 1.0e-10, "batched U");
            assert_isometric(slab_vt, n, n, 1.0e-10, "batched Vt");
            for col in 0..n {
                for row in 0..m {
                    let reconstructed: Complex64 = (0..m.min(n))
                        .map(|index| {
                            slab_u[row + index * m] * slab_s[index] * slab_vt[index + col * n]
                        })
                        .sum();
                    assert!((reconstructed - slab_source[row + col * m]).norm() < 1.0e-10);
                }
            }
        }
    });
}

#[test]
fn typed_full_svd_returns_real_singular_values() {
    let mut host = faer_backend();
    let input = TypedTensor::<Complex64>::from_vec_col_major(
        vec![3, 2],
        sample_real(3, 2)
            .iter()
            .zip(sample_imag(3, 2))
            .map(|(&re, im)| Complex64::new(re, im))
            .collect(),
    )
    .unwrap();

    host.with_backend_session(|session| {
        let (u, s, vt) = input.svd_full(session).unwrap();
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);
        // `TypedSvd<Complex64>` types the singular values as f64.
        let values: &[f64] = s.as_slice().unwrap();
        assert!(values[0] >= values[1] && values[1] >= 0.0);
    });
}

#[test]
fn full_svd_read_rejects_unsupported_dtypes_before_provider_entry() {
    let mut host = faer_backend();
    let input = TypedTensor::<i64>::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap();
    host.with_backend_session(|session| {
        let error = TensorRead::from_view(TensorView::I64(input.as_view()))
            .svd_full_read(session)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    });
}

#[test]
fn faer_view_path_does_not_pool_an_input_copy() {
    // Same shape, same outputs, two eligibility stories. A compact owned read
    // is handed to faer as a `MatRef` directly; a reversed view has negative
    // strides, so the provider must pack it into a pooled compact tensor first
    // and then returns that buffer to the pool. The packed input copy is the
    // only structural difference between the two calls, so it has to show up
    // as extra retained pool capacity.
    let (m, n) = (3_usize, 5_usize);
    let base = TypedTensor::<f64>::from_vec_col_major(vec![m, n], sample_real(m, n)).unwrap();
    let owned = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![m, n], sample_real(m, n)).unwrap(),
    );

    let mut view_host = faer_backend();
    view_host.with_backend_session(|session| {
        TensorRead::from_tensor(&owned)
            .svd_full_read(session)
            .unwrap();
    });
    let view_stats = view_host.buffer_pool_stats().unwrap();

    let mut packed_host = faer_backend();
    packed_host.with_backend_session(|session| {
        let reversed = base
            .as_view()
            .try_slice(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])
            .unwrap();
        TensorRead::from_view(TensorView::F64(reversed))
            .svd_full_read(session)
            .unwrap();
    });
    let packed_stats = packed_host.buffer_pool_stats().unwrap();

    let input_copy_bytes = m * n * std::mem::size_of::<f64>();
    assert!(
        packed_stats.capacity_bytes >= view_stats.capacity_bytes + input_copy_bytes,
        "the packing path must retain at least the {input_copy_bytes}-byte input copy \
         that the faer view path never takes: view {} vs packed {}",
        view_stats.capacity_bytes,
        packed_stats.capacity_bytes
    );
}

#[test]
fn owned_full_svd_reports_unsupported_dtypes_from_the_provider_boundary() {
    // The owned entry point has no dtype pre-check, so an integer tensor must
    // be rejected by the provider dispatch itself rather than reaching faer.
    let mut host = faer_backend();
    let input = Tensor::from_typed::<i64>(
        TypedTensor::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap(),
    );
    host.with_backend_session(|session| {
        let error = input.svd_full(session).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    });
}
