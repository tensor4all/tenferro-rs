//! Borrowed-input parity for the general eigensolver and the solve dispatch.
//!
//! `eig_values` gained a `_read` hook, and `eig`/`eig_values` gained faer
//! strided-view fast paths, so a borrowed view now reaches the provider that
//! can read it instead of being packed first. The assertions below are the
//! borrowed contract itself: the same result as the owned call, an untouched
//! source, and a typed refusal for dtypes the provider never sees.

#![cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]

use num_complex::{Complex32, Complex64};
use tenferro_cpu::{CpuBackend, CpuBackendKind};
use tenferro_linalg::{TensorLinalgExt, TensorReadLinalgExt};
use tenferro_tensor::{
    BackendSessionHost, DType, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorView,
    TypedTensor,
};

/// Every CPU linalg provider compiled into this build.
fn providers() -> Vec<(&'static str, CpuBackend)> {
    const KINDS: &[(&str, CpuBackendKind)] = &[
        #[cfg(feature = "cpu-faer")]
        ("faer", CpuBackendKind::Faer),
        #[cfg(feature = "cpu-blas")]
        ("blas", CpuBackendKind::Blas),
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

/// A non-symmetric matrix with distinct real eigenvalues, so the spectrum is
/// well separated and ordering comparisons stay meaningful.
fn sample_real(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let row = (index % n) as f64;
            let col = (index / n) as f64;
            if row == col {
                3.0 + row * 2.0
            } else {
                0.25 * (row - col)
            }
        })
        .collect()
}

fn sample_imag(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let row = (index % n) as f64;
            let col = (index / n) as f64;
            0.125 * (row + 1.0) - 0.0625 * col
        })
        .collect()
}

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

/// Eigenvalues are always complex; read them at f64 precision for comparison.
fn complex_values(tensor: &Tensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DType::C32 => tensor
            .as_slice::<Complex32>()
            .unwrap()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        DType::C64 => tensor.as_slice::<Complex64>().unwrap().to_vec(),
        other => panic!("eigenvalues must be complex, got {other:?}"),
    }
}

fn tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F32 | DType::C32 => 2.0e-4,
        _ => 1.0e-9,
    }
}

/// Compare two spectra as multisets: the solver may order them differently
/// between the view and the packed path, and neither order is a contract.
fn assert_same_spectrum(actual: &[Complex64], expected: &[Complex64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: spectrum length");
    let mut remaining: Vec<Complex64> = expected.to_vec();
    for value in actual {
        let Some(position) = remaining
            .iter()
            .position(|candidate| (candidate - value).norm() < tol)
        else {
            panic!("{label}: eigenvalue {value} has no match in {expected:?}");
        };
        remaining.swap_remove(position);
    }
}

const DTYPES: [DType; 4] = [DType::F32, DType::F64, DType::C32, DType::C64];

#[test]
fn borrowed_eig_and_eigvals_match_the_owned_call_for_every_dtype() {
    for (provider, mut host) in providers() {
        for dtype in DTYPES {
            let n = 4;
            let input = tensor_of(dtype, &[n, n], &sample_real(n), &sample_imag(n));
            let tol = tolerance(dtype);

            host.with_backend_session(|session| {
                let (owned_values, owned_vectors) = input.eig(session).unwrap();
                let (read_values, read_vectors) =
                    TensorRead::from_tensor(&input).eig_read(session).unwrap();
                assert_eq!(read_values.shape(), owned_values.shape());
                assert_eq!(read_vectors.shape(), owned_vectors.shape());
                assert_same_spectrum(
                    &complex_values(&read_values),
                    &complex_values(&owned_values),
                    tol,
                    &format!("{provider}/{dtype:?} eig_read"),
                );

                let owned_only = input.eigvals(session).unwrap();
                let read_only = TensorRead::from_tensor(&input)
                    .eigvals_read(session)
                    .unwrap();
                assert_eq!(read_only.shape(), owned_only.shape());
                assert_same_spectrum(
                    &complex_values(&read_only),
                    &complex_values(&owned_only),
                    tol,
                    &format!("{provider}/{dtype:?} eigvals_read"),
                );
            });
        }
    }
}

#[test]
fn borrowed_eig_consumes_strided_and_reversed_views_without_touching_the_source() {
    // The transposed view is faer-eligible and reaches the eigensolver as a
    // strided `MatRef`; the reversed view has negative strides, so the provider
    // packs it first. Both must agree with the owned decomposition of the same
    // logical elements, and neither may modify the source.
    for (provider, mut host) in providers() {
        let n = 3;
        let base = TypedTensor::<f64>::from_vec_col_major(vec![n, n], sample_real(n)).unwrap();
        let original = base.host_data().unwrap().to_vec();

        for (label, view) in [
            ("transposed", base.as_view().transpose_view([1, 0]).unwrap()),
            (
                "reversed",
                base.as_view()
                    .try_slice(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])
                    .unwrap(),
            ),
        ] {
            let mut expected_data = vec![0.0_f64; n * n];
            for col in 0..n {
                for row in 0..n {
                    expected_data[row + col * n] = *view.get(&[row, col]).unwrap();
                }
            }
            let expected = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![n, n], expected_data).unwrap(),
            );

            host.with_backend_session(|session| {
                let owned = expected.eigvals(session).unwrap();
                let borrowed = TensorRead::from_view(TensorView::F64(view.clone()))
                    .eigvals_read(session)
                    .unwrap();
                assert_same_spectrum(
                    &complex_values(&borrowed),
                    &complex_values(&owned),
                    1.0e-9,
                    &format!("{provider}/{label}"),
                );

                let (values, vectors) = TensorRead::from_view(TensorView::F64(view.clone()))
                    .eig_read(session)
                    .unwrap();
                assert_eq!(vectors.shape(), &[n, n]);
                assert_same_spectrum(
                    &complex_values(&values),
                    &complex_values(&owned),
                    1.0e-9,
                    &format!("{provider}/{label} eig_read"),
                );
            });
        }

        assert_eq!(
            base.host_data().unwrap(),
            original.as_slice(),
            "{provider}: borrowed eigensolver must not modify its source"
        );
    }
}

#[test]
fn borrowed_eig_falls_back_for_batched_views_and_keeps_batch_shapes() {
    // A rank-3 view is not a faer `MatRef`, so this exercises the packing
    // fallback while still owing the same batched output shapes.
    for (provider, mut host) in providers() {
        let (n, batch) = (2_usize, 2_usize);
        let data: Vec<f64> = (0..n * n * batch).map(|value| 1.0 + value as f64).collect();
        let input = Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![n, n, batch], data).unwrap(),
        );
        host.with_backend_session(|session| {
            let owned = input.eigvals(session).unwrap();
            let borrowed = TensorRead::from_tensor(&input)
                .eigvals_read(session)
                .unwrap();
            assert_eq!(owned.shape(), &[n, batch]);
            assert_eq!(borrowed.shape(), owned.shape());
            assert_same_spectrum(
                &complex_values(&borrowed),
                &complex_values(&owned),
                1.0e-9,
                provider,
            );
        });
    }
}

#[test]
fn borrowed_eig_keeps_empty_shapes() {
    for (provider, mut host) in providers() {
        let input = Tensor::from_typed::<f64>(
            TypedTensor::from_vec_col_major(vec![0, 0], Vec::new()).unwrap(),
        );
        host.with_backend_session(|session| {
            let values = TensorRead::from_tensor(&input)
                .eigvals_read(session)
                .unwrap();
            assert_eq!(values.shape(), &[0], "{provider}: empty eigenvalue shape");
            assert_eq!(values.dtype(), DType::C64);
            let (values, vectors) = TensorRead::from_tensor(&input).eig_read(session).unwrap();
            assert_eq!(values.shape(), &[0]);
            assert_eq!(vectors.shape(), &[0, 0]);
        });
    }
}

#[test]
fn borrowed_eigvals_rejects_unsupported_dtypes_before_provider_entry() {
    for (provider, mut host) in providers() {
        let input =
            TypedTensor::<i64>::from_vec_col_major(vec![2, 2], vec![1_i64, 2, 3, 4]).unwrap();
        host.with_backend_session(|session| {
            let error = TensorRead::from_view(TensorView::I64(input.as_view()))
                .eigvals_read(session)
                .unwrap_err();
            assert_eq!(
                error.kind(),
                ErrorKind::Unsupported,
                "{provider}: integer eigenvalues must be refused"
            );
        });
    }
}

#[cfg(feature = "cpu-faer")]
#[test]
fn faer_eig_view_path_does_not_pool_an_input_copy() {
    // Same shape, same outputs, two eligibility stories: a compact owned read
    // reaches faer as a `MatRef`, while a reversed view must be packed into a
    // pooled compact tensor that is then returned to the pool. That packed
    // input copy is the only structural difference between the two calls.
    let n = 6_usize;
    let base = TypedTensor::<f64>::from_vec_col_major(vec![n, n], sample_real(n)).unwrap();
    let owned = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![n, n], sample_real(n)).unwrap(),
    );

    let mut view_host =
        CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).expect("faer CPU backend");
    view_host.with_backend_session(|session| {
        TensorRead::from_tensor(&owned)
            .eigvals_read(session)
            .unwrap();
    });
    let view_stats = view_host.buffer_pool_stats().unwrap();

    let mut packed_host =
        CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).expect("faer CPU backend");
    packed_host.with_backend_session(|session| {
        let reversed = base
            .as_view()
            .try_slice(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])
            .unwrap();
        TensorRead::from_view(TensorView::F64(reversed))
            .eigvals_read(session)
            .unwrap();
    });
    let packed_stats = packed_host.buffer_pool_stats().unwrap();

    let input_copy_bytes = n * n * std::mem::size_of::<f64>();
    assert!(
        packed_stats.capacity_bytes >= view_stats.capacity_bytes + input_copy_bytes,
        "the packing path must retain at least the {input_copy_bytes}-byte input copy \
         that the faer view path never takes: view {} vs packed {}",
        view_stats.capacity_bytes,
        packed_stats.capacity_bytes
    );
}

// ---------------------------------------------------------------------------
// Rank-revealing QR and triangular solve. Both are destructive, so they copy
// into their own work storage either way; the borrowed path makes the view the
// source of that single copy instead of packing it into a compact tensor first.
// ---------------------------------------------------------------------------

fn transposed_pair(base: &TypedTensor<f64>, rows: usize, cols: usize) -> Tensor {
    let view = base.as_view().transpose_view([1, 0]).unwrap();
    let mut data = vec![0.0_f64; rows * cols];
    for col in 0..cols {
        for row in 0..rows {
            data[row + col * rows] = *view.get(&[row, col]).unwrap();
        }
    }
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![rows, cols], data).unwrap())
}

#[test]
fn borrowed_rank_revealing_qr_matches_the_owned_call() {
    use tenferro_linalg::RankRevealingQrOptions;

    // Columns are a, a, b: a prefix-only rank guard would miss the later
    // independent column, so CPQR must report rank two through both routes.
    let base = TypedTensor::<f64>::from_vec_col_major(
        vec![3, 4],
        vec![
            1.0_f64, 1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 0.0, 1.0, 1.0, 1.0, 0.0,
        ],
    )
    .unwrap();
    let original = base.host_data().unwrap().to_vec();
    let (rows, cols) = (4_usize, 3_usize);
    let expected = transposed_pair(&base, rows, cols);

    for (provider, mut host) in providers() {
        let options = RankRevealingQrOptions::default().rtol(1.0e-12);
        host.with_backend_session(|session| {
            let owned = expected.rank_revealing_qr(options, session).unwrap();
            let view = base.as_view().transpose_view([1, 0]).unwrap();
            let borrowed = TensorRead::from_view(TensorView::F64(view))
                .rank_revealing_qr_read(options, session)
                .unwrap();

            assert_eq!(borrowed.q.shape(), owned.q.shape(), "{provider}: Q shape");
            assert_eq!(borrowed.r.shape(), owned.r.shape(), "{provider}: R shape");
            assert_eq!(
                borrowed.rank.as_slice::<i64>().unwrap(),
                owned.rank.as_slice::<i64>().unwrap(),
                "{provider}: rank"
            );
            assert_eq!(
                borrowed.column_permutation.as_slice::<i64>().unwrap(),
                owned.column_permutation.as_slice::<i64>().unwrap(),
                "{provider}: permutation"
            );

            // The factorization is verified by its own contract, not by
            // comparing bytes with the owned run: Q is isometric and
            // `Q R = A P`.
            let q = borrowed.q.as_slice::<f64>().unwrap();
            let r = borrowed.r.as_slice::<f64>().unwrap();
            let permutation = borrowed.column_permutation.as_slice::<i64>().unwrap();
            let source = expected.as_slice::<f64>().unwrap();
            let k = rows.min(cols);
            for left in 0..k {
                for right in 0..k {
                    let inner: f64 = (0..rows)
                        .map(|row| q[row + left * rows] * q[row + right * rows])
                        .sum();
                    let want = if left == right { 1.0 } else { 0.0 };
                    assert!((inner - want).abs() < 1.0e-10, "{provider}: Q^T Q");
                }
            }
            for factor_col in 0..cols {
                let source_col = usize::try_from(permutation[factor_col]).unwrap();
                for row in 0..rows {
                    let reconstructed: f64 = (0..k)
                        .map(|inner| q[row + inner * rows] * r[inner + factor_col * k])
                        .sum();
                    assert!(
                        (reconstructed - source[row + source_col * rows]).abs() < 1.0e-10,
                        "{provider}: Q R = A P"
                    );
                }
            }
        });
    }

    assert_eq!(
        base.host_data().unwrap(),
        original.as_slice(),
        "rank_revealing_qr_read must not modify its borrowed source"
    );
}

#[test]
fn borrowed_rank_revealing_qr_keeps_the_zero_and_non_finite_guards() {
    use tenferro_linalg::RankRevealingQrOptions;

    for (provider, mut host) in providers() {
        let zeros = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![0.0; 6]).unwrap();
        host.with_backend_session(|session| {
            let view = zeros.as_view().transpose_view([1, 0]).unwrap();
            let result = TensorRead::from_view(TensorView::F64(view))
                .rank_revealing_qr_read(RankRevealingQrOptions::default(), session)
                .unwrap();
            assert_eq!(result.rank.as_slice::<i64>().unwrap(), &[0], "{provider}");
            assert_eq!(result.q.shape(), &[3, 2]);
            assert_eq!(result.r.shape(), &[2, 2]);
        });

        let non_finite =
            TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, f64::NAN, 0.0, 1.0])
                .unwrap();
        host.with_backend_session(|session| {
            let view = non_finite.as_view().transpose_view([1, 0]).unwrap();
            let error = TensorRead::from_view(TensorView::F64(view))
                .rank_revealing_qr_read(RankRevealingQrOptions::default(), session)
                .unwrap_err();
            assert_eq!(
                error.kind(),
                ErrorKind::NumericalFailure,
                "{provider}: non-finite input must be refused"
            );
        });
    }
}

#[test]
fn borrowed_triangular_solve_matches_the_owned_call_on_both_sides() {
    // A lower-triangular coefficient and a strided right-hand side: the direct
    // path hands `a` to faer as a `MatRef` and gathers `b` straight into the
    // destructible buffer, so the result must still match the owned solve.
    let a_base = TypedTensor::<f64>::from_vec_col_major(
        vec![3, 3],
        vec![2.0, 1.0, 0.5, 0.0, 3.0, -1.0, 0.0, 0.0, 4.0],
    )
    .unwrap();
    let b_base =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, -2.0, 0.5, 3.0, 1.5])
            .unwrap();
    let a_original = a_base.host_data().unwrap().to_vec();
    let b_original = b_base.host_data().unwrap().to_vec();

    let a_owned = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![3, 3], a_base.host_data().unwrap().to_vec()).unwrap(),
    );
    // `b` transposed is 3 x 2, the shape a left-side solve needs.
    let b_owned = transposed_pair(&b_base, 3, 2);

    for (provider, mut host) in providers() {
        for (lower, transpose_a, unit_diagonal) in [
            (true, false, false),
            (true, true, false),
            (false, false, false),
            (true, false, true),
        ] {
            host.with_backend_session(|session| {
                let owned = a_owned
                    .triangular_solve(&b_owned, true, lower, transpose_a, unit_diagonal, session)
                    .unwrap();
                let b_view = b_base.as_view().transpose_view([1, 0]).unwrap();
                let borrowed = TensorRead::from_tensor(&a_owned)
                    .triangular_solve_read(
                        TensorRead::from_view(TensorView::F64(b_view)),
                        true,
                        lower,
                        transpose_a,
                        unit_diagonal,
                        session,
                    )
                    .unwrap();
                assert_eq!(borrowed.shape(), owned.shape());
                for (borrowed, owned) in borrowed
                    .as_slice::<f64>()
                    .unwrap()
                    .iter()
                    .zip(owned.as_slice::<f64>().unwrap())
                {
                    assert!(
                        (borrowed - owned).abs() < 1.0e-10,
                        "{provider}: borrowed and owned triangular solve disagree"
                    );
                }
            });
        }

        // Right-side solve `X A = B` with a strided `A` view.
        host.with_backend_session(|session| {
            let rhs = Tensor::from_typed::<f64>(
                TypedTensor::from_vec_col_major(vec![2, 3], vec![1.0, 4.0, -2.0, 0.5, 3.0, 1.5])
                    .unwrap(),
            );
            let owned = a_owned
                .triangular_solve(&rhs, false, true, false, false, session)
                .unwrap();
            let a_view = a_base.as_view().transpose_view([1, 0]).unwrap();
            let borrowed = TensorRead::from_view(TensorView::F64(a_view))
                .triangular_solve_read(
                    TensorRead::from_tensor(&rhs),
                    false,
                    // The view is Aᵀ, so an upper-triangular read of it is the
                    // same system as the lower-triangular owned solve.
                    false,
                    true,
                    false,
                    session,
                )
                .unwrap();
            assert_eq!(borrowed.shape(), owned.shape(), "{provider}");
            for (borrowed, owned) in borrowed
                .as_slice::<f64>()
                .unwrap()
                .iter()
                .zip(owned.as_slice::<f64>().unwrap())
            {
                assert!(
                    (borrowed - owned).abs() < 1.0e-10,
                    "{provider}: right-side borrowed solve disagrees"
                );
            }
        });
    }

    assert_eq!(a_base.host_data().unwrap(), a_original.as_slice());
    assert_eq!(b_base.host_data().unwrap(), b_original.as_slice());
}

#[test]
fn borrowed_triangular_solve_rejects_a_vector_right_hand_side_like_the_owned_call() {
    // Provider selection must never change the accepted input shape: triangular
    // solve takes a matrix right-hand side everywhere, so the borrowed route has
    // to refuse a rank-1 RHS exactly as the owned route does.
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 4.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![2.0, 8.0]).unwrap();
    let b_owned = Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![2], vec![2.0, 8.0]).unwrap(),
    );
    for (provider, mut host) in providers() {
        host.with_backend_session(|session| {
            let owned = a
                .triangular_solve(&b_owned, true, true, false, false, session)
                .unwrap_err();
            let borrowed = TensorRead::from_tensor(&a)
                .triangular_solve_read(
                    TensorRead::from_view(TensorView::F64(b.as_view())),
                    true,
                    true,
                    false,
                    false,
                    session,
                )
                .unwrap_err();
            assert_eq!(
                owned.kind(),
                borrowed.kind(),
                "{provider}: owned and borrowed triangular solve must refuse alike"
            );
        });
    }
}
