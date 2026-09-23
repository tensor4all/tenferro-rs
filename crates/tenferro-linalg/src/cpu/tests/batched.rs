//! Batch x size sweeps for the tight batched CPU linalg loops (#1878).
//!
//! Each case runs on both CPU providers and all four linalg dtypes, and checks
//! every matrix of the batch against a residual or reconstruction oracle, so a
//! stride or workspace reuse error in one batch slot cannot hide behind the
//! first matrix.

use super::*;
use tenferro_cpu::CpuBackendKind;

const SIZES: [usize; 5] = [1, 2, 3, 8, 33];
const BATCHES: [usize; 2] = [1, 3];
const FLAGS: [(bool, bool); 4] = [(false, false), (true, false), (true, true), (false, true)];

#[derive(Clone, Copy, Debug)]
enum Dt {
    F32,
    F64,
    C32,
    C64,
}

const DTYPES: [Dt; 4] = [Dt::F32, Dt::F64, Dt::C32, Dt::C64];

impl Dt {
    fn complex(self) -> bool {
        matches!(self, Dt::C32 | Dt::C64)
    }

    fn tol(self) -> f64 {
        match self {
            Dt::F32 | Dt::C32 => 2.0e-4,
            Dt::F64 | Dt::C64 => 1.0e-11,
        }
    }

    fn tensor(self, shape: &[usize], data: &[Complex64]) -> Tensor {
        let shape = shape.to_vec();
        match self {
            Dt::F32 => {
                Tensor::from_vec_col_major(shape, data.iter().map(|z| z.re as f32).collect())
            }
            Dt::F64 => Tensor::from_vec_col_major(shape, data.iter().map(|z| z.re).collect()),
            Dt::C32 => Tensor::from_vec_col_major(
                shape,
                data.iter()
                    .map(|z| Complex32::new(z.re as f32, z.im as f32))
                    .collect(),
            ),
            Dt::C64 => Tensor::from_vec_col_major(shape, data.to_vec()),
        }
        .unwrap()
    }
}

fn backends() -> [(CpuBackendKind, CpuBackend); 2] {
    [CpuBackendKind::Faer, CpuBackendKind::Blas]
        .map(|kind| (kind, CpuBackend::with_threads_and_kind(1, kind).unwrap()))
}

/// Host data of any linalg or pivot dtype, widened to `Complex64`.
fn data(t: &Tensor) -> Vec<Complex64> {
    fn widen<T: tenferro_tensor::TensorScalar>(
        t: &Tensor,
        f: impl Fn(T) -> Complex64,
    ) -> Vec<Complex64> {
        t.as_typed::<T>()
            .unwrap()
            .host_data()
            .unwrap()
            .iter()
            .map(|&v| f(v))
            .collect()
    }
    match t.dtype() {
        DType::F32 => widen(t, |v: f32| Complex64::new(v as f64, 0.0)),
        DType::F64 => widen(t, |v: f64| Complex64::new(v, 0.0)),
        DType::C32 => widen(t, |v: Complex32| Complex64::new(v.re as f64, v.im as f64)),
        DType::C64 => widen(t, |v: Complex64| v),
        DType::I32 => widen(t, |v: i32| Complex64::new(v as f64, 0.0)),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// Deterministic values in `[-1, 1]` (imaginary part zero for real dtypes).
fn values(len: usize, seed: u64, complex: bool) -> Vec<Complex64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..len)
        .map(|_| {
            let re = next();
            let im = if complex { next() } else { 0.0 };
            Complex64::new(re, im)
        })
        .collect()
}

/// Well conditioned batched matrices whose dominant entries sit on the cyclic
/// subdiagonal, so partial pivoting must permute every row for `n >= 2`.
fn pivoting_matrices(n: usize, batch: usize, seed: u64, complex: bool) -> Vec<Complex64> {
    let mut a = values(n * n * batch, seed, complex);
    let dominant = Complex64::new(n as f64 + 2.0, 0.0);
    for k in 0..batch {
        for col in 0..n {
            let row = (col + 1) % n;
            a[k * n * n + row + col * n] += dominant;
        }
    }
    a
}

fn op_entry(a: &[Complex64], n: usize, row: usize, col: usize, t: bool, c: bool) -> Complex64 {
    let v = if t {
        a[col + row * n]
    } else {
        a[row + col * n]
    };
    if c {
        v.conj()
    } else {
        v
    }
}

/// Max relative residual of `op(A_k) X_k = B_k` over the batch.
fn solve_residual(
    a: &[Complex64],
    x: &[Complex64],
    b: &[Complex64],
    n: usize,
    nrhs: usize,
    batch: usize,
    (t, c): (bool, bool),
) -> f64 {
    let mut worst = 0.0_f64;
    for k in 0..batch {
        let ak = &a[k * n * n..(k + 1) * n * n];
        let xk = &x[k * n * nrhs..(k + 1) * n * nrhs];
        let bk = &b[k * n * nrhs..(k + 1) * n * nrhs];
        let scale = 1.0 + bk.iter().map(|v| v.norm()).fold(0.0, f64::max);
        for j in 0..nrhs {
            for i in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for p in 0..n {
                    acc += op_entry(ak, n, i, p, t, c) * xk[p + j * n];
                }
                worst = worst.max((acc - bk[i + j * n]).norm() / scale);
            }
        }
    }
    worst
}

fn max_diff(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs)
        .map(|(l, r)| (l - r).norm())
        .fold(0.0, f64::max)
}

/// RHS shapes for one case: the vector form `[n, batch]` and a matrix form.
fn rhs_cases(n: usize, batch: usize) -> [(Vec<usize>, usize); 2] {
    [(vec![n, batch], 1), (vec![n, 2, batch], 2)]
}

#[test]
fn batched_lu_solves_match_residual_oracle_across_sizes_dtypes_and_flags() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            for n in SIZES {
                for batch in BATCHES {
                    let a_data = pivoting_matrices(n, batch, (n * 31 + batch) as u64, dt.complex());
                    let a = dt.tensor(&[n, n, batch], &a_data);
                    let factors = with_cpu_linalg(&mut backend, |s| s.lu_factor(&a)).unwrap();
                    if n >= 2 {
                        let pivots = data(&factors[1]);
                        assert!(
                            pivots
                                .iter()
                                .enumerate()
                                .any(|(i, p)| p.re as usize != i % n + 1),
                            "{kind:?} {dt:?} n={n}: the cyclic dominant entries must force pivoting"
                        );
                    }
                    for (b_shape, nrhs) in rhs_cases(n, batch) {
                        let b_data = values(n * nrhs * batch, (7 * n + nrhs) as u64, dt.complex());
                        let b = dt.tensor(&b_shape, &b_data);
                        let a_used = dt.tensor(&[n, n, batch], &data(&a));
                        let b_used = data(&b);
                        for flags in FLAGS {
                            let x = with_cpu_linalg(&mut backend, |s| {
                                s.lu_solve_prepared(
                                    &a,
                                    &factors[0],
                                    &factors[1],
                                    &b,
                                    flags.0,
                                    flags.1,
                                )
                            })
                            .unwrap();
                            assert_eq!(x.shape(), b.shape());
                            let residual = solve_residual(
                                &data(&a_used),
                                &data(&x),
                                &b_used,
                                n,
                                nrhs,
                                batch,
                                flags,
                            );
                            assert!(
                                residual < dt.tol() * n as f64,
                                "{kind:?} {dt:?} n={n} batch={batch} nrhs={nrhs} flags={flags:?}: residual {residual}"
                            );
                        }

                        let fused =
                            with_cpu_linalg(&mut backend, |s| s.lu_factor_solve(&a, &b)).unwrap();
                        assert_eq!(fused.len(), 3);
                        let split = with_cpu_linalg(&mut backend, |s| {
                            s.lu_solve_prepared(&a, &factors[0], &factors[1], &b, false, false)
                        })
                        .unwrap();
                        let scale = dt.tol() * n as f64;
                        assert!(
                            max_diff(&data(&fused[0]), &data(&split)) < scale * 10.0,
                            "{kind:?} {dt:?} n={n}: fused and split solves disagree"
                        );
                        assert!(max_diff(&data(&fused[1]), &data(&factors[0])) < scale);
                        assert_eq!(data(&fused[2]), data(&factors[1]));
                    }
                }
            }
        }
    }
}

#[test]
fn batched_lu_solves_agree_across_providers() {
    let [(_, mut faer), (_, mut blas)] = backends();
    for dt in DTYPES {
        for n in SIZES {
            let batch = 3;
            let a = dt.tensor(
                &[n, n, batch],
                &pivoting_matrices(n, batch, 5, dt.complex()),
            );
            let b = dt.tensor(&[n, 2, batch], &values(n * 2 * batch, 9, dt.complex()));
            let lhs = with_cpu_linalg(&mut faer, |s| s.lu_factor_solve(&a, &b)).unwrap();
            let rhs = with_cpu_linalg(&mut blas, |s| s.lu_factor_solve(&a, &b)).unwrap();
            assert!(
                max_diff(&data(&lhs[0]), &data(&rhs[0])) < dt.tol() * 10.0 * n as f64,
                "{dt:?} n={n}: faer and LAPACK solutions disagree"
            );
            // Both providers follow the LAPACK getrf pivot convention.
            assert_eq!(data(&lhs[2]), data(&rhs[2]), "{dt:?} n={n}: pivots differ");
        }
    }
}

#[test]
fn batched_lu_solves_report_a_singular_batch_member() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            let n = 3;
            let batch = 3;
            let mut a_data = pivoting_matrices(n, batch, 11, dt.complex());
            // Zero the second matrix's first column.
            for row in 0..n {
                a_data[n * n + row] = Complex64::new(0.0, 0.0);
            }
            let a = dt.tensor(&[n, n, batch], &a_data);
            let b = dt.tensor(&[n, batch], &values(n * batch, 3, dt.complex()));
            let fused = with_cpu_linalg(&mut backend, |s| s.lu_factor_solve(&a, &b));
            assert!(
                fused.is_err(),
                "{kind:?} {dt:?}: singular fused solve must fail"
            );
            let solve = with_cpu_linalg(&mut backend, |s| s.solve(&a, &b));
            assert!(solve.is_err(), "{kind:?} {dt:?}: singular solve must fail");

            let factors = with_cpu_linalg(&mut backend, |s| s.lu_factor(&a)).unwrap();
            for flags in FLAGS {
                let prepared = with_cpu_linalg(&mut backend, |s| {
                    s.lu_solve_prepared(&a, &factors[0], &factors[1], &b, flags.0, flags.1)
                });
                assert!(
                    prepared.is_err(),
                    "{kind:?} {dt:?} {flags:?}: singular prepared solve must fail"
                );
            }

            // A zero-column RHS still factors, and there is nothing to solve.
            let empty_rhs = dt.tensor(&[n, 0, batch], &[]);
            let outputs =
                with_cpu_linalg(&mut backend, |s| s.lu_factor_solve(&a, &empty_rhs)).unwrap();
            assert_eq!(outputs[0].shape(), &[n, 0, batch]);
            assert_eq!(outputs[1].shape(), &[n, n, batch]);
        }
    }
}

#[test]
fn batched_lu_solves_accept_empty_batches_and_zero_sizes() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            for (a_shape, b_shape) in [
                (vec![3, 3, 0], vec![3, 2, 0]),
                (vec![3, 3, 0], vec![3, 0]),
                (vec![0, 0, 4], vec![0, 1, 4]),
            ] {
                let a = dt.tensor(&a_shape, &[]);
                let b = dt.tensor(&b_shape, &[]);
                let fused = with_cpu_linalg(&mut backend, |s| s.lu_factor_solve(&a, &b))
                    .unwrap_or_else(|err| panic!("{kind:?} {dt:?} {a_shape:?}: {err}"));
                assert_eq!(fused[0].shape(), b.shape());
                assert_eq!(fused[1].shape(), a.shape());
                let factors = with_cpu_linalg(&mut backend, |s| s.lu_factor(&a)).unwrap();
                for flags in FLAGS {
                    let x = with_cpu_linalg(&mut backend, |s| {
                        s.lu_solve_prepared(&a, &factors[0], &factors[1], &b, flags.0, flags.1)
                    })
                    .unwrap();
                    assert_eq!(x.shape(), b.shape());
                }
            }
        }
    }
}

/// Upper or lower triangular batch with a dominant diagonal.
fn triangular_matrices(n: usize, batch: usize, lower: bool, complex: bool) -> Vec<Complex64> {
    let mut a = values(n * n * batch, 17, complex);
    for k in 0..batch {
        for col in 0..n {
            for row in 0..n {
                let entry = &mut a[k * n * n + row + col * n];
                if (lower && row < col) || (!lower && row > col) {
                    *entry = Complex64::new(0.0, 0.0);
                } else if row == col {
                    *entry += Complex64::new(2.0, 0.0);
                }
            }
        }
    }
    a
}

#[test]
fn batched_triangular_solve_matches_residual_oracle() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            for n in SIZES {
                let batch = 3;
                for lower in [false, true] {
                    let a_data = triangular_matrices(n, batch, lower, dt.complex());
                    let a = dt.tensor(&[n, n, batch], &a_data);
                    let b = dt.tensor(&[n, 2, batch], &values(n * 2 * batch, 23, dt.complex()));
                    for transpose in [false, true] {
                        let x = with_cpu_linalg(&mut backend, |s| {
                            s.triangular_solve(&a, &b, true, lower, transpose, false)
                        })
                        .unwrap();
                        let residual = solve_residual(
                            &data(&a),
                            &data(&x),
                            &data(&b),
                            n,
                            2,
                            batch,
                            (transpose, false),
                        );
                        assert!(
                            residual < dt.tol() * n as f64,
                            "{kind:?} {dt:?} n={n} lower={lower} transpose={transpose}: residual {residual}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn batched_full_piv_lu_solve_matches_residual_oracle() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            for n in SIZES {
                let batch = 3;
                let a_data = pivoting_matrices(n, batch, 29, dt.complex());
                let a = dt.tensor(&[n, n, batch], &a_data);
                let b = dt.tensor(&[n, 2, batch], &values(n * 2 * batch, 31, dt.complex()));
                for transpose in [false, true] {
                    let x =
                        with_cpu_linalg(&mut backend, |s| s.full_piv_lu_solve(&a, &b, transpose))
                            .unwrap();
                    let residual = solve_residual(
                        &a_data_for(dt, &a_data),
                        &data(&x),
                        &data(&b),
                        n,
                        2,
                        batch,
                        (transpose, false),
                    );
                    assert!(
                        residual < dt.tol() * n as f64,
                        "{kind:?} {dt:?} n={n} transpose={transpose}: residual {residual}"
                    );
                }
            }
        }
    }
}

/// The oracle must use the matrix at the tensor's own precision.
fn a_data_for(dt: Dt, a: &[Complex64]) -> Vec<Complex64> {
    data(&dt.tensor(&[a.len()], a))
}

fn hermitian_matrices(n: usize, batch: usize, complex: bool) -> Vec<Complex64> {
    let raw = values(n * n * batch, 37, complex);
    let mut a = vec![Complex64::new(0.0, 0.0); raw.len()];
    for k in 0..batch {
        let off = k * n * n;
        for col in 0..n {
            for row in 0..n {
                let v = raw[off + row + col * n] + raw[off + col + row * n].conj();
                a[off + row + col * n] = v * 0.5;
            }
        }
    }
    a
}

#[test]
fn batched_eigh_reconstructs_every_matrix() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            for n in SIZES {
                let batch = 3;
                let a_data = a_data_for(dt, &hermitian_matrices(n, batch, dt.complex()));
                let a = dt.tensor(&[n, n, batch], &a_data);
                let outputs = with_cpu_linalg(&mut backend, |s| s.eigh(&a)).unwrap();
                let w = data(&outputs[0]);
                let v = data(&outputs[1]);
                assert_eq!(w.len(), n * batch);
                let only = data(&with_cpu_linalg(&mut backend, |s| s.eigh_values(&a)).unwrap());
                let tol = dt.tol() * 10.0 * n as f64;
                assert!(
                    max_diff(&w, &only) < tol,
                    "{kind:?} {dt:?} n={n}: eigh_values"
                );
                for k in 0..batch {
                    let ak = &a_data[k * n * n..(k + 1) * n * n];
                    let vk = &v[k * n * n..(k + 1) * n * n];
                    for j in 0..n {
                        let lambda = w[k * n + j];
                        for i in 0..n {
                            let mut av = Complex64::new(0.0, 0.0);
                            for p in 0..n {
                                av += ak[i + p * n] * vk[p + j * n];
                            }
                            let err = (av - lambda * vk[i + j * n]).norm();
                            assert!(err < tol, "{kind:?} {dt:?} n={n} k={k}: A v != w v ({err})");
                        }
                    }
                }
            }
        }
    }
}

fn check_svd_reconstruction(
    label: &str,
    a: &[Complex64],
    (u, s, vt): (&[Complex64], &[Complex64], &[Complex64]),
    (m, n, batch): (usize, usize, usize),
    (u_cols, vt_rows): (usize, usize),
    tol: f64,
) {
    let k = m.min(n);
    for b in 0..batch {
        for col in 0..n {
            for row in 0..m {
                let mut acc = Complex64::new(0.0, 0.0);
                for p in 0..k {
                    acc += u[b * m * u_cols + row + p * m]
                        * s[b * k + p]
                        * vt[b * vt_rows * n + p + col * vt_rows];
                }
                let err = (acc - a[b * m * n + row + col * m]).norm();
                assert!(err < tol, "{label} batch {b}: U S Vt != A ({err})");
            }
        }
    }
}

#[test]
fn batched_svd_variants_reconstruct_every_matrix() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            for (m, n) in [(1, 1), (2, 2), (3, 3), (8, 8), (33, 33), (3, 5), (5, 3)] {
                let batch = 3;
                let a_data = a_data_for(dt, &values(m * n * batch, 41, dt.complex()));
                let a = dt.tensor(&[m, n, batch], &a_data);
                let k = m.min(n);
                let tol = dt.tol() * 10.0 * m.max(n) as f64;
                let label = format!("{kind:?} {dt:?} {m}x{n}");

                let thin = with_cpu_linalg(&mut backend, |s| s.svd(&a)).unwrap();
                assert_eq!(thin[0].shape(), &[m, k, batch]);
                assert_eq!(thin[2].shape(), &[k, n, batch]);
                let s_thin = data(&thin[1]);
                check_svd_reconstruction(
                    &format!("{label} thin"),
                    &a_data,
                    (&data(&thin[0]), &s_thin, &data(&thin[2])),
                    (m, n, batch),
                    (k, k),
                    tol,
                );

                let full = with_cpu_linalg(&mut backend, |s| s.svd_full(&a)).unwrap();
                assert_eq!(full[0].shape(), &[m, m, batch]);
                assert_eq!(full[2].shape(), &[n, n, batch]);
                check_svd_reconstruction(
                    &format!("{label} full"),
                    &a_data,
                    (&data(&full[0]), &data(&full[1]), &data(&full[2])),
                    (m, n, batch),
                    (m, n),
                    tol,
                );

                let only = data(&with_cpu_linalg(&mut backend, |s| s.svd_values(&a)).unwrap());
                assert!(max_diff(&only, &s_thin) < tol, "{label}: svd_values");
            }
        }
    }
}

#[test]
fn batched_eigh_and_svd_accept_empty_batches() {
    for (kind, mut backend) in backends() {
        for dt in DTYPES {
            let a = dt.tensor(&[3, 3, 0], &[]);
            let eigh = with_cpu_linalg(&mut backend, |s| s.eigh(&a))
                .unwrap_or_else(|err| panic!("{kind:?} {dt:?} eigh: {err}"));
            assert_eq!(eigh[1].shape(), &[3, 3, 0]);
            let w = with_cpu_linalg(&mut backend, |s| s.eigh_values(&a)).unwrap();
            assert_eq!(w.shape(), &[3, 0]);
            let svd = with_cpu_linalg(&mut backend, |s| s.svd(&a)).unwrap();
            assert_eq!(svd[0].shape(), &[3, 3, 0]);
            let sv = with_cpu_linalg(&mut backend, |s| s.svd_values(&a)).unwrap();
            assert_eq!(sv.shape(), &[3, 0]);
        }
    }
}
