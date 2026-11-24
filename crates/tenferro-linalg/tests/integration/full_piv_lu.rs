use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{DType, DotGeneralConfig, Tensor, TensorDot, TensorStructural, TypedTensor};

use super::support;
use tenferro_tensor::TensorRead;

fn f64_tensor(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().expect("expected f64 tensor")
}

fn f32_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    Tensor::from_typed::<f32>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().expect("expected c64 tensor")
}

fn c32_data(tensor: &Tensor) -> &[Complex32] {
    tensor.as_slice::<Complex32>().expect("expected c32 tensor")
}

fn matmul(backend: &mut CpuEngine, lhs: &Tensor, rhs: &Tensor) -> Tensor {
    backend
        .dot_general_read(
            TensorRead::from_tensor(lhs),
            TensorRead::from_tensor(rhs),
            &DotGeneralConfig {
                lhs_contracting_dims: [1].as_slice().into(),
                rhs_contracting_dims: [0].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        )
        .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1.0e-10,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn full_piv_lu_reconstructs_permuted_matrix() {
    let a = f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]);
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let outputs =
        support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&a)).unwrap();
    let [p, l, u, q, parity]: [Tensor; 5] = outputs.try_into().unwrap();
    let pa = matmul(&mut backend, &p, &a);
    let qt = backend
        .transpose_read(TensorRead::from_tensor(&q), &[1, 0])
        .unwrap();
    let paqt = matmul(&mut backend, &pa, &qt);
    let lu = matmul(&mut backend, &l, &u);

    assert_eq!(p.shape(), &[2, 2]);
    assert_eq!(l.shape(), &[2, 2]);
    assert_eq!(u.shape(), &[2, 2]);
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(parity.shape(), &[] as &[usize]);
    assert_close(f64_data(&paqt), f64_data(&lu));
}

#[test]
fn full_piv_lu_complex_parity_uses_real_counterpart_dtype() {
    let a = Tensor::from_typed::<tenferro_tensor::Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        )
        .unwrap(),
    );
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let outputs =
        support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&a)).unwrap();

    assert_eq!(outputs[0].dtype(), DType::C64);
    assert_eq!(outputs[1].dtype(), DType::C64);
    assert_eq!(outputs[2].dtype(), DType::C64);
    assert_eq!(outputs[3].dtype(), DType::C64);
    assert_eq!(outputs[4].dtype(), DType::F64);
    assert_eq!(outputs[4].shape(), &[] as &[usize]);
    let parity = outputs[4].as_slice::<f64>().unwrap()[0];
    assert!(parity == 1.0 || parity == -1.0);
}

#[test]
fn full_piv_lu_uses_column_pivot_when_max_pivot_is_off_column() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 100.0, 3.0]);
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let outputs =
        support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&a)).unwrap();
    let [_p, _l, u, q, _parity]: [Tensor; 5] = outputs.try_into().unwrap();

    assert_close(f64_data(&q), &[0.0, 1.0, 1.0, 0.0]);
    assert_close(&f64_data(&u)[..1], &[100.0]);
}

/// `(P, Q, rank)` of one `n x n` factorization, the rank counted as the `U` diagonal entries above
/// `ε · max|U_ii|`, the same threshold `full_piv_lu_solve` rejects.
fn factor_summary(a: &Tensor, n: usize) -> (Vec<f64>, Vec<f64>, usize) {
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let outputs = support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(a)).unwrap();
    let [p, l, u, q, _parity]: [Tensor; 5] = outputs.try_into().unwrap();
    let paqt = {
        let pa = matmul(&mut backend, &p, a);
        let qt = backend
            .transpose_read(TensorRead::from_tensor(&q), &[1, 0])
            .unwrap();
        matmul(&mut backend, &pa, &qt)
    };
    assert_close(f64_data(&paqt), f64_data(&matmul(&mut backend, &l, &u)));
    let diagonal: Vec<f64> = (0..n).map(|i| f64_data(&u)[i + n * i].abs()).collect();
    let largest = diagonal.iter().copied().fold(0.0_f64, f64::max);
    let rank = diagonal
        .iter()
        .filter(|&&d| d > f64::EPSILON * largest)
        .count();
    (f64_data(&p).to_vec(), f64_data(&q).to_vec(), rank)
}

// Complete-pivoting LU is one kernel, so these cases assert fixed values: every build
// returns the same P, Q and rank.

#[test]
fn full_piv_lu_factorizes_singular_matrix_and_reveals_rank() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]);
    let (p, q, rank) = factor_summary(&a, 2);
    assert_eq!(rank, 1);
    assert_eq!(p, vec![0.0, 1.0, 1.0, 0.0]);
    assert_eq!(q, vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn full_piv_lu_factorizes_zero_matrix_with_rank_zero() {
    let a = f64_tensor(vec![3, 3], vec![0.0; 9]);
    let (p, q, rank) = factor_summary(&a, 3);
    assert_eq!(rank, 0);
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(p, identity);
    assert_eq!(q, identity);
}

#[test]
fn full_piv_lu_keeps_the_identity_permutation_on_diagonal_ties() {
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let (p, q, rank) = factor_summary(&f64_tensor(vec![3, 3], identity.clone()), 3);
    assert_eq!(rank, 3);
    assert_eq!(p, identity);
    assert_eq!(q, identity);
}

#[test]
fn full_piv_lu_takes_the_first_column_major_maximum_on_off_diagonal_ties() {
    // The largest entries tie at (1, 0) and (0, 1). The rest is a unit diagonal plus small
    // distinct entries, so the matrix has full rank and no other entry reaches 2. The first pivot
    // is the first maximum in column-major order, (1, 0), on every SIMD width: P moves row 1 first
    // and Q keeps column 0 first.
    for n in [8, 16] {
        let mut data: Vec<f64> = (0..n * n)
            .map(|k| {
                let diagonal = if k % (n + 1) == 0 { 1.0 } else { 0.0 };
                diagonal + 1e-5 * ((k * 7919 + 13) % 997 + 1) as f64
            })
            .collect();
        data[1] = 2.0;
        data[n] = 2.0;
        let (p, q, rank) = factor_summary(&f64_tensor(vec![n, n], data), n);
        assert_eq!(rank, n, "n = {n}");
        assert_eq!(p[n], 1.0, "n = {n}: P must take row 1 first");
        assert_eq!(q[0], 1.0, "n = {n}: Q must keep column 0 first");
    }
}

#[test]
fn full_piv_lu_keeps_pivots_below_the_lapack_floor_unperturbed() {
    // `?getc2` would flag a pivot below `EPS * max` and overwrite it with that floor.
    let a = f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0e-17]);
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let outputs =
        support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&a)).unwrap();
    assert_eq!(f64_data(&outputs[2]), &[1.0, 0.0, 0.0, 1.0e-17]);
    assert_eq!(factor_summary(&a, 2).2, 1);
}

#[test]
fn full_piv_lu_solve_decides_singularity_by_rank() {
    let mut backend = tenferro_cpu::CpuEngine::serial();
    let small = f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0e-12]);
    let b = f64_tensor(vec![2, 1], vec![2.0, 3.0e-12]);
    let x = support::with_cpu_linalg(&mut backend, |backend| {
        backend.full_piv_lu_solve(&small, &b, false)
    })
    .unwrap();
    assert_close(f64_data(&x), &[2.0, 3.0]);

    let deficient = f64_tensor(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0e-17]);
    let error = support::with_cpu_linalg(&mut backend, |backend| {
        backend.full_piv_lu_solve(&deficient, &b, false)
    })
    .unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::NumericalFailure);
}

#[test]
fn full_piv_lu_solve_returns_expected_solution() {
    let a = f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]);
    let b = f64_tensor(vec![2, 1], vec![-1.0, 5.0]);
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let x = support::with_cpu_linalg(&mut backend, |backend| {
        backend.full_piv_lu_solve(&a, &b, false)
    })
    .unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    assert_close(f64_data(&x), &[4.0, -1.0]);
}

#[test]
fn full_piv_lu_solve_accepts_vector_rhs() {
    let a = f64_tensor(vec![2, 2], vec![0.0, 2.0, 1.0, 3.0]);
    let b = f64_tensor(vec![2], vec![-1.0, 5.0]);
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let x = support::with_cpu_linalg(&mut backend, |backend| {
        backend.full_piv_lu_solve(&a, &b, false)
    })
    .unwrap();

    assert_eq!(x.shape(), &[2]);
    assert_close(f64_data(&x), &[4.0, -1.0]);
}

#[test]
fn full_piv_lu_solve_accepts_small_and_large_scaled_f64_systems() {
    let base_a = [3.0, 0.0, 0.0, 2.0];
    let base_b = [6.0, 4.0];
    let mut backend = tenferro_cpu::CpuEngine::serial();

    for scale in [1.0e-20, 1.0e20] {
        let a = f64_tensor(
            vec![2, 2],
            base_a.iter().map(|value| value * scale).collect(),
        );
        let b = f64_tensor(
            vec![2, 1],
            base_b.iter().map(|value| value * scale).collect(),
        );
        let x = support::with_cpu_linalg(&mut backend, |backend| {
            backend.full_piv_lu_solve(&a, &b, false)
        })
        .unwrap();
        assert_close(f64_data(&x), &[2.0, 2.0]);
    }
}

#[test]
fn full_piv_lu_solve_accepts_small_and_large_scaled_f32_and_complex_systems() {
    let mut backend = tenferro_cpu::CpuEngine::serial();
    for scale in [1.0e-10_f32, 1.0e10_f32] {
        let a = f32_tensor(vec![2, 2], vec![3.0 * scale, 0.0, 0.0, 2.0 * scale]);
        let b = f32_tensor(vec![2, 1], vec![6.0 * scale, 4.0 * scale]);
        let x = support::with_cpu_linalg(&mut backend, |backend| {
            backend.full_piv_lu_solve(&a, &b, false)
        })
        .unwrap();
        let values = x.as_slice::<f32>().unwrap();
        assert!((values[0] - 2.0).abs() < 1.0e-5);
        assert!((values[1] - 2.0).abs() < 1.0e-5);
    }

    for scale in [1.0e-20_f64, 1.0e20_f64] {
        let a = Tensor::from_typed::<tenferro_tensor::Complex64>(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex64::new(3.0 * scale, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(2.0 * scale, 0.0),
                ],
            )
            .unwrap(),
        );
        let b = Tensor::from_typed::<tenferro_tensor::Complex64>(
            TypedTensor::from_vec_col_major(
                vec![2, 1],
                vec![
                    Complex64::new(6.0 * scale, 0.0),
                    Complex64::new(4.0 * scale, 0.0),
                ],
            )
            .unwrap(),
        );
        let x = support::with_cpu_linalg(&mut backend, |backend| {
            backend.full_piv_lu_solve(&a, &b, false)
        })
        .unwrap();
        let values = c64_data(&x);
        assert!((values[0].re - 2.0).abs() < 1.0e-12);
        assert!((values[1].re - 2.0).abs() < 1.0e-12);
    }

    for scale in [1.0e-10_f32, 1.0e10_f32] {
        let a = Tensor::from_typed::<tenferro_tensor::Complex32>(
            TypedTensor::from_vec_col_major(
                vec![2, 2],
                vec![
                    Complex32::new(3.0 * scale, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(0.0, 0.0),
                    Complex32::new(2.0 * scale, 0.0),
                ],
            )
            .unwrap(),
        );
        let b = Tensor::from_typed::<tenferro_tensor::Complex32>(
            TypedTensor::from_vec_col_major(
                vec![2, 1],
                vec![
                    Complex32::new(6.0 * scale, 0.0),
                    Complex32::new(4.0 * scale, 0.0),
                ],
            )
            .unwrap(),
        );
        let x = support::with_cpu_linalg(&mut backend, |backend| {
            backend.full_piv_lu_solve(&a, &b, false)
        })
        .unwrap();
        let values = c32_data(&x);
        assert!((values[0].re - 2.0).abs() < 1.0e-5);
        assert!((values[1].re - 2.0).abs() < 1.0e-5);
    }
}

#[test]
fn full_piv_lu_solve_rejects_exactly_singular_system() {
    let a = f64_tensor(vec![2, 2], vec![1.0, 2.0, 2.0, 4.0]);
    let b = f64_tensor(vec![2, 1], vec![1.0, 2.0]);
    let mut backend = tenferro_cpu::CpuEngine::serial();

    let error = support::with_cpu_linalg(&mut backend, |backend| {
        backend.full_piv_lu_solve(&a, &b, false)
    })
    .unwrap_err();
    assert_eq!(error.kind(), tenferro_tensor::ErrorKind::NumericalFailure);
}

// ---------------------------------------------------------------------------------------------
// Rectangular input (#2488 step 1): `A` is `m x n`, `k = min(m, n)`, and the factors are
// `P (m x m)`, `L (m x k)`, `U (k x n)`, `Q (n x n)` with `P A Qᵀ = L U`.
// ---------------------------------------------------------------------------------------------

fn transpose(backend: &mut CpuEngine, tensor: &Tensor) -> Tensor {
    backend
        .transpose_read(TensorRead::from_tensor(tensor), &[1, 0])
        .unwrap()
}

/// Every entry is 0 or 1 and every row and column holds exactly one 1.
fn assert_permutation_matrix(name: &str, data: &[f64], n: usize) {
    assert_eq!(data.len(), n * n, "{name}: {n} x {n} expected");
    assert!(
        data.iter().all(|&x| x == 0.0 || x == 1.0),
        "{name}: {data:?}"
    );
    for i in 0..n {
        let row: f64 = (0..n).map(|j| data[i + n * j]).sum();
        let col: f64 = (0..n).map(|j| data[j + n * i]).sum();
        assert_eq!((row, col), (1.0, 1.0), "{name}: row/column {i} of {data:?}");
    }
}

/// Factor one `m x n` f64 matrix, check the shapes, the triangular structure, the permutations
/// and `P A Qᵀ = L U`, and return the `|U_ii|`, `i < k`.
fn check_rectangular_factorization(a: &Tensor, m: usize, n: usize) -> Vec<f64> {
    let k = m.min(n);
    let mut backend = CpuEngine::serial();
    let outputs = support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(a)).unwrap();
    let [p, l, u, q, parity]: [Tensor; 5] = outputs.try_into().unwrap();

    assert_eq!(p.shape(), &[m, m]);
    assert_eq!(l.shape(), &[m, k]);
    assert_eq!(u.shape(), &[k, n]);
    assert_eq!(q.shape(), &[n, n]);
    assert_eq!(parity.shape(), &[] as &[usize]);
    assert!(f64_data(&parity)[0] == 1.0 || f64_data(&parity)[0] == -1.0);
    assert_permutation_matrix("P", f64_data(&p), m);
    assert_permutation_matrix("Q", f64_data(&q), n);
    let (l_data, u_data) = (f64_data(&l), f64_data(&u));
    for j in 0..k {
        assert_eq!(l_data[j + m * j], 1.0, "L must be unit diagonal");
        for i in 0..j {
            assert_eq!(l_data[i + m * j], 0.0, "L must be lower triangular");
        }
    }
    for j in 0..n {
        for i in (j + 1)..k {
            assert_eq!(u_data[i + k * j], 0.0, "U must be upper triangular");
        }
    }

    let qt = transpose(&mut backend, &q);
    let pa = matmul(&mut backend, &p, a);
    let paqt = matmul(&mut backend, &pa, &qt);
    let lu = matmul(&mut backend, &l, &u);
    assert_eq!(paqt.shape(), &[m, n]);
    assert_close(f64_data(&paqt), f64_data(&lu));
    // The documented inverse form, `A = Pᵀ L U Q`.
    let pt = transpose(&mut backend, &p);
    let ptlu = matmul(&mut backend, &pt, &lu);
    assert_close(f64_data(&matmul(&mut backend, &ptlu, &q)), f64_data(a));

    (0..k).map(|i| u_data[i + k * i].abs()).collect()
}

fn numerical_rank(diagonal: &[f64]) -> usize {
    let largest = diagonal.iter().copied().fold(0.0_f64, f64::max);
    diagonal
        .iter()
        .filter(|&&d| d > 64.0 * f64::EPSILON * largest)
        .count()
}

#[test]
fn full_piv_lu_factors_a_tall_matrix() {
    // 4 x 3, column major, full column rank.
    let a = f64_tensor(
        vec![4, 3],
        vec![
            1.0, 2.0, -1.0, 0.5, //
            3.0, -0.5, 2.0, 1.0, //
            -2.0, 4.0, 0.25, 7.0,
        ],
    );
    let diagonal = check_rectangular_factorization(&a, 4, 3);
    assert_eq!(numerical_rank(&diagonal), 3);
    // Complete pivoting takes the largest entry first: |U_00| = max |A_ij| = 7.
    assert_eq!(diagonal[0], 7.0);
}

#[test]
fn full_piv_lu_factors_a_wide_matrix() {
    // 2 x 5, column major, full row rank.
    let a = f64_tensor(
        vec![2, 5],
        vec![1.0, 2.0, -3.0, 0.5, 4.0, -1.0, 0.25, 9.0, 2.0, 2.0],
    );
    let diagonal = check_rectangular_factorization(&a, 2, 5);
    assert_eq!(numerical_rank(&diagonal), 2);
    assert_eq!(diagonal[0], 9.0);
}

#[test]
fn full_piv_lu_reveals_the_rank_of_rank_deficient_rectangular_matrices() {
    // Columns: c0, c1, c2 = c0 + 2 c1, so the 4 x 3 matrix has rank 2.
    let c0 = [1.0, 2.0, -1.0, 3.0];
    let c1 = [0.5, -1.0, 2.0, 1.0];
    let tall: Vec<f64> = c0
        .iter()
        .chain(&c1)
        .copied()
        .chain(c0.iter().zip(&c1).map(|(x, y)| x + 2.0 * y))
        .collect();
    let a = f64_tensor(vec![4, 3], tall.clone());
    assert_eq!(
        numerical_rank(&check_rectangular_factorization(&a, 4, 3)),
        2
    );

    // Its 3 x 4 transpose has the same rank.
    let wide: Vec<f64> = (0..4)
        .flat_map(|j| (0..3).map(move |i| (i, j)))
        .map(|(i, j)| tall[j + 4 * i])
        .collect();
    let at = f64_tensor(vec![3, 4], wide);
    assert_eq!(
        numerical_rank(&check_rectangular_factorization(&at, 3, 4)),
        2
    );

    // A rank-1 wide matrix: every row a multiple of the first.
    let rank_one = f64_tensor(vec![2, 3], vec![1.0, -2.0, 3.0, -6.0, 0.5, -1.0]);
    assert_eq!(
        numerical_rank(&check_rectangular_factorization(&rank_one, 2, 3)),
        1
    );
}

#[test]
fn full_piv_lu_factors_a_batch_of_rectangular_matrices() {
    // Two 3 x 2 items in batch order; each item must equal its own unbatched factorization.
    let items = [
        vec![1.0, 2.0, 3.0, -4.0, 0.5, 6.0],
        vec![0.0, 1.0, 0.0, 2.0, 0.0, 5.0],
    ];
    let batched = f64_tensor(vec![3, 2, 2], items.concat());
    let mut backend = CpuEngine::serial();
    let outputs =
        support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&batched)).unwrap();
    let shapes: Vec<&[usize]> = outputs.iter().map(|t| t.shape()).collect();
    assert_eq!(
        shapes,
        vec![
            &[3, 3, 2][..],
            &[3, 2, 2][..],
            &[2, 2, 2][..],
            &[2, 2, 2][..],
            &[2][..]
        ]
    );
    for (index, item) in items.iter().enumerate() {
        let single = f64_tensor(vec![3, 2], item.clone());
        check_rectangular_factorization(&single, 3, 2);
        let expected =
            support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&single)).unwrap();
        for (factor, (batch_out, item_out)) in outputs.iter().zip(&expected).enumerate() {
            let len = item_out.shape().iter().product::<usize>();
            assert_eq!(
                &f64_data(batch_out)[index * len..(index + 1) * len],
                f64_data(item_out),
                "batch item {index}, factor {factor}"
            );
        }
    }
}

#[test]
fn full_piv_lu_factors_rectangular_f32_and_complex_matrices() {
    let mut backend = CpuEngine::serial();
    let a = f32_tensor(vec![3, 2], vec![1.0, -2.0, 0.5, 4.0, 3.0, -1.0]);
    let outputs =
        support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&a)).unwrap();
    let shapes: Vec<&[usize]> = outputs.iter().map(|t| t.shape()).collect();
    assert_eq!(
        shapes,
        vec![&[3, 3][..], &[3, 2][..], &[2, 2][..], &[2, 2][..], &[][..]]
    );
    let lu = matmul(&mut backend, &outputs[1], &outputs[2]);
    let pa = matmul(&mut backend, &outputs[0], &a);
    let qt = transpose(&mut backend, &outputs[3]);
    let paqt = matmul(&mut backend, &pa, &qt);
    for (x, y) in paqt
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .zip(lu.as_slice::<f32>().unwrap())
    {
        assert!((x - y).abs() < 1.0e-5, "f32: {x} vs {y}");
    }

    let a = Tensor::from_typed::<Complex64>(
        TypedTensor::from_vec_col_major(
            vec![2, 3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(-1.0, 0.0),
                Complex64::new(0.25, 4.0),
                Complex64::new(2.0, 2.0),
            ],
        )
        .unwrap(),
    );
    let outputs =
        support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&a)).unwrap();
    let shapes: Vec<&[usize]> = outputs.iter().map(|t| t.shape()).collect();
    assert_eq!(
        shapes,
        vec![&[2, 2][..], &[2, 2][..], &[2, 3][..], &[3, 3][..], &[][..]]
    );
    assert_eq!(outputs[4].dtype(), DType::F64);
    let lu = matmul(&mut backend, &outputs[1], &outputs[2]);
    let pa = matmul(&mut backend, &outputs[0], &a);
    let qt = transpose(&mut backend, &outputs[3]);
    let paqt = matmul(&mut backend, &pa, &qt);
    for (x, y) in c64_data(&paqt).iter().zip(c64_data(&lu)) {
        assert!((x - y).norm() < 1.0e-12, "c64: {x} vs {y}");
    }
}

#[test]
fn full_piv_lu_accepts_empty_rectangular_shapes() {
    let mut backend = CpuEngine::serial();
    // (input shape, P, L, U, Q, parity)
    let cases: [(&[usize], [&[usize]; 5]); 4] = [
        (&[3, 0], [&[3, 3], &[3, 0], &[0, 0], &[0, 0], &[]]),
        (&[0, 3], [&[0, 0], &[0, 0], &[0, 3], &[3, 3], &[]]),
        (&[0, 0], [&[0, 0], &[0, 0], &[0, 0], &[0, 0], &[]]),
        (
            &[2, 3, 0],
            [&[2, 2, 0], &[2, 2, 0], &[2, 3, 0], &[3, 3, 0], &[0]],
        ),
    ];
    for (shape, expected) in cases {
        let a = f64_tensor(shape.to_vec(), vec![]);
        let outputs =
            support::with_cpu_linalg(&mut backend, |backend| backend.full_piv_lu(&a)).unwrap();
        let shapes: Vec<&[usize]> = outputs.iter().map(|t| t.shape()).collect();
        assert_eq!(shapes, expected.to_vec(), "input {shape:?}");
        if shape.len() == 2 {
            // One item: the permutations are identities and the parity is +1.
            let (m, n) = (shape[0], shape[1]);
            assert_permutation_matrix("P", f64_data(&outputs[0]), m);
            assert_permutation_matrix("Q", f64_data(&outputs[3]), n);
            assert_eq!(f64_data(&outputs[4]), &[1.0], "input {shape:?}");
        }
    }
}
