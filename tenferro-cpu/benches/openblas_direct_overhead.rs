use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;

use cblas_sys::{cblas_zgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};

#[cfg(feature = "provider-src")]
extern crate blas_src as _;
#[cfg(feature = "provider-src")]
extern crate cblas_src as _;

const PHYS_DIM: usize = 2;
const CHIS: &[usize] = &[1, 2, 4, 8, 16, 32];

fn complex_vec(len: usize, seed: usize) -> Vec<Complex64> {
    (0..len)
        .map(|idx| {
            let real = ((idx * 17 + seed * 13 + 3) % 97) as f64 / 97.0 - 0.5;
            let imag = ((idx * 29 + seed * 7 + 5) % 89) as f64 / 89.0 - 0.5;
            Complex64::new(real, imag)
        })
        .collect()
}

fn conj_vec(input: &[Complex64]) -> Vec<Complex64> {
    input.iter().map(|value| value.conj()).collect()
}

#[allow(clippy::too_many_arguments)]
fn zgemm(
    trans_a: CBLAS_TRANSPOSE,
    trans_b: CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
    k: usize,
    a: &[Complex64],
    lda: usize,
    b: &[Complex64],
    ldb: usize,
    c: &mut [Complex64],
    ldc: usize,
) {
    let alpha = [1.0_f64, 0.0_f64];
    let beta = [0.0_f64, 0.0_f64];
    unsafe {
        cblas_zgemm(
            CBLAS_LAYOUT::CblasColMajor,
            trans_a,
            trans_b,
            m as i32,
            n as i32,
            k as i32,
            &alpha as *const _,
            a.as_ptr() as *const _,
            lda as i32,
            b.as_ptr() as *const _,
            ldb as i32,
            &beta as *const _,
            c.as_mut_ptr() as *mut _,
            ldc as i32,
        );
    }
}

fn bench_openblas_direct_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("openblas_direct_overhead/c64/one_thread");

    for &chi in CHIS {
        let params = format!("chi_{chi}_d_{PHYS_DIM}");

        let first_a = complex_vec(PHYS_DIM * chi, 1);
        let first_b = complex_vec(PHYS_DIM * chi, 2);
        let mut first_c = vec![Complex64::new(0.0, 0.0); chi * chi];
        group.bench_function(BenchmarkId::new("first_site_zgemm", &params), |b| {
            b.iter(|| {
                zgemm(
                    CBLAS_TRANSPOSE::CblasConjTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    chi,
                    chi,
                    PHYS_DIM,
                    black_box(&first_a),
                    PHYS_DIM,
                    black_box(&first_b),
                    PHYS_DIM,
                    black_box(&mut first_c),
                    chi,
                );
                black_box(first_c[0]);
            });
        });

        let env = complex_vec(chi * chi, 3);
        let bra = complex_vec(chi * PHYS_DIM * chi, 4);
        let bra_conj = conj_vec(&bra);
        let mut tmp = vec![Complex64::new(0.0, 0.0); chi * PHYS_DIM * chi];
        group.bench_function(BenchmarkId::new("env_bra_zgemm_preconj", &params), |b| {
            b.iter(|| {
                zgemm(
                    CBLAS_TRANSPOSE::CblasTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    chi,
                    PHYS_DIM * chi,
                    chi,
                    black_box(&env),
                    chi,
                    black_box(&bra_conj),
                    chi,
                    black_box(&mut tmp),
                    chi,
                );
                black_box(tmp[0]);
            });
        });

        group.bench_function(BenchmarkId::new("env_bra_conj_copy", &params), |b| {
            let mut out = vec![Complex64::new(0.0, 0.0); bra.len()];
            b.iter(|| {
                for (dst, src) in out.iter_mut().zip(black_box(&bra)) {
                    *dst = src.conj();
                }
                black_box(out[0]);
            });
        });

        let ket = complex_vec(chi * PHYS_DIM * chi, 6);
        let mut out = vec![Complex64::new(0.0, 0.0); chi * chi];
        group.bench_function(BenchmarkId::new("tmp_ket_zgemm", &params), |b| {
            b.iter(|| {
                zgemm(
                    CBLAS_TRANSPOSE::CblasTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    chi,
                    chi,
                    chi * PHYS_DIM,
                    black_box(&tmp),
                    chi * PHYS_DIM,
                    black_box(&ket),
                    chi * PHYS_DIM,
                    black_box(&mut out),
                    chi,
                );
                black_box(out[0]);
            });
        });

        group.bench_function(
            BenchmarkId::new("site_update_2_zgemm_preconj", &params),
            |b| {
                b.iter(|| {
                    zgemm(
                        CBLAS_TRANSPOSE::CblasTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        chi,
                        PHYS_DIM * chi,
                        chi,
                        black_box(&env),
                        chi,
                        black_box(&bra_conj),
                        chi,
                        black_box(&mut tmp),
                        chi,
                    );
                    zgemm(
                        CBLAS_TRANSPOSE::CblasTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        chi,
                        chi,
                        chi * PHYS_DIM,
                        black_box(&tmp),
                        chi * PHYS_DIM,
                        black_box(&ket),
                        chi * PHYS_DIM,
                        black_box(&mut out),
                        chi,
                    );
                    black_box(out[0]);
                });
            },
        );

        group.bench_function(
            BenchmarkId::new("site_update_conj_copy_2_zgemm", &params),
            |b| {
                let mut bra_conj_each_iter = vec![Complex64::new(0.0, 0.0); bra.len()];
                b.iter(|| {
                    for (dst, src) in bra_conj_each_iter.iter_mut().zip(black_box(&bra)) {
                        *dst = src.conj();
                    }
                    zgemm(
                        CBLAS_TRANSPOSE::CblasTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        chi,
                        PHYS_DIM * chi,
                        chi,
                        black_box(&env),
                        chi,
                        black_box(&bra_conj_each_iter),
                        chi,
                        black_box(&mut tmp),
                        chi,
                    );
                    zgemm(
                        CBLAS_TRANSPOSE::CblasTrans,
                        CBLAS_TRANSPOSE::CblasNoTrans,
                        chi,
                        chi,
                        chi * PHYS_DIM,
                        black_box(&tmp),
                        chi * PHYS_DIM,
                        black_box(&ket),
                        chi * PHYS_DIM,
                        black_box(&mut out),
                        chi,
                    );
                    black_box(out[0]);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_openblas_direct_overhead);
criterion_main!(benches);
