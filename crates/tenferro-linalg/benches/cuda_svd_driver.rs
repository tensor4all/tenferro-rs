//! CUDA SVD driver comparison: cuSOLVER `gesvdj` / `gesvd` / `Xgesvdp` against
//! the default `Auto` policy, across dtype, order and driver (#2182, #1839, #1849).
//!
//! For each `(dtype, n, driver)` the harness measures one thin SVD
//! (`full_matrices = false`, vectors computed — the only form
//! [`SvdOptions`](tenferro_linalg::SvdOptions) exposes) followed by the
//! backend-native device synchronize, on a caller-held CUDA engine. Inputs are
//! uploaded to the device once, before timing; no iteration downloads, so the
//! timed region is the cuSOLVER call, its workspace/output allocation, and its
//! barrier and nothing else. The factors are dropped inside the timed region
//! because their allocation is part of the operation's cost.
//!
//! This lives in `tenferro-linalg` rather than the shared benchmark crate
//! because linalg owns `SvdOptions::driver`, and the repository rule keeps a
//! performance case with the code it exercises.
//!
//! `SvdDriver::Auto` is measured but stays the accuracy-preserving,
//! JAX-compatible policy (`gesvdj` up to the 1024 order threshold, `gesvd`
//! beyond it - #971); `Xgesvdp` is opt-in. The harness exists so the opt-in
//! tradeoff can be judged on per-driver time *and* per-driver accuracy:
//! `Xgesvdp` is faster for `min(m, n) >= 256` but its small singular values are
//! only accurate to about `1e-6` of the largest (cuSOLVER's `h_err_sigma`), so
//! an `eps`-scale truncation below that is not reliable.
//!
//! An untimed accuracy table is printed to stderr before the timing groups. It
//! reports the relative reconstruction error `||A - U diag(S) Vᴴ||_F / ||A||_F`
//! and the orthogonality residuals `||UᴴU - I||_F` and `||VᴴV - I||_F` from a
//! host download, because #1849 showed `Xgesvdp` accuracy is input-dependent
//! and can lose digits against `gesvd` on some families: time without accuracy
//! would mislead the policy decision.
//!
//! A `(driver, dtype)` pair the library reports as a typed
//! [`ErrorKind::Unsupported`] is skipped with a printed line rather than a
//! panic; any other failure is fatal.
//!
//! Arms: drivers `Auto`, `Gesvdj`, `Gesvd`, `Xgesvdp`; orders 64, 256, 1024;
//! dtypes `f32`, `f64`, `c32`, `c64`. The Criterion group id is
//! `cuda_svd_driver/<dtype>/n<n>/<driver>`.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::{Complex32, Complex64};
use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime,
};
use tenferro_linalg::{LinalgBackend, SvdDriver, SvdOptions};
use tenferro_tensor::{DType, ErrorKind, Tensor};

/// Square orders. 64 and 256 stay under the `Auto` `gesvdj` threshold; 1024 is
/// the JAX-compatible boundary itself, where `Auto` still chooses `gesvdj`.
const ORDERS: &[usize] = &[64, 256, 1024];

/// dtypes the CUDA SVD path accepts.
const DTYPES: &[DType] = &[DType::F32, DType::F64, DType::C32, DType::C64];

/// Drivers compared. `Auto` is the default policy; the rest are explicit.
const DRIVERS: &[SvdDriver] = &[
    SvdDriver::Auto,
    SvdDriver::Gesvdj,
    SvdDriver::Gesvd,
    SvdDriver::Xgesvdp,
];

/// One deterministic square input, host copy kept for the accuracy residuals.
struct MatrixCase {
    dtype: DType,
    n: usize,
    host: Tensor,
    device: Tensor,
}

/// A supported `(matrix, driver)` pair to time, in first-seen order.
struct TimingCase {
    matrix: usize,
    driver: SvdDriver,
}

/// Untimed host accuracy for one `(dtype, n, driver)`.
struct AccuracyRow {
    dtype: &'static str,
    n: usize,
    driver: &'static str,
    reconstruction: f64,
    orthogonality_u: f64,
    orthogonality_v: f64,
}

impl std::fmt::Display for AccuracyRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<6} {:>6} {:<9} {:>12.3e} {:>12.3e} {:>12.3e}",
            self.dtype,
            self.n,
            self.driver,
            self.reconstruction,
            self.orthogonality_u,
            self.orthogonality_v,
        )
    }
}

fn dtype_label(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::C32 => "c32",
        DType::C64 => "c64",
        other => panic!("the SVD driver bench has no label for {other:?}"),
    }
}

fn driver_label(driver: SvdDriver) -> &'static str {
    match driver {
        SvdDriver::Auto => "Auto",
        SvdDriver::Gesvdj => "Gesvdj",
        SvdDriver::Gesvd => "Gesvd",
        SvdDriver::Xgesvdp => "Xgesvdp",
    }
}

fn case_label(dtype: DType, n: usize, driver: SvdDriver) -> String {
    format!("{}/n{n}/{}", dtype_label(dtype), driver_label(driver))
}

/// Fixed-seed splitmix64 stream (public-domain algorithm) for a dense matrix.
///
/// A small deterministic PRNG avoids an external crate while giving a matrix
/// with no low-rank structure: the previous generator keyed every entry off
/// `(13*row + 17*col) mod 31`, so it had rank at most 31 and, with the `2I`
/// shift, a spectrum clustered about 2 for large `n`. Jacobi, QR and polar
/// drivers behave differently on that cluster, which biases the comparison.
struct SplitMix64(u64);

impl SplitMix64 {
    /// Start the stream at `seed`.
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Next 64-bit state (splitmix64).
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform draw in `[-1, 1)` from the top 53 bits.
    fn next_uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) / (1_u64 << 52) as f64 - 1.0
    }
}

/// The deterministic `n x n` input for `dtype`, in column-major order.
///
/// Every entry is an independent uniform draw in `[-1, 1)` (real and imaginary
/// parts independently for complex), so the matrix is dense and full rank with
/// probability one and its singular values follow the usual random-matrix
/// spread rather than concentrating at one value. Real dtypes share one draw
/// stream and complex dtypes share another, so `f32`/`f64` sample the same
/// underlying matrix and `c32`/`c64` sample a common one.
fn host_matrix(dtype: DType, n: usize) -> Tensor {
    let shape = vec![n, n];
    // Fixed seed: every run and every dtype observes the same input.
    let mut rng = SplitMix64::new(0x5EED_2182_0000_0001);
    match dtype {
        DType::F32 => Tensor::from_vec_col_major(
            shape,
            (0..n * n).map(|_| rng.next_uniform() as f32).collect(),
        )
        .expect("host f32 matrix"),
        DType::F64 => {
            Tensor::from_vec_col_major(shape, (0..n * n).map(|_| rng.next_uniform()).collect())
                .expect("host f64 matrix")
        }
        DType::C32 => Tensor::from_vec_col_major(
            shape,
            (0..n * n)
                .map(|_| {
                    let real = rng.next_uniform();
                    let imag = rng.next_uniform();
                    Complex32::new(real as f32, imag as f32)
                })
                .collect(),
        )
        .expect("host c32 matrix"),
        DType::C64 => Tensor::from_vec_col_major(
            shape,
            (0..n * n)
                .map(|_| {
                    let real = rng.next_uniform();
                    let imag = rng.next_uniform();
                    Complex64::new(real, imag)
                })
                .collect(),
        )
        .expect("host c64 matrix"),
        other => panic!("the SVD driver bench has no builder for {other:?}"),
    }
}

/// Decode any host SVD operand to `Complex64` for a dtype-generic checker.
///
/// The accuracy pass is untimed, so it trades the four scalar types for one
/// representation instead of duplicating the residual arithmetic per dtype.
fn host_values(tensor: &Tensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DType::F32 => tensor
            .as_slice::<f32>()
            .expect("host f32")
            .iter()
            .map(|&value| Complex64::new(f64::from(value), 0.0))
            .collect(),
        DType::F64 => tensor
            .as_slice::<f64>()
            .expect("host f64")
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        DType::C32 => tensor
            .as_slice::<Complex32>()
            .expect("host c32")
            .iter()
            .map(|&value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect(),
        DType::C64 => tensor.as_slice::<Complex64>().expect("host c64").to_vec(),
        other => panic!("the SVD accuracy pass has no decoder for {other:?}"),
    }
}

/// `||A - U diag(S) Vt||_F / ||A||_F` for square `n x n` column-major factors.
// INVARIANT: the index arithmetic below is the column-major layout contract,
// verified by the reconstruction test in `gpu_linalg.rs`; the loops are over
// elements, not over a slice whose iteration clippy can rewrite.
#[allow(clippy::needless_range_loop)]
fn reconstruction_error(
    a: &[Complex64],
    u: &[Complex64],
    s: &[Complex64],
    vt: &[Complex64],
    n: usize,
) -> f64 {
    let a_sq: f64 = a.iter().map(|value| value.norm_sqr()).sum();
    let diff_sq: f64 = (0..n)
        .into_iter()
        .map(|col| {
            let mut local = 0.0;
            for row in 0..n {
                let mut acc = Complex64::new(0.0, 0.0);
                for (j, sigma) in s.iter().enumerate() {
                    acc += u[row + n * j] * sigma.re * vt[j + n * col];
                }
                local += (acc - a[row + n * col]).norm_sqr();
            }
            local
        })
        .sum();
    (diff_sq / a_sq).sqrt()
}

/// `||M^H M - I||_F` for a column-major `rows x cols` `M` (so `cols x cols` Gram).
// INVARIANT: index arithmetic encodes column-major storage; see
// `reconstruction_error`.
#[allow(clippy::needless_range_loop)]
fn orthogonality_columns(matrix: &[Complex64], rows: usize, cols: usize) -> f64 {
    let total: f64 = (0..cols)
        .into_iter()
        .map(|p| {
            let mut local = 0.0;
            for q in 0..cols {
                let mut acc = Complex64::new(0.0, 0.0);
                for i in 0..rows {
                    acc += matrix[i + rows * p].conj() * matrix[i + rows * q];
                }
                local += (acc - identity(p, q)).norm_sqr();
            }
            local
        })
        .sum();
    total.sqrt()
}

/// `||M M^H - I||_F` for a column-major `rows x cols` `M` (so `rows x rows` Gram).
///
/// For `Vt` this is the requested `||V^H V - I||_F` because `V = Vt^H`.
// INVARIANT: index arithmetic encodes column-major storage; see
// `reconstruction_error`.
#[allow(clippy::needless_range_loop)]
fn orthogonality_rows(matrix: &[Complex64], rows: usize, cols: usize) -> f64 {
    let total: f64 = (0..rows)
        .into_iter()
        .map(|p| {
            let mut local = 0.0;
            for q in 0..rows {
                let mut acc = Complex64::new(0.0, 0.0);
                for j in 0..cols {
                    acc += matrix[p + rows * j] * matrix[q + rows * j].conj();
                }
                local += (acc - identity(p, q)).norm_sqr();
            }
            local
        })
        .sum();
    total.sqrt()
}

fn identity(p: usize, q: usize) -> Complex64 {
    if p == q {
        Complex64::new(1.0, 0.0)
    } else {
        Complex64::new(0.0, 0.0)
    }
}

/// Run one SVD through the scoped entry the CUDA linalg tests use, download the
/// factors, and compute the residuals on the host. Untimed.
fn accuracy_row(
    backend: &mut CudaEngine,
    runtime: &CudaRuntime,
    host: &Tensor,
    device: &Tensor,
    options: SvdOptions,
) -> tenferro_tensor::Result<AccuracyRow> {
    let outputs = backend.svd_with_options(device, options)?;
    let u = download_tensor(runtime, &outputs[0]).expect("download U");
    let s = download_tensor(runtime, &outputs[1]).expect("download S");
    let vt = download_tensor(runtime, &outputs[2]).expect("download Vt");
    let n = host.shape()[0];
    let a = host_values(host);
    let u = host_values(&u);
    let s = host_values(&s);
    let vt = host_values(&vt);
    Ok(AccuracyRow {
        dtype: dtype_label(host.dtype()),
        n,
        driver: driver_label(options.driver),
        reconstruction: reconstruction_error(&a, &u, &s, &vt, n),
        orthogonality_u: orthogonality_columns(&u, n, n),
        orthogonality_v: orthogonality_rows(&vt, n, n),
    })
}

/// One timed iteration: the SVD on the engine's captured stream, then the
/// backend-native device synchronize. No host download.
fn time_svd(engine: &mut CudaEngine, device: &Tensor, options: SvdOptions) {
    let outputs = engine
        .svd_with_options(device, options)
        .expect("svd_with_options");
    black_box(outputs);
    engine.runtime().synchronize().expect("device synchronize");
}

fn bench_svd_driver(c: &mut Criterion) {
    assert!(
        gpu_available(),
        "the CUDA SVD driver comparison requires a CUDA device"
    );
    let mut backend = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("CUDA backend");
    let runtime = backend.runtime().clone();

    // Deterministic inputs, host and device, prepared once before any timing.
    let mut matrices = Vec::new();
    for &dtype in DTYPES {
        for &n in ORDERS {
            let host = host_matrix(dtype, n);
            let device = upload_tensor(&runtime, &host).expect("upload input");
            matrices.push(MatrixCase {
                dtype,
                n,
                host,
                device,
            });
        }
    }

    // Untimed accuracy table and typed support detection, through the scoped
    // route the CUDA linalg tests use.
    eprintln!("cuda_svd_driver accuracy (host, after download):");
    eprintln!(
        "{:<6} {:>6} {:<9} {:>12} {:>12} {:>12}",
        "dtype", "n", "driver", "rel_recon", "orth_u", "orth_v"
    );
    let mut timings = Vec::new();
    for (index, matrix) in matrices.iter().enumerate() {
        for &driver in DRIVERS {
            let options = SvdOptions::default().driver(driver);
            let label = case_label(matrix.dtype, matrix.n, driver);
            match accuracy_row(
                &mut backend,
                &runtime,
                &matrix.host,
                &matrix.device,
                options,
            ) {
                Err(error) if error.kind() == ErrorKind::Unsupported => {
                    eprintln!("skip {label}: {error}");
                }
                Err(error) => panic!("{label}: {error}"),
                Ok(row) => {
                    eprintln!("{row}");
                    timings.push(TimingCase {
                        matrix: index,
                        driver,
                    });
                }
            }
        }
    }

    // Timing at steady state: one caller-held CUDA engine for every arm, so the
    // cuSOLVER call and its barrier are not charged a per-operation admission.
    let mut group = c.benchmark_group("cuda_svd_driver");
    for timing in &timings {
        let matrix = &matrices[timing.matrix];
        let options = SvdOptions::default().driver(timing.driver);
        let label = case_label(matrix.dtype, matrix.n, timing.driver);
        group.bench_function(&label, |bencher| {
            bencher.iter(|| time_svd(&mut backend, &matrix.device, options));
        });
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_svd_driver
}
criterion_main!(benches);
