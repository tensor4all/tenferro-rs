//! Shared helpers for the FFTW planning-concurrency test binaries.
#![allow(dead_code)]

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftOptions, Fftw, FftwError, FftwPlanning};
use tenferro_tensor::Tensor;

/// Plans each planning thread creates. Override with
/// `TENFERRO_TEST_FFTW_PLANNING_ROUNDS` to run a longer stress.
pub fn rounds() -> usize {
    std::env::var("TENFERRO_TEST_FFTW_PLANNING_ROUNDS")
        .ok()
        .map_or(600, |value| value.parse().expect("a plan count"))
}

pub fn test_directory() -> Option<PathBuf> {
    std::env::var_os("TENFERRO_TEST_FFTW_DIR").map(PathBuf::from)
}

pub fn fftw_required() -> bool {
    test_directory().is_some() || std::env::var_os("TENFERRO_TEST_REQUIRE_FFTW").is_some()
}

/// Load a provider from `directory`, or from the system search path; `None`
/// when FFTW is not installed and the run does not require it.
pub fn load(directory: Option<&Path>, threads: usize) -> Option<Fftw> {
    let threads = NonZeroUsize::new(threads).unwrap();
    let result = match directory {
        Some(directory) => Fftw::load_from(directory, FftwPlanning::Estimate, threads),
        None => Fftw::load(FftwPlanning::Estimate, threads),
    };
    match result {
        Ok(fftw) => Some(fftw),
        Err(error @ FftwError::Load { .. }) if !fftw_required() => {
            eprintln!("FFTW integration requires a runtime: {error}");
            None
        }
        Err(error) => panic!("{error}"),
    }
}

/// Create and destroy `rounds` distinct FFTW plans through the public API:
/// complex f64, real f64 and complex f32 transforms of lengths no other
/// `thread` uses. A fresh engine every few steps destroys the cached plans, so
/// destruction interleaves with other threads' planning.
pub fn plan_many(provider: &Fftw, thread: usize, threads: usize, rounds: usize) {
    let mut engine = CpuEngine::serial();
    provider.install(&mut engine);
    let mut executor = FftExecutor::default();
    let options = FftOptions::default();
    for step in 0..rounds {
        if step % 8 == 0 {
            engine = CpuEngine::serial();
            provider.install(&mut engine);
            executor = FftExecutor::default();
        }
        let len = 16 + threads * step + thread;
        match step % 3 {
            0 => {
                let x =
                    Tensor::from_vec_col_major([len], vec![Complex64::new(1.0, 0.0); len]).unwrap();
                let y = executor.fft_with(&x, &options, &mut engine).unwrap();
                let dc = y.as_slice::<Complex64>().unwrap()[0];
                assert!((dc - Complex64::new(len as f64, 0.0)).norm() < 1e-9, "{dc}");
            }
            1 => {
                let x = Tensor::from_vec_col_major([len], vec![1.0_f64; len]).unwrap();
                let y = executor.rfft_with(&x, &options, &mut engine).unwrap();
                let dc = y.as_slice::<Complex64>().unwrap()[0];
                assert!((dc - Complex64::new(len as f64, 0.0)).norm() < 1e-9, "{dc}");
            }
            _ => {
                let x =
                    Tensor::from_vec_col_major([len], vec![Complex32::new(1.0, 0.0); len]).unwrap();
                let y = executor.fft_with(&x, &options, &mut engine).unwrap();
                let dc = y.as_slice::<Complex32>().unwrap()[0];
                let tolerance = 1e-4 * len as f32;
                assert!(
                    (dc - Complex32::new(len as f32, 0.0)).norm() < tolerance,
                    "{dc}"
                );
            }
        }
    }
}

/// Run `plan_many` on `threads` threads at once, released together.
pub fn plan_concurrently(provider: &Fftw, threads: usize, rounds: usize) {
    let start = std::sync::Arc::new(std::sync::Barrier::new(threads));
    let workers: Vec<_> = (0..threads)
        .map(|thread| {
            let (provider, start) = (provider.clone(), start.clone());
            std::thread::spawn(move || {
                start.wait();
                plan_many(&provider, thread, threads, rounds);
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}

pub fn library_path(directory: Option<&Path>, stem: &str) -> PathBuf {
    let name = if cfg!(target_os = "macos") {
        format!("lib{stem}.3.dylib")
    } else if cfg!(windows) {
        format!("lib{stem}-3.dll")
    } else {
        format!("lib{stem}.so.3")
    };
    directory.map_or_else(|| PathBuf::from(&name), |path| path.join(&name))
}
