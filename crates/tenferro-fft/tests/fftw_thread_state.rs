#![cfg(feature = "fftw")]
//! FFTW's planner thread count is process-wide, so this check runs in its own
//! test binary where no concurrently running test plans with FFTW.

use std::ffi::c_int;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use libloading::Library;
use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftOptions, Fftw, FftwError, FftwPlanning};
use tenferro_tensor::Tensor;

type SetThreads = unsafe extern "C" fn(c_int);
type GetThreads = unsafe extern "C" fn() -> c_int;

fn library_path(directory: Option<&Path>, stem: &str) -> PathBuf {
    let name = if cfg!(target_os = "macos") {
        format!("lib{stem}.3.dylib")
    } else if cfg!(windows) {
        format!("lib{stem}-3.dll")
    } else {
        format!("lib{stem}.so.3")
    };
    directory.map_or_else(|| PathBuf::from(&name), |path| path.join(&name))
}

#[test]
fn threaded_planning_restores_the_planner_thread_count() {
    let directory = std::env::var_os("TENFERRO_TEST_FFTW_DIR").map(PathBuf::from);
    let threads = NonZeroUsize::new(2).unwrap();
    let loaded = match directory.as_deref() {
        Some(directory) => Fftw::load_from(directory, FftwPlanning::Estimate, threads),
        None => Fftw::load(FftwPlanning::Estimate, threads),
    };
    let provider = match loaded {
        Ok(provider) => provider,
        Err(error @ FftwError::Load { .. })
            if directory.is_none() && std::env::var_os("TENFERRO_TEST_REQUIRE_FFTW").is_none() =>
        {
            eprintln!("FFTW integration requires a runtime: {error}");
            return;
        }
        Err(error) => panic!("{error}"),
    };

    // The provider has already loaded these libraries, so these handles refer
    // to the same FFTW instance and its global planner state.
    // SAFETY: the libraries come from the FFTW installation the provider loaded.
    let core = unsafe { Library::new(library_path(directory.as_deref(), "fftw3")) }.unwrap();
    // SAFETY: `fftw_planner_nthreads` has this C signature.
    let threads_in_core = unsafe { core.get::<GetThreads>(b"fftw_planner_nthreads") }.is_ok();
    let companion;
    let thread_library = if threads_in_core {
        &core
    } else {
        // SAFETY: as above, from the same FFTW installation.
        companion =
            unsafe { Library::new(library_path(directory.as_deref(), "fftw3_threads")) }.unwrap();
        &companion
    };
    // SAFETY: both symbols have these C signatures.
    let (set, get) = unsafe {
        (
            *thread_library
                .get::<SetThreads>(b"fftw_plan_with_nthreads")
                .unwrap(),
            *thread_library
                .get::<GetThreads>(b"fftw_planner_nthreads")
                .unwrap(),
        )
    };

    // SAFETY: threading was initialized by `Fftw::load`, and no other thread
    // in this test binary calls FFTW.
    unsafe { set(3) };
    let mut engine = CpuEngine::serial();
    provider.install(&mut engine);
    let x = Tensor::from_vec_col_major([8], vec![Complex64::new(1.0, 0.0); 8]).unwrap();
    let y = FftExecutor::default()
        .fft_with(&x, &FftOptions::default(), &mut engine)
        .unwrap();
    assert_eq!(
        y.as_slice::<Complex64>().unwrap()[0],
        Complex64::new(8.0, 0.0)
    );
    // SAFETY: as above.
    assert_eq!(unsafe { get() }, 3);
    // SAFETY: as above.
    unsafe { set(1) };
}
