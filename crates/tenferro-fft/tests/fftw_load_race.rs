#![cfg(feature = "fftw")]
//! The first threaded `Fftw::load` of an installation initializes FFTW's
//! threads, which adds solvers to that installation's global planner. A plan
//! being created on another thread at that moment must not see the planner
//! change under it. The race can happen only once per process, so each attempt
//! runs in a fresh child process of this test binary.

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Barrier};

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftOptions, Fftw, FftwError, FftwPlanning};
use tenferro_tensor::Tensor;

const CHILD: &str = "TENFERRO_TEST_FFTW_LOAD_RACE_CHILD";
const ATTEMPTS: usize = 60;

fn load(threads: usize) -> Option<Fftw> {
    let directory = std::env::var_os("TENFERRO_TEST_FFTW_DIR").map(PathBuf::from);
    let threads = NonZeroUsize::new(threads).unwrap();
    let result = match directory.as_deref() {
        Some(directory) => Fftw::load_from(directory, FftwPlanning::Estimate, threads),
        None => Fftw::load(FftwPlanning::Estimate, threads),
    };
    match result {
        Ok(fftw) => Some(fftw),
        Err(error @ FftwError::Load { .. })
            if directory.is_none() && std::env::var_os("TENFERRO_TEST_REQUIRE_FFTW").is_none() =>
        {
            eprintln!("FFTW integration requires a runtime: {error}");
            None
        }
        Err(error) => panic!("{error}"),
    }
}

/// One attempt in a fresh process: several threads create their first plans
/// with a single-threaded provider while the main thread performs the first
/// threaded load of the same installation. Before any plan exists, both sides
/// reach FFTW's global planner: the plans to create it, the threaded load to
/// initialize FFTW's threads in it.
fn race_once() {
    const PLANNERS: usize = 4;
    let Some(serial) = load(1) else { return };
    let start = Arc::new(Barrier::new(PLANNERS + 1));
    let planners: Vec<_> = (0..PLANNERS)
        .map(|k| {
            let (serial, start) = (serial.clone(), start.clone());
            std::thread::spawn(move || {
                let mut engine = CpuEngine::serial();
                serial.install(&mut engine);
                let mut executor = FftExecutor::default();
                start.wait();
                for step in 0..16 {
                    let len = 64 + PLANNERS * step + k;
                    let x = Tensor::from_vec_col_major([len], vec![Complex64::new(1.0, 0.0); len])
                        .unwrap();
                    let y = executor
                        .fft_with(&x, &FftOptions::default(), &mut engine)
                        .unwrap();
                    let dc = y.as_slice::<Complex64>().unwrap()[0];
                    assert!((dc - Complex64::new(len as f64, 0.0)).norm() < 1e-9, "{dc}");
                }
            })
        })
        .collect();
    start.wait();
    let threaded = load(2).expect("the serial load found the runtime");
    for planner in planners {
        planner.join().unwrap();
    }
    drop(threaded);
}

#[test]
fn first_threaded_load_does_not_race_concurrent_planning() {
    if std::env::var_os(CHILD).is_some() {
        race_once();
        return;
    }
    if load(1).is_none() {
        return;
    }
    let exe = std::env::current_exe().unwrap();
    for attempt in 0..ATTEMPTS {
        let output = Command::new(&exe)
            .args([
                "first_threaded_load_does_not_race_concurrent_planning",
                "--exact",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "attempt {attempt} failed with {:?}:\n{}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
