#![cfg(all(feature = "fftw", target_os = "linux"))]
//! FFTW starts its worker threads from the thread that plans or executes a
//! threaded plan, and they keep that thread's CPU mask. This binary pins the
//! calling thread to one CPU, as a placed team leader is, runs a threaded FFTW
//! transform on a team engine, and checks that every thread FFTW started may run
//! on every CPU of the original mask. FFTW keeps its workers for the rest of the
//! process and `Measure` starts them while planning, so each planning mode runs
//! in a child process of its own.

use std::collections::BTreeSet;
use std::ffi::c_int;
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::Command;

use num_complex::Complex64;
use tenferro_cpu::{CpuEngine, Pool};
use tenferro_fft::{FftExecutor, FftOptions, Fftw, FftwError, FftwPlanning};
use tenferro_tensor::Tensor;

const MODE: &str = "TENFERRO_TEST_FFTW_PLACEMENT_MODE";

/// glibc's `cpu_set_t`: 1024 CPUs.
#[repr(C)]
#[derive(Clone, Copy)]
struct CpuSet([u64; 16]);

extern "C" {
    fn sched_getaffinity(pid: c_int, size: usize, mask: *mut CpuSet) -> c_int;
    fn sched_setaffinity(pid: c_int, size: usize, mask: *const CpuSet) -> c_int;
}

/// CPUs of thread `tid`; 0 is the calling thread.
fn cpus(tid: c_int) -> BTreeSet<usize> {
    let mut set = CpuSet([0; 16]);
    // SAFETY: the mask has glibc's `cpu_set_t` size and layout.
    let status = unsafe { sched_getaffinity(tid, std::mem::size_of::<CpuSet>(), &mut set) };
    assert_eq!(status, 0, "sched_getaffinity({tid}) failed");
    (0..1024)
        .filter(|cpu| set.0[cpu / 64] & (1 << (cpu % 64)) != 0)
        .collect()
}

fn pin_to(cpu: usize) {
    let mut set = CpuSet([0; 16]);
    set.0[cpu / 64] |= 1 << (cpu % 64);
    // SAFETY: as above.
    let status = unsafe { sched_setaffinity(0, std::mem::size_of::<CpuSet>(), &set) };
    assert_eq!(status, 0, "sched_setaffinity to CPU {cpu} failed");
}

fn tasks() -> BTreeSet<c_int> {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .map(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_str()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

#[test]
fn fftw_threads_keep_the_callers_cpu_set() {
    if let Some(mode) = std::env::var_os(MODE) {
        let planning = match mode.to_str() {
            Some("estimate") => FftwPlanning::Estimate,
            Some("measure") => FftwPlanning::Measure,
            _ => panic!("unknown {MODE}: {mode:?}"),
        };
        check(planning);
        return;
    }
    for mode in ["estimate", "measure"] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "fftw_threads_keep_the_callers_cpu_set",
                "--exact",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(MODE, mode)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{mode} planning:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
    }
}

fn check(planning: FftwPlanning) {
    let directory = std::env::var_os("TENFERRO_TEST_FFTW_DIR");
    let threads = NonZeroUsize::new(3).unwrap();
    let loaded = match directory.as_deref() {
        Some(directory) => Fftw::load_from(Path::new(directory), planning, threads),
        None => Fftw::load(planning, threads),
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
    let original = cpus(0);
    if original.len() < 2 {
        eprintln!("placement check needs at least two CPUs, found {original:?}");
        return;
    }
    // The pool's workers start with the original mask.
    let mut engine = CpuEngine::new(Pool::builder().threads(2).build().unwrap());
    assert!(engine.has_team());
    provider.install(&mut engine);
    let before = tasks();
    pin_to(*original.first().unwrap());

    // One lane, fewer than the engine's two slots, so the lane runs a threaded plan.
    let len = 1 << 20;
    let x = Tensor::from_vec_col_major([len], vec![Complex64::new(1.0, 0.0); len]).unwrap();
    let y = FftExecutor::default()
        .fft_with(&x, &FftOptions::default(), &mut engine)
        .unwrap();
    assert_eq!(
        y.as_slice::<Complex64>().unwrap()[0],
        Complex64::new(len as f64, 0.0)
    );

    let started: Vec<_> = tasks().difference(&before).copied().collect();
    assert!(
        !started.is_empty(),
        "the threaded FFTW plan started no worker threads"
    );
    for tid in started {
        assert_eq!(cpus(tid), original, "FFTW thread {tid}");
    }
}
