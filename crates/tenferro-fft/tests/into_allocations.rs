//! Counts every Rust allocation, including provider-worker allocations. Run this
//! binary alone: its single test keeps the harness and other cases out of the
//! measurement window. The allocator follows the repository's eager_allocations
//! test; no pencil-rs implementation is copied.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use num_complex::{Complex32, Complex64};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{FftExecutor, FftOps, FftOptions, FftnOptions};
use tenferro_tensor::{AsView, Dest, Tensor, TensorRead, TensorView, TypedTensorView};

static ENABLED: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
static LARGE: AtomicUsize = AtomicUsize::new(0);
static LARGE_THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
struct CountingAllocator;
fn record(size: usize) {
    if ENABLED.load(Ordering::Relaxed) {
        COUNT.fetch_add(1, Ordering::Relaxed);
        if size >= LARGE_THRESHOLD.load(Ordering::Relaxed) {
            LARGE.fetch_add(1, Ordering::Relaxed);
        }
    }
}
// SAFETY: every allocator argument is forwarded unchanged to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: original allocation layout.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        // SAFETY: original allocation layout.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        // SAFETY: original allocation and requested size.
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: this allocation originated from System.
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn measure(f: impl FnOnce()) -> (usize, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.store(false, Ordering::SeqCst);
        }
    }
    COUNT.store(0, Ordering::Relaxed);
    LARGE.store(0, Ordering::Relaxed);
    ENABLED.store(true, Ordering::SeqCst);
    let reset = Reset;
    f();
    drop(reset);
    (COUNT.load(Ordering::Relaxed), LARGE.load(Ordering::Relaxed))
}

macro_rules! no_alloc {
    ($call:expr, $label:expr) => {{
        for _ in 0..4 {
            $call.unwrap();
        }
        let counts = measure(|| {
            for _ in 0..12 {
                std::hint::black_box($call).unwrap();
            }
        });
        assert_eq!(counts.0, 0, "{} allocated after warm-up", $label);
    }};
}

#[test]
fn warmed_fft_family_and_column_major_gather_contract() {
    // Keep both pools alive across scalar widths: teardown of a previous pool
    // must not enter another case's process-wide allocation window.
    let pool = tenferro_cpu::Pool::builder().threads(4).build().unwrap();
    // Warm Rayon's first-sleep allocations before reserving workers, as in
    // threadteam 0.5.1 tests/team_alloc_free.rs::park_workers.
    std::thread::sleep(std::time::Duration::from_millis(300));
    let mut engines = [CpuEngine::serial(), CpuEngine::new(pool)];
    assert_eq!(engines[0].num_threads(), 1);
    assert!((1..=4).contains(&engines[1].num_threads()));
    eprintln!("effective CPU widths: 1, {}", engines[1].num_threads());
    let startup =
        Tensor::from_vec_col_major([64, 32, 32], vec![Complex64::new(1.0, 0.0); 65536]).unwrap();
    for engine in &mut engines {
        for _ in 0..4 {
            drop(
                startup
                    .fft_with(&FftOptions::default().axis(1), engine)
                    .unwrap(),
            );
        }
    }
    macro_rules! check_width {
        ($real:ty, $complex:ty, $tol:expr) => {{
            for (threads, engine) in [1, 4].into_iter().zip(engines.iter_mut()) {
                for shape in [vec![4, 6], vec![17, 7], vec![64, 32, 32], vec![2; 9]] {
                    let count: usize = shape.iter().product();
                    let input =
                        Tensor::from_vec_col_major(shape.clone(), vec![1.0 as $real; count])
                            .unwrap();
                    let complex = Tensor::from_vec_col_major(
                        shape.clone(),
                        vec![<$complex>::new(1.0, 0.0); count],
                    )
                    .unwrap();
                    let mut executor = FftExecutor::default();
                    for axis in 0..shape.len() {
                        eprintln!(
                            "allocation case {} threads={threads} shape={shape:?} axis={axis}",
                            stringify!($real)
                        );
                        let opts = FftOptions::default().axis(axis as isize);
                        let mut full = complex.fft_with(&opts, engine).unwrap();
                        no_alloc!(
                            executor.fft_into(&complex, &opts, engine, Dest::overwrite(&mut full)),
                            "c2c forward"
                        );
                        no_alloc!(
                            executor.ifft_into(&complex, &opts, engine, Dest::overwrite(&mut full)),
                            "c2c inverse"
                        );
                        no_alloc!(
                            executor.fft_into(&input, &opts, engine, Dest::overwrite(&mut full)),
                            "full r2c"
                        );
                        let mut spectrum = input.rfft_with(&opts, engine).unwrap();
                        no_alloc!(
                            executor.rfft_into(
                                &input,
                                &opts,
                                engine,
                                Dest::overwrite(&mut spectrum)
                            ),
                            "r2c"
                        );
                        let opts = opts.n(Some(shape[axis]));
                        let mut restored =
                            Tensor::from_vec_col_major(shape.clone(), vec![<$real>::NAN; count])
                                .unwrap();
                        no_alloc!(
                            executor.irfft_into(
                                &spectrum,
                                &opts,
                                engine,
                                Dest::overwrite(&mut restored)
                            ),
                            "c2r"
                        );
                        let error = restored
                            .as_slice::<$real>()
                            .unwrap()
                            .iter()
                            .map(|v| (*v - 1.0).abs())
                            .fold(0.0 as $real, <$real>::max);
                        assert!(error < $tol, "real roundtrip max error {error}");
                    }
                    let opts = FftnOptions::default();
                    let mut full = complex.fftn_with(&opts, engine).unwrap();
                    no_alloc!(
                        executor.fftn_into(&complex, &opts, engine, Dest::overwrite(&mut full)),
                        "nd c2c forward"
                    );
                    no_alloc!(
                        executor.ifftn_into(&complex, &opts, engine, Dest::overwrite(&mut full)),
                        "nd c2c inverse"
                    );
                    no_alloc!(
                        executor.fftn_into(&input, &opts, engine, Dest::overwrite(&mut full)),
                        "nd full r2c"
                    );
                    let mut spectrum = input.rfftn_with(&opts, engine).unwrap();
                    no_alloc!(
                        executor.rfftn_into(&input, &opts, engine, Dest::overwrite(&mut spectrum)),
                        "nd r2c"
                    );
                    let opts = opts.shape(Some(shape.clone()));
                    let mut restored =
                        Tensor::from_vec_col_major(shape, vec![<$real>::NAN; count]).unwrap();
                    no_alloc!(
                        executor.irfftn_into(
                            &spectrum,
                            &opts,
                            engine,
                            Dest::overwrite(&mut restored)
                        ),
                        "nd c2r"
                    );
                    let error = restored
                        .as_slice::<$real>()
                        .unwrap()
                        .iter()
                        .map(|v| (*v - 1.0).abs())
                        .fold(0.0 as $real, <$real>::max);
                    assert!(error < $tol, "nd real roundtrip max error {error}");
                }
            }
        }};
    }
    check_width!(f32, Complex32, 2e-5);
    check_width!(f64, Complex64, 1e-12);

    // A compact tensor's non-leading axis has strided lanes, but must not
    // canonicalize the entire tensor. A transposed input is the negative control
    // for the retained full-input gather boundary. Lanes here are only 64 values.
    let shape = [32, 64];
    let data = vec![Complex64::new(1.0, 0.0); 2048];
    let compact = TensorRead::from_view(TensorView::C64(
        TypedTensorView::from_slice(shape, [1, 32], 0, &data).unwrap(),
    ));
    let strided = TensorRead::from_view(TensorView::C64(
        TypedTensorView::from_slice(shape, [64, 1], 0, &data).unwrap(),
    ));
    let opts = FftOptions::default().axis(1);
    let mut out =
        Tensor::from_vec_col_major(shape, vec![Complex64::new(f64::NAN, f64::NAN); 2048]).unwrap();
    LARGE_THRESHOLD.store(data.len() * size_of::<Complex64>(), Ordering::Relaxed);
    let mut compact_engine = CpuEngine::serial();
    let (_, large) = measure(|| {
        compact
            .fft_into(&opts, &mut compact_engine, Dest::overwrite(&mut out))
            .unwrap()
    });
    assert_eq!(large, 0, "compact higher-axis FFT gathered a full tensor");
    let mut executor = FftExecutor::default();
    no_alloc!(
        executor.fft_into(
            &compact,
            &opts,
            &mut compact_engine,
            Dest::overwrite(&mut out)
        ),
        "compact borrowed higher-axis c2c"
    );
    let mut strided_engine = CpuEngine::serial();
    let (_, large) = measure(|| {
        strided
            .fft_into(&opts, &mut strided_engine, Dest::overwrite(&mut out))
            .unwrap()
    });
    assert!(
        large > 0,
        "negative control did not observe the documented full gather"
    );
    assert_eq!(
        out.as_slice::<Complex64>().unwrap()[0],
        Complex64::new(64.0, 0.0)
    );
    assert_eq!(compact.as_read().shape(), &shape);
}
