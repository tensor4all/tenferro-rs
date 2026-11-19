use super::*;
use crate::FftPlanCache;
use num_complex::Complex64;
use tenferro_cpu::linalg_interop::BufferPool;

#[test]
fn impulse_reference_covers_borrowed_and_consuming_lane_modes() {
    let impulse = [
        Complex64::new(1., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
        Complex64::new(0., 0.),
    ];
    let real = [1., 0., 0., 0.];
    let mut plans = FftPlanCache::default();
    let mut buffers = BufferPool::new();
    for mode in 0..4 {
        for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
            let (input, shape, operation) = match mode {
                0 => (
                    Some(Input::Complex(&impulse[..])),
                    [4],
                    FftOperation::C2cForward,
                ),
                1 => (Some(Input::Real(&real[..])), [4], FftOperation::R2cFull),
                2 => (None, [4], FftOperation::C2cForward),
                _ => (Some(Input::Complex(&impulse[..3])), [3], FftOperation::C2r),
            };
            let mut output = impulse.map(MaybeUninit::new);
            // SAFETY: borrowed modes use disjoint arrays. Absent input uses
            // initialized Complex64 storage, shape-preserving c2c, identity
            // projection, and no other reference to the output allocation.
            unsafe {
                execute(
                    input,
                    &shape,
                    0,
                    4,
                    4,
                    operation,
                    norm,
                    &mut plans,
                    &mut buffers,
                    &mut output,
                    std::convert::identity,
                    Exec::serial(),
                )
                .unwrap();
            }
            // DFT of a unit impulse is all ones. The inverse DFT of a unit DC
            // coefficient is also constant, with the inverse normalization.
            let expected = match norm {
                FftNorm::Ortho => 0.5,
                FftNorm::Backward if mode == 3 => 0.25,
                FftNorm::Forward if mode != 3 => 0.25,
                _ => 1.0,
            };
            for value in output {
                // SAFETY: initialized before execution; successful execution
                // preserves initialization and overwrites the complete output.
                let value = unsafe { value.assume_init() };
                assert_eq!(
                    value,
                    Complex64::new(expected, 0.),
                    "mode={mode} norm={norm:?}"
                );
            }
        }
    }
}

#[test]
fn checked_lane_boundary_rejects_mismatched_spans_and_overflow_before_writes() {
    // A real pool: eight lanes fork over an eight-slot token.
    let pool = tenferro_cpu::Pool::builder().threads(8).build().unwrap();
    for (shape, input_len, output_len, valid) in [
        ([1, 2], 2, 2, true),
        ([1, 2], 1, 2, false),
        ([1, 2], 2, 1, false),
        ([usize::MAX, 2], 0, 0, false),
        ([0, 2], 0, 0, true),
    ] {
        let input = vec![Complex64::new(1., 0.); input_len];
        let mut output = vec![MaybeUninit::uninit(); output_len];
        let mut plans = FftPlanCache::default();
        let mut buffers = BufferPool::new();
        // SAFETY: Some(input) selects the disjoint borrowed-input path. The
        // kernel must reject invalid spans before any unchecked output access.
        let result = unsafe {
            execute(
                Some(Input::Complex(&input)),
                &shape,
                1,
                2,
                2,
                FftOperation::C2cForward,
                FftNorm::Backward,
                &mut plans,
                &mut buffers,
                &mut output,
                |value| value,
                Exec::new(&pool, 8).unwrap(),
            )
        };
        assert_eq!(
            result.is_ok(),
            valid,
            "shape={shape:?} input={input_len} output={output_len}"
        );
    }
}

#[test]
fn complex_lane_inverse_rejects_an_incomplete_initialized_spectrum() {
    let input = [Complex64::new(1., 0.)];
    let sentinel = Complex64::new(17., -2.);
    let mut output = [MaybeUninit::new(sentinel); 4];
    let mut plans = FftPlanCache::default();
    let mut buffers = BufferPool::new();
    // SAFETY: borrowed input/output are disjoint; malformed spectrum metadata
    // must be refused before any uninitialized mirror slot is inspected.
    let error = unsafe {
        execute(
            Some(Input::Complex(&input)),
            &[1],
            0,
            4,
            4,
            FftOperation::C2r,
            FftNorm::Backward,
            &mut plans,
            &mut buffers,
            &mut output,
            std::convert::identity,
            Exec::serial(),
        )
    }
    .unwrap_err();
    assert!(error.to_string().contains("spectrum"));
    for value in output {
        // SAFETY: destination started initialized, and validation precedes writes.
        assert_eq!(unsafe { value.assume_init() }, sentinel);
    }
}

#[test]
fn complex_lane_invalid_transform_lengths_preserve_destination() {
    let input = [Complex64::new(1., 0.); 4];
    let sentinel = Complex64::new(17., -2.);
    for (fft_len, out_axis_len) in [(0, 4), (4, 5)] {
        let mut output = [MaybeUninit::new(sentinel); 5];
        let mut plans = FftPlanCache::default();
        let mut buffers = BufferPool::new();
        // SAFETY: input/output are disjoint initialized spans. Invalid transform
        // lengths must be refused before creating scratch or writing output.
        let error = unsafe {
            execute(
                Some(Input::Complex(&input)),
                &[4],
                0,
                fft_len,
                out_axis_len,
                FftOperation::C2cForward,
                FftNorm::Backward,
                &mut plans,
                &mut buffers,
                &mut output[..out_axis_len],
                std::convert::identity,
                Exec::serial(),
            )
        }
        .unwrap_err();
        assert!(error.to_string().contains("transform lengths"));
        assert_eq!(plans.stats().entries, 0);
        assert!(buffers.is_empty());
        for value in output {
            // SAFETY: initialized before the call; refusal must precede writes.
            assert_eq!(unsafe { value.assume_init() }, sentinel);
        }
    }
}

#[test]
fn real_lane_span_validation_preserves_destination_and_execution_resources() {
    let real = [1.; 8];
    let spectrum = [Complex64::new(1., 0.); 6];
    let sentinel = Complex64::new(17., -2.);
    for (operation, input_len, output_len) in [
        (FftOperation::R2cOnesided, 7, 6),
        (FftOperation::R2cOnesided, 8, 5),
        (FftOperation::C2r, 5, 8),
        (FftOperation::C2r, 6, 7),
    ] {
        let (input, shape, out_axis_len) = if operation == FftOperation::C2r {
            (Input::Complex(&spectrum[..input_len]), [2, 3], 4)
        } else {
            (Input::Real(&real[..input_len]), [2, 4], 3)
        };
        let mut output = [MaybeUninit::new(sentinel); 8];
        let mut plans = FftPlanCache::default();
        let mut buffers = BufferPool::new();
        // SAFETY: initialized input/output storage is disjoint. The checked
        // span boundary must reject these mismatches before unchecked writes.
        let error = unsafe {
            execute_real(
                Some(input),
                &shape,
                1,
                4,
                out_axis_len,
                operation,
                FftNorm::Backward,
                &mut plans,
                &mut buffers,
                &mut output[..output_len],
                std::convert::identity,
                Exec::serial(),
            )
        }
        .unwrap_err();
        assert!(error.to_string().contains("buffer length"));
        assert_eq!(plans.stats().entries, 0);
        assert!(buffers.is_empty());
        for value in output {
            // SAFETY: initialized before the call; refusal must precede writes.
            assert_eq!(unsafe { value.assume_init() }, sentinel);
        }
    }
}

#[test]
fn empty_real_batches_do_not_touch_storage_or_create_plans() {
    let sentinel = Complex64::new(17., -2.);
    for operation in [
        FftOperation::R2cFull,
        FftOperation::R2cOnesided,
        FftOperation::C2r,
    ] {
        let (input, shape, out_axis_len) = match operation {
            FftOperation::C2r => (Input::Complex(&[]), [0, 3], 4),
            FftOperation::R2cOnesided => (Input::Real(&[]), [0, 4], 3),
            _ => (Input::Real(&[]), [0, 4], 4),
        };
        let mut output = [MaybeUninit::new(sentinel); 2];
        let mut plans = FftPlanCache::default();
        let mut buffers = BufferPool::new();
        // SAFETY: both logical spans are empty and match the zero-batch layout;
        // the nonempty initialized backing array guards the zero-write boundary.
        unsafe {
            execute_real(
                Some(input),
                &shape,
                1,
                4,
                out_axis_len,
                operation,
                FftNorm::Backward,
                &mut plans,
                &mut buffers,
                &mut output[1..1],
                std::convert::identity,
                Exec::serial(),
            )
        }
        .unwrap();
        assert_eq!(plans.stats().entries, 0);
        assert!(buffers.is_empty());
        for value in output {
            // SAFETY: initialized backing storage must survive an empty transform.
            assert_eq!(unsafe { value.assume_init() }, sentinel);
        }
    }
}

// Test-only provider fault: all operand/output spans remain valid, but realfft
// receives a plan for a different length and rejects it before payload writes.
#[derive(Default)]
struct WrongLengthRealPlans(FftPlanCache);

impl FftPlanProvider for WrongLengthRealPlans {
    fn plan_f32(&mut self, len: usize, forward: bool) -> std::sync::Arc<dyn rustfft::Fft<f32>> {
        self.0.plan_f32(len, forward)
    }

    fn plan_f64(&mut self, len: usize, forward: bool) -> std::sync::Arc<dyn rustfft::Fft<f64>> {
        self.0.plan_f64(len, forward)
    }

    fn real_plan_f32(&mut self, len: usize, forward: bool) -> RealPlan<f32> {
        self.0.real_plan_f32(len + 2, forward)
    }

    fn real_plan_f64(&mut self, len: usize, forward: bool) -> RealPlan<f64> {
        self.0.real_plan_f64(len + 2, forward)
    }
}

#[test]
fn real_plan_failure_recycles_both_slabs_after_serial_and_parallel_jobs() {
    let pool = tenferro_cpu::Pool::builder().threads(4).build().unwrap();
    for budget in [1, 4] {
        for operation in [FftOperation::R2cOnesided, FftOperation::C2r] {
            let fft_len = if budget == 1 { 8 } else { 8192 };
            let spectrum_len = fft_len / 2 + 1;
            let lanes = 9;
            let (input_axis_len, out_axis_len) = if operation == FftOperation::C2r {
                (spectrum_len, fft_len)
            } else {
                (fft_len, spectrum_len)
            };
            let real = vec![1.; lanes * input_axis_len];
            let spectrum = vec![Complex64::new(1., 0.); lanes * input_axis_len];
            let input = if operation == FftOperation::C2r {
                Input::Complex(&spectrum)
            } else {
                Input::Real(&real)
            };
            let mut plans = WrongLengthRealPlans::default();
            let scratch_len = match plans.real_plan_f64(fft_len, operation.is_forward()) {
                RealPlan::Forward(plan) => plan.get_scratch_len(),
                RealPlan::Inverse(plan) => plan.get_scratch_len(),
            };
            let mut buffers = BufferPool::new();
            let sentinel = Complex64::new(17., -2.);
            let mut output = vec![MaybeUninit::new(sentinel); lanes * out_axis_len];
            let exec = if budget == 1 {
                Exec::serial()
            } else {
                assert!(output.len() >= MIN_PARALLEL_ELEMENTS);
                Exec::new(&pool, budget).unwrap()
            };
            assert_eq!(exec.budget().get(), budget);
            // SAFETY: disjoint initialized spans match the lane coverage. Only
            // the provider is faulty; realfft checks its input length before
            // accessing payloads. This is not a user-error or rollback contract.
            let error = unsafe {
                execute_real(
                    Some(input),
                    &[lanes, input_axis_len],
                    1,
                    fft_len,
                    out_axis_len,
                    operation,
                    FftNorm::Backward,
                    &mut plans,
                    &mut buffers,
                    &mut output,
                    std::convert::identity,
                    exec,
                )
            }
            .unwrap_err();
            assert_eq!(error.kind(), tenferro_tensor::ErrorKind::RuntimeState);
            let diagnostic = error.to_string();
            let context = if operation == FftOperation::C2r {
                "complex-to-real plan rejected the lane buffers"
            } else {
                "real-to-complex plan rejected the lane buffers"
            };
            assert!(diagnostic.contains(context), "{diagnostic}");
            assert!(diagnostic.contains("Wrong length of input"), "{diagnostic}");
            for value in output {
                // SAFETY: initialized sentinel storage was never scattered into.
                assert_eq!(unsafe { value.assume_init() }, sentinel);
            }
            // Nine lanes over four jobs leave one empty job; all jobs must join
            // before both checked-out slabs are returned on the error path.
            let jobs = budget;
            assert_eq!(buffers.len(), 2);
            let recycled = buffers.stats();
            let real_slab = buffers.acquire_with_capacity::<f64>(jobs * fft_len);
            assert_eq!(buffers.len(), 1);
            assert_eq!(real_slab.len(), 0);
            assert!(real_slab.capacity() >= jobs * fft_len);
            let complex_slab =
                buffers.acquire_with_capacity::<Complex64>(jobs * (spectrum_len + scratch_len));
            assert_eq!(buffers.len(), 0);
            assert_eq!(complex_slab.len(), 0);
            assert!(complex_slab.capacity() >= jobs * (spectrum_len + scratch_len));
            f64::pool_release(&mut buffers, real_slab);
            Complex64::pool_release(&mut buffers, complex_slab);
            assert_eq!(buffers.stats(), recycled);
        }
    }
}
