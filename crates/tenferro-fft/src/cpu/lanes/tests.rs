use super::*;
use crate::FftPlanCache;
use num_complex::Complex64;

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
                    &mut output,
                    1,
                    std::convert::identity,
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
                &mut output,
                8,
                |value| value,
            )
        };
        assert_eq!(
            result.is_ok(),
            valid,
            "shape={shape:?} input={input_len} output={output_len}"
        );
    }
}
