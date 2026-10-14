#![cfg(feature = "autodiff")]

use num_complex::{Complex32, Complex64};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;
use tenferro_fft::{EagerFftInPlaceError, EagerSessionFftExt, EagerTensorFftExt, FftNorm};
use tenferro_tensor::Tensor;

fn input(runtime: &std::sync::Arc<EagerRuntime>) -> EagerTensor {
    EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([2], vec![Complex64::new(1., 0.), Complex64::new(2., 0.)])
            .unwrap(),
        runtime.clone(),
    )
    .unwrap()
}
fn rejected(result: Result<EagerTensor, EagerFftInPlaceError>) -> EagerTensor {
    match result {
        Err(EagerFftInPlaceError::Rejected { input, .. }) => *input,
        other => panic!("expected unchanged rejected input, got {other:?}"),
    }
}

#[test]
fn consuming_fft_preserves_allocation_and_matches_borrowed_fft_on_every_axis() {
    let shape = [32, 32, 32];
    for threads in [1, 8] {
        let runtime =
            EagerRuntime::with_cpu_backend(CpuBackend::with_threads(threads).unwrap()).unwrap();
        for axis in 0..3 {
            for norm in [FftNorm::Backward, FftNorm::Forward, FftNorm::Ortho] {
                let source = Tensor::from_vec_col_major(
                    shape,
                    (0..32768)
                        .map(|i| Complex64::new((i % 13) as f64, (i % 11) as f64))
                        .collect(),
                )
                .unwrap();
                let x = EagerTensor::from_tensor_in(source, runtime.clone()).unwrap();
                let expected = runtime
                    .with_eager_session(|s| s.fft(&x, None, axis, norm))
                    .unwrap();
                let pointer = x.value().unwrap().as_slice::<Complex64>().unwrap().as_ptr();
                let y = x.fft_in_place(axis, norm).unwrap();
                assert_eq!(
                    y.value().unwrap().as_slice::<Complex64>().unwrap().as_ptr(),
                    pointer
                );
                assert_eq!(
                    y.value().unwrap().as_slice::<Complex64>().unwrap(),
                    expected.value().unwrap().as_slice::<Complex64>().unwrap()
                );
                let expected = runtime
                    .with_eager_session(|s| s.ifft(&y, None, axis, norm))
                    .unwrap();
                let z = y.ifft_in_place(axis, norm).unwrap();
                assert_eq!(
                    z.value().unwrap().as_slice::<Complex64>().unwrap().as_ptr(),
                    pointer
                );
                assert_eq!(
                    z.value().unwrap().as_slice::<Complex64>().unwrap(),
                    expected.value().unwrap().as_slice::<Complex64>().unwrap()
                );
            }
        }
    }
}

#[test]
fn consuming_fft_respects_compact_slice_extent_and_offset() {
    for start in [0, 1] {
        let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
        let source = EagerTensor::from_tensor_in(
            Tensor::from_vec_col_major(
                [4],
                (1..=4).map(|x| Complex64::new(x as f64, 0.)).collect(),
            )
            .unwrap(),
            runtime.clone(),
        )
        .unwrap();
        let slice = runtime
            .with_eager_session(|session| {
                session.slice(
                    &source,
                    tenferro_ad::SliceConfig {
                        starts: vec![start],
                        limits: vec![start + 2],
                        strides: vec![1],
                    },
                )
            })
            .unwrap();
        drop(source);
        let pointer = slice
            .tensor_read()
            .as_slice::<Complex64>()
            .unwrap()
            .as_ptr();
        let output = slice.fft_in_place(0, FftNorm::Backward).unwrap();
        assert_eq!(
            output
                .tensor_read()
                .as_slice::<Complex64>()
                .unwrap()
                .as_ptr(),
            pointer
        );
        assert_eq!(
            output.tensor_read().as_slice::<Complex64>().unwrap(),
            &[
                Complex64::new((2 * start + 3) as f64, 0.),
                Complex64::new(-1., 0.)
            ]
        );
        let root = output
            .into_value()
            .unwrap()
            .into_typed::<tenferro_tensor::Complex64>()
            .unwrap_or_else(|_| unreachable!());
        let mut expected: Vec<_> = (1..=4).map(|x| Complex64::new(x as f64, 0.)).collect();
        expected[start] = Complex64::new((2 * start + 3) as f64, 0.);
        expected[start + 1] = Complex64::new(-1., 0.);
        // The typed view retains the whole root, so outside-the-slice bytes stay
        // observable even though logical accessors export only the slice.
        assert_eq!(
            root.as_view().host_storage().unwrap(),
            expected,
            "outside the slice must remain unchanged"
        );
    }
}

#[test]
fn consuming_fft_rejects_noncompact_layout_before_ownership_extraction() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let source = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            [2, 3],
            (1..=6).map(|x| Complex64::new(x as f64, 0.)).collect(),
        )
        .unwrap(),
        runtime,
    )
    .unwrap();
    let view = source
        .runtime()
        .with_eager_session(|s| s.transpose(&source, &[1, 0]))
        .unwrap();
    drop(source);
    let view = match view.fft_in_place(0, FftNorm::Backward) {
        Err(EagerFftInPlaceError::Rejected { input, source }) => {
            assert!(source.to_string().contains("compact column-major"));
            *input
        }
        other => panic!("noncompact ownership extraction must fail: {other:?}"),
    };
    assert_eq!(
        view.to_tensor().unwrap().as_slice::<Complex64>().unwrap(),
        &[1., 3., 5., 2., 4., 6.].map(|x| Complex64::new(x, 0.))
    );
    assert_eq!(
        view.runtime()
            .with_eager_session(|s| s.fft(&view, None, 0, FftNorm::Backward))
            .unwrap()
            .shape(),
        &[3, 2]
    );
}

#[test]
fn consuming_fft_preserves_a_retained_reshape_source() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let source = input(&runtime);
    let reshaped = source
        .runtime()
        .with_eager_session(|s| s.reshape(&source, [1, 2]))
        .unwrap();
    // A reshape may share the physical owner or materialize independently.
    // Mutation must never change the retained source in either representation.
    match reshaped.fft_in_place(1, FftNorm::Backward) {
        Ok(result) => assert_eq!(
            result.value().unwrap().as_slice::<Complex64>().unwrap(),
            &[Complex64::new(3., 0.), Complex64::new(-1., 0.)]
        ),
        Err(EagerFftInPlaceError::Rejected { input, .. }) => assert_eq!(input.shape(), &[1, 2]),
        Err(error) => panic!("unexpected execution failure: {error}"),
    }
    assert_eq!(
        source.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(1., 0.), Complex64::new(2., 0.)]
    );
}

#[test]
fn consuming_fft_preserves_values_needed_by_later_backward() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let input = input(&runtime);
    let coefficient = runtime
        .with_eager_session(|s| s.fft(&input, None, 0, FftNorm::Backward))
        .unwrap();
    let variable = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major([2], vec![Complex64::new(1., 0.); 2]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let product = runtime
        .with_eager_session(|s| s.mul(&variable, &coefficient))
        .unwrap();
    // A separately saved value permits mutation; a retained physical owner
    // requires rejection. Either way backward must observe the original data.
    match coefficient.fft_in_place(0, FftNorm::Backward) {
        Ok(changed) => assert_eq!(
            changed.value().unwrap().as_slice::<Complex64>().unwrap(),
            &[Complex64::new(2., 0.), Complex64::new(4., 0.)]
        ),
        Err(EagerFftInPlaceError::Rejected { input, .. }) => {
            assert_eq!(
                input.value().unwrap().as_slice::<Complex64>().unwrap(),
                &[Complex64::new(3., 0.), Complex64::new(-1., 0.)]
            );
        }
        Err(error) => panic!("unexpected execution failure: {error}"),
    }
    let seed = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([2], vec![Complex64::new(1., 0.); 2]).unwrap(),
        runtime,
    )
    .unwrap();
    product.backward_with(&seed).unwrap();
    assert_eq!(
        variable
            .grad()
            .unwrap()
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        &[Complex64::new(3., 0.), Complex64::new(-1., 0.)]
    );
}

#[test]
fn consuming_fft_c32_and_rejected_alias_remain_usable() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([2], vec![Complex32::new(1., 0.), Complex32::new(2., 0.)])
            .unwrap(),
        runtime,
    )
    .unwrap();
    let alias = x.clone();
    let x = rejected(x.fft_in_place(0, FftNorm::Backward));
    assert_eq!(
        alias.value().unwrap().as_slice::<Complex32>().unwrap(),
        &[Complex32::new(1., 0.), Complex32::new(2., 0.)]
    );
    drop(alias);
    let x = x
        .fft_in_place(0, FftNorm::Backward)
        .unwrap()
        .ifft_in_place(0, FftNorm::Backward)
        .unwrap();
    assert_eq!(
        x.value().unwrap().as_slice::<Complex32>().unwrap(),
        &[Complex32::new(1., 0.), Complex32::new(2., 0.)]
    );
}

#[test]
fn consuming_fft_rejects_invalid_axis_dtype_tracking_and_capture_before_mutation() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let x = input(&runtime);
    let x = rejected(x.fft_in_place(2, FftNorm::Backward));
    let capture = runtime.capture_trace();
    let no_grad = runtime.no_grad();
    let x = rejected(x.fft_in_place(0, FftNorm::Backward));
    drop(no_grad);
    drop(capture);
    assert_eq!(
        x.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(1., 0.), Complex64::new(2., 0.)]
    );
    x.fft_in_place(0, FftNorm::Backward).unwrap();
    let real = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([1], vec![1.0_f64]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let real = rejected(real.fft_in_place(0, FftNorm::Backward));
    assert_eq!(real.value().unwrap().as_slice::<f64>().unwrap(), &[1.]);
    let tracked = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major([1], vec![Complex64::new(2., 0.)]).unwrap(),
        runtime.clone(),
    )
    .unwrap();
    let _no_grad = runtime.no_grad();
    let tracked = rejected(tracked.ifft_in_place(0, FftNorm::Backward));
    assert!(tracked.tracks_grad());
    assert_eq!(
        tracked.value().unwrap().as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2., 0.)]
    );
}

/// #1858 (B2): both error arms convert into `tenferro_ad::Error` so `?` works.
#[test]
fn in_place_errors_convert_into_the_eager_error_type() {
    let runtime = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap()).unwrap();
    let real = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major([1], vec![1.0_f64]).unwrap(),
        runtime,
    )
    .unwrap();
    let rejected = real.fft_in_place(0, FftNorm::Backward).unwrap_err();
    let converted = tenferro_ad::Error::from(rejected);
    assert!(converted.to_string().contains("C32 or C64"), "{converted}");

    let execution = EagerFftInPlaceError::Execution(tenferro_ad::Error::runtime_state(
        "fft_in_place",
        tenferro_ad::ErrorPhase::Execution,
        "intentional",
    ));
    let converted = tenferro_ad::Error::from(execution);
    assert!(converted.to_string().contains("intentional"), "{converted}");
}
