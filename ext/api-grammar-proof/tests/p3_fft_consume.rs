use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_api_grammar_proof::p3_fft_consume::{FftOptions, IfftOptions, SessionFftConsumeExt};
use tenferro_api_grammar_proof::policy::MaterializationPolicy::{Allow, Forbid};
use tenferro_cpu::CpuEngine;
use tenferro_fft::{EagerFftInPlaceError, EagerSessionFftExt, FftNorm};

fn data(n: usize) -> Vec<Complex64> {
    (0..n)
        .map(|i| Complex64::new(i as f64 + 1.0, 0.5 * (i as f64) - 1.0))
        .collect()
}

fn tensor(shape: Vec<usize>, v: Vec<Complex64>) -> Tensor {
    Tensor::from_vec_col_major(shape, v).unwrap()
}

fn close(a: &[Complex64], b: &[Complex64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).norm() < 1e-9, "{x} vs {y}");
    }
}

#[test]
fn matches_out_of_place_and_roundtrips() {
    let rt = EagerRuntime::new().unwrap();
    let mut engine = CpuEngine::serial();
    let v = data(8);
    let (inplace, oop, back) = rt
        .with_eager_session(&mut engine, |s| {
            let x = s.constant_from(tensor(vec![8], v.clone()))?;
            let oop = s.fft(&x, None, 0, FftNorm::Backward)?;
            drop(x);
            let x2 = s.constant_from(tensor(vec![8], v.clone()))?;
            let y = s
                .fft_in_place(
                    x2,
                    &FftOptions {
                        axis: 0,
                        norm: FftNorm::Backward,
                    },
                    Allow,
                )
                .map_err(tenferro_ad::Error::from)?;
            let y_copy = y.value()?.as_slice::<Complex64>()?.to_vec();
            let z = s
                .ifft_in_place(
                    y,
                    &IfftOptions {
                        axis: 0,
                        norm: FftNorm::Backward,
                    },
                    Allow,
                )
                .map_err(tenferro_ad::Error::from)?;
            Ok::<_, tenferro_ad::Error>((
                y_copy,
                oop.value()?.as_slice::<Complex64>()?.to_vec(),
                z.value()?.as_slice::<Complex64>()?.to_vec(),
            ))
        })
        .unwrap();
    close(&inplace, &oop);
    close(&back, &v);
}

#[test]
fn negative_axis_and_multidim() {
    let rt = EagerRuntime::new().unwrap();
    let mut engine = CpuEngine::serial();
    let v = data(12);
    let (a, b) = rt
        .with_eager_session(&mut engine, |s| {
            let x = s.constant_from(tensor(vec![3, 4], v.clone()))?;
            let oop = s.fft(&x, None, -1, FftNorm::Ortho)?;
            let x2 = s.constant_from(tensor(vec![3, 4], v.clone()))?;
            let y = s
                .fft_in_place(
                    x2,
                    &FftOptions {
                        axis: -1,
                        norm: FftNorm::Ortho,
                    },
                    Allow,
                )
                .map_err(tenferro_ad::Error::from)?;
            Ok::<_, tenferro_ad::Error>((
                y.value()?.as_slice::<Complex64>()?.to_vec(),
                oop.value()?.as_slice::<Complex64>()?.to_vec(),
            ))
        })
        .unwrap();
    close(&a, &b);
}

#[test]
fn shared_owner_rejected_and_returned_intact() {
    let rt = EagerRuntime::new().unwrap();
    let mut engine = CpuEngine::serial();
    let v = data(4);
    rt.with_eager_session(&mut engine, |s| {
        let x = s.constant_from(tensor(vec![4], v.clone()))?;
        let alias = x.clone();
        let err = s
            .fft_in_place(x, &FftOptions::default(), Allow)
            .unwrap_err();
        match err {
            EagerFftInPlaceError::Rejected { input, source } => {
                assert!(source.to_string().contains("retained"), "{source}");
                close(input.value()?.as_slice::<Complex64>()?, &v);
                close(alias.value()?.as_slice::<Complex64>()?, &v);
                // After the alias is dropped the recovered owner works.
                drop(alias);
                let out = s
                    .fft_in_place(*input, &FftOptions::default(), Allow)
                    .map_err(tenferro_ad::Error::from)?;
                assert_eq!(out.shape(), &[4]);
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
        Ok::<_, tenferro_ad::Error>(())
    })
    .unwrap();
}

#[test]
fn tracked_input_rejected() {
    let rt = EagerRuntime::new().unwrap();
    let mut engine = CpuEngine::serial();
    let v = data(4);
    let x = EagerTensor::requires_grad_in(tensor(vec![4], v.clone()), rt.clone()).unwrap();
    rt.with_eager_session(&mut engine, |s| {
        match s.fft_in_place(x, &FftOptions::default(), Allow) {
            Err(EagerFftInPlaceError::Rejected { input, source }) => {
                assert!(source.to_string().contains("untracked"), "{source}");
                assert!(input.tracks_grad());
                close(input.value()?.as_slice::<Complex64>()?, &v);
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
        Ok::<_, tenferro_ad::Error>(())
    })
    .unwrap();
}

#[test]
fn invalid_axis_and_real_dtype_return_owner() {
    let rt = EagerRuntime::new().unwrap();
    let mut engine = CpuEngine::serial();
    rt.with_eager_session(&mut engine, |s| {
        let x = s.constant_from(tensor(vec![4], data(4)))?;
        match s.fft_in_place(
            x,
            &FftOptions {
                axis: 5,
                norm: FftNorm::Backward,
            },
            Allow,
        ) {
            Err(EagerFftInPlaceError::Rejected { input, .. }) => assert_eq!(input.shape(), &[4]),
            other => panic!("expected Rejected, got {other:?}"),
        }
        let r = s.constant_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?)?;
        match s.ifft_in_place(r, &IfftOptions::default(), Allow) {
            Err(EagerFftInPlaceError::Rejected { input, .. }) => assert_eq!(input.shape(), &[2]),
            other => panic!("expected Rejected, got {other:?}"),
        }
        Ok::<_, tenferro_ad::Error>(())
    })
    .unwrap();
}

#[test]
fn forbid_returns_owner_unchanged() {
    let rt = EagerRuntime::new().unwrap();
    let mut engine = CpuEngine::serial();
    let v = data(4);
    rt.with_eager_session(&mut engine, |s| {
        let x = s.constant_from(tensor(vec![4], v.clone()))?;
        for inverse in [false, true] {
            let r = if inverse {
                s.ifft_in_place(x.clone(), &IfftOptions::default(), Forbid)
            } else {
                s.fft_in_place(x.clone(), &FftOptions::default(), Forbid)
            };
            match r {
                Err(EagerFftInPlaceError::Rejected { input, source }) => {
                    assert!(source.to_string().contains("not implemented"), "{source}");
                    close(input.value()?.as_slice::<Complex64>()?, &v);
                }
                other => panic!("expected Rejected, got {other:?}"),
            }
        }
        Ok::<_, tenferro_ad::Error>(())
    })
    .unwrap();
}

#[test]
fn error_converts_with_question_mark() {
    let rt = EagerRuntime::new().unwrap();
    let mut engine = CpuEngine::serial();
    let r: Result<(), tenferro_ad::Error> = rt.with_eager_session(&mut engine, |s| {
        let x = s.constant_from(tensor(vec![4], data(4)))?;
        s.fft_in_place(x, &FftOptions::default(), Forbid)?;
        Ok(())
    });
    assert!(r.unwrap_err().to_string().contains("not implemented"));
}
