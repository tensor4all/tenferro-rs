//! The guards of the generated FFT grammar (#2552 §4.4), the exception
//! entries E01/E02 and the generated documentation check (§4.5).

use num_complex::Complex64;
use tenferro_cpu::{BufferPool, CpuEngine};
use tenferro_tensor::{
    AsView, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorScalar, TypedTensor,
    TypedTensorView,
};

use super::cases;
use super::support::*;
use crate::api::registry::{api_docs_sync, API_EXCEPTIONS, API_LAYERS, API_OPS};
use crate::{FftExecutor, FftNorm, FftOps, FftOptions, FftnOptions};

// ------------------------------------------------------------ 1. forwarding

#[test]
fn every_registry_entry_has_a_forwarding_case() {
    let cases = cases::all();
    for op in API_OPS {
        for layer in op.layers {
            if *layer == "eager" && cfg!(not(feature = "autodiff")) {
                continue;
            }
            assert!(
                cases.iter().any(|c| c.op == op.name && c.layer == *layer),
                "no forwarding case for {layer}.{}",
                op.name
            );
        }
    }
    assert_eq!(API_OPS.len(), 8);
    assert!(API_LAYERS.iter().any(|(name, _)| *name == "concrete"));
}

#[test]
fn op_and_op_with_default_options_agree_bit_for_bit() {
    let mut failures = Vec::new();
    for case in cases::all() {
        match (case.run)() {
            Ok((plain, with)) if plain == with => {}
            Ok(_) => failures.push(format!(
                "{}.{} ({}): bits differ",
                case.layer, case.op, case.input
            )),
            Err(e) => failures.push(format!("{}.{} ({}): {e}", case.layer, case.op, case.input)),
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn no_operation_declares_into() {
    assert!(API_OPS.iter().all(|op| op.into.is_empty()));
}

// ---------------------------------------------------------- 2. input shapes

/// The logical elements of `view` in a fresh compact tensor.
fn compact_copy<T: TensorScalar + Copy>(view: &TypedTensorView<'_, T>) -> TestResult<Tensor> {
    let (m, n) = (view.shape()[0], view.shape()[1]);
    let mut data = Vec::with_capacity(m * n);
    for col in 0..n {
        for row in 0..m {
            data.push(*view.get(&[row, col]).ok_or("index out of range")?);
        }
    }
    Ok(Tensor::from_typed::<T>(TypedTensor::from_vec_col_major(
        vec![m, n],
        data,
    )?))
}

/// Owned tensor, compact borrowed read, transposed view, negative-stride view,
/// typed tensor and typed view all return the owned result on the same logical
/// matrix, bit for bit (strided and negative-stride views are gathered once,
/// then run the same kernel).
#[test]
fn every_input_kind_returns_the_owned_result() -> TestResult<()> {
    let base = TypedTensor::<f64>::from_vec_col_major(
        vec![3, 4],
        (0..12).map(|i| f64::from(i) * 0.75 - 2.0).collect(),
    )?;
    let owned = Tensor::from_typed::<f64>(base.duplicate()?);
    let mut e = CpuEngine::serial();
    let check = |label: &str,
                 logical: &Tensor,
                 got1: Tensor,
                 gotn: Tensor,
                 e: &mut CpuEngine|
     -> TestResult<()> {
        assert_eq!(
            tensor_bits(&got1)?,
            tensor_bits(&logical.fft(e)?)?,
            "fft of the {label} input differs"
        );
        assert_eq!(
            tensor_bits(&gotn)?,
            tensor_bits(&logical.fftn(e)?)?,
            "fftn of the {label} input differs"
        );
        Ok(())
    };

    check(
        "owned",
        &owned,
        owned.fft(&mut e)?,
        owned.fftn(&mut e)?,
        &mut e,
    )?;
    let read = owned.as_read();
    check(
        "compact read",
        &owned,
        read.fft(&mut e)?,
        read.fftn(&mut e)?,
        &mut e,
    )?;

    let compact_view = base.as_view();
    let r = TensorRead::from_view(f64::tensor_view(compact_view.clone()));
    check(
        "compact view",
        &owned,
        r.fft(&mut e)?,
        r.fftn(&mut e)?,
        &mut e,
    )?;

    let views = [
        ("transposed", base.as_view().transpose_view([1, 0])?),
        (
            "negative-stride",
            base.as_view()
                .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])?,
        ),
    ];
    for (label, view) in views {
        let logical = compact_copy(&view)?;
        let r = TensorRead::from_view(f64::tensor_view(view.clone()));
        check(label, &logical, r.fft(&mut e)?, r.fftn(&mut e)?, &mut e)?;
        // The typed view returns typed (complex) results with the same bits.
        let t1 = Tensor::from_typed::<Complex64>(view.fft(&mut e)?);
        let tn = Tensor::from_typed::<Complex64>(view.fftn(&mut e)?);
        check(&format!("typed {label}"), &logical, t1, tn, &mut e)?;
    }

    let t1 = Tensor::from_typed::<Complex64>(base.fft(&mut e)?);
    let tn = Tensor::from_typed::<Complex64>(base.fftn(&mut e)?);
    check("typed", &owned, t1, tn, &mut e)?;
    // irfft of a typed complex input is typed real.
    let spectrum = base.rfft(&mut e)?;
    let back: TypedTensor<f64> = spectrum.irfft_with(&FftOptions::default().n(Some(4)), &mut e)?;
    assert_eq!(back.shape(), &[3, 4]);
    Ok(())
}

#[test]
fn fftn_default_axes_follow_numpy() -> TestResult<()> {
    let mut e = CpuEngine::serial();
    let x = real()?;
    let all = x.fftn(&mut e)?;
    let explicit = x.fftn_with(&FftnOptions::default().axes(vec![0, 1]), &mut e)?;
    assert_eq!(tensor_bits(&all)?, tensor_bits(&explicit)?);
    // With `shape` and no `axes`: the last `shape.len()` axes.
    let last = x.fftn_with(&FftnOptions::default().shape(Some(vec![4])), &mut e)?;
    let explicit = x.fftn_with(
        &FftnOptions::default().shape(Some(vec![4])).axes(vec![1]),
        &mut e,
    )?;
    assert_eq!(tensor_bits(&last)?, tensor_bits(&explicit)?);
    // One axis through fftn is the one-axis transform.
    let one = x.fftn_with(&FftnOptions::default().axes(vec![-1]), &mut e)?;
    assert_eq!(tensor_bits(&one)?, tensor_bits(&x.fft(&mut e)?)?);
    Ok(())
}

// --------------------------------------------- 4. declared omissions (lanes)

#[test]
fn an_fft_on_a_lane_engine_is_unsupported() -> TestResult<()> {
    let mut engine = CpuEngine::serial();
    let mut pools = [BufferPool::new()];
    let x = complex()?;
    let results = engine.with_lanes(&[1], &mut pools, |_, lane| {
        [x.fft(lane).map(|_| ()), x.fftn(lane).map(|_| ())]
    })?;
    for result in results.into_iter().flatten() {
        let err = result.expect_err("an FFT in a lane");
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
    }
    Ok(())
}

// --------------------------------------------------------- E02 executor

#[test]
fn executor_with_forms_equal_fft_ops_and_fill_the_plan_cache() -> TestResult<()> {
    let mut e = CpuEngine::serial();
    let mut ex = FftExecutor::default();
    let (r, c) = (real()?, complex()?);
    let one = FftOptions::default().norm(FftNorm::Ortho);
    let nd = FftnOptions::default().norm(FftNorm::Forward);
    let pairs = [
        (ex.fft_with(&r, &one, &mut e)?, r.fft_with(&one, &mut e)?),
        (ex.ifft_with(&c, &one, &mut e)?, c.ifft_with(&one, &mut e)?),
        (ex.rfft_with(&r, &one, &mut e)?, r.rfft_with(&one, &mut e)?),
        (
            ex.irfft_with(&c, &one, &mut e)?,
            c.irfft_with(&one, &mut e)?,
        ),
        (ex.fftn_with(&r, &nd, &mut e)?, r.fftn_with(&nd, &mut e)?),
        (ex.ifftn_with(&c, &nd, &mut e)?, c.ifftn_with(&nd, &mut e)?),
        (ex.rfftn_with(&r, &nd, &mut e)?, r.rfftn_with(&nd, &mut e)?),
        (
            ex.irfftn_with(&c, &nd, &mut e)?,
            c.irfftn_with(&nd, &mut e)?,
        ),
    ];
    for (i, (a, b)) in pairs.iter().enumerate() {
        assert_eq!(tensor_bits(a)?, tensor_bits(b)?, "executor case {i}");
    }
    assert!(ex.cache_stats().entries > 0);
    assert!(API_EXCEPTIONS.iter().any(|x| x.id == "E02"));
    Ok(())
}

// --------------------------------------------------------- E01 in place

#[cfg(feature = "autodiff")]
mod in_place {
    use super::*;
    use crate::{EagerFftInPlaceError, EagerFftOps};
    use tenferro_ad::{EagerRuntime, EagerTensor};

    #[test]
    fn a_length_change_is_rejected_and_returns_the_input_unchanged() -> TestResult<()> {
        let runtime = EagerRuntime::new()?;
        let x = runtime.constant_from(complex()?)?;
        let before = tensor_bits(&x.to_tensor(&mut CpuEngine::serial())?)?;
        let outcome = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
            Ok::<_, tenferro_ad::Error>(s.ifft_in_place(x, &FftOptions::default().n(Some(8))))
        })?;
        match outcome {
            Err(EagerFftInPlaceError::Rejected { input, source }) => {
                assert!(
                    matches!(
                        source.kind(),
                        ErrorKind::Validation(tenferro_tensor::ValidationKind::InvalidArgument)
                    ),
                    "{source}"
                );
                assert!(source.to_string().contains("ifft_in_place"), "{source}");
                assert_eq!(
                    tensor_bits(&input.to_tensor(&mut CpuEngine::serial())?)?,
                    before
                );
            }
            other => panic!("n = Some(8) must be rejected: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn a_real_input_is_rejected_with_the_ifft_label() -> TestResult<()> {
        let runtime = EagerRuntime::new()?;
        let x = runtime.constant_from(real()?)?;
        let outcome = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
            Ok::<_, tenferro_ad::Error>(s.ifft_in_place(x, &FftOptions::default()))
        })?;
        match outcome {
            Err(EagerFftInPlaceError::Rejected { input, source }) => {
                assert!(source.to_string().contains("ifft_in_place"), "{source}");
                assert_eq!(input.shape(), &[3, 4]);
            }
            other => panic!("a real input must be rejected: {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn question_mark_drops_a_rejected_input_and_keeps_the_error() -> TestResult<()> {
        fn transform(
            s: &mut tenferro_ad::EagerSession<'_>,
            x: EagerTensor,
        ) -> tenferro_ad::Result<EagerTensor> {
            Ok(s.fft_in_place(x, &FftOptions::default())?)
        }
        let runtime = EagerRuntime::new()?;
        let x = runtime.constant_from(real()?)?;
        let err = runtime
            .with_eager_session(&mut CpuEngine::serial(), |s| transform(s, x))
            .expect_err("real input");
        assert!(err.to_string().contains("C32 or C64"), "{err}");
        Ok(())
    }

    #[test]
    fn in_place_rejects_a_tensor_from_another_runtime_and_returns_it() -> TestResult<()> {
        let mut e = CpuEngine::serial();
        let (a, b) = (EagerRuntime::new()?, EagerRuntime::new()?);
        let x = b.constant_from(complex()?)?;
        let before = tensor_bits(&x.to_tensor(&mut e)?)?;
        let outcome = a.with_eager_session(&mut e, |s| {
            let out_of_place = s.fft(&x).map(|_| ()).expect_err("foreign tensor");
            let rejected = s
                .fft_in_place(x, &FftOptions::default())
                .expect_err("foreign tensor");
            Ok::<_, tenferro_ad::Error>((out_of_place, rejected))
        })?;
        let (out_of_place, rejected) = outcome;
        assert!(
            matches!(out_of_place, tenferro_ad::Error::ContextMismatch { .. }),
            "{out_of_place}"
        );
        match rejected {
            EagerFftInPlaceError::Rejected { input, source } => {
                assert!(
                    matches!(source, tenferro_ad::Error::ContextMismatch { .. }),
                    "{source}"
                );
                assert!(input.same_context(&b.constant_from(complex()?)?));
                assert_eq!(tensor_bits(&input.to_tensor(&mut e)?)?, before);
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn an_accepted_in_place_transform_equals_the_out_of_place_one() -> TestResult<()> {
        let mut e = CpuEngine::serial();
        let want =
            complex()?.fft_with(&FftOptions::default().axis(0).norm(FftNorm::Ortho), &mut e)?;
        let runtime = EagerRuntime::new()?;
        let x = runtime.constant_from(complex()?)?;
        let y = runtime.with_eager_session(&mut e, |s| {
            Ok::<_, tenferro_ad::Error>(
                s.fft_in_place(x, &FftOptions::default().axis(0).norm(FftNorm::Ortho))?,
            )
        })?;
        assert_eq!(tensor_bits(&y.to_tensor(&mut e)?)?, tensor_bits(&want)?);
        assert!(API_EXCEPTIONS.iter().any(|x| x.id == "E01"));
        Ok(())
    }
}

// --------------------------------------------------------- generated docs

#[test]
fn api_docs_are_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/spec/generated/api-ops.md"
    );
    let current = std::fs::read_to_string(path).expect("read docs/spec/generated/api-ops.md");
    let expected = api_docs_sync(&current);
    if std::env::var_os("UPDATE_API_DOCS").is_some() {
        std::fs::write(path, &expected).expect("write docs/spec/generated/api-ops.md");
        return;
    }
    assert!(
        current == expected,
        "docs/spec/generated/api-ops.md is stale; run \
         `UPDATE_API_DOCS=1 cargo test -p tenferro-fft --features autodiff api_docs_are_current`"
    );
}

// ------------------------------------------------ options only where they mean something

/// #2552 Q14: an operation has an options type and an `_with` form only when
/// it has semantic options. The set of options-bearing operations is fixed, every
/// options type has at least one field, and the generated method names of each
/// operation are exactly `op`, `op_with` (iff options) and `op_into` (iff `into`).
#[test]
fn options_exist_exactly_for_the_operations_that_declare_them() {
    use crate::api::registry::api_ops_markdown;

    let mut with_options: Vec<&str> = API_OPS
        .iter()
        .filter(|op| op.options != "none")
        .map(|op| op.name)
        .collect();
    with_options.sort_unstable();
    assert_eq!(
        with_options,
        ["fft", "fftn", "ifft", "ifftn", "irfft", "irfftn", "rfft", "rfftn"]
    );

    // Every declared options type is non-empty: a derived `Debug` prints
    // `Name { field: .. }` for a struct with fields and only `Name` for an
    // empty one.
    let defaults: &[(&str, String)] = &[
        ("FftOptions", format!("{:?}", crate::FftOptions::default())),
        (
            "FftnOptions",
            format!("{:?}", crate::FftnOptions::default()),
        ),
    ];
    for op in API_OPS.iter().filter(|op| op.options != "none") {
        let name = op.options.rsplit("::").next().unwrap_or(op.options);
        let (_, shown) = defaults
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("no default for options type {name} of {}", op.name));
        assert!(
            shown.contains(" { "),
            "{name} of {} has no field: {shown}",
            op.name
        );
    }

    let md = api_ops_markdown();
    for op in API_OPS {
        let mut want = vec![format!("`{}`", op.name)];
        if op.options != "none" {
            want.push(format!("`{}_with`", op.name));
        }
        if !op.into.is_empty() {
            want.push(format!("`{}_into` ({})", op.name, op.into.join(", ")));
        }
        let row = md
            .lines()
            .find(|line| line.starts_with(&format!("| `{}` |", op.name)))
            .unwrap_or_else(|| panic!("no table row for {}", op.name));
        let cell = row.trim_end_matches(" |").rsplit("| ").next().unwrap_or("");
        assert_eq!(cell, want.join(", "), "{}", op.name);
    }
}
