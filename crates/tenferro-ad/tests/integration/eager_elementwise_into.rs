//! The eager elementwise output-reuse forms `EagerSession::{add,sub,mul,div,neg,conj}_into`
//! (#2701).
//!
//! Pins the `_into` contract on the eager layer: the destination is overwritten
//! without being read, every rejection (undeclared `Dest` mode, tracked operand,
//! active capture, foreign runtime, dtype or shape mismatch) happens before the
//! destination is written, and view-backed eager operands are read in place.

use tenferro_ad::{EagerRuntime, EagerTensor, Error};
use tenferro_cpu::CpuEngine;
use tenferro_tensor::ErrorKind;
use tenferro_tensor::{Dest, Tensor};

fn vector(values: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values.to_vec()).expect("host vector")
}

fn constant(ctx: &std::sync::Arc<EagerRuntime>, tensor: Tensor) -> EagerTensor {
    ctx.constant_from(tensor).expect("constant leaf")
}

/// A destination whose contents are a recognizable sentinel; a rejected call
/// must leave every element bit for bit unchanged.
const SENTINEL: [f64; 3] = [7.5, -0.0, 9.25];

fn assert_untouched(out: &Tensor) {
    let got = out.as_slice::<f64>().expect("host destination");
    for (got, want) in got.iter().zip(SENTINEL) {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "a rejected _into wrote {got}"
        );
    }
}

#[test]
fn add_into_overwrites_without_reading_the_destination() {
    let ctx = EagerRuntime::new().expect("runtime");
    let x = constant(&ctx, vector(&[1.0, 2.0, 3.0]));
    let y = constant(&ctx, vector(&[0.5, -4.0, 1e300]));
    let mut out = vector(&[f64::NAN, f64::INFINITY, -0.0]);
    let fresh = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.add_into(&x, &y, Dest::overwrite(&mut out))?;
            s.add(&x, &y)
        })
        .expect("add_into and add");
    let fresh = fresh.into_value().expect("unique result");
    let got = out.as_slice::<f64>().expect("host destination");
    let want = fresh.as_slice::<f64>().expect("host result");
    assert_eq!(
        got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "add_into must equal add bit for bit and ignore NaN/inf/-0 in the destination"
    );
}

#[test]
fn add_into_reads_view_backed_and_broadcast_operands() {
    let ctx = EagerRuntime::new().expect("runtime");
    // a is 2x3 column-major; its eager transpose is a 3x2 view, not a copy.
    let a = constant(
        &ctx,
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("matrix"),
    );
    let scalar = constant(
        &ctx,
        Tensor::from_vec_col_major(vec![], vec![100.0_f64]).expect("scalar"),
    );
    let mut out = Tensor::from_vec_col_major(vec![3, 2], vec![f64::NAN; 6]).expect("output");
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        let at = s.transpose(&a, &[1, 0])?;
        assert!(
            matches!(at.tensor_read(), tenferro_tensor::TensorRead::View(_)),
            "the eager transpose must stay a view for this test to cover a strided operand"
        );
        s.add_into(&at, &scalar, Dest::overwrite(&mut out))
    })
    .expect("strided add_into");
    // out[i, j] = a[j, i] + 100, column-major over [3, 2].
    assert_eq!(
        out.as_slice::<f64>().expect("host destination"),
        &[101.0, 103.0, 105.0, 102.0, 104.0, 106.0]
    );
}

#[test]
fn add_into_rejects_undeclared_dest_modes_before_writing() {
    let ctx = EagerRuntime::new().expect("runtime");
    let x = constant(&ctx, vector(&[1.0, 2.0, 3.0]));
    for mode in ["accumulate", "scaled"] {
        let mut out = vector(&SENTINEL);
        let err = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                let dest = match mode {
                    "accumulate" => Dest::accumulate(&mut out),
                    _ => Dest::scaled(&mut out, 2.0_f64, 0.0_f64),
                };
                s.add_into(&x, &x, dest)
            })
            .expect_err("only Dest::Overwrite is declared for add_into");
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{mode}: {err}");
        assert_untouched(&out);
    }
}

#[test]
fn add_into_rejects_tracked_operands_and_capture_before_writing() {
    let ctx = EagerRuntime::new().expect("runtime");
    let x = ctx
        .variable_from(vector(&[1.0, 2.0, 3.0]))
        .expect("variable");
    let c = constant(&ctx, vector(&[1.0, 1.0, 1.0]));

    for (lhs, rhs) in [(&x, &c), (&c, &x)] {
        let mut out = vector(&SENTINEL);
        let err = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                s.add_into(lhs, rhs, Dest::overwrite(&mut out))
            })
            .expect_err("a tracked operand has no gradient path into a caller-owned destination");
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
        assert_untouched(&out);
    }

    {
        let _capture = ctx.capture_trace();
        let mut out = vector(&SENTINEL);
        let err = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                s.add_into(&c, &c, Dest::overwrite(&mut out))
            })
            .expect_err("an active trace capture cannot record a caller-owned destination");
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{err}");
        assert_untouched(&out);
    }

    // Under no_grad the same tracked operand is a plain value.
    let _guard = ctx.no_grad();
    let mut out = vector(&SENTINEL);
    ctx.with_eager_session(&mut CpuEngine::serial(), |s| {
        s.add_into(&x, &c, Dest::overwrite(&mut out))
    })
    .expect("no_grad add_into");
    assert_eq!(out.as_slice::<f64>().expect("host"), &[2.0, 3.0, 4.0]);
}

#[test]
fn add_into_rejects_foreign_runtime_and_mismatched_destination_before_writing() {
    let ctx = EagerRuntime::new().expect("runtime");
    let other = EagerRuntime::new().expect("other runtime");
    let x = constant(&ctx, vector(&[1.0, 2.0, 3.0]));
    let foreign = constant(&other, vector(&[1.0, 2.0, 3.0]));

    let mut out = vector(&SENTINEL);
    let err = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.add_into(&x, &foreign, Dest::overwrite(&mut out))
        })
        .expect_err("an operand of another runtime");
    assert!(matches!(err, Error::ContextMismatch { .. }), "{err}");
    assert_untouched(&out);

    // An f32 operand: add_into does not promote into an f64 destination.
    let narrow = constant(
        &ctx,
        Tensor::from_vec_col_major(vec![3], vec![1.0_f32, 2.0, 3.0]).expect("f32 vector"),
    );
    let mut out = vector(&SENTINEL);
    let err = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.add_into(&x, &narrow, Dest::overwrite(&mut out))
        })
        .expect_err("mixed operand dtypes");
    assert!(matches!(err.kind(), ErrorKind::Validation(_)), "{err}");
    assert_untouched(&out);

    // A destination whose shape is not the operands' broadcast shape.
    let long = constant(&ctx, vector(&[1.0, 2.0]));
    let mut out = vector(&SENTINEL);
    let err = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            s.add_into(&long, &long, Dest::overwrite(&mut out))
        })
        .expect_err("destination shape mismatch");
    assert!(matches!(err.kind(), ErrorKind::Validation(_)), "{err}");
    assert_untouched(&out);
}

type Session<'a> = tenferro_ad::EagerSession<'a>;
type IntoFn = fn(
    &mut Session<'_>,
    &EagerTensor,
    &EagerTensor,
    Dest<tenferro_tensor::TensorWrite<'_>>,
) -> tenferro_ad::Result<()>;
type OwnedFn = fn(&mut Session<'_>, &EagerTensor, &EagerTensor) -> tenferro_ad::Result<EagerTensor>;

/// Every eager `_into` beside its allocating form; unary forms ignore `rhs`.
fn eager_into_cases() -> [(&'static str, IntoFn, OwnedFn); 6] {
    [
        (
            "add",
            |s, a, b, d| s.add_into(a, b, d),
            |s, a, b| s.add(a, b),
        ),
        (
            "sub",
            |s, a, b, d| s.sub_into(a, b, d),
            |s, a, b| s.sub(a, b),
        ),
        (
            "mul",
            |s, a, b, d| s.mul_into(a, b, d),
            |s, a, b| s.mul(a, b),
        ),
        (
            "div",
            |s, a, b, d| s.div_into(a, b, d),
            |s, a, b| s.div(a, b),
        ),
        ("neg", |s, a, _, d| s.neg_into(a, d), |s, a, _| s.neg(a)),
        ("conj", |s, a, _, d| s.conj_into(a, d), |s, a, _| s.conj(a)),
    ]
}

#[test]
fn every_eager_into_matches_its_allocating_form_and_rejects_before_writing() {
    let ctx = EagerRuntime::new().expect("runtime");
    let x = constant(&ctx, vector(&[1.5, -2.0, 3.25]));
    let y = constant(&ctx, vector(&[0.5, 4.0, -8.0]));
    let tracked = ctx
        .variable_from(vector(&[1.0, 2.0, 3.0]))
        .expect("variable");
    for (name, into, owned) in eager_into_cases() {
        let mut out = vector(&[f64::NAN, -0.0, f64::INFINITY]);
        let want = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                into(s, &x, &y, Dest::overwrite(&mut out))?;
                owned(s, &x, &y)
            })
            .unwrap_or_else(|err| panic!("{name}: {err}"))
            .into_value()
            .expect("unique result");
        assert_eq!(
            out.as_slice::<f64>()
                .expect("host")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            want.as_slice::<f64>()
                .expect("host")
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "{name}_into must equal {name} bit for bit"
        );

        let mut out = vector(&SENTINEL);
        let err = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                into(s, &x, &y, Dest::accumulate(&mut out))
            })
            .expect_err("only Dest::Overwrite is declared");
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{name}: {err}");
        assert_untouched(&out);

        let mut out = vector(&SENTINEL);
        let err = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                into(s, &tracked, &tracked, Dest::overwrite(&mut out))
            })
            .expect_err("a tracked operand");
        assert_eq!(err.kind(), ErrorKind::Unsupported, "{name}: {err}");
        assert_untouched(&out);
    }
}

#[test]
fn eager_into_rejects_a_real_operand_against_a_complex_destination_before_writing() {
    use num_complex::Complex64;
    let ctx = EagerRuntime::new().expect("runtime");
    // Rank-zero operands: an allocating CPU kernel would promote the real scalar.
    let real = constant(
        &ctx,
        Tensor::from_vec_col_major(vec![], vec![1.0_f64]).expect("real scalar"),
    );
    let complex = constant(
        &ctx,
        Tensor::from_vec_col_major(vec![], vec![Complex64::new(2.0, 1.0)]).expect("complex scalar"),
    );
    let sentinel = Complex64::new(7.5, -3.25);
    for (name, into, _) in eager_into_cases() {
        let mut out = Tensor::from_vec_col_major(vec![], vec![sentinel]).expect("destination");
        let err = ctx
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                into(s, &real, &complex, Dest::overwrite(&mut out))
            })
            .expect_err("a real operand cannot overwrite a complex destination");
        assert!(
            matches!(err.kind(), ErrorKind::Validation(_)),
            "{name}: {err}"
        );
        assert_eq!(
            out.as_slice::<Complex64>().expect("host"),
            &[sentinel],
            "{name}_into wrote a rejected result"
        );
    }
}
