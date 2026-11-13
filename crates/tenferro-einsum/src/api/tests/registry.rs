//! The guards of the generated einsum grammar (#2552 §4.4), the error causes
//! (§6.5) and the generated documentation check (§4.5).

use std::error::Error as StdError;

use num_complex::Complex64;
use tenferro_cpu::CpuEngine;
use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime, TraceContext, TracedTensor};
use tenferro_tensor::{
    Conj, DType, Dest, ErrorKind, StridedSliceSpec, Tensor, TensorRead, TensorScalar,
    ValidationKind,
};

use crate::api::registry::{api_docs_sync, API_OPS};
use crate::{
    ConcreteEinsumPlan, ConjMask, EinsumOps, EinsumOptimize, EinsumOptions, TensorDotAxes,
    TensordotOps, TracedEinsumOps, TracedTensordotOps,
};

type R<T> = Result<T, Box<dyn StdError>>;

fn bits(t: &Tensor) -> R<Vec<u64>> {
    let mut out: Vec<u64> = t.shape().iter().map(|&d| d as u64).collect();
    out.push(u64::MAX);
    match t.dtype() {
        DType::F64 => out.extend(t.as_slice::<f64>()?.iter().map(|x| x.to_bits())),
        DType::C64 => out.extend(
            t.as_slice::<Complex64>()?
                .iter()
                .flat_map(|x| [x.re.to_bits(), x.im.to_bits()]),
        ),
        other => return Err(format!("dtype {other:?}").into()),
    }
    Ok(out)
}

/// Equal values, shape and dtype; `+0.0` and `-0.0` compare equal. The
/// allocating and `_into` routes agree on values, but on some hosts (notably
/// x86_64) they may differ in the sign of an exactly cancelled zero.
fn same_values(a: &Tensor, b: &Tensor) -> R<bool> {
    if a.shape() != b.shape() || a.dtype() != b.dtype() {
        return Ok(false);
    }
    Ok(match a.dtype() {
        DType::F64 => a
            .as_slice::<f64>()?
            .iter()
            .zip(b.as_slice::<f64>()?)
            .all(|(x, y)| x == y),
        DType::C64 => a
            .as_slice::<Complex64>()?
            .iter()
            .zip(b.as_slice::<Complex64>()?)
            .all(|(x, y)| x.re == y.re && x.im == y.im),
        other => return Err(format!("dtype {other:?}").into()),
    })
}

/// Integer-valued data, so every summation order is exact and layouts can be
/// compared bit for bit.
fn ints(shape: &[usize], seed: i64) -> R<Tensor> {
    let n: usize = shape.iter().product();
    Ok(Tensor::from_vec_col_major(
        shape.to_vec(),
        (0..n as i64)
            .map(|i| ((i * 7 + seed) % 11 - 5) as f64)
            .collect::<Vec<_>>(),
    )?)
}

fn cints(shape: &[usize], seed: i64) -> R<Tensor> {
    let n: usize = shape.iter().product();
    Ok(Tensor::from_vec_col_major(
        shape.to_vec(),
        (0..n as i64)
            .map(|i| {
                Complex64::new(
                    ((i * 7 + seed) % 11 - 5) as f64,
                    ((i * 3 + seed) % 5 - 2) as f64,
                )
            })
            .collect::<Vec<_>>(),
    )?)
}

fn run_traced(outputs: &[&TracedTensor]) -> R<Vec<Tensor>> {
    let program = GraphCompiler::new().compile_many(outputs)?;
    let mut engine = CpuEngine::serial();
    let mut builder = Runtime::builder();
    builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
    builder.install_extension_module(crate::extension_module::<CpuEngine>(
        tenferro_cpu::runtime_engine_id()?,
    )?)?;
    let runtime = builder.build()?;
    let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
    let results = {
        let mut engines: [&mut dyn ExecutionEngine; 1] = [&mut engine];
        runtime.run(&mut engines, &prepared, &[])?
    };
    results
        .into_iter()
        .map(|r| r.into_tensor(&mut engine).map_err(Into::into))
        .collect()
}

fn traced_of(trace: &TraceContext, t: &Tensor) -> R<TracedTensor> {
    Ok(trace.from_tensor_concrete_shape(t.duplicate()?)?)
}

// ------------------------------------------------------------ 1. forwarding

/// `(op, layer)` -> bits of the plain call and of `_with(default)`.
fn forwarding(op: &str, layer: &str, complex: bool) -> R<(Vec<u64>, Vec<u64>)> {
    let mk = |s: &[usize], seed| {
        if complex {
            cints(s, seed)
        } else {
            ints(s, seed)
        }
    };
    let (a, b, c) = (mk(&[2, 3], 1)?, mk(&[3, 4], 2)?, mk(&[4, 2], 3)?);
    let mut e = CpuEngine::serial();
    match (op, layer) {
        ("einsum", "concrete") => Ok((
            bits(&[&a, &b, &c].einsum("ij,jk,kl->il", &mut e)?)?,
            bits(&[&a, &b, &c].einsum_with("ij,jk,kl->il", &EinsumOptions::default(), &mut e)?)?,
        )),
        ("tensordot", "concrete_tensordot") => Ok((
            bits(&a.tensordot(&b, TensorDotAxes::Count(1), &mut e)?)?,
            bits(&a.tensordot(&b, TensorDotAxes::Count(1), &mut e)?)?,
        )),
        ("einsum", "traced") => {
            let trace = TraceContext::new();
            let (ta, tb, tc) = (
                traced_of(&trace, &a)?,
                traced_of(&trace, &b)?,
                traced_of(&trace, &c)?,
            );
            let p = trace.einsum(&[&ta, &tb, &tc], "ij,jk,kl->il")?;
            let w =
                trace.einsum_with(&[&ta, &tb, &tc], "ij,jk,kl->il", &EinsumOptions::default())?;
            let out = run_traced(&[&p, &w])?;
            Ok((bits(&out[0])?, bits(&out[1])?))
        }
        ("tensordot", "traced_tensordot") => {
            let trace = TraceContext::new();
            let (ta, tb) = (traced_of(&trace, &a)?, traced_of(&trace, &b)?);
            let p = ta.tensordot(&tb, TensorDotAxes::Count(1))?;
            let w = ta.tensordot(&tb, TensorDotAxes::Count(1))?;
            let out = run_traced(&[&p, &w])?;
            Ok((bits(&out[0])?, bits(&out[1])?))
        }
        #[cfg(feature = "autodiff")]
        (op, "eager") => {
            use crate::EagerEinsumOps;
            let runtime = tenferro_ad::EagerRuntime::new()?;
            let (p, w) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
                let xa = s.constant_from(a.duplicate()?)?;
                let xb = s.constant_from(b.duplicate()?)?;
                let xc = s.constant_from(c.duplicate()?)?;
                if op == "einsum" {
                    Ok::<_, tenferro_ad::Error>((
                        s.einsum(&[&xa, &xb, &xc], "ij,jk,kl->il")?,
                        s.einsum_with(&[&xa, &xb, &xc], "ij,jk,kl->il", &EinsumOptions::default())?,
                    ))
                } else {
                    Ok((
                        s.tensordot(&xa, &xb, TensorDotAxes::Count(1))?,
                        s.tensordot(&xa, &xb, TensorDotAxes::Count(1))?,
                    ))
                }
            })?;
            Ok((
                bits(
                    &p.runtime()
                        .with_eager_session(&mut e, |s| s.duplicate_value(&p))?,
                )?,
                bits(
                    &w.runtime()
                        .with_eager_session(&mut e, |s| s.duplicate_value(&w))?,
                )?,
            ))
        }
        other => Err(format!("no forwarding case for {other:?}").into()),
    }
}

#[test]
fn every_registry_op_and_layer_forwards_plain_to_with_default_bit_for_bit() -> R<()> {
    let mut checked = 0;
    for op in API_OPS {
        for layer in op.layers {
            if *layer == "eager" && cfg!(not(feature = "autodiff")) {
                continue;
            }
            for complex in [false, true] {
                let (plain, with) = forwarding(op.name, layer, complex)?;
                assert_eq!(plain, with, "{layer}.{} complex={complex}", op.name);
                checked += 1;
            }
        }
    }
    let expected = if cfg!(feature = "autodiff") { 12 } else { 8 };
    assert_eq!(checked, expected);
    Ok(())
}

// ---------------------------------------------------------- 2. input shapes

#[test]
fn every_input_kind_returns_the_owned_result() -> R<()> {
    let mut e = CpuEngine::serial();
    let a = ints(&[3, 3], 1)?;
    let b = ints(&[3, 3], 4)?;
    let ta = a.as_typed::<f64>().ok_or("typed")?.duplicate()?;
    let tb = b.as_typed::<f64>().ok_or("typed")?.duplicate()?;
    let want = |l: &Tensor, r: &Tensor, e: &mut CpuEngine| -> R<Vec<u64>> {
        bits(&[l, r].einsum("ij,jk->ik", e)?)
    };
    let base = want(&a, &b, &mut e)?;

    // compact borrowed view
    let reads = [TensorRead::from_tensor(&a), TensorRead::from_tensor(&b)];
    assert_eq!(
        bits(&reads.einsum("ij,jk->ik", &mut e)?)?,
        base,
        "compact view"
    );
    // typed tensors
    let typed = [&ta, &tb].einsum("ij,jk->ik", &mut e)?;
    assert_eq!(bits(&Tensor::from_typed(typed))?, base, "typed");
    // transposed and negative-stride views against the materialized matrix
    for (label, view) in [
        ("transposed", ta.as_view().transpose_view([1, 0])?),
        (
            "negative-stride",
            ta.as_view()
                .slice_view(&[StridedSliceSpec::reverse(), StridedSliceSpec::reverse()])?,
        ),
    ] {
        let mut data = Vec::new();
        for col in 0..3 {
            for row in 0..3 {
                data.push(*view.get(&[row, col]).ok_or("index")?);
            }
        }
        let logical = Tensor::from_vec_col_major(vec![3, 3], data)?;
        let expected = want(&logical, &b, &mut e)?;
        let views = [view.clone(), tb.as_view()];
        let got = views.einsum("ij,jk->ik", &mut e)?;
        assert_eq!(
            bits(&Tensor::from_typed(got))?,
            expected,
            "{label} typed view"
        );
        let erased = [
            TensorRead::from_view(f64::tensor_view(view)),
            TensorRead::from_tensor(&b),
        ];
        assert_eq!(
            bits(&erased.einsum("ij,jk->ik", &mut e)?)?,
            expected,
            "{label} erased view"
        );
    }
    Ok(())
}

// ------------------------------------------------------------------ 3. _into

fn filled(shape: &[usize], v: f64) -> R<Tensor> {
    Ok(Tensor::from_vec_col_major(
        shape.to_vec(),
        vec![v; shape.iter().product()],
    )?)
}

#[test]
fn einsum_into_honours_every_destination_mode() -> R<()> {
    let mut e = CpuEngine::serial();
    let o = EinsumOptions::default();
    for (spec, inputs, shape) in [
        (
            "ij,jk->ik",
            vec![ints(&[2, 3], 1)?, ints(&[3, 2], 2)?],
            vec![2, 2],
        ),
        (
            "ij,jk,kl->il",
            vec![ints(&[2, 3], 1)?, ints(&[3, 4], 2)?, ints(&[4, 2], 3)?],
            vec![2, 2],
        ),
    ] {
        let refs: Vec<&Tensor> = inputs.iter().collect();
        let r = refs.einsum(spec, &mut e)?;
        let r = r.as_slice::<f64>()?.to_vec();
        // Overwrite never reads the destination.
        for poison in [f64::NAN, -0.0, f64::INFINITY] {
            let mut out = filled(&shape, poison)?;
            refs.einsum_into(spec, &o, &mut e, Dest::overwrite(&mut out))?;
            assert_eq!(out.as_slice::<f64>()?, &r[..], "{spec} overwrite {poison}");
        }
        // Accumulate adds.
        let mut out = filled(&shape, 1.5)?;
        refs.einsum_into(spec, &o, &mut e, Dest::accumulate(&mut out))?;
        let want: Vec<f64> = r.iter().map(|x| x + 1.5).collect();
        assert_eq!(out.as_slice::<f64>()?, &want[..], "{spec} accumulate");
        // Scaled with beta = 0 leaks no NaN and no -0.
        for poison in [f64::NAN, -0.0] {
            let mut out = filled(&shape, poison)?;
            refs.einsum_into(spec, &o, &mut e, Dest::scaled(&mut out, 2.0, 0.0))?;
            let want: Vec<u64> = r.iter().map(|x| (2.0 * x + 0.0).to_bits()).collect();
            let got: Vec<u64> = out.as_slice::<f64>()?.iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, want, "{spec} scaled beta=0 over {poison}");
        }
        // Scaled with alpha and beta.
        let mut out = filled(&shape, 2.0)?;
        refs.einsum_into(spec, &o, &mut e, Dest::scaled(&mut out, 3.0, 0.5))?;
        let want: Vec<f64> = r.iter().map(|x| 3.0 * x + 1.0).collect();
        assert_eq!(out.as_slice::<f64>()?, &want[..], "{spec} scaled");
    }
    Ok(())
}

// ----------------------------------------------- 4. refusals and omissions

#[test]
fn refusals_happen_before_any_write() -> R<()> {
    let mut e = CpuEngine::serial();
    let (a, b) = (ints(&[2, 3], 1)?, ints(&[3, 2], 2)?);
    // A conjugation mask longer than the input list.
    let o = EinsumOptions::default().conj(ConjMask::new([Conj::No, Conj::No, Conj::Yes]));
    let mut out = filled(&[2, 2], 7.0)?;
    let err = [&a, &b]
        .einsum_into("ij,jk->ik", &o, &mut e, Dest::accumulate(&mut out))
        .expect_err("mask longer than the inputs");
    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument),
        "{err}"
    );
    assert_eq!(out.as_slice::<f64>()?, &[7.0; 4]);
    // A scaled destination whose scalars have another dtype.
    let mut out = filled(&[2, 2], 7.0)?;
    let err = [&a, &b]
        .einsum_into(
            "ij,jk->ik",
            &EinsumOptions::default(),
            &mut e,
            Dest::scaled(&mut out, 1.0_f32, 0.0_f32),
        )
        .expect_err("scalar dtype");
    assert!(matches!(err.kind(), ErrorKind::Validation(_)), "{err}");
    assert_eq!(out.as_slice::<f64>()?, &[7.0; 4]);
    // An explicit path that does not fit is refused, never ignored.
    let bad = EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(0, 5)]));
    let c = ints(&[2, 2], 3)?;
    assert!([&a, &b, &c]
        .einsum_with("ij,jk,kl->il", &bad, &mut e)
        .is_err());
    assert!(ConcreteEinsumPlan::prepare(&[&a, &b, &c], "ij,jk,kl->il", &bad).is_err());
    Ok(())
}

#[cfg(feature = "native")]
#[test]
fn einsum_in_a_lane_is_unsupported_and_leaves_the_destination() -> R<()> {
    use tenferro_cpu::linalg_interop::BufferPool;
    let (a, b) = (ints(&[2, 3], 1)?, ints(&[3, 2], 2)?);
    let mut engine = CpuEngine::serial();
    let mut pools = [BufferPool::new()];
    let kinds = engine.with_lanes(&[1], &mut pools, |_, lane| {
        let mut out = Tensor::from_vec_col_major(vec![2, 2], vec![7.0_f64; 4]).ok();
        let fresh = [&a, &b]
            .einsum("ij,jk->ik", lane)
            .map(|_| ())
            .map_err(|e| e.kind());
        let into = match out.as_mut() {
            Some(out) => [&a, &b]
                .einsum_into(
                    "ij,jk->ik",
                    &EinsumOptions::default(),
                    lane,
                    Dest::overwrite(out),
                )
                .map_err(|e| e.kind()),
            None => Ok(()),
        };
        let untouched = out
            .as_ref()
            .and_then(|o| o.as_slice::<f64>().ok().map(|s| s == [7.0; 4]));
        (fresh, into, untouched)
    })?;
    let (fresh, into, untouched) = &kinds[0];
    assert_eq!(fresh, &Err(ErrorKind::Unsupported));
    assert_eq!(into, &Err(ErrorKind::Unsupported));
    assert_eq!(untouched, &Some(true));
    Ok(())
}

// ------------------------------------------------- options are honoured

#[test]
fn conjugation_mask_conjugates_the_selected_operands_on_every_layer() -> R<()> {
    let mut e = CpuEngine::serial();
    let (a, b, c) = (cints(&[2, 3], 1)?, cints(&[3, 2], 2)?, cints(&[2, 2], 3)?);
    let conj = |t: &Tensor, e: &mut CpuEngine| -> R<Tensor> {
        Ok(tenferro_tensor::TensorElementwise::conj_read(
            e,
            TensorRead::from_tensor(t),
        )?)
    };
    let (ca, cc) = (conj(&a, &mut e)?, conj(&c, &mut e)?);
    for (spec, inputs, explicit) in [
        ("ij,jk->ik", vec![&a, &b], vec![&ca, &b]),
        ("ij,jk,kl->il", vec![&a, &b, &c], vec![&ca, &b, &cc]),
    ] {
        let mask = ConjMask::new([Conj::Yes, Conj::No, Conj::Yes]);
        let mask = ConjMask::new((0..inputs.len()).map(|i| mask.get(i)));
        let o = EinsumOptions::default().conj(mask);
        let want = bits(&explicit.einsum(spec, &mut e)?)?;
        assert_eq!(
            bits(&inputs.einsum_with(spec, &o, &mut e)?)?,
            want,
            "concrete {spec}"
        );
        // `_into`: the selected operands are conjugated first (#2580). The same
        // route on explicitly conjugated inputs is the bit-exact reference.
        let zero = || Tensor::from_vec_col_major(vec![2, 2], vec![Complex64::new(0.0, 0.0); 4]);
        let (mut out, mut want_into) = (zero()?, zero()?);
        inputs.einsum_into(spec, &o, &mut e, Dest::overwrite(&mut out))?;
        explicit.einsum_into(
            spec,
            &EinsumOptions::default(),
            &mut e,
            Dest::overwrite(&mut want_into),
        )?;
        assert_eq!(bits(&out)?, bits(&want_into)?, "concrete into {spec}");
        assert!(
            same_values(&out, &explicit.einsum(spec, &mut e)?)?,
            "concrete into values {spec}"
        );
        // the plan
        let plan = ConcreteEinsumPlan::prepare(&inputs, spec, &o)?;
        assert_eq!(bits(&plan.execute(&inputs, &mut e)?)?, want, "plan {spec}");
        // traced
        let trace = TraceContext::new();
        let ts: Vec<TracedTensor> = inputs
            .iter()
            .map(|t| traced_of(&trace, t))
            .collect::<R<_>>()?;
        let refs: Vec<&TracedTensor> = ts.iter().collect();
        let t = trace.einsum_with(&refs, spec, &o)?;
        assert_eq!(bits(&run_traced(&[&t])?[0])?, want, "traced {spec}");
    }
    Ok(())
}

#[test]
fn explicit_orders_are_used_and_give_the_automatic_values() -> R<()> {
    let mut e = CpuEngine::serial();
    let (a, b, c) = (ints(&[2, 3], 1)?, ints(&[3, 4], 2)?, ints(&[4, 2], 3)?);
    let want = bits(&[&a, &b, &c].einsum("ij,jk,kl->il", &mut e)?)?;
    let subs = crate::Subscripts::parse("ij,jk,kl->il")?;
    let tree =
        crate::ContractionTree::from_pairs(&subs, &[&[2, 3], &[3, 4], &[4, 2]], &[(1, 2), (0, 3)])?;
    for optimize in [
        EinsumOptimize::False,
        EinsumOptimize::Path(vec![(1, 2), (0, 1)]),
        EinsumOptimize::Tree(tree),
    ] {
        let o = EinsumOptions::default().optimize(optimize.clone());
        // Integer data: every order is exact.
        assert_eq!(
            bits(&[&a, &b, &c].einsum_with("ij,jk,kl->il", &o, &mut e)?)?,
            want,
            "{optimize:?}"
        );
        let plan = ConcreteEinsumPlan::prepare(&[&a, &b, &c], "ij,jk,kl->il", &o)?;
        assert_eq!(
            bits(&plan.execute(&[&a, &b, &c], &mut e)?)?,
            want,
            "plan {optimize:?}"
        );
    }
    // The plan really follows the requested pairs.
    let o = EinsumOptions::default().optimize(EinsumOptimize::Path(vec![(1, 2), (0, 1)]));
    let plan = ConcreteEinsumPlan::prepare(&[&a, &b, &c], "ij,jk,kl->il", &o)?;
    assert_eq!(plan.tree_step_pair(0), Some((1, 2)));
    Ok(())
}

// ------------------------------------------------------- 6.5 error causes

fn is_invalid_subscripts(err: &(dyn StdError + 'static)) -> bool {
    let mut cur = Some(err);
    while let Some(e) = cur {
        if matches!(
            e.downcast_ref::<crate::Error>(),
            Some(crate::Error::InvalidSubscripts { .. })
        ) {
            return true;
        }
        cur = e.source();
    }
    false
}

#[test]
fn an_invalid_spec_has_one_kind_and_cause_on_every_layer() -> R<()> {
    let invalid = ErrorKind::Validation(ValidationKind::InvalidArgument);
    let (a, b) = (ints(&[2, 3], 1)?, ints(&[3, 2], 2)?);
    let spec = "ij,(jk)->ik";
    let concrete = [&a, &b]
        .einsum(spec, &mut CpuEngine::serial())
        .expect_err("concrete");
    assert_eq!(concrete.kind(), invalid);
    assert!(is_invalid_subscripts(&concrete), "concrete: {concrete:?}");
    let plan =
        ConcreteEinsumPlan::prepare(&[&a, &b], spec, &EinsumOptions::default()).expect_err("plan");
    assert_eq!(plan.kind(), invalid);
    assert!(is_invalid_subscripts(&plan), "plan: {plan:?}");

    let trace = TraceContext::new();
    let (ta, tb) = (traced_of(&trace, &a)?, traced_of(&trace, &b)?);
    let traced = trace.einsum(&[&ta, &tb], spec).expect_err("traced");
    assert_eq!(traced.kind(), invalid);
    assert!(is_invalid_subscripts(&traced), "traced: {traced:?}");

    #[cfg(feature = "autodiff")]
    {
        use crate::EagerEinsumOps;
        let runtime = tenferro_ad::EagerRuntime::new()?;
        let eager = runtime
            .with_eager_session(&mut CpuEngine::serial(), |s| {
                let xa = s.constant_from(a.duplicate()?)?;
                let xb = s.constant_from(b.duplicate()?)?;
                s.einsum(&[&xa, &xb], spec)
            })
            .expect_err("eager");
        assert_eq!(eager.kind(), invalid);
        assert!(is_invalid_subscripts(&eager), "eager: {eager:?}");
    }
    Ok(())
}

#[test]
fn a_traced_input_from_another_trace_is_refused() -> R<()> {
    let (one, other) = (TraceContext::new(), TraceContext::new());
    let a = traced_of(&other, &ints(&[2, 2], 1)?)?;
    let err = one.einsum(&[&a], "ij->ji").expect_err("foreign input");
    assert!(matches!(err.kind(), ErrorKind::Validation(_)), "{err}");
    Ok(())
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
         `UPDATE_API_DOCS=1 cargo test -p tenferro-einsum --features autodiff api_docs_are_current`"
    );
}

/// #2580: `einsum_into` with a conjugation mask gives the same bits as the
/// same `_into` route on the explicitly conjugated inputs, for every
/// destination mode, and the same values as the owned entrance (only the sign
/// of an exact zero may differ between the allocating and `_into` routes).
/// Exactly cancelling imaginary parts are where the former two-operand route
/// (mask as contraction operand flags, evaluated as `conj(a b)`) wrote `-0.0`
/// where the explicitly conjugated inputs write `+0.0`.
#[test]
fn conjugation_mask_into_matches_explicit_conjugation_bit_for_bit_in_every_mode() -> R<()> {
    let mut e = CpuEngine::serial();
    let conj = |t: &Tensor, e: &mut CpuEngine| -> R<Tensor> {
        Ok(tenferro_tensor::TensorElementwise::conj_read(
            e,
            TensorRead::from_tensor(t),
        )?)
    };
    let neg_zero = Complex64::new(-0.0, -0.0);
    let masks = [
        [Conj::Yes, Conj::No, Conj::Yes],
        [Conj::No, Conj::Yes, Conj::No],
        [Conj::Yes; 3],
    ];
    for (m, k, n) in [(1, 2, 4), (1, 2, 5), (2, 3, 2), (3, 4, 3)] {
        for seed in 0..6 {
            let (a, b, c) = (
                cints(&[m, k], seed)?,
                cints(&[k, n], seed + 3)?,
                cints(&[n, 2], seed + 5)?,
            );
            let (ca, cb, cc) = (conj(&a, &mut e)?, conj(&b, &mut e)?, conj(&c, &mut e)?);
            for mask in masks {
                let pick = |i: usize| mask[i].is_yes();
                let ea = if pick(0) { &ca } else { &a };
                let eb = if pick(1) { &cb } else { &b };
                let ec = if pick(2) { &cc } else { &c };
                for (spec, inputs, explicit, cols) in [
                    ("ij,jk->ik", vec![&a, &b], vec![ea, eb], n),
                    ("ij,jk,kl->il", vec![&a, &b, &c], vec![ea, eb, ec], 2),
                ] {
                    let o = EinsumOptions::default()
                        .conj(ConjMask::new((0..inputs.len()).map(|i| mask[i])));
                    let plain = EinsumOptions::default();
                    let at = format!("{spec} m{m} k{k} n{n} seed {seed} {mask:?}");
                    let owned = inputs.einsum_with(spec, &o, &mut e)?;
                    assert!(
                        same_values(&owned, &explicit.einsum(spec, &mut e)?)?,
                        "owned {at}"
                    );
                    let fill = |v| Tensor::from_vec_col_major(vec![m, cols], vec![v; m * cols]);
                    let zero = Complex64::new(0.0, 0.0);
                    let (mut out, mut want) = (fill(zero)?, fill(zero)?);
                    inputs.einsum_into(spec, &o, &mut e, Dest::overwrite(&mut out))?;
                    explicit.einsum_into(spec, &plain, &mut e, Dest::overwrite(&mut want))?;
                    assert_eq!(bits(&out)?, bits(&want)?, "overwrite {at}");
                    assert!(same_values(&out, &owned)?, "overwrite values {at}");
                    // accumulating modes: the same mode on the explicit inputs
                    let (mut got, mut want) = (fill(neg_zero)?, fill(neg_zero)?);
                    inputs.einsum_into(spec, &o, &mut e, Dest::accumulate(&mut got))?;
                    explicit.einsum_into(spec, &plain, &mut e, Dest::accumulate(&mut want))?;
                    assert_eq!(bits(&got)?, bits(&want)?, "accumulate {at}");
                    let (alpha, beta) = (Complex64::new(2.0, 0.0), Complex64::new(0.5, 0.0));
                    let (mut got, mut want) = (
                        fill(Complex64::new(1.0, -1.0))?,
                        fill(Complex64::new(1.0, -1.0))?,
                    );
                    inputs.einsum_into(spec, &o, &mut e, Dest::scaled(&mut got, alpha, beta))?;
                    explicit.einsum_into(
                        spec,
                        &plain,
                        &mut e,
                        Dest::scaled(&mut want, alpha, beta),
                    )?;
                    assert_eq!(bits(&got)?, bits(&want)?, "scaled {at}");
                }
            }
        }
    }
    Ok(())
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
    assert_eq!(with_options, ["einsum"]);

    // Every declared options type is non-empty: a derived `Debug` prints
    // `Name { field: .. }` for a struct with fields and only `Name` for an
    // empty one.
    let defaults: &[(&str, String)] = &[(
        "EinsumOptions",
        format!("{:?}", crate::EinsumOptions::default()),
    )];
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
