//! Gradients of the sampled einsum (#2675) through the eager layer: the
//! sampled plan against a dense einsum followed by a select, both recorded on
//! one eager tape over the same variables.
//!
//! Provenance: ported from `tests/ad.rs` of Fanerst/sampled-einsum at commit
//! c482b79a0f1e407390ef7822a89e7706220021d1 (Feng Pan, Apache-2.0): the dense
//! route (einsum to the full output, move the sampled labels last, merge them,
//! select one row per sample), the weighted-sum loss, the two planner settings
//! and the 60 random networks. The prototype records each route on its own
//! runtime; here both routes share one tape and their VJPs are taken from it.
//! Added here: the eager executor's measured peak
//! against the planned eager peak, a forward-mode (JVP) check against the
//! dense route, the eager `einsum_at` spelling and conjugation.
#![cfg(feature = "autodiff")]

use tenferro_ad::{AdContext, EagerRuntime, EagerSession, EagerTensor};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::{
    ConjMask, EagerEinsumOps, EinsumOptions, EinsumSubscripts, SampleOptions, SampledEinsumPlan,
    Samples,
};
use tenferro_tensor::{Conj, DType, Tensor};

use super::sampled_einsum::{operands, random_case, sizes, Lcg, Network};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// `y` by the dense route: einsum to the full output, move the sampled labels
/// last, merge them, and select one row per sample.
fn dense_route(
    s: &mut EagerSession<'_>,
    net: &Network,
    samples: &Samples,
    vars: &[&EagerTensor],
) -> tenferro_ad::Result<EagerTensor> {
    let full = s.einsum(vars, &net.subscripts())?;
    let rest: Vec<u32> = net
        .output
        .iter()
        .copied()
        .filter(|l| !samples.labels.contains(l))
        .collect();
    let mut order = rest.clone();
    order.extend(&samples.labels);
    let moved = s.einsum(
        &[&full],
        &EinsumSubscripts::new(&[net.output.as_slice()], &order),
    )?;
    let mut shape: Vec<usize> = rest.iter().map(|l| net.sizes[l]).collect();
    shape.push(samples.labels.iter().map(|l| net.sizes[l]).product());
    let merged = s.reshape(&moved, shape)?;
    let positions: Vec<usize> = (0..samples.n)
        .map(|i| {
            let mut lin = 0;
            let mut stride = 1;
            for (j, l) in samples.labels.iter().enumerate() {
                lin += samples.value(i, j).expect("in range") * stride;
                stride *= net.sizes[l];
            }
            lin
        })
        .collect();
    s.index_select(&merged, rest.len() as isize, &positions)
}

fn ad_runtime() -> Res<std::sync::Arc<EagerRuntime>> {
    // Einsum's derivative rules live in tenferro-einsum and are installed through the AD context.
    let ad = AdContext::builder()
        .with_semantic_extension_rules(tenferro_einsum::semantic_ad_rules()?)?
        .build()?;
    Ok(EagerRuntime::with_ad_context(&ad)?)
}

type Route<'r> =
    dyn Fn(&mut EagerSession<'_>, &[&EagerTensor]) -> tenferro_ad::Result<EagerTensor> + 'r;

/// Both routes recorded on one eager tape over the same variables: their
/// values and, for each operand, the VJP of each with cotangent `weights`
/// (the gradient of `sum(weights * y)`).
#[allow(clippy::type_complexity)]
fn values_and_grads_on_one_tape(
    sampled: &Route<'_>,
    dense: &Route<'_>,
    operands: &[Tensor],
    weights: &Tensor,
) -> Res<((Vec<f64>, Vec<f64>), Vec<(Vec<f64>, Vec<f64>)>)> {
    let runtime = ad_runtime()?;
    let mut engine = CpuEngine::serial();
    let vars = operands
        .iter()
        .map(|t| runtime.variable_from(t.duplicate()?))
        .collect::<tenferro_ad::Result<Vec<_>>>()?;
    let w = runtime.constant_from(weights.duplicate()?)?;
    let (ys, yd, grads) = runtime.with_eager_session(&mut engine, |s| {
        let refs: Vec<&EagerTensor> = vars.iter().collect();
        let ys = sampled(s, &refs)?;
        let yd = dense(s, &refs)?;
        let mut grads = Vec::new();
        for x in &vars {
            let gs = s.vjp_optional(&ys, x, &w)?;
            let gd = s.vjp_optional(&yd, x, &w)?;
            grads.push((gs, gd));
        }
        Ok::<_, tenferro_ad::Error>((ys, yd, grads))
    })?;
    let host = |t: &EagerTensor| -> Res<Vec<f64>> { Ok(t.value()?.as_slice::<f64>()?.to_vec()) };
    let mut out = Vec::new();
    for (gs, gd) in grads {
        let gs = gs.ok_or("the sampled route gave no gradient")?;
        let gd = gd.ok_or("the dense route gave no gradient")?;
        out.push((host(&gs)?, host(&gd)?));
    }
    Ok(((host(&ys)?, host(&yd)?), out))
}

fn assert_close(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-11 * (1.0 + w.abs()),
            "{what}: entry {k}: got {g}, want {w}"
        );
    }
}

fn weights_for(rng: &mut Lcg, shape: &[usize]) -> Res<Tensor> {
    let len: usize = shape.iter().product();
    Ok(Tensor::from_vec_col_major(
        shape.to_vec(),
        (0..len).map(|_| rng.unit()).collect::<Vec<f64>>(),
    )?)
}

#[test]
fn gradients_match_the_dense_route() -> Res<()> {
    let mut rng = Lcg(856);
    let settings = [
        SampleOptions::default(),
        SampleOptions::default()
            .max_chunk_elements(3)
            .outer_select_ratio(0.0),
    ];
    let mut checked = 0;
    while checked < 60 {
        let case = random_case(&mut rng);
        if case.samples.n == 0 {
            continue;
        }
        let tensors = operands(&mut rng, &case.net, DType::F64);
        let net = case.net.clone();
        let samples = case.samples.clone();
        let reference =
            move |s: &mut EagerSession<'_>, v: &[&EagerTensor]| dense_route(s, &net, &samples, v);
        for options in &settings {
            let options = case.net.along(&case.path, options.clone());
            let refs: Vec<&Tensor> = tensors.iter().collect();
            let plan =
                SampledEinsumPlan::prepare(&refs, &case.net.subscripts(), &case.samples, &options)?;
            let weights = weights_for(&mut rng, plan.output_shape())?;
            let sampled = |s: &mut EagerSession<'_>, v: &[&EagerTensor]| {
                let (y, stats) = plan.execute_eager_with_stats(s, v)?;
                assert_eq!(
                    stats.peak_elements,
                    plan.report().eager_peak_elements,
                    "measured vs planned eager peak\n{case:?}"
                );
                Ok(y)
            };
            let ((y, y_ref), grads) =
                values_and_grads_on_one_tape(&sampled, &reference, &tensors, &weights)?;
            assert_close("y", &y, &y_ref);
            for (i, (g, g_ref)) in grads.iter().enumerate() {
                assert_close(
                    &format!("grad of operand {i} ({:?})", case.net.inputs[i]),
                    g,
                    g_ref,
                );
            }
        }
        checked += 1;
    }
    Ok(())
}

/// Forward mode: the JVP of the sampled plan equals the JVP of the dense route.
#[test]
fn forward_mode_matches_the_dense_route() -> Res<()> {
    let mut rng = Lcg(2552);
    let mut checked = 0;
    while checked < 20 {
        let case = random_case(&mut rng);
        if case.samples.n == 0 {
            continue;
        }
        let tensors = operands(&mut rng, &case.net, DType::F64);
        let tangents = operands(&mut rng, &case.net, DType::F64);
        let options = case.net.along(
            &case.path,
            SampleOptions::default()
                .max_chunk_elements(3)
                .outer_select_ratio(0.0),
        );
        let refs: Vec<&Tensor> = tensors.iter().collect();
        let plan =
            SampledEinsumPlan::prepare(&refs, &case.net.subscripts(), &case.samples, &options)?;
        // Total JVP: the sum over operands of each operand's JVP.
        let jvp = |route: &Route<'_>| -> Res<Vec<f64>> {
            let runtime = ad_runtime()?;
            let mut engine = CpuEngine::serial();
            let vars = tensors
                .iter()
                .map(|t| runtime.variable_from(t.duplicate()?))
                .collect::<tenferro_ad::Result<Vec<_>>>()?;
            let dirs = tangents
                .iter()
                .map(|t| runtime.constant_from(t.duplicate()?))
                .collect::<tenferro_ad::Result<Vec<_>>>()?;
            let parts = runtime.with_eager_session(&mut engine, |s| {
                let refs: Vec<&EagerTensor> = vars.iter().collect();
                let y = route(s, &refs)?;
                let mut parts = Vec::new();
                for (x, d) in vars.iter().zip(&dirs) {
                    if let Some(dy) = s.jvp_optional(&y, x, d)? {
                        parts.push(dy);
                    }
                }
                Ok::<_, tenferro_ad::Error>(parts)
            })?;
            let mut total: Option<Vec<f64>> = None;
            for part in parts {
                let v = part.value()?.as_slice::<f64>()?.to_vec();
                total = Some(match total {
                    None => v,
                    Some(t) => t.iter().zip(&v).map(|(a, b)| a + b).collect(),
                });
            }
            Ok(total.unwrap_or_default())
        };
        let net = case.net.clone();
        let samples = case.samples.clone();
        let got = jvp(&|s, v| plan.execute_eager(s, v))?;
        let want = jvp(&move |s, v| dense_route(s, &net, &samples, v))?;
        assert_close("jvp", &got, &want);
        checked += 1;
    }
    Ok(())
}

#[test]
fn eager_einsum_at_and_conjugation() -> Res<()> {
    let net = Network {
        inputs: vec![vec![0, 1], vec![1, 2]],
        output: vec![0, 2],
        sizes: sizes(&[(0, 3), (1, 4), (2, 5)]),
    };
    let mut rng = Lcg(9);
    let tensors = operands(&mut rng, &net, DType::C64);
    let samples = Samples::new([0, 2], 3, vec![1, 0, 1, 2, 0, 2]);
    let options = SampleOptions::default()
        .einsum(EinsumOptions::default().conj(ConjMask::new([Conj::Yes])))
        .max_chunk_elements(4)
        .outer_select_ratio(0.0);
    let runtime = EagerRuntime::new()?;
    let (got, want) = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(tensors[0].duplicate()?)?;
        let b = s.constant_from(tensors[1].duplicate()?)?;
        let got = s.einsum_at_with(&[&a, &b], &net.subscripts(), &samples, &options)?;
        let a_conj = s.conj(&a)?;
        let want = dense_route(s, &net, &samples, &[&a_conj, &b])?;
        Ok::<_, tenferro_ad::Error>((got, want))
    })?;
    let got = got.value()?;
    let want = want.value()?;
    for (g, w) in got
        .as_slice::<num_complex::Complex64>()?
        .iter()
        .zip(want.as_slice::<num_complex::Complex64>()?)
    {
        assert!((g - w).norm() <= 1e-12 * (1.0 + w.norm()));
    }
    // The concrete layer conjugates the same way.
    let refs: Vec<&Tensor> = tensors.iter().collect();
    let concrete = SampledEinsumPlan::prepare(&refs, &net.subscripts(), &samples, &options)?
        .execute(&refs, &mut CpuEngine::serial())?;
    assert_eq!(
        concrete.as_slice::<num_complex::Complex64>()?.len(),
        got.as_slice::<num_complex::Complex64>()?.len()
    );
    for (g, w) in concrete
        .as_slice::<num_complex::Complex64>()?
        .iter()
        .zip(got.as_slice::<num_complex::Complex64>()?)
    {
        assert!((g - w).norm() <= 1e-12 * (1.0 + w.norm()));
    }
    Ok(())
}

#[test]
fn eager_einsum_at_rejects_a_mask_longer_than_the_inputs() -> Res<()> {
    let runtime = EagerRuntime::new()?;
    let samples = Samples::new([u32::from(b'i')], 1, vec![0]);
    let options = SampleOptions::default().einsum(EinsumOptions::default().conj(ConjMask::new([
        Conj::No,
        Conj::No,
        Conj::Yes,
    ])));
    let result = runtime.with_eager_session(&mut CpuEngine::serial(), |s| {
        let a = s.constant_from(Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64; 4])?)?;
        s.einsum_at_with(&[&a, &a], "ij,jk->ik", &samples, &options)
    });
    assert!(result.is_err());
    Ok(())
}
