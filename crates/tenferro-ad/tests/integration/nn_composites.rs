//! Composite operations (#2010 PR-B1): values, edge-case policy and AD on the
//! eager, traced and concrete-session surfaces, plus the `erf` primitive.

use std::sync::{Arc, OnceLock};

use crate::support::{cpu_runtime, RunTraced};
use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor, TracedTensorAdExt};
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    DType, Runtime, Tensor, TensorSessionOpsExt, TracedTensor, TypedTensor,
    TypedTensorSessionOpsExt,
};
use tenferro_tensor::{BackendSessionHost, Engine};

fn ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()).unwrap())
        .clone()
}

fn runtime() -> &'static Runtime {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    RUNTIME.get_or_init(cpu_runtime)
}

fn t64(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn data64(tensor: &Tensor) -> Vec<f64> {
    tensor.as_slice::<f64>().unwrap().to_vec()
}

fn data32(tensor: &Tensor) -> Vec<f32> {
    tensor.as_slice::<f32>().unwrap().to_vec()
}

fn assert_close(actual: &[f64], expected: &[f64], rtol: f64, atol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if e.is_nan() {
            assert!(a.is_nan(), "{what}[{index}]: expected NaN, got {a}");
        } else if e.is_infinite() {
            assert_eq!(a, e, "{what}[{index}]");
        } else {
            let tol = atol + rtol * e.abs();
            assert!(
                (a - e).abs() <= tol,
                "{what}[{index}]: actual={a}, expected={e}, tol={tol}"
            );
        }
    }
}

fn eager_value(tensor: &EagerTensor) -> Tensor {
    tensor.to_tensor().unwrap()
}

fn traced_value(tensor: &TracedTensor) -> Tensor {
    tensor.run_with(runtime()).unwrap()
}

fn with_session<R>(f: impl FnOnce(&mut dyn Engine) -> R + Send) -> R
where
    R: Send,
{
    let mut backend = CpuBackend::new();
    backend.with_backend_session(f).unwrap()
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Act {
    Sigmoid,
    Silu,
    Softplus,
    Gelu,
    GeluTanh,
}

const ACTS: [Act; 5] = [
    Act::Sigmoid,
    Act::Silu,
    Act::Softplus,
    Act::Gelu,
    Act::GeluTanh,
];

fn sigmoid_ref(x: f64) -> f64 {
    if x > 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// `(f(x), f'(x), f''(x))` in closed form.
fn act_ref(act: Act, x: f64) -> (f64, f64, f64) {
    // `1 - s` is written as `sigmoid(-x)` to avoid cancellation in the tails.
    let s = sigmoid_ref(x);
    let c = sigmoid_ref(-x);
    match act {
        Act::Sigmoid => (s, s * c, s * c * (c - s)),
        Act::Silu => (x * s, s + x * s * c, s * c * (2.0 + x * (c - s))),
        Act::Softplus => (x.max(0.0) + (-x.abs()).exp().ln_1p(), s, s * c),
        Act::Gelu => {
            let cdf = 0.5 * (1.0 + libm::erf(x * std::f64::consts::FRAC_1_SQRT_2));
            let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
            (x * cdf, cdf + x * pdf, pdf * (2.0 - x * x))
        }
        Act::GeluTanh => {
            let a = 0.044_715;
            let u = SQRT_2_OVER_PI * (x + a * x * x * x);
            let du = SQRT_2_OVER_PI * (1.0 + 3.0 * a * x * x);
            let d2u = SQRT_2_OVER_PI * 6.0 * a * x;
            let t = u.tanh();
            let sech2 = 1.0 - t * t;
            let f = 0.5 * x * (1.0 + t);
            let df = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du;
            let d2f = sech2 * du + 0.5 * x * (-2.0 * t * sech2 * du * du + sech2 * d2u);
            (f, df, d2f)
        }
    }
}

fn eager_act(
    s: &mut EagerSession<'_>,
    act: Act,
    x: &EagerTensor,
) -> tenferro_ad::Result<EagerTensor> {
    match act {
        Act::Sigmoid => s.sigmoid(x),
        Act::Silu => s.silu(x),
        Act::Softplus => s.softplus(x),
        Act::Gelu => s.gelu(x),
        Act::GeluTanh => s.gelu_tanh(x),
    }
}

fn traced_act(act: Act, x: &TracedTensor) -> tenferro_ad::Result<TracedTensor> {
    match act {
        Act::Sigmoid => x.sigmoid(),
        Act::Silu => x.silu(),
        Act::Softplus => x.softplus(),
        Act::Gelu => x.gelu(),
        Act::GeluTanh => x.gelu_tanh(),
    }
}

fn session_act(act: Act, x: &Tensor, session: &mut dyn Engine) -> tenferro_tensor::Result<Tensor> {
    match act {
        Act::Sigmoid => x.sigmoid(session),
        Act::Silu => x.silu(session),
        Act::Softplus => x.softplus(session),
        Act::Gelu => x.gelu(session),
        Act::GeluTanh => x.gelu_tanh(session),
    }
}

const ACT_POINTS: [f64; 13] = [
    -30.0, -5.0, -1.0, -1.0e-3, -0.0, 0.0, 1.0e-3, 0.5, 1.0, 2.5, 5.0, 30.0, 20.0,
];

#[test]
fn activations_match_closed_forms_on_all_surfaces_f64() {
    let n = ACT_POINTS.len();
    for act in ACTS {
        let expected: Vec<f64> = ACT_POINTS.iter().map(|&x| act_ref(act, x).0).collect();
        let input = t64(&[n], ACT_POINTS.to_vec());

        let eager = ctx()
            .with_eager_session(|s| {
                let x = s.constant_from(input.duplicate()?)?;
                eager_act(s, act, &x)
            })
            .unwrap();
        let eager = data64(&eager_value(&eager));
        assert_close(&eager, &expected, 1e-14, 1e-300, &format!("eager {act:?}"));

        let traced = traced_act(
            act,
            &TracedTensor::from_tensor_concrete_shape(input.duplicate().unwrap()).unwrap(),
        )
        .unwrap();
        let traced = data64(&traced_value(&traced));
        assert_close(
            &traced,
            &expected,
            4e-16,
            1e-300,
            &format!("traced {act:?}"),
        );

        let concrete = with_session(|s| session_act(act, &input, s)).unwrap();
        assert_close(
            &data64(&concrete),
            &expected,
            4e-16,
            1e-300,
            &format!("session {act:?}"),
        );

        let typed = TypedTensor::<f64>::from_vec_col_major(vec![n], ACT_POINTS.to_vec()).unwrap();
        let typed = with_session(|s| match act {
            Act::Sigmoid => typed.sigmoid(s),
            Act::Silu => typed.silu(s),
            Act::Softplus => typed.softplus(s),
            Act::Gelu => typed.gelu(s),
            Act::GeluTanh => typed.gelu_tanh(s),
        })
        .unwrap();
        assert_close(
            typed.host_data().unwrap(),
            &expected,
            4e-16,
            1e-300,
            &format!("typed {act:?}"),
        );
    }
}

#[test]
fn activations_match_closed_forms_f32_and_saturate_without_overflow() {
    let points = [-100.0_f32, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0];
    for act in ACTS {
        let input = Tensor::from_vec_col_major(vec![points.len()], points.to_vec()).unwrap();
        let out = with_session(|s| session_act(act, &input, s)).unwrap();
        for (&x, &y) in points.iter().zip(&data32(&out)) {
            let expected = act_ref(act, f64::from(x)).0;
            assert!(y.is_finite(), "{act:?}({x}) = {y}");
            let tol = 4.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0e-30) + 1.0e-37;
            assert!(
                (f64::from(y) - expected).abs() <= tol.max(1.0e-6 * expected.abs()),
                "{act:?}({x}) = {y}, expected {expected}"
            );
        }
    }
}

#[test]
fn activation_infinities_and_nan_follow_the_documented_policy() {
    let input = t64(&[3], vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    let out = |act| data64(&with_session(|s| session_act(act, &input, s)).unwrap());
    assert_close(
        &out(Act::Sigmoid),
        &[1.0, 0.0, f64::NAN],
        0.0,
        0.0,
        "sigmoid",
    );
    assert_close(
        &out(Act::Softplus),
        &[f64::INFINITY, 0.0, f64::NAN],
        0.0,
        0.0,
        "softplus",
    );
    // x * gate(x) is inf * 1 at +inf and the IEEE product -inf * 0 = NaN at -inf,
    // matching PyTorch's silu and gelu.
    for act in [Act::Silu, Act::Gelu, Act::GeluTanh] {
        let y = out(act);
        assert_eq!(y[0], f64::INFINITY, "{act:?}(+inf)");
        assert!(y[1].is_nan() && y[2].is_nan(), "{act:?}: {y:?}");
    }
}

/// First derivatives via eager backward and traced `grad`, second
/// derivatives via traced forward-over-reverse, against closed forms.
#[test]
fn activation_first_and_second_derivatives_at_zero_and_extremes() {
    let points = [-1000.0, -20.0, -1.0, 0.0, 0.5, 1.0, 20.0, 1000.0];
    let n = points.len();
    for act in ACTS {
        let first: Vec<f64> = points.iter().map(|&x| act_ref(act, x).1).collect();
        let second: Vec<f64> = points.iter().map(|&x| act_ref(act, x).2).collect();

        // Eager reverse mode.
        let x = EagerTensor::requires_grad_in(t64(&[n], points.to_vec()), ctx()).unwrap();
        let loss = ctx()
            .with_eager_session(|s| {
                let y = eager_act(s, act, &x)?;
                s.reduce_sum(&y, None)
            })
            .unwrap();
        loss.backward().unwrap();
        let grad = x.grad().unwrap().unwrap().to_tensor().unwrap();
        assert_close(
            &data64(&grad),
            &first,
            1e-12,
            1e-300,
            &format!("eager d{act:?}"),
        );

        // Traced reverse mode, then forward-over-reverse for f''.
        let xt = TracedTensor::from_tensor_concrete_shape(t64(&[n], points.to_vec())).unwrap();
        let y = traced_act(act, &xt).unwrap().reduce_sum(None).unwrap();
        let g = y.grad(&xt).unwrap();
        assert_close(
            &data64(&traced_value(&g)),
            &first,
            1e-12,
            1e-300,
            &format!("traced d{act:?}"),
        );
        let ones = TracedTensor::from_tensor_concrete_shape(t64(&[n], vec![1.0; n])).unwrap();
        let h = g.jvp(&xt, &ones).unwrap();
        let h = data64(&traced_value(&h));
        assert!(
            h.iter().all(|v| v.is_finite()),
            "d2 {act:?} not finite: {h:?}"
        );
        assert_close(&h, &second, 1e-12, 1e-15, &format!("traced d2{act:?}"));
    }
}

#[test]
fn softplus_derivatives_at_zero_are_one_half_and_one_quarter() {
    let xt = TracedTensor::from_tensor_concrete_shape(t64(&[], vec![0.0])).unwrap();
    let g = xt.softplus().unwrap().grad(&xt).unwrap();
    let one = TracedTensor::from_tensor_concrete_shape(t64(&[], vec![1.0])).unwrap();
    let h = g.jvp(&xt, &one).unwrap();
    assert_eq!(data64(&traced_value(&g)), vec![0.5]);
    assert_eq!(data64(&traced_value(&h)), vec![0.25]);
}

#[test]
fn activations_reject_non_real_dtypes_on_every_surface() {
    let complex = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 0.0)]).unwrap();
    let int = Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap();
    for act in ACTS {
        for input in [&complex, &int] {
            let err = with_session(|s| session_act(act, input, s)).unwrap_err();
            assert!(
                matches!(err, tenferro_tensor::Error::UnsupportedDType { .. }),
                "{act:?}: {err:?}"
            );
            let traced =
                TracedTensor::from_tensor_concrete_shape(input.duplicate().unwrap()).unwrap();
            assert!(traced_act(act, &traced).is_err(), "traced {act:?}");
            let eager = ctx().with_eager_session(|s| {
                let x = s.constant_from(input.duplicate()?)?;
                eager_act(s, act, &x)
            });
            assert!(eager.is_err(), "eager {act:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// erf
// ---------------------------------------------------------------------------

#[test]
fn erf_gradient_and_second_derivative_match_closed_forms() {
    let points: [f64; 8] = [-6.0, -1.0, -0.25, 0.0, 0.25, 1.0, 6.0, 30.0];
    let n = points.len();
    let c = std::f64::consts::FRAC_2_SQRT_PI;
    let first: Vec<f64> = points.iter().map(|&x| c * (-x * x).exp()).collect();
    let second: Vec<f64> = points
        .iter()
        .map(|&x| -2.0 * x * c * (-x * x).exp())
        .collect();

    let x = EagerTensor::requires_grad_in(t64(&[n], points.to_vec()), ctx()).unwrap();
    let loss = ctx()
        .with_eager_session(|s| {
            let y = s.erf(&x)?;
            s.reduce_sum(&y, None)
        })
        .unwrap();
    loss.backward().unwrap();
    let grad = x.grad().unwrap().unwrap().to_tensor().unwrap();
    assert_close(&data64(&grad), &first, 1e-15, 0.0, "eager d erf");

    let xt = TracedTensor::from_tensor_concrete_shape(t64(&[n], points.to_vec())).unwrap();
    let g = xt
        .erf()
        .unwrap()
        .reduce_sum(None)
        .unwrap()
        .grad(&xt)
        .unwrap();
    assert_close(
        &data64(&traced_value(&g)),
        &first,
        1e-15,
        0.0,
        "traced d erf",
    );
    let ones = TracedTensor::from_tensor_concrete_shape(t64(&[n], vec![1.0; n])).unwrap();
    let h = g.jvp(&xt, &ones).unwrap();
    assert_close(
        &data64(&traced_value(&h)),
        &second,
        1e-14,
        1e-300,
        "traced d2 erf",
    );

    // Forward mode directly.
    let v = TracedTensor::from_tensor_concrete_shape(t64(&[n], vec![2.0; n])).unwrap();
    let jvp = xt.erf().unwrap().jvp(&xt, &v).unwrap();
    let doubled: Vec<f64> = first.iter().map(|d| 2.0 * d).collect();
    assert_close(
        &data64(&traced_value(&jvp)),
        &doubled,
        1e-15,
        0.0,
        "traced jvp erf",
    );
}

#[test]
fn erf_f32_gradient_uses_the_f32_scale() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f32, 1.0]).unwrap(),
        ctx(),
    )
    .unwrap();
    let loss = ctx()
        .with_eager_session(|s| {
            let y = s.erf(&x)?;
            s.reduce_sum(&y, None)
        })
        .unwrap();
    loss.backward().unwrap();
    let grad = x.grad().unwrap().unwrap().to_tensor().unwrap();
    let grad = data32(&grad);
    let c = std::f32::consts::FRAC_2_SQRT_PI;
    assert_eq!(grad[0], c);
    assert!((grad[1] - c * (-1.0_f32).exp()).abs() <= 2.0 * f32::EPSILON);
}

#[test]
fn erf_rejects_complex_input_on_every_surface() {
    let complex = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(0.5, 0.0)]).unwrap();
    let err = with_session(|s| complex.erf(s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::UnsupportedDType { .. }),
        "{err:?}"
    );
    let traced = TracedTensor::from_tensor_concrete_shape(complex.duplicate().unwrap()).unwrap();
    let err = traced.erf().unwrap_err();
    assert_eq!(
        err.kind(),
        tenferro_tensor::ErrorKind::Unsupported,
        "{err:?}"
    );
    let eager = ctx().with_eager_session(|s| {
        let x = s.constant_from(complex.duplicate()?)?;
        s.erf(&x)
    });
    assert!(eager.is_err());
}

/// `strided_fused` has no erf instruction, so the CPU fusion adapter declines
/// a region containing it and the ops run unfused with the same values.
#[test]
fn compiled_cpu_region_with_erf_runs_unfused_with_correct_values() {
    let n = 32 * 1024;
    let data: Vec<f64> = (0..n)
        .map(|i| (i as f64) / (n as f64) * 6.0 - 3.0)
        .collect();
    let x = TracedTensor::from_tensor_concrete_shape(t64(&[n], data.clone())).unwrap();
    let y = x
        .exp()
        .unwrap()
        .neg()
        .unwrap()
        .erf()
        .unwrap()
        .tanh()
        .unwrap();
    let out = data64(&traced_value(&y));
    let expected: Vec<f64> = data.iter().map(|&v| libm::erf(-v.exp()).tanh()).collect();
    assert_close(&out, &expected, 1e-15, 1e-300, "fused-region erf");
}

// ---------------------------------------------------------------------------
// reduce_mean
// ---------------------------------------------------------------------------

#[test]
fn reduce_mean_values_identity_and_empty_policy() {
    let x = t64(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rows = with_session(|s| x.reduce_mean(Some(&[1]), s)).unwrap();
    assert_eq!(data64(&rows), vec![3.0, 4.0]);
    let all = with_session(|s| x.reduce_mean(None, s)).unwrap();
    assert_eq!(data64(&all), vec![3.5]);
    let identity = with_session(|s| x.reduce_mean(Some(&[]), s)).unwrap();
    assert_eq!(data64(&identity), data64(&x));

    let empty = t64(&[0, 3], vec![]);
    let mean = with_session(|s| empty.reduce_mean(Some(&[0]), s)).unwrap();
    assert_eq!(mean.shape(), &[3]);
    assert!(data64(&mean).iter().all(|v| v.is_nan()));
    let traced = TracedTensor::from_tensor_concrete_shape(empty.duplicate().unwrap()).unwrap();
    let traced = traced_value(&traced.reduce_mean(None).unwrap());
    assert_eq!(traced.shape(), &[] as &[usize]);
    assert!(data64(&traced)[0].is_nan());

    let complex = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    )
    .unwrap();
    let mean = with_session(|s| complex.reduce_mean(None, s)).unwrap();
    assert_eq!(
        mean.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.0, -1.0)]
    );

    let int = Tensor::from_vec_col_major(vec![2], vec![1_i32, 2]).unwrap();
    let err = with_session(|s| int.reduce_mean(None, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::UnsupportedDType { .. }),
        "{err:?}"
    );
    let err = with_session(|s| x.reduce_mean(Some(&[2]), s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let err = with_session(|s| x.reduce_mean(Some(&[1, 1]), s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
}

#[test]
fn reduce_mean_gradient_is_one_over_count() {
    let x = EagerTensor::requires_grad_in(t64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]), ctx()).unwrap();
    let loss = ctx()
        .with_eager_session(|s| {
            let m = s.reduce_mean(&x, Some(&[0]))?;
            s.reduce_sum(&m, None)
        })
        .unwrap();
    loss.backward().unwrap();
    let grad = x.grad().unwrap().unwrap().to_tensor().unwrap();
    assert_eq!(data64(&grad), vec![0.5; 4]);
}

// ---------------------------------------------------------------------------
// softmax / log_softmax
// ---------------------------------------------------------------------------

/// Host reference along `axis` of a column-major rank-2 tensor; masked-out
/// entries are excluded; an empty slice gives 0 / -inf.
fn softmax_ref(
    shape: [usize; 2],
    x: &[f64],
    mask: Option<&[bool]>,
    axis: usize,
    log: bool,
) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    let (len, other) = if axis == 0 {
        (shape[0], shape[1])
    } else {
        (shape[1], shape[0])
    };
    for o in 0..other {
        let idx = |k: usize| {
            if axis == 0 {
                k + shape[0] * o
            } else {
                o + shape[0] * k
            }
        };
        let live = |k: usize| mask.is_none_or(|m| m[idx(k)]);
        let max = (0..len)
            .filter(|&k| live(k))
            .map(|k| x[idx(k)])
            .fold(f64::NEG_INFINITY, f64::max);
        let max = if max == f64::NEG_INFINITY { 0.0 } else { max };
        let sum: f64 = (0..len)
            .filter(|&k| live(k))
            .map(|k| (x[idx(k)] - max).exp())
            .sum();
        let sum = if sum == 0.0 { 1.0 } else { sum };
        for k in 0..len {
            let v = if live(k) {
                x[idx(k)] - max
            } else {
                f64::NEG_INFINITY
            };
            out[idx(k)] = if log { v - sum.ln() } else { v.exp() / sum };
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct SoftmaxCase {
    log: bool,
    masked: bool,
}

const SOFTMAX_CASES: [SoftmaxCase; 4] = [
    SoftmaxCase {
        log: false,
        masked: false,
    },
    SoftmaxCase {
        log: true,
        masked: false,
    },
    SoftmaxCase {
        log: false,
        masked: true,
    },
    SoftmaxCase {
        log: true,
        masked: true,
    },
];

fn eager_softmax(
    s: &mut EagerSession<'_>,
    case: SoftmaxCase,
    x: &EagerTensor,
    mask: &EagerTensor,
    axis: usize,
) -> tenferro_ad::Result<EagerTensor> {
    match (case.log, case.masked) {
        (false, false) => s.softmax(x, axis),
        (true, false) => s.log_softmax(x, axis),
        (false, true) => s.masked_softmax(x, mask, axis),
        (true, true) => s.masked_log_softmax(x, mask, axis),
    }
}

fn traced_softmax(
    case: SoftmaxCase,
    x: &TracedTensor,
    mask: &TracedTensor,
    axis: usize,
) -> tenferro_ad::Result<TracedTensor> {
    match (case.log, case.masked) {
        (false, false) => x.softmax(axis),
        (true, false) => x.log_softmax(axis),
        (false, true) => x.masked_softmax(mask, axis),
        (true, true) => x.masked_log_softmax(mask, axis),
    }
}

fn session_softmax(
    case: SoftmaxCase,
    x: &Tensor,
    mask: &Tensor,
    axis: usize,
    s: &mut dyn Engine,
) -> tenferro_tensor::Result<Tensor> {
    match (case.log, case.masked) {
        (false, false) => x.softmax(axis, s),
        (true, false) => x.log_softmax(axis, s),
        (false, true) => x.masked_softmax(mask, axis, s),
        (true, true) => x.masked_log_softmax(mask, axis, s),
    }
}

/// Run one softmax case on every surface and check it against the reference.
fn check_softmax_surfaces(shape: [usize; 2], x: Vec<f64>, mask: Vec<bool>, axis: usize) {
    for case in SOFTMAX_CASES {
        let reference = softmax_ref(shape, &x, case.masked.then_some(&mask[..]), axis, case.log);
        let xt = t64(&shape, x.clone());
        let mt = Tensor::from_vec_col_major(shape.to_vec(), mask.clone()).unwrap();

        let concrete = with_session(|s| session_softmax(case, &xt, &mt, axis, s)).unwrap();
        assert_close(
            &data64(&concrete),
            &reference,
            1e-15,
            1e-300,
            &format!("session {case:?}"),
        );

        let eager = ctx()
            .with_eager_session(|s| {
                let x = s.constant_from(xt.duplicate()?)?;
                let m = s.constant_from(mt.duplicate()?)?;
                eager_softmax(s, case, &x, &m, axis)
            })
            .unwrap();
        assert_close(
            &data64(&eager_value(&eager)),
            &reference,
            1e-15,
            1e-300,
            &format!("eager {case:?}"),
        );

        let x_tr = TracedTensor::from_tensor_concrete_shape(xt.duplicate().unwrap()).unwrap();
        let m_tr = TracedTensor::from_tensor_concrete_shape(mt.duplicate().unwrap()).unwrap();
        let traced = traced_softmax(case, &x_tr, &m_tr, axis).unwrap();
        assert_close(
            &data64(&traced_value(&traced)),
            &reference,
            1e-15,
            1e-300,
            &format!("traced {case:?}"),
        );
    }
}

#[test]
fn softmax_family_matches_reference_on_all_surfaces() {
    let x = vec![
        0.5, -1.0, 2.0, 3.0, 0.0, -2.5, 1.5, 1.5, 7.0, -0.25, 4.0, 0.125,
    ];
    let mask = vec![
        true, false, true, true, true, false, false, true, true, true, true, false,
    ];
    for axis in [0, 1] {
        check_softmax_surfaces([3, 4], x.clone(), mask.clone(), axis);
    }
}

#[test]
fn all_masked_slice_is_zero_or_neg_inf_with_a_finite_zero_gradient() {
    // Column 1 is fully masked; column 0 is partially masked.
    let x = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, -1.0];
    let mask = vec![true, false, true, false, false, false];
    check_softmax_surfaces([3, 2], x.clone(), mask.clone(), 0);

    for log in [false, true] {
        let xe = EagerTensor::requires_grad_in(t64(&[3, 2], x.clone()), ctx()).unwrap();
        let y = ctx()
            .with_eager_session(|s| {
                let m = s.constant_from(Tensor::from_vec_col_major(vec![3, 2], mask.clone())?)?;
                if log {
                    s.masked_log_softmax(&xe, &m, 0)
                } else {
                    s.masked_softmax(&xe, &m, 0)
                }
            })
            .unwrap();
        let values = data64(&eager_value(&y));
        let fill = if log { f64::NEG_INFINITY } else { 0.0 };
        assert_eq!(&values[3..], &[fill; 3], "all-masked column (log={log})");

        let ones = ctx().constant_from(t64(&[3, 2], vec![1.0; 6])).unwrap();
        let grad = ctx().vjp(&y, &xe, &ones).unwrap().to_tensor().unwrap();
        let grad = data64(&grad);
        assert!(
            grad.iter().all(|g| g.is_finite()),
            "gradient not finite: {grad:?}"
        );
        assert_eq!(&grad[3..], &[0.0; 3], "all-masked gradient (log={log})");
        assert_eq!(grad[1], 0.0, "masked entry gradient (log={log})");

        // The traced transform agrees.
        let xt = TracedTensor::from_tensor_concrete_shape(t64(&[3, 2], x.clone())).unwrap();
        let mt = TracedTensor::from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![3, 2], mask.clone()).unwrap(),
        )
        .unwrap();
        let yt = if log {
            xt.masked_log_softmax(&mt, 0)
        } else {
            xt.masked_softmax(&mt, 0)
        }
        .unwrap();
        // sum over the finite (column 0, unmasked) outputs plus a zero-weighted rest.
        let w = TracedTensor::from_tensor_concrete_shape(t64(
            &[3, 2],
            vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ))
        .unwrap();
        let zero = TracedTensor::from_tensor_concrete_shape(t64(&[], vec![0.0])).unwrap();
        let mask_w = TracedTensor::from_tensor_concrete_shape(
            Tensor::from_vec_col_major(vec![3, 2], vec![true, false, true, false, false, false])
                .unwrap(),
        )
        .unwrap();
        let picked = TracedTensor::where_select(&mask_w, &yt, &zero).unwrap();
        let loss = (&picked * &w).unwrap().reduce_sum(None).unwrap();
        let gt = data64(&traced_value(&loss.grad(&xt).unwrap()));
        assert!(gt.iter().all(|g| g.is_finite()), "traced gradient: {gt:?}");
        assert_eq!(&gt[3..], &[0.0; 3]);
        assert_eq!(gt[1], 0.0);
    }
}

#[test]
fn unmasked_all_neg_inf_slice_follows_the_all_masked_policy() {
    let x = t64(
        &[2, 2],
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY, 1.0, 1.0],
    );
    let sm = with_session(|s| x.softmax(0, s)).unwrap();
    assert_eq!(data64(&sm), vec![0.0, 0.0, 0.5, 0.5]);
    let lsm = with_session(|s| x.log_softmax(0, s)).unwrap();
    let ln_half = -std::f64::consts::LN_2;
    assert_eq!(
        data64(&lsm),
        vec![f64::NEG_INFINITY, f64::NEG_INFINITY, ln_half, ln_half]
    );

    let xe = EagerTensor::requires_grad_in(x.duplicate().unwrap(), ctx()).unwrap();
    let y = ctx().with_eager_session(|s| s.softmax(&xe, 0)).unwrap();
    let ones = ctx().constant_from(t64(&[2, 2], vec![1.0; 4])).unwrap();
    let grad = data64(&ctx().vjp(&y, &xe, &ones).unwrap().to_tensor().unwrap());
    assert!(grad.iter().all(|g| g.is_finite()), "{grad:?}");
    assert_eq!(&grad[..2], &[0.0, 0.0]);
}

#[test]
fn softmax_nan_and_pos_inf_poison_only_their_slice() {
    // Row 0 is [NaN, 1, 2] and row 1 is [1, +inf, 3] (column-major [2, 3]).
    let x = t64(&[2, 3], vec![f64::NAN, 1.0, 1.0, f64::INFINITY, 2.0, 3.0]);
    for log in [false, true] {
        let y = with_session(|s| {
            if log {
                x.log_softmax(1, s)
            } else {
                x.softmax(1, s)
            }
        })
        .unwrap();
        let y = data64(&y);
        for (index, value) in y.iter().enumerate() {
            assert!(value.is_nan(), "log={log} index {index}: {value}");
        }
    }
    let finite_row = t64(&[2, 3], vec![1.0, f64::NAN, 1.0, 0.0, 1.0, 0.0]);
    let y = data64(&with_session(|s| finite_row.softmax(1, s)).unwrap());
    let third = 1.0 / 3.0;
    assert_close(
        &[y[0], y[2], y[4]],
        &[third, third, third],
        1e-15,
        0.0,
        "finite row",
    );
    assert!(y[1].is_nan() && y[3].is_nan() && y[5].is_nan());
}

#[test]
fn softmax_over_an_empty_axis_returns_an_empty_result() {
    for (shape, axis) in [([0_usize, 3], 0), ([3, 0], 0), ([2, 0], 1)] {
        let x = t64(&shape, vec![]);
        let mask = Tensor::from_vec_col_major(shape.to_vec(), Vec::<bool>::new()).unwrap();
        for case in SOFTMAX_CASES {
            let y = with_session(|s| session_softmax(case, &x, &mask, axis, s)).unwrap();
            assert_eq!(y.shape(), &shape, "{case:?}");
            let xt = TracedTensor::from_tensor_concrete_shape(x.duplicate().unwrap()).unwrap();
            let mt = TracedTensor::from_tensor_concrete_shape(mask.duplicate().unwrap()).unwrap();
            let yt = traced_value(&traced_softmax(case, &xt, &mt, axis).unwrap());
            assert_eq!(yt.shape(), &shape, "traced {case:?}");
        }
    }
}

#[test]
fn softmax_gradient_matches_the_closed_form() {
    // d/dx_j sum_i w_i softmax_i = s_j (w_j - sum_i w_i s_i).
    let x = vec![0.3, -1.2, 2.0, 0.7];
    let w = vec![1.0, -2.0, 0.5, 3.0];
    let s = softmax_ref([4, 1], &x, None, 0, false);
    let dot: f64 = s.iter().zip(&w).map(|(a, b)| a * b).sum();
    let expected: Vec<f64> = s.iter().zip(&w).map(|(sj, wj)| sj * (wj - dot)).collect();
    let xe = EagerTensor::requires_grad_in(t64(&[4], x.clone()), ctx()).unwrap();
    let y = ctx().with_eager_session(|s| s.softmax(&xe, 0)).unwrap();
    let ct = ctx().constant_from(t64(&[4], w.clone())).unwrap();
    let grad = data64(&ctx().vjp(&y, &xe, &ct).unwrap().to_tensor().unwrap());
    assert_close(&grad, &expected, 1e-14, 1e-16, "softmax vjp");

    // log_softmax: d/dx_j sum_i w_i lsm_i = w_j - s_j sum_i w_i.
    let total: f64 = w.iter().sum();
    let expected: Vec<f64> = s.iter().zip(&w).map(|(sj, wj)| wj - sj * total).collect();
    let y = ctx().with_eager_session(|s| s.log_softmax(&xe, 0)).unwrap();
    let grad = data64(&ctx().vjp(&y, &xe, &ct).unwrap().to_tensor().unwrap());
    assert_close(&grad, &expected, 1e-14, 1e-16, "log_softmax vjp");
}

#[test]
fn softmax_validates_axis_dtype_and_mask() {
    let x = t64(&[2, 2], vec![1.0; 4]);
    let bool_mask = Tensor::from_vec_col_major(vec![2, 2], vec![true; 4]).unwrap();
    let err = with_session(|s| x.softmax(2, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let float_mask = t64(&[2, 2], vec![1.0; 4]);
    let err = with_session(|s| x.masked_softmax(&float_mask, 0, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let wide_mask = Tensor::from_vec_col_major(vec![3, 2], vec![true; 6]).unwrap();
    let err = with_session(|s| x.masked_softmax(&wide_mask, 0, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    // A row mask broadcasts across the leading axis.
    let row_mask = Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap();
    let y = with_session(|s| x.masked_softmax(&row_mask, 1, s)).unwrap();
    assert_eq!(data64(&y), vec![1.0, 1.0, 0.0, 0.0]);
    let int = Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap();
    let err = with_session(|s| int.softmax(0, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::UnsupportedDType { .. }),
        "{err:?}"
    );
    let _ = bool_mask;
}

#[test]
fn softmax_f32_matches_reference() {
    let x: Vec<f32> = vec![0.5, -1.0, 2.0, 30.0, -30.0, 0.0];
    let xt = Tensor::from_vec_col_major(vec![3, 2], x.clone()).unwrap();
    let y = data32(&with_session(|s| xt.softmax(0, s)).unwrap());
    let reference = softmax_ref(
        [3, 2],
        &x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        None,
        0,
        false,
    );
    for (a, e) in y.iter().zip(&reference) {
        assert!(
            (f64::from(*a) - e).abs() <= 4.0 * f64::from(f32::EPSILON) * e.max(1e-30),
            "{a} vs {e}"
        );
    }
}

// ---------------------------------------------------------------------------
// layer_norm / rms_norm
// ---------------------------------------------------------------------------

fn norm_ref(
    shape: [usize; 2],
    x: &[f64],
    axis: usize,
    weight: Option<&[f64]>,
    bias: Option<&[f64]>,
    eps: f64,
    rms: bool,
) -> Vec<f64> {
    let mut out = vec![0.0; x.len()];
    let (len, other) = if axis == 0 {
        (shape[0], shape[1])
    } else {
        (shape[1], shape[0])
    };
    for o in 0..other {
        let idx = |k: usize| {
            if axis == 0 {
                k + shape[0] * o
            } else {
                o + shape[0] * k
            }
        };
        let n = len as f64;
        let mean = if rms {
            0.0
        } else {
            (0..len).map(|k| x[idx(k)]).sum::<f64>() / n
        };
        let var = (0..len).map(|k| (x[idx(k)] - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for k in 0..len {
            let mut v = (x[idx(k)] - mean) * inv;
            if let Some(w) = weight {
                v *= w[k];
            }
            if let Some(b) = bias {
                v += b[k];
            }
            out[idx(k)] = v;
        }
    }
    out
}

#[test]
fn layer_norm_and_rms_norm_match_reference_on_all_surfaces() {
    let x = vec![
        0.5, -1.0, 2.0, 3.0, 0.0, -2.5, 1.5, 1.5, 7.0, -0.25, 4.0, 0.125,
    ];
    for axis in [0, 1] {
        let len = [3, 4][axis];
        let w: Vec<f64> = (0..len).map(|k| 0.5 + k as f64).collect();
        let b: Vec<f64> = (0..len).map(|k| 1.0 - 0.25 * k as f64).collect();
        for rms in [false, true] {
            let reference = norm_ref([3, 4], &x, axis, Some(&w), Some(&b), 1e-5, rms);
            let (xt, wt, bt) = (
                t64(&[3, 4], x.clone()),
                t64(&[len], w.clone()),
                t64(&[len], b.clone()),
            );

            let concrete = with_session(|s| {
                if rms {
                    xt.rms_norm(axis, Some(&wt), Some(&bt), 1e-5, s)
                } else {
                    xt.layer_norm(axis, Some(&wt), Some(&bt), 1e-5, s)
                }
            })
            .unwrap();
            assert_close(
                &data64(&concrete),
                &reference,
                1e-14,
                1e-15,
                &format!("session rms={rms}"),
            );

            let eager = ctx()
                .with_eager_session(|s| {
                    let x = s.constant_from(xt.duplicate()?)?;
                    let w = s.constant_from(wt.duplicate()?)?;
                    let b = s.constant_from(bt.duplicate()?)?;
                    if rms {
                        s.rms_norm(&x, axis, Some(&w), Some(&b), 1e-5)
                    } else {
                        s.layer_norm(&x, axis, Some(&w), Some(&b), 1e-5)
                    }
                })
                .unwrap();
            assert_close(
                &data64(&eager_value(&eager)),
                &reference,
                1e-14,
                1e-15,
                &format!("eager rms={rms}"),
            );

            let x_tr = TracedTensor::from_tensor_concrete_shape(xt.duplicate().unwrap()).unwrap();
            let w_tr = TracedTensor::from_tensor_concrete_shape(wt.duplicate().unwrap()).unwrap();
            let b_tr = TracedTensor::from_tensor_concrete_shape(bt.duplicate().unwrap()).unwrap();
            let traced = if rms {
                x_tr.rms_norm(axis, Some(&w_tr), Some(&b_tr), 1e-5)
            } else {
                x_tr.layer_norm(axis, Some(&w_tr), Some(&b_tr), 1e-5)
            }
            .unwrap();
            assert_close(
                &data64(&traced_value(&traced)),
                &reference,
                1e-14,
                1e-15,
                &format!("traced rms={rms}"),
            );
        }
    }
}

/// Central finite differences of `f` at `x` along every coordinate.
fn finite_difference(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    (0..x.len())
        .map(|i| {
            let mut plus = x.to_vec();
            let mut minus = x.to_vec();
            plus[i] += h;
            minus[i] -= h;
            (f(&plus) - f(&minus)) / (2.0 * h)
        })
        .collect()
}

#[test]
fn norm_gradients_match_finite_differences_including_a_zero_variance_slice() {
    // Column 1 is constant: zero variance.
    let x = vec![0.4, -1.3, 2.2, 0.9, 0.9, 0.9];
    let w = vec![0.7, -1.1, 0.3, 2.0, 0.5, -0.4];
    let gamma = vec![1.5, 0.5, -1.0];
    let beta = vec![0.25, -0.5, 1.0];
    for rms in [false, true] {
        let eps = 1e-3;
        let loss_ref = |xs: &[f64]| {
            norm_ref([3, 2], xs, 0, Some(&gamma), Some(&beta), eps, rms)
                .iter()
                .zip(&w)
                .map(|(a, b)| a * b)
                .sum::<f64>()
        };
        let expected = finite_difference(loss_ref, &x);

        let xe = EagerTensor::requires_grad_in(t64(&[3, 2], x.clone()), ctx()).unwrap();
        let ge = EagerTensor::requires_grad_in(t64(&[3], gamma.clone()), ctx()).unwrap();
        let y = ctx()
            .with_eager_session(|s| {
                let b = s.constant_from(t64(&[3], beta.clone()))?;
                if rms {
                    s.rms_norm(&xe, 0, Some(&ge), Some(&b), eps)
                } else {
                    s.layer_norm(&xe, 0, Some(&ge), Some(&b), eps)
                }
            })
            .unwrap();
        let values = data64(&eager_value(&y));
        if !rms {
            // A zero-variance slice normalizes to exactly the bias.
            assert_close(&values[3..], &beta, 0.0, 1e-12, "zero-variance layer_norm");
        }
        let ct = ctx().constant_from(t64(&[3, 2], w.clone())).unwrap();
        let grad = data64(&ctx().vjp(&y, &xe, &ct).unwrap().to_tensor().unwrap());
        assert!(grad.iter().all(|g| g.is_finite()), "{grad:?}");
        assert_close(
            &grad,
            &expected,
            1e-6,
            1e-7,
            &format!("norm grad rms={rms}"),
        );

        let ggamma = data64(&ctx().vjp(&y, &ge, &ct).unwrap().to_tensor().unwrap());
        let gamma_fd = finite_difference(
            |gs: &[f64]| {
                norm_ref([3, 2], &x, 0, Some(gs), Some(&beta), eps, rms)
                    .iter()
                    .zip(&w)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            },
            &gamma,
        );
        assert_close(
            &ggamma,
            &gamma_fd,
            1e-6,
            1e-7,
            &format!("weight grad rms={rms}"),
        );
    }
}

#[test]
fn norms_validate_eps_and_affine_parameters() {
    let x = t64(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    for eps in [-1e-5, f64::NAN, f64::INFINITY] {
        let err = with_session(|s| x.layer_norm(0, None, None, eps, s)).unwrap_err();
        assert!(
            matches!(err, tenferro_tensor::Error::Validation { .. }),
            "{eps}: {err:?}"
        );
        let err = with_session(|s| x.rms_norm(0, None, None, eps, s)).unwrap_err();
        assert!(
            matches!(err, tenferro_tensor::Error::Validation { .. }),
            "{eps}: {err:?}"
        );
    }
    let bad_weight = t64(&[3], vec![1.0; 3]);
    let err = with_session(|s| x.layer_norm(0, Some(&bad_weight), None, 1e-5, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let f32_bias = Tensor::from_vec_col_major(vec![2], vec![1.0_f32; 2]).unwrap();
    let err = with_session(|s| x.rms_norm(0, None, Some(&f32_bias), 1e-5, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let err = with_session(|s| x.layer_norm(5, None, None, 1e-5, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );

    // eps = 0 on a constant slice is 0/0, like PyTorch.
    let constant = t64(&[2], vec![3.0, 3.0]);
    let y = with_session(|s| constant.layer_norm(0, None, None, 0.0, s)).unwrap();
    assert!(data64(&y).iter().all(|v| v.is_nan()));
    // An empty normalized axis returns an empty result.
    let empty = t64(&[0, 2], vec![]);
    let y = with_session(|s| empty.rms_norm(0, None, None, 1e-6, s)).unwrap();
    assert_eq!(y.shape(), &[0, 2]);
}

// ---------------------------------------------------------------------------
// take_along_axis
// ---------------------------------------------------------------------------

/// NumPy `take_along_axis` reference for column-major rank-3 input.
fn take_ref(
    shape: [usize; 3],
    x: &[f64],
    idx_shape: [usize; 3],
    idx: &[i64],
    axis: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut out_shape = shape;
    out_shape[axis] = idx_shape[axis];
    let count: usize = out_shape.iter().product();
    let mut out = vec![0.0; count];
    for (linear, slot) in out.iter_mut().enumerate() {
        let c = [
            linear % out_shape[0],
            (linear / out_shape[0]) % out_shape[1],
            linear / (out_shape[0] * out_shape[1]),
        ];
        let ic: Vec<usize> = (0..3)
            .map(|d| if idx_shape[d] == 1 { 0 } else { c[d] })
            .collect();
        let picked = idx[ic[0] + idx_shape[0] * (ic[1] + idx_shape[1] * ic[2])] as usize;
        let mut src = c;
        src[axis] = picked;
        *slot = x[src[0] + shape[0] * (src[1] + shape[1] * src[2])];
    }
    (out_shape.to_vec(), out)
}

#[test]
fn take_along_axis_rows_columns_and_batches_match_numpy() {
    let shape = [3, 3, 2];
    let x: Vec<f64> = (0..18).map(|v| f64::from(v) * 1.5 - 4.0).collect();
    let cases: Vec<([usize; 3], Vec<i64>, usize)> = vec![
        // Per-batch row pivots: out[i, j, b] = x[p[i, b], j, b].
        ([3, 1, 2], vec![2, 0, 1, 1, 1, 0], 0),
        // Per-batch column pivots: out[i, j, b] = x[i, p[j, b], b].
        ([1, 3, 2], vec![1, 2, 0, 0, 0, 2], 1),
        // Fully batch-varying indices, fewer rows than the operand.
        ([2, 3, 2], vec![0, 2, 1, 1, 2, 0, 1, 0, 0, 0, 2, 2], 0),
        // Indices shared across batches and columns.
        ([4, 1, 1], vec![2, 2, 0, 1], 0),
        // Gather along the trailing (batch) axis.
        ([3, 3, 1], vec![1, 0, 1, 0, 1, 0, 1, 1, 0], 2),
    ];
    for (idx_shape, idx, axis) in cases {
        let (out_shape, expected) = take_ref(shape, &x, idx_shape, &idx, axis);
        let xt = t64(&shape, x.clone());
        let it64 = Tensor::from_vec_col_major(idx_shape.to_vec(), idx.clone()).unwrap();
        let it32 = Tensor::from_vec_col_major(
            idx_shape.to_vec(),
            idx.iter().map(|&v| v as i32).collect::<Vec<_>>(),
        )
        .unwrap();
        for it in [&it64, &it32] {
            let y = with_session(|s| xt.take_along_axis(it, axis, s)).unwrap();
            assert_eq!(y.shape(), &out_shape[..], "{idx_shape:?} axis {axis}");
            assert_eq!(
                data64(&y),
                expected,
                "session {idx_shape:?} axis {axis} {:?}",
                it.dtype()
            );
        }

        let eager = ctx()
            .with_eager_session(|s| {
                let x = s.constant_from(xt.duplicate()?)?;
                let i = s.constant_from(it64.duplicate()?)?;
                s.take_along_axis(&x, &i, axis)
            })
            .unwrap();
        assert_eq!(
            data64(&eager_value(&eager)),
            expected,
            "eager {idx_shape:?}"
        );

        let x_tr = TracedTensor::from_tensor_concrete_shape(xt.duplicate().unwrap()).unwrap();
        let i_tr = TracedTensor::from_tensor_concrete_shape(it32.duplicate().unwrap()).unwrap();
        let traced = x_tr.take_along_axis(&i_tr, axis).unwrap();
        assert_eq!(
            data64(&traced_value(&traced)),
            expected,
            "traced {idx_shape:?}"
        );
    }
}

#[test]
fn take_along_axis_gradient_scatters_back_to_the_operand() {
    let shape = [3, 2, 2];
    let x: Vec<f64> = (0..12).map(f64::from).collect();
    // Row 0 picked twice in batch 0 so the gradient accumulates.
    let idx = vec![0_i64, 0, 2, 1];
    let idx_shape = [2, 1, 2];
    let w: Vec<f64> = (0..8).map(|v| 1.0 + f64::from(v)).collect();
    let loss_ref = |xs: &[f64]| {
        take_ref(shape, xs, idx_shape, &idx, 0)
            .1
            .iter()
            .zip(&w)
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };
    let expected = finite_difference(loss_ref, &x);

    let xe = EagerTensor::requires_grad_in(t64(&shape, x.clone()), ctx()).unwrap();
    let y = ctx()
        .with_eager_session(|s| {
            let i =
                s.constant_from(Tensor::from_vec_col_major(idx_shape.to_vec(), idx.clone())?)?;
            s.take_along_axis(&xe, &i, 0)
        })
        .unwrap();
    let ct = ctx().constant_from(t64(&[2, 2, 2], w.clone())).unwrap();
    let grad = data64(&ctx().vjp(&y, &xe, &ct).unwrap().to_tensor().unwrap());
    assert_close(&grad, &expected, 1e-8, 1e-8, "take_along_axis grad");
}

#[test]
fn take_along_axis_validates_shapes_and_dtypes() {
    let x = t64(&[2, 3], vec![1.0; 6]);
    let rank1 = Tensor::from_vec_col_major(vec![2], vec![0_i64, 1]).unwrap();
    let err = with_session(|s| x.take_along_axis(&rank1, 0, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let bad_extent = Tensor::from_vec_col_major(vec![1, 2], vec![0_i64, 1]).unwrap();
    let err = with_session(|s| x.take_along_axis(&bad_extent, 0, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let float_idx = t64(&[1, 3], vec![0.0; 3]);
    let err = with_session(|s| x.take_along_axis(&float_idx, 0, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::UnsupportedDType { .. }),
        "{err:?}"
    );
    let ok_idx = Tensor::from_vec_col_major(vec![1, 3], vec![0_i64, 1, 0]).unwrap();
    let err = with_session(|s| x.take_along_axis(&ok_idx, 2, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    let empty = t64(&[0, 3], vec![]);
    let err = with_session(|s| empty.take_along_axis(&ok_idx, 0, s)).unwrap_err();
    assert!(
        matches!(err, tenferro_tensor::Error::Validation { .. }),
        "{err:?}"
    );
    // Zero indices along the axis give an empty result.
    let none = Tensor::from_vec_col_major(vec![0, 3], Vec::<i64>::new()).unwrap();
    let y = with_session(|s| x.take_along_axis(&none, 0, s)).unwrap();
    assert_eq!(y.shape(), &[0, 3]);
    assert_eq!(DType::F64, y.dtype());
}

#[test]
fn reduce_mean_covers_complex_dtypes_on_traced_and_session_surfaces() {
    let c64 = Tensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    )
    .unwrap();
    let traced = TracedTensor::from_tensor_concrete_shape(c64).unwrap();
    let mean = traced_value(&traced.reduce_mean(None).unwrap());
    assert_eq!(
        mean.as_slice::<Complex64>().unwrap(),
        &[Complex64::new(2.0, -1.0)]
    );

    let c32 = Tensor::from_vec_col_major(
        vec![2],
        vec![
            num_complex::Complex32::new(1.0, 1.0),
            num_complex::Complex32::new(2.0, 3.0),
        ],
    )
    .unwrap();
    let mean = with_session(|s| c32.reduce_mean(Some(&[0]), s)).unwrap();
    assert_eq!(
        mean.as_slice::<num_complex::Complex32>().unwrap(),
        &[num_complex::Complex32::new(1.5, 2.0)]
    );
}

#[test]
fn traced_composites_reject_symbolic_shapes_at_graph_build() {
    let x = TracedTensor::input_symbolic_shape(DType::F64, 2).unwrap();
    for result in [
        x.sigmoid(),
        x.gelu(),
        x.softmax(0),
        x.reduce_mean(None),
        x.layer_norm(0, None, None, 1e-5),
    ] {
        let err = result.unwrap_err();
        assert_eq!(
            err.kind(),
            tenferro_tensor::ErrorKind::Validation(
                tenferro_tensor::ValidationKind::InvalidArgument
            ),
            "{err:?}"
        );
    }
}

/// The typed concrete-session surface forwards to the same composites.
#[test]
fn typed_session_composites_match_the_erased_surface() {
    let data = vec![0.5, -1.0, 2.0, 3.0, 0.0, -2.5];
    let erased = t64(&[3, 2], data.clone());
    let typed = TypedTensor::<f64>::from_vec_col_major(vec![3, 2], data).unwrap();
    let mask_values = vec![true, false, true, false, false, false];
    let erased_mask = Tensor::from_vec_col_major(vec![3, 2], mask_values.clone()).unwrap();
    let typed_mask = TypedTensor::<bool>::from_vec_col_major(vec![3, 2], mask_values).unwrap();
    let erased_w = t64(&[3], vec![1.0, 2.0, 0.5]);
    let typed_w = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, 2.0, 0.5]).unwrap();
    let erased_b = t64(&[3], vec![0.1, 0.2, 0.3]);
    let typed_b = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![0.1, 0.2, 0.3]).unwrap();

    let pairs: Vec<(Tensor, Vec<f64>)> = with_session(|s| -> tenferro_tensor::Result<_> {
        Ok(vec![
            (
                erased.reduce_mean(Some(&[0]), s)?,
                typed.reduce_mean(Some(&[0]), s)?.host_data()?.to_vec(),
            ),
            (
                erased.softmax(0, s)?,
                typed.softmax(0, s)?.host_data()?.to_vec(),
            ),
            (
                erased.log_softmax(1, s)?,
                typed.log_softmax(1, s)?.host_data()?.to_vec(),
            ),
            (
                erased.masked_softmax(&erased_mask, 0, s)?,
                typed
                    .masked_softmax(&typed_mask, 0, s)?
                    .host_data()?
                    .to_vec(),
            ),
            (
                erased.masked_log_softmax(&erased_mask, 0, s)?,
                typed
                    .masked_log_softmax(&typed_mask, 0, s)?
                    .host_data()?
                    .to_vec(),
            ),
            (
                erased.layer_norm(0, Some(&erased_w), Some(&erased_b), 1e-5, s)?,
                typed
                    .layer_norm(0, Some(&typed_w), Some(&typed_b), 1e-5, s)?
                    .host_data()?
                    .to_vec(),
            ),
            (
                erased.rms_norm(0, Some(&erased_w), None, 1e-5, s)?,
                typed
                    .rms_norm(0, Some(&typed_w), None, 1e-5, s)?
                    .host_data()?
                    .to_vec(),
            ),
        ])
    })
    .unwrap();
    for (index, (erased, typed)) in pairs.iter().enumerate() {
        assert_close(&data64(erased), typed, 0.0, 0.0, &format!("case {index}"));
    }
}
