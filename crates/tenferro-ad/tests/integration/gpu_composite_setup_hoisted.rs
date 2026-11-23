#![cfg(feature = "cuda")]

//! Composite constants are hoisted out of the per-call path on CUDA (#2701).
//!
//! A composite (an activation, a normalization, a softmax, a mean, a scale) used to upload each
//! of its constants per call and to stage every broadcast operand into a compact copy: softmax
//! cost 15 allocations, 11 launches and 2 host-to-device transfers on the direct route and more on
//! the eager one. The concrete engine now keeps its scalars resident (the scalar-operand engine
//! hooks), an eager session reuses its constants, an untracked eager broadcast is read as a
//! zero-stride view, and the layout-taking catalogue entries read every such view in place.
//!
//! This pins the contract with the runtime's own counters. In steady state every composite makes
//! no host-to-device copy (counted on every upload path), launches exactly the primitives of its
//! formulation, and every device allocation is a kernel's own result (allocations equal launches: no upload, no staging copy). The eager route costs
//! exactly what the direct route costs and returns the same bits.
//!
//! Run with: `cargo test -p tenferro-ad --features cuda --test integration
//! gpu_composite_setup_hoisted -- --ignored`

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_gpu::cuda::{
    download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime,
};
use tenferro_runtime::{CoreOps, Tensor, TypedTensor};
use tenferro_tensor::{DType, Engine};

const ROWS: usize = 64;
const COLS: usize = 48;

fn f64_matrix(salt: usize) -> Tensor {
    Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![ROWS, COLS],
            (0..ROWS * COLS)
                .map(|i| ((i * 7 + salt) % 97) as f64 / 24.0 - 2.0)
                .collect(),
        )
        .expect("host matrix"),
    )
}

fn c64_matrix() -> Tensor {
    Tensor::from_typed::<Complex64>(
        TypedTensor::from_vec_col_major(
            vec![ROWS, COLS],
            (0..ROWS * COLS)
                .map(|i| Complex64::new((i % 11) as f64 - 5.0, (i % 7) as f64 * 0.5))
                .collect(),
        )
        .expect("host matrix"),
    )
}

fn mask() -> Tensor {
    Tensor::from_typed::<bool>(
        TypedTensor::from_vec_col_major(
            vec![ROWS, COLS],
            (0..ROWS * COLS).map(|i| i % 5 != 0).collect(),
        )
        .expect("host mask"),
    )
}

fn vector(salt: usize) -> Tensor {
    Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![COLS],
            (0..COLS)
                .map(|i| ((i + salt) % 13) as f64 / 6.0 + 0.25)
                .collect(),
        )
        .expect("host vector"),
    )
}

/// Cumulative device allocations, launches and host-to-device copies.
fn counters(runtime: &CudaRuntime) -> [u64; 3] {
    let stats = runtime.stats().retirement;
    [stats.allocations, stats.launches, runtime.host_uploads()]
}

fn delta(before: [u64; 3], after: [u64; 3]) -> [u64; 3] {
    [
        after[0] - before[0],
        after[1] - before[1],
        after[2] - before[2],
    ]
}

fn host_bits(runtime: &CudaRuntime, tensor: &Tensor) -> Vec<u64> {
    let host = download_tensor(runtime, tensor).expect("download");
    match host.dtype() {
        DType::C64 => host
            .as_slice::<Complex64>()
            .expect("c64")
            .iter()
            .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
            .collect(),
        _ => host
            .as_slice::<f64>()
            .expect("f64")
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    }
}

/// The composite inputs, uploaded once: `x`, a complex `z`, a Bool mask, a weight and a bias.
struct Inputs {
    x: Tensor,
    z: Tensor,
    mask: Tensor,
    weight: Tensor,
    bias: Tensor,
}

impl Inputs {
    fn upload(runtime: &CudaRuntime) -> Self {
        let up = |host: Tensor| upload_tensor(runtime, &host).expect("upload");
        Self {
            x: up(f64_matrix(3)),
            z: up(c64_matrix()),
            mask: up(mask()),
            weight: up(vector(1)),
            bias: up(vector(7)),
        }
    }
}

/// The same inputs as eager constants of one session.
struct EagerInputs {
    x: EagerTensor,
    z: EagerTensor,
    mask: EagerTensor,
    weight: EagerTensor,
    bias: EagerTensor,
}

type Direct = fn(&Inputs, &mut dyn Engine) -> tenferro_tensor::Result<Tensor>;
type Eager = fn(&EagerInputs, &mut EagerSession<'_>) -> tenferro_ad::Result<EagerTensor>;

/// Every composite the GPU benchmark measures, on the direct and the eager route.
fn cases() -> Vec<(&'static str, Direct, Eager)> {
    vec![
        ("gelu", |i, s| i.x.gelu(s), |i, s| s.gelu(&i.x)),
        (
            "gelu_tanh",
            |i, s| i.x.gelu_tanh(s),
            |i, s| s.gelu_tanh(&i.x),
        ),
        ("sigmoid", |i, s| i.x.sigmoid(s), |i, s| s.sigmoid(&i.x)),
        ("silu", |i, s| i.x.silu(s), |i, s| s.silu(&i.x)),
        (
            "softmax",
            |i, s| i.x.softmax(1, s),
            |i, s| s.softmax(&i.x, 1),
        ),
        (
            "log_softmax",
            |i, s| i.x.log_softmax(1, s),
            |i, s| s.log_softmax(&i.x, 1),
        ),
        (
            "masked_softmax",
            |i, s| i.x.masked_softmax(&i.mask, 1, s),
            |i, s| s.masked_softmax(&i.x, &i.mask, 1),
        ),
        (
            "masked_log_softmax",
            |i, s| i.x.masked_log_softmax(&i.mask, 1, s),
            |i, s| s.masked_log_softmax(&i.x, &i.mask, 1),
        ),
        (
            "layer_norm",
            |i, s| i.x.layer_norm(1, Some(&i.weight), Some(&i.bias), 1e-5, s),
            |i, s| s.layer_norm(&i.x, 1, Some(&i.weight), Some(&i.bias), 1e-5),
        ),
        (
            "rms_norm",
            |i, s| i.x.rms_norm(1, Some(&i.weight), None, 1e-5, s),
            |i, s| s.rms_norm(&i.x, 1, Some(&i.weight), None, 1e-5),
        ),
        (
            "reduce_mean",
            |i, s| i.x.reduce_mean(Some(&[1isize]), s),
            |i, s| s.reduce_mean(&i.x, Some(&[1isize])),
        ),
        (
            "scale_real",
            |i, s| i.x.scale_real(0.5, s),
            |i, s| s.scale_real(&i.x, 0.5),
        ),
        (
            "scale_complex",
            |i, s| i.z.scale_complex(Complex64::new(0.5, -1.5), s),
            |i, s| s.scale_complex(&i.z, Complex64::new(0.5, -1.5)),
        ),
    ]
}

/// The primitive kernels each composite formulation runs on a 64x48 input along axis 1
/// (`tenferro_runtime::composite`): gelu is scale, erf, add one, multiply by x and halve; softmax is
/// max, floor maximum, subtract, exp, sum, tiny maximum and divide; and so on. A staging copy, an
/// upload or a materialized broadcast would add to it.
fn primitive_kernels(name: &str) -> u64 {
    match name {
        "gelu" => 5,
        "gelu_tanh" => 9,
        "sigmoid" => 7,
        "silu" => 8,
        "softmax" => 7,
        "log_softmax" => 8,
        "masked_softmax" => 8,
        "masked_log_softmax" => 9,
        "layer_norm" => 12,
        "rms_norm" => 7,
        "reduce_mean" => 2,
        "scale_real" | "scale_complex" => 1,
        other => panic!("no primitive count for {other}"),
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn composites_transfer_nothing_and_allocate_only_results_in_steady_state() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let runtime = gpu.runtime().clone();
    let inputs = Inputs::upload(&runtime);
    let ctx = EagerRuntime::new().expect("eager runtime");
    let _no_grad = ctx.no_grad();

    for (name, direct, eager) in cases() {
        // Direct route: the first call uploads the composite's constants and prepares its
        // modules; the steady-state call is counted.
        drop(direct(&inputs, &mut gpu).unwrap_or_else(|error| panic!("{name}: {error}")));
        let before = counters(&runtime);
        let reference = direct(&inputs, &mut gpu).unwrap_or_else(|error| panic!("{name}: {error}"));
        let [allocations, launches, transfers] = delta(before, counters(&runtime));
        assert_eq!(
            launches,
            primitive_kernels(name),
            "{name}: one launch per primitive of the formulation (no extra copy kernel)"
        );
        assert_eq!(
            transfers, 0,
            "{name}: a steady-state call transfers nothing to the device"
        );
        assert_eq!(
            allocations, launches,
            "{name}: every allocation is a kernel's result (no upload, no staging copy)"
        );

        // Eager route, in one session as the benchmark holds one.
        let (eager_cost, out) = ctx
            .with_eager_session(&mut gpu, |s| {
                let fresh = Inputs::upload(&runtime);
                let values = EagerInputs {
                    x: s.constant_from(fresh.x)?,
                    z: s.constant_from(fresh.z)?,
                    mask: s.constant_from(fresh.mask)?,
                    weight: s.constant_from(fresh.weight)?,
                    bias: s.constant_from(fresh.bias)?,
                };
                drop(eager(&values, s)?);
                let before = counters(&runtime);
                let out = eager(&values, s)?;
                let cost = delta(before, counters(&runtime));
                Ok::<_, tenferro_ad::Error>((cost, s.duplicate_value(&out)?))
            })
            .unwrap_or_else(|error| panic!("eager {name}: {error}"));
        assert_eq!(
            eager_cost,
            [allocations, launches, 0],
            "eager {name} costs what the direct route costs"
        );
        assert_eq!(
            host_bits(&runtime, &out),
            host_bits(&runtime, &reference),
            "eager {name} matches the direct route bit for bit"
        );
    }
}
