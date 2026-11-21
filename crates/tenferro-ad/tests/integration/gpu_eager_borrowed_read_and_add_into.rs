#![cfg(feature = "cuda")]

//! The eager output-reuse routes on a CUDA device read their operands in place (#2701).
//!
//! An eager value's `tensor_read()` is a descriptor view of its retained allocation, not an
//! owned `Tensor`. The CUDA elementwise `_into` path used to stage every view into a compact
//! copy before its flat kernel, so `add_into` on two eager values cost two device allocations
//! and three launches where the owned-tensor hook costs none and one. A flat view (one compact
//! span at its allocation's base) is now read where it lies.
//!
//! This pins the borrowed contract with the runtime's own counters, not timings: every eager
//! `_into` on compact eager operands performs exactly one launch and no device allocation (no
//! staging copy, no upload, no result), and writes the same bits as the owned-tensor hook. A
//! permuted (transposed) eager view is still staged once, because the flat kernels address no
//! layout; that path is checked for values only. Complex and integer division are checked for
//! values too: their kernels need a zeroed 4-byte status word (a device memset, not an upload).
//!
//! Run with: `cargo test -p tenferro-ad --features cuda --test integration
//! gpu_eager_borrowed_read_and_add_into -- --ignored --include-ignored`

use num_complex::Complex64;
use tenferro_ad::{EagerRuntime, EagerSession, EagerTensor};
use tenferro_gpu::cuda::{download_tensor, gpu_available, upload_tensor, CudaDeviceId, CudaEngine};
use tenferro_runtime::{Tensor, TypedTensor};
use tenferro_tensor::{Dest, TensorElementwise, TensorRead, TensorWrite};

const M: usize = 96;
const N: usize = 80;

fn f64_matrix(rows: usize, cols: usize, value: impl Fn(usize) -> f64) -> Tensor {
    Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(vec![rows, cols], (0..rows * cols).map(value).collect())
            .expect("host matrix"),
    )
}

fn c64_matrix(rows: usize, cols: usize) -> Tensor {
    Tensor::from_typed::<Complex64>(
        TypedTensor::from_vec_col_major(
            vec![rows, cols],
            (0..rows * cols)
                .map(|i| Complex64::new((i % 11) as f64 - 5.0, (i % 7) as f64 * 0.5))
                .collect(),
        )
        .expect("host matrix"),
    )
}

/// The runtime's cumulative device allocations and kernel launches.
fn counters(gpu: &CudaEngine) -> (u64, u64) {
    let stats = gpu.runtime().stats().retirement;
    (stats.allocations, stats.launches)
}

fn host_bits(gpu: &CudaEngine, tensor: &Tensor) -> Vec<u64> {
    let host = download_tensor(gpu.runtime(), tensor).expect("download");
    match host.dtype() {
        tenferro_tensor::DType::C64 => host
            .as_slice::<Complex64>()
            .expect("host c64")
            .iter()
            .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
            .collect(),
        _ => host
            .as_slice::<f64>()
            .expect("host f64")
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    }
}

type EagerInto = fn(
    &mut EagerSession<'_>,
    &EagerTensor,
    &EagerTensor,
    Dest<TensorWrite<'_>>,
) -> tenferro_ad::Result<()>;
type HookInto =
    fn(&mut CudaEngine, &Tensor, &Tensor, TensorWrite<'_>) -> tenferro_tensor::Result<()>;

/// Each eager `_into` beside the owned-tensor engine hook it must match; unary forms ignore `rhs`.
fn cases() -> [(&'static str, EagerInto, HookInto); 6] {
    [
        (
            "add",
            |s, a, b, d| s.add_into(a, b, d),
            |g, a, b, o| g.add_into(a, b, o),
        ),
        (
            "sub",
            |s, a, b, d| s.sub_into(a, b, d),
            |g, a, b, o| g.sub_into(a, b, o),
        ),
        (
            "mul",
            |s, a, b, d| s.mul_into(a, b, d),
            |g, a, b, o| g.mul_into(a, b, o),
        ),
        (
            "div",
            |s, a, b, d| s.div_into(a, b, d),
            |g, a, b, o| g.div_into(a, b, o),
        ),
        (
            "neg",
            |s, a, _, d| s.neg_into(a, d),
            |g, a, _, o| g.neg_into(a, o),
        ),
        (
            "conj",
            |s, a, _, d| s.conj_into(a, d),
            // The concrete form: conjugation is the identity copy on a real dtype.
            |g, a, _, o| tenferro_runtime::CoreOps::conj_into(a, g, Dest::Overwrite(o)),
        ),
    ]
}

#[test]
#[ignore = "requires a CUDA device"]
fn eager_into_on_compact_operands_allocates_nothing_and_launches_once() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let ctx = EagerRuntime::new().expect("eager runtime");
    let real = (
        f64_matrix(M, N, |i| (i % 97) as f64 / 8.0 + 0.5),
        f64_matrix(M, N, |i| (i % 31) as f64 * 1.5 + 1.0),
    );
    let complex = (c64_matrix(M, N), c64_matrix(M, N));

    for (name, eager_into, hook_into) in cases() {
        let pairs: Vec<&(Tensor, Tensor)> = if name == "conj" {
            vec![&real, &complex]
        } else {
            vec![&real]
        };
        for (lhs_host, rhs_host) in pairs {
            let up = |gpu: &CudaEngine, host: &Tensor| {
                upload_tensor(gpu.runtime(), host).expect("upload")
            };
            let (lhs_owned, rhs_owned) = (up(&gpu, lhs_host), up(&gpu, rhs_host));
            let (lhs_value, rhs_value) = (up(&gpu, lhs_host), up(&gpu, rhs_host));
            let (lhs, rhs) = ctx
                .with_eager_session(&mut gpu, |s| {
                    Ok::<_, tenferro_ad::Error>((
                        s.constant_from(lhs_value)?,
                        s.constant_from(rhs_value)?,
                    ))
                })
                .expect("eager constants");
            assert!(
                matches!(lhs.tensor_read(), TensorRead::View(_)),
                "an eager value reads as a descriptor view; that is the path under test"
            );
            // Destinations are prepared once, outside the counted window.
            let mut hook_out = up(&gpu, lhs_host);
            let mut eager_out = up(&gpu, lhs_host);

            hook_into(
                &mut gpu,
                &lhs_owned,
                &rhs_owned,
                TensorWrite::from_tensor(&mut hook_out),
            )
            .unwrap_or_else(|error| panic!("{name} hook: {error}"));

            let before = counters(&gpu);
            ctx.with_eager_session(&mut gpu, |s| {
                eager_into(s, &lhs, &rhs, Dest::overwrite(&mut eager_out))
            })
            .unwrap_or_else(|error| panic!("eager {name}_into: {error}"));
            let after = counters(&gpu);
            assert_eq!(
                after.0 - before.0,
                0,
                "eager {name}_into on {:?} allocated device memory (a staging copy or upload)",
                lhs_host.dtype()
            );
            assert_eq!(
                after.1 - before.1,
                1,
                "eager {name}_into on {:?} must be exactly its own kernel launch",
                lhs_host.dtype()
            );
            assert_eq!(
                host_bits(&gpu, &eager_out),
                host_bits(&gpu, &hook_out),
                "eager {name}_into must write the owned hook's bits"
            );
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn eager_into_reads_transposed_eager_views_correctly() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let ctx = EagerRuntime::new().expect("eager runtime");
    let lhs_host = f64_matrix(M, N, |i| (i % 97) as f64 / 8.0);
    let rhs_host = f64_matrix(M, N, |i| -((i % 31) as f64) * 1.5);
    let x_value = upload_tensor(gpu.runtime(), &lhs_host).expect("upload");
    let y_value = upload_tensor(gpu.runtime(), &rhs_host).expect("upload");
    let (xt, yt) = ctx
        .with_eager_session(&mut gpu, |s| {
            let x = s.constant_from(x_value)?;
            let y = s.constant_from(y_value)?;
            Ok::<_, tenferro_ad::Error>((s.transpose(&x, &[1, 0])?, s.transpose(&y, &[1, 0])?))
        })
        .expect("eager transposes");
    let mut out = upload_tensor(gpu.runtime(), &f64_matrix(N, M, |_| f64::NAN)).expect("upload");
    ctx.with_eager_session(&mut gpu, |s| {
        s.add_into(&xt, &yt, Dest::overwrite(&mut out))
    })
    .expect("eager strided add_into");
    let got = download_tensor(gpu.runtime(), &out).expect("download");
    let got = got.as_slice::<f64>().expect("host f64");
    let (a, b) = (
        lhs_host.as_slice::<f64>().expect("host"),
        rhs_host.as_slice::<f64>().expect("host"),
    );
    for i in 0..M {
        for j in 0..N {
            // out is [N, M] column-major: out[j, i] = x[i, j] + y[i, j].
            assert_eq!(got[j + N * i], a[i + M * j] + b[i + M * j]);
        }
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn eager_div_into_serves_complex_and_integer_dtypes() {
    assert!(gpu_available(), "CUDA test requires an available device");
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).expect("open CUDA engine");
    let ctx = EagerRuntime::new().expect("eager runtime");
    let complex = c64_matrix(M, N);
    let complex_rhs = Tensor::from_typed::<Complex64>(
        TypedTensor::from_vec_col_major(
            vec![M, N],
            (0..M * N)
                .map(|i| Complex64::new(1.0 + (i % 5) as f64, -0.5 * (i % 3) as f64))
                .collect(),
        )
        .expect("host matrix"),
    );
    let ints = |values: Vec<i32>| {
        Tensor::from_typed::<i32>(
            TypedTensor::from_vec_col_major(vec![values.len()], values).expect("host ints"),
        )
    };
    let cases = [
        (complex, complex_rhs, false),
        (ints(vec![7, -9, 10, 3]), ints(vec![2, 4, -3, 1]), false),
        (ints(vec![7, -9, 10, 3]), ints(vec![2, 0, -3, 1]), true),
    ];
    for (lhs_host, rhs_host, divides_by_zero) in cases {
        let lhs_value = upload_tensor(gpu.runtime(), &lhs_host).expect("upload");
        let rhs_value = upload_tensor(gpu.runtime(), &rhs_host).expect("upload");
        let mut out = upload_tensor(gpu.runtime(), &lhs_host).expect("upload destination");
        let result = ctx.with_eager_session(&mut gpu, |s| {
            let lhs = s.constant_from(lhs_value)?;
            let rhs = s.constant_from(rhs_value)?;
            s.div_into(&lhs, &rhs, Dest::overwrite(&mut out))?;
            s.div(&lhs, &rhs)
        });
        if divides_by_zero {
            let error = result.expect_err("an integer division by zero is reported");
            assert!(error.to_string().contains("division by zero"), "{error}");
            continue;
        }
        let fresh = result
            .unwrap_or_else(|error| panic!("{:?} div_into: {error}", lhs_host.dtype()))
            .into_value()
            .expect("unique result");
        let (got, want) = (
            download_tensor(gpu.runtime(), &out).expect("download"),
            download_tensor(gpu.runtime(), &fresh).expect("download"),
        );
        match lhs_host.dtype() {
            tenferro_tensor::DType::C64 => assert_eq!(
                got.as_slice::<Complex64>().expect("host"),
                want.as_slice::<Complex64>().expect("host")
            ),
            _ => assert_eq!(
                got.as_slice::<i32>().expect("host"),
                want.as_slice::<i32>().expect("host")
            ),
        }
    }
}
