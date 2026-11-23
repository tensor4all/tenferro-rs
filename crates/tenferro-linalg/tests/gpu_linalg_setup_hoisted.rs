#![cfg(all(feature = "cuda", feature = "autodiff"))]

//! The linalg composites transfer nothing to the device per call on CUDA (#2701).
//!
//! `inv` built its identity from an uploaded one, `pinv` uploaded its threshold factor and a ones
//! vector, on the concrete and eager routes alike. They now fill from host scalars the engine keeps
//! resident ([`tenferro_tensor::Engine::fill_scalar`], `binary_scalar_read`), and an eager session
//! reuses its constants. This pins it with the runtime's own counter: in steady state no call
//! allocates a pinned staging chunk, which every host-to-device transfer does. `lstsq` is pinned
//! too; its rank check uploads nothing either.
//!
//! Run with: `cargo test -p tenferro-linalg --features cuda,autodiff --test gpu_linalg_setup_hoisted
//! -- --ignored`
use tenferro_ad::EagerRuntime;
use tenferro_gpu::cuda::{upload_tensor, CudaDeviceId, CudaEngine, CudaRuntime};
use tenferro_linalg::{EagerLinalgOps, LinalgOps, PinvOptions};
use tenferro_runtime::{Tensor, TypedTensor};
use tenferro_tensor::Engine;

fn counters(r: &CudaRuntime) -> [u64; 4] {
    let r = r.stats().retirement;
    [r.allocations, r.launches, r.zero_fills, r.chunk_allocs]
}

/// Assert that a steady-state call made no host-to-device transfer.
fn assert_no_transfer(route: &str, name: &str, before: [u64; 4], after: [u64; 4]) {
    assert_eq!(
        after[3] - before[3],
        0,
        "{route} {name}: a steady-state call transfers nothing to the device"
    );
}

fn matrix(rows: usize, cols: usize, diag: f64) -> Tensor {
    Tensor::from_typed::<f64>(
        TypedTensor::from_vec_col_major(
            vec![rows, cols],
            (0..rows * cols)
                .map(|i| {
                    if i % (rows + 1) == 0 {
                        diag
                    } else {
                        ((i % 13) as f64) / 13.0
                    }
                })
                .collect(),
        )
        .unwrap(),
    )
}

#[test]
#[ignore = "requires a CUDA device"]
fn linalg_composites_transfer_nothing_in_steady_state() {
    let mut gpu = CudaEngine::open(CudaDeviceId::from_ordinal(0)).unwrap();
    let rt = gpu.runtime().clone();
    let a = upload_tensor(&rt, &matrix(32, 32, 40.0)).unwrap();
    let tall = upload_tensor(&rt, &matrix(64, 32, 40.0)).unwrap();
    let b = upload_tensor(&rt, &matrix(64, 4, 1.0)).unwrap();
    type Op<'t> = Box<dyn Fn(&mut dyn Engine) -> Tensor + 't>;
    let ops: Vec<(&str, Op<'_>)> = vec![
        ("inv", Box::new(|s| a.inv(s).unwrap())),
        (
            "pinv",
            Box::new(|s| {
                a.pinv_with(&PinvOptions::default().rtol(Some(1e-5)), s)
                    .unwrap()
            }),
        ),
        ("lstsq", Box::new(|s| tall.lstsq(&b, s).unwrap())),
    ];
    for (name, f) in &ops {
        for _ in 0..3 {
            let _ = f(&mut gpu);
        }
        let before = counters(&rt);
        let _r = f(&mut gpu);
        assert_no_transfer("direct", name, before, counters(&rt));
    }
    let ctx = EagerRuntime::new().unwrap();
    let _guard = ctx.no_grad();
    let (ea, et, eb) = ctx
        .with_eager_session(&mut gpu, |s| {
            Ok::<_, tenferro_ad::Error>((
                s.constant_from(upload_tensor(&rt, &matrix(32, 32, 40.0)).unwrap())?,
                s.constant_from(upload_tensor(&rt, &matrix(64, 32, 40.0)).unwrap())?,
                s.constant_from(upload_tensor(&rt, &matrix(64, 4, 1.0)).unwrap())?,
            ))
        })
        .unwrap();
    for name in ["inv", "pinv", "lstsq"] {
        let (before, after) = ctx
            .with_eager_session(&mut gpu, |s| {
                let mut run = |s: &mut tenferro_ad::EagerSession<'_>| match name {
                    "inv" => s.inv(&ea),
                    "pinv" => s.pinv_with(&ea, &PinvOptions::default().rtol(Some(1e-5))),
                    _ => s.lstsq(&et, &eb),
                };
                for _ in 0..3 {
                    run(s)?;
                }
                let before = counters(&rt);
                run(s)?;
                Ok::<_, tenferro_ad::Error>((before, counters(&rt)))
            })
            .unwrap();
        assert_no_transfer("eager", name, before, after);
    }
}
