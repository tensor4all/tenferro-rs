//! Bounded issue #1762 diagnostic: 2x2 F64, ten dependent matmuls, explicit 1T.
use std::hint::black_box;
use std::time::{Duration, Instant};

use tenferro_ad::EagerRuntime;
use tenferro_cpu::CpuBackend;
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor};

fn sample(f: &mut impl FnMut(), duration: Duration) -> (usize, Duration) {
    let start = Instant::now();
    let mut calls = 0;
    while start.elapsed() < duration {
        for _ in 0..100 {
            f();
        }
        calls += 100;
    }
    (calls, start.elapsed())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut backend = CpuBackend::with_threads(1)?;
    let eager_backend = CpuBackend::with_threads(1)?;
    assert_eq!(backend.num_threads(), 1);
    assert_eq!(eager_backend.num_threads(), 1);
    eprintln!(
        "concrete_workers={} eager_workers={}",
        backend.num_threads(),
        eager_backend.num_threads()
    );
    let runtime = EagerRuntime::with_cpu_backend(eager_backend)?;
    let lhs = Tensor::from_vec_col_major([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0])?;
    let rhs = Tensor::from_vec_col_major([2, 2], vec![0.75_f64, 0.125, -0.25, 0.5])?;
    let a = runtime.constant_from(Tensor::from_vec_col_major(
        [2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let b = runtime.constant_from(Tensor::from_vec_col_major(
        [2, 2],
        vec![0.75_f64, 0.125, -0.25, 0.5],
    )?)?;
    let active_a = runtime.variable_from(Tensor::from_vec_col_major(
        [2, 2],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )?)?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    // Independent fixed 2x2 reference, column-major; never timed.
    let mut expected = [1.0_f64, 2.0, 3.0, 4.0];
    for _ in 0..10 {
        let [a, b, c, d] = expected;
        expected = [
            0.75 * a + 0.125 * c,
            0.75 * b + 0.125 * d,
            -0.25 * a + 0.5 * c,
            -0.25 * b + 0.5 * d,
        ];
    }
    let mut cases = vec![
        "shared",
        "per_op",
        "eager",
        "active_forward",
        "empty_x10",
        "leaf",
    ];
    if std::env::args().any(|arg| arg == "reverse") {
        cases.reverse();
    }
    println!("case,sample,calls,elapsed_ns,ns_per_call");
    for case in cases {
        let mut run = || -> Option<Tensor> {
            match case {
                "shared" => backend.with_backend_session(|session| {
                    let mut out = None;
                    for _ in 0..10 {
                        out = Some(
                            session
                                .dot_general(
                                    out.as_ref().unwrap_or(black_box(&lhs)),
                                    black_box(&rhs),
                                    black_box(&config),
                                )
                                .unwrap(),
                        );
                    }
                    out
                }),
                "per_op" => {
                    let mut out = None;
                    for _ in 0..10 {
                        out = Some(
                            backend
                                .with_backend_session(|session| {
                                    session.dot_general(
                                        out.as_ref().unwrap_or(black_box(&lhs)),
                                        black_box(&rhs),
                                        black_box(&config),
                                    )
                                })
                                .unwrap(),
                        );
                    }
                    out
                }
                "eager" | "active_forward" => {
                    let mut out = black_box(if case == "eager" { &a } else { &active_a }).clone();
                    for _ in 0..10 {
                        out = out
                            .dot_general(black_box(&b), black_box(config.clone()))
                            .unwrap();
                    }
                    Some(out.to_tensor().unwrap())
                }
                "empty_x10" => {
                    for _ in 0..10 {
                        backend.with_backend_session(|session| {
                            black_box(session);
                        });
                    }
                    None
                }
                // Tensor construction is included in this standalone diagnostic.
                "leaf" => Some(
                    runtime
                        .constant_from(
                            Tensor::from_vec_col_major(
                                [2, 2],
                                black_box(vec![1.0_f64, 2.0, 3.0, 4.0]),
                            )
                            .unwrap(),
                        )
                        .unwrap()
                        .to_tensor()
                        .unwrap(),
                ),
                _ => unreachable!(),
            }
        };
        if let Some(output) = run() {
            assert_eq!(output.shape(), &[2, 2]);
            let reference = if matches!(case, "empty_x10" | "leaf") {
                &[1.0, 2.0, 3.0, 4.0]
            } else {
                &expected
            };
            for (actual, expected) in output.as_slice::<f64>()?.iter().zip(reference) {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{case}: {actual} != {expected}"
                );
            }
        }
        let mut observed = || {
            if let Some(output) = run() {
                black_box(output.shape());
                black_box(output.as_slice::<f64>().unwrap());
            }
        };
        sample(&mut observed, Duration::from_millis(100));
        for index in 0..5 {
            let (calls, elapsed) = sample(&mut observed, Duration::from_millis(100));
            println!(
                "{case},{index},{calls},{},{:.3}",
                elapsed.as_nanos(),
                elapsed.as_nanos() as f64 / calls as f64
            );
        }
    }
    Ok(())
}
