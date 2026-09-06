//! Small public-API experiment for #1761; see docs/worklogs/issue-1761-binary-einsum.md.
use std::hint::black_box;
use std::time::{Duration, Instant};

use tenferro_cpu::CpuBackend;
use tenferro_einsum::{
    parse_einsum_subscripts, ConcreteEinsumPlan, TensorEinsumExt, TensorReadEinsumExt,
};
use tenferro_tensor::{BackendSessionHost, Tensor, TensorRead, TensorView, TypedTensorView};

fn measure(name: &str, mut call: impl FnMut()) {
    for _ in 0..100 {
        call();
    }
    for sample in 0..5 {
        let start = Instant::now();
        let mut iterations = 0;
        loop {
            for _ in 0..100 {
                call();
            }
            iterations += 100;
            if start.elapsed() >= Duration::from_millis(50) {
                break;
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "{name},{sample},{iterations},{elapsed:.9},{:.3}",
            elapsed * 1e9 / iterations as f64
        );
    }
}

fn observe(tensor: Tensor) {
    black_box(tensor.as_slice::<f64>().unwrap());
}

fn check(tensor: &Tensor, expected: &[f64]) {
    let actual = tensor.as_slice::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(error < 1e-9, "maximum absolute error: {error}");
}

fn data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 7 + seed) % 19) as f64 / 19.0 - 0.5)
        .collect()
}

fn main() {
    let threads = std::env::args()
        .nth(1)
        .expect("argument: default or thread count");
    let mut backend = if threads == "default" {
        CpuBackend::new()
    } else {
        CpuBackend::with_threads(threads.parse().unwrap()).unwrap()
    };
    println!("case,sample,iterations,seconds,ns_per_call");
    for n in [2, 8, 32, 256] {
        let a = Tensor::from_vec_col_major([n, n], data(n * n, 1)).unwrap();
        let b = Tensor::from_vec_col_major([n, n], data(n * n, 3)).unwrap();
        let inputs = [&a, &b];
        let subs = parse_einsum_subscripts("ij,jk->ik").unwrap();
        let plan = ConcreteEinsumPlan::prepare(inputs, "ij,jk->ik").unwrap();
        let av = a.as_slice::<f64>().unwrap();
        let bv = b.as_slice::<f64>().unwrap();
        let expected: Vec<f64> = (0..n * n)
            .map(|p| {
                (0..n)
                    .map(|k| av[p % n + n * k] * bv[k + n * (p / n)])
                    .sum()
            })
            .collect();
        check(
            &backend
                .with_backend_session(|s| inputs.einsum("ij,jk->ik", s))
                .unwrap(),
            &expected,
        );
        measure(&format!("prepare_string/{n}"), || {
            black_box(
                ConcreteEinsumPlan::prepare(black_box(inputs), black_box("ij,jk->ik")).unwrap(),
            );
        });
        measure(&format!("prepare_labels/{n}"), || {
            black_box(
                ConcreteEinsumPlan::prepare_subscripts(black_box(inputs), black_box(&subs))
                    .unwrap(),
            );
        });
        measure(&format!("ordinary_string/{n}"), || {
            observe(
                backend
                    .with_backend_session(|s| black_box(inputs).einsum(black_box("ij,jk->ik"), s))
                    .unwrap(),
            )
        });
        measure(&format!("ordinary_labels/{n}"), || {
            observe(
                backend
                    .with_backend_session(|s| {
                        black_box(inputs).einsum_subscripts(black_box(&subs), s)
                    })
                    .unwrap(),
            )
        });
        measure(&format!("prepared/{n}"), || {
            observe(
                backend
                    .with_backend_session(|s| plan.execute(black_box(inputs), s))
                    .unwrap(),
            )
        });
    }

    // Changing operand order and host strides may change the selected dot orientation.
    for n in [8, 256] {
        let adata = data(n * n, 1);
        let b = Tensor::from_vec_col_major([n, n], data(n * n, 3)).unwrap();
        for (layout, strides, offset) in [
            ("compact", [1, n as isize], 0),
            ("transpose", [n as isize, 1], 0),
            ("reverse", [-1, n as isize], n - 1),
        ] {
            let a = TypedTensorView::from_slice([n, n], strides, offset as isize, &adata).unwrap();
            let read = TensorRead::from_view(TensorView::F64(a));
            let bv = b.as_slice::<f64>().unwrap();
            let expected: Vec<f64> = (0..n * n)
                .map(|p| {
                    (0..n)
                        .map(|k| {
                            let index = offset as isize
                                + (p % n) as isize * strides[0]
                                + k as isize * strides[1];
                            adata[index as usize] * bv[k + n * (p / n)]
                        })
                        .sum()
                })
                .collect();
            for (order, inputs, equation) in [
                (
                    "ab",
                    [read.clone(), TensorRead::from_tensor(&b)],
                    "ij,jk->ik",
                ),
                (
                    "ba",
                    [TensorRead::from_tensor(&b), read.clone()],
                    "jk,ij->ik",
                ),
            ] {
                check(
                    &backend
                        .with_backend_session(|s| inputs.einsum_read(equation, s))
                        .unwrap(),
                    &expected,
                );
                measure(&format!("read_{layout}_{order}/{n}"), || {
                    observe(
                        backend
                            .with_backend_session(|s| {
                                black_box(&inputs).einsum_read(black_box(equation), s)
                            })
                            .unwrap(),
                    )
                });
            }
        }
    }

    let a = Tensor::from_vec_col_major([8, 8, 4], vec![0.5_f64; 256]).unwrap();
    let b = Tensor::from_vec_col_major([8, 8, 4], vec![0.25_f64; 256]).unwrap();
    let batch = [&a, &b];
    check(
        &backend
            .with_backend_session(|s| batch.einsum("ijb,jkb->ikb", s))
            .unwrap(),
        &[1.0; 256],
    );
    measure("batch_prepare/8", || {
        black_box(ConcreteEinsumPlan::prepare(black_box(batch), "ijb,jkb->ikb").unwrap());
    });
    measure("batch_ordinary/8", || {
        observe(
            backend
                .with_backend_session(|s| black_box(batch).einsum("ijb,jkb->ikb", s))
                .unwrap(),
        )
    });
    let c = Tensor::from_vec_col_major([8, 8, 4], vec![0.5_f64; 256]).unwrap();
    let nary = [&a, &b, &c];
    check(
        &backend
            .with_backend_session(|s| nary.einsum("ijb,jkb,klb->ilb", s))
            .unwrap(),
        &[4.0; 256],
    );
    measure("nary_prepare/8", || {
        black_box(ConcreteEinsumPlan::prepare(black_box(nary), "ijb,jkb,klb->ilb").unwrap());
    });
    measure("nary_ordinary/8", || {
        observe(
            backend
                .with_backend_session(|s| black_box(nary).einsum("ijb,jkb,klb->ilb", s))
                .unwrap(),
        )
    });
}
