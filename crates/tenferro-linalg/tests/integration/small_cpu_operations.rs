//! Small linalg operations stay on the calling thread without a pool entry.
//!
//! The direct faer lane (`HouseholderQr::<Tensor>::from_factors`) and the tlinalg lane
//! (`qr`) decide their width before entering the engine's pool. A tiny input
//! must therefore run inline: the engine's provider-pool entry counter stays
//! zero and the result matches a serial engine.

use tenferro_cpu::CpuEngine;
use tenferro_linalg::{HouseholderQr, QrOptions, TensorLinalgExt};
use tenferro_tensor::Tensor;

fn matrix(shape: [usize; 2], data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data.to_vec()).expect("matrix")
}

fn values(tensor: &Tensor) -> Vec<f64> {
    tensor.as_slice::<f64>().expect("f64 values").to_vec()
}

fn assert_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (left - right).abs() < 1.0e-12,
            "{label}[{index}]: {left} != {right}"
        );
    }
}

fn pooled(threads: usize) -> CpuEngine {
    CpuEngine::new(
        tenferro_cpu::Pool::builder()
            .threads(threads)
            .build()
            .expect("test pool"),
    )
}

#[test]
fn tiny_householder_qr_from_factors_uses_the_direct_faer_lane_inline() {
    let q = matrix([2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let r = matrix([2, 2], &[2.0, 0.0, 1.0, 3.0]);

    let mut serial = CpuEngine::serial();
    let serial_state =
        HouseholderQr::<Tensor>::from_factors(&q, &r, &mut serial).expect("serial compact qr");
    let serial_r = serial_state
        .r(QrOptions::default(), &mut serial)
        .expect("serial r");

    let mut engine = pooled(4);
    let state =
        HouseholderQr::<Tensor>::from_factors(&q, &r, &mut engine).expect("pooled compact qr");
    let actual_r = state
        .r(QrOptions::default(), &mut engine)
        .expect("pooled r");

    assert_close(&values(&actual_r), &values(&serial_r), "direct faer r");
    #[cfg(feature = "ledger")]
    assert_eq!(
        engine.take_ledger_installs(),
        0,
        "a tiny direct faer product must not enter the pool"
    );
}

#[test]
fn tiny_qr_uses_the_tlinalg_lane_inline() {
    let a = matrix([2, 2], &[2.0, 1.0, 1.0, 3.0]);

    let mut serial = CpuEngine::serial();
    let (serial_q, serial_r) = a.qr(&mut serial).expect("serial qr");

    let mut engine = pooled(4);
    let (q, r) = a.qr(&mut engine).expect("pooled qr");

    assert_close(&values(&q), &values(&serial_q), "tlinalg q");
    assert_close(&values(&r), &values(&serial_r), "tlinalg r");
    #[cfg(feature = "ledger")]
    assert_eq!(
        engine.take_ledger_installs(),
        0,
        "a tiny tlinalg factorization must not enter the pool"
    );
}

#[test]
fn linalg_families_match_serial_on_pooled_engines_at_one_two_and_four_threads() {
    let n = 8;
    let mut data = Vec::with_capacity(n * n);
    for col in 0..n {
        for row in 0..n {
            data.push(if row == col {
                4.0 + col as f64
            } else {
                ((row * 3 + col * 5) % 7) as f64 * 0.125
            });
        }
    }
    let a = matrix([n, n], &data);
    let b = matrix(
        [n, 2],
        &(0..2 * n)
            .map(|index| index as f64 * 0.5 - 1.0)
            .collect::<Vec<_>>(),
    );

    let mut serial = CpuEngine::serial();
    let serial_qr = a.qr(&mut serial).expect("serial qr");
    let serial_svd = a.svd(&mut serial).expect("serial svd");
    let serial_lu = a.lu(&mut serial).expect("serial lu");
    let serial_solve = a.solve(&b, &mut serial).expect("serial solve");

    for threads in [1, 2, 4] {
        let mut engine = pooled(threads);
        let qr = a.qr(&mut engine).expect("pooled qr");
        let svd = a.svd(&mut engine).expect("pooled svd");
        let lu = a.lu(&mut engine).expect("pooled lu");
        let solve = a.solve(&b, &mut engine).expect("pooled solve");

        assert_close(&values(&qr.0), &values(&serial_qr.0), "qr q");
        assert_close(&values(&qr.1), &values(&serial_qr.1), "qr r");
        assert_close(&values(&svd.0), &values(&serial_svd.0), "svd u");
        assert_close(&values(&svd.1), &values(&serial_svd.1), "svd s");
        assert_close(&values(&svd.2), &values(&serial_svd.2), "svd vt");
        assert_close(&values(&lu.0), &values(&serial_lu.0), "lu p");
        assert_close(&values(&lu.1), &values(&serial_lu.1), "lu l");
        assert_close(&values(&lu.2), &values(&serial_lu.2), "lu u");
        assert_close(&values(&lu.3), &values(&serial_lu.3), "lu q");
        assert_close(&values(&solve), &values(&serial_solve), "solve");
    }
}
