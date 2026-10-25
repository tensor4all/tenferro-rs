//! A tiny contraction stays on the calling thread without a pool entry.

use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

fn matrix(shape: [usize; 2], data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data.to_vec()).expect("matrix")
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
fn tiny_einsum_uses_the_contraction_lane_inline_and_matches_serial() {
    // d[i, l] = sum_j a[i, j] b[j, l]
    let a = matrix([2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = matrix([2, 2], &[5.0, 6.0, 7.0, 8.0]);

    let mut serial = CpuEngine::serial();
    let expected = [&a, &b]
        .einsum("ij,jl->il", &mut serial)
        .expect("serial einsum");

    let mut engine = pooled(4);
    let actual = [&a, &b]
        .einsum("ij,jl->il", &mut engine)
        .expect("pooled einsum");

    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(
        actual.as_slice::<f64>().expect("f64 values"),
        expected.as_slice::<f64>().expect("f64 values")
    );
    #[cfg(feature = "ledger")]
    assert_eq!(
        engine.take_ledger_installs(),
        0,
        "a tiny contraction must not enter the pool"
    );
}

#[test]
fn contraction_families_match_serial_on_pooled_engines_at_one_two_and_four_threads() {
    use tenferro_einsum::prelude::TensorTensordotExt;
    use tenferro_einsum::TensorDotAxes;

    let n = 32;
    let make = |seed: usize| {
        Tensor::from_vec_col_major(
            vec![n, n],
            (0..n * n)
                .map(|index| ((index + seed) % 17) as f64 * 0.125 - 1.0)
                .collect::<Vec<_>>(),
        )
        .expect("matrix")
    };
    let a = make(0);
    let b = make(3);
    let c = make(7);

    let mut serial = CpuEngine::serial();
    let serial_dot = a
        .tensordot(&b, TensorDotAxes::Count(1), &mut serial)
        .expect("serial tensordot");
    let serial_nary = [&a, &b, &c]
        .einsum("ij,jk,kl->il", &mut serial)
        .expect("serial n-ary einsum");

    for threads in [1, 2, 4] {
        let mut engine = pooled(threads);
        let dot = a
            .tensordot(&b, TensorDotAxes::Count(1), &mut engine)
            .expect("pooled tensordot");
        let nary = [&a, &b, &c]
            .einsum("ij,jk,kl->il", &mut engine)
            .expect("pooled n-ary einsum");

        assert_eq!(
            dot.as_slice::<f64>().expect("dot values"),
            serial_dot.as_slice::<f64>().expect("serial dot values")
        );
        assert_eq!(
            nary.as_slice::<f64>().expect("n-ary values"),
            serial_nary.as_slice::<f64>().expect("serial n-ary values")
        );
    }
}
