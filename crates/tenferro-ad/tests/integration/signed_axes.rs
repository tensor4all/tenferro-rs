//! Signed axes on the eager layer (#2552 §3.2): `-1` is the last axis.

use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
use tenferro_cpu::CpuEngine;

fn bits(t: &EagerTensor) -> (Vec<usize>, Vec<u64>) {
    let v = t
        .runtime()
        .with_eager_session(&mut CpuEngine::serial(), |s| s.duplicate_value(&t))
        .unwrap();
    (
        v.shape().to_vec(),
        v.as_slice::<f64>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect(),
    )
}

#[test]
fn negative_axes_equal_rank_minus_one_on_the_eager_layer() {
    let ctx = EagerRuntime::new().unwrap();
    let pairs = ctx
        .with_eager_session(&mut CpuEngine::serial(), |s| {
            let x = s.constant_from(Tensor::from_vec_col_major(
                vec![2, 3],
                vec![0.5_f64, -1.25, 2.0, 0.75, 3.5, -0.25],
            )?)?;
            assert!(s.reduce_sum(&x, Some(&[-3])).is_err());
            assert!(s.softmax(&x, 2).is_err());
            Ok::<_, tenferro_ad::Error>(vec![
                (
                    s.reduce_sum(&x, Some(&[-1]))?,
                    s.reduce_sum(&x, Some(&[1]))?,
                ),
                (
                    s.reduce_max(&x, Some(&[-2]))?,
                    s.reduce_max(&x, Some(&[0]))?,
                ),
                (s.softmax(&x, -1)?, s.softmax(&x, 1)?),
                (s.concatenate(&[&x, &x], -1)?, s.concatenate(&[&x, &x], 1)?),
                (s.transpose(&x, &[-1, 0])?, s.transpose(&x, &[1, 0])?),
            ])
        })
        .unwrap();
    for (signed, unsigned) in &pairs {
        assert_eq!(bits(signed), bits(unsigned));
    }
}
