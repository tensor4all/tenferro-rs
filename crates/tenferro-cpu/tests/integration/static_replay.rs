use tenferro_tensor::backend::{ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan};
use tenferro_tensor::{DType, StridedSliceSpec, Tensor, TensorRead, TensorView, TypedTensor};

/// Distance between two values in units in the last place, so that a check names a
/// precision rather than a particular libm bit pattern.
fn ulp_distance(left: f64, right: f64) -> u64 {
    fn ordered(value: f64) -> i64 {
        let bits = value.to_bits() as i64;
        if bits < 0 {
            i64::MIN.wrapping_sub(bits)
        } else {
            bits
        }
    }
    ordered(left).abs_diff(ordered(right))
}

/// A contiguous run goes through the vector kernel and every other layout through `map_into`, and
/// `simd_math` documents that the two agree element for element but differ in rounding by at most
/// one ulp. So both the layout comparison and the libm reference are bounded in ulp, not exact; on
/// aarch64 macOS `sin(4.0)` is one ulp away on both counts.
fn assert_within_one_ulp(got: &[f64], want: &[f64], what: &str) {
    const MAX_ULP: u64 = 1;
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        let ulps = ulp_distance(got, want);
        assert!(
            ulps <= MAX_ULP,
            "{what}: element {index} is {ulps} ulp apart (got {got:?}, want {want:?})"
        );
    }
}

#[test]
fn static_analytic_replay_preserves_owned_and_reversed_values() {
    let mut backend = crate::engine(1);
    assert_eq!(backend.num_threads(), 1);
    let values = [0.25_f64, 1.5, 4.0];
    let typed = TypedTensor::<f64>::from_vec_col_major([3], values.to_vec()).unwrap();
    let owned = Tensor::from_typed::<f64>(typed.duplicate().unwrap());
    let reversed = typed
        .as_view()
        .slice_view(&[StridedSliceSpec::new(0, Some(3), -1)])
        .unwrap();
    for op in 0..9 {
        let expected: Vec<_> = values
            .iter()
            .map(|&x| match op {
                0 => x.exp(),
                1 => x.ln(),
                2 => x.sin(),
                3 => x.cos(),
                4 => x.tanh(),
                5 => x.sqrt(),
                6 => 1.0 / x.sqrt(),
                7 => x.exp_m1(),
                8 => x.ln_1p(),
                _ => unreachable!(),
            })
            .collect();
        let output = match op {
            0 => crate::run(&mut backend, |__s| {
                __s.exp_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            1 => crate::run(&mut backend, |__s| {
                __s.log_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            2 => crate::run(&mut backend, |__s| {
                __s.sin_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            3 => crate::run(&mut backend, |__s| {
                __s.cos_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            4 => crate::run(&mut backend, |__s| {
                __s.tanh_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            5 => crate::run(&mut backend, |__s| {
                __s.sqrt_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            6 => crate::run(&mut backend, |__s| {
                __s.rsqrt_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            7 => crate::run(&mut backend, |__s| {
                __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            8 => crate::run(&mut backend, |__s| {
                __s.log1p_read(TensorRead::from_tensor(&owned))
            })
            .unwrap(),
            _ => unreachable!(),
        }
        .unwrap();
        let owned_values = output.as_slice::<f64>().unwrap().to_vec();
        assert_within_one_ulp(&owned_values, &expected, "owned read against libm");
        crate::run(&mut backend, |__s| __s.reclaim_buffer(output)).unwrap();
        let read = TensorRead::from_view(TensorView::F64(reversed.clone()));
        let output = match op {
            0 => crate::run(&mut backend, |__s| __s.exp_read(read)).unwrap(),
            1 => crate::run(&mut backend, |__s| __s.log_read(read)).unwrap(),
            2 => crate::run(&mut backend, |__s| __s.sin_read(read)).unwrap(),
            3 => crate::run(&mut backend, |__s| __s.cos_read(read)).unwrap(),
            4 => crate::run(&mut backend, |__s| __s.tanh_read(read)).unwrap(),
            5 => crate::run(&mut backend, |__s| __s.sqrt_read(read)).unwrap(),
            6 => crate::run(&mut backend, |__s| __s.rsqrt_read(read)).unwrap(),
            7 => crate::run(&mut backend, |__s| __s.expm1_read(read)).unwrap(),
            8 => crate::run(&mut backend, |__s| __s.log1p_read(read)).unwrap(),
            _ => unreachable!(),
        }
        .unwrap();
        let reversed_values = output.as_slice::<f64>().unwrap();
        assert_within_one_ulp(
            reversed_values,
            &owned_values.iter().rev().copied().collect::<Vec<_>>(),
            "reversed read against the owned read",
        );
        assert_within_one_ulp(
            reversed_values,
            &expected.iter().rev().copied().collect::<Vec<_>>(),
            "reversed read against libm",
        );
        crate::run(&mut backend, |__s| __s.reclaim_buffer(output)).unwrap();
    }
}

#[test]
fn static_pow_replay_preserves_wrapping_and_domain_checks() {
    let mut backend = crate::engine(1);
    let bases = [2_i64, i64::MAX, i64::MIN, -1];
    let powers = [63_i64, 2, 1, 0];
    let lhs = Tensor::from_vec_col_major([4], bases.to_vec()).unwrap();
    let rhs = Tensor::from_vec_col_major([4], powers.to_vec()).unwrap();
    let expected: Vec<_> = bases
        .iter()
        .zip(powers)
        .map(|(&x, n)| x.wrapping_pow(n as u32))
        .collect();
    let out = crate::run(&mut backend, |__s| {
        __s.pow_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
    })
    .unwrap()
    .unwrap();
    assert_eq!(out.as_slice::<i64>().unwrap(), expected);
    crate::run(&mut backend, |__s| __s.reclaim_buffer(out)).unwrap();
    let out = crate::run(&mut backend, |__s| {
        __s.pow_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
    })
    .unwrap()
    .unwrap();
    assert_eq!(out.as_slice::<i64>().unwrap(), expected);
    crate::run(&mut backend, |__s| __s.reclaim_buffer(out)).unwrap();
    let negative = Tensor::from_vec_col_major([], vec![-1_i64]).unwrap();
    assert!(crate::run(&mut backend, |__s| __s.pow_read(
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&negative)
    ))
    .unwrap()
    .is_err());
    assert!(crate::run(&mut backend, |__s| __s.pow_read(
        TensorRead::from_tensor(&lhs),
        TensorRead::from_tensor(&negative)
    ))
    .unwrap()
    .is_err());
}

#[test]
fn ordinary_cpu_retains_supported_fusion_hook() {
    let mut backend = crate::engine(1);
    const N: usize = 16 * 1024;
    let x = Tensor::from_vec_col_major([N], vec![1.0_f64; N]).unwrap();
    let y = Tensor::from_vec_col_major([N], vec![2.0_f64; N]).unwrap();
    let z = Tensor::from_vec_col_major([N], vec![3.0_f64; N]).unwrap();
    let plan = ElementwiseFusionPlan::new(
        DType::F64,
        3,
        vec![3, 4],
        vec![
            ElementwiseFusionInst::new(ElementwiseFusionOp::Add, vec![0, 1]),
            ElementwiseFusionInst::new(ElementwiseFusionOp::Multiply, vec![3, 2]),
        ],
    );
    for _ in 0..3 {
        let outputs = crate::run(&mut backend, |__s| {
            __s.execute_elementwise_fusion(&[&x, &y, &z], &plan)
        })
        .unwrap()
        .unwrap()
        .expect("ordinary CPU must retain its supported fusion hook");
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].as_slice::<f64>().unwrap(), vec![3.0; N]);
        assert_eq!(outputs[1].as_slice::<f64>().unwrap(), vec![9.0; N]);
        for output in outputs {
            crate::run(&mut backend, |__s| __s.reclaim_buffer(output)).unwrap();
        }
    }
}
