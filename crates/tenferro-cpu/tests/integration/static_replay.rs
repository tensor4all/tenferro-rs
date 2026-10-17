use tenferro_cpu::CpuBackend;
use tenferro_tensor::backend::{ElementwiseFusionInst, ElementwiseFusionOp, ElementwiseFusionPlan};
use tenferro_tensor::{
    BackendSessionHost, DType, StridedSliceSpec, Tensor, TensorRead, TensorView, TypedTensor,
};

#[test]
fn static_analytic_replay_preserves_owned_and_reversed_values() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
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
            0 => backend
                .with_backend_session(|__s| __s.exp_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            1 => backend
                .with_backend_session(|__s| __s.log_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            2 => backend
                .with_backend_session(|__s| __s.sin_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            3 => backend
                .with_backend_session(|__s| __s.cos_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            4 => backend
                .with_backend_session(|__s| __s.tanh_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            5 => backend
                .with_backend_session(|__s| __s.sqrt_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            6 => backend
                .with_backend_session(|__s| __s.rsqrt_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            7 => backend
                .with_backend_session(|__s| {
                    __s.expm1_read(tenferro_tensor::TensorRead::from_tensor(&owned))
                })
                .unwrap(),
            8 => backend
                .with_backend_session(|__s| __s.log1p_read(TensorRead::from_tensor(&owned)))
                .unwrap(),
            _ => unreachable!(),
        }
        .unwrap();
        assert_eq!(output.as_slice::<f64>().unwrap(), expected);
        backend
            .with_backend_session(|__s| __s.reclaim_buffer(output))
            .unwrap();
        let read = TensorRead::from_view(TensorView::F64(reversed.clone()));
        let output = match op {
            0 => backend
                .with_backend_session(|__s| __s.exp_read(read))
                .unwrap(),
            1 => backend
                .with_backend_session(|__s| __s.log_read(read))
                .unwrap(),
            2 => backend
                .with_backend_session(|__s| __s.sin_read(read))
                .unwrap(),
            3 => backend
                .with_backend_session(|__s| __s.cos_read(read))
                .unwrap(),
            4 => backend
                .with_backend_session(|__s| __s.tanh_read(read))
                .unwrap(),
            5 => backend
                .with_backend_session(|__s| __s.sqrt_read(read))
                .unwrap(),
            6 => backend
                .with_backend_session(|__s| __s.rsqrt_read(read))
                .unwrap(),
            7 => backend
                .with_backend_session(|__s| __s.expm1_read(read))
                .unwrap(),
            8 => backend
                .with_backend_session(|__s| __s.log1p_read(read))
                .unwrap(),
            _ => unreachable!(),
        }
        .unwrap();
        assert_eq!(
            output.as_slice::<f64>().unwrap(),
            expected.into_iter().rev().collect::<Vec<_>>()
        );
        backend
            .with_backend_session(|__s| __s.reclaim_buffer(output))
            .unwrap();
    }
}

#[test]
fn static_pow_replay_preserves_wrapping_and_domain_checks() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let bases = [2_i64, i64::MAX, i64::MIN, -1];
    let powers = [63_i64, 2, 1, 0];
    let lhs = Tensor::from_vec_col_major([4], bases.to_vec()).unwrap();
    let rhs = Tensor::from_vec_col_major([4], powers.to_vec()).unwrap();
    let expected: Vec<_> = bases
        .iter()
        .zip(powers)
        .map(|(&x, n)| x.wrapping_pow(n as u32))
        .collect();
    let out = backend
        .with_backend_session(|__s| {
            __s.pow_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        })
        .unwrap()
        .unwrap();
    assert_eq!(out.as_slice::<i64>().unwrap(), expected);
    backend
        .with_backend_session(|__s| __s.reclaim_buffer(out))
        .unwrap();
    let out = backend
        .with_backend_session(|__s| {
            __s.pow_read(TensorRead::from_tensor(&lhs), TensorRead::from_tensor(&rhs))
        })
        .unwrap()
        .unwrap();
    assert_eq!(out.as_slice::<i64>().unwrap(), expected);
    backend
        .with_backend_session(|__s| __s.reclaim_buffer(out))
        .unwrap();
    let negative = Tensor::from_vec_col_major([], vec![-1_i64]).unwrap();
    assert!(backend
        .with_backend_session(|__s| __s.pow_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&negative)
        ))
        .unwrap()
        .is_err());
    assert!(backend
        .with_backend_session(|__s| __s.pow_read(
            TensorRead::from_tensor(&lhs),
            TensorRead::from_tensor(&negative)
        ))
        .unwrap()
        .is_err());
}

#[test]
fn ordinary_cpu_retains_supported_fusion_hook() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    // Small inputs deliberately use the unfused path; exercise the existing
    // minimum fusion size rather than requiring fusion for every tensor.
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
        let outputs = backend
            .with_backend_session(|__s| __s.execute_elementwise_fusion(&[&x, &y, &z], &plan))
            .unwrap()
            .unwrap()
            .expect("ordinary CPU must retain its supported fusion hook");
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].as_slice::<f64>().unwrap(), vec![3.0; N]);
        assert_eq!(outputs[1].as_slice::<f64>().unwrap(), vec![9.0; N]);
        for output in outputs {
            backend
                .with_backend_session(|__s| __s.reclaim_buffer(output))
                .unwrap();
        }
    }
}
