//! #1858: the tensor-receiver session methods forward to the borrowed-session
//! operations unchanged. Each family compares the receiver form with the
//! session form on the same inputs.

use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{Tensor, TensorSessionOpsExt, TypedTensor, TypedTensorSessionOpsExt};
use tenferro_tensor::{
    BackendSession, BackendSessionHost, DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig,
    SliceConfig, TensorRead,
};

fn t(shape: &[usize], data: Vec<f64>) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data).unwrap()
}

fn same(receiver: &Tensor, session: &Tensor) {
    assert_eq!(receiver.shape(), session.shape());
    assert_eq!(receiver.dtype(), session.dtype());
    assert_eq!(
        format!("{:?}", receiver.as_slice::<f64>().ok()),
        format!("{:?}", session.as_slice::<f64>().ok())
    );
}

fn in_session<R: Send>(f: impl FnOnce(&mut dyn BackendSession) -> R + Send) -> R {
    CpuBackend::new().with_backend_session(f).unwrap()
}

#[test]
fn indexing_family_matches_the_session_form() {
    let x = t(&[4], vec![1.0, 2.0, 3.0, 4.0]);
    let idx = Tensor::from_vec_col_major(vec![2, 1], vec![3_i64, 1]).unwrap();
    let gather = GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    };
    let scatter = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    let updates = t(&[2], vec![9.0, 8.0]);
    let slice = SliceConfig {
        starts: vec![0],
        limits: vec![4],
        strides: vec![2],
    };
    let pad = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![0],
        interior_padding: vec![1],
    };
    let starts = Tensor::from_vec_col_major(vec![1], vec![2_i64]).unwrap();
    in_session(|s| {
        same(
            &x.gather(&idx, gather.clone(), s).unwrap(),
            &s.gather(&x, &idx, &gather).unwrap(),
        );
        same(
            &x.scatter(&idx, &updates, scatter.clone(), s).unwrap(),
            &s.scatter(&x, &idx, &updates, &scatter).unwrap(),
        );
        same(
            &x.slice(slice.clone(), s).unwrap(),
            &s.slice(&x, &slice).unwrap(),
        );
        same(
            &x.dynamic_slice(&starts, &[2], s).unwrap(),
            &s.dynamic_slice(&x, &starts, &[2]).unwrap(),
        );
        same(&x.pad(pad.clone(), s).unwrap(), &s.pad(&x, &pad).unwrap());
        same(
            &Tensor::concatenate(&[&x, &x], 0, s).unwrap(),
            &s.concatenate(&[&x, &x], 0).unwrap(),
        );
        same(&x.reverse(&[0], s).unwrap(), &s.reverse(&x, &[0]).unwrap());
    });
}

#[test]
fn reduction_family_matches_the_session_form() {
    let x = t(&[2, 3], vec![1.0, -2.0, 3.0, 0.5, -1.5, 4.0]);
    let tx =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, -2.0, 3.0, 0.5, -1.5, 4.0])
            .unwrap();
    in_session(|s| {
        let read = || TensorRead::from_tensor(&x);
        for axes in [vec![0], vec![1], vec![0, 1]] {
            same(
                &x.reduce_max(Some(&axes), s).unwrap(),
                &s.reduce_max_read(read(), &axes).unwrap(),
            );
            same(
                &x.reduce_min(Some(&axes), s).unwrap(),
                &s.reduce_min_read(read(), &axes).unwrap(),
            );
            same(
                &x.reduce_prod(Some(&axes), s).unwrap(),
                &s.reduce_prod_read(read(), &axes).unwrap(),
            );
            same(
                &x.reduce_sum_squares(Some(&axes), s).unwrap(),
                &s.reduce_sum_squares_read(read(), &axes).unwrap(),
            );
            let typed = tx.reduce_max(Some(&axes), s).unwrap();
            same(
                &Tensor::from_typed(typed),
                &s.reduce_max_read(read(), &axes).unwrap(),
            );
            let typed = tx.reduce_sum_squares(Some(&axes), s).unwrap();
            same(
                &Tensor::from_typed(typed),
                &s.reduce_sum_squares_read(read(), &axes).unwrap(),
            );
        }
        // `None` is every axis.
        same(
            &x.reduce_prod(None, s).unwrap(),
            &s.reduce_prod_read(read(), &[0, 1]).unwrap(),
        );
        same(
            &Tensor::from_typed(tx.reduce_min(None, s).unwrap()),
            &s.reduce_min_read(read(), &[0, 1]).unwrap(),
        );
        same(
            &Tensor::from_typed(tx.reduce_prod(Some(&[]), s).unwrap()),
            &s.reduce_prod_read(read(), &[]).unwrap(),
        );
    });
}

#[test]
fn structural_family_matches_the_session_form() {
    let m = t(&[3, 3], (1..=9).map(f64::from).collect());
    let v = t(&[3], vec![1.0, 2.0, 3.0]);
    in_session(|s| {
        same(
            &v.broadcast_in_dim(&[3, 2], &[0], s).unwrap(),
            &s.broadcast_in_dim_read(TensorRead::from_tensor(&v), &[3, 2], &[0])
                .unwrap(),
        );
        for k in [-1, 0, 1] {
            same(&m.tril(k, s).unwrap(), &s.tril(&m, k).unwrap());
            same(&m.triu(k, s).unwrap(), &s.triu(&m, k).unwrap());
        }
        same(
            &m.extract_diag(0, 1, s).unwrap(),
            &s.extract_diagonal(&m, 0, 1).unwrap(),
        );
        same(
            &v.embed_diag(0, 1, s).unwrap(),
            &s.embed_diagonal(&v, 0, 1).unwrap(),
        );
    });
}

#[test]
fn dot_family_matches_the_session_form() {
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    };
    let lhs = t(&[2, 3, 2], (0..12).map(f64::from).collect());
    let rhs = t(&[3, 2, 2], (0..12).map(|v| f64::from(v) * 0.5).collect());
    let tl =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3, 2], (0..12).map(f64::from).collect())
            .unwrap();
    let tr = TypedTensor::<f64>::from_vec_col_major(
        vec![3, 2, 2],
        (0..12).map(|v| f64::from(v) * 0.5).collect(),
    )
    .unwrap();
    let cl = Tensor::from_vec_col_major(
        vec![1, 2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.5)],
    )
    .unwrap();
    let cr = Tensor::from_vec_col_major(
        vec![2, 1],
        vec![Complex64::new(0.5, -1.0), Complex64::new(2.0, 1.0)],
    )
    .unwrap();
    let matrix = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [].as_slice().into(),
        rhs_batch_dims: [].as_slice().into(),
    };
    in_session(|s| {
        let expected = s
            .dot_general_read(
                TensorRead::from_tensor(&lhs),
                TensorRead::from_tensor(&rhs),
                &config,
            )
            .unwrap();
        same(
            &lhs.dot_general(&rhs, config.clone(), s).unwrap(),
            &expected,
        );
        same(
            &Tensor::from_typed(tl.dot_general(&tr, config.clone(), s).unwrap()),
            &expected,
        );
        for (lc, rc) in [(false, false), (true, false), (false, true), (true, true)] {
            let receiver = cl
                .dot_general_with_conj(&cr, matrix.clone(), lc, rc, s)
                .unwrap();
            let session = s.dot_general_with_conj(&cl, &cr, &matrix, lc, rc).unwrap();
            assert_eq!(
                receiver.as_slice::<Complex64>().unwrap(),
                session.as_slice::<Complex64>().unwrap()
            );
            let typed_l = TypedTensor::<Complex64>::from_vec_col_major(
                vec![1, 2],
                cl.as_slice::<Complex64>().unwrap().to_vec(),
            )
            .unwrap();
            let typed_r = TypedTensor::<Complex64>::from_vec_col_major(
                vec![2, 1],
                cr.as_slice::<Complex64>().unwrap().to_vec(),
            )
            .unwrap();
            let typed = typed_l
                .dot_general_with_conj(&typed_r, matrix.clone(), lc, rc, s)
                .unwrap();
            assert_eq!(
                typed.host_data().unwrap(),
                session.as_slice::<Complex64>().unwrap()
            );
        }
    });
}

#[test]
fn scaling_family_matches_multiplication_by_a_rank0_factor() {
    let x = t(&[3], vec![1.0, -2.0, 0.5]);
    let ints = Tensor::from_vec_col_major(vec![2], vec![3_i64, -4]).unwrap();
    let c = Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)]).unwrap();
    let tx = TypedTensor::<f64>::from_vec_col_major(vec![3], vec![1.0, -2.0, 0.5]).unwrap();
    let tc = TypedTensor::<Complex64>::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)])
        .unwrap();
    in_session(|s| {
        assert_eq!(
            x.scale_real(-2.0, s).unwrap().as_slice::<f64>().unwrap(),
            &[-2.0, 4.0, -1.0]
        );
        assert_eq!(
            tx.scale_real(-2.0, s).unwrap().host_data().unwrap(),
            &[-2.0, 4.0, -1.0]
        );
        // Integer factors round, as on the eager surface.
        assert_eq!(
            ints.scale_real(1.6, s).unwrap().as_slice::<i64>().unwrap(),
            &[6, -8]
        );
        assert!(ints.scale_real(f64::NAN, s).is_err());
        let i = Complex64::new(0.0, 1.0);
        assert_eq!(
            c.scale_complex(i, s)
                .unwrap()
                .as_slice::<Complex64>()
                .unwrap(),
            &[Complex64::new(-2.0, 1.0)]
        );
        assert_eq!(
            tc.scale_complex(i, s).unwrap().host_data().unwrap(),
            &[Complex64::new(-2.0, 1.0)]
        );
        assert!(x.scale_complex(i, s).is_err());
        assert!(tx.scale_complex(i, s).is_err());
    });
}

#[test]
fn scale_factors_follow_the_shared_dtype_rules() {
    use num_complex::Complex32;
    use tenferro_runtime::scale::{complex_scale_scalar, real_scale_scalar};
    use tenferro_tensor::DType;

    assert_eq!(
        real_scale_scalar(DType::F32, 0.5)
            .unwrap()
            .as_slice::<f32>()
            .unwrap(),
        &[0.5]
    );
    assert_eq!(
        real_scale_scalar(DType::I32, -2.5)
            .unwrap()
            .as_slice::<i32>()
            .unwrap(),
        &[-3]
    );
    assert_eq!(
        real_scale_scalar(DType::Bool, 0.0)
            .unwrap()
            .as_slice::<bool>()
            .unwrap(),
        &[false]
    );
    assert_eq!(
        real_scale_scalar(DType::C32, 2.0)
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap(),
        &[Complex32::new(2.0, 0.0)]
    );
    assert_eq!(
        real_scale_scalar(DType::C64, 2.0)
            .unwrap()
            .as_slice::<Complex64>()
            .unwrap(),
        &[Complex64::new(2.0, 0.0)]
    );
    assert!(real_scale_scalar(DType::I32, 1e10).is_err());
    assert!(real_scale_scalar(DType::I64, 1e30).is_err());
    assert!(real_scale_scalar(DType::Bool, f64::INFINITY).is_err());
    assert_eq!(
        complex_scale_scalar(DType::C32, Complex64::new(1.0, -1.0))
            .unwrap()
            .as_slice::<Complex32>()
            .unwrap(),
        &[Complex32::new(1.0, -1.0)]
    );
    assert!(complex_scale_scalar(DType::I64, Complex64::new(1.0, 0.0)).is_err());
}
