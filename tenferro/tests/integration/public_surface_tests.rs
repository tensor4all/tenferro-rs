use tenferro::{ScalarType, Tensor};

#[test]
fn tensor_from_slice_reports_dtype_shape_and_layout_flags() {
    let value = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

    assert_eq!(value.scalar_type(), ScalarType::F64);
    assert_eq!(value.dims(), &[2, 2]);
    assert!(value.is_dense());
    assert!(!value.is_diag());
    assert_eq!(value.try_to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_detach_drops_reverse_tracking() {
    let value = Tensor::from_slice(&[1.0_f64, 2.0], &[2])
        .unwrap()
        .with_requires_grad(true);

    let diag = Tensor::diag(&Tensor::from_tensor(vector_f64(&[3.0, 4.0]))).unwrap();
    assert!(diag.is_diag());
    assert_eq!(diag.dims(), &[2, 2]);
}

#[test]
fn tensor_public_surface_reexports_memory_order() {
    let dense = DenseTensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    assert_eq!(dense.dims(), &[2]);
}

#[test]
fn tensor_typed_accessor_exposes_scalar_type_via_wrapper() {
    let x = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    assert_eq!(x.as_f64().unwrap().scalar_type(), ScalarType::F64);
}

#[test]
fn tensor_debug_includes_dense_value_preview() {
    let x = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    let rendered = format!("{x:?}");
    assert!(rendered.contains("Tensor"));
    assert!(rendered.contains("scalar_type: F64"));
    assert!(rendered.contains("dims: [2]"));
    assert!(rendered.contains("axis_classes: [0]"));
    assert!(rendered.contains("mode: Primal"));
    assert!(rendered.contains("is_dense: true"));
    assert!(rendered.contains("is_diag: true"));
    assert!(rendered.contains("preview"));
    assert!(rendered.contains("[1.0, 2.0]"));
}

#[test]
fn tensor_debug_structured_preview_uses_logical_values() {
    let x = Tensor::diag(&Tensor::from_tensor(vector_f64(&[3.0, 4.0]))).unwrap();
    let rendered = format!("{x:?}");
    assert!(rendered.contains("dims: [2, 2]"));
    assert!(rendered.contains("axis_classes: [0, 0]"));
    assert!(rendered.contains("is_dense: false"));
    assert!(rendered.contains("is_diag: true"));
    assert!(rendered.contains("preview"));
    assert!(rendered.contains("[[3.0, 0.0], [0.0, 4.0]]"));
}

#[test]
fn tensor_debug_large_tensor_omits_preview_values() {
    let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let x = Tensor::from_tensor(
        DenseTensor::from_slice(&values, &[20], MemoryOrder::ColumnMajor).unwrap(),
    );
    let rendered = format!("{x:?}");
    assert!(rendered.contains("dims: [20]"));
    assert!(rendered.contains("preview: <omitted: 20 logical values>"));
    assert!(!rendered.contains("[0.0, 1.0, 2.0"));
}

#[test]
fn tensor_public_forward_constructor_preserves_tangent() {
    let x = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    let dx = Tensor::from_tensor(vector_f64(&[0.5, -0.5]));

    let (primal, tangent) = forward_ad::dual_level(|fw| {
        let dual = fw.make_dual(&x, &dx)?;
        fw.unpack_dual(&dual)
    })
    .unwrap();

    assert_eq!(primal.dims(), &[2]);
    assert_eq!(
        tangent
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.5, -0.5]
    );
}

#[test]
fn tensor_public_reverse_api_tracks_requested_gradients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(scalar_f64(2.0))
        .with_requires_grad(true)
        .unwrap();
    let out = x.exp().unwrap();
    backward(&[&out], None, &[&x], BackwardOptions::default()).unwrap();
    assert!(x.requires_grad());
    assert!(x.grad().unwrap().is_some());
    assert!(x.is_leaf());
}

#[test]
fn tensor_with_requires_grad_detaches_nonleaf_outputs_into_new_leafs() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(scalar_f64(2.0))
        .with_requires_grad(true)
        .unwrap();
    let y = x.exp().unwrap();
    let z = y.with_requires_grad(true).unwrap();
    let out = z.exp().unwrap();

    backward(&[&out], None, &[&z], BackwardOptions::default()).unwrap();

    assert!(x.grad().unwrap().is_none());
    assert!(z.grad().unwrap().is_some());
    assert!(z.is_leaf());
}

#[test]
fn tensor_reverse_add_joins_independent_nonleaf_graphs_without_helper() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(scalar_f64(2.0))
        .with_requires_grad(true)
        .unwrap();
    let y = Tensor::from_tensor(scalar_f64(3.0))
        .with_requires_grad(true)
        .unwrap();

    let out_x = x.exp().unwrap();
    let out_y = y.exp().unwrap();
    let out = out_x.add(&out_y).unwrap();

    backward(&[&out], None, &[&x, &y], BackwardOptions::default()).unwrap();

    let grad_x_tensor = x.grad().unwrap().unwrap();
    let grad_y_tensor = y.grad().unwrap().unwrap();
    let grad_x = grad_x_tensor.as_f64().unwrap();
    let grad_y = grad_y_tensor.as_f64().unwrap();
    let values_x = grad_x.primal().buffer().as_slice().unwrap();
    let values_y = grad_y.primal().buffer().as_slice().unwrap();

    assert!((values_x[0] - 2.0_f64.exp()).abs() < 1e-12);
    assert!((values_y[0] - 3.0_f64.exp()).abs() < 1e-12);
}

#[test]
fn tensor_reverse_einsum_joins_independent_nonleaf_graphs_without_helper() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(vector_f64(&[1.0, 2.0]))
        .with_requires_grad(true)
        .unwrap();
    let y = Tensor::from_tensor(vector_f64(&[3.0, 4.0]))
        .with_requires_grad(true)
        .unwrap();

    let out_x = x.exp().unwrap();
    let out_y = y.exp().unwrap();
    let loss = Tensor::einsum("i,i->", &[&out_x, &out_y]).unwrap();

    backward(&[&loss], None, &[&x, &y], BackwardOptions::default()).unwrap();

    let grad_x = x
        .grad()
        .unwrap()
        .unwrap()
        .as_f64()
        .unwrap()
        .primal()
        .clone();
    let grad_y = y
        .grad()
        .unwrap()
        .unwrap()
        .as_f64()
        .unwrap()
        .primal()
        .clone();
    let values_x = grad_x.buffer().as_slice().unwrap();
    let values_y = grad_y.buffer().as_slice().unwrap();

    let expected = [4.0_f64.exp(), 6.0_f64.exp()];
    for (actual, want) in values_x.iter().zip(expected) {
        assert!((*actual - want).abs() < 1e-12);
    }
    for (actual, want) in values_y.iter().zip(expected) {
        assert!((*actual - want).abs() < 1e-12);
    }
}

#[test]
fn tensor_reverse_qr_outputs_join_independent_nonleaf_graphs_without_helper() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(
        // Keep exp(x) full-rank so QR backward does not hit a singular R.
        DenseTensor::<f64>::from_slice(&[1.0, 3.0, 1.5, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
    .with_requires_grad(true)
    .unwrap();
    let y = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[2.0, 1.0, 0.5, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
    .with_requires_grad(true)
    .unwrap();

    let left = x.exp().unwrap().qr().unwrap().r;
    let right = y.exp().unwrap().qr().unwrap().r;
    let loss = Tensor::einsum("ij,ij->", &[&left, &right]).unwrap();

    backward(&[&loss], None, &[&x, &y], BackwardOptions::default()).unwrap();

    assert!(x.grad().unwrap().is_some());
    assert!(y.grad().unwrap().is_some());
}

#[test]
fn tensor_reverse_svd_outputs_join_independent_nonleaf_graphs_without_helper() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
    .with_requires_grad(true)
    .unwrap();
    let y = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[2.0, 1.0, 0.5, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    )
    .with_requires_grad(true)
    .unwrap();

    let left = x.exp().unwrap().svd().unwrap().vt;
    let right = y.exp().unwrap().svd().unwrap().vt;
    let loss = Tensor::einsum("ij,ij->", &[&left, &right]).unwrap();

    backward(&[&loss], None, &[&x, &y], BackwardOptions::default()).unwrap();

    assert!(x.grad().unwrap().is_some());
    assert!(y.grad().unwrap().is_some());
}

#[test]
fn tensor_public_rank0_complex_scale_does_not_require_adtensor() {
    let x = Tensor::from_tensor(scalar_f64(2.0));
    let alpha = Tensor::from_tensor(
        DenseTensor::from_slice(&[Complex64::new(0.0, 3.0)], &[], MemoryOrder::ColumnMajor)
            .unwrap(),
    );

    let y = x.scale(&alpha).unwrap();
    assert!(y.dims().is_empty());
    assert_eq!(
        y.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(0.0, 6.0)]
    );
}

#[test]
fn tensor_public_to_scalar_type_supports_cross_precision_cast() {
    let x = Tensor::from_tensor(scalar_f64(2.0));
    let y = x.to_scalar_type(ScalarType::F32).unwrap();
    assert_eq!(y.scalar_type(), ScalarType::F32);
    assert_eq!(
        y.as_f32().unwrap().primal().buffer().as_slice().unwrap(),
        &[2.0]
    );

    let detached = y.detach();
    assert_eq!(detached.scalar_type(), ScalarType::F32);
    assert!(!detached.requires_grad());
}

#[test]
fn tensor_public_to_scalar_type_supports_all_pairs_and_preserves_dense_layout() {
    let real32 = Tensor::from_tensor(vector_f32(&[1.5, -2.0]));
    let real64 = Tensor::from_tensor(vector_f64(&[2.5, -3.0]));
    let complex32 = Tensor::from_tensor(vector_c32(&[
        Complex32::new(3.0, -4.0),
        Complex32::new(-1.0, 2.0),
    ]));
    let complex64 = Tensor::from_tensor(vector_c64(&[
        Complex64::new(-2.0, 5.0),
        Complex64::new(1.0, -3.0),
    ]));

    assert_cast_preserves_layout(&real32, ScalarType::F64, ScalarType::F64, |cast| {
        assert_cast_values(cast, &[1.5, -2.0]);
    });
    assert_cast_preserves_layout(&real32, ScalarType::C32, ScalarType::C32, |cast| {
        assert_cast_values_c32(cast, &[Complex32::new(1.5, 0.0), Complex32::new(-2.0, 0.0)]);
    });
    assert_cast_preserves_layout(&real32, ScalarType::C64, ScalarType::C64, |cast| {
        assert_cast_values_c64(cast, &[Complex64::new(1.5, 0.0), Complex64::new(-2.0, 0.0)]);
    });

    assert_cast_preserves_layout(&real64, ScalarType::F32, ScalarType::F32, |cast| {
        assert_cast_values_f32(cast, &[2.5, -3.0]);
    });
    assert_cast_preserves_layout(&real64, ScalarType::C32, ScalarType::C32, |cast| {
        assert_cast_values_c32(cast, &[Complex32::new(2.5, 0.0), Complex32::new(-3.0, 0.0)]);
    });
    assert_cast_preserves_layout(&real64, ScalarType::C64, ScalarType::C64, |cast| {
        assert_cast_values_c64(cast, &[Complex64::new(2.5, 0.0), Complex64::new(-3.0, 0.0)]);
    });

    assert_cast_preserves_layout(&complex32, ScalarType::F32, ScalarType::F32, |cast| {
        assert_cast_values_f32(cast, &[3.0, -1.0]);
    });
    assert_cast_preserves_layout(&complex32, ScalarType::F64, ScalarType::F64, |cast| {
        assert_cast_values(cast, &[3.0, -1.0]);
    });
    assert_cast_preserves_layout(&complex32, ScalarType::C64, ScalarType::C64, |cast| {
        assert_cast_values_c64(
            cast,
            &[Complex64::new(3.0, -4.0), Complex64::new(-1.0, 2.0)],
        );
    });

    assert_cast_preserves_layout(&complex64, ScalarType::F32, ScalarType::F32, |cast| {
        assert_cast_values_f32(cast, &[-2.0, 1.0]);
    });
    assert_cast_preserves_layout(&complex64, ScalarType::F64, ScalarType::F64, |cast| {
        assert_cast_values(cast, &[-2.0, 1.0]);
    });
    assert_cast_preserves_layout(&complex64, ScalarType::C32, ScalarType::C32, |cast| {
        assert_cast_values_c32(
            cast,
            &[Complex32::new(-2.0, 5.0), Complex32::new(1.0, -3.0)],
        );
    });
}

#[test]
fn tensor_public_to_scalar_type_preserves_diag_axis_classes() {
    let diag_real = diag_f32(&[1.0, -2.0]);
    let diag_complex = diag_c64(&[Complex64::new(2.0, 1.0), Complex64::new(-3.0, 0.5)]);

    assert_cast_preserves_layout(&diag_real, ScalarType::C64, ScalarType::C64, |cast| {
        assert_cast_values_c64(cast, &[Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)]);
    });
    assert_cast_preserves_layout(&diag_complex, ScalarType::F32, ScalarType::F32, |cast| {
        assert_cast_values_f32(cast, &[2.0, -3.0]);
    });
}

#[test]
fn tensor_public_scalar_eager_methods_do_not_require_typed_api() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(vector_f64(&[0.0, 1.0]));
    let y = x.exp().unwrap();
    assert_eq!(y.scalar_type(), tenferro::ScalarType::F64);
    let y_vals = y.as_f64().unwrap().primal().buffer().as_slice().unwrap();
    assert!((y_vals[0] - 1.0).abs() < 1e-12);
    assert!((y_vals[1] - std::f64::consts::E).abs() < 1e-12);

    let a = Tensor::from_tensor(scalar_f64(2.0));
    let b = Tensor::from_tensor(scalar_f64(3.0));
    let c = a.add(&b).unwrap();
    assert_eq!(
        c.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[5.0]
    );

    let m = x.mean().unwrap();
    assert!(m.dims().is_empty());
    assert_eq!(
        m.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[0.5]
    );

    let s = x.sum().unwrap();
    assert!(s.dims().is_empty());
    assert_eq!(
        s.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[1.0]
    );

    let t = x.sin().unwrap().cos().unwrap().tanh().unwrap();
    assert_eq!(t.scalar_type(), tenferro::ScalarType::F64);

    let v = x.var().unwrap();
    assert!(v.dims().is_empty());

    let std = x.std().unwrap();
    assert!(std.dims().is_empty());
}

#[test]
fn tensor_public_einsum_uses_dynamic_operands_only() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );
    let b = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
            &[2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );

    let out = Tensor::einsum("i,i->", &[&a, &b]).unwrap();
    assert_eq!(out.scalar_type(), tenferro::ScalarType::C64);
    assert_eq!(
        out.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(5.0, -1.0)]
    );
}

#[test]
fn tensor_public_einsum_owned_accepts_owned_dynamic_operands() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let operands = vec![
        Tensor::from_tensor(
            DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
        ),
        Tensor::from_tensor(
            DenseTensor::<Complex64>::from_slice(
                &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
                &[2],
                MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        ),
    ];

    let out = Tensor::einsum_owned("i,i->", operands).unwrap();
    assert_eq!(out.scalar_type(), tenferro::ScalarType::C64);
    assert_eq!(
        out.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(5.0, -1.0)]
    );
}

#[test]
fn tensor_public_einsum_transposed_scalar_contraction_matches_manual() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let lhs = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[
                Complex64::new(1.0, 0.2),
                Complex64::new(-0.5, 0.4),
                Complex64::new(2.0, -0.1),
                Complex64::new(0.3, -1.0),
                Complex64::new(1.2, 0.7),
                Complex64::new(-0.8, 0.5),
            ],
            &[2, 3],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    let rhs = Tensor::from_tensor(
        DenseTensor::<Complex64>::from_slice(
            &[
                Complex64::new(0.5, -0.1),
                Complex64::new(1.0, 0.6),
                Complex64::new(-1.2, 0.3),
                Complex64::new(0.7, -0.9),
                Complex64::new(0.2, 1.1),
                Complex64::new(-0.4, 0.8),
            ],
            &[3, 2],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );

    let out = Tensor::einsum("ab,ba->", &[&lhs, &rhs]).unwrap();
    let actual = match out.try_scalar_value().unwrap() {
        tenferro::ScalarValue::C64(z) => z,
        other => panic!("expected C64 scalar, got {other:?}"),
    };

    let lhs_vals = lhs.as_c64().unwrap().primal().buffer().as_slice().unwrap();
    let rhs_vals = rhs.as_c64().unwrap().primal().buffer().as_slice().unwrap();
    let mut expected = Complex64::new(0.0, 0.0);
    for a in 0..2 {
        for b in 0..3 {
            expected += lhs_vals[a + 2 * b] * rhs_vals[b + 3 * a];
        }
    }

    assert!(
        (actual - expected).norm() < 1e-12,
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn tensor_public_linalg_single_result_methods_do_not_require_typed_api() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[4.0, 1.0, 1.0, 3.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );
    let b = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
    );

    let det = a.det().unwrap();
    assert!(det.dims().is_empty());
    assert_eq!(
        det.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[11.0]
    );

    let x = a.solve(&b).unwrap();
    let x_vals = x.as_f64().unwrap().primal().buffer().as_slice().unwrap();
    assert!((x_vals[0] - (1.0 / 11.0)).abs() < 1e-12);
    assert!((x_vals[1] - (7.0 / 11.0)).abs() < 1e-12);
}

#[test]
fn tensor_public_linalg_multi_result_methods_return_dynamic_results() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let a = Tensor::from_tensor(
        DenseTensor::<f64>::from_slice(&[1.0, 0.0, 0.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor)
            .unwrap(),
    );

    let svd = a.svd().unwrap();
    assert_eq!(svd.u.scalar_type(), tenferro::ScalarType::F64);
    assert_eq!(svd.s.dims(), &[2]);
    assert_eq!(svd.vt.dims(), &[2, 2]);

    let qr = a.qr().unwrap();
    assert_eq!(qr.q.dims(), &[2, 2]);
    assert_eq!(qr.r.dims(), &[2, 2]);
}

#[test]
fn tensor_public_pullback_wrt_does_not_require_typed_api() {
    let mut x = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    let mut a = Tensor::from_tensor(scalar_f64(3.0));
    x.set_requires_grad(true).unwrap();
    a.set_requires_grad(true).unwrap();
    let out = x.scale(&a).unwrap();
    let cotangent = Tensor::from_tensor(vector_f64(&[0.5, 1.25]));

    out.backward(Some(&cotangent), &[&x, &a], BackwardOptions::default())
        .unwrap();
    assert_eq!(
        x.grad()
            .unwrap()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.5, 3.75]
    );
    assert_eq!(
        a.grad()
            .unwrap()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.0]
    );
}

#[test]
fn tensor_public_backward_handles_edge_outputs_without_typed_api() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let x = Tensor::from_tensor(scalar_f64(2.0))
        .with_requires_grad(true)
        .unwrap();
    let y = Tensor::from_tensor(scalar_f64(3.0))
        .with_requires_grad(true)
        .unwrap();
    let out = x.exp().unwrap().add(&y.exp().unwrap()).unwrap();
    let cotangent = Tensor::from_tensor(scalar_f64(1.5));

    out.backward(Some(&cotangent), &[&x, &y], BackwardOptions::default())
        .unwrap();

    let grad_x = x.grad().unwrap().unwrap();
    let grad_y = y.grad().unwrap().unwrap();
    let grad_x_values = grad_x
        .as_f64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap();
    let grad_y_values = grad_y
        .as_f64()
        .unwrap()
        .primal()
        .buffer()
        .as_slice()
        .unwrap();

    assert!((grad_x_values[0] - 1.5 * 2.0_f64.exp()).abs() < 1e-12);
    assert!((grad_y_values[0] - 1.5 * 3.0_f64.exp()).abs() < 1e-12);
}
