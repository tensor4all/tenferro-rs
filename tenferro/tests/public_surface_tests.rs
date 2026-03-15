use num_complex::{Complex32, Complex64};
use tenferro::{
    backward, forward_ad, set_default_runtime, BackwardOptions, RuntimeContext, ScalarType, Tensor,
};
use tenferro_prims::CpuContext;
use tenferro_tensor::{MemoryOrder, Tensor as DenseTensor};

fn scalar_f64(value: f64) -> DenseTensor<f64> {
    DenseTensor::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_f64(values: &[f64]) -> DenseTensor<f64> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_f32(values: &[f32]) -> DenseTensor<f32> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_c32(values: &[Complex32]) -> DenseTensor<Complex32> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn vector_c64(values: &[Complex64]) -> DenseTensor<Complex64> {
    DenseTensor::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn diag_f32(values: &[f32]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector_f32(values))).unwrap()
}

fn diag_c64(values: &[Complex64]) -> Tensor {
    Tensor::diag(&Tensor::from_tensor(vector_c64(values))).unwrap()
}

fn assert_cast_values(tensor: &Tensor, expected: &[f64]) {
    assert_eq!(
        tensor
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        expected
    );
}

fn assert_cast_values_f32(tensor: &Tensor, expected: &[f32]) {
    assert_eq!(
        tensor
            .as_f32()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        expected
    );
}

fn assert_cast_values_c32(tensor: &Tensor, expected: &[Complex32]) {
    assert_eq!(
        tensor
            .as_c32()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        expected
    );
}

fn assert_cast_values_c64(tensor: &Tensor, expected: &[Complex64]) {
    assert_eq!(
        tensor
            .as_c64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        expected
    );
}

fn assert_cast_preserves_layout(
    source: &Tensor,
    target: ScalarType,
    expected_type: ScalarType,
    assert_values: impl Fn(&Tensor),
) {
    let cast = source.to_scalar_type(target).unwrap();
    assert_eq!(cast.scalar_type(), expected_type);
    assert_eq!(cast.dims(), source.dims());
    assert_eq!(cast.axis_classes(), source.axis_classes());
    assert_eq!(cast.is_dense(), source.is_dense());
    assert_eq!(cast.is_diag(), source.is_diag());
    assert_values(&cast);
}

#[test]
fn tensor_public_primal_constructor_handles_dense_and_diag() {
    let dense = Tensor::from_tensor(vector_f64(&[1.0, 2.0]));
    assert!(dense.is_dense());
    assert_eq!(dense.dims(), &[2]);

    let diag = Tensor::diag(&Tensor::from_tensor(vector_f64(&[3.0, 4.0]))).unwrap();
    assert!(diag.is_diag());
    assert_eq!(diag.dims(), &[2, 2]);
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
    let mut x = Tensor::from_tensor(scalar_f64(2.0));
    x.set_requires_grad(true).unwrap();
    let out = x.exp().unwrap();
    backward(&[&out], None, &[&x], BackwardOptions::default()).unwrap();
    assert!(x.requires_grad());
    assert!(x.grad().is_some());
}

#[test]
fn tensor_public_rank0_complex_scale_does_not_require_adtensor() {
    let x = Tensor::from_tensor(scalar_f64(2.0));
    let alpha = Tensor::from_tensor(
        DenseTensor::from_slice(&[Complex64::new(0.0, 3.0)], &[], MemoryOrder::ColumnMajor)
            .unwrap(),
    );

    let y = x.scale(&alpha).unwrap();
    assert_eq!(y.dims(), &[]);
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
    assert_eq!(m.dims(), &[]);
    assert_eq!(
        m.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[0.5]
    );

    let s = x.sum().unwrap();
    assert_eq!(s.dims(), &[]);
    assert_eq!(
        s.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[1.0]
    );

    let t = x.sin().unwrap().cos().unwrap().tanh().unwrap();
    assert_eq!(t.scalar_type(), tenferro::ScalarType::F64);

    let v = x.var().unwrap();
    assert_eq!(v.dims(), &[]);

    let std = x.std().unwrap();
    assert_eq!(std.dims(), &[]);
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
    assert_eq!(det.dims(), &[]);
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
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.0]
    );
}
