use num_complex::Complex64;
use tenferro_dyadtensor::{
    ad, AdMode, AdScalar, AdTensor, AdValue, DynAdScalar, DynAdTensor, NodeId, TapeId,
};
use tenferro_tensor::{MemoryOrder, Tensor};

fn c64_vec(values: &[Complex64]) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar_f64(value: &DynAdScalar) -> AdScalar<f64> {
    AdScalar::from(value.as_f64().unwrap().clone())
}

#[test]
fn c64_tensor_scale_accepts_f64_scalar_in_forward_mode() {
    let x: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .into();
    let a = DynAdScalar::from(AdValue::forward(2.0_f64, 0.25_f64));

    let out = x.scale(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Forward);

    let out_t = out.as_c64().unwrap();
    assert_eq!(
        out_t.primal().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(-6.0, 0.0)]
    );
    assert_eq!(
        out_t.tangent().unwrap().buffer().as_slice().unwrap(),
        &[Complex64::new(1.25, 0.0), Complex64::new(2.25, 0.0)]
    );
}

#[test]
fn c64_tensor_div_scalar_accepts_f64_scalar_in_forward_mode() {
    let x: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .into();
    let a = DynAdScalar::from(AdValue::forward(2.0_f64, 0.5_f64));

    let out = x.div_scalar(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Forward);

    let out_t = out.as_c64().unwrap();
    assert_eq!(
        out_t.primal().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(-3.0, 0.0)]
    );
    assert_eq!(
        out_t.tangent().unwrap().buffer().as_slice().unwrap(),
        &[Complex64::new(-0.25, 0.0), Complex64::new(1.5, 0.0)]
    );
}

#[test]
fn c64_tensor_axpby_accepts_real_coefficients() {
    let x: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    )
    .into();
    let y: DynAdTensor = AdTensor::new_forward(
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0)]),
        c64_vec(&[Complex64::new(-0.5, 0.0), Complex64::new(0.25, 0.0)]),
    )
    .into();
    let a = DynAdScalar::from(AdValue::forward(2.0_f64, 0.25_f64));
    let b = DynAdScalar::from(AdValue::forward(-0.5_f64, 1.0_f64));

    let out = x.axpby(&a, &y, &b).unwrap();
    assert_eq!(out.mode(), AdMode::Forward);

    let out_t = out.as_c64().unwrap();
    assert_eq!(
        out_t.primal().buffer().as_slice().unwrap(),
        &[Complex64::new(1.75, 0.0), Complex64::new(-7.0, 0.0)]
    );
    assert_eq!(
        out_t.tangent().unwrap().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, 0.0), Complex64::new(4.125, 0.0)]
    );
}

#[test]
fn c64_tensor_scale_pullback_reaches_real_scalar_input() {
    let x: DynAdTensor = AdTensor::new_reverse(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        NodeId(21),
        TapeId(91),
        None::<tenferro_dyadtensor::StructuredTensor<Complex64>>,
    )
    .into();
    let a = DynAdScalar::from(AdValue::reverse(2.0_f64, NodeId(22), TapeId(91), None));

    let out = x.scale(&a).unwrap();
    let cotangent = AdTensor::new_primal(c64_vec(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(1.0, 0.0),
    ]));

    let tensor_grads =
        ad::pullback_wrt(out.as_c64().unwrap(), &cotangent, &[x.as_c64().unwrap()]).unwrap();
    let scalar_grads =
        ad::pullback_wrt_scalars(out.as_c64().unwrap(), &cotangent, &[&scalar_f64(&a)]).unwrap();

    assert_eq!(
        tensor_grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]
    );
    assert_eq!(scalar_grads, vec![Some(-2.5)]);
}

#[test]
fn c64_tensor_div_scalar_pullback_reaches_real_scalar_input() {
    let x: DynAdTensor = AdTensor::new_reverse(
        c64_vec(&[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)]),
        NodeId(31),
        TapeId(92),
        None::<tenferro_dyadtensor::StructuredTensor<Complex64>>,
    )
    .into();
    let a = DynAdScalar::from(AdValue::reverse(2.0_f64, NodeId(32), TapeId(92), None));

    let out = x.div_scalar(&a).unwrap();
    let cotangent = AdTensor::new_primal(c64_vec(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(1.0, 0.0),
    ]));

    let tensor_grads =
        ad::pullback_wrt(out.as_c64().unwrap(), &cotangent, &[x.as_c64().unwrap()]).unwrap();
    let scalar_grads =
        ad::pullback_wrt_scalars(out.as_c64().unwrap(), &cotangent, &[&scalar_f64(&a)]).unwrap();

    assert_eq!(
        tensor_grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(0.25, 0.0), Complex64::new(0.5, 0.0)]
    );
    assert_eq!(scalar_grads, vec![Some(1.0)]);
}
