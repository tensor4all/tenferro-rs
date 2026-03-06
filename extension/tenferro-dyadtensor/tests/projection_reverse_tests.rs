use num_complex::Complex64;
use tenferro_dyadtensor::{
    ad, AdScalar, AdTensor, AdValue, DynAdScalar, DynAdTensor, NodeId, TapeId,
};
use tenferro_tensor::{MemoryOrder, Tensor};

fn f64_vec(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn c64_vec(values: &[Complex64]) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar_f64(value: &DynAdScalar) -> AdScalar<f64> {
    AdScalar::from(value.as_f64().unwrap().clone())
}

fn scalar_c64(value: &DynAdScalar) -> AdScalar<Complex64> {
    AdScalar::from(value.as_c64().unwrap().clone())
}

#[test]
fn scalar_complex_real_part_reverse_flows_into_real_lane() {
    let x: DynAdTensor = AdTensor::new_primal(f64_vec(&[2.0, -1.0])).into();
    let z = DynAdScalar::from(AdValue::reverse(
        Complex64::new(3.0, -4.0),
        NodeId(1),
        TapeId(71),
        None,
    ));

    let out = x.scale(&z.real_part()).unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -1.25]));
    let grads =
        ad::pullback_wrt_scalars(out.as_f64().unwrap(), &cotangent, &[&scalar_c64(&z)]).unwrap();

    assert_eq!(grads, vec![Some(Complex64::new(2.25, 0.0))]);
}

#[test]
fn scalar_complex_imag_part_reverse_flows_into_imag_lane() {
    let x: DynAdTensor = AdTensor::new_primal(f64_vec(&[2.0, -1.0])).into();
    let z = DynAdScalar::from(AdValue::reverse(
        Complex64::new(3.0, -4.0),
        NodeId(2),
        TapeId(72),
        None,
    ));

    let out = x.scale(&z.imag_part()).unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -1.25]));
    let grads =
        ad::pullback_wrt_scalars(out.as_f64().unwrap(), &cotangent, &[&scalar_c64(&z)]).unwrap();

    assert_eq!(grads, vec![Some(Complex64::new(0.0, 2.25))]);
}

#[test]
fn scalar_real_imag_part_reverse_returns_zero_gradient() {
    let x: DynAdTensor = AdTensor::new_primal(f64_vec(&[2.0, -1.0])).into();
    let a = DynAdScalar::from(AdValue::reverse(3.0_f64, NodeId(3), TapeId(73), None));

    let out = x.scale(&a.imag_part()).unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -1.25]));
    let grads =
        ad::pullback_wrt_scalars(out.as_f64().unwrap(), &cotangent, &[&scalar_f64(&a)]).unwrap();

    assert_eq!(grads, vec![Some(0.0)]);
}

#[test]
fn scalar_compose_complex_reverse_splits_complex_cotangent() {
    let x: DynAdTensor = AdTensor::new_primal(c64_vec(&[Complex64::new(1.0, 0.0); 2])).into();
    let re = DynAdScalar::from(AdValue::reverse(2.0_f64, NodeId(4), TapeId(74), None));
    let im = DynAdScalar::from(AdValue::reverse(-3.0_f64, NodeId(5), TapeId(74), None));
    let z = DynAdScalar::compose_complex(re.clone(), im.clone()).unwrap();

    let out = x.scale(&z).unwrap();
    let cotangent = AdTensor::new_primal(c64_vec(&[
        Complex64::new(0.5, -0.25),
        Complex64::new(1.0, 0.75),
    ]));
    let grads = ad::pullback_wrt_scalars(
        out.as_c64().unwrap(),
        &cotangent,
        &[&scalar_f64(&re), &scalar_f64(&im)],
    )
    .unwrap();

    assert_eq!(grads, vec![Some(1.5), Some(0.5)]);
}

#[test]
fn tensor_complex_real_part_reverse_via_pullback_wrt_mixed() {
    let x: DynAdTensor = AdTensor::new_reverse(
        c64_vec(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]),
        NodeId(11),
        TapeId(81),
        None::<tenferro_dyadtensor::StructuredTensor<Complex64>>,
    )
    .unwrap()
    .into();

    let out = x.real_part().unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -1.25]));
    let grads =
        ad::pullback_wrt_mixed(out.as_f64().unwrap(), &cotangent, &[x.as_c64().unwrap()]).unwrap();

    assert!(grads[0].as_ref().unwrap().is_dense());
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(0.5, 0.0), Complex64::new(-1.25, 0.0)]
    );
}

#[test]
fn tensor_complex_imag_part_reverse_via_pullback_wrt_mixed() {
    let x: DynAdTensor = AdTensor::new_reverse(
        c64_vec(&[Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)]),
        NodeId(12),
        TapeId(82),
        None::<tenferro_dyadtensor::StructuredTensor<Complex64>>,
    )
    .unwrap()
    .into();

    let out = x.imag_part().unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -1.25]));
    let grads =
        ad::pullback_wrt_mixed(out.as_f64().unwrap(), &cotangent, &[x.as_c64().unwrap()]).unwrap();

    assert!(grads[0].as_ref().unwrap().is_dense());
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(0.0, 0.5), Complex64::new(0.0, -1.25)]
    );
}

#[test]
fn tensor_compose_complex_reverse_via_pullback_wrt_mixed() {
    let re: DynAdTensor = AdTensor::new_reverse(
        f64_vec(&[1.0, -3.0]),
        NodeId(13),
        TapeId(83),
        None::<tenferro_dyadtensor::StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let im: DynAdTensor = AdTensor::new_reverse(
        f64_vec(&[2.0, 4.0]),
        NodeId(14),
        TapeId(83),
        None::<tenferro_dyadtensor::StructuredTensor<f64>>,
    )
    .unwrap()
    .into();

    let out = DynAdTensor::compose_complex(re.clone(), im.clone()).unwrap();
    let cotangent = AdTensor::new_primal(c64_vec(&[
        Complex64::new(0.5, -0.25),
        Complex64::new(1.0, 0.75),
    ]));
    let grads = ad::pullback_wrt_mixed(
        out.as_c64().unwrap(),
        &cotangent,
        &[re.as_f64().unwrap(), im.as_f64().unwrap()],
    )
    .unwrap();

    assert!(grads[0].as_ref().unwrap().is_dense());
    assert!(grads[1].as_ref().unwrap().is_dense());
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.5, 1.0]
    );
    assert_eq!(
        grads[1]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[-0.25, 0.75]
    );
}
