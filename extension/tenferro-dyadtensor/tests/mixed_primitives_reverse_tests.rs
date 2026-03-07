use num_complex::Complex64;
use tenferro_dyadtensor::{
    ad, AdMode, AdScalar, AdTensor, AdValue, DynAdScalar, DynAdTensor, NodeId, TapeId,
};
use tenferro_tensor::{MemoryOrder, Tensor};

fn f64_vec(values: &[f64]) -> Tensor<f64> {
    Tensor::<f64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn c64_vec(values: &[Complex64]) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(values, &[values.len()], MemoryOrder::ColumnMajor).unwrap()
}

fn scalar_from_dyn(value: &DynAdScalar) -> AdScalar<f64> {
    AdScalar::from(value.as_f64().unwrap().clone())
}

#[test]
fn scale_registers_reverse_gradients_for_tensor_and_scalar_inputs() {
    let x: DynAdTensor = AdTensor::new_reverse(
        f64_vec(&[1.0, 2.0]),
        NodeId(1),
        TapeId(21),
        None::<tenferro_dyadtensor::StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let a = DynAdScalar::from(AdValue::reverse(3.0_f64, NodeId(2), TapeId(21), None));

    let out = x.scale(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, 1.25]));

    let tensor_grads = ad::pullback_wrt(out_t, &cotangent, &[x.as_f64().unwrap()]).unwrap();
    let scalar_grads =
        ad::pullback_wrt_scalars(out_t, &cotangent, &[&scalar_from_dyn(&a)]).unwrap();

    assert_eq!(
        tensor_grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.5, 3.75]
    );
    assert_eq!(scalar_grads, vec![Some(3.0)]);
}

#[test]
fn axpby_registers_reverse_gradients_for_all_inputs() {
    let x: DynAdTensor = AdTensor::new_reverse(
        f64_vec(&[1.0, 4.0]),
        NodeId(11),
        TapeId(31),
        None::<tenferro_dyadtensor::StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let y: DynAdTensor = AdTensor::new_reverse(
        f64_vec(&[3.0, -1.0]),
        NodeId(12),
        TapeId(31),
        None::<tenferro_dyadtensor::StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let a = DynAdScalar::from(AdValue::reverse(2.0_f64, NodeId(13), TapeId(31), None));
    let b = DynAdScalar::from(AdValue::reverse(-1.0_f64, NodeId(14), TapeId(31), None));

    let out = x.axpby(&a, &y, &b).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -0.25]));

    let tensor_grads = ad::pullback_wrt(
        out_t,
        &cotangent,
        &[x.as_f64().unwrap(), y.as_f64().unwrap()],
    )
    .unwrap();
    let scalar_grads = ad::pullback_wrt_scalars(
        out_t,
        &cotangent,
        &[&scalar_from_dyn(&a), &scalar_from_dyn(&b)],
    )
    .unwrap();

    assert_eq!(
        tensor_grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.0, -0.5]
    );
    assert_eq!(
        tensor_grads[1]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[-0.5, 0.25]
    );
    assert_eq!(scalar_grads, vec![Some(-0.5), Some(1.75)]);
}

#[test]
fn div_scalar_registers_reverse_gradients_for_tensor_and_scalar_inputs() {
    let x: DynAdTensor = AdTensor::new_reverse(
        f64_vec(&[2.0, 4.0]),
        NodeId(21),
        TapeId(41),
        None::<tenferro_dyadtensor::StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let a = DynAdScalar::from(AdValue::reverse(2.0_f64, NodeId(22), TapeId(41), None));

    let out = x.div_scalar(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -1.0]));

    let tensor_grads = ad::pullback_wrt(out_t, &cotangent, &[x.as_f64().unwrap()]).unwrap();
    let scalar_grads =
        ad::pullback_wrt_scalars(out_t, &cotangent, &[&scalar_from_dyn(&a)]).unwrap();

    assert_eq!(
        tensor_grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.25, -0.5]
    );
    assert_eq!(scalar_grads, vec![Some(0.75)]);
}

#[test]
fn scale_propagates_reverse_gradients_through_unary_scalar_coefficients() {
    let x: DynAdTensor = AdTensor::new_reverse(
        f64_vec(&[1.0, 2.0]),
        NodeId(31),
        TapeId(51),
        None::<tenferro_dyadtensor::StructuredTensor<f64>>,
    )
    .unwrap()
    .into();
    let a = DynAdScalar::from(AdValue::reverse(4.0_f64, NodeId(32), TapeId(51), None));
    let coeff = a.sqrt();

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, 1.25]));

    let tensor_grads = ad::pullback_wrt(out_t, &cotangent, &[x.as_f64().unwrap()]).unwrap();
    let scalar_grads =
        ad::pullback_wrt_scalars(out_t, &cotangent, &[&scalar_from_dyn(&a)]).unwrap();

    assert_eq!(
        tensor_grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.0, 2.5]
    );
    assert_eq!(scalar_grads, vec![Some(0.75)]);
}

#[test]
fn scale_propagates_reverse_gradients_through_negated_scalar_coefficients() {
    let x: DynAdTensor = AdTensor::new_primal(f64_vec(&[2.0, -1.0])).into();
    let a = DynAdScalar::from(AdValue::reverse(3.0_f64, NodeId(41), TapeId(61), None));
    let coeff = -a.clone();

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = AdTensor::new_primal(f64_vec(&[0.5, -1.25]));
    let scalar_grads =
        ad::pullback_wrt_scalars(out_t, &cotangent, &[&scalar_from_dyn(&a)]).unwrap();

    assert_eq!(scalar_grads, vec![Some(-2.25)]);
}

#[test]
fn scale_propagates_reverse_gradients_through_negative_real_sqrt_promotion() {
    let x: DynAdTensor = AdTensor::new_primal(c64_vec(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
    ]))
    .into();
    let a = DynAdScalar::from(AdValue::reverse(-4.0_f64, NodeId(51), TapeId(62), None));
    let coeff = a.sqrt();

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_c64().unwrap();
    let cotangent = AdTensor::new_primal(c64_vec(&[
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 2.0),
    ]));
    let scalar_grads =
        ad::pullback_wrt_scalars(out_t, &cotangent, &[&scalar_from_dyn(&a)]).unwrap();

    assert_eq!(scalar_grads, vec![Some(-1.25)]);
}
