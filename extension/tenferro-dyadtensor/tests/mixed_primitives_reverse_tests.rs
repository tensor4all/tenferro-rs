use chainrules::Tape;
use num_complex::Complex64;
use tenferro_dyadtensor::{
    ad, set_default_runtime, AdMode, AdTensor, DynAdTensor, RuntimeContext, StructuredTensor,
};
use tenferro_prims::CpuContext;

mod support;

use support::{
    primal_rank0_c64, primal_rank0_f64, rank0_value_f64, reverse_rank0_f64, reverse_vector_f64,
    vector_c64, vector_f64,
};

#[test]
fn scale_registers_reverse_gradients_for_tensor_and_scalar_inputs() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x = reverse_vector_f64(&[1.0, 2.0], &tape);
    let a = reverse_rank0_f64(3.0_f64, &tape);

    let out = x.scale(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = tenferro_dyadtensor::AdTensor::new_primal(vector_f64(&[0.5, 1.25]));

    let grads = ad::pullback_wrt(
        out_t,
        &cotangent,
        &[x.as_f64().unwrap(), a.as_f64().unwrap()],
    )
    .unwrap();

    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.5, 3.75]
    );
    assert_eq!(rank0_value_f64(grads[1].as_ref().unwrap()), 3.0);
}

#[test]
fn axpby_registers_reverse_gradients_for_all_inputs() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x = reverse_vector_f64(&[1.0, 4.0], &tape);
    let y = reverse_vector_f64(&[3.0, -1.0], &tape);
    let a = reverse_rank0_f64(2.0_f64, &tape);
    let b = reverse_rank0_f64(-1.0_f64, &tape);

    let out = x.axpby(&a, &y, &b).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = tenferro_dyadtensor::AdTensor::new_primal(vector_f64(&[0.5, -0.25]));

    let grads = ad::pullback_wrt(
        out_t,
        &cotangent,
        &[
            x.as_f64().unwrap(),
            y.as_f64().unwrap(),
            a.as_f64().unwrap(),
            b.as_f64().unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.0, -0.5]
    );
    assert_eq!(
        grads[1]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[-0.5, 0.25]
    );
    assert_eq!(rank0_value_f64(grads[2].as_ref().unwrap()), -0.5);
    assert_eq!(rank0_value_f64(grads[3].as_ref().unwrap()), 1.75);
}

#[test]
fn div_scalar_registers_reverse_gradients_for_tensor_and_scalar_inputs() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x = reverse_vector_f64(&[2.0, 4.0], &tape);
    let a = reverse_rank0_f64(2.0_f64, &tape);

    let out = x.div_scalar(&a).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = tenferro_dyadtensor::AdTensor::new_primal(vector_f64(&[0.5, -1.0]));

    let grads = ad::pullback_wrt(
        out_t,
        &cotangent,
        &[x.as_f64().unwrap(), a.as_f64().unwrap()],
    )
    .unwrap();

    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.25, -0.5]
    );
    assert_eq!(rank0_value_f64(grads[1].as_ref().unwrap()), 0.75);
}

#[test]
fn scale_propagates_reverse_gradients_through_unary_scalar_coefficients() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x = reverse_vector_f64(&[1.0, 2.0], &tape);
    let a = reverse_rank0_f64(4.0_f64, &tape);
    let coeff: DynAdTensor = ad::sqrt(a.as_f64().unwrap()).unwrap().into();

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = tenferro_dyadtensor::AdTensor::new_primal(vector_f64(&[0.5, 1.25]));

    let grads = ad::pullback_wrt(
        out_t,
        &cotangent,
        &[x.as_f64().unwrap(), a.as_f64().unwrap()],
    )
    .unwrap();

    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .payload()
            .buffer()
            .as_slice()
            .unwrap(),
        &[1.0, 2.5]
    );
    assert_eq!(rank0_value_f64(grads[1].as_ref().unwrap()), 0.75);
}

#[test]
fn scale_propagates_reverse_gradients_through_negated_scalar_coefficients() {
    let tape = Tape::<StructuredTensor<f64>>::new();
    let x: DynAdTensor = tenferro_dyadtensor::AdTensor::new_primal(vector_f64(&[2.0, -1.0])).into();
    let a = reverse_rank0_f64(3.0_f64, &tape);
    let coeff = a.scale(&primal_rank0_f64(-1.0)).unwrap();

    let out = x.scale(&coeff).unwrap();
    assert_eq!(out.mode(), AdMode::Reverse);

    let out_t = out.as_f64().unwrap();
    let cotangent = tenferro_dyadtensor::AdTensor::new_primal(vector_f64(&[0.5, -1.25]));
    let grads = ad::pullback_wrt(out_t, &cotangent, &[a.as_f64().unwrap()]).unwrap();

    assert_eq!(rank0_value_f64(grads[0].as_ref().unwrap()), -2.25);
}

#[test]
fn scale_propagates_reverse_gradients_through_negative_real_sqrt_promotion() {
    let x: DynAdTensor = AdTensor::new_primal(c64_vec(&[
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
    ]))
    .into();
    let tape = Tape::<StructuredTensor<f64>>::new();
    let coeff = reverse_rank0_f64(-4.0_f64, &tape);

    let err = match x.scale(&coeff) {
        Ok(_) => panic!("mixed-dtype reverse scale should be unsupported"),
        Err(err) => err,
    };
    assert!(
        matches!(err, tenferro_dyadtensor::Error::UnsupportedAdOp { op } if op == "mixed_dtype_tensor_reverse")
    );

    let out = x
        .scale(&primal_rank0_c64(Complex64::new(0.0, 2.0)))
        .unwrap();
    assert_eq!(out.mode(), AdMode::Primal);
}

fn c64_vec(values: &[Complex64]) -> tenferro_tensor::Tensor<Complex64> {
    vector_c64(values)
}
