use super::support::{
    forward_rank0_f64, forward_tensor_c64, grad_wrt, reverse_rank0_f64_like, reverse_vector_c64,
    scalar_c32, scalar_c64, scalar_f32, scalar_f64, vector_c64, vector_f32,
};
use num_complex::{Complex32, Complex64};
use tenferro::{set_default_runtime, RuntimeContext, ScalarType, Tensor};
use tenferro_prims::CpuContext;
use tenferro_tensor::Tensor as DenseTensor;

fn c64_vec(values: &[Complex64]) -> DenseTensor<Complex64> {
    vector_c64(values)
}

#[test]
fn c64_tensor_scale_accepts_f64_scalar_in_forward_mode() {
    let x = forward_tensor_c64(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    );
    let a = forward_rank0_f64(2.0_f64, 0.25_f64);

    let out = x.scale(&a).unwrap();
    assert!(!out.requires_grad());
    assert!(out.grad().is_none());

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
    let x = forward_tensor_c64(
        c64_vec(&[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    );
    let a = forward_rank0_f64(2.0_f64, 0.5_f64);

    let out = x.div_scalar(&a).unwrap();
    assert!(!out.requires_grad());
    assert!(out.grad().is_none());

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
    let x = forward_tensor_c64(
        c64_vec(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]),
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)]),
    );
    let y = forward_tensor_c64(
        c64_vec(&[Complex64::new(0.5, 0.0), Complex64::new(2.0, 0.0)]),
        c64_vec(&[Complex64::new(-0.5, 0.0), Complex64::new(0.25, 0.0)]),
    );
    let a = forward_rank0_f64(2.0_f64, 0.25_f64);
    let b = forward_rank0_f64(-0.5_f64, 1.0_f64);

    let out = x.axpby(&a, &y, &b).unwrap();
    assert!(!out.requires_grad());
    assert!(out.grad().is_none());

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
fn c64_tensor_scale_reverse_casts_back_scalar_gradient_to_real_dtype() {
    let x = reverse_vector_c64(&[Complex64::new(1.0, 0.0), Complex64::new(-3.0, 0.0)]);
    let a = reverse_rank0_f64_like(2.0_f64, &x);

    let out = x.scale(&a).unwrap();
    assert!(out.requires_grad());

    let cotangent = Tensor::from_tensor(c64_vec(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(-1.0, 0.0),
    ]));
    let grads = grad_wrt(&out, &cotangent, &[&x, &a]);
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .as_c64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(1.0, 0.0), Complex64::new(-2.0, 0.0)]
    );
    assert_eq!(
        grads[1]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[3.5]
    );
}

#[test]
fn c64_tensor_div_scalar_reverse_casts_back_scalar_gradient_to_real_dtype() {
    let x = reverse_vector_c64(&[Complex64::new(4.0, 0.0), Complex64::new(-6.0, 0.0)]);
    let a = reverse_rank0_f64_like(2.0_f64, &x);

    let out = x.div_scalar(&a).unwrap();
    assert!(out.requires_grad());

    let cotangent = Tensor::from_tensor(c64_vec(&[
        Complex64::new(0.5, 0.0),
        Complex64::new(-1.0, 0.0),
    ]));
    let grads = grad_wrt(&out, &cotangent, &[&x, &a]);
    assert_eq!(
        grads[0]
            .as_ref()
            .unwrap()
            .as_c64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(0.25, 0.0), Complex64::new(-0.5, 0.0)]
    );
    assert_eq!(
        grads[1]
            .as_ref()
            .unwrap()
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[-2.0]
    );
}

#[test]
fn mixed_precision_ops_follow_result_type_lattice() {
    let _guard = set_default_runtime(RuntimeContext::Cpu(CpuContext::new(1)));
    let coeff_f32 = Tensor::from_tensor(scalar_f32(1.0));
    let coeff_f64 = Tensor::from_tensor(scalar_f64(2.0));
    let coeff_c32 = Tensor::from_tensor(scalar_c32(Complex32::new(3.0, -0.5)));
    let real_vec_f32 = Tensor::from_tensor(vector_f32(&[4.0, -2.0]));
    let complex_vec_c32 = Tensor::from_tensor(
        DenseTensor::<Complex32>::from_slice(
            &[Complex32::new(1.0, 0.25), Complex32::new(-2.0, 1.5)],
            &[2],
            tenferro_tensor::MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    let complex_vec_c64 = Tensor::from_tensor(c64_vec(&[
        Complex64::new(1.0, -0.5),
        Complex64::new(2.0, 1.25),
    ]));
    let coeff_c64 = Tensor::from_tensor(scalar_c64(Complex64::new(1.0, -0.5)));

    assert_eq!(
        coeff_f32.add(&coeff_f64).unwrap().scalar_type(),
        ScalarType::F64
    );
    assert_eq!(
        coeff_f64.add(&coeff_c32).unwrap().scalar_type(),
        ScalarType::C64
    );
    assert_eq!(
        coeff_f32.add(&coeff_c64).unwrap().scalar_type(),
        ScalarType::C64
    );
    assert_eq!(
        complex_vec_c32.scale(&coeff_f64).unwrap().scalar_type(),
        ScalarType::C64
    );
    assert_eq!(
        complex_vec_c32
            .div_scalar(&coeff_f64)
            .unwrap()
            .scalar_type(),
        ScalarType::C64
    );
    assert_eq!(
        real_vec_f32
            .axpby(
                &coeff_f64,
                &Tensor::from_tensor(vector_f32(&[0.5, 1.0])),
                &coeff_f32
            )
            .unwrap()
            .scalar_type(),
        ScalarType::F64
    );
    assert_eq!(
        complex_vec_c32
            .axpby(&coeff_f64, &complex_vec_c64, &coeff_c32)
            .unwrap()
            .scalar_type(),
        ScalarType::C64
    );
}
