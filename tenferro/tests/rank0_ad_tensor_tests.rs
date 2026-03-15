use num_complex::{Complex32, Complex64};
use tenferro::{ScalarType, Tensor};

mod support;

use support::{
    forward_rank0_c32, forward_rank0_f32, forward_rank0_f64, grad_wrt, primal_rank0_c64,
    rank0_value_c32, rank0_value_c64, reverse_rank0_c32, reverse_rank0_c64, reverse_rank0_f32,
};

#[test]
fn rank0_forward_tensor_exposes_primal_tangent_and_metadata() {
    let x = forward_rank0_f64(2.0_f64, 0.5_f64);
    assert_eq!(x.dims(), &[]);
    assert_eq!(x.scalar_type(), tenferro::ScalarType::F64);
    assert!(!x.requires_grad());
    assert!(x.grad().is_none());
    assert_eq!(
        x.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[2.0_f64]
    );
    assert_eq!(
        x.as_f64()
            .unwrap()
            .tangent()
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.5_f64]
    );
}

#[test]
fn rank0_reverse_tensor_roundtrips_complex_primal_and_node_metadata() {
    let x = reverse_rank0_c64(Complex64::new(1.0, -2.0));

    assert_eq!(x.dims(), &[]);
    assert!(x.requires_grad());
    assert!(x.grad().is_none());
    assert_eq!(rank0_value_c64(&x), Complex64::new(1.0, -2.0));
}

#[test]
fn rank0_real_imag_compose_roundtrip_preserves_forward_mode() {
    let z = primal_rank0_c64(Complex64::new(2.0, -3.0));
    let re = z.real_part().unwrap();
    let im = z.imag_part().unwrap();
    let roundtrip = Tensor::compose_complex(re.clone(), im.clone()).unwrap();

    assert!(!re.requires_grad());
    assert!(re.as_f64().unwrap().tangent().is_none());
    assert!(!im.requires_grad());
    assert!(im.as_f64().unwrap().tangent().is_none());
    assert!(!roundtrip.requires_grad());
    assert!(roundtrip.as_c64().unwrap().tangent().is_none());
    assert_eq!(
        roundtrip
            .as_c64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(2.0, -3.0)]
    );
}

#[test]
fn rank0_explicit_cast_preserves_rank0_forward_tangent_across_precision_changes() {
    let x = forward_rank0_f32(2.5_f32, -0.75_f32);
    let y = x.to_scalar_type(ScalarType::F64).unwrap();
    assert_eq!(y.dims(), &[]);
    assert_eq!(y.scalar_type(), ScalarType::F64);
    assert!(!y.requires_grad());
    assert!(y.grad().is_none());
    assert_eq!(
        y.as_f64().unwrap().primal().buffer().as_slice().unwrap(),
        &[2.5_f64]
    );
    assert_eq!(
        y.as_f64()
            .unwrap()
            .tangent()
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap(),
        &[-0.75_f64]
    );

    let z = forward_rank0_c32(Complex32::new(1.5, -2.0), Complex32::new(-0.5, 3.0));
    let w = z.to_scalar_type(ScalarType::C64).unwrap();
    assert_eq!(w.scalar_type(), ScalarType::C64);
    assert_eq!(
        w.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(1.5, -2.0)]
    );
    assert_eq!(
        w.as_c64()
            .unwrap()
            .tangent()
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(-0.5, 3.0)]
    );
}

#[test]
fn rank0_explicit_cast_reverse_pullback_casts_back_to_source_dtype() {
    let real = reverse_rank0_f32(1.25_f32);
    let cast_real = real.to_scalar_type(ScalarType::C64).unwrap();
    let real_grads = grad_wrt(
        &cast_real,
        &Tensor::from_tensor(
            tenferro_tensor::Tensor::<Complex64>::from_slice(
                &[Complex64::new(2.0, -3.0)],
                &[],
                tenferro_tensor::MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        ),
        &[&real],
    );
    assert_eq!(
        real_grads[0]
            .as_ref()
            .unwrap()
            .as_f32()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[2.0_f32]
    );

    let complex = reverse_rank0_c32(Complex32::new(2.0, -1.5));
    let cast_complex = complex.to_scalar_type(ScalarType::F64).unwrap();
    let complex_grads = grad_wrt(
        &cast_complex,
        &Tensor::from_tensor(
            tenferro_tensor::Tensor::<f64>::from_slice(
                &[3.5_f64],
                &[],
                tenferro_tensor::MemoryOrder::ColumnMajor,
            )
            .unwrap(),
        ),
        &[&complex],
    );
    assert_eq!(
        rank0_value_c32(complex_grads[0].as_ref().unwrap()),
        Complex32::new(3.5, 0.0)
    );
}
