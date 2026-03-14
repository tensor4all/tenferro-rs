use num_complex::Complex64;
use tenferro_dyadtensor::{AdMode, DynAdTensor};

mod support;

use support::{forward_rank0_f64, primal_rank0_c64, rank0_value_c64, reverse_rank0_c64};

#[test]
fn rank0_forward_tensor_exposes_primal_tangent_and_metadata() {
    let x = forward_rank0_f64(2.0_f64, 0.5_f64);
    assert_eq!(x.mode(), AdMode::Forward);
    assert_eq!(x.dims(), &[]);
    assert_eq!(x.scalar_type(), tenferro_dyadtensor::ScalarType::F64);
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
    let value = x.as_c64().unwrap();

    assert_eq!(x.mode(), AdMode::Reverse);
    assert_eq!(x.dims(), &[]);
    assert!(value.node_id().is_some());
    assert!(x.tape_id().is_some());
    assert_eq!(
        rank0_value_c64(x.as_c64().unwrap().structured_primal()),
        Complex64::new(1.0, -2.0)
    );
}

#[test]
fn rank0_real_imag_compose_roundtrip_preserves_forward_mode() {
    let z = primal_rank0_c64(Complex64::new(2.0, -3.0));
    let re = z.real_part().unwrap();
    let im = z.imag_part().unwrap();
    let roundtrip = DynAdTensor::compose_complex(re.clone(), im.clone()).unwrap();

    assert_eq!(re.mode(), AdMode::Primal);
    assert_eq!(im.mode(), AdMode::Primal);
    assert_eq!(roundtrip.mode(), AdMode::Primal);
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
