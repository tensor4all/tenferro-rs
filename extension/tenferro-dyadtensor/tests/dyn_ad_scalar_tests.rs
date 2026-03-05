use num_complex::Complex64;
use tenferro_dyadtensor::{AdMode, AdValue, DynAdScalar, DynScalar, NodeId, ScalarType, TapeId};

#[test]
fn dyn_ad_scalar_exposes_primal_tangent_and_metadata() {
    let x = DynAdScalar::from(AdValue::forward(2.0_f64, 0.5_f64));
    assert_eq!(x.scalar_type(), ScalarType::F64);
    assert_eq!(x.mode(), AdMode::Forward);
    assert_eq!(x.primal(), DynScalar::F64(2.0));
    assert_eq!(x.tangent(), Some(DynScalar::F64(0.5)));
    assert_eq!(x.node_id(), None);
    assert_eq!(x.tape_id(), None);
}

#[test]
fn dyn_ad_scalar_detach_and_primal_into_drop_metadata_explicitly() {
    let x = DynAdScalar::from(AdValue::reverse(
        Complex64::new(1.0, -2.0),
        NodeId(3),
        TapeId(7),
        Some(Complex64::new(0.25, 0.5)),
    ));
    assert_eq!(x.detach(), DynScalar::C64(Complex64::new(1.0, -2.0)));
    assert_eq!(
        x.clone().primal_into(),
        DynScalar::C64(Complex64::new(1.0, -2.0))
    );
    assert_eq!(x.node_id(), Some(NodeId(3)));
    assert_eq!(x.tape_id(), Some(TapeId(7)));
}

#[test]
fn dyn_ad_scalar_real_imag_compose_preserve_forward_mode() {
    let z = DynAdScalar::from(AdValue::forward(
        Complex64::new(2.0, -3.0),
        Complex64::new(0.5, 0.25),
    ));
    let re = z.real_part();
    let im = z.imag_part();
    let roundtrip = DynAdScalar::compose_complex(re.clone(), im.clone()).unwrap();

    assert_eq!(re.mode(), AdMode::Forward);
    assert_eq!(im.mode(), AdMode::Forward);
    assert_eq!(roundtrip.mode(), AdMode::Forward);
    assert_eq!(
        roundtrip.primal(),
        DynScalar::C64(Complex64::new(2.0, -3.0))
    );
    assert_eq!(
        roundtrip.tangent(),
        Some(DynScalar::C64(Complex64::new(0.5, 0.25)))
    );
}
