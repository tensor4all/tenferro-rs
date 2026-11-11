use super::*;
use crate::ErrorKind;

#[test]
fn overwrite_maps_to_overwrite_accumulation() {
    let (out, accum) = Dest::Overwrite(1)
        .into_accumulation(DType::F64, Conj::No, Conj::No)
        .unwrap();
    assert_eq!(out, 1);
    assert_eq!(
        accum,
        DotGeneralAccumulation::overwrite(DType::F64).unwrap()
    );
}

#[test]
fn accumulate_maps_to_add_to() {
    let (_, accum) = Dest::Accumulate(())
        .into_accumulation(DType::C64, Conj::No, Conj::No)
        .unwrap();
    assert_eq!(accum, DotGeneralAccumulation::add_to(DType::C64).unwrap());
}

#[test]
fn scaled_keeps_alpha_beta_and_conj_flags() {
    let dest = Dest::Scaled {
        out: (),
        alpha: 2.0_f32.into(),
        beta: 3.0_f32.into(),
    };
    let (_, accum) = dest
        .into_accumulation(DType::F32, Conj::Yes, Conj::No)
        .unwrap();
    assert_eq!(accum.alpha, ContractionScalar::F32(2.0));
    assert_eq!(accum.beta, ContractionScalar::F32(3.0));
    assert!(accum.lhs_conj);
    assert!(!accum.rhs_conj);

    let (_, accum) = Dest::Overwrite(())
        .into_accumulation(DType::F32, Conj::No, Conj::Yes)
        .unwrap();
    assert!(!accum.lhs_conj);
    assert!(accum.rhs_conj);
}

#[test]
fn scaled_dtype_mismatch_is_validation_error() {
    let dest = Dest::Scaled {
        out: (),
        alpha: 1.0_f64.into(),
        beta: 0.0_f32.into(),
    };
    let err = dest
        .into_accumulation(DType::F64, Conj::No, Conj::No)
        .unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Validation(_)));
    assert!(matches!(
        err,
        Error::Validation {
            source: ValidationError::DTypeMismatch { .. },
            ..
        }
    ));
}

#[test]
fn unsupported_dtype_is_validation_error() {
    let err = Dest::Overwrite(())
        .into_accumulation(DType::I32, Conj::No, Conj::No)
        .unwrap_err();
    assert!(matches!(err, Error::Validation { .. }));
}

#[test]
fn accessors_and_scalar_conversions() {
    let dest = Dest::<i32>::scaled(4, 1.0_f64, Complex64::new(0.0, 1.0));
    assert_eq!(dest.mode(), DestMode::Scaled);
    assert_eq!(*dest.target(), 4);
    assert_eq!(dest.into_target(), 4);
    assert_eq!(
        ContractionScalar::from(Complex32::new(1.0, 0.0)),
        ContractionScalar::C32(Complex32::new(1.0, 0.0))
    );
    assert_eq!(Conj::default(), Conj::No);
}
