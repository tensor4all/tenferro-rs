use super::*;

#[test]
fn compact_host_accumulation_slice_selects_only_compact_host_views() {
    let mut compact_data = [0.0_f64; 4];
    let mut compact = TypedTensorViewMut::from_slice([2, 2], [1, 2], 0, &mut compact_data).unwrap();
    assert_eq!(
        compact_host_accumulation_slice(&mut compact, 4)
            .unwrap()
            .unwrap()
            .len(),
        4
    );

    let mut strided_data = [0.0_f64; 3];
    let mut strided = TypedTensorViewMut::from_slice([2], [2], 0, &mut strided_data).unwrap();
    assert!(compact_host_accumulation_slice(&mut strided, 2)
        .unwrap()
        .is_none());
}

#[test]
fn contraction_scalar_identity_errors_name_the_public_constructor() {
    let one_error = ContractionScalar::one(DType::I32).unwrap_err();
    assert!(matches!(
        one_error,
        Error::Validation {
            op: "ContractionScalar::one",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let zero_error = ContractionScalar::zero(DType::Bool).unwrap_err();
    assert!(matches!(
        zero_error,
        Error::Validation {
            op: "ContractionScalar::zero",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let overwrite_error = DotGeneralAccumulation::overwrite(DType::I32).unwrap_err();
    assert!(matches!(
        overwrite_error,
        Error::Validation {
            op: "DotGeneralAccumulation::overwrite",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let add_to_error = DotGeneralAccumulation::add_to(DType::Bool).unwrap_err();
    assert!(matches!(
        add_to_error,
        Error::Validation {
            op: "DotGeneralAccumulation::add_to",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));

    let scaled_error =
        DotGeneralAccumulation::scaled(ContractionScalar::F32(1.0), ContractionScalar::F64(1.0))
            .unwrap_err();
    assert!(matches!(
        scaled_error,
        Error::Validation {
            op: "DotGeneralAccumulation::scaled",
            source: ValidationError::DTypeMismatch { .. },
        }
    ));
}
