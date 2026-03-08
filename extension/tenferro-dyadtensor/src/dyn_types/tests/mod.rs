use super::*;
use crate::{Error, TapeId};
use tenferro_tensor::MemoryOrder;

#[test]
fn dyn_scalar_metadata() {
    let x: DynScalar = 1.0_f64.into();
    assert_eq!(x.scalar_type(), ScalarType::F64);
    assert_eq!(x.as_f64(), Some(1.0));
}

#[test]
fn dyn_ad_value_mode_and_tangent() {
    let x: DynAdScalar = AdValue::forward(2.0_f32, 0.5_f32).into();
    assert_eq!(x.scalar_type(), ScalarType::F32);
    assert_eq!(x.mode(), AdMode::Forward);
    assert_eq!(x.tangent(), Some(DynScalar::F32(0.5)));
}

#[test]
fn dyn_tensor_and_dyn_ad_tensor_dims() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let d: DynTensor = t.clone().into();
    assert_eq!(d.dims(), &[2]);

    let ad = AdTensor::new_primal(t);
    let dad: DynAdTensor = ad.into();
    assert_eq!(dad.dims(), &[2]);
    assert_eq!(dad.mode(), AdMode::Primal);
}

#[test]
fn dyn_ad_value_mul_mixed_real_complex_promotes_to_complex() {
    let lhs = DynAdScalar::from(2.0_f64);
    let rhs = DynAdScalar::from(Complex64::new(1.0, -3.0));
    let out = (lhs * rhs).unwrap();
    assert_eq!(out.scalar_type(), ScalarType::C64);
    assert_eq!(out.primal(), DynScalar::C64(Complex64::new(2.0, -6.0)));
}

#[test]
fn dyn_ad_value_div_with_scalar_lhs_is_supported() {
    let rhs = DynAdScalar::from(2.0_f64);
    let out = (Complex64::new(4.0, -2.0) / rhs).unwrap();
    assert_eq!(out.scalar_type(), ScalarType::C64);
    assert_eq!(out.primal(), DynScalar::C64(Complex64::new(2.0, -1.0)));
}

#[test]
fn dyn_ad_value_try_add_rejects_cross_precision_pairs() {
    let lhs = DynAdScalar::from(1.0_f32);
    let rhs = DynAdScalar::from(2.0_f64);
    let err = lhs.try_add(rhs).unwrap_err();
    assert!(matches!(err, Error::InvalidAdScalar { .. }));
}

#[test]
fn dyn_ad_value_try_mul_checks_reverse_tape_compatibility() {
    let lhs: DynAdScalar = AdValue::reverse(2.0_f64, crate::NodeId(1), TapeId(7), None).into();
    let rhs: DynAdScalar = AdValue::reverse(3.0_f64, crate::NodeId(2), TapeId(8), None).into();
    let err = lhs.try_mul(rhs).unwrap_err();
    assert!(
        matches!(err, Error::InvalidAdScalar { message } if message.contains("reverse-mode tape mismatch"))
    );
}

#[test]
fn dyn_ad_value_operator_mul_checks_reverse_tape_compatibility() {
    let lhs: DynAdScalar = AdValue::reverse(2.0_f64, crate::NodeId(1), TapeId(7), None).into();
    let rhs: DynAdScalar = AdValue::reverse(3.0_f64, crate::NodeId(2), TapeId(8), None).into();
    let err = (lhs * rhs).unwrap_err();
    assert!(
        matches!(err, Error::InvalidAdScalar { message } if message.contains("reverse-mode tape mismatch"))
    );
}

#[test]
fn dyn_tensor_max_abs_diff_is_zero_for_same_logical_tensor_with_different_memory_order() {
    let base = Tensor::<f64>::from_slice(
        &(0..12).map(|x| x as f64).collect::<Vec<_>>(),
        &[2, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let row_major = base.contiguous(MemoryOrder::RowMajor);

    let lhs: DynTensor = base.into();
    let rhs: DynTensor = row_major.into();
    let diff = lhs.max_abs_diff(&rhs).unwrap();
    assert!(diff < 1e-12, "expected zero diff, got {diff}");
}

#[test]
fn dyn_tensor_sub_abs_max_pipeline_matches_expected() {
    let lhs_t = Tensor::<f64>::from_slice(
        &(0..12).map(|x| x as f64).collect::<Vec<_>>(),
        &[2, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let rhs_t = Tensor::<f64>::from_slice(
        &(0..12)
            .map(|x| if x == 7 { (x as f64) + 4.0 } else { x as f64 })
            .collect::<Vec<_>>(),
        &[2, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .contiguous(MemoryOrder::RowMajor);

    let lhs: DynTensor = lhs_t.into();
    let rhs: DynTensor = rhs_t.into();

    let diff = lhs
        .try_sub(&rhs)
        .unwrap()
        .abs_tensor()
        .unwrap()
        .max_as_f64()
        .unwrap();
    assert!((diff - 4.0).abs() < 1e-12, "expected diff=4, got {diff}");
}

#[test]
fn dyn_tensor_abs_tensor_on_complex_returns_real_dtype() {
    let t = Tensor::<Complex64>::from_slice(
        &[Complex64::new(3.0, 4.0), Complex64::new(0.0, -2.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let x: DynTensor = t.into();
    let y = x.abs_tensor().unwrap();
    assert_eq!(y.scalar_type(), ScalarType::F64);
    let yr = y.as_f64().unwrap();
    let data = yr.buffer().as_slice().unwrap();
    assert!((data[0] - 5.0).abs() < 1e-12);
    assert!((data[1] - 2.0).abs() < 1e-12);
}

#[test]
fn dyn_tensor_max_on_complex_requires_abs_first() {
    let t = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 2.0)],
        &[1],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let x: DynTensor = t.into();
    let err = x.max().unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { message } if message.contains("abs_tensor")));
}

#[test]
fn dyn_tensor_max_abs_diff_detects_value_difference() {
    let lhs_t = Tensor::<f64>::from_slice(
        &(0..12).map(|x| x as f64).collect::<Vec<_>>(),
        &[2, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let rhs_t = Tensor::<f64>::from_slice(
        &(0..12)
            .map(|x| if x == 7 { (x as f64) + 4.0 } else { x as f64 })
            .collect::<Vec<_>>(),
        &[2, 3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let lhs: DynTensor = lhs_t.into();
    let rhs: DynTensor = rhs_t.into();

    let diff = lhs.max_abs_diff(&rhs).unwrap();
    assert!((diff - 4.0).abs() < 1e-12, "expected diff=4, got {diff}");
}

#[test]
fn dyn_tensor_max_abs_diff_rejects_dtype_mismatch() {
    let lhs = Tensor::<f32>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let rhs = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let lhs: DynTensor = lhs.into();
    let rhs: DynTensor = rhs.into();

    let err = lhs.max_abs_diff(&rhs).unwrap_err();
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn dyn_ad_tensor_max_abs_diff_primal_uses_primal_values() {
    let lhs = Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let rhs = Tensor::<f64>::from_slice(&[1.0, 1.5, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let lhs: DynAdTensor = AdTensor::new_primal(lhs).into();
    let rhs: DynAdTensor = AdTensor::new_primal(rhs).into();

    let diff = lhs.max_abs_diff_primal(&rhs).unwrap();
    assert!((diff - 0.5).abs() < 1e-12, "expected diff=0.5, got {diff}");
}

#[test]
fn dyn_ad_tensor_real_imag_part_preserve_forward_mode() {
    let primal = Tensor::<Complex64>::from_slice(
        &[Complex64::new(2.5, -1.25)],
        &[1],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let tangent = Tensor::<Complex64>::from_slice(
        &[Complex64::new(0.5, 0.75)],
        &[1],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let x: DynAdTensor = AdTensor::new_forward(primal, tangent).unwrap().into();
    assert!(x.is_complex());
    assert!(!x.is_real());

    let xr = x.real_part().unwrap();
    let xi = x.imag_part().unwrap();
    assert!(xr.is_real());
    assert!(xi.is_real());
    assert_eq!(xr.scalar_type(), ScalarType::F64);
    assert_eq!(xi.scalar_type(), ScalarType::F64);
    assert_eq!(xr.mode(), AdMode::Forward);
    assert_eq!(xi.mode(), AdMode::Forward);

    let xr_t = xr.as_f64().unwrap();
    let xi_t = xi.as_f64().unwrap();
    let xr_primal = xr_t.primal().buffer().as_slice().unwrap()[0];
    let xr_tangent = xr_t.tangent().unwrap().buffer().as_slice().unwrap()[0];
    let xi_primal = xi_t.primal().buffer().as_slice().unwrap()[0];
    let xi_tangent = xi_t.tangent().unwrap().buffer().as_slice().unwrap()[0];

    assert!((xr_primal - 2.5).abs() < 1e-12);
    assert!((xr_tangent - 0.5).abs() < 1e-12);
    assert!((xi_primal - (-1.25)).abs() < 1e-12);
    assert!((xi_tangent - 0.75).abs() < 1e-12);
}

#[test]
fn dyn_ad_tensor_compose_complex_roundtrip_forward() {
    let re = AdTensor::new_forward(
        Tensor::<f64>::from_slice(&[1.5], &[1], MemoryOrder::ColumnMajor).unwrap(),
        Tensor::<f64>::from_slice(&[0.25], &[1], MemoryOrder::ColumnMajor).unwrap(),
    )
    .unwrap();
    let im = AdTensor::new_forward(
        Tensor::<f64>::from_slice(&[-2.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
        Tensor::<f64>::from_slice(&[0.75], &[1], MemoryOrder::ColumnMajor).unwrap(),
    )
    .unwrap();
    let z = DynAdTensor::compose_complex(DynAdTensor::F64(re), DynAdTensor::F64(im)).unwrap();
    assert_eq!(z.scalar_type(), ScalarType::C64);
    assert_eq!(z.mode(), AdMode::Forward);

    let zc = z.as_c64().unwrap();
    let primal = zc.primal().buffer().as_slice().unwrap()[0];
    let tangent = zc.tangent().unwrap().buffer().as_slice().unwrap()[0];
    assert!((primal - Complex64::new(1.5, -2.0)).norm() < 1e-12);
    assert!((tangent - Complex64::new(0.25, 0.75)).norm() < 1e-12);
}

#[test]
fn dyn_ad_tensor_compose_complex_rejects_non_real_inputs() {
    let re = AdTensor::new_primal(
        Tensor::<Complex64>::from_slice(
            &[Complex64::new(1.0, 0.0)],
            &[1],
            MemoryOrder::ColumnMajor,
        )
        .unwrap(),
    );
    let im = AdTensor::new_primal(
        Tensor::<f64>::from_slice(&[2.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
    );
    let err = match DynAdTensor::compose_complex(DynAdTensor::C64(re), DynAdTensor::F64(im)) {
        Ok(_) => panic!("compose_complex should reject non-real input"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::InvalidAdTensor { .. }));
}

#[test]
fn dyn_ad_tensor_compose_complex_checks_reverse_tape_compatibility() {
    let re = AdTensor::new_reverse(
        Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
        crate::NodeId(1),
        TapeId(7),
        None,
    )
    .unwrap();
    let im = AdTensor::new_reverse(
        Tensor::<f64>::from_slice(&[2.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
        crate::NodeId(2),
        TapeId(8),
        None,
    )
    .unwrap();
    let err = match DynAdTensor::compose_complex(DynAdTensor::F64(re), DynAdTensor::F64(im)) {
        Ok(_) => panic!("compose_complex should reject mixed reverse tapes"),
        Err(err) => err,
    };
    assert!(
        matches!(err, Error::InvalidAdTensor { message } if message.contains("reverse-mode tape mismatch"))
    );
}
