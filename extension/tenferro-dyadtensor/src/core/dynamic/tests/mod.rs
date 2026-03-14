mod organization;

use super::*;
use chainrules::Tape;
use chainrules_core::Differentiable;
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{AdMode, AdTensor, Error, StructuredTensor};

fn rank0_f64(value: f64) -> Tensor<f64> {
    Tensor::<f64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn rank0_f32(value: f32) -> Tensor<f32> {
    Tensor::<f32>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn rank0_c32(value: Complex32) -> Tensor<Complex32> {
    Tensor::<Complex32>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

fn rank0_c64(value: Complex64) -> Tensor<Complex64> {
    Tensor::<Complex64>::from_slice(&[value], &[], MemoryOrder::ColumnMajor).unwrap()
}

#[test]
fn dyn_scalar_metadata() {
    let x: DynScalar = 1.0_f64.into();
    assert_eq!(x.scalar_type(), ScalarType::F64);
    assert_eq!(x.as_f64(), Some(1.0));
}

#[test]
fn rank0_dyn_ad_tensor_mode_and_tangent() {
    let x: DynAdTensor = AdTensor::new_forward(
        Tensor::<f32>::from_slice(&[2.0_f32], &[], MemoryOrder::ColumnMajor).unwrap(),
        Tensor::<f32>::from_slice(&[0.5_f32], &[], MemoryOrder::ColumnMajor).unwrap(),
    )
    .unwrap()
    .into();
    assert_eq!(x.scalar_type(), ScalarType::F32);
    assert_eq!(x.mode(), AdMode::Forward);
    assert_eq!(
        x.as_f32()
            .unwrap()
            .tangent()
            .unwrap()
            .buffer()
            .as_slice()
            .unwrap(),
        &[0.5_f32]
    );
}

#[test]
fn dyn_tensor_and_dyn_ad_tensor_dims() {
    let t = Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let d: DynTensor = t.clone().into();
    assert_eq!(d.dims(), &[2]);
    assert_eq!(d.axis_classes(), &[0]);
    assert!(d.is_dense());

    let ad = AdTensor::new_primal(t);
    let dad: DynAdTensor = ad.into();
    assert_eq!(dad.dims(), &[2]);
    assert_eq!(dad.mode(), AdMode::Primal);
}

#[test]
fn dyn_tensor_preserves_diag_structure() {
    let diag = StructuredTensor::from_diagonal_vector(
        Tensor::<f64>::from_slice(&[2.0, 3.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
        2,
    )
    .unwrap();
    let x: DynTensor = diag.clone().into();
    assert!(x.is_diag());
    assert!(!x.is_dense());
    assert_eq!(x.dims(), &[2, 2]);
    assert_eq!(x.axis_classes(), &[0, 0]);
    let structured = x.as_f64().unwrap();
    assert_eq!(structured.logical_dims(), diag.logical_dims());
    assert_eq!(structured.axis_classes(), diag.axis_classes());
    assert_eq!(structured.payload().dims(), &[2]);
}

#[test]
fn dyn_tensor_is_valid_homogeneous_tape_payload() {
    let tape = Tape::<DynTensor>::new();
    let leaf = tape.leaf(
        StructuredTensor::from_diagonal_vector(
            Tensor::<f64>::from_slice(&[1.0, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
            2,
        )
        .unwrap()
        .into(),
    );
    assert!(leaf.requires_grad());
    assert!(leaf.tape().unwrap().same_tape(&tape));
    assert!(leaf.value().is_diag());
    assert_eq!(leaf.value().dims(), &[2, 2]);
}

#[test]
fn dyn_tensor_tangents_preserve_diag_layout() {
    let diag: DynTensor = StructuredTensor::from_diagonal_vector(
        Tensor::<f64>::from_slice(&[3.0, 4.0], &[2], MemoryOrder::ColumnMajor).unwrap(),
        2,
    )
    .unwrap()
    .into();
    let zero = diag.zero_tangent();
    let seed = diag.seed_cotangent();
    assert!(zero.is_diag());
    assert!(seed.is_diag());
    assert_eq!(zero.dims(), diag.dims());
    assert_eq!(seed.axis_classes(), diag.axis_classes());
}

#[test]
fn dyn_tensor_differentiable_contract_covers_all_runtime_variants() {
    let cases: Vec<(DynTensor, DynTensor, usize)> = vec![
        (
            StructuredTensor::from_dense(
                Tensor::<f32>::from_slice(&[1.0_f32, 2.0_f32], &[2], MemoryOrder::ColumnMajor)
                    .unwrap(),
            )
            .into(),
            StructuredTensor::from_dense(
                Tensor::<f32>::from_slice(&[0.5_f32, -0.5_f32], &[2], MemoryOrder::ColumnMajor)
                    .unwrap(),
            )
            .into(),
            2,
        ),
        (
            StructuredTensor::from_dense(
                Tensor::<f64>::from_slice(&[3.0_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
            )
            .into(),
            StructuredTensor::from_dense(
                Tensor::<f64>::from_slice(&[1.25_f64], &[], MemoryOrder::ColumnMajor).unwrap(),
            )
            .into(),
            1,
        ),
        (
            StructuredTensor::from_dense(
                Tensor::<Complex32>::from_slice(
                    &[Complex32::new(1.0, 2.0)],
                    &[],
                    MemoryOrder::ColumnMajor,
                )
                .unwrap(),
            )
            .into(),
            StructuredTensor::from_dense(
                Tensor::<Complex32>::from_slice(
                    &[Complex32::new(-0.25, 0.5)],
                    &[],
                    MemoryOrder::ColumnMajor,
                )
                .unwrap(),
            )
            .into(),
            1,
        ),
        (
            StructuredTensor::from_dense(
                Tensor::<Complex64>::from_slice(
                    &[Complex64::new(2.0, -1.0), Complex64::new(0.0, 3.0)],
                    &[2],
                    MemoryOrder::ColumnMajor,
                )
                .unwrap(),
            )
            .into(),
            StructuredTensor::from_dense(
                Tensor::<Complex64>::from_slice(
                    &[Complex64::new(0.5, 0.25), Complex64::new(-1.0, 1.0)],
                    &[2],
                    MemoryOrder::ColumnMajor,
                )
                .unwrap(),
            )
            .into(),
            2,
        ),
    ];

    for (primal, tangent, num_elements) in cases {
        let zero = primal.zero_tangent();
        let seed = primal.seed_cotangent();
        let accumulated = DynTensor::accumulate_tangent(zero.clone(), &tangent);

        assert_eq!(primal.num_elements(), num_elements);
        assert_eq!(zero.scalar_type(), primal.scalar_type());
        assert_eq!(seed.scalar_type(), primal.scalar_type());
        assert_eq!(accumulated.scalar_type(), primal.scalar_type());
        assert_eq!(zero.dims(), primal.dims());
        assert_eq!(seed.dims(), primal.dims());
        assert_eq!(accumulated.dims(), primal.dims());
    }
}

#[test]
fn rank0_dyn_ad_tensor_scale_mixed_real_complex_promotes_to_complex() {
    let lhs: DynAdTensor = AdTensor::new_primal(rank0_f64(2.0_f64)).into();
    let rhs: DynAdTensor = AdTensor::new_primal(rank0_c64(Complex64::new(1.0, -3.0))).into();
    let out = lhs.scale(&rhs).unwrap();
    assert_eq!(out.scalar_type(), ScalarType::C64);
    assert_eq!(
        out.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, -6.0)]
    );
}

#[test]
fn dyn_ad_tensor_promote_to_c64_lifts_real_rank0_tensor() {
    let x: DynAdTensor = AdTensor::new_primal(rank0_f64(2.0_f64)).into();
    let promoted = x.promote_to(ScalarType::C64).unwrap();
    assert_eq!(promoted.scalar_type(), ScalarType::C64);
    assert_eq!(
        promoted
            .as_c64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex64::new(2.0, 0.0)]
    );
}

#[test]
fn dyn_ad_tensor_promote_to_c32_lifts_real_rank0_tensor() {
    let x: DynAdTensor = AdTensor::new_primal(rank0_f32(2.0_f32)).into();
    let promoted = x.promote_to(ScalarType::C32).unwrap();
    assert_eq!(promoted.scalar_type(), ScalarType::C32);
    assert_eq!(
        promoted
            .as_c32()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex32::new(2.0, 0.0)]
    );
}

#[test]
fn dyn_ad_tensor_promote_to_identity_keeps_all_supported_variants() {
    let f32_value: DynAdTensor = AdTensor::new_primal(rank0_f32(1.0_f32)).into();
    let f64_value: DynAdTensor = AdTensor::new_primal(rank0_f64(2.0_f64)).into();
    let c32_value: DynAdTensor = AdTensor::new_primal(rank0_c32(Complex32::new(3.0, 1.0))).into();
    let c64_value: DynAdTensor = AdTensor::new_primal(rank0_c64(Complex64::new(4.0, -2.0))).into();

    assert!(matches!(
        f32_value.promote_to(ScalarType::F32).unwrap(),
        DynAdTensor::F32(_)
    ));
    assert!(matches!(
        f64_value.promote_to(ScalarType::F64).unwrap(),
        DynAdTensor::F64(_)
    ));
    assert!(matches!(
        c32_value.promote_to(ScalarType::C32).unwrap(),
        DynAdTensor::C32(_)
    ));
    assert!(matches!(
        c64_value.promote_to(ScalarType::C64).unwrap(),
        DynAdTensor::C64(_)
    ));
}

#[test]
fn dyn_ad_tensor_promote_to_rejects_cross_precision_casts() {
    let x: DynAdTensor = AdTensor::new_primal(rank0_f64(2.0_f64)).into();
    let err = match x.promote_to(ScalarType::F32) {
        Ok(_) => panic!("cross-precision promote_to should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(err, Error::InvalidAdTensor { message } if message.contains("unsupported promotion"))
    );
}

#[test]
fn rank0_dyn_ad_tensor_scale_rejects_cross_precision_join() {
    let lhs: DynAdTensor = AdTensor::new_primal(rank0_f32(2.0_f32)).into();
    let rhs: DynAdTensor = AdTensor::new_primal(rank0_f64(3.0_f64)).into();
    let err = match lhs.scale(&rhs) {
        Ok(_) => panic!("cross-precision scale should be rejected"),
        Err(err) => err,
    };
    assert!(
        matches!(err, Error::InvalidAdTensor { message } if message.contains("unsupported promotion"))
    );
}

#[test]
fn dyn_ad_tensor_promote_to_preserves_forward_tangent() {
    let x: DynAdTensor = AdTensor::new_forward(rank0_f64(2.0_f64), rank0_f64(0.5_f64))
        .unwrap()
        .into();
    let promoted = x.promote_to(ScalarType::C64).unwrap();
    assert_eq!(promoted.mode(), AdMode::Forward);
    let promoted = promoted.as_c64().unwrap();
    assert_eq!(
        promoted.tangent().unwrap().buffer().as_slice().unwrap(),
        &[Complex64::new(0.5, 0.0)]
    );
}

#[test]
fn dyn_ad_tensor_promote_to_rejects_mixed_dtype_reverse_promotion() {
    let tape = Tape::<crate::DynTensor>::new();
    let x: DynAdTensor = AdTensor::new_reverse_leaf(rank0_f64(2.0_f64), &tape)
        .unwrap()
        .into();
    let err = match x.promote_to(ScalarType::C64) {
        Ok(_) => panic!("mixed reverse promotion should stay unsupported"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "mixed_dtype_tensor_reverse"));
}

#[test]
fn dyn_ad_tensor_to_scalar_type_supports_cross_precision_primal_casts() {
    let x = DynAdTensor::new_primal(rank0_f32(2.0_f32));
    let as_f64 = x.to_scalar_type(ScalarType::F64).unwrap();
    assert_eq!(as_f64.scalar_type(), ScalarType::F64);
    assert_eq!(
        as_f64
            .as_f64()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[2.0_f64]
    );

    let y = DynAdTensor::new_primal(rank0_c64(Complex64::new(1.5, -0.25)));
    let as_c32 = y.to_scalar_type(ScalarType::C32).unwrap();
    assert_eq!(as_c32.scalar_type(), ScalarType::C32);
    assert_eq!(
        as_c32
            .as_c32()
            .unwrap()
            .primal()
            .buffer()
            .as_slice()
            .unwrap(),
        &[Complex32::new(1.5_f32, -0.25_f32)]
    );
}

#[test]
fn dyn_ad_tensor_to_scalar_type_casts_reverse_grad_back_to_input_dtype() {
    let tape = Tape::<crate::DynTensor>::new();
    let x: DynAdTensor = AdTensor::new_reverse_leaf(rank0_f32(2.0_f32), &tape)
        .unwrap()
        .into();
    let y = x.to_scalar_type(ScalarType::F64).unwrap();
    let y = y.as_f64().unwrap();
    let tracked = tape
        .tracked_existing(
            y.node_id().unwrap(),
            crate::DynTensor::from(y.structured_primal().clone()),
            y.structured_tangent().cloned().map(crate::DynTensor::from),
        )
        .unwrap();
    let grads = tape
        .pullback_with_seed(
            &tracked,
            crate::DynTensor::from(
                y.structured_primal()
                    .with_payload_like(rank0_f64(1.5_f64))
                    .unwrap(),
            ),
        )
        .unwrap();
    let grad = grads.get(x.node_id().unwrap()).unwrap();
    assert_eq!(
        grad.payload_f32().unwrap().buffer().as_slice().unwrap(),
        &[1.5_f32]
    );
}

#[test]
fn dyn_ad_tensor_div_with_scalar_lhs_is_supported() {
    let rhs: DynAdTensor = AdTensor::new_primal(rank0_f64(2.0_f64)).into();
    let lhs: DynAdTensor = AdTensor::new_primal(rank0_c64(Complex64::new(4.0, -2.0))).into();
    let out = lhs.div_scalar(&rhs).unwrap();
    assert_eq!(out.scalar_type(), ScalarType::C64);
    assert_eq!(
        out.as_c64().unwrap().primal().buffer().as_slice().unwrap(),
        &[Complex64::new(2.0, -1.0)]
    );
}

#[test]
fn dyn_ad_tensor_scale_checks_reverse_tape_compatibility() {
    let tensor_tape = Tape::<crate::DynTensor>::new();
    let scalar_tape = Tape::<crate::DynTensor>::new();
    let lhs: DynAdTensor = AdTensor::new_reverse_leaf(rank0_f64(2.0_f64), &tensor_tape)
        .unwrap()
        .into();
    let rhs: DynAdTensor = AdTensor::new_reverse_leaf(rank0_f64(3.0_f64), &scalar_tape)
        .unwrap()
        .into();
    let err = match lhs.scale(&rhs) {
        Ok(_) => panic!("mixed reverse tapes should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        Error::MixedReverseTape {
            expected: e,
            found: f
        } if e == tensor_tape.id() as u64 && f == scalar_tape.id() as u64
    ));
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
    let data = yr.payload().buffer().as_slice().unwrap();
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
    let re = AdTensor::new_reverse_leaf(
        Tensor::<f64>::from_slice(&[1.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
        &Tape::<crate::DynTensor>::new(),
    )
    .unwrap();
    let im = AdTensor::new_reverse_leaf(
        Tensor::<f64>::from_slice(&[2.0], &[1], MemoryOrder::ColumnMajor).unwrap(),
        &Tape::<crate::DynTensor>::new(),
    )
    .unwrap();
    let err = match DynAdTensor::compose_complex(DynAdTensor::F64(re), DynAdTensor::F64(im)) {
        Ok(_) => panic!("compose_complex should reject mixed reverse tapes"),
        Err(err) => err,
    };
    assert!(matches!(err, Error::UnsupportedAdOp { op } if op == "mixed_dtype_tensor_reverse"));
}
