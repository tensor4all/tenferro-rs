use super::*;
use num_complex::{Complex32, Complex64};
use std::hint::black_box;
use tenferro_prims::CpuContext;

mod batch_a_contracts;
mod batch_b_contracts;
mod organization;
mod runtime_capability;

fn tensor_data(tensor: &Tensor<f64>) -> Vec<f64> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

fn tensor_data_complex64(tensor: &Tensor<Complex64>) -> Vec<Complex64> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[test]
fn vector_norm_primal_rrule_and_frule_match_expected_values() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::from_slice(&[3.0_f64, -4.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let dx = Tensor::from_slice(&[0.25_f64, -0.5], &[2], MemoryOrder::ColumnMajor).unwrap();
    let cotangent = Tensor::from_vec(vec![2.0_f64], &[], &[], 0).unwrap();

    let lp = norm(&mut ctx, &x, NormKind::Lp(2.0)).unwrap();
    assert!((tensor_data(&lp)[0] - 5.0).abs() < 1e-12);

    let grad = norm_rrule(&mut ctx, &x, &cotangent, NormKind::Fro).unwrap();
    let grad_data = tensor_data(&grad);
    assert!((grad_data[0] - 1.2).abs() < 1e-12);
    assert!((grad_data[1] + 1.6).abs() < 1e-12);

    let (nrm, dnrm) = norm_frule(&mut ctx, &x, &dx, NormKind::Lp(2.0)).unwrap();
    assert!((tensor_data(&nrm)[0] - 5.0).abs() < 1e-12);
    assert!((tensor_data(&dnrm)[0] - 0.55).abs() < 1e-12);
}

#[test]
fn vector_lp_norm_matches_manual_formula_and_rejects_invalid_p() {
    let mut ctx = CpuContext::new(1);
    let x = Tensor::from_slice(&[2.0_f64, -3.0, 4.0], &[3], MemoryOrder::ColumnMajor).unwrap();

    let lp = norm(&mut ctx, &x, NormKind::Lp(3.0)).unwrap();
    let expected = (2.0_f64.powi(3) + 3.0_f64.powi(3) + 4.0_f64.powi(3)).powf(1.0 / 3.0);
    assert!((tensor_data(&lp)[0] - expected).abs() < 1e-12);

    assert!(norm(&mut ctx, &x, NormKind::Lp(0.5)).is_err());
}

#[test]
fn linalg_scalar_helpers_and_validation_accept_valid_inputs() {
    assert_eq!(<f64 as LinalgScalar>::abs_real(&black_box(-1.5_f64)), 1.5);
    assert_eq!(<f32 as LinalgScalar>::abs_real(&black_box(-1.5_f32)), 1.5);
    assert!(black_box(<f32 as LinalgScalar>::real_epsilon()) > 0.0);
    assert_eq!(<f64 as LinalgScalar>::conj(&1.5), 1.5);
    assert_eq!(<f32 as LinalgScalar>::conj(&1.5_f32), 1.5_f32);

    let z64 = Complex64::new(3.0, -4.0);
    assert_eq!(<Complex64 as LinalgScalar>::abs_real(&z64), 5.0);
    assert!(<Complex64 as LinalgScalar>::real_epsilon() > 0.0);
    assert_eq!(
        <Complex64 as LinalgScalar>::conj(&z64),
        Complex64::new(3.0, 4.0)
    );

    let z32 = Complex32::new(-2.0, 1.5);
    assert_eq!(<Complex32 as LinalgScalar>::abs_real(&z32), z32.norm());
    assert!(<Complex32 as LinalgScalar>::real_epsilon() > 0.0);
    assert_eq!(
        <Complex32 as LinalgScalar>::conj(&z32),
        Complex32::new(-2.0, -1.5)
    );

    let square =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let (n, batch) = validate_square(&square).unwrap();
    assert_eq!(n, 2);
    assert!(batch.is_empty());

    let rhs = Tensor::from_slice(&[1.0_f64, -2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    validate_lstsq_rhs(&rhs, 2, &[]).unwrap();

    let scalar_cotangent = Tensor::from_vec(vec![1.0_f64], &[], &[], 0).unwrap();
    validate_norm_cotangent(&scalar_cotangent, &[]).unwrap();

    let hermitian = [
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, -2.0),
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 0.0),
    ];
    validate_hermitian_batches(&hermitian, 0, 2, 1, "eigh").unwrap();

    let slice = extract_slice(&square).unwrap();
    assert_eq!(slice, &[1.0, 0.0, 0.0, 1.0]);

    let scalar = scalar_from::<f32>(1.25).unwrap();
    assert_eq!(scalar, 1.25_f32);

    let ad_err = to_ad_err(Error::InvalidArgument("invalid shape".into()));
    assert!(
        matches!(ad_err, chainrules_core::AutodiffError::InvalidArgument(msg) if msg.contains("invalid shape"))
    );
}

#[test]
fn validation_helpers_reject_invalid_shapes_and_axes() {
    let wrong_rank_rhs =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = validate_lstsq_rhs(&wrong_rank_rhs, 2, &[2]).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("lstsq expects b shape")));

    let wrong_leading_rhs =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    let err = validate_lstsq_rhs(&wrong_leading_rhs, 2, &[]).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("dim[0] == m")));

    let wrong_batch_rhs =
        Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();
    let err = validate_lstsq_rhs(&wrong_batch_rhs, 2, &[3]).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("batch dims mismatch")));

    let nonscalar_cotangent =
        Tensor::from_slice(&[1.0_f64], &[1], MemoryOrder::ColumnMajor).unwrap();
    let err = validate_norm_cotangent(&nonscalar_cotangent, &[]).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("expected scalar")));

    let wrong_batch_cotangent =
        Tensor::from_slice(&[1.0_f64, 2.0], &[2], MemoryOrder::ColumnMajor).unwrap();
    let err = validate_norm_cotangent(&wrong_batch_cotangent, &[3]).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("shape mismatch")));

    let err = validate_tensor_solve_axes(4, 2, Some(&[0])).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("expects 2 solution axes")));

    let err = validate_tensor_solve_axes(3, 2, Some(&[0, 3])).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("out of bounds")));

    let err = validate_tensor_solve_axes(3, 2, Some(&[1, 1])).unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(msg) if msg.contains("must be unique")));

    assert_eq!(validate_tensor_solve_axes(4, 2, None).unwrap(), vec![2, 3]);
}

#[test]
fn svdvals_matches_svd_singular_values() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[1.0_f64, 0.0, 0.0, 2.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let svd_s = svd(&mut ctx, &a, None).unwrap().s;
    let values = svdvals(&mut ctx, &a).unwrap();

    assert_eq!(tensor_data(&values), tensor_data(&svd_s));
}

#[test]
fn slogdet_matches_expected_sign_and_logabsdet() {
    let mut ctx = CpuContext::new(1);
    let a =
        Tensor::from_slice(&[1.0_f64, 3.0, 2.0, 4.0], &[2, 2], MemoryOrder::ColumnMajor).unwrap();

    let result = slogdet(&mut ctx, &a).unwrap();
    let sign = tensor_data(&result.sign);
    let logabsdet = tensor_data(&result.logabsdet);

    assert!((sign[0] - (-1.0)).abs() < 1e-12);
    assert!((logabsdet[0] - (2.0_f64).ln()).abs() < 1e-12);
}

#[test]
fn matrix_exp_batch_1_norms_keep_batches_separate() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            6.0_f64, 0.0, 0.0, 6.0, // batch 0
            2.0, 0.5, -1.0, 1.5, // batch 1
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let norms = crate::ad_helpers::matrix_exp_batch_1_norms_tensor(&mut ctx, &a).unwrap();
    assert_eq!(norms.dims(), &[2]);
    assert_eq!(tensor_data(&norms), vec![6.0, 2.5]);
}

#[test]
fn blend_tensor_by_real_mask_same_shape_selects_complex_batches() {
    let mut ctx = CpuContext::new(1);
    let on_true = Tensor::from_slice(
        &[
            Complex64::new(2.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(3.0, -1.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let on_false = Tensor::from_slice(
        &[
            Complex64::new(5.0, 0.5),
            Complex64::new(5.0, 0.5),
            Complex64::new(5.0, 0.5),
            Complex64::new(5.0, 0.5),
            Complex64::new(7.0, 2.0),
            Complex64::new(7.0, 2.0),
            Complex64::new(7.0, 2.0),
            Complex64::new(7.0, 2.0),
        ],
        &[2, 2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let mask = Tensor::from_slice(&[1.0_f64, 0.0], &[2], MemoryOrder::ColumnMajor)
        .unwrap()
        .reshape(&[1, 1, 2])
        .unwrap()
        .broadcast(&[2, 2, 2])
        .unwrap();

    let blended = crate::ad_helpers::blend_tensor_by_real_mask_same_shape(
        &mut ctx, &on_true, &on_false, &mask,
    )
    .unwrap();
    assert_eq!(
        tensor_data_complex64(&blended),
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(7.0, 2.0),
            Complex64::new(7.0, 2.0),
            Complex64::new(7.0, 2.0),
            Complex64::new(7.0, 2.0),
        ]
    );
}

#[test]
fn svd_cutoff_fixed_shape_zero_fill_semantics_hold() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[3.0_f64, 0.0, 0.0, 0.25],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let opts = SvdOptions {
        max_rank: Some(2),
        cutoff: Some(0.5),
    };

    let result = svd(&mut ctx, &a, Some(&opts)).unwrap();

    assert_eq!(result.u.dims(), &[2, 2]);
    assert_eq!(result.s.dims(), &[2]);
    assert_eq!(result.vt.dims(), &[2, 2]);
    assert_eq!(tensor_data(&result.s), vec![3.0, 0.0]);
    assert_eq!(tensor_data(&result.u), vec![1.0, 0.0, 0.0, 0.0]);
    assert_eq!(tensor_data(&result.vt), vec![1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn complex_pinv_diagonal_matrix_matches_expected_inverse() {
    let mut ctx = CpuContext::new(1);
    let a = Tensor::from_slice(
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, -1.0),
        ],
        &[2, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let ap = pinv(&mut ctx, &a, None).unwrap();
    let data = tensor_data_complex64(&ap);
    let expected = [
        Complex64::new(0.5, -0.5),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.4, 0.2),
    ];

    assert_eq!(ap.dims(), &[2, 2]);
    for (got, want) in data.iter().zip(expected.iter()) {
        assert!(
            (*got - *want).norm() < 1.0e-10,
            "got {got:?}, want {want:?}"
        );
    }
}
