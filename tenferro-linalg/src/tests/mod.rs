use super::*;
use num_complex::{Complex32, Complex64};
use std::hint::black_box;
use tenferro_prims::CpuContext;

mod batch_a_contracts;
mod batch_b_contracts;

fn tensor_data(tensor: &Tensor<f64>) -> Vec<f64> {
    let contiguous = tensor.contiguous(MemoryOrder::ColumnMajor);
    let offset = contiguous.offset() as usize;
    let len = contiguous.dims().iter().product::<usize>().max(1);
    contiguous.buffer().as_slice().unwrap()[offset..offset + len].to_vec()
}

#[test]
fn vector_norm_paths_are_covered_in_crate_unit_tests() {
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
fn private_scalar_and_validation_helpers_are_covered_in_crate_unit_tests() {
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

    assert_eq!(<f64 as LinalgScalar>::eig_buffer_sizes(2), (4, 8));
    assert_eq!(<f32 as LinalgScalar>::eig_buffer_sizes(2), (4, 8));
    assert_eq!(<Complex64 as LinalgScalar>::eig_buffer_sizes(2), (2, 4));
    assert_eq!(<Complex32 as LinalgScalar>::eig_buffer_sizes(2), (2, 4));

    let mut real_vals = vec![Complex64::new(0.0, 0.0); 2];
    let mut real_vecs = vec![Complex64::new(0.0, 0.0); 4];
    <f64 as LinalgScalar>::eig_ri_to_complex(
        2,
        &[1.0, 0.5, -2.0, 1.25],
        &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &mut real_vals,
        &mut real_vecs,
    );
    assert_eq!(
        real_vals,
        vec![Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.25)]
    );
    assert_eq!(
        real_vecs,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ]
    );

    let mut complex_vals = vec![Complex32::new(0.0, 0.0); 2];
    let mut complex_vecs = vec![Complex32::new(0.0, 0.0); 4];
    <Complex32 as LinalgScalar>::eig_ri_to_complex(
        2,
        &[Complex32::new(1.0, -0.5), Complex32::new(-2.0, 0.25)],
        &[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ],
        &mut complex_vals,
        &mut complex_vecs,
    );
    assert_eq!(
        complex_vals,
        vec![Complex32::new(1.0, -0.5), Complex32::new(-2.0, 0.25)]
    );
    assert_eq!(
        complex_vecs,
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(1.0, 0.0),
        ]
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

    let ad_err = to_ad_err(Error::InvalidArgument("coverage".into()));
    assert!(
        matches!(ad_err, chainrules_core::AutodiffError::InvalidArgument(msg) if msg.contains("coverage"))
    );
}

#[test]
fn private_validation_helpers_cover_remaining_error_paths() {
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
