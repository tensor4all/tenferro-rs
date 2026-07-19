use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::{TensorLinalgExt, TensorReadLinalgExt, TypedTensorLinalgExt};
use tenferro_tensor::{Tensor, TensorRead, TensorScalar, TypedTensor};

#[test]
fn dynamic_and_read_surfaces_return_fixed_tuples() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut backend = CpuBackend::new();

    let (_u, s, _vt) = input.svd(&mut backend).unwrap();
    let (_q, r) = TensorRead::from_tensor(&input)
        .qr_read(&mut backend)
        .unwrap();
    let (sign, logabsdet) = input.slogdet(&mut backend).unwrap();

    assert_eq!(s.as_slice::<f64>().unwrap(), &[4.0, 2.0]);
    assert_eq!(r.shape(), &[2, 2]);
    assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
    assert!((logabsdet.as_slice::<f64>().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn typed_surface_exposes_associated_real_and_complex_outputs() {
    let real =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let complex = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let mut backend = CpuBackend::new();

    let (_u, singular_values, _vt): (TypedTensor<f64>, TypedTensor<f64>, TypedTensor<f64>) =
        real.svd(&mut backend).unwrap();
    let (eigenvalues, _vectors): (TypedTensor<Complex64>, TypedTensor<Complex64>) =
        real.eig(&mut backend).unwrap();
    let (_cu, complex_singular_values, _cvt): (
        TypedTensor<Complex64>,
        TypedTensor<f64>,
        TypedTensor<Complex64>,
    ) = complex.svd(&mut backend).unwrap();

    assert_eq!(singular_values.as_slice().unwrap(), &[4.0, 2.0]);
    assert_eq!(complex_singular_values.as_slice().unwrap(), &[4.0, 2.0]);
    assert_eq!(eigenvalues.shape(), &[2]);
}

#[test]
fn typed_input_is_erased_as_a_borrowed_read() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let read = f64::tensor_read(&input);
    let mut backend = CpuBackend::new();

    let factor = read.cholesky_read(&mut backend).unwrap();

    assert_eq!(factor.shape(), &[2, 2]);
}

#[test]
fn dynamic_composites_cover_inverse_pseudoinverse_eigenvalues_and_norm() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut backend = CpuBackend::new();

    let det = input.det(&mut backend).unwrap();
    let inv = input.inv(&mut backend).unwrap();
    let pinv = input.pinv(&mut backend).unwrap();
    let eigvalsh = input.eigvalsh(&mut backend).unwrap();
    let eigvals = input.eigvals(&mut backend).unwrap();
    let norm = input.norm(None, Some(&[0, 1]), true, &mut backend).unwrap();

    assert!((det.as_slice::<f64>().unwrap()[0] - 8.0).abs() < 1.0e-12);
    assert_eq!(inv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
    assert_eq!(pinv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
    assert_eq!(eigvalsh.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
    assert_eq!(eigvals.dtype(), tenferro_tensor::DType::C64);
    assert_eq!(norm.shape(), &[1, 1]);
    assert!((norm.as_slice::<f64>().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn read_surface_accepts_a_strided_view_without_an_input_clone() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
    let transposed = input.as_view().transpose_view([1, 0]).unwrap();
    let read = TensorRead::from_view(f64::tensor_view(transposed));
    let mut backend = CpuBackend::new();

    let (_u, singular_values, _vt) = read.svd_read(&mut backend).unwrap();

    assert_eq!(singular_values.as_slice::<f64>().unwrap(), &[2.0, 1.0]);
}

#[test]
fn typed_complex_composites_return_real_outputs_where_required() {
    let input = TypedTensor::<Complex64>::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(4.0, 0.0),
        ],
    )
    .unwrap();
    let mut backend = CpuBackend::new();

    let (_sign, logabsdet): (TypedTensor<Complex64>, TypedTensor<f64>) =
        input.slogdet(&mut backend).unwrap();
    let norm: TypedTensor<f64> = input
        .norm(None, Some(&[0, 1]), false, &mut backend)
        .unwrap();

    assert!((logabsdet.as_slice().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
    assert!((norm.as_slice().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn typed_solve_surfaces_accept_vector_and_matrix_rhs() {
    let a =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let vector = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 8.0]).unwrap();
    let matrix = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap();
    let mut backend = CpuBackend::new();

    let vector_x = a.full_piv_lu_solve(&vector, &mut backend).unwrap();
    let matrix_x = a.full_piv_lu_solve(&matrix, &mut backend).unwrap();
    let triangular_x = a
        .triangular_solve(&matrix, true, false, false, false, &mut backend)
        .unwrap();

    assert_eq!(vector_x.as_slice().unwrap(), &[2.0, 2.0]);
    assert_eq!(matrix_x.as_slice().unwrap(), &[2.0, 2.0]);
    assert_eq!(triangular_x.as_slice().unwrap(), &[2.0, 2.0]);
}

#[test]
fn concrete_norm_distinguishes_empty_axes_and_rejects_invalid_axes() {
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut backend = CpuBackend::new();

    let identity = input
        .norm(Some(2.0), Some(&[]), false, &mut backend)
        .unwrap();
    let error = input
        .norm(Some(2.0), Some(&[1]), false, &mut backend)
        .unwrap_err();

    assert_eq!(identity.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
    assert!(error.to_string().contains("axis 1"));
}
