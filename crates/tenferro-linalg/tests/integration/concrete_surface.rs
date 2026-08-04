use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::{
    EighOptions, LinalgBackend, QrOptions, SvdOptions, TensorLinalgExt, TensorReadLinalgExt,
    TypedTensorLinalgExt,
};
use tenferro_tensor::{Tensor, TensorRead, TensorScalar, TypedTensor};

use super::support;

#[test]
fn cpu_execution_session_implements_linalg_backend() {
    fn assert_linalg_backend<B: LinalgBackend>() {}

    assert_linalg_backend::<tenferro_cpu::CpuExecSession<'static>>();
}

#[test]
fn dynamic_and_read_surfaces_return_fixed_tuples() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        let (_u, s, _vt) = input.svd(backend).unwrap();
        let (_q, r) = TensorRead::from_tensor(&input).qr_read(backend).unwrap();
        let (sign, logabsdet) = input.slogdet(backend).unwrap();

        assert_eq!(s.as_slice::<f64>().unwrap(), &[4.0, 2.0]);
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(sign.as_slice::<f64>().unwrap(), &[1.0]);
        assert!((logabsdet.as_slice::<f64>().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
    });
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

    support::with_cpu_linalg(&mut backend, |backend| {
        let (_u, singular_values, _vt): (TypedTensor<f64>, TypedTensor<f64>, TypedTensor<f64>) =
            real.svd(backend).unwrap();
        let (eigenvalues, _vectors): (TypedTensor<Complex64>, TypedTensor<Complex64>) =
            real.eig(backend).unwrap();
        let (_cu, complex_singular_values, _cvt): (
            TypedTensor<Complex64>,
            TypedTensor<f64>,
            TypedTensor<Complex64>,
        ) = complex.svd(backend).unwrap();

        assert_eq!(singular_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(complex_singular_values.as_slice().unwrap(), &[4.0, 2.0]);
        assert_eq!(eigenvalues.shape(), &[2]);
    });
}

#[test]
fn typed_input_is_erased_as_a_borrowed_read() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0_f64, 2.0, 2.0, 3.0]).unwrap();
    let read = f64::tensor_read(&input);
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        let factor = read.cholesky_read(backend).unwrap();
        assert_eq!(factor.shape(), &[2, 2]);
    });
}

#[test]
fn dynamic_composites_cover_inverse_pseudoinverse_eigenvalues_and_norm() {
    let input = Tensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        let det = input.det(backend).unwrap();
        let inv = input.inv(backend).unwrap();
        let pinv = input.pinv(backend).unwrap();
        let eigvalsh = input.eigvalsh(backend).unwrap();
        let eigvals = input.eigvals(backend).unwrap();
        let norm = input.norm(None, Some(&[0, 1]), true, backend).unwrap();

        assert!((det.as_slice::<f64>().unwrap()[0] - 8.0).abs() < 1.0e-12);
        assert_eq!(inv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
        assert_eq!(pinv.as_slice::<f64>().unwrap(), &[0.5, 0.0, 0.0, 0.25]);
        assert_eq!(eigvalsh.as_slice::<f64>().unwrap(), &[2.0, 4.0]);
        assert_eq!(eigvals.dtype(), tenferro_tensor::DType::C64);
        assert_eq!(norm.shape(), &[1, 1]);
        assert!((norm.as_slice::<f64>().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
    });
}

#[test]
fn read_surface_accepts_a_strided_view_without_an_input_clone() {
    let input =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0])
            .unwrap();
    let transposed = input.as_view().transpose_view([1, 0]).unwrap();
    let read = TensorRead::from_view(f64::tensor_view(transposed));
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        let (_u, singular_values, _vt) = read.svd_read(backend).unwrap();
        assert_eq!(singular_values.as_slice::<f64>().unwrap(), &[2.0, 1.0]);
    });
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

    support::with_cpu_linalg(&mut backend, |backend| {
        let (_sign, logabsdet): (TypedTensor<Complex64>, TypedTensor<f64>) =
            input.slogdet(backend).unwrap();
        let norm: TypedTensor<f64> = input.norm(None, Some(&[0, 1]), false, backend).unwrap();

        assert!((logabsdet.as_slice().unwrap()[0] - 8.0_f64.ln()).abs() < 1.0e-12);
        assert!((norm.as_slice().unwrap()[0] - 20.0_f64.sqrt()).abs() < 1.0e-12);
    });
}

#[test]
fn typed_solve_surfaces_accept_vector_and_matrix_rhs() {
    let a =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![2.0_f64, 0.0, 0.0, 4.0]).unwrap();
    let vector = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![4.0, 8.0]).unwrap();
    let matrix = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![4.0, 8.0]).unwrap();
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        let vector_x = a.full_piv_lu_solve(&vector, backend).unwrap();
        let matrix_x = a.full_piv_lu_solve(&matrix, backend).unwrap();
        let triangular_x = a
            .triangular_solve(&matrix, true, false, false, false, backend)
            .unwrap();

        assert_eq!(vector_x.as_slice().unwrap(), &[2.0, 2.0]);
        assert_eq!(matrix_x.as_slice().unwrap(), &[2.0, 2.0]);
        assert_eq!(triangular_x.as_slice().unwrap(), &[2.0, 2.0]);
    });
}

#[test]
fn concrete_norm_distinguishes_empty_axes_and_rejects_invalid_axes() {
    let input = Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap();
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        let identity = input.norm(Some(2.0), Some(&[]), false, backend).unwrap();
        let error = input
            .norm(Some(2.0), Some(&[1]), false, backend)
            .unwrap_err();

        assert_eq!(identity.as_slice::<f64>().unwrap(), &[3.0, 4.0]);
        assert!(error.to_string().contains("axis 1"));
    });
}

#[test]
fn read_surface_covers_factorizations_and_composites() {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![4.0_f64, 1.0, 1.0, 3.0]).unwrap();
    let b = Tensor::from_vec_col_major(vec![2], vec![9.0_f64, 7.0]).unwrap();
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        TensorRead::from_tensor(&a)
            .svd_with_options_read(SvdOptions::default(), backend)
            .unwrap();
        TensorRead::from_tensor(&a)
            .qr_with_options_read(QrOptions::default(), backend)
            .unwrap();
        TensorRead::from_tensor(&a).lu_read(backend).unwrap();
        TensorRead::from_tensor(&a)
            .full_piv_lu_read(backend)
            .unwrap();
        let x = TensorRead::from_tensor(&a)
            .full_piv_lu_solve_read(TensorRead::from_tensor(&b), backend)
            .unwrap();
        TensorRead::from_tensor(&a)
            .solve_read(TensorRead::from_tensor(&b), backend)
            .unwrap();
        TensorRead::from_tensor(&a)
            .eigh_with_options_read(EighOptions::default(), backend)
            .unwrap();
        TensorRead::from_tensor(&a).eig_read(backend).unwrap();
        TensorRead::from_tensor(&a).slogdet_read(backend).unwrap();
        TensorRead::from_tensor(&a).det_read(backend).unwrap();
        TensorRead::from_tensor(&a).inv_read(backend).unwrap();
        TensorRead::from_tensor(&a).eigvalsh_read(backend).unwrap();
        TensorRead::from_tensor(&a).eigvals_read(backend).unwrap();
        TensorRead::from_tensor(&a).pinv_read(backend).unwrap();
        TensorRead::from_tensor(&a)
            .pinv_with_rtol_read(1.0e-12, backend)
            .unwrap();
        TensorRead::from_tensor(&a)
            .norm_read(Some(2.0), Some(&[0, 1]), false, backend)
            .unwrap();

        let actual = x.as_slice::<f64>().unwrap();
        assert!((actual[0] - 20.0 / 11.0).abs() < 1.0e-12);
        assert!((actual[1] - 19.0 / 11.0).abs() < 1.0e-12);
    });
}

#[test]
fn typed_surface_covers_all_receiver_adapters() {
    let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![4.0, 1.0, 1.0, 3.0]).unwrap();
    let b = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![9.0, 7.0]).unwrap();
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        a.svd_with_options(SvdOptions::default(), backend).unwrap();
        a.qr(backend).unwrap();
        a.qr_with_options(QrOptions::default(), backend).unwrap();
        a.lu(backend).unwrap();
        a.full_piv_lu(backend).unwrap();
        a.solve(&b, backend).unwrap();
        a.cholesky(backend).unwrap();
        a.eigh(backend).unwrap();
        a.eigh_with_options(EighOptions::default(), backend)
            .unwrap();
        a.det(backend).unwrap();
        a.inv(backend).unwrap();
        a.eigvalsh(backend).unwrap();
        a.eigvals(backend).unwrap();
        a.pinv(backend).unwrap();
        a.pinv_with_rtol(1.0e-12, backend).unwrap();
        let norm = a
            .norm(Some(f64::INFINITY), Some(&[0, 1]), false, backend)
            .unwrap();

        assert_eq!(norm.as_slice().unwrap(), &[5.0]);
    });
}

#[test]
fn concrete_norm_covers_orders_axis_permutation_and_validation() {
    let matrix = Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let tensor = Tensor::from_vec_col_major(
        vec![2, 2, 2],
        vec![1.0_f64, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
    )
    .unwrap();
    let mut backend = CpuBackend::new();

    support::with_cpu_linalg(&mut backend, |backend| {
        for order in [
            Some(0.0),
            Some(1.0),
            Some(-1.0),
            Some(2.0),
            Some(-2.0),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ] {
            matrix.norm(order, Some(&[0, 1]), false, backend).unwrap();
        }
        tensor.norm(None, Some(&[2, 0]), true, backend).unwrap();
        tensor.norm(Some(3.0), None, false, backend).unwrap();

        assert!(tensor.norm(None, Some(&[0, 0]), false, backend).is_err());
        assert!(tensor.norm(Some(0.0), None, false, backend).is_ok());
    });
}
