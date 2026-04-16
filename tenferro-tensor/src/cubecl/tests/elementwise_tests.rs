use crate::config::CompareDir;
use crate::{DType, TensorBackend};

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c64, tensor_f64, upload,
};

#[test]
fn test_cubecl_binary_float_elementwise_matches_cpu() {
    let lhs = tensor_f64(vec![4], vec![1.5, -2.0, 3.0, 4.5]);
    let rhs = tensor_f64(vec![4], vec![0.5, 4.0, 2.0, -1.5]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);

    let expected = cpu.add(&lhs, &rhs).unwrap();
    let gpu_out = gpu.add(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.mul(&lhs, &rhs).unwrap();
    let gpu_out = gpu.mul(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.div(&lhs, &rhs).unwrap();
    let gpu_out = gpu.div(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.maximum(&lhs, &rhs).unwrap();
    let gpu_out = gpu.maximum(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.minimum(&lhs, &rhs).unwrap();
    let gpu_out = gpu.minimum(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu
        .pow(
            &tensor_f64(vec![4], vec![1.5, 2.0, 3.0, 4.0]),
            &tensor_f64(vec![4], vec![2.0, 3.0, 0.5, 1.0]),
        )
        .unwrap();
    let gpu_base = upload(&gpu, &tensor_f64(vec![4], vec![1.5, 2.0, 3.0, 4.0]));
    let gpu_exp = upload(&gpu, &tensor_f64(vec![4], vec![2.0, 3.0, 0.5, 1.0]));
    let gpu_out = gpu.pow(&gpu_base, &gpu_exp).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
fn test_cubecl_unary_float_elementwise_matches_cpu() {
    let positive = tensor_f64(vec![4], vec![0.25, 0.5, 1.5, 3.0]);
    let signed = tensor_f64(vec![4], vec![-2.0, -0.0, 3.5, -4.5]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_positive = upload(&gpu, &positive);
    let gpu_signed = upload(&gpu, &signed);

    let expected = cpu.neg(&signed).unwrap();
    let gpu_out = gpu.neg(&gpu_signed).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.abs(&signed).unwrap();
    let gpu_out = gpu.abs(&gpu_signed).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.sign(&signed).unwrap();
    let gpu_out = gpu.sign(&gpu_signed).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.exp(&positive).unwrap();
    let gpu_out = gpu.exp(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.log(&positive).unwrap();
    let gpu_out = gpu.log(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.sin(&positive).unwrap();
    let gpu_out = gpu.sin(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.cos(&positive).unwrap();
    let gpu_out = gpu.cos(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.tanh(&positive).unwrap();
    let gpu_out = gpu.tanh(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.sqrt(&positive).unwrap();
    let gpu_out = gpu.sqrt(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.rsqrt(&positive).unwrap();
    let gpu_out = gpu.rsqrt(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.expm1(&positive).unwrap();
    let gpu_out = gpu.expm1(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.log1p(&positive).unwrap();
    let gpu_out = gpu.log1p(&gpu_positive).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
fn test_cubecl_float_compare_select_and_clamp_match_cpu() {
    let lhs = tensor_f64(vec![4], vec![1.0, 3.0, 2.0, 4.0]);
    let rhs = tensor_f64(vec![4], vec![2.0, 3.0, 1.0, 5.0]);
    let lower = tensor_f64(vec![4], vec![0.5, 2.0, 1.5, 3.5]);
    let upper = tensor_f64(vec![4], vec![1.5, 4.0, 2.5, 4.5]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let gpu_lower = upload(&gpu, &lower);
    let gpu_upper = upload(&gpu, &upper);

    let expected = cpu.compare(&lhs, &rhs, &CompareDir::Ge).unwrap();
    let gpu_out = gpu.compare(&gpu_lhs, &gpu_rhs, &CompareDir::Ge).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.select(&expected, &lhs, &rhs).unwrap();
    let gpu_pred = upload(&gpu, &actual);
    let gpu_out = gpu.select(&gpu_pred, &gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.clamp(&lhs, &lower, &upper).unwrap();
    let gpu_out = gpu.clamp(&gpu_lhs, &gpu_lower, &gpu_upper).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
fn test_cubecl_complex_elementwise_matches_cpu_and_rejects_unsupported_ops() {
    let lhs = tensor_c64(
        vec![3],
        vec![
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(-3.0, 0.5),
            num_complex::Complex64::new(0.25, -1.25),
        ],
    );
    let rhs = tensor_c64(
        vec![3],
        vec![
            num_complex::Complex64::new(-0.5, 1.0),
            num_complex::Complex64::new(2.0, -1.5),
            num_complex::Complex64::new(0.5, 0.25),
        ],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);

    let expected = cpu.add(&lhs, &rhs).unwrap();
    let gpu_out = gpu.add(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.mul(&lhs, &rhs).unwrap();
    let gpu_out = gpu.mul(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.div(&lhs, &rhs).unwrap();
    let gpu_out = gpu.div(&gpu_lhs, &gpu_rhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.neg(&lhs).unwrap();
    let gpu_out = gpu.neg(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.conj(&lhs).unwrap();
    let gpu_out = gpu.conj(&gpu_lhs).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let err = gpu.abs(&gpu_lhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "abs", .. }
    ));

    let err = gpu.exp(&gpu_lhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "exp", .. }
    ));

    let err = gpu
        .compare(&gpu_lhs, &gpu_rhs, &CompareDir::Eq)
        .unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "compare", .. }
    ));

    let err = gpu.select(&gpu_lhs, &gpu_lhs, &gpu_rhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "select", .. }
    ));

    let err = gpu.clamp(&gpu_lhs, &gpu_lhs, &gpu_rhs).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure { op: "clamp", .. }
    ));

    let err = gpu.convert(&gpu_lhs, DType::C64).unwrap();
    let actual = download(&gpu, &err);
    assert_tensor_close(&actual, &lhs, 1e-12);
}
