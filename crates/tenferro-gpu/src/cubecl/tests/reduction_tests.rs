// Run with: cargo test --features cuda -- --ignored
use tenferro_tensor::TensorReduction;

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_bool, tensor_c64, tensor_f64,
    tensor_i32, tensor_i64, upload,
};

#[test]
#[ignore]
fn test_cubecl_float_reductions_match_cpu() {
    let input = tensor_f64(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_prod(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_max(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_max(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_min(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_min(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_complex_sum_and_prod_match_cpu() {
    let input = tensor_c64(
        vec![2, 2],
        vec![
            num_complex::Complex64::new(1.0, 1.0),
            num_complex::Complex64::new(2.0, -1.0),
            num_complex::Complex64::new(-0.5, 0.25),
            num_complex::Complex64::new(3.0, 2.0),
        ],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reduce_prod(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let err = gpu.reduce_max(&gpu_input, &[0]).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure {
            op: "reduce_max",
            ..
        }
    ));

    let err = gpu.reduce_min(&gpu_input, &[0]).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure {
            op: "reduce_min",
            ..
        }
    ));
}

#[test]
#[ignore]
fn test_cubecl_i64_sum_and_prod_match_cpu() {
    let input = tensor_i64(vec![2, 3, 2], vec![1, 2, 3, 4, 5, 6, -1, -2, 2, 3, -3, 4]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_prod(&input, &[2]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_i32_sum_and_prod_match_cpu() {
    let input = tensor_i32(vec![2, 3, 2], vec![1, 2, 3, 4, 5, 6, -1, -2, 2, 3, -3, 4]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_prod(&input, &[2]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_integer_reductions_wrap_on_overflow() {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let input = tensor_i32(vec![2, 2], vec![i32::MAX, 1, i32::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![2, 2], vec![i32::MIN, -1, i32::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_prod(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i64(vec![2, 1], vec![i64::MAX, 2]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_sum(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_sum(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_prod(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_prod(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i32(vec![2, 2], vec![i32::MIN, 1, i32::MAX, -5]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_max(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_max(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_min(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_min(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let input = tensor_i64(vec![2, 2], vec![i64::MIN, 7, i64::MAX, -9]);
    let gpu_input = upload(&gpu, &input);
    let expected = cpu.reduce_max(&input, &[0]).unwrap();
    let gpu_out = gpu.reduce_max(&gpu_input, &[0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);

    let expected = cpu.reduce_min(&input, &[1]).unwrap();
    let gpu_out = gpu.reduce_min(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 0.0);
}

#[test]
#[ignore]
fn test_cubecl_bool_reductions_are_unsupported() {
    let input = tensor_bool(vec![2, 3], vec![true, false, true, true, false, false]);

    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    let err = gpu.reduce_sum(&gpu_input, &[0]).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure {
            op: "reduce_sum",
            ..
        }
    ));

    let err = gpu.reduce_prod(&gpu_input, &[1]).unwrap_err();
    assert!(matches!(
        err,
        crate::Error::BackendFailure {
            op: "reduce_prod",
            ..
        }
    ));
}

#[test]
#[ignore]
fn test_cubecl_reductions_column_major_3d_axes_match_cpu() {
    let input = tensor_f64(
        vec![2, 3, 4],
        (1..=24).map(|value| value as f64 - 7.0).collect(),
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);

    for axes in [&[0][..], &[1][..], &[2][..], &[0, 2][..]] {
        let expected = cpu.reduce_sum(&input, axes).unwrap();
        let gpu_out = gpu.reduce_sum(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_prod(&input, axes).unwrap();
        let gpu_out = gpu.reduce_prod(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_max(&input, axes).unwrap();
        let gpu_out = gpu.reduce_max(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);

        let expected = cpu.reduce_min(&input, axes).unwrap();
        let gpu_out = gpu.reduce_min(&gpu_input, axes).unwrap();
        let actual = download(&gpu, &gpu_out);
        assert_eq!(actual.shape(), expected.shape());
        assert_tensor_close(&actual, &expected, 1e-12);
    }
}
