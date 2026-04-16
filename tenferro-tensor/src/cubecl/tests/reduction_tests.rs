use crate::TensorBackend;

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c64, tensor_f64, upload,
};

#[test]
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
