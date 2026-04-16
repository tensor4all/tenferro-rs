use crate::DType;
use crate::TensorBackend;

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c64, tensor_f64, upload,
};

#[test]
#[ignore]
fn test_cubecl_structural_ops_match_cpu() {
    let input = tensor_f64(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let scalar = tensor_f64(vec![], vec![7.5]);
    let vector = tensor_f64(vec![3], vec![10.0, 20.0, 30.0]);
    let matrix = tensor_f64(
        vec![3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let gpu_scalar = upload(&gpu, &scalar);
    let gpu_vector = upload(&gpu, &vector);

    let expected = cpu.transpose(&input, &[1, 0]).unwrap();
    let gpu_out = gpu.transpose(&gpu_input, &[1, 0]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reshape(&input, &[3, 2]).unwrap();
    let gpu_out = gpu.reshape(&gpu_input, &[3, 2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.broadcast_in_dim(&scalar, &[2, 3], &[]).unwrap();
    let gpu_out = gpu.broadcast_in_dim(&gpu_scalar, &[2, 3], &[]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.reverse(&input, &[1]).unwrap();
    let gpu_out = gpu.reverse(&gpu_input, &[1]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.concatenate(&[&input, &input], 1).unwrap();
    let gpu_concat = gpu.concatenate(&[&gpu_input, &gpu_input], 1).unwrap();
    let actual = download(&gpu, &gpu_concat);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.extract_diagonal(&matrix, 0, 1).unwrap();
    let gpu_matrix = upload(&gpu, &matrix);
    let gpu_out = gpu.extract_diagonal(&gpu_matrix, 0, 1).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.embed_diagonal(&vector, 0, 1).unwrap();
    let gpu_out = gpu.embed_diagonal(&gpu_vector, 0, 1).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.tril(&matrix, 0).unwrap();
    let gpu_out = gpu.tril(&gpu_matrix, 0).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.triu(&matrix, -1).unwrap();
    let gpu_out = gpu.triu(&gpu_matrix, -1).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_convert_matches_cpu() {
    let real = tensor_f64(vec![3], vec![1.5, -2.25, 3.75]);
    let complex = tensor_c64(
        vec![2],
        vec![
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(-3.5, 0.5),
        ],
    );

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_real = upload(&gpu, &real);
    let gpu_complex = upload(&gpu, &complex);

    let expected = cpu.convert(&real, DType::F32).unwrap();
    let gpu_out = gpu.convert(&gpu_real, DType::F32).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-6);

    let expected = cpu.convert(&real, DType::C64).unwrap();
    let gpu_out = gpu.convert(&gpu_real, DType::C64).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.convert(&complex, DType::F64).unwrap();
    let gpu_out = gpu.convert(&gpu_complex, DType::F64).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}
