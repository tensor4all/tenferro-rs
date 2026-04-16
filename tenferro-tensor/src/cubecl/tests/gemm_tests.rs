use num_complex::{Complex32, Complex64};

use crate::DotGeneralConfig;
use crate::Tensor;
use crate::TensorBackend;

use super::{
    assert_tensor_close, cpu_backend, download, gpu_backend, tensor_c32, tensor_c64, tensor_f32,
    tensor_f64, upload,
};

fn run_dot_general_case(lhs: Tensor, rhs: Tensor, config: DotGeneralConfig, tol: f64) {
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let expected = cpu.dot_general(&lhs, &rhs, &config).unwrap();
    let gpu_lhs = upload(&gpu, &lhs);
    let gpu_rhs = upload(&gpu, &rhs);
    let actual_gpu = gpu.dot_general(&gpu_lhs, &gpu_rhs, &config).unwrap();
    let actual = download(&gpu, &actual_gpu);

    assert_eq!(actual.shape(), expected.shape());
    assert_tensor_close(&actual, &expected, tol);
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
        lhs_rank: 2,
        rhs_rank: 2,
    }
}

#[test]
fn test_dot_general_matmul_f32() {
    run_dot_general_case(
        tensor_f32(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        tensor_f32(
            vec![3, 4],
            vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ],
        ),
        matmul_config(),
        1e-4,
    );
}

#[test]
fn test_dot_general_matmul_f64() {
    run_dot_general_case(
        tensor_f64(vec![2, 3], vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]),
        tensor_f64(
            vec![3, 4],
            vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ],
        ),
        matmul_config(),
        1e-9,
    );
}

#[test]
fn test_dot_general_batched_matmul_f32() {
    run_dot_general_case(
        tensor_f32(
            vec![2, 3, 5],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            ],
        ),
        tensor_f32(
            vec![3, 4, 5],
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                10.0, 11.0, 12.0, 13.0, 14.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
        ),
        DotGeneralConfig {
            lhs_contracting_dims: vec![1],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![2],
            rhs_batch_dims: vec![2],
            lhs_rank: 3,
            rhs_rank: 3,
        },
        1e-4,
    );
}

#[test]
fn test_dot_general_complex_matmul_c32() {
    run_dot_general_case(
        tensor_c32(
            vec![2, 3],
            vec![
                Complex32::new(1.0, 0.5),
                Complex32::new(2.0, -1.0),
                Complex32::new(3.0, 0.25),
                Complex32::new(4.0, 1.0),
                Complex32::new(5.0, -0.5),
                Complex32::new(6.0, 0.75),
            ],
        ),
        tensor_c32(
            vec![3, 4],
            vec![
                Complex32::new(1.0, -1.0),
                Complex32::new(2.0, 0.25),
                Complex32::new(3.0, 0.5),
                Complex32::new(4.0, -0.75),
                Complex32::new(5.0, 0.0),
                Complex32::new(6.0, 1.0),
                Complex32::new(7.0, -0.5),
                Complex32::new(8.0, 0.75),
                Complex32::new(9.0, -1.25),
                Complex32::new(10.0, 0.5),
                Complex32::new(11.0, 1.5),
                Complex32::new(12.0, -0.25),
            ],
        ),
        matmul_config(),
        1e-4,
    );
}

#[test]
fn test_dot_general_complex_matmul_c64() {
    run_dot_general_case(
        tensor_c64(
            vec![2, 3],
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.25),
                Complex64::new(4.0, 1.0),
                Complex64::new(5.0, -0.5),
                Complex64::new(6.0, 0.75),
            ],
        ),
        tensor_c64(
            vec![3, 4],
            vec![
                Complex64::new(1.0, -1.0),
                Complex64::new(2.0, 0.25),
                Complex64::new(3.0, 0.5),
                Complex64::new(4.0, -0.75),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 1.0),
                Complex64::new(7.0, -0.5),
                Complex64::new(8.0, 0.75),
                Complex64::new(9.0, -1.25),
                Complex64::new(10.0, 0.5),
                Complex64::new(11.0, 1.5),
                Complex64::new(12.0, -0.25),
            ],
        ),
        matmul_config(),
        1e-9,
    );
}
