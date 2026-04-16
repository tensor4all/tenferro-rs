use crate::config::{PadConfig, ScatterConfig, SliceConfig};
use crate::TensorBackend;

use super::{
    assert_tensor_close, cpu_backend, diagonal_scatter_config, download, gpu_backend,
    simple_gather_config, tensor_f64, upload,
};

#[test]
#[ignore]
fn test_cubecl_slice_dynamic_slice_and_pad_match_cpu() {
    let input = tensor_f64(
        vec![4, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    );
    let slice_config = SliceConfig {
        starts: vec![1, 1],
        limits: vec![4, 4],
        strides: vec![2, 1],
    };
    let pad_config = PadConfig {
        edge_padding_low: vec![1, 1],
        edge_padding_high: vec![1, 0],
        interior_padding: vec![0, 1],
    };
    let starts = tensor_f64(vec![2], vec![2.0, 3.0]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_input = upload(&gpu, &input);
    let gpu_starts = upload(&gpu, &starts);

    let expected = cpu.slice(&input, &slice_config).unwrap();
    let gpu_out = gpu.slice(&gpu_input, &slice_config).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.dynamic_slice(&input, &starts, &[2, 2]).unwrap();
    let gpu_out = gpu.dynamic_slice(&gpu_input, &gpu_starts, &[2, 2]).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let expected = cpu.pad(&input, &pad_config).unwrap();
    let gpu_out = gpu.pad(&gpu_input, &pad_config).unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_gather_and_scatter_match_cpu() {
    let operand = tensor_f64(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let start_indices = tensor_f64(vec![3, 1], vec![0.0, 2.0, 4.0]);

    let scatter_operand = tensor_f64(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );
    let scatter_indices = tensor_f64(vec![3, 2], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    let updates = tensor_f64(vec![3], vec![5.0, 6.0, 7.0]);

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();

    let gpu_operand = upload(&gpu, &operand);
    let gpu_start_indices = upload(&gpu, &start_indices);
    let expected = cpu
        .gather(&operand, &start_indices, &simple_gather_config())
        .unwrap();
    let gpu_out = gpu
        .gather(&gpu_operand, &gpu_start_indices, &simple_gather_config())
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);

    let gpu_scatter_operand = upload(&gpu, &scatter_operand);
    let gpu_scatter_indices = upload(&gpu, &scatter_indices);
    let gpu_updates = upload(&gpu, &updates);
    let expected = cpu
        .scatter(
            &scatter_operand,
            &scatter_indices,
            &updates,
            &diagonal_scatter_config(),
        )
        .unwrap();
    let gpu_out = gpu
        .scatter(
            &gpu_scatter_operand,
            &gpu_scatter_indices,
            &gpu_updates,
            &diagonal_scatter_config(),
        )
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}

#[test]
#[ignore]
fn test_cubecl_scatter_skips_invalid_windows_like_cpu() {
    let operand = tensor_f64(vec![4], vec![9.0, 8.0, 7.0, 6.0]);
    let scatter_indices = tensor_f64(vec![3, 1], vec![-1.0, 2.0, 4.0]);
    let updates = tensor_f64(vec![3], vec![5.0, 6.0, 7.0]);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gpu_operand = upload(&gpu, &operand);
    let gpu_indices = upload(&gpu, &scatter_indices);
    let gpu_updates = upload(&gpu, &updates);

    let expected = cpu
        .scatter(&operand, &scatter_indices, &updates, &config)
        .unwrap();
    let gpu_out = gpu
        .scatter(&gpu_operand, &gpu_indices, &gpu_updates, &config)
        .unwrap();
    let actual = download(&gpu, &gpu_out);
    assert_tensor_close(&actual, &expected, 1e-12);
}
