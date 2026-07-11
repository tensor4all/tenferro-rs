// Run with: cargo test --features cuda -- --ignored
use crate::config::{PadConfig, ScatterConfig, SliceConfig};
use num_complex::Complex64;
use tenferro_tensor::TensorIndexing;

use super::{
    assert_tensor_close, cpu_backend, diagonal_scatter_config, download, gpu_backend,
    simple_gather_config, tensor_bool, tensor_c64, tensor_f64, tensor_i64, upload,
};

#[test]
#[ignore = "requires CUDA 12.8+ GPU"]
fn cuda_bool_indexing_ops_match_cpu() {
    let input = tensor_bool(vec![4], vec![true, false, true, false]);
    let starts = tensor_i64(vec![1], vec![1]);
    let indices = tensor_i64(vec![2, 1], vec![0, 3]);
    let slice = SliceConfig {
        starts: vec![1],
        limits: vec![4],
        strides: vec![2],
    };
    let pad = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![1],
    };
    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    let gi = upload(&gpu, &input);
    let gs = upload(&gpu, &starts);
    let gx = upload(&gpu, &indices);
    macro_rules! parity {
        ($cpu:expr, $gpu:expr) => {{
            let expected = $cpu.unwrap();
            let out = $gpu.unwrap();
            let actual = download(&gpu, &out);
            assert_tensor_close(&actual, &expected, 0.0);
        }};
    }
    parity!(cpu.slice(&input, &slice), gpu.slice(&gi, &slice));
    parity!(
        cpu.dynamic_slice(&input, &starts, &[2]),
        gpu.dynamic_slice(&gi, &gs, &[2])
    );
    parity!(cpu.pad(&input, &pad), gpu.pad(&gi, &pad));
    parity!(
        cpu.gather(&input, &indices, &simple_gather_config()),
        gpu.gather(&gi, &gx, &simple_gather_config())
    );
    let invalid = SliceConfig {
        starts: vec![0],
        limits: vec![5],
        strides: vec![1],
    };
    let cpu_err = cpu.slice(&input, &invalid).unwrap_err();
    let gpu_err = gpu.slice(&gi, &invalid).unwrap_err();
    assert_eq!(
        std::mem::discriminant(&cpu_err),
        std::mem::discriminant(&gpu_err)
    );

    let updates = tensor_bool(vec![2], vec![true, false]);
    let scatter_indices = tensor_i64(vec![2, 1], vec![0, 1]);
    let config = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    assert!(cpu
        .scatter(&input, &scatter_indices, &updates, &config)
        .is_err());
}

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
    let starts = tensor_i64(vec![2], vec![2, 3]);

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
#[ignore = "requires CUDA 12.8+ GPU"]
fn test_cubecl_signed_pad_cropping_matches_cpu_values() {
    let cases = [
        (
            tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
            PadConfig {
                edge_padding_low: vec![-1],
                edge_padding_high: vec![0],
                interior_padding: vec![0],
            },
            tensor_f64(vec![3], vec![2.0, 3.0, 4.0]),
        ),
        (
            tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
            PadConfig {
                edge_padding_low: vec![0],
                edge_padding_high: vec![-1],
                interior_padding: vec![0],
            },
            tensor_f64(vec![3], vec![1.0, 2.0, 3.0]),
        ),
        (
            tensor_f64(vec![4], vec![1.0, 2.0, 3.0, 4.0]),
            PadConfig {
                edge_padding_low: vec![-1],
                edge_padding_high: vec![0],
                interior_padding: vec![1],
            },
            tensor_f64(vec![6], vec![0.0, 2.0, 0.0, 3.0, 0.0, 4.0]),
        ),
        (
            tensor_f64(vec![2], vec![1.0, 2.0]),
            PadConfig {
                edge_padding_low: vec![i64::MIN],
                edge_padding_high: vec![i64::MAX],
                interior_padding: vec![0],
            },
            tensor_f64(vec![1], vec![0.0]),
        ),
    ];

    let mut cpu = cpu_backend();
    let mut gpu = gpu_backend();
    for (input, config, expected_values) in cases {
        let expected = cpu.pad(&input, &config).unwrap();
        assert_eq!(expected.shape(), expected_values.shape());
        assert_tensor_close(&expected, &expected_values, 0.0);

        let gpu_input = upload(&gpu, &input);
        let gpu_output = gpu.pad(&gpu_input, &config).unwrap();
        let actual = download(&gpu, &gpu_output);
        assert_eq!(actual.shape(), expected_values.shape());
        assert_tensor_close(&actual, &expected_values, 0.0);
        assert_tensor_close(&actual, &expected, 0.0);
    }
}

#[test]
#[ignore]
fn test_cubecl_gather_and_scatter_match_cpu() {
    let operand = tensor_f64(vec![5], vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let start_indices = tensor_i64(vec![3, 1], vec![0, 2, 4]);

    let scatter_operand = tensor_f64(
        vec![3, 3],
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
    );
    let scatter_indices = tensor_i64(vec![3, 2], vec![0, 1, 2, 0, 1, 2]);
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
    let scatter_indices = tensor_i64(vec![3, 1], vec![-1, 2, 4]);
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

#[test]
#[ignore]
fn test_cubecl_scatter_accumulates_overlapping_updates_like_cpu() {
    let operand = tensor_f64(vec![4], vec![1.0, 10.0, 100.0, 1000.0]);
    let scatter_indices = tensor_i64(vec![4, 1], vec![1, 1, -1, 3]);
    let updates = tensor_f64(vec![4], vec![2.0, 3.0, 99.0, 5.0]);
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

#[test]
#[ignore]
fn test_cubecl_complex_scatter_accumulates_overlapping_updates_like_cpu() {
    let operand = tensor_c64(
        vec![3],
        vec![
            Complex64::new(1.0, 10.0),
            Complex64::new(2.0, 20.0),
            Complex64::new(3.0, 30.0),
        ],
    );
    let scatter_indices = tensor_i64(vec![3, 1], vec![0, 0, 2]);
    let updates = tensor_c64(
        vec![3],
        vec![
            Complex64::new(4.0, 1.0),
            Complex64::new(5.0, 2.0),
            Complex64::new(6.0, 3.0),
        ],
    );
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
