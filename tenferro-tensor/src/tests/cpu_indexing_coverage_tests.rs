use num_complex::{Complex32, Complex64};

use crate::backend::TensorBackend;
use crate::config::{DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use crate::cpu::{dynamic_slice, gather, pad, scatter, CpuBackend};
use crate::types::{Tensor, TypedTensor};

fn simple_gather_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1],
    }
}

fn valid_gather_2d_config() -> GatherConfig {
    GatherConfig {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 2],
    }
}

fn diagonal_scatter_config() -> ScatterConfig {
    ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    }
}

fn expect_invalid_config(result: crate::Result<Tensor>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::InvalidConfig { op: actual, .. }) if actual == op
    ));
}

fn expect_rank_mismatch(result: crate::Result<Tensor>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::RankMismatch { op: actual, .. }) if actual == op
    ));
}

fn expect_axis_oob(result: crate::Result<Tensor>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::AxisOutOfBounds { op: actual, .. }) if actual == op
    ));
}

fn expect_duplicate_axis(result: crate::Result<Tensor>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::DuplicateAxis { op: actual, .. }) if actual == op
    ));
}

#[test]
fn cpu_indexing_dispatch_covers_supported_dtypes() {
    let indices = Tensor::from_vec_col_major(vec![2], vec![0_i64, 2]);

    let f32_operand = Tensor::F32(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    assert_eq!(
        gather(&f32_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let c32_operand = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 1.0),
            Complex32::new(3.0, 2.0),
        ],
    ));
    assert_eq!(
        gather(&c32_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let c64_operand = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.0),
            Complex64::new(3.0, 2.0),
        ],
    ));
    assert_eq!(
        gather(&c64_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let i64_operand = Tensor::from_vec_col_major(vec![3], vec![1_i64, 2, 3]);
    assert!(matches!(
        gather(&i64_operand, &indices, &simple_gather_config()),
        Err(crate::Error::BackendFailure { op: "gather", .. })
    ));

    let scatter_indices = Tensor::from_vec_col_major(vec![2, 2], vec![0_i64, 1, 0, 1]);

    let f32_updates = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![5.0, 6.0]));
    assert_eq!(
        scatter(
            &Tensor::F32(TypedTensor::zeros(vec![2, 2])),
            &scatter_indices,
            &f32_updates,
            &diagonal_scatter_config(),
        )
        .unwrap()
        .shape(),
        &[2, 2]
    );

    let c32_updates = Tensor::C32(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex32::new(5.0, 1.0), Complex32::new(6.0, 2.0)],
    ));
    assert_eq!(
        scatter(
            &Tensor::C32(TypedTensor::zeros(vec![2, 2])),
            &scatter_indices,
            &c32_updates,
            &diagonal_scatter_config(),
        )
        .unwrap()
        .shape(),
        &[2, 2]
    );

    let c64_updates = Tensor::C64(TypedTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(5.0, 1.0), Complex64::new(6.0, 2.0)],
    ));
    assert_eq!(
        scatter(
            &Tensor::C64(TypedTensor::zeros(vec![2, 2])),
            &scatter_indices,
            &c64_updates,
            &diagonal_scatter_config(),
        )
        .unwrap()
        .shape(),
        &[2, 2]
    );

    assert!(matches!(
        scatter(
            &Tensor::from_vec_col_major(vec![2, 2], vec![0_i64; 4]),
            &scatter_indices,
            &Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]),
            &diagonal_scatter_config(),
        ),
        Err(crate::Error::BackendFailure { op: "scatter", .. })
    ));
    assert!(matches!(
        scatter(
            &Tensor::F32(TypedTensor::zeros(vec![2, 2])),
            &scatter_indices,
            &Tensor::F64(TypedTensor::zeros(vec![2])),
            &diagonal_scatter_config(),
        ),
        Err(crate::Error::DTypeMismatch { op: "scatter", .. })
    ));

    let slice_cfg = SliceConfig {
        starts: vec![0],
        limits: vec![2],
        strides: vec![1],
    };
    assert_eq!(
        crate::cpu::indexing::slice(&f32_operand, &slice_cfg)
            .unwrap()
            .shape(),
        &[2]
    );
    assert_eq!(
        crate::cpu::indexing::slice(&i64_operand, &slice_cfg)
            .unwrap()
            .shape(),
        &[2]
    );
    assert_eq!(
        crate::cpu::indexing::slice(&c32_operand, &slice_cfg)
            .unwrap()
            .shape(),
        &[2]
    );
    assert_eq!(
        crate::cpu::indexing::slice(&c64_operand, &slice_cfg)
            .unwrap()
            .shape(),
        &[2]
    );

    let starts = Tensor::from_vec_col_major(vec![1], vec![1_i64]);
    assert_eq!(
        dynamic_slice(&f32_operand, &starts, &[2]).unwrap().shape(),
        &[2]
    );
    assert_eq!(
        dynamic_slice(&c32_operand, &starts, &[2]).unwrap().shape(),
        &[2]
    );
    assert_eq!(
        dynamic_slice(&c64_operand, &starts, &[2]).unwrap().shape(),
        &[2]
    );
    assert!(matches!(
        dynamic_slice(&i64_operand, &starts, &[2]),
        Err(crate::Error::BackendFailure {
            op: "dynamic_slice",
            ..
        })
    ));

    let pad_cfg = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    };
    assert_eq!(pad(&f32_operand, &pad_cfg).unwrap().shape(), &[5]);
    assert_eq!(pad(&i64_operand, &pad_cfg).unwrap().shape(), &[5]);
    assert_eq!(pad(&c32_operand, &pad_cfg).unwrap().shape(), &[5]);
    assert_eq!(pad(&c64_operand, &pad_cfg).unwrap().shape(), &[5]);

    let mut backend = CpuBackend::new();
    assert_eq!(
        backend
            .concatenate(&[&f32_operand, &f32_operand], 0)
            .unwrap()
            .shape(),
        &[6]
    );
    assert_eq!(
        backend
            .concatenate(&[&i64_operand, &i64_operand], 0)
            .unwrap()
            .shape(),
        &[6]
    );
    assert_eq!(
        backend
            .concatenate(&[&c32_operand, &c32_operand], 0)
            .unwrap()
            .shape(),
        &[6]
    );
    assert_eq!(
        backend
            .concatenate(&[&c64_operand, &c64_operand], 0)
            .unwrap()
            .shape(),
        &[6]
    );

    assert_eq!(backend.reverse(&f32_operand, &[0]).unwrap().shape(), &[3]);
    assert_eq!(backend.reverse(&c32_operand, &[0]).unwrap().shape(), &[3]);
    assert_eq!(backend.reverse(&c64_operand, &[0]).unwrap().shape(), &[3]);
}

#[test]
fn cpu_indexing_validation_covers_error_branches() {
    let mut backend = CpuBackend::new();
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));

    expect_rank_mismatch(
        backend.slice(
            &input,
            &SliceConfig {
                starts: vec![0],
                limits: vec![2, 2],
                strides: vec![1],
            },
        ),
        "slice",
    );
    expect_rank_mismatch(
        backend.slice(
            &input,
            &SliceConfig {
                starts: vec![0],
                limits: vec![2],
                strides: vec![1, 1],
            },
        ),
        "slice",
    );
    expect_invalid_config(
        backend.slice(
            &input,
            &SliceConfig {
                starts: vec![2],
                limits: vec![1],
                strides: vec![1],
            },
        ),
        "slice",
    );
    expect_axis_oob(
        backend.slice(
            &input,
            &SliceConfig {
                starts: vec![0],
                limits: vec![3],
                strides: vec![1],
            },
        ),
        "slice",
    );
    expect_invalid_config(
        backend.slice(
            &input,
            &SliceConfig {
                starts: vec![0],
                limits: vec![2],
                strides: vec![0],
            },
        ),
        "slice",
    );

    let matrix = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    expect_rank_mismatch(
        backend.dynamic_slice(
            &matrix,
            &Tensor::from_vec_col_major(vec![2], vec![0_i64, 0]),
            &[1],
        ),
        "dynamic_slice",
    );
    expect_invalid_config(
        backend.dynamic_slice(
            &matrix,
            &Tensor::from_vec_col_major(vec![1, 1], vec![0_i64]),
            &[1, 1],
        ),
        "dynamic_slice",
    );
    expect_invalid_config(
        backend.dynamic_slice(
            &matrix,
            &Tensor::from_vec_col_major(vec![1], vec![0_i64]),
            &[1, 1],
        ),
        "dynamic_slice",
    );

    expect_rank_mismatch(
        backend.pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![0, 0],
                edge_padding_high: vec![0],
                interior_padding: vec![0],
            },
        ),
        "pad",
    );
    expect_rank_mismatch(
        backend.pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![0],
                edge_padding_high: vec![0],
                interior_padding: vec![0, 0],
            },
        ),
        "pad",
    );
    expect_invalid_config(
        backend.pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![0],
                edge_padding_high: vec![0],
                interior_padding: vec![-1],
            },
        ),
        "pad",
    );
    expect_invalid_config(
        backend.pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![-3],
                edge_padding_high: vec![0],
                interior_padding: vec![0],
            },
        ),
        "pad",
    );

    let operand_2d = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![1.0, 2.0, 3.0, 4.0],
    ));
    let idx = Tensor::from_vec_col_major(vec![1, 1], vec![0_i64]);
    let idx2 = Tensor::from_vec_col_major(vec![1, 2], vec![0_i64, 0]);

    expect_invalid_config(
        gather(
            &operand_2d,
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![1, 1], vec![0.5])),
            &valid_gather_2d_config(),
        ),
        "index_tensor",
    );
    expect_invalid_config(
        gather(
            &operand_2d,
            &Tensor::F32(TypedTensor::from_vec_col_major(
                vec![1, 1],
                vec![16_777_218.0],
            )),
            &valid_gather_2d_config(),
        ),
        "index_tensor",
    );
    expect_invalid_config(
        gather(
            &operand_2d,
            &Tensor::F64(TypedTensor::from_vec_col_major(
                vec![1, 1],
                vec![9_007_199_254_740_994.0],
            )),
            &valid_gather_2d_config(),
        ),
        "index_tensor",
    );

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.slice_sizes = vec![1];
    expect_rank_mismatch(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.collapsed_slice_dims = vec![2];
    expect_axis_oob(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.collapsed_slice_dims = vec![0, 0];
    expect_duplicate_axis(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.start_index_map = vec![0, 1];
    expect_invalid_config(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.start_index_map = vec![2];
    expect_axis_oob(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.start_index_map = vec![0, 0];
    expect_duplicate_axis(gather(&operand_2d, &idx2, &gather_cfg), "gather");

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.offset_dims = vec![];
    expect_invalid_config(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let mut gather_cfg = valid_gather_2d_config();
    gather_cfg.offset_dims = vec![2];
    expect_axis_oob(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let gather_cfg = GatherConfig {
        offset_dims: vec![0, 0],
        collapsed_slice_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, 1],
    };
    expect_duplicate_axis(gather(&operand_2d, &idx, &gather_cfg), "gather");

    let updates = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![5.0]));
    let mut scatter_cfg = diagonal_scatter_config();
    scatter_cfg.inserted_window_dims = vec![2];
    expect_axis_oob(
        scatter(&operand_2d, &idx2, &updates, &scatter_cfg),
        "scatter",
    );

    let mut scatter_cfg = diagonal_scatter_config();
    scatter_cfg.inserted_window_dims = vec![0, 0];
    expect_duplicate_axis(
        scatter(&operand_2d, &idx2, &updates, &scatter_cfg),
        "scatter",
    );

    let mut scatter_cfg = diagonal_scatter_config();
    scatter_cfg.scatter_dims_to_operand_dims = vec![0];
    expect_invalid_config(
        scatter(&operand_2d, &idx2, &updates, &scatter_cfg),
        "scatter",
    );

    let mut scatter_cfg = diagonal_scatter_config();
    scatter_cfg.scatter_dims_to_operand_dims = vec![0, 2];
    expect_axis_oob(
        scatter(&operand_2d, &idx2, &updates, &scatter_cfg),
        "scatter",
    );

    let mut scatter_cfg = diagonal_scatter_config();
    scatter_cfg.scatter_dims_to_operand_dims = vec![0, 0];
    expect_duplicate_axis(
        scatter(&operand_2d, &idx2, &updates, &scatter_cfg),
        "scatter",
    );

    let scatter_cfg = ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    };
    expect_invalid_config(
        scatter(&operand_2d, &idx2, &updates, &scatter_cfg),
        "scatter",
    );

    let scatter_cfg = ScatterConfig {
        update_window_dims: vec![0, 1],
        inserted_window_dims: vec![],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    };
    expect_invalid_config(
        scatter(&operand_2d, &idx2, &updates, &scatter_cfg),
        "scatter",
    );

    let scatter_cfg = diagonal_scatter_config();
    let bad_batch_updates = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![5.0]));
    expect_invalid_config(
        scatter(&operand_2d, &idx2, &bad_batch_updates, &scatter_cfg),
        "scatter",
    );

    let scatter_cfg = ScatterConfig {
        update_window_dims: vec![2],
        inserted_window_dims: vec![0],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    };
    let updates_2d = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![5.0]));
    expect_axis_oob(
        scatter(&operand_2d, &idx2, &updates_2d, &scatter_cfg),
        "scatter",
    );

    let scatter_cfg = ScatterConfig {
        update_window_dims: vec![0, 0],
        inserted_window_dims: vec![],
        scatter_dims_to_operand_dims: vec![0, 1],
        index_vector_dim: 1,
    };
    let updates_3d = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1, 1], vec![5.0]));
    expect_duplicate_axis(
        scatter(&operand_2d, &idx2, &updates_3d, &scatter_cfg),
        "scatter",
    );

    let scatter_cfg = diagonal_scatter_config();
    let mismatched_updates = Tensor::F64(TypedTensor::from_vec_col_major(
        vec![3],
        vec![1.0, 2.0, 3.0],
    ));
    expect_invalid_config(
        scatter(&operand_2d, &idx2, &mismatched_updates, &scatter_cfg),
        "scatter",
    );
}

#[test]
fn cpu_exec_session_covers_complex_linalg_and_error_dispatch() {
    let mut backend = CpuBackend::new();
    backend.with_exec_session(|exec| {
        let eye = Tensor::C64(TypedTensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ));
        let rhs = Tensor::C64(TypedTensor::from_vec_col_major(
            vec![2, 1],
            vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
        ));

        assert_eq!(exec.cholesky(&eye).unwrap().shape(), &[2, 2]);
        assert_eq!(exec.lu(&eye).unwrap().len(), 4);
        assert_eq!(exec.full_piv_lu(&eye).unwrap().len(), 5);
        assert_eq!(
            exec.full_piv_lu_solve(&eye, &rhs, false).unwrap().shape(),
            &[2, 1]
        );
        assert_eq!(
            exec.triangular_solve(&eye, &rhs, true, true, false, false)
                .unwrap()
                .shape(),
            &[2, 1]
        );
        assert_eq!(exec.svd(&eye).unwrap().len(), 3);
        assert_eq!(exec.qr(&eye).unwrap().len(), 2);
        assert_eq!(exec.eigh(&eye).unwrap().len(), 2);
        assert_eq!(exec.eig(&eye).unwrap().len(), 2);

        let f32_vec = Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
        let i64_vec = Tensor::I64(TypedTensor::from_vec_col_major(vec![2], vec![1_i64, 2]));
        assert!(matches!(
            exec.eig(&i64_vec),
            Err(crate::Error::BackendFailure { op: "eig", .. })
        ));
        assert!(matches!(
            exec.triangular_solve(&i64_vec, &i64_vec, true, true, false, false),
            Err(crate::Error::BackendFailure {
                op: "triangular_solve",
                ..
            })
        ));
        assert!(matches!(
            exec.full_piv_lu_solve(&i64_vec, &i64_vec, false),
            Err(crate::Error::BackendFailure {
                op: "full_piv_lu_solve",
                ..
            })
        ));

        let f64_vec = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]));
        assert!(matches!(
            exec.triangular_solve(&f64_vec, &f32_vec, true, true, false, false),
            Err(crate::Error::DTypeMismatch {
                op: "triangular_solve",
                ..
            })
        ));
        assert!(matches!(
            exec.full_piv_lu_solve(&f64_vec, &f32_vec, false),
            Err(crate::Error::DTypeMismatch {
                op: "full_piv_lu_solve",
                ..
            })
        ));

        let dot_cfg = DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        assert!(matches!(
            exec.dot_general(&f64_vec, &f32_vec, &dot_cfg),
            Err(crate::Error::DTypeMismatch {
                op: "dot_general",
                ..
            })
        ));

        exec.reclaim_buffer(Tensor::F32(TypedTensor::zeros(vec![1])));
        exec.reclaim_buffer(Tensor::F64(TypedTensor::zeros(vec![1])));
        exec.reclaim_buffer(Tensor::C32(TypedTensor::zeros(vec![1])));
        exec.reclaim_buffer(Tensor::C64(TypedTensor::zeros(vec![1])));
    });
}
