use num_complex::{Complex32, Complex64};

use crate::{dynamic_slice, gather, pad, scatter, CpuBackend};
use tenferro_tensor::{BackendSessionHost, TensorIndexing};
use tenferro_tensor::{DotGeneralConfig, GatherConfig, PadConfig, ScatterConfig, SliceConfig};
use tenferro_tensor::{Tensor, TypedTensor};

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
        Err(crate::Error::Validation { op: actual, .. }) if actual == op
    ));
}

fn expect_rank_mismatch(result: crate::Result<Tensor>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::Validation { op: actual, source: tenferro_tensor::ValidationError::RankMismatch { .. } }) if actual == op
    ));
}

fn expect_axis_oob(result: crate::Result<Tensor>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::Validation { op: actual, source: tenferro_tensor::ValidationError::AxisOutOfBounds { .. } }) if actual == op
    ));
}

fn expect_duplicate_axis(result: crate::Result<Tensor>, op: &'static str) {
    assert!(matches!(
        result,
        Err(crate::Error::Validation { op: actual, source: tenferro_tensor::ValidationError::DuplicateAxis { .. } }) if actual == op
    ));
}

#[test]
fn cpu_indexing_dispatch_covers_supported_dtypes() {
    let mut backend = CpuBackend::new();
    let indices = Tensor::from_vec_col_major(vec![2], vec![0_i64, 2]).unwrap();

    let f32_operand =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    assert_eq!(
        gather(&f32_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let c32_operand = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![3],
            vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(2.0, 1.0),
                Complex32::new(3.0, 2.0),
            ],
        )
        .unwrap(),
    );
    assert_eq!(
        gather(&c32_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let c64_operand = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 1.0),
                Complex64::new(3.0, 2.0),
            ],
        )
        .unwrap(),
    );
    assert_eq!(
        gather(&c64_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let i32_operand = Tensor::from_vec_col_major(vec![3], vec![1_i32, 2, 3]).unwrap();
    assert_eq!(
        gather(&i32_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let i64_operand = Tensor::from_vec_col_major(vec![3], vec![1_i64, 2, 3]).unwrap();
    assert_eq!(
        gather(&i64_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let bool_operand = Tensor::from_vec_col_major(vec![3], vec![true, false, true]).unwrap();
    assert_eq!(
        gather(&bool_operand, &indices, &simple_gather_config())
            .unwrap()
            .shape(),
        &[2]
    );

    let scatter_indices = Tensor::from_vec_col_major(vec![2, 2], vec![0_i64, 1, 0, 1]).unwrap();

    let f32_updates =
        Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![5.0, 6.0]).unwrap());
    assert_eq!(
        scatter(
            &Tensor::F32(TypedTensor::zeros(vec![2, 2]).unwrap()),
            &scatter_indices,
            &f32_updates,
            &diagonal_scatter_config(),
        )
        .unwrap()
        .shape(),
        &[2, 2]
    );

    let c32_updates = Tensor::C32(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex32::new(5.0, 1.0), Complex32::new(6.0, 2.0)],
        )
        .unwrap(),
    );
    assert_eq!(
        scatter(
            &Tensor::C32(TypedTensor::zeros(vec![2, 2]).unwrap()),
            &scatter_indices,
            &c32_updates,
            &diagonal_scatter_config(),
        )
        .unwrap()
        .shape(),
        &[2, 2]
    );

    let c64_updates = Tensor::C64(
        TypedTensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(5.0, 1.0), Complex64::new(6.0, 2.0)],
        )
        .unwrap(),
    );
    assert_eq!(
        scatter(
            &Tensor::C64(TypedTensor::zeros(vec![2, 2]).unwrap()),
            &scatter_indices,
            &c64_updates,
            &diagonal_scatter_config(),
        )
        .unwrap()
        .shape(),
        &[2, 2]
    );

    assert_eq!(
        scatter(
            &Tensor::from_vec_col_major(vec![2, 2], vec![0_i64; 4]).unwrap(),
            &scatter_indices,
            &Tensor::from_vec_col_major(vec![2], vec![1_i64, 2]).unwrap(),
            &diagonal_scatter_config(),
        )
        .unwrap()
        .shape(),
        &[2, 2]
    );
    assert!(matches!(
        scatter(
            &Tensor::from_vec_col_major(vec![2, 2], vec![false; 4]).unwrap(),
            &scatter_indices,
            &Tensor::from_vec_col_major(vec![2], vec![true, false]).unwrap(),
            &diagonal_scatter_config(),
        ),
        Err(crate::Error::Unsupported { op: "scatter", .. })
    ));
    assert!(matches!(
        scatter(
            &Tensor::F32(TypedTensor::zeros(vec![2, 2]).unwrap()),
            &scatter_indices,
            &Tensor::F64(TypedTensor::zeros(vec![2]).unwrap()),
            &diagonal_scatter_config(),
        ),
        Err(crate::Error::Validation {
            op: "scatter",
            source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
        })
    ));

    let slice_cfg = SliceConfig {
        starts: vec![0],
        limits: vec![2],
        strides: vec![1],
    };
    assert_eq!(
        backend.slice(&f32_operand, &slice_cfg).unwrap().shape(),
        &[2]
    );
    assert_eq!(
        backend.slice(&i64_operand, &slice_cfg).unwrap().shape(),
        &[2]
    );
    assert_eq!(
        backend.slice(&bool_operand, &slice_cfg).unwrap().shape(),
        &[2]
    );
    assert_eq!(
        backend.slice(&c32_operand, &slice_cfg).unwrap().shape(),
        &[2]
    );
    assert_eq!(
        backend.slice(&c64_operand, &slice_cfg).unwrap().shape(),
        &[2]
    );

    let starts = Tensor::from_vec_col_major(vec![1], vec![1_i64]).unwrap();
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
    assert_eq!(
        dynamic_slice(&i64_operand, &starts, &[2]).unwrap().shape(),
        &[2]
    );
    assert_eq!(
        dynamic_slice(&bool_operand, &starts, &[2]).unwrap().shape(),
        &[2]
    );

    let pad_cfg = PadConfig {
        edge_padding_low: vec![1],
        edge_padding_high: vec![1],
        interior_padding: vec![0],
    };
    assert_eq!(pad(&f32_operand, &pad_cfg).unwrap().shape(), &[5]);
    assert_eq!(pad(&i64_operand, &pad_cfg).unwrap().shape(), &[5]);
    assert_eq!(pad(&bool_operand, &pad_cfg).unwrap().shape(), &[5]);
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
    assert_eq!(backend.reverse(&i64_operand, &[0]).unwrap().shape(), &[3]);
    assert_eq!(backend.reverse(&bool_operand, &[0]).unwrap().shape(), &[3]);
    assert_eq!(backend.reverse(&c32_operand, &[0]).unwrap().shape(), &[3]);
    assert_eq!(backend.reverse(&c64_operand, &[0]).unwrap().shape(), &[3]);
}

#[test]
fn static_erased_indexing_preserves_bool_values_and_empty_shapes() {
    let mut backend = CpuBackend::with_threads(2).unwrap();
    let mut input = Tensor::from_vec_col_major(vec![4], vec![true, false, true, false]).unwrap();
    let sliced = backend
        .slice(
            &input,
            &SliceConfig {
                starts: vec![1],
                limits: vec![4],
                strides: vec![2],
            },
        )
        .unwrap();
    assert_eq!(sliced.as_slice::<bool>().unwrap(), &[false, false]);

    let reversed = backend.reverse(&input, &[0]).unwrap();
    assert_eq!(
        reversed.as_slice::<bool>().unwrap(),
        &[false, true, false, true]
    );
    let concatenated = backend.concatenate(&[&sliced, &reversed], 0).unwrap();
    assert_eq!(
        concatenated.as_slice::<bool>().unwrap(),
        &[false, false, false, true, false, true]
    );
    let Tensor::Bool(input) = &mut input else {
        panic!("test input must remain Bool");
    };
    input.host_data_mut().unwrap().fill(true);
    assert_eq!(
        sliced.as_slice::<bool>().unwrap(),
        &[false, false],
        "mutating the input after handoff must not change the copied output"
    );
    assert_eq!(
        reversed.as_slice::<bool>().unwrap(),
        &[false, true, false, true],
        "static replay outputs must own their destination storage"
    );

    let empty = Tensor::from_vec_col_major(vec![0], Vec::<bool>::new()).unwrap();
    let empty_slice = backend
        .slice(
            &empty,
            &SliceConfig {
                starts: vec![0],
                limits: vec![0],
                strides: vec![1],
            },
        )
        .unwrap();
    assert!(empty_slice.as_slice::<bool>().unwrap().is_empty());
    assert!(backend
        .reverse(&empty, &[0])
        .unwrap()
        .as_slice::<bool>()
        .unwrap()
        .is_empty());
    assert!(backend
        .concatenate(&[&empty, &empty], 0)
        .unwrap()
        .as_slice::<bool>()
        .unwrap()
        .is_empty());

    let padded = backend
        .pad(
            &empty,
            &PadConfig {
                edge_padding_low: vec![1],
                edge_padding_high: vec![2],
                interior_padding: vec![0],
            },
        )
        .unwrap();
    assert_eq!(padded.as_slice::<bool>().unwrap(), &[false; 3]);
}

#[test]
fn cpu_indexing_validation_covers_error_branches() {
    let mut backend = CpuBackend::new();
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());

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

    let matrix =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    expect_rank_mismatch(
        backend.dynamic_slice(
            &matrix,
            &Tensor::from_vec_col_major(vec![2], vec![0_i64, 0]).unwrap(),
            &[1],
        ),
        "dynamic_slice",
    );
    expect_invalid_config(
        backend.dynamic_slice(
            &matrix,
            &Tensor::from_vec_col_major(vec![1, 1], vec![0_i64]).unwrap(),
            &[1, 1],
        ),
        "dynamic_slice",
    );
    expect_invalid_config(
        backend.dynamic_slice(
            &matrix,
            &Tensor::from_vec_col_major(vec![1], vec![0_i64]).unwrap(),
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
                edge_padding_low: vec![0],
                edge_padding_high: vec![0],
                interior_padding: vec![i64::MAX],
            },
        ),
        "pad",
    );
    expect_invalid_config(
        backend.pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![i64::MAX],
                edge_padding_high: vec![1],
                interior_padding: vec![0],
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

    let operand_2d =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap());
    let idx = Tensor::from_vec_col_major(vec![1, 1], vec![0_i64]).unwrap();
    let idx2 = Tensor::from_vec_col_major(vec![1, 2], vec![0_i64, 0]).unwrap();

    expect_invalid_config(
        gather(
            &operand_2d,
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![1, 1], vec![0.5]).unwrap()),
            &valid_gather_2d_config(),
        ),
        "index_tensor",
    );
    expect_invalid_config(
        gather(
            &operand_2d,
            &Tensor::F32(TypedTensor::from_vec_col_major(vec![1, 1], vec![16_777_218.0]).unwrap()),
            &valid_gather_2d_config(),
        ),
        "index_tensor",
    );
    expect_invalid_config(
        gather(
            &operand_2d,
            &Tensor::F64(
                TypedTensor::from_vec_col_major(vec![1, 1], vec![9_007_199_254_740_994.0]).unwrap(),
            ),
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
    gather_cfg.slice_sizes = vec![2, 1];
    expect_invalid_config(gather(&operand_2d, &idx, &gather_cfg), "gather");

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

    let updates = Tensor::F64(TypedTensor::from_vec_col_major(vec![1], vec![5.0]).unwrap());
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
    let bad_batch_updates =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![5.0]).unwrap());
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
    let updates_2d = Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1], vec![5.0]).unwrap());
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
    let updates_3d =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![1, 1, 1], vec![5.0]).unwrap());
    expect_duplicate_axis(
        scatter(&operand_2d, &idx2, &updates_3d, &scatter_cfg),
        "scatter",
    );

    let scatter_cfg = diagonal_scatter_config();
    let mismatched_updates =
        Tensor::F64(TypedTensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap());
    expect_invalid_config(
        scatter(&operand_2d, &idx2, &mismatched_updates, &scatter_cfg),
        "scatter",
    );
}

#[test]
fn cpu_pad_supports_signed_edge_cropping() {
    let input = Tensor::F64(
        TypedTensor::from_vec_col_major(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
    );

    for (low, high, expected) in [
        (-2, 0, vec![3.0, 4.0, 5.0]),
        (0, -2, vec![1.0, 2.0, 3.0]),
        (-1, -1, vec![2.0, 3.0, 4.0]),
    ] {
        let output = pad(
            &input,
            &PadConfig {
                edge_padding_low: vec![low],
                edge_padding_high: vec![high],
                interior_padding: vec![0],
            },
        )
        .unwrap();
        assert_eq!(output.as_slice::<f64>().unwrap(), expected);
    }
}

#[test]
fn cpu_pad_skips_extreme_signed_positions_without_overflow() {
    let input = Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
    let output = pad(
        &input,
        &PadConfig {
            edge_padding_low: vec![i64::MAX],
            edge_padding_high: vec![i64::MIN],
            interior_padding: vec![0],
        },
    )
    .unwrap();

    assert_eq!(output.shape(), &[1]);
    assert_eq!(output.as_slice::<f64>().unwrap(), &[0.0]);
}

#[test]
fn cpu_pad_does_not_reject_signed_edges_before_checked_shape_validation() {
    let indexing_source = include_str!("../indexing.rs");
    assert!(!indexing_source.contains("config.edge_padding_low[axis] < 0"));
    assert!(!indexing_source.contains("config.edge_padding_high[axis] < 0"));
}

#[test]
fn cpu_exec_session_covers_dot_errors_and_reclaim_dispatch() {
    let mut backend = CpuBackend::new();
    backend.with_backend_session(|exec| {
        let f32_vec =
            Tensor::F32(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
        let f64_vec =
            Tensor::F64(TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap());
        let dot_cfg = DotGeneralConfig {
            lhs_contracting_dims: vec![0],
            rhs_contracting_dims: vec![0],
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
        };
        assert!(matches!(
            exec.dot_general(&f64_vec, &f32_vec, &dot_cfg),
            Err(crate::Error::Validation {
                op: "dot_general",
                source: tenferro_tensor::ValidationError::DTypeMismatch { .. },
            })
        ));

        exec.reclaim_buffer(Tensor::F32(TypedTensor::zeros(vec![1]).unwrap()));
        exec.reclaim_buffer(Tensor::F64(TypedTensor::zeros(vec![1]).unwrap()));
        exec.reclaim_buffer(Tensor::C32(TypedTensor::zeros(vec![1]).unwrap()));
        exec.reclaim_buffer(Tensor::C64(TypedTensor::zeros(vec![1]).unwrap()));
    });
}
