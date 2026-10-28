//! Tests for non-trivial logic in `exec_op_on_tensors`.
//!
//! Trivial forwarding branches (Add, Mul, Exp, etc.) are exercised
//! via EagerTensor AD tests. These tests target branches with real logic:
//! edge-case handling, byte parsing, multi-output unpacking, etc.

use super::{eager_input_promotion_plan, exec_op_on_tensors};
use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;
use tenferro_ops::dim_expr::DimExpr;
use tenferro_ops::ext_op::ExtensionOp;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_ops::SymDim;
use tenferro_tensor::{
    CompareDir, DType, DotGeneralConfig, ErrorKind, ScatterConfig, Tensor, TypedTensor,
    ValidationKind,
};

fn f64t(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(shape, data).unwrap())
}

fn scalar(v: f64) -> Tensor {
    f64t(vec![], vec![v])
}

fn i64_scalar(v: i64) -> Tensor {
    Tensor::from_typed::<i64>(TypedTensor::from_vec_col_major(vec![], vec![v]).unwrap())
}

fn planned_input_dtypes(op: &StdTensorOp, input_dtypes: &[DType]) -> Vec<DType> {
    let plan = eager_input_promotion_plan(op, input_dtypes.len(), |index| input_dtypes[index]);
    input_dtypes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, dtype)| plan.target_dtype(index, dtype))
        .collect()
}

#[test]
fn eager_read_dispatch_only_materializes_when_the_operation_requires_it() {
    use super::exec_standard_op_on_tensor_reads_in_session as execute;
    use crate::eager_backend::RecordingEngine;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tenferro_tensor::{TensorRead, TensorView};

    let copies = Arc::new(AtomicUsize::new(0));
    let mut backend = RecordingEngine::new(Arc::clone(&copies));
    let input = f64t(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
    let TensorView::F64(view) = TensorRead::from_tensor(&input).tensor_view() else {
        panic!("f64 view")
    };
    let read = TensorRead::from_view(TensorView::F64(view.transpose_view([1, 0]).unwrap()));
    let result = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        execute(&StdTensorOp::Mul, &[read.clone(), read.clone()], session)
    }
    .unwrap();
    assert_eq!(data(&result[0]), vec![1., 9., 25., 4., 16., 36.]);
    assert_eq!(
        copies.load(Ordering::Relaxed),
        0,
        "same-dtype read-capable operation must borrow inputs"
    );
    assert_eq!(data(&input), vec![1., 2., 3., 4., 5., 6.]);

    let rhs = Tensor::from_vec_col_major(vec![3, 2], vec![2.0_f32; 6]).unwrap();
    let rhs = TensorRead::from_view(TensorRead::from_tensor(&rhs).tensor_view());
    let result = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        execute(&StdTensorOp::Mul, &[read.clone(), rhs], session)
    }
    .unwrap();
    assert_eq!(data(&result[0]), vec![2., 6., 10., 4., 8., 12.]);
    assert_eq!(
        copies.swap(0, Ordering::Relaxed),
        1,
        "only the promoted input needs conversion storage"
    );

    // A structural read entry consumes the borrowed view, so the eager caller
    // hands it over instead of staging a compact copy first (#2223).
    let result = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        execute(
            &StdTensorOp::Tril { k: 0 },
            std::slice::from_ref(&read),
            session,
        )
    }
    .unwrap();
    assert_eq!(data(&result[0]), vec![1., 3., 5., 0., 4., 6.]);
    assert_eq!(
        copies.load(Ordering::Relaxed),
        0,
        "tril reads the borrowed operand instead of staging it"
    );

    // A value-changing operation still needs owned compact storage: the
    // dispatch stages the view before the owned `cast` entry, and that copy
    // must stay visible to this counter.
    let result = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        execute(
            &StdTensorOp::Convert {
                from: DType::F64,
                to: DType::F32,
            },
            std::slice::from_ref(&read),
            session,
        )
    }
    .unwrap();
    assert_eq!(result[0].dtype(), DType::F32);
    assert_eq!(
        copies.swap(0, Ordering::Relaxed),
        1,
        "a conversion still stages its borrowed operand"
    );

    // Default read hooks reject views; never conceal unsupported capabilities
    // with an implicit eager copy. Production CPU sessions override exp_read.
    let error = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        execute(&StdTensorOp::Exp, &[read], session)
    }
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
    assert_eq!(copies.load(Ordering::Relaxed), 0);

    let empty = f64t(vec![0, 3], vec![]);
    let read = TensorRead::from_view(TensorRead::from_tensor(&empty).tensor_view());
    let result = {
        let session: &mut dyn tenferro_tensor::Engine = &mut backend;
        execute(&StdTensorOp::Mul, &[read.clone(), read], session)
    }
    .unwrap();
    assert_eq!(result[0].shape(), &[0, 3]);
    assert!(data(&result[0]).is_empty());
    assert_eq!(copies.load(Ordering::Relaxed), 0);
}

#[test]
fn eager_input_promotion_plan_covers_all_promoted_families() {
    let binary_ops = [
        StdTensorOp::Add,
        StdTensorOp::Sub,
        StdTensorOp::Mul,
        StdTensorOp::Div,
        StdTensorOp::Rem,
        StdTensorOp::Pow,
        StdTensorOp::Maximum,
        StdTensorOp::Minimum,
        StdTensorOp::Compare(CompareDir::Eq),
        StdTensorOp::DotGeneral {
            config: DotGeneralConfig {
                lhs_contracting_dims: [].as_slice().into(),
                rhs_contracting_dims: [].as_slice().into(),
                lhs_batch_dims: [].as_slice().into(),
                rhs_batch_dims: [].as_slice().into(),
            },
        },
    ];
    for op in binary_ops {
        assert_eq!(
            planned_input_dtypes(&op, &[DType::F64, DType::C32]),
            vec![DType::C64, DType::C64],
            "binary promotion plan for {op:?}"
        );
    }

    let cases = [
        (
            StdTensorOp::Select,
            vec![DType::Bool, DType::F64, DType::C32],
            vec![DType::Bool, DType::C64, DType::C64],
        ),
        (
            StdTensorOp::Clamp,
            vec![DType::F32, DType::F64, DType::C32],
            vec![DType::C64, DType::C64, DType::C64],
        ),
        (
            StdTensorOp::Concatenate {
                axis: 0,
                input_count: 3,
            },
            vec![DType::F32, DType::F64, DType::C32],
            vec![DType::C64, DType::C64, DType::C64],
        ),
        (
            StdTensorOp::Scatter(ScatterConfig {
                update_window_dims: vec![],
                inserted_window_dims: vec![],
                scatter_dims_to_operand_dims: vec![],
                index_vector_dim: 0,
            }),
            vec![DType::F32, DType::I64, DType::C32],
            vec![DType::C32, DType::I64, DType::C32],
        ),
        (
            StdTensorOp::DynamicUpdateSlice,
            vec![DType::F32, DType::C64, DType::I64],
            vec![DType::C64, DType::C64, DType::I64],
        ),
        (StdTensorOp::Neg, vec![DType::C32], vec![DType::C32]),
    ];
    for (op, input_dtypes, expected) in cases {
        assert_eq!(
            planned_input_dtypes(&op, &input_dtypes),
            expected,
            "promotion plan for {op:?}"
        );
    }
}

fn data(t: &Tensor) -> Vec<f64> {
    t.as_typed::<f64>()
        .expect("expected F64")
        .host_data()
        .unwrap()
        .to_vec()
}

#[test]
fn dynamic_truncate_rounds_non_integer_size() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let size = scalar(2.7);
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0]);
}

#[test]
fn dynamic_truncate_clamps_negative_to_zero() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let size = scalar(-5.0);
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[0]);
}

#[test]
fn dynamic_truncate_clamps_oversize() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let size = scalar(100.0);
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0]);
}

#[test]
fn dynamic_truncate_rejects_nan_size() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let size = scalar(f64::NAN);
    let err = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap_err();

    assert!(
        err.to_string().contains("finite"),
        "expected finite-value error, got {err:?}"
    );
}

#[test]
fn dynamic_truncate_accepts_i64_scalar_size() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let size = i64_scalar(2);
    let result = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0]);
}

#[test]
fn dynamic_truncate_rejects_non_scalar_size() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![4], vec![1.0, 2.0, 3.0, 4.0]);
    let size = f64t(vec![2], vec![1.0, 2.0]);
    let err = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[&x, &size],
        &mut b,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("scalar"),
        "unexpected error: {err}"
    );
}

#[test]
fn pad_to_match_noop_when_already_larger() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let reference = f64t(vec![3], vec![0.0; 3]);
    let result = exec_op_on_tensors(
        &StdTensorOp::PadToMatch { axis: 0 },
        &[&x, &reference],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn pad_to_match_pads_with_zeros() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![2], vec![1.0, 2.0]);
    let reference = f64t(vec![5], vec![0.0; 5]);
    let result = exec_op_on_tensors(
        &StdTensorOp::PadToMatch { axis: 0 },
        &[&x, &reference],
        &mut b,
    )
    .unwrap();
    assert_eq!(data(&result[0]), vec![1.0, 2.0, 0.0, 0.0, 0.0]);
}

#[test]
fn dynamic_truncate_invalid_axis_returns_tensor_runtime_error() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![3], vec![1.0, 2.0, 3.0]);
    let size = scalar(2.0);
    let err = exec_op_on_tensors(
        &StdTensorOp::DynamicTruncate { axis: 1 },
        &[&x, &size],
        &mut b,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        tenferro_runtime::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            op: "DynamicTruncate",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: 1, rank: 1 },
        })
    ));
}

#[test]
fn pad_to_match_rejects_reference_axis_out_of_bounds_without_panicking() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![2], vec![1.0, 2.0]);
    let reference = scalar(0.0);
    let err = exec_op_on_tensors(
        &StdTensorOp::PadToMatch { axis: 0 },
        &[&x, &reference],
        &mut b,
    )
    .unwrap_err();

    assert!(matches!(
        err,
        tenferro_runtime::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            op: "PadToMatch",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: 0, rank: 0 },
        })
    ));
}

#[test]
fn eager_shape_expr_resolution_returns_error_instead_of_panicking() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![2], vec![1.0, 2.0]);
    let err = exec_op_on_tensors(
        &StdTensorOp::Reshape {
            to_shape: vec![DimExpr::InputDim {
                input_idx: 1,
                axis: 0,
            }],
        },
        &[&x],
        &mut b,
    )
    .unwrap_err();

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert_eq!(err.phase(), Some(tenferro_runtime::ErrorPhase::Execution));
    assert!(matches!(
        err,
        tenferro_runtime::Error::ShapeExpressionEvaluation { .. }
    ));
}

#[test]
fn constant_f64_parses_bytes() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let result = exec_op_on_tensors(
        &StdTensorOp::Constant {
            dtype: DType::F64,
            bytes: 3.125_f64.to_le_bytes().to_vec(),
        },
        &[],
        &mut b,
    )
    .unwrap();
    assert!((data(&result[0])[0] - 3.125).abs() < 1e-12);
}

#[test]
fn constant_f32_parses_bytes() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let result = exec_op_on_tensors(
        &StdTensorOp::Constant {
            dtype: DType::F32,
            bytes: 2.5_f32.to_le_bytes().to_vec(),
        },
        &[],
        &mut b,
    )
    .unwrap();
    assert_eq!(result[0].dtype(), DType::F32);
}

#[test]
fn reshape_resolves_dim_exprs() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let result = exec_op_on_tensors(
        &StdTensorOp::Reshape {
            to_shape: vec![DimExpr::Const(2), DimExpr::Const(3)],
        },
        &[&x],
        &mut b,
    )
    .unwrap();
    assert_eq!(result[0].shape(), &[2, 3]);
}

#[test]
fn shape_of_each_axis() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = f64t(vec![3, 7, 5], vec![0.0; 105]);
    for (axis, expected) in [(0, 3.0), (1, 7.0), (2, 5.0)] {
        let result = exec_op_on_tensors(&StdTensorOp::ShapeOf { axis }, &[&x], &mut b).unwrap();
        assert_eq!(data(&result[0]), vec![expected]);
    }
}

#[test]
fn generated_outputs_use_the_callers_borrowed_session() {
    use super::{
        exec_standard_op_on_tensor_reads_with_session, exec_standard_op_on_tensors_with_session,
    };
    use tenferro_tensor::TensorRead;

    let mut backend = tenferro_cpu::CpuEngine::serial();
    let input = f64t(vec![2, 3], vec![0.0; 6]);
    (|session: &mut dyn tenferro_tensor::Engine| {
        let constant = StdTensorOp::Constant {
            dtype: DType::F64,
            bytes: 3.125_f64.to_le_bytes().to_vec(),
        };
        let values = exec_standard_op_on_tensor_reads_with_session(&constant, &[], session)?;
        assert_eq!(data(&values[0]), vec![3.125]);
        let shape = StdTensorOp::ShapeOf { axis: 1 };
        let reads = [TensorRead::from_tensor(&input)];
        let values = exec_standard_op_on_tensor_reads_with_session(&shape, &reads, session)?;
        assert_eq!(data(&values[0]), vec![3.0]);
        let values = exec_standard_op_on_tensors_with_session(&shape, &[&input], session)?;
        assert_eq!(data(&values[0]), vec![3.0]);
        Ok::<_, tenferro_runtime::Error>(())
    })(&mut backend)
    .unwrap();
}

/// #2223: the eager dispatch hands a borrowed operand to the read entry.
///
/// Each case runs the same op twice through the eager dispatch: once with the
/// compact copies its operands stand for, once with the borrowed views — a
/// compact one and an offset/non-compact one. The values must agree, and the
/// caller must not canonicalize either operand: `RecordingEngine` counts every
/// `to_contiguous_read`, and the CPU engine reads a borrowed layout in place.
#[test]
fn eager_read_entries_consume_borrowed_views_without_staging() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::exec_standard_op_on_tensor_reads_in_session;
    use crate::eager_backend::RecordingEngine;
    use tenferro_cpu::CpuEngine;
    use tenferro_tensor::{
        PadConfig, ScatterConfig, SliceConfig, TensorRead, TensorStructural, TensorValue,
    };

    fn assert_outputs_match(
        actual: &[Tensor],
        expected: &[Tensor],
        op: &StdTensorOp,
        flavour: &str,
    ) {
        assert_eq!(actual.len(), expected.len(), "{op:?} {flavour}");
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.shape(), expected.shape(), "{op:?} {flavour}");
            assert_eq!(data(actual), data(expected), "{op:?} {flavour}");
        }
    }

    fn assert_views_match_materialized(op: &StdTensorOp, views: &[TensorRead<'_>]) {
        let copies = Arc::new(AtomicUsize::new(0));
        let mut backend = RecordingEngine::new(Arc::clone(&copies));

        let mut materializer = CpuEngine::serial();
        let owned: Vec<Tensor> = views
            .iter()
            .cloned()
            .map(|read| {
                materializer
                    .to_contiguous_read(read)
                    .expect("materialize a compact copy of the view")
            })
            .collect();
        let owned_reads: Vec<TensorRead<'_>> = owned.iter().map(TensorRead::from_tensor).collect();
        let expected = exec_standard_op_on_tensor_reads_in_session(op, &owned_reads, &mut backend)
            .unwrap_or_else(|error| panic!("{op:?} on owned copies: {error}"));
        assert_eq!(
            copies.swap(0, Ordering::Relaxed),
            0,
            "{op:?}: an owned operand must not be staged"
        );

        // The same values bound as a compact borrowed read, not as an owned
        // `TensorRead::Tensor`. A value input that the arm already passes owned
        // (the dynamic-truncate size, the pad-to-match reference) keeps that
        // flavour: it is not an operand view, and its value read may legitimately
        // materialize.
        let compact_reads: Vec<TensorRead<'_>> = views
            .iter()
            .zip(&owned)
            .map(|(original, compact)| match original {
                TensorRead::Tensor(_) => TensorRead::from_tensor(compact),
                TensorRead::View(_) => {
                    TensorRead::from_view(TensorRead::from_tensor(compact).tensor_view())
                }
            })
            .collect();
        let from_compact =
            exec_standard_op_on_tensor_reads_in_session(op, &compact_reads, &mut backend)
                .unwrap_or_else(|error| panic!("{op:?} on compact borrowed reads: {error}"));
        assert_eq!(
            copies.swap(0, Ordering::Relaxed),
            0,
            "{op:?}: a compact borrowed operand must not be staged"
        );
        assert_outputs_match(&from_compact, &expected, op, "on compact borrowed reads");

        let actual = exec_standard_op_on_tensor_reads_in_session(op, views, &mut backend)
            .unwrap_or_else(|error| panic!("{op:?} on borrowed views: {error}"));
        assert_eq!(
            copies.load(Ordering::Relaxed),
            0,
            "{op:?} staged a borrowed operand instead of reading it"
        );
        assert_outputs_match(&actual, &expected, op, "on the view");
    }

    // A `5 x 4` column-major source, so an element is `row + column * 5`.
    let base = || TensorValue::from_tensor(f64t(vec![5, 4], (0..20).map(|i| i as f64).collect()));
    let compact = base();
    // Rows 1..4 keep the source's column stride of 5, so this is offset and non-compact.
    let offset = base()
        .slice_view(&SliceConfig {
            starts: vec![1, 0],
            limits: vec![4, 4],
            strides: vec![1, 1],
        })
        .expect("offset view");
    let columns = base()
        .slice_view(&SliceConfig {
            starts: vec![0, 0],
            limits: vec![4, 4],
            strides: vec![1, 1],
        })
        .expect("column view");
    let transposed = base().transpose_view([1, 0]).expect("transposed view");
    let update = base()
        .slice_view(&SliceConfig {
            starts: vec![1, 1],
            limits: vec![3, 3],
            strides: vec![1, 1],
        })
        .expect("update view");
    assert_eq!(compact.tensor_read().shape(), &[5, 4]);
    assert_eq!(offset.shape(), &[3, 4]);
    assert_eq!(offset.strides(), &[1, 5]);

    assert_views_match_materialized(&StdTensorOp::Tril { k: -1 }, &[offset.tensor_read()]);
    assert_views_match_materialized(&StdTensorOp::Triu { k: 1 }, &[columns.tensor_read()]);
    assert_views_match_materialized(
        &StdTensorOp::ExtractDiag {
            axis_a: 0,
            axis_b: 1,
        },
        &[offset.tensor_read()],
    );
    assert_views_match_materialized(
        &StdTensorOp::EmbedDiag {
            axis_a: 0,
            axis_b: 1,
        },
        &[offset.tensor_read()],
    );
    assert_views_match_materialized(
        &StdTensorOp::Slice(SliceConfig {
            starts: vec![0, 1],
            limits: vec![3, 4],
            strides: vec![1, 2],
        }),
        &[offset.tensor_read()],
    );
    assert_views_match_materialized(
        &StdTensorOp::Pad(PadConfig {
            edge_padding_low: vec![1, 0],
            edge_padding_high: vec![0, 1],
            interior_padding: vec![0, 1],
        }),
        &[transposed.tensor_read()],
    );
    assert_views_match_materialized(
        &StdTensorOp::Reverse { axes: vec![0] },
        &[offset.tensor_read()],
    );
    assert_views_match_materialized(
        &StdTensorOp::Concatenate {
            axis: 0,
            input_count: 2,
        },
        &[offset.tensor_read(), columns.tensor_read()],
    );

    // The dynamic-truncate size is a value the dispatch reads, not an operand.
    let truncated_size = scalar(2.0);
    assert_views_match_materialized(
        &StdTensorOp::DynamicTruncate { axis: 0 },
        &[
            offset.tensor_read(),
            TensorRead::from_tensor(&truncated_size),
        ],
    );
    // Pad-to-match pads against a wider reference, so the pad entry runs.
    let wider = f64t(vec![5, 4], vec![0.0; 20]);
    assert_views_match_materialized(
        &StdTensorOp::PadToMatch { axis: 0 },
        &[offset.tensor_read(), TensorRead::from_tensor(&wider)],
    );

    let starts = Tensor::from_typed::<i64>(
        TypedTensor::from_vec_col_major(vec![2], vec![1, 1]).expect("starts tensor"),
    );
    assert_views_match_materialized(
        &StdTensorOp::DynamicSlice {
            slice_sizes: vec![2, 2],
        },
        &[offset.tensor_read(), TensorRead::from_tensor(&starts)],
    );
    assert_views_match_materialized(
        &StdTensorOp::DynamicUpdateSlice,
        &[
            offset.tensor_read(),
            update.tensor_read(),
            TensorRead::from_tensor(&starts),
        ],
    );

    let scatter_indices = Tensor::from_typed::<i64>(
        TypedTensor::from_vec_col_major(vec![4, 1], vec![0, 1, 1, 2]).expect("scatter indices"),
    );
    let scatter_updates = f64t(vec![4], vec![100.0, 200.0, 300.0, 400.0]);
    assert_views_match_materialized(
        &StdTensorOp::Scatter(ScatterConfig {
            update_window_dims: vec![],
            inserted_window_dims: vec![0, 1],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        }),
        &[
            offset.tensor_read(),
            TensorRead::from_tensor(&scatter_indices),
            TensorRead::from_tensor(&scatter_updates),
        ],
    );

    // A compact borrowed read is a view too; it must reach the hook unchanged.
    assert_views_match_materialized(
        &StdTensorOp::Tril { k: 0 },
        &[TensorRead::from_view(compact.tensor_view())],
    );
}

/// #2223: a dtype mismatch still converts, but the operand that already has
/// the promoted dtype stays a borrowed read, and the typed errors are the
/// same as the owned entries report.
#[test]
fn eager_promoting_read_entries_convert_only_the_mismatched_operand() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::exec_standard_op_on_tensor_reads_in_session;
    use crate::eager_backend::RecordingEngine;
    use tenferro_cpu::CpuEngine;
    use tenferro_tensor::{Error as TensorError, TensorRead, ValidationError};

    fn view(tensor: &Tensor) -> TensorRead<'_> {
        TensorRead::from_view(TensorRead::from_tensor(tensor).tensor_view())
    }

    fn expected(op: &StdTensorOp, inputs: &[&Tensor]) -> Vec<Tensor> {
        let reads: Vec<TensorRead<'_>> =
            inputs.iter().map(|t| TensorRead::from_tensor(t)).collect();
        exec_standard_op_on_tensor_reads_in_session(op, &reads, &mut CpuEngine::serial())
            .expect("the promoted owned baseline")
    }

    fn promoted(op: &StdTensorOp, inputs: &[TensorRead<'_>]) -> (Vec<Tensor>, usize) {
        let copies = Arc::new(AtomicUsize::new(0));
        let mut backend = RecordingEngine::new(Arc::clone(&copies));
        let outputs = exec_standard_op_on_tensor_reads_in_session(op, inputs, &mut backend)
            .expect("the borrowed mixed-dtype run");
        (outputs, copies.load(Ordering::Relaxed))
    }

    let concat = StdTensorOp::Concatenate {
        axis: 0,
        input_count: 2,
    };
    let left = f64t(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let right64 = f64t(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let right32 = Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f32, 6.0, 7.0, 8.0])
        .expect("f32 right operand");
    let want = expected(&concat, &[&left, &right64]);
    let (got, copies) = promoted(&concat, &[view(&left), view(&right32)]);
    assert_eq!(got[0].dtype(), DType::F64);
    assert_eq!(
        copies, 1,
        "only the F32 concatenate operand needs conversion storage"
    );
    assert_eq!(data(&got[0]), data(&want[0]));

    let scatter = StdTensorOp::Scatter(ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    });
    let operand = f64t(vec![3, 2], vec![0.0; 6]);
    let indices = Tensor::from_typed::<i64>(
        TypedTensor::from_vec_col_major(vec![4, 1], vec![0, 1, 1, 2]).expect("scatter indices"),
    );
    let updates64 = f64t(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
    let updates32 =
        Tensor::from_vec_col_major(vec![4], vec![10.0_f32, 20.0, 30.0, 40.0]).expect("f32 updates");
    let want = expected(&scatter, &[&operand, &indices, &updates64]);
    let (got, copies) = promoted(
        &scatter,
        &[view(&operand), view(&indices), view(&updates32)],
    );
    assert_eq!(got[0].dtype(), DType::F64);
    assert_eq!(
        copies, 1,
        "only the F32 scatter updates need conversion storage"
    );
    assert_eq!(data(&got[0]), data(&want[0]));

    let operand = f64t(vec![3, 2], vec![0.0; 6]);
    let update64 = f64t(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let update32 =
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).expect("f32 update");
    let starts = Tensor::from_typed::<i64>(
        TypedTensor::from_vec_col_major(vec![2], vec![1, 0]).expect("starts"),
    );
    let want = expected(
        &StdTensorOp::DynamicUpdateSlice,
        &[&operand, &update64, &starts],
    );
    let (got, copies) = promoted(
        &StdTensorOp::DynamicUpdateSlice,
        &[view(&operand), view(&update32), view(&starts)],
    );
    assert_eq!(got[0].dtype(), DType::F64);
    assert_eq!(
        copies, 1,
        "only the F32 dynamic-update update needs conversion storage"
    );
    assert_eq!(data(&got[0]), data(&want[0]));

    // One representative invalid configuration per arm: the read entries keep
    // the typed validation the owned entries reported.
    let err = {
        let mut backend = RecordingEngine::new(Arc::new(AtomicUsize::new(0)));
        exec_standard_op_on_tensor_reads_in_session(
            &StdTensorOp::Concatenate {
                axis: 2,
                input_count: 2,
            },
            &[view(&left), view(&right64)],
            &mut backend,
        )
        .expect_err("axis 2 is out of bounds for rank 2")
    };
    assert!(
        matches!(
            &err,
            crate::Error::TensorRuntime(TensorError::Validation {
                op: "concatenate",
                source: ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
                ..
            })
        ),
        "unexpected concatenate error: {err}"
    );

    let bad_scatter = StdTensorOp::Scatter(ScatterConfig {
        update_window_dims: vec![],
        inserted_window_dims: vec![0, 1],
        scatter_dims_to_operand_dims: vec![2],
        index_vector_dim: 1,
    });
    let err = {
        let mut backend = RecordingEngine::new(Arc::new(AtomicUsize::new(0)));
        exec_standard_op_on_tensor_reads_in_session(
            &bad_scatter,
            &[view(&operand), view(&indices), view(&updates64)],
            &mut backend,
        )
        .expect_err("scatter dimension 2 is out of bounds for rank 2")
    };
    assert!(
        matches!(
            &err,
            crate::Error::TensorRuntime(TensorError::Validation {
                op: "scatter",
                source: ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
                ..
            })
        ),
        "unexpected scatter error: {err}"
    );

    let short_starts = Tensor::from_typed::<i64>(
        TypedTensor::from_vec_col_major(vec![1], vec![0]).expect("short starts"),
    );
    let err = {
        let mut backend = RecordingEngine::new(Arc::new(AtomicUsize::new(0)));
        exec_standard_op_on_tensor_reads_in_session(
            &StdTensorOp::DynamicUpdateSlice,
            &[view(&operand), view(&update64), view(&short_starts)],
            &mut backend,
        )
        .expect_err("starts must match the operand rank")
    };
    assert!(
        matches!(
            &err,
            crate::Error::TensorRuntime(TensorError::Validation {
                op: "dynamic_update_slice",
                source: ValidationError::InvalidArgument { argument: "configuration", message },
                ..
            }) if message.contains("starts length 1")
        ),
        "unexpected dynamic_update_slice error: {err}"
    );
}

#[test]
fn traced_host_ops_do_not_stage_a_borrowed_operand_or_shape_reference() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::eager_backend::RecordingEngine;
    use tenferro_runtime::{DType, ExecutionEngine, GraphCompiler, Runtime, TracedTensor};
    use tenferro_tensor::{SliceConfig, TensorValue};

    fn cpu_runtime() -> Runtime {
        let backend = tenferro_cpu::CpuEngine::serial();
        let mut builder = Runtime::builder();
        builder
            .register_engine(
                tenferro_cpu::runtime_engine_registration(&backend)
                    .expect("CPU engine registration"),
            )
            .expect("register the CPU engine");
        builder.build().expect("CPU runtime")
    }

    let copies = Arc::new(AtomicUsize::new(0));
    let placeholder = TracedTensor::input_symbolic_shape(DType::F64, 2).expect("placeholder");
    let source = f64t(vec![5, 4], (0..20).map(|index| index as f64).collect());
    let view = TensorValue::from_tensor(source)
        .slice_view(&SliceConfig {
            starts: vec![1, 0],
            limits: vec![4, 4],
            strides: vec![1, 1],
        })
        .expect("offset view");

    let size = TracedTensor::from_vec_col_major(vec![], vec![2.0_f64]).expect("size scalar");
    let truncated = placeholder
        .dynamic_truncate(&size, 0)
        .expect("traced dynamic_truncate");
    let mut engine = RecordingEngine::new(Arc::clone(&copies));
    let shape = view.shape().to_vec();
    let mut compiler = GraphCompiler::new();
    let program = compiler
        .compile_with_input_specs(&truncated, &[(&placeholder, DType::F64, shape.as_slice())])
        .expect("compile the traced dynamic_truncate");
    let runtime = cpu_runtime();
    let target = engine.target();
    let reads = [view.tensor_read()];
    let prepared = runtime
        .prepare(&program, &reads, std::slice::from_ref(&target))
        .expect("prepare the traced dynamic_truncate");
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &reads)
        .expect("run the traced dynamic_truncate");
    assert_eq!(outputs[0].read().shape(), &[2, 4]);
    assert_eq!(
        copies.load(Ordering::Relaxed),
        0,
        "dynamic_truncate staged its borrowed operand"
    );

    // The reference is a borrowed view too, so the old staging of both inputs
    // would move the counter; the new path must read only its shape.
    let reference_placeholder =
        TracedTensor::input_symbolic_shape(DType::F64, 2).expect("reference placeholder");
    let wider = TensorValue::from_tensor(f64t(vec![5, 4], vec![0.0; 20]));
    let wider_shape = wider.shape().to_vec();
    let padded = placeholder
        .pad_to_match(&reference_placeholder, 0)
        .expect("traced pad_to_match");
    let program = compiler
        .compile_with_input_specs(
            &padded,
            &[
                (&placeholder, DType::F64, shape.as_slice()),
                (&reference_placeholder, DType::F64, wider_shape.as_slice()),
            ],
        )
        .expect("compile the traced pad_to_match");
    let padded_reads = [view.tensor_read(), wider.tensor_read()];
    let prepared = runtime
        .prepare(&program, &padded_reads, std::slice::from_ref(&target))
        .expect("prepare the traced pad_to_match");
    copies.store(0, Ordering::Relaxed);
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &padded_reads)
        .expect("run the traced pad_to_match");
    assert_eq!(outputs[0].read().shape(), &[5, 4]);
    assert_eq!(
        copies.load(Ordering::Relaxed),
        0,
        "pad_to_match staged its operand or its shape-only reference"
    );

    // The unpadded branch must still return a fresh owned tensor, so its one
    // output materialization proves the counter is live.
    let same = TensorValue::from_tensor(f64t(vec![3, 4], vec![0.0; 12]));
    let same_shape = same.shape().to_vec();
    let unpadded = placeholder
        .pad_to_match(&reference_placeholder, 0)
        .expect("traced unpadded pad_to_match");
    let program = compiler
        .compile_with_input_specs(
            &unpadded,
            &[
                (&placeholder, DType::F64, shape.as_slice()),
                (&reference_placeholder, DType::F64, same_shape.as_slice()),
            ],
        )
        .expect("compile the traced unpadded pad_to_match");
    let same_reads = [view.tensor_read(), same.tensor_read()];
    let prepared = runtime
        .prepare(&program, &same_reads, std::slice::from_ref(&target))
        .expect("prepare the traced unpadded pad_to_match");
    copies.store(0, Ordering::Relaxed);
    let outputs = runtime
        .run(&mut [&mut engine], &prepared, &same_reads)
        .expect("run the traced unpadded pad_to_match");
    assert_eq!(outputs[0].read().shape(), &[3, 4]);
    assert_eq!(
        copies.load(Ordering::Relaxed),
        1,
        "the unpadded branch materializes only its fresh output"
    );
}

#[test]
fn extension_op_requires_installed_extension_module_runtime() {
    let mut b = tenferro_cpu::CpuEngine::serial();
    let x = scalar(1.0);
    let err = exec_op_on_tensors(
        &StdTensorOp::Extension(Arc::new(TestExtension)),
        &[&x],
        &mut b,
    )
    .unwrap_err();

    let message = err.to_string();
    assert!(message.contains("missing extension module"), "{message}");
    assert!(
        message.contains("install an ExtensionModule on the Runtime"),
        "{message}"
    );
    assert!(
        message.contains("tenferro-tests.eager_exec.v1"),
        "{message}"
    );
}

#[derive(Clone, Debug)]
struct TestExtension;

impl ExtensionOp for TestExtension {
    fn family_id(&self) -> &'static str {
        "tenferro-tests.eager_exec.v1"
    }

    fn payload_hash(&self, hasher: &mut dyn Hasher) {
        hasher.write_u8(0);
    }

    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }

    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn input_count(&self) -> usize {
        1
    }

    fn output_count(&self) -> usize {
        1
    }

    fn infer_output_meta(
        &self,
        ctx: &mut tenferro_ops::ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<SymDim>)>> {
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}
