use std::sync::{Arc, OnceLock};
use tenferro_ad::{EagerRuntime, EagerTensor};

use num_complex::Complex64;
use tenferro_cpu::CpuBackend;
use tenferro_runtime::{
    DType, DotGeneralConfig, Error as RuntimeError, ErrorPhase, GatherConfig, PadConfig,
    SliceConfig, Tensor, TensorRead, TensorView,
};
use tenferro_tensor::{Error as TensorError, ErrorKind, ValidationError, ValidationKind};

#[path = "eager_tensor/context_and_promotion.rs"]
mod context_and_promotion;

const FD_H: f64 = 1.0e-6;
const TOL: f64 = 1.0e-5;
const FD_TOL: f64 = 1.0e-4;

fn assert_close_slice(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tol,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn assert_close_c64_slice(actual: &[Complex64], expected: &[Complex64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).norm() <= tol,
            "index {index}: expected {expected}, got {actual}"
        );
    }
}

fn f64_data(tensor: &Tensor) -> &[f64] {
    tensor.as_slice::<f64>().unwrap()
}

fn c64_data(tensor: &Tensor) -> &[Complex64] {
    tensor.as_slice::<Complex64>().unwrap()
}

fn assert_send_sync<T: Send + Sync>() {}

fn finite_diff_scalar(f: impl Fn(&[f64]) -> f64, x: &[f64], index: usize) -> f64 {
    let mut plus = x.to_vec();
    let mut minus = x.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(&plus) - f(&minus)) / (2.0 * FD_H)
}

fn finite_diff_lhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    index: usize,
) -> f64 {
    let mut plus = lhs.to_vec();
    let mut minus = lhs.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(&plus, rhs) - f(&minus, rhs)) / (2.0 * FD_H)
}

fn finite_diff_rhs(
    f: impl Fn(&[f64], &[f64]) -> f64,
    lhs: &[f64],
    rhs: &[f64],
    index: usize,
) -> f64 {
    let mut plus = rhs.to_vec();
    let mut minus = rhs.to_vec();
    plus[index] += FD_H;
    minus[index] -= FD_H;
    (f(lhs, &plus) - f(lhs, &minus)) / (2.0 * FD_H)
}

fn matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    }
}

fn eager_matmul_sum(lhs: &[f64], rhs: &[f64]) -> f64 {
    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], lhs.to_vec()).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3, 2], rhs.to_vec()).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let loss = a
        .dot_general(&b, matmul_config())
        .unwrap()
        .reduce_sum(&[0, 1])
        .unwrap();
    f64_data(loss.materialized().unwrap().as_ref())[0]
}

fn test_ctx() -> Arc<EagerRuntime> {
    static CTX: OnceLock<Arc<EagerRuntime>> = OnceLock::new();
    CTX.get_or_init(|| EagerRuntime::with_cpu_backend(CpuBackend::new()))
        .clone()
}

#[test]
fn eager_tensor_exposes_metadata_read_and_materialization_without_data_accessor() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    assert_eq!(x.shape(), &[2, 3]);
    assert_eq!(x.dtype(), DType::F64);

    let read = x.tensor_read();
    assert_eq!(read.shape(), &[2, 3]);
    assert_eq!(read.dtype(), DType::F64);

    let materialized = x.to_tensor().unwrap();
    assert_eq!(materialized.shape(), &[2, 3]);
    assert_eq!(f64_data(&materialized), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn untracked_eager_transpose_is_exposed_as_borrowed_view_until_materialized() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let transposed = x.transpose(&[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);

    match transposed.tensor_read() {
        TensorRead::View(view) => {
            assert_eq!(view.shape(), &[3, 2]);
        }
        TensorRead::Tensor(_) => panic!("untracked transpose should stay as a borrowed view"),
    }

    let materialized = transposed.to_tensor().unwrap();
    assert_eq!(materialized.shape(), &[3, 2]);
    assert_eq!(f64_data(&materialized), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn matrix_eager_input_uses_column_major_values() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x.add(&x).unwrap();

    assert_eq!(y.shape(), &[2, 3]);
    assert_eq!(
        f64_data(y.materialized().unwrap().as_ref()),
        &[2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
    );
}

#[test]
fn eager_slice_axis_and_builder_preserve_column_major_values() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 4],
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();

    let rows = x.slice_axis(0, 1..3).unwrap();
    assert_eq!(rows.shape(), &[2, 4]);
    assert_eq!(
        f64_data(rows.materialized().unwrap().as_ref()),
        &[2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]
    );

    let strided = x
        .slice_builder()
        .axis(0, 1..3)
        .axis_step(1, 0..4, 2)
        .apply()
        .unwrap();
    assert_eq!(strided.shape(), &[2, 2]);
    assert_eq!(
        f64_data(strided.materialized().unwrap().as_ref()),
        &[2.0, 3.0, 8.0, 9.0]
    );

    let mixed = x
        .slice_builder()
        .axis(0, 0..2)
        .take_axis(1, &[3, 1, 3])
        .apply()
        .unwrap();
    assert_eq!(mixed.shape(), &[2, 3]);
    assert_eq!(
        f64_data(mixed.materialized().unwrap().as_ref()),
        &[10.0, 11.0, 4.0, 5.0, 10.0, 11.0]
    );
}

#[test]
fn eager_concatenate_empty_reports_typed_validation_error() {
    let err = EagerTensor::concatenate(&[], 0).unwrap_err();

    assert!(matches!(
        err,
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            op: "concatenate",
            ..
        })
    ));
}

#[test]
fn eager_reductions_and_reverse_validate_axes_before_ad_recording() {
    let ctx = test_ctx();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64; 6]).unwrap(),
        ctx,
    )
    .unwrap();

    let out_of_bounds = x.reduce_sum(&[2]).unwrap_err();
    assert!(matches!(
        out_of_bounds,
        tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::Validation {
            op: "EagerTensor::reduce_sum",
            source: tenferro_tensor::ValidationError::AxisOutOfBounds { axis: 2, rank: 2 },
        })
    ));

    for err in [
        x.reduce_sum(&[0, 0]).unwrap_err(),
        x.reduce_prod(&[0, 0]).unwrap_err(),
        x.reduce_max(&[0, 0]).unwrap_err(),
        x.reduce_min(&[0, 0]).unwrap_err(),
        x.reverse(&[0, 0]).unwrap_err(),
    ] {
        assert!(
            matches!(
                err,
                tenferro_ad::Error::TensorRuntime(tenferro_tensor::Error::Validation {
                    op: _,
                    source: tenferro_tensor::ValidationError::DuplicateAxis {
                        axis: 0,
                        role: "axis",
                    },
                })
            ),
            "{err}"
        );
    }
}

#[test]
fn eager_backward_seed_for_integer_scalar_does_not_call_float_analytic_ops() {
    let ctx = test_ctx();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![], vec![3_i32]).unwrap(),
        ctx,
    )
    .unwrap();

    let cotangents = x.backward().unwrap();

    assert_eq!(cotangents.len(), 1);
    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.as_slice::<i32>().unwrap(), &[1]);
}

#[test]
fn untracked_eager_intermediate_can_later_feed_tracked_ad() {
    let ctx = test_ctx();
    let plain = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let scale = plain.add(&plain).unwrap();
    assert!(!scale.tracks_grad());

    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let loss = x.mul(&scale).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[2.0, 4.0, 6.0],
        TOL,
    );
    x.clear_grad().unwrap();
}

#[test]
fn eager_dot_general_with_conj_uses_untracked_fast_path() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -0.25),
                Complex64::new(-1.0, 0.75),
                Complex64::new(0.5, 1.5),
            ],
        )
        .unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![2, 2],
            vec![
                Complex64::new(0.25, -1.0),
                Complex64::new(3.0, 0.5),
                Complex64::new(-2.0, 0.25),
                Complex64::new(1.5, -0.75),
            ],
        )
        .unwrap(),
        ctx,
    )
    .unwrap();
    let config = matmul_config();

    let fused = lhs
        .dot_general_with_conj(&rhs, config.clone(), true, false)
        .unwrap();
    let explicit = lhs.conj().unwrap().dot_general(&rhs, config).unwrap();

    assert!(!fused.tracks_grad());
    assert_eq!(fused.shape(), explicit.shape());
    assert_close_c64_slice(
        c64_data(fused.materialized().unwrap().as_ref()),
        c64_data(explicit.materialized().unwrap().as_ref()),
        TOL,
    );
}

#[test]
fn eager_dot_general_with_conj_validates_config_before_untracked_backend_dispatch() {
    let ctx = test_ctx();
    let lhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let rhs = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![5.0_f64, 6.0, 7.0, 8.0]).unwrap(),
        ctx,
    )
    .unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![3],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };

    let err = lhs
        .dot_general_with_conj(&rhs, config, true, false)
        .unwrap_err();

    assert_eq!(
        err.kind(),
        ErrorKind::Validation(ValidationKind::AxisOutOfBounds)
    );
    assert_eq!(err.phase(), Some(ErrorPhase::Execution));
    assert!(matches!(
        err,
        RuntimeError::TensorRuntime(TensorError::Validation { .. })
    ));
}

#[test]
fn eager_scalar_scaling_matches_traced_dtype_semantics() {
    let ctx = test_ctx();
    let real = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let scaled = real.scale_real(2.5).unwrap();
    assert_eq!(
        f64_data(scaled.materialized().unwrap().as_ref()),
        &[2.5, -5.0]
    );
    assert!(real.scale_complex(Complex64::new(0.0, 1.0)).is_err());

    let integer = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![3_i64]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    assert_eq!(
        integer
            .scale_real(2.5)
            .unwrap()
            .materialized()
            .unwrap()
            .as_slice::<i64>()
            .unwrap(),
        &[9]
    );
    assert!(integer.scale_real(f64::NAN).is_err());

    let complex = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![1], vec![Complex64::new(1.0, 2.0)]).unwrap(),
        ctx,
    )
    .unwrap();
    assert_eq!(
        c64_data(
            complex
                .scale_complex(Complex64::new(0.0, 1.0))
                .unwrap()
                .materialized()
                .unwrap()
                .as_ref()
        ),
        &[Complex64::new(-2.0, 1.0)]
    );

    let ctx = test_ctx();
    let tracked = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, -2.0]).unwrap(),
        ctx,
    )
    .unwrap();
    tracked
        .scale_real(2.5)
        .unwrap()
        .reduce_sum(&[0])
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(
        f64_data(tracked.grad().unwrap().unwrap().as_ref()),
        &[2.5, 2.5]
    );
}

#[test]
fn eager_tensor_has_one_step_column_major_constructor() {
    let eager =
        EagerTensor::from_vec_col_major_in([2, 2], vec![1.0_f64, 3.0, 2.0, 4.0], test_ctx())
            .unwrap();

    assert_eq!(eager.shape(), &[2, 2]);
    assert_eq!(
        eager.materialized().unwrap().as_slice::<f64>().unwrap(),
        &[1.0, 3.0, 2.0, 4.0]
    );
}

#[test]
fn eager_gather_keeps_indices_integer_for_complex_operand() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();
    let indices = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 1], vec![2_i64, 0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let y = x
        .gather(
            &indices,
            GatherConfig {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            },
        )
        .unwrap();

    assert_eq!(y.shape(), &[2]);
    assert_eq!(
        c64_data(y.materialized().unwrap().as_ref()),
        &[Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)]
    );
}

#[test]
fn eager_index_select_keeps_indices_integer_for_complex_operand() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3],
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(3.0, 0.5),
            ],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();

    let y = x.index_select(-1, &[2, 0]).unwrap();

    assert_eq!(y.shape(), &[2]);
    assert_eq!(
        c64_data(y.materialized().unwrap().as_ref()),
        &[Complex64::new(3.0, 0.5), Complex64::new(1.0, 1.0)]
    );
}

#[test]
fn eager_stack_trailing_axis_and_index_select_primal() {
    let x0 = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let x1 = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let stacked = EagerTensor::stack(&[&x0, &x1], -1).unwrap();
    let selected = stacked.index_select(-1, &[1, 0, 1]).unwrap();

    assert_eq!(selected.shape(), &[2, 3]);
    assert_close_slice(
        f64_data(selected.materialized().unwrap().as_ref()),
        &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        TOL,
    );
}

#[test]
fn eager_index_select_rejects_invalid_axis_and_position() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let axis_err = x.index_select(1, &[0]).err().unwrap().to_string();
    assert!(axis_err.contains("index_select"), "got: {axis_err}");
    assert!(axis_err.contains("axis"), "got: {axis_err}");

    let position_err = x.index_select(0, &[2]).err().unwrap().to_string();
    assert!(position_err.contains("index_select"), "got: {position_err}");
    assert!(
        position_err.contains("position 2 out of bounds"),
        "got: {position_err}"
    );
}

#[test]
fn eager_stack_rejects_empty_mismatched_shapes_and_invalid_axis() {
    let empty: [&EagerTensor; 0] = [];
    let empty_err = EagerTensor::stack(&empty, 0).err().unwrap();
    assert_eq!(
        empty_err.kind(),
        ErrorKind::Validation(ValidationKind::InvalidArgument)
    );
    assert!(matches!(
        empty_err,
        RuntimeError::TensorRuntime(TensorError::Validation {
            source: ValidationError::InvalidArgument {
                argument: "inputs",
                ..
            },
            ..
        })
    ));

    let a = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let b = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![3.0_f64, 4.0, 5.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let shape_err = EagerTensor::stack(&[&a, &b], -1).err().unwrap();
    assert_eq!(
        shape_err.kind(),
        ErrorKind::Validation(ValidationKind::ShapeMismatch)
    );
    assert!(matches!(
        shape_err,
        RuntimeError::TensorRuntime(TensorError::Validation {
            source: ValidationError::ShapeMismatch(_),
            ..
        })
    ));

    let axis_err = EagerTensor::stack(&[&a], 2).err().unwrap();
    assert_eq!(
        axis_err.kind(),
        ErrorKind::Validation(ValidationKind::AxisOutOfBounds)
    );
    assert!(matches!(
        axis_err,
        RuntimeError::TensorRuntime(TensorError::Validation {
            source: ValidationError::AxisOutOfBounds { .. },
            ..
        })
    ));

    let c = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let out = EagerTensor::stack(&[&a, &c], 0).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    assert_close_slice(
        f64_data(out.materialized().unwrap().as_ref()),
        &[1.0, 3.0, 2.0, 4.0],
        TOL,
    );
}

#[test]
fn eager_index_select_repeated_positions_accumulates_grad() {
    let ctx = test_ctx();
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let weights = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap(),
        ctx,
    )
    .unwrap();

    let selected = x.index_select(0, &[1, 1, 2]).unwrap();
    let loss = selected.mul(&weights).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[0.0, 30.0, 30.0],
        TOL,
    );
}

#[test]
fn eager_x_squared_gradient_matches_finite_difference() {
    let x_data = vec![1.0, 2.0, 3.0];
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], x_data.clone()).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();
    let grad = x.grad().unwrap().unwrap();

    let grad_data = f64_data(grad.as_ref());
    let expected: Vec<f64> = (0..x_data.len())
        .map(|index| {
            finite_diff_scalar(
                |values| values.iter().map(|v| v * v).sum::<f64>(),
                &x_data,
                index,
            )
        })
        .collect();
    assert_close_slice(grad_data, &expected, FD_TOL);
}

#[test]
fn eager_repeated_backward_accumulates_across_calls() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[2.0, 4.0, 6.0],
        TOL,
    );

    let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();
    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[4.0, 8.0, 12.0],
        TOL,
    );
}

#[test]
fn eager_matmul_gradients_match_finite_difference() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let a = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], a_data.clone()).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let b = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3, 2], b_data.clone()).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let loss = a
        .dot_general(&b, matmul_config())
        .unwrap()
        .reduce_sum(&[0, 1])
        .unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad_a = a.grad().unwrap().unwrap();
    let grad_b = b.grad().unwrap().unwrap();
    let grad_a_data = f64_data(grad_a.as_ref());
    let grad_b_data = f64_data(grad_b.as_ref());

    let expected_a: Vec<f64> = (0..a_data.len())
        .map(|index| finite_diff_lhs(eager_matmul_sum, &a_data, &b_data, index))
        .collect();
    let expected_b: Vec<f64> = (0..b_data.len())
        .map(|index| finite_diff_rhs(eager_matmul_sum, &a_data, &b_data, index))
        .collect();

    assert_close_slice(grad_a_data, &expected_a, FD_TOL);
    assert_close_slice(grad_b_data, &expected_b, FD_TOL);
}

#[test]
fn eager_exp_gradient_matches_primal() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![0.0, 1.0, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let loss = x.exp().unwrap().reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    let expected = vec![1.0, 1.0_f64.exp(), 2.0_f64.exp()];
    assert_close_slice(f64_data(grad.as_ref()), &expected, TOL);
}

#[test]
fn eager_fan_out_accumulates_gradient() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let loss = x.add(&x).unwrap().reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[2.0, 2.0, 2.0], TOL);
}

#[test]
fn eager_clear_grad_resets_only_one_leaf() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let loss = x.mul(&y).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    x.clear_grad().unwrap();

    assert!(x.grad().unwrap().is_none());
    assert_close_slice(
        f64_data(y.grad().unwrap().unwrap().as_ref()),
        &[1.0, 2.0, 3.0],
        TOL,
    );

    let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[2.0, 4.0, 6.0],
        TOL,
    );
    assert_close_slice(
        f64_data(y.grad().unwrap().unwrap().as_ref()),
        &[1.0, 2.0, 3.0],
        TOL,
    );
}

#[test]
fn eager_context_clear_grads_resets_all_live_leaves() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let loss = x.mul(&y).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    ctx.clear_grads().unwrap();

    assert!(x.grad().unwrap().is_none());
    assert!(y.grad().unwrap().is_none());

    let loss = x.mul(&y).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss.backward().unwrap();

    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[4.0, 5.0, 6.0],
        TOL,
    );
    assert_close_slice(
        f64_data(y.grad().unwrap().unwrap().as_ref()),
        &[1.0, 2.0, 3.0],
        TOL,
    );
}

#[test]
fn eager_unrelated_backward_keeps_existing_leaf_grad() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let y = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();

    let loss_x = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss_x.backward().unwrap();
    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[2.0, 4.0, 6.0],
        TOL,
    );

    let loss_y = y.mul(&y).unwrap().reduce_sum(&[0]).unwrap();
    let _ = loss_y.backward().unwrap();

    assert_close_slice(
        f64_data(x.grad().unwrap().unwrap().as_ref()),
        &[2.0, 4.0, 6.0],
        TOL,
    );
    assert_close_slice(
        f64_data(y.grad().unwrap().unwrap().as_ref()),
        &[8.0, 10.0, 12.0],
        TOL,
    );
}

#[test]
fn eager_tracks_grad_reports_leaf_state() {
    let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new());
    let plain = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        ctx.clone(),
    )
    .unwrap();
    let leaf = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0_f64, 5.0, 6.0]).unwrap(),
        ctx,
    )
    .unwrap();

    assert!(!plain.tracks_grad());
    assert!(leaf.tracks_grad());
    assert!(!leaf.detach().tracks_grad());
}

#[test]
fn eager_send_sync_contracts_compile() {
    assert_send_sync::<EagerTensor>();
    assert_send_sync::<EagerRuntime>();
}

#[test]
fn eager_context_and_tensor_are_backend_erased_public_types() {
    assert_send_sync::<EagerTensor>();
    assert_send_sync::<EagerRuntime>();

    let ctx: Arc<EagerRuntime> =
        EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1).unwrap());
    let x = ctx
        .variable_from(Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap())
        .unwrap();
    let loss = x.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
    loss.backward().unwrap();

    assert_eq!(
        x.grad().unwrap().unwrap().as_slice::<f64>().unwrap(),
        &[2.0, 4.0]
    );
}

#[test]
fn eager_detach_cuts_one_gradient_path() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let detached = x.detach();
    let loss = detached.mul(&x).unwrap().reduce_sum(&[0]).unwrap();
    let _cotangents = loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[1.0, 2.0, 3.0], TOL);
    assert!(detached.grad().unwrap().is_none());
}

#[test]
fn eager_untracked_tensor_behaves_like_plain_tensor() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![4.0, 5.0, 6.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let z = x.mul(&y).unwrap();

    assert_close_slice(
        f64_data(z.materialized().unwrap().as_ref()),
        &[4.0, 10.0, 18.0],
        TOL,
    );
    assert!(x.grad().unwrap().is_none());
    assert!(y.grad().unwrap().is_none());
    assert!(z.grad().unwrap().is_none());
}

#[test]
fn eager_structural_primal_ops_transpose_and_reshape() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let transposed = x.transpose(&[1, 0]).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_close_slice(
        f64_data(transposed.materialized().unwrap().as_ref()),
        &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
        TOL,
    );

    let reshaped = x.reshape(&[6]).unwrap();
    assert_eq!(reshaped.shape(), &[6]);
    assert_close_slice(
        f64_data(reshaped.materialized().unwrap().as_ref()),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        TOL,
    );
}

#[test]
fn eager_untracked_structural_ops_return_lazy_views() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let transposed = x.transpose(&[1, 0]).unwrap();
    match transposed.tensor_read() {
        TensorRead::View(TensorView::F64(view)) => {
            assert_eq!(view.shape(), &[3, 2]);
            assert_eq!(view.strides(), &[2, 1]);
        }
        other => panic!("expected transpose to remain a lazy f64 view, got {other:?}"),
    }

    let reshaped = x.reshape(&[6]).unwrap();
    match reshaped.tensor_read() {
        TensorRead::View(TensorView::F64(view)) => {
            assert_eq!(view.shape(), &[6]);
            assert_eq!(view.strides(), &[1]);
        }
        other => panic!("expected reshape to remain a lazy f64 view, got {other:?}"),
    }
}

#[test]
fn eager_tracked_structural_ops_return_lazy_views_and_backprop() {
    let x = EagerTensor::requires_grad_in(
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let transposed = x.transpose(&[1, 0]).unwrap();
    match transposed.tensor_read() {
        TensorRead::View(TensorView::F64(view)) => {
            assert_eq!(view.shape(), &[3, 2]);
            assert_eq!(view.strides(), &[2, 1]);
        }
        other => panic!("expected tracked transpose to remain a lazy f64 view, got {other:?}"),
    }

    let loss = transposed.reduce_sum(&[0, 1]).unwrap();
    let _cotangents = loss.backward().unwrap();
    let grad = x.grad().unwrap().unwrap();
    assert_close_slice(f64_data(grad.as_ref()), &[1.0; 6], TOL);
}

#[test]
fn eager_elementwise_primal_ops_div_abs_and_sin() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![8.0_f64, -6.0, 9.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 3.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let div = x.div(&y).unwrap();
    assert_close_slice(
        f64_data(div.materialized().unwrap().as_ref()),
        &[4.0, -2.0, 3.0],
        TOL,
    );

    let abs = x.abs().unwrap();
    assert_close_slice(
        f64_data(abs.materialized().unwrap().as_ref()),
        &[8.0, 6.0, 9.0],
        TOL,
    );

    let angles = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::FRAC_PI_2]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let sin = angles.sin().unwrap();
    assert_close_slice(
        f64_data(sin.materialized().unwrap().as_ref()),
        &[0.0, 1.0],
        TOL,
    );
}

#[test]
fn eager_diagonal_primal_ops_extract_diag_and_tril() {
    let matrix = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 3],
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();
    let diag = matrix.extract_diag(0, 1).unwrap();
    assert_close_slice(
        f64_data(diag.materialized().unwrap().as_ref()),
        &[1.0, 5.0, 9.0],
        TOL,
    );

    let lower = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap()
    .tril(0)
    .unwrap();
    assert_close_slice(
        f64_data(lower.materialized().unwrap().as_ref()),
        &[1.0, 2.0, 0.0, 4.0],
        TOL,
    );
}

#[test]
fn eager_reduction_primal_ops_reduce_prod() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();

    let prod = x.reduce_prod(&[0, 1]).unwrap();
    assert_close_slice(
        f64_data(prod.materialized().unwrap().as_ref()),
        &[24.0],
        TOL,
    );

    let max = x.reduce_max(&[0, 1]).unwrap();
    assert_close_slice(f64_data(max.materialized().unwrap().as_ref()), &[4.0], TOL);

    let min = x.reduce_min(&[0, 1]).unwrap();
    assert_close_slice(f64_data(min.materialized().unwrap().as_ref()), &[1.0], TOL);
}

#[test]
fn eager_slice_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![4, 3],
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();

    let y = x
        .slice(SliceConfig {
            starts: vec![0, 0],
            limits: vec![4, 3],
            strides: vec![2, 2],
        })
        .unwrap();

    assert_eq!(y.shape(), &[2, 2]);
    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[1.0, 3.0, 9.0, 11.0],
        TOL,
    );
}

#[test]
fn eager_untracked_slice_returns_lazy_view() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![4, 3],
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();

    let y = x
        .slice(SliceConfig {
            starts: vec![0, 0],
            limits: vec![4, 3],
            strides: vec![2, 2],
        })
        .unwrap();

    match y.tensor_read() {
        TensorRead::View(TensorView::F64(view)) => {
            assert_eq!(view.shape(), &[2, 2]);
            assert_eq!(view.strides(), &[2, 8]);
        }
        other => panic!("expected slice to remain a lazy f64 view, got {other:?}"),
    }
    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[1.0, 3.0, 9.0, 11.0],
        TOL,
    );
}

#[test]
fn eager_broadcast_in_dim_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x.broadcast_in_dim(&[3, 2], &[0]).unwrap();

    assert_eq!(y.shape(), &[3, 2]);
    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        TOL,
    );
}

#[test]
fn eager_untracked_broadcast_in_dim_returns_lazy_view() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x.broadcast_in_dim(&[3, 2], &[0]).unwrap();

    match y.tensor_read() {
        TensorRead::View(TensorView::F64(view)) => {
            assert_eq!(view.shape(), &[3, 2]);
            assert_eq!(view.strides(), &[1, 0]);
        }
        other => panic!("expected broadcast_in_dim to remain a lazy f64 view, got {other:?}"),
    }
    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        TOL,
    );
}

#[test]
fn eager_pad_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x
        .pad(PadConfig {
            edge_padding_low: vec![1],
            edge_padding_high: vec![1],
            interior_padding: vec![1],
        })
        .unwrap();

    assert_eq!(y.shape(), &[5]);
    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[0.0, 1.0, 0.0, 2.0, 0.0],
        TOL,
    );
}

#[test]
fn eager_reverse_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x.reverse(&[0]).unwrap();

    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[4.0, 3.0, 2.0, 1.0],
        TOL,
    );
}

#[test]
fn eager_concatenate_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let z = EagerTensor::concatenate(&[&x, &y], 0).unwrap();

    assert_eq!(z.shape(), &[4]);
    assert_close_slice(
        f64_data(z.materialized().unwrap().as_ref()),
        &[1.0, 2.0, 3.0, 4.0],
        TOL,
    );
}

#[test]
fn eager_gather_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![5], vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let indices = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![4_i64, 1, 0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x
        .gather(
            &indices,
            GatherConfig {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            },
        )
        .unwrap();

    assert_eq!(y.shape(), &[3]);
    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[50.0, 20.0, 10.0],
        TOL,
    );
}

#[test]
fn eager_dynamic_slice_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![4, 4],
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();
    let starts = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2_i64, 3]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x.dynamic_slice(&starts, &[2, 2]).unwrap();

    assert_eq!(y.shape(), &[2, 2]);
    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[11.0, 12.0, 15.0, 16.0],
        TOL,
    );
}

#[test]
fn eager_conj_primal() {
    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![2],
            vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 0.5)],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = x.conj().unwrap();

    assert_eq!(
        c64_data(y.materialized().unwrap().as_ref()),
        &[Complex64::new(1.0, -2.0), Complex64::new(-3.0, -0.5)]
    );
}

#[test]
fn eager_analytic_primal_ops_sign_log_sqrt_rsqrt_cos_tanh_expm1_log1p() {
    let sign_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![-2.0_f64, 0.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let sign = sign_input.sign().unwrap();
    assert_close_slice(
        f64_data(sign.materialized().unwrap().as_ref()),
        &[-1.0, 0.0, 1.0],
        TOL,
    );

    let log_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, std::f64::consts::E]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let log = log_input.log().unwrap();
    assert_close_slice(
        f64_data(log.materialized().unwrap().as_ref()),
        &[0.0, 1.0],
        TOL,
    );

    let sqrt_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let sqrt = sqrt_input.sqrt().unwrap();
    let rsqrt = sqrt_input.rsqrt().unwrap();
    assert_close_slice(
        f64_data(sqrt.materialized().unwrap().as_ref()),
        &[1.0, 2.0],
        TOL,
    );
    assert_close_slice(
        f64_data(rsqrt.materialized().unwrap().as_ref()),
        &[1.0, 0.5],
        TOL,
    );

    let angles = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, std::f64::consts::PI]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let cos = angles.cos().unwrap();
    assert_close_slice(
        f64_data(cos.materialized().unwrap().as_ref()),
        &[1.0, -1.0],
        TOL,
    );

    let tanh_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let tanh = tanh_input.tanh().unwrap();
    assert_close_slice(
        f64_data(tanh.materialized().unwrap().as_ref()),
        &[0.0, 1.0_f64.tanh()],
        TOL,
    );

    let expm1_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![0.0_f64, 1.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let expm1 = expm1_input.expm1().unwrap();
    assert_close_slice(
        f64_data(expm1.materialized().unwrap().as_ref()),
        &[0.0, 1.0_f64.exp_m1()],
        TOL,
    );

    let log1p_input = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 4.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let log1p = log1p_input.log1p().unwrap();
    assert_close_slice(
        f64_data(log1p.materialized().unwrap().as_ref()),
        &[2.0_f64.ln(), 5.0_f64.ln()],
        TOL,
    );
}

#[test]
fn eager_pow_maximum_and_minimum_primal() {
    let base = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![2.0_f64, 9.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let exp = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![2], vec![3.0_f64, 0.5]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let pow = base.pow(&exp).unwrap();
    assert_close_slice(
        f64_data(pow.materialized().unwrap().as_ref()),
        &[8.0, 3.0],
        TOL,
    );

    let x = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![8.0_f64, -2.0, 9.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![2.0_f64, 5.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let maximum = x.maximum(&y).unwrap();
    let minimum = x.minimum(&y).unwrap();
    assert_close_slice(
        f64_data(maximum.materialized().unwrap().as_ref()),
        &[8.0, 5.0, 9.0],
        TOL,
    );
    assert_close_slice(
        f64_data(minimum.materialized().unwrap().as_ref()),
        &[2.0, -2.0, 3.0],
        TOL,
    );
}

#[test]
fn eager_select_primal() {
    let condition = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![false, true, true]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let on_true = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![10.0_f64, 20.0, 30.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let on_false = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let y = EagerTensor::select(&condition, &on_true, &on_false).unwrap();

    assert_close_slice(
        f64_data(y.materialized().unwrap().as_ref()),
        &[1.0, 20.0, 30.0],
        TOL,
    );
}

#[test]
fn eager_embed_diag_and_triu_primal() {
    let diagonal = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(vec![3], vec![1.0_f64, 2.0, 3.0]).unwrap(),
        test_ctx(),
    )
    .unwrap();
    let embedded = diagonal.embed_diag(0, 1).unwrap();
    assert_eq!(embedded.shape(), &[3, 3]);
    assert_close_slice(
        f64_data(embedded.materialized().unwrap().as_ref()),
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        TOL,
    );

    let matrix = EagerTensor::from_tensor_in(
        Tensor::from_vec_col_major(
            vec![3, 3],
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap(),
        test_ctx(),
    )
    .unwrap();
    let upper = matrix.triu(0).unwrap();
    assert_close_slice(
        f64_data(upper.materialized().unwrap().as_ref()),
        &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0],
        TOL,
    );
}
