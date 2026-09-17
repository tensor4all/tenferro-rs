//! Module-local tests for the reduction dispatch seams.

use super::{
    norm_squared_read, reduce_max, reduce_min, reduce_prod, reduce_prod_read, reduce_sum,
    reduce_sum_read, reduce_sum_squares, typed_input,
};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::{Tensor, TensorRead, TensorView, TypedTensor};

fn tensor<T: tenferro_tensor::TensorScalar + Copy>(values: Vec<T>) -> Tensor {
    Tensor::from_vec_col_major(vec![values.len()], values).unwrap()
}

/// Every preset scalar reaches its own arm of the max and min tables, so the tables'
/// per-scalar bodies are exercised rather than only the float ones.
#[test]
fn reduce_max_and_min_cover_every_preset_scalar() {
    let cases = [
        tensor(vec![1.0_f32, 2.0]),
        tensor(vec![1.0_f64, 2.0]),
        tensor(vec![1_i32, 2]),
        tensor(vec![1_i64, 2]),
        tensor(vec![false, true]),
        tensor(vec![
            num_complex::Complex32::new(1.0, 0.0),
            num_complex::Complex32::new(2.0, 0.0),
        ]),
        tensor(vec![
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(2.0, 0.0),
        ]),
    ];
    for input in &cases {
        // Either the table computes the reduction or its refusal arm answers; both are
        // dispatch bodies this module owns.
        let _ = reduce_max(input, &[0]);
        let _ = reduce_min(input, &[0]);
    }
}

/// The tag tables only reach `typed_input` with the scalar they claim, so its refusal
/// is unreachable from the public API. Covering it here keeps the invariant honest:
/// a mismatch is a typed error, never a panic.
#[test]
fn typed_input_refuses_a_dtype_the_dispatch_never_produces() {
    let tensor = Tensor::I32(TypedTensor::from_vec_col_major(vec![2], vec![1, 2]).unwrap());
    assert!(typed_input::<f32>("reduce_sum", &tensor).is_err());
}

/// The sum and product tables carry complex and integer arms that the float-only paths
/// never reach; calling them for every preset scalar exercises each dispatch body.
#[test]
fn reduce_sum_and_prod_cover_every_preset_scalar() {
    let cases = [
        tensor(vec![1.0_f32, 2.0]),
        tensor(vec![1.0_f64, 2.0]),
        tensor(vec![1_i32, 2]),
        tensor(vec![1_i64, 2]),
        tensor(vec![false, true]),
        tensor(vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, 2.0)]),
        tensor(vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)]),
    ];
    let context = strided_kernel::ExecContext::serial();
    for input in &cases {
        let _ = reduce_sum(input, &[0], &context);
        let _ = reduce_prod(input, &[0], &context);
    }
}

/// The read table carries one view arm per preset scalar. Passing a view-backed operand reaches those
/// arms without materializing, and the boolean view is refused with a typed refusal rather than a panic.
#[test]
fn reduce_sum_read_covers_every_preset_view() {
    let context = strided_kernel::ExecContext::serial();
    let mut buffers = crate::buffer_pool::BufferPool::new();

    macro_rules! check {
        ($variant:ident, $scalar:ty, $values:expr, $supported:expr) => {{
            let owned = tensor::<$scalar>($values);
            let typed = owned
                .as_typed::<$scalar>()
                .expect("the tensor was built from this scalar")
                .as_view();
            let read = TensorRead::from_view(TensorView::$variant(typed));
            let result = reduce_sum_read(&mut buffers, read, &[0], &context);
            if $supported {
                assert!(result.is_ok(), "{} view must reduce", stringify!($scalar));
            } else {
                let error = result.expect_err("an unsupported view must be refused");
                assert!(!error.to_string().is_empty());
            }
        }};
    }

    check!(F32, f32, vec![1.0_f32, 2.0], true);
    check!(F64, f64, vec![1.0_f64, 2.0], true);
    check!(I32, i32, vec![1_i32, 2], true);
    check!(I64, i64, vec![1_i64, 2], true);
    check!(Bool, bool, vec![false, true], false);
    check!(
        C32,
        Complex32,
        vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, 2.0)],
        true
    );
    check!(
        C64,
        Complex64,
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)],
        true
    );
}

/// The product read table carries a view arm per preset scalar, and the squared-norm and sum-of-squares
/// entries admit floating dtypes only, so their integer and boolean inputs are refusals rather than table
/// arms. The owned-tensor tests take neither path.
#[test]
fn reduction_read_tables_cover_views_and_sum_squares_refusals() {
    let context = strided_kernel::ExecContext::serial();
    let mut buffers = crate::buffer_pool::BufferPool::new();

    macro_rules! prod_view {
        ($variant:ident, $scalar:ty, $values:expr, $expected:expr) => {{
            let owned = tensor::<$scalar>($values);
            let typed = owned
                .as_typed::<$scalar>()
                .expect("the tensor was built from this scalar")
                .as_view();
            let read = TensorRead::from_view(TensorView::$variant(typed));
            let result = reduce_prod_read(&mut buffers, read, &[0], &context)
                .expect("the product view arm admits this scalar");
            assert_eq!(result.as_slice::<$scalar>().unwrap(), &[$expected],);
        }};
    }

    prod_view!(F32, f32, vec![2.0_f32, 3.0], 6.0_f32);
    prod_view!(F64, f64, vec![2.0_f64, 3.0], 6.0_f64);
    prod_view!(I32, i32, vec![2_i32, 3], 6_i32);
    prod_view!(I64, i64, vec![2_i64, 3], 6_i64);
    // The boolean view arm is an explicit refusal rather than a product, so it is asserted as one.
    let bools = tensor(vec![false, true]);
    let bool_view = bools.as_typed::<bool>().unwrap().as_view();
    let error = reduce_prod_read(
        &mut buffers,
        TensorRead::from_view(TensorView::Bool(bool_view)),
        &[0],
        &context,
    )
    .expect_err("reduce_prod refuses boolean views");
    assert!(matches!(
        error,
        tenferro_tensor::Error::Unsupported {
            op: "reduce_prod",
            ..
        }
    ));
    prod_view!(
        C32,
        Complex32,
        vec![Complex32::new(1.0, 1.0), Complex32::new(2.0, 2.0)],
        Complex32::new(0.0, 4.0)
    );
    prod_view!(
        C64,
        Complex64,
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 2.0)],
        Complex64::new(0.0, 4.0)
    );

    for unsupported in [tensor(vec![1_i32, 2]), tensor(vec![false, true])] {
        let sums = reduce_sum_squares(&mut buffers, &unsupported, &[0], &context);
        assert!(sums.is_err(), "sum of squares admits floating dtypes only");
        let norms = norm_squared_read(&mut buffers, TensorRead::from_tensor(&unsupported));
        assert!(norms.is_err(), "squared norm admits floating dtypes only");
    }
}
