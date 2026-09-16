//! Module-local tests for this file's test-only materialization helpers.
//!
//! These helpers are compiled under `cfg(test)`, so they are covered only when a test calls
//! them; the crate's other modules exercise the public operations, not this seam.

use super::{host_typed, materialize_tensor_read};
use num_complex::{Complex32, Complex64};
use tenferro_cpu_basic::BufferPool;
use tenferro_tensor::{Tensor, TensorRead, TypedTensor};

fn column<T: tenferro_tensor::TensorScalar + Copy>(values: Vec<T>) -> Tensor {
    let rows = values.len();
    Tensor::from_vec_col_major(vec![rows, 1], values).unwrap()
}

fn cases() -> Vec<Tensor> {
    vec![
        column(vec![1.0_f32, 2.0]),
        column(vec![1.0_f64, 2.0]),
        column(vec![1_i32, 2]),
        column(vec![1_i64, 2]),
        column(vec![true, false]),
        column(vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)]),
        column(vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)]),
    ]
}

/// Reading an owned tensor goes through the clone path for every preset scalar.
#[test]
fn materialize_tensor_read_clones_every_preset_scalar() {
    let mut pool = BufferPool::new();
    for input in cases() {
        let out =
            materialize_tensor_read(&mut pool, "materialize", TensorRead::from_tensor(&input))
                .expect("reading an owned preset tensor must clone it");
        assert_eq!(out.dtype(), input.dtype());
        assert_eq!(out.shape(), input.shape());
    }
}

/// A borrowed view takes the other branch, which materializes rather than clones.
#[test]
fn materialize_tensor_read_materializes_a_view() {
    let mut pool = BufferPool::new();
    let owned = TypedTensor::<f64>::from_vec_col_major(vec![2, 1], vec![1.0, 2.0]).unwrap();
    let read = TensorRead::from_view(tenferro_tensor::TensorView::F64(owned.as_view()));
    let out = materialize_tensor_read(&mut pool, "materialize", read)
        .expect("reading a preset view must materialize it");
    assert_eq!(out.shape(), &[2, 1]);
}

/// The tag tables only reach the accessor with the scalar they claim, so its refusal is unreachable
/// from the public API. Covering it here keeps the invariant honest: a mismatch is a typed error
/// rather than a panic.
#[test]
fn host_typed_refuses_a_dtype_the_dispatch_never_produces() {
    let tensor = column(vec![1_i32, 2]);
    assert!(host_typed::<f32>("materialize", &tensor).is_err());
}
