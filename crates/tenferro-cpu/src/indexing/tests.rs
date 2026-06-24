use std::panic::{catch_unwind, AssertUnwindSafe};

use super::{index_component, typed_concatenate, BufferPool, IndexTensor};
use tenferro_tensor::{Error, TypedTensor};

#[test]
fn typed_concatenate_rejects_empty_typed_inputs_without_panicking() {
    let mut buffers = BufferPool::new();

    let result = catch_unwind(AssertUnwindSafe(|| {
        typed_concatenate::<f64>(&mut buffers, &[], 0)
    }));

    assert!(result.is_ok(), "empty typed concatenate should return Err");
    assert!(matches!(
        result.unwrap().unwrap_err(),
        Error::InvalidConfig {
            op: "concatenate",
            ..
        }
    ));
}

#[test]
fn index_component_rejects_mismatched_scratch_len_without_panicking() {
    let mut scratch = vec![0usize; 1];
    let indices = IndexTensor {
        shape: vec![1, 1],
        values: vec![7],
    };

    let result = catch_unwind(AssertUnwindSafe(|| {
        index_component("gather", &indices, &[0], 1, 0, &mut scratch)
    }));

    assert!(
        result.is_ok(),
        "index_component should return Err for a malformed scratch buffer"
    );
    assert!(matches!(
        result.unwrap().unwrap_err(),
        Error::InvalidConfig { op: "gather", .. }
    ));
}

#[test]
fn typed_concatenate_accepts_nonempty_typed_inputs() {
    let mut buffers = BufferPool::new();
    let a = TypedTensor::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let b = TypedTensor::from_vec_col_major(vec![1], vec![3.0]).unwrap();
    let inputs = vec![&a, &b];

    let out = typed_concatenate(&mut buffers, &inputs, 0).unwrap();

    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.host_data().unwrap(), &[1.0, 2.0, 3.0]);
}
