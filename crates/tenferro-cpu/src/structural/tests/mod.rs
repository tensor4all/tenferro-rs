//! Module-local tests for `structural`.
//!
//! The host-view helpers live here, so the layout-borrowing check stays next to
//! the code it constrains (REPOSITORY_RULES.md, Unit Test Organization).

use crate::structural::host_view;
use tenferro_tensor::TypedTensor;

/// The compact layout lives with the tensor, so the view borrows it: this is
/// what makes a per-operation view free of allocation.
#[test]
fn host_view_borrows_the_tensor_layout() {
    let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();

    let view = host_view("transpose", &tensor).unwrap();

    assert_eq!(view.dims(), tensor.shape());
    assert_eq!(view.strides(), tensor.strides());
    assert_eq!(view.strides().as_ptr(), tensor.strides().as_ptr());
}
