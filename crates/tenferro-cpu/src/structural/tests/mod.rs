//! Module-local tests for `structural`.
//!
//! The host-view helpers live here, so the layout-borrowing check stays next to
//! the code it constrains (REPOSITORY_RULES.md, Unit Test Organization).

use crate::typed_view;
use tenferro_tensor::{SliceConfig, Tensor, TensorScalar, TensorValue, TypedTensor};

/// A group-backed owner extracted from a larger root keeps its descriptor
/// layout, so the CPU adapters must read the logical slice rather than the root
/// prefix (#2023 review).
fn offset_slice_owner() -> TypedTensor<f64> {
    let value = TensorValue::from_tensor(
        Tensor::from_vec_col_major(vec![4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
    )
    .slice_view(&SliceConfig {
        starts: vec![1],
        limits: vec![3],
        strides: vec![1],
    })
    .unwrap();
    let (group, slot, _, _) = value
        .try_into_group_parts()
        .unwrap_or_else(|_| unreachable!("a pooled value enters a group"));
    let tensor = group
        .into_tensor(slot)
        .unwrap_or_else(|_| unreachable!("a compact root-spanning slice extracts"));
    <f64 as TensorScalar>::into_typed(tensor).unwrap_or_else(|_| unreachable!())
}

#[test]
fn per_operation_views_read_a_root_preserving_slice() {
    let tensor = offset_slice_owner();

    let borrowed = typed_view("convert", &tensor).unwrap();
    assert_eq!(borrowed.dims(), &[2]);
    assert_eq!(borrowed.offset(), 0);
    assert_eq!([borrowed.get(&[0]), borrowed.get(&[1])], [2.0, 3.0]);

    let host = super::host_view("extract_diagonal", &tensor).unwrap();
    assert_eq!([host.get(&[0]), host.get(&[1])], [2.0, 3.0]);
}

/// The view carries the tensor's canonical layout. Since strided-view 0.6 the
/// view owns its metadata inline, so a per-operation view at rank
/// `<= INLINE_RANK` is free of allocation.
#[test]
fn typed_view_exposes_the_tensor_layout() {
    let tensor = TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0; 6]).unwrap();

    let view = typed_view("transpose", &tensor).unwrap();

    assert_eq!(view.dims(), tensor.shape());
    assert_eq!(view.strides(), tensor.strides());
}
