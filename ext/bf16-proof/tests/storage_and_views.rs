//! Bfloat16 storage, views, and materialization through the erased boundary.
//!
//! #1785 asks for construction, views, and materialization to work for bfloat16 the way they do
//! for the other external scalars. The payload is caller-owned, so nothing here depends on the
//! pooled storage boundary. The materialization contract asserted below is the same one
//! `ext/df64-proof/tests/external_views.rs` verifies for the extended scalar: materializing a
//! permuted view keeps the payload's own element order under a fresh dense layout, so an erased
//! read sees the permuted order without any narrowing.

use tenferro_bf16_proof::Bf16;
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn payload(values: &[f32], shape: &[usize]) -> ErasedHostTensor {
    let host = HostTensor::from_vec_col_major(
        shape.to_vec(),
        values.iter().copied().map(Bf16::from_f32).collect(),
    )
    .expect("shape matches data");
    ErasedHostTensor::new(host)
}

fn elements(tensor: &ErasedHostTensor) -> Vec<f32> {
    tensor
        .as_dense::<Bf16>()
        .expect("a dense payload")
        .0
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

#[test]
fn construction_records_the_actual_type_and_shape() {
    let tensor = payload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    assert!(tensor.is::<Bf16>(), "the stored type is bfloat16");
    assert!(
        !tensor.is::<f32>(),
        "the stored type is not the widened one"
    );
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.element_count(), 4);
    assert!(tensor.is_contiguous());
    assert_eq!(tensor.strides(), &[1, 2]);
    assert_eq!(elements(&tensor), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_permutation_is_metadata_only_and_reads_in_the_permuted_order() {
    let tensor = payload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let permuted = tensor.permuted(&[1, 0]).expect("permutation");

    assert!(permuted.shares_payload_with(&tensor));
    assert_eq!(permuted.shape(), &[2, 2]);
    assert_eq!(permuted.strides(), &[2, 1]);
    assert!(!permuted.is_contiguous());

    // The permuted view reads the same memory in the other order, so its logical (0, 1) is the
    // stored matrix's (1, 0). A strided view is not a dense slice, so a typed projection refuses
    // it rather than reinterpreting bytes.
    let transposed = permuted
        .element_at::<Bf16>(&[0, 1])
        .expect("the view is inside its payload")
        .to_f32();
    let stored = tensor
        .element_at::<Bf16>(&[1, 0])
        .expect("the payload has that element")
        .to_f32();
    assert_eq!(transposed, stored);
    assert!(permuted.downcast_ref::<Bf16>().is_none());
}

#[test]
fn materialization_keeps_the_payload_order_under_a_dense_layout() {
    let tensor = payload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let dense = tensor
        .permuted(&[1, 0])
        .expect("permutation")
        .to_contiguous()
        .expect("materialization");

    assert!(dense.is_contiguous());
    assert!(!dense.shares_payload_with(&tensor));
    assert_eq!(dense.shape(), &[2, 2]);
    assert_eq!(
        elements(&dense),
        vec![1.0, 3.0, 2.0, 4.0],
        "materialization keeps the payload order and gives it a dense layout"
    );
    // Under the dense layout the element the view called (0, 1) is the payload's third element.
    assert_eq!(
        dense.element_at::<Bf16>(&[0, 1]).expect("element").to_f32(),
        2.0
    );
}

#[test]
fn a_clone_shares_the_payload_and_a_duplicate_copies_it() {
    let mut tensor = payload(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    // A uniquely owned payload accepts a mutable borrow.
    *tensor
        .element_at_mut::<Bf16>(&[1, 0])
        .expect("a unique payload accepts a mutable element borrow") = Bf16::from_f32(9.0);
    assert_eq!(elements(&tensor), vec![1.0, 9.0, 3.0, 4.0]);

    // A duplicate copies the payload, so a later write to the original must not reach it.
    let copied = tensor.duplicate();
    assert!(!copied.shares_payload_with(&tensor));
    *tensor
        .element_at_mut::<Bf16>(&[0, 1])
        .expect("the original is still uniquely owned") = Bf16::from_f32(7.0);
    assert_eq!(
        elements(&tensor),
        vec![1.0, 9.0, 7.0, 4.0],
        "the write addresses the payload element the index names"
    );
    assert_eq!(
        elements(&copied),
        vec![1.0, 9.0, 3.0, 4.0],
        "a duplicate owns its own buffer and does not observe a later write"
    );

    // A clone shares the buffer, and sharing withdraws the mutable borrow: the guard refuses
    // rather than letting one owner observe a write through another's reference.
    let shared = tensor.clone();
    assert!(shared.shares_payload_with(&tensor));
    assert!(
        tensor.element_at_mut::<Bf16>(&[1, 0]).is_none(),
        "a shared payload must refuse a mutable element borrow"
    );
    assert_eq!(elements(&shared), vec![1.0, 9.0, 7.0, 4.0]);
}

#[test]
fn a_mismatched_projection_returns_nothing() {
    let tensor = payload(&[1.0], &[1]);
    assert!(tensor.downcast_ref::<f32>().is_none());
    assert!(tensor.element_at::<f32>(&[0]).is_none());
}
