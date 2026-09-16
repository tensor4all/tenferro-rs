//! Views over a caller-owned payload.
//!
//! #1785 requires the low-order component to survive typed and erased reads,
//! mutable views, a metadata-only axis permutation, explicit contiguous
//! materialization, and a sum reduction. The layout of an erased payload is what
//! makes the first four possible without a variant in the typed view types.

use tenferro_cpu::scalar_fold;
use tenferro_df64_proof::{Df64, Df64Add};
use tenferro_tensor::{DType, Tensor};
use tenferro_tensor_core::{ErasedHostTensor, HostTensor};

fn payload(values: &[Df64], shape: &[usize]) -> ErasedHostTensor {
    ErasedHostTensor::new(
        HostTensor::from_vec_col_major(shape.to_vec(), values.to_vec())
            .expect("shape matches data"),
    )
}

fn value(hi: f64, lo: f64) -> Df64 {
    Df64 { hi, lo }
}

#[test]
fn a_permutation_is_metadata_only_and_preserves_every_component() {
    let low = 2f64.powi(-80);
    let original = payload(
        &[
            value(1.0, low),
            value(2.0, 0.0),
            value(3.0, 0.0),
            value(4.0, 0.0),
        ],
        &[2, 2],
    );

    let permuted = original.permuted(&[1, 0]).expect("a valid permutation");

    // No element moved: the view shares the payload and only its layout changed.
    assert!(permuted.shares_payload_with(&original));
    assert_eq!(permuted.shape(), &[2, 2]);
    assert_eq!(permuted.strides(), &[2, 1]);
    assert!(!permuted.is_contiguous());

    // A typed read applies the layout, so the two axes really are exchanged.
    assert_eq!(permuted.element_at::<Df64>(&[0, 0]), Some(&value(1.0, low)));
    assert_eq!(permuted.element_at::<Df64>(&[1, 0]), Some(&value(3.0, 0.0)));
    assert_eq!(permuted.element_at::<Df64>(&[0, 1]), Some(&value(2.0, 0.0)));
    assert_eq!(permuted.element_at::<Df64>(&[1, 1]), Some(&value(4.0, 0.0)));
    assert_eq!(permuted.element_at::<Df64>(&[0, 5]), None);

    // The dense accessors refuse the strided view instead of presenting the
    // payload as if it were the view.
    assert!(permuted.as_dense::<Df64>().is_none());
    assert!(permuted.downcast_ref::<Df64>().is_none());
    assert!(permuted.clone().downcast_mut::<Df64>().is_none());
}

#[test]
fn a_mutable_view_writes_the_element_the_view_names() {
    let low = 2f64.powi(-80);
    let owned = payload(
        &[
            value(1.0, low),
            value(2.0, 0.0),
            value(3.0, 0.0),
            value(4.0, 0.0),
        ],
        &[2, 2],
    );

    // A mutable element borrow through a view needs the caller to be the only
    // holder of the payload, so the source is released first.
    let source = owned.duplicate();
    let mut view = source.permuted(&[1, 0]).expect("a valid permutation");
    drop(source);
    *view
        .element_at_mut::<Df64>(&[1, 0])
        .expect("the only holder") = value(30.0, low);

    // The write landed on the element the view named.
    assert_eq!(view.element_at::<Df64>(&[1, 0]), Some(&value(30.0, low)));
    assert_eq!(view.element_at::<Df64>(&[0, 0]), Some(&value(1.0, low)));

    // Permuting the same two axes again returns the dense raw order, which shows
    // the write in the third element and in no other one.
    let raw = view.permuted(&[1, 0]).expect("a valid permutation");
    assert!(raw.is_contiguous());
    assert_eq!(
        raw.as_dense::<Df64>().expect("dense raw order").0,
        &[
            value(1.0, low),
            value(2.0, 0.0),
            value(30.0, low),
            value(4.0, 0.0)
        ]
    );
}

#[test]
fn a_shared_payload_refuses_a_mutable_element_borrow() {
    let original = payload(&[value(1.0, 0.0), value(2.0, 0.0)], &[2]);

    // Two live views of one payload never produce two mutable borrows.
    let mut first = original.clone();
    let mut second = original.clone();
    assert!(first.element_at_mut::<Df64>(&[0]).is_none());
    assert!(second.element_at_mut::<Df64>(&[0]).is_none());

    // The independent copy is what accepts the write.
    let mut independent = original.duplicate();
    *independent
        .element_at_mut::<Df64>(&[0])
        .expect("an independent payload") = value(9.0, 2f64.powi(-80));
    assert_eq!(
        independent.element_at::<Df64>(&[0]),
        Some(&value(9.0, 2f64.powi(-80)))
    );
    assert_eq!(original.element_at::<Df64>(&[0]), Some(&value(1.0, 0.0)));
}

#[test]
fn a_materialized_view_keeps_the_low_component_in_logical_order() {
    let low = 2f64.powi(-80);
    let original = payload(
        &[
            value(1.0, low),
            value(2.0, 0.0),
            value(3.0, 0.0),
            value(4.0, 0.0),
        ],
        &[2, 2],
    );

    let contiguous = original
        .permuted(&[1, 0])
        .expect("a valid permutation")
        .to_contiguous()
        .expect("the view is inside its payload");

    assert!(contiguous.is_contiguous());
    assert!(!contiguous.shares_payload_with(&original));
    assert_eq!(contiguous.shape(), &[2, 2]);
    assert_eq!(
        contiguous
            .as_dense::<Df64>()
            .expect("dense after materialization")
            .0,
        &[
            value(1.0, low),
            value(3.0, 0.0),
            value(2.0, 0.0),
            value(4.0, 0.0)
        ]
    );

    // The materialized payload is the value the erased tensor type carries, so an
    // erased read sees the permuted order without narrowing anything.
    let tensor = Tensor::external(contiguous);
    assert_eq!(tensor.shape(), &[2, 2]);
    match &tensor {
        Tensor::External(payload, _) => {
            assert!(payload.is_contiguous());
            assert_eq!(payload.element_count(), 4);
            assert_eq!(payload.element_at::<Df64>(&[0, 1]), Some(&value(2.0, 0.0)));
        }
        other => panic!("expected an external payload, found {:?}", other.dtype()),
    }
    assert_eq!(
        tensor.dtype(),
        DType::External(std::any::TypeId::of::<Df64>())
    );
}

#[test]
fn a_sum_reduction_over_a_view_retains_low_order_information() {
    let low = 2f64.powi(-80);
    // Two elements contribute a low component, so the reduction only equals the
    // exact result if both survive.
    let erased = payload(
        &[
            value(1.0, low),
            value(low, 0.0),
            value(0.0, 0.0),
            value(0.0, 0.0),
        ],
        &[2, 2],
    );

    let view = erased.permuted(&[1, 0]).expect("a valid permutation");
    let mut values = Vec::new();
    for first in 0..2 {
        for second in 0..2 {
            values.push(*view.element_at::<Df64>(&[first, second]).expect("in range"));
        }
    }
    let dense = HostTensor::from_vec_col_major(vec![4], values).expect("shape matches data");

    let total = scalar_fold::<Df64, Df64Add>("sum", &dense, Df64::zero()).expect("reduction");
    // 1 + 2^-80 + 2^-80 = 1 + 2^-79 exactly.
    assert_eq!(total, value(1.0, 2f64.powi(-79)));
    assert_eq!(total.narrow_to_f64(), 1.0);
}

#[test]
fn an_invalid_permutation_is_rejected_with_a_typed_error() {
    let erased = payload(&[value(1.0, 0.0), value(2.0, 0.0)], &[2]);

    // One entry per axis, in range, without repetition.
    assert!(erased.permuted(&[0, 1]).is_err());
    assert!(erased.permuted(&[3]).is_err());
    assert!(erased.permuted(&[0, 0]).is_err());

    // The identity permutation is valid and stays dense.
    let identity = erased.permuted(&[0]).expect("the identity permutation");
    assert!(identity.is_contiguous());
    assert!(identity.shares_payload_with(&erased));
}
