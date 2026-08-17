use super::{notation_without_ellipsis, resolve_einsum_notation};
use crate::{parse_einsum_notation, EinsumAxis, EinsumNotation};

#[test]
fn resolves_right_aligned_ellipsis_labels() {
    let notation = parse_einsum_notation("...ij,...jk->...ik").unwrap();
    let resolved = resolve_einsum_notation(&notation, &[&[2, 3, 4][..], &[1, 4, 5]]).unwrap();
    assert_eq!(resolved.inputs[0].len(), 3);
    assert_eq!(resolved.inputs[1].len(), 3);
    assert_eq!(resolved.output.len(), 3);
    assert_eq!(resolved.inputs[0][0], resolved.inputs[1][0]);
    assert_eq!(resolved.output[0], resolved.inputs[0][0]);
}

#[test]
fn resolves_zero_rank_ellipsis_and_programmatic_tokens() {
    let notation = EinsumNotation::new(
        &[&[EinsumAxis::Ellipsis, EinsumAxis::Label(7)]],
        &[EinsumAxis::Ellipsis, EinsumAxis::Label(7)],
    );
    let resolved = resolve_einsum_notation(&notation, &[&[3][..]]).unwrap();
    assert_eq!(resolved.inputs, vec![vec![0]]);
    assert_eq!(resolved.output, vec![0]);
}

#[test]
fn resolves_explicit_notation_without_ellipsis() {
    let notation = EinsumNotation::new(&[&[EinsumAxis::Label(7)]], &[EinsumAxis::Label(7)]);
    let resolved = notation_without_ellipsis(&notation).unwrap();
    assert_eq!(resolved.inputs, vec![vec![7]]);
    assert_eq!(resolved.output, vec![7]);

    let with_ellipsis = EinsumNotation::new(&[&[EinsumAxis::Ellipsis]], &[]);
    assert!(notation_without_ellipsis(&with_ellipsis).is_err());
}

#[test]
fn rejects_structural_and_label_errors() {
    let empty = EinsumNotation {
        inputs: Vec::new(),
        output: Vec::new(),
    };
    assert!(resolve_einsum_notation(&empty, &[]).is_err());

    let notation = EinsumNotation::new(&[&[EinsumAxis::Label(0)]], &[EinsumAxis::Label(0)]);
    assert!(resolve_einsum_notation(&notation, &[]).is_err());
    assert!(resolve_einsum_notation(&notation, &[&[]]).is_err());

    let ellipsis = EinsumNotation::new(
        &[&[EinsumAxis::Ellipsis, EinsumAxis::Label(0)]],
        &[EinsumAxis::Ellipsis, EinsumAxis::Label(0)],
    );
    assert!(resolve_einsum_notation(&ellipsis, &[&[]]).is_err());

    let repeated = EinsumNotation::new(
        &[&[EinsumAxis::Label(0), EinsumAxis::Label(0)]],
        &[EinsumAxis::Label(0)],
    );
    assert!(resolve_einsum_notation(&repeated, &[&[2, 3]]).is_err());

    let missing = EinsumNotation::new(&[&[EinsumAxis::Label(0)]], &[EinsumAxis::Label(1)]);
    assert!(resolve_einsum_notation(&missing, &[&[2]]).is_err());
}

#[test]
fn accepts_both_orders_of_singleton_broadcast() {
    let notation = parse_einsum_notation("...i,...i->...i").unwrap();
    assert!(resolve_einsum_notation(&notation, &[&[1, 3][..], &[2, 3]]).is_ok());
    assert!(resolve_einsum_notation(&notation, &[&[2, 3][..], &[1, 3]]).is_ok());
}

#[test]
fn rejects_missing_output_ellipsis_and_incompatible_broadcast() {
    let missing = parse_einsum_notation("...i->i").unwrap();
    assert!(resolve_einsum_notation(&missing, &[&[2, 3][..]]).is_err());

    let incompatible = parse_einsum_notation("...i,...i->...i").unwrap();
    assert!(resolve_einsum_notation(&incompatible, &[&[2, 3][..], &[4, 3]]).is_err());

    let invalid_output = parse_einsum_notation("i->...i").unwrap();
    assert!(resolve_einsum_notation(&invalid_output, &[&[3][..]]).is_err());
}
