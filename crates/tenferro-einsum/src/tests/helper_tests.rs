use crate::planning::classify::classify_modes;
use crate::syntax::notation::{char_to_label, split_and_validate_notation};
use crate::Error;

#[test]
fn classify_modes_preserves_input_order_and_deduplicates_labels() {
    let (batch, lo, ro, sum) = classify_modes(
        &[10, 20, 20, 30, 40, 50],
        &[20, 30, 30, 60, 50],
        &[20, 40, 60],
    );

    assert_eq!(batch, vec![20]);
    assert_eq!(lo, vec![40]);
    assert_eq!(ro, vec![60]);
    assert_eq!(sum, vec![30, 50]);
}

#[test]
fn classify_modes_ignores_labels_missing_from_rhs_and_output() {
    let (batch, lo, ro, sum) = classify_modes(&[1, 2, 3], &[2, 4], &[4]);

    assert!(batch.is_empty());
    assert!(lo.is_empty());
    assert_eq!(ro, vec![4]);
    assert_eq!(sum, vec![2]);
}

#[test]
fn char_to_label_accepts_unicode_alphanumeric_and_private_use() {
    assert_eq!(char_to_label('7').unwrap(), '7' as u32);
    assert_eq!(char_to_label('β').unwrap(), 'β' as u32);
    assert_eq!(char_to_label('\u{E123}').unwrap(), 0xE123);
}

#[test]
fn char_to_label_accepts_unicode_symbols() {
    assert_eq!(char_to_label('×').unwrap(), '×' as u32);
    assert_eq!(char_to_label('÷').unwrap(), '÷' as u32);
    assert_eq!(char_to_label('\u{03A2}').unwrap(), 0x03A2);
}

#[test]
fn char_to_label_rejects_reserved_syntax_chars() {
    let err = char_to_label('-').unwrap_err();
    match err {
        Error::InvalidSubscripts { message: msg } => {
            assert!(msg.contains("invalid einsum label character"));
            assert!(msg.contains("reserved syntax character"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert!(char_to_label(',').is_err());
    assert!(char_to_label('>').is_err());
    assert!(char_to_label('(').is_err());
    assert!(char_to_label(')').is_err());
    assert!(char_to_label(' ').is_err());
}

#[test]
fn split_and_validate_notation_accepts_balanced_parentheses() {
    let (lhs, rhs) = split_and_validate_notation("(ij,jk),kl->il").unwrap();
    assert_eq!(lhs, "(ij,jk),kl");
    assert_eq!(rhs, "il");
}

#[test]
fn split_and_validate_notation_rejects_missing_or_extra_arrow() {
    let missing = split_and_validate_notation("ij,jk").unwrap_err();
    let extra = split_and_validate_notation("ij->jk->ik").unwrap_err();

    match missing {
        Error::InvalidSubscripts { message: msg } => assert!(msg.contains("exactly one '->'")),
        other => panic!("unexpected error: {other:?}"),
    }
    match extra {
        Error::InvalidSubscripts { message: msg } => assert!(msg.contains("exactly one '->'")),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn split_and_validate_notation_rejects_unbalanced_parentheses() {
    let unmatched_open = split_and_validate_notation("(ij,jk->ik").unwrap_err();
    let unmatched_close = split_and_validate_notation("ij),jk->ik").unwrap_err();

    match unmatched_open {
        Error::InvalidSubscripts { message: msg } => assert!(msg.contains("unmatched '('")),
        other => panic!("unexpected error: {other:?}"),
    }
    match unmatched_close {
        Error::InvalidSubscripts { message: msg } => assert!(msg.contains("unmatched ')'")),
        other => panic!("unexpected error: {other:?}"),
    }
}
