use super::*;

#[test]
fn parse_matches_subscripts_integer_labels() {
    let parsed = parse_einsum_subscripts("ij,jk->ik").unwrap();

    assert_eq!(
        parsed.inputs,
        vec![
            vec![b'i' as u32, b'j' as u32],
            vec![b'j' as u32, b'k' as u32]
        ]
    );
    assert_eq!(
        Subscripts::from(&parsed),
        Subscripts::parse("ij,jk->ik").unwrap()
    );
}

#[test]
fn integer_subscripts_convert_to_and_from_raw_subscripts() {
    let raw = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let borrowed = EinsumSubscripts::from(&raw);
    let owned = EinsumSubscripts::from(raw.clone());

    assert_eq!(borrowed, owned);
    assert_eq!(borrowed.input_count(), 2);
    assert_eq!(Subscripts::from(borrowed.clone()), raw);
    assert_eq!(Subscripts::from(&borrowed), raw);
}

#[test]
fn parse_einsum_subscripts_reports_invalid_notation() {
    let err = parse_einsum_subscripts("ij,(jk)->ik").unwrap_err();

    assert!(err.to_string().contains("invalid einsum subscripts"));
}
