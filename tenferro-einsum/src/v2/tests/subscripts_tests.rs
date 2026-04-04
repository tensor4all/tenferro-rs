use crate::v2::types::Subscripts;

#[test]
fn parse_matmul() {
    let s = Subscripts::parse("ij,jk->ik");
    assert_eq!(s.n_inputs(), 2);
    assert_eq!(s.input_labels[0], vec!['i' as u32, 'j' as u32]);
    assert_eq!(s.input_labels[1], vec!['j' as u32, 'k' as u32]);
    assert_eq!(s.output_labels, vec!['i' as u32, 'k' as u32]);
}

#[test]
fn parse_trace() {
    let s = Subscripts::parse("ii->");
    assert_eq!(s.n_inputs(), 1);
    assert_eq!(s.input_labels[0], vec!['i' as u32, 'i' as u32]);
    assert!(s.output_labels.is_empty());
}

#[test]
fn parse_row_sum() {
    let s = Subscripts::parse("ij->i");
    assert_eq!(s.n_inputs(), 1);
    assert_eq!(s.input_labels[0], vec!['i' as u32, 'j' as u32]);
    assert_eq!(s.output_labels, vec!['i' as u32]);
}

#[test]
fn parse_outer_product() {
    let s = Subscripts::parse("i,j->ij");
    assert_eq!(s.n_inputs(), 2);
    assert_eq!(s.input_labels[0], vec!['i' as u32]);
    assert_eq!(s.input_labels[1], vec!['j' as u32]);
    assert_eq!(s.output_labels, vec!['i' as u32, 'j' as u32]);
}

#[test]
fn all_labels_includes_all() {
    let s = Subscripts::parse("ij,jk->ik");
    let all = s.all_labels();
    assert!(all.contains(&('i' as u32)));
    assert!(all.contains(&('j' as u32)));
    assert!(all.contains(&('k' as u32)));
    assert_eq!(all.len(), 3);
}

#[test]
fn parse_three_input() {
    let s = Subscripts::parse("ij,jk,kl->il");
    assert_eq!(s.n_inputs(), 3);
    assert_eq!(s.input_labels[2], vec!['k' as u32, 'l' as u32]);
    assert_eq!(s.output_labels, vec!['i' as u32, 'l' as u32]);
}
