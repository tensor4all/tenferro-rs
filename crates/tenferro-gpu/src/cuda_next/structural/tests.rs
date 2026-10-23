//! Unit tests for the structural slice's arithmetic helpers.

use super::super::structural::element_count;

#[test]
fn the_element_count_checks_both_extremes() {
    assert_eq!(element_count(&[]).expect("scalar"), 1);
    assert_eq!(element_count(&[2, 3, 4]).expect("24"), 24);
    assert!(element_count(&[usize::MAX, 2]).is_err());
}
