use num_complex::Complex64;
use tenferro_tensor::{MemoryOrder, Tensor};

use super::canonicalize_col_major_operands_owned;

#[test]
fn owned_canonicalization_preserves_unique_buffer_for_already_canonical_operands() {
    let tensor =
        Tensor::<f64>::from_slice(&[1.0, 2.0, 3.0], &[3], MemoryOrder::ColumnMajor).unwrap();
    assert!(tensor.buffer().is_unique());

    let canonical = canonicalize_col_major_operands_owned(vec![tensor]);

    assert_eq!(canonical.len(), 1);
    assert!(canonical[0].buffer().is_unique());
    assert!(canonical[0].is_col_major_contiguous());
    assert_eq!(canonical[0].offset(), 0);
    assert!(!canonical[0].is_conjugated());
}

#[test]
fn owned_canonicalization_materializes_lazy_conjugation_into_plain_col_major_storage() {
    let tensor = Tensor::<Complex64>::from_slice(
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
        &[2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap()
    .conj();

    let canonical = canonicalize_col_major_operands_owned(vec![tensor]);

    assert_eq!(canonical.len(), 1);
    assert!(canonical[0].is_col_major_contiguous());
    assert_eq!(canonical[0].offset(), 0);
    assert!(!canonical[0].is_conjugated());
    assert_eq!(
        canonical[0].buffer().as_slice().unwrap(),
        &[Complex64::new(1.0, -2.0), Complex64::new(3.0, 4.0)]
    );
}
