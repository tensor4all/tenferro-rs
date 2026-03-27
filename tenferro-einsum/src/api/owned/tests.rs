use num_complex::Complex64;
use tenferro_algebra::Standard;
use tenferro_prims::{CpuBackend, CpuContext};
use tenferro_tensor::{MemoryOrder, Tensor};

use crate::{einsum_binary_with_subscripts, Subscripts};

use super::{canonicalize_col_major_operands_owned, einsum_with_subscripts_owned};

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

#[test]
fn owned_binary_entry_matches_borrowed_binary_entry() {
    let subs = Subscripts::new(&[&[0, 1], &[1, 2]], &[0, 2]);
    let a = Tensor::<f64>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();
    let b = Tensor::<f64>::from_slice(
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        &[3, 2],
        MemoryOrder::ColumnMajor,
    )
    .unwrap();

    let mut borrowed_ctx = CpuContext::new(1);
    let expected = einsum_binary_with_subscripts::<Standard<f64>, CpuBackend>(
        &mut borrowed_ctx,
        &subs,
        &a,
        &b,
        None,
    )
    .unwrap();

    let mut owned_ctx = CpuContext::new(1);
    let actual = einsum_with_subscripts_owned::<Standard<f64>, CpuBackend>(
        &mut owned_ctx,
        &subs,
        vec![a, b],
        None,
    )
    .unwrap();

    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.to_vec(), expected.to_vec());
}
